"""
Does ESM3 fold cleaved zero_polymer (HA1 | HA2) as ONE molecule, and does it
put Cys14(HA1) and Cys137(HA2) in disulfide-bonding geometry?

This is the structural question that tests/references.py's cleaved block sets
up but cannot answer: the bridge is RECORDED there, never imposed, so whether
the model finds it unaided is exactly what is measured here.

WHY THIS SCRIPT GOES AROUND THE SAMPLER. The rest of the suite reaches the
model through ExtendedProtein + predict_attention(sequence_probs=...), the
custom differentiable path. That path cannot carry a chainbreak at all:
expand() maps characters through C.SEQUENCE_USED_VOCAB (25 amino acids, no
"|"), and the encoder's format_sequence_probs() zero-pads over token 31, so
the break is unreachable as sequence_probs (see CHAINBREAK_UNSUPPORTED in
references.py). This script therefore uses the STOCK SDK route --
model.encode(ESMProtein(...)) -- which runs the real sequence tokenizer, for
which "|" is a registered special token. Confirmed working: 551 positions in,
551 out, chainbreak intact, to_protein_complex() recovers 2 chains.

Two tempting shortcuts are NOT used here, because both are wrong:
  - chain_id= on predict_attention is consumed only by geometric attention
    (transformer_stack.py "Only used in geometric attention"). The soft path
    feeds all-NaN coords, so affine_mask is all-False, geometric attention is
    masked and zeroed, and chain_id has no effect at all -- a silent no-op.
  - sequence_id= is an attention SEPARATOR (blocks.py: self.attn(x,
    sequence_id)). Different ids would make the chains ignore each other --
    the opposite of two chains tethered by a disulfide.

DECODING STEPS ARE THE WHOLE BALLGAME. The first run of this script used
utils/predictions.py:init_structure_config(), whose num_steps defaults to 1,
and produced ptm=0.18 with a 1.64 A CB-CB "contact" -- shorter than a C-C
covalent bond, i.e. a steric collapse, not a disulfide. All 551 structure
tokens had been sampled in a single step. For scale, the SDK's own
generation_test.py uses num_steps=10 for a FIVE-residue sequence and
num_steps=1 only to assert you cannot request more steps than masks.
So --steps is now explicit and defaults to 64, and a GEOMETRY SANITY GATE
runs before the bridge verdict: a structure with low ptm or CB-CB clashes is
reported as untrustworthy rather than silently measured. A distance from a
collapsed structure is not a finding about the model.

Run from the softmax/code directory:
    python tests/test_cleaved_complex.py                        # cleaved, 64 steps
    python tests/test_cleaved_complex.py --steps 256
    python tests/test_cleaved_complex.py --ref mature           # control: no break
    python tests/test_cleaved_complex.py --ref one              # control: HA1 alone
    python tests/test_cleaved_complex.py --pdb /tmp/ha.pdb
The controls are the point of comparison: if `mature` (same 550 residues, no
chainbreak) folds no better, the problem is the protein or the budget, not
the cleavage. Note also that HA is natively a TRIMER -- a lone HA1+HA2
protomer may simply not have a confident monomeric fold.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESMProtein
from custom_esm.utils.residue_constants import atom_order

from utils.predictions import init_structure_config

from references import (
	ZERO_POLYMER_CLEAVED,
	ZERO_POLYMER_MATURE,
	POLYMER_ONE,
	POLYMER_TWO,
	DISULFIDE_BRIDGES,
	CHAIN_BREAK,
	chain_index,
)

# A disulfide is SG-SG ~2.05 A; 2.5 A is a generous bonded cutoff.
SG_BONDED = 2.5
# With no side-chain S predicted, CB-CB < 4.5 A is the usual stand-in.
CB_BONDED = 4.5
# Two CB atoms closer than this are clashing: vdW contact is ~3.4 A and a
# C-C covalent bond is 1.54 A, so anything under 3.0 is unphysical.
CB_CLASH = 3.0
# Below this ptm the fold is not worth measuring.
PTM_FLOOR = 0.40

REFS = {
	"cleaved": ZERO_POLYMER_CLEAVED,
	"mature": ZERO_POLYMER_MATURE,
	"one": POLYMER_ONE,
	"two": POLYMER_TWO,
}


def arg(flag: str, default=None):
	return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def bridge_indices(ref: str) -> tuple[int, int] | None:
	"""0-based coordinate indices of the two disulfide anchors for the active
	reference, or None where the pair does not exist (a single chain). The
	cleaved and break-free forms differ by the chainbreak position, which is
	exactly the off-by-one chain_index() exists to absorb."""
	(c1, p1), (c2, p2) = DISULFIDE_BRIDGES[0]
	if ref == "cleaved":
		return chain_index(c1, p1), chain_index(c2, p2)
	if ref == "mature":  # same residues, no break occupying a position
		return p1 - 1, len(POLYMER_ONE) + p2 - 1
	return None


def cb_distances(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Full CB-CB distance matrix and a validity mask. Vectorised: the suite's
	compute_distance_matrix is a Python double loop over residues and atoms,
	which at L=551 would take minutes."""
	cb = coords[:, atom_order["CB"], :]
	ok = torch.isfinite(cb).all(dim=-1)
	safe = torch.where(ok[:, None], cb, torch.zeros_like(cb))
	return torch.cdist(safe, safe), ok


def atom_distance(coords: torch.Tensor, i: int, j: int, atom: str) -> float | None:
	a = atom_order[atom]
	xi, xj = coords[i, a], coords[j, a]
	if not (torch.isfinite(xi).all() and torch.isfinite(xj).all()):
		return None
	return torch.sqrt(((xi - xj) ** 2.).sum()).item()


def present_atoms(coords: torch.Tensor, i: int) -> list[str]:
	return [n for n, a in atom_order.items() if bool(torch.isfinite(coords[i, a]).all())]


def fmt(d: float | None) -> str:
	return "   n/a" if d is None else f"{d:6.2f}"


def main() -> None:
	ref = arg("--ref", "cleaved")
	if ref not in REFS:
		raise SystemExit(f"--ref must be one of {sorted(REFS)}, got {ref!r}")
	seq = REFS[ref]
	steps = int(arg("--steps", 64))
	pdb_out = arg("--pdb")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	nres = len(seq) - seq.count(CHAIN_BREAK)

	print(f"# reference: zero_polymer_{ref} (L={nres} residues"
		  + (f" + {seq.count(CHAIN_BREAK)} chainbreak, {len(seq)} positions)"
			 if CHAIN_BREAK in seq else ")"))
	print(f"# decoding steps: {steps}   device: {device}")
	print()

	model = ESM3.from_pretrained("esm3-open").to(device)
	for p in model.parameters():
		p.requires_grad = False

	tensor = model.encode(ESMProtein(sequence=seq))
	out = model.predict_protein(tensor, init_structure_config(num_steps=steps))

	coords = out.coordinates.detach().to("cpu").float()
	ptm = None if out.ptm is None else float(out.ptm)
	print(f"predicted sequence length: {len(out.sequence)}   "
		  f"coordinates: {tuple(coords.shape)}")

	# ---------------------------------------------------------------- gate
	# Decide whether this structure is worth measuring BEFORE measuring it.
	# The 1-step run scored ptm=0.18 with a 1.64 A CB-CB pair; reporting a
	# bridge distance from that would have been reporting noise.
	d_cb, ok = cb_distances(coords)
	L = d_cb.shape[0]
	sep = (torch.arange(L)[:, None] - torch.arange(L)[None, :]).abs()
	valid = ok[:, None] & ok[None, :] & (sep >= 2) & torch.triu(
		torch.ones(L, L, dtype=torch.bool), diagonal=1
	)
	clashes = int(((d_cb < CB_CLASH) & valid).sum())
	min_cb = float(d_cb[valid].min()) if bool(valid.any()) else float("nan")

	print(f"\n--- geometry sanity ---")
	print(f"  ptm                        {'n/a' if ptm is None else f'{ptm:.4f}'}"
		  f"   (floor {PTM_FLOOR})")
	print(f"  min CB-CB (|i-j|>=2)      {min_cb:6.2f} A   (clash below {CB_CLASH})")
	print(f"  CB-CB clashes              {clashes}   (a physical fold has ~0)")

	bad = []
	if ptm is not None and ptm < PTM_FLOOR:
		bad.append(f"ptm {ptm:.3f} < {PTM_FLOOR}")
	if clashes:
		bad.append(f"{clashes} CB-CB clashes")
	if bad:
		print(f"\n  [UNTRUSTWORTHY] {'; '.join(bad)}.")
		print(f"  Distances below describe a structure the model does not")
		print(f"  believe in. Raise --steps and re-run before concluding")
		print(f"  anything about the bridge. Compare --ref mature as a control.")
	else:
		print(f"\n  [OK] fold passes the sanity gate; distances are meaningful.")

	atoms = present_atoms(coords, 0)
	print(f"\natoms predicted (residue 0): {' '.join(atoms)}")
	if "SG" not in atoms:
		print(f"  no side-chain S in the output -- the decoder emits backbone"
			  f" + CB only, so CB-CB is the operative criterion.")

	# -------------------------------------------------------------- bridge
	idx = bridge_indices(ref)
	if idx is None:
		print(f"\n(ref {ref!r} is a single chain -- no interchain bridge to measure)")
	else:
		i1, i2 = idx
		(c1, p1), (c2, p2) = DISULFIDE_BRIDGES[0]
		for i, chain, pos in ((i1, c1, p1), (i2, c2, p2)):
			got = out.sequence[i] if i < len(out.sequence) else "?"
			flag = "" if got == "C" else "   <-- NOT a cysteine, indices are off!"
			print(f"  {chain}:{pos} -> position {i}: {got}{flag}")

		sg = atom_distance(coords, i1, i2, "SG")
		cb = atom_distance(coords, i1, i2, "CB")
		ca = atom_distance(coords, i1, i2, "CA")
		print(f"\n--- the bridge: {c1}:Cys{p1} <-> {c2}:Cys{p2} ---")
		print(f"  SG-SG  {fmt(sg)} A   (bonded ~2.05, cutoff {SG_BONDED})")
		print(f"  CB-CB  {fmt(cb)} A   (bonded <{CB_BONDED})")
		print(f"  CA-CA  {fmt(ca)} A   (bonded ~5.5-6.8)")
		if sg is not None:
			formed, basis = sg <= SG_BONDED, f"SG-SG={sg:.2f} A"
		elif cb is not None:
			formed, basis = cb <= CB_BONDED, f"CB-CB={cb:.2f} A, no SG predicted"
		else:
			formed, basis = None, "neither SG nor CB predicted"
		verdict = "cannot tell" if formed is None else ("FORMED" if formed else "NOT formed")
		caveat = "  (but see UNTRUSTWORTHY above)" if bad else ""
		print(f"\n  verdict: bridge {verdict}  [{basis}]{caveat}")

	# ------------------------------------------------------- cys pair scan
	# Scale for the single number above: what did the model do with the
	# cysteines it DID pair up? Needs no outside knowledge of which HA
	# cysteines natively bond.
	cys = [i for i, a in enumerate(out.sequence) if a == "C"]
	pairs = sorted(
		(float(d_cb[a, b]), a, b)
		for n, a in enumerate(cys) for b in cys[n + 1:]
		if bool(ok[a] and ok[b])
	)

	brk = out.sequence.find(CHAIN_BREAK)

	def label(i: int) -> str:
		if brk == -1 or i < brk:
			return f"one:{i + 1}"
		return f"two:{i - brk}"

	target = set(idx) if idx else set()
	print(f"\n--- all {len(pairs)} cysteine pairs by CB-CB, closest first ---")
	for rank, (d, a, b) in enumerate(pairs, start=1):
		if rank > 12 and {a, b} != target:
			continue
		mark = "  <== the HA1-HA2 bridge" if {a, b} == target else ""
		tag = "!" if d < CB_CLASH else ("*" if d <= CB_BONDED else " ")
		print(f"  {rank:3d}. {tag} {d:6.2f} A  {label(a):>9s} -- {label(b):<9s}{mark}")

	n_bonded = sum(1 for d, _, _ in pairs if CB_CLASH <= d <= CB_BONDED)
	n_clash = sum(1 for d, _, _ in pairs if d < CB_CLASH)
	print(f"\n  {n_bonded} pairs in plausible bonding range "
		  f"[{CB_CLASH}, {CB_BONDED}] A (*)")
	print(f"  {n_clash} pairs CLASHING below {CB_CLASH} A (!) "
		  f"-- these are not disulfides, they are broken geometry")

	if pdb_out:
		out.to_pdb(pdb_out)
		print(f"\nwrote {pdb_out}")
		if CHAIN_BREAK in seq:
			pc = out.to_protein_complex()
			print(f"to_protein_complex() -> {len(list(pc.chain_iter()))} chains "
				  f"(expected 2)")


if __name__ == "__main__":
	main()
