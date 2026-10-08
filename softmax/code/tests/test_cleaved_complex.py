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
which "|" is a registered special token. No plumbing change needed.

Two tempting shortcuts are NOT used here, because both are wrong:
  - chain_id= on predict_attention is consumed only by geometric attention
    (transformer_stack.py "Only used in geometric attention", and
    geom_attention.py's chain_id_mask). The soft path feeds all-NaN coords,
    so affine_mask is all-False, geometric attention is fully masked and
    zeroed, and chain_id has no effect whatsoever -- a silent no-op.
  - sequence_id= is an attention SEPARATOR (blocks.py: self.attn(x,
    sequence_id)). Giving the two chains different ids would make them ignore
    each other -- the opposite of two chains tethered by a disulfide.

HOW THE VERDICT IS REACHED. A disulfide has SG-SG ~2.05 A. But ESM3's
structure decoder may emit only backbone + CB, in which case SG does not
exist in the output and an SG-SG number cannot be had. So three measures are
reported, and the script says which atoms are actually present rather than
assuming:
    SG-SG   ~2.05 A bonded            (only if side-chain S is predicted)
    CB-CB   <4.5 A is the standard geometric criterion when SG is absent
    CA-CA   ~5.5-6.8 A for a bonded pair
The target pair is then ranked against EVERY cysteine pair in the molecule.
That ranking is the point: if Cys14-Cys137 is among the closest Cys pairs,
the model placed it in bonding geometry; if it sits mid-pack at 20+ A, it
did not, and no single distance could have told you which.

Needs a working ESM3 install + downloaded weights + network for the first
from_pretrained call. Run from the softmax/code directory:
    python tests/test_cleaved_complex.py
    python tests/test_cleaved_complex.py --pdb out.pdb   # also write a PDB
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESMProtein
from custom_esm.utils.residue_constants import atom_order

from utils.predictions import init_structure_config
from utils.matrices import compute_residue_distance

from references import (
	ZERO_POLYMER_CLEAVED,
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


def present_atoms(coords: torch.Tensor, i: int) -> list[str]:
	"""Names of the atoms actually predicted at residue i. Absent atoms come
	back as inf or nan depending on the decoder, so test for finiteness
	rather than for one sentinel."""
	return [
		name for name, a in atom_order.items()
		if bool(torch.isfinite(coords[i, a]).all())
	]


def atom_distance(coords: torch.Tensor, i: int, j: int, atom: str) -> float | None:
	"""Distance (A) between `atom` of residue i and the same atom of residue
	j, or None if either is missing from the prediction."""
	a = atom_order[atom]
	xi, xj = coords[i, a], coords[j, a]
	if not (torch.isfinite(xi).all() and torch.isfinite(xj).all()):
		return None
	return torch.sqrt(((xi - xj) ** 2.).sum()).item()


def min_interatomic(coords: torch.Tensor, i: int, j: int) -> float | None:
	"""Closest approach between any predicted atom of residue i and any of
	residue j, via the suite's own compute_residue_distance -- the same
	measure utils/matrices.py builds contact maps from."""
	mi = torch.isfinite(coords[i]).all(dim=-1)
	mj = torch.isfinite(coords[j]).all(dim=-1)
	if not (bool(mi.any()) and bool(mj.any())):
		return None
	return compute_residue_distance(coords[i][mi], coords[j][mj])


def fmt(d: float | None) -> str:
	return "   n/a" if d is None else f"{d:6.2f}"


def main() -> None:
	pdb_out = None
	if "--pdb" in sys.argv:
		pdb_out = sys.argv[sys.argv.index("--pdb") + 1]

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	seq = ZERO_POLYMER_CLEAVED
	print(f"# reference: zero_polymer_cleaved "
		  f"(L={len(seq) - seq.count(CHAIN_BREAK)} residues + "
		  f"{seq.count(CHAIN_BREAK)} chainbreak, {len(seq)} positions)")
	print(f"# polymer_one (HA1): {len(POLYMER_ONE)} aa      "
		  f"polymer_two (HA2): {len(POLYMER_TWO)} aa")
	print(f"# device: {device}")
	print()

	model = ESM3.from_pretrained("esm3-open").to(device)
	for p in model.parameters():
		p.requires_grad = False

	# The stock route: the real tokenizer, for which "|" is a special token.
	# ExtendedProtein.expand() would raise here instead -- that is the whole
	# reason this script exists.
	tensor = model.encode(ESMProtein(sequence=seq))
	out = model.predict_protein(tensor, init_structure_config())

	coords = out.coordinates.detach().to("cpu").float()
	print(f"predicted sequence length: {len(out.sequence)}  "
		  f"coordinates: {tuple(coords.shape)}")
	if out.ptm is not None:
		print(f"ptm: {float(out.ptm):.4f}")

	# Locate the cysteines in the RETURNED sequence rather than trusting a
	# precomputed index: if the decoder strips or shifts the break, a
	# precomputed index would quietly measure the wrong two atoms. Any
	# disagreement with chain_index() is reported, not absorbed.
	if len(out.sequence) != len(seq):
		print(f"\n[WARN] returned sequence is {len(out.sequence)} positions, "
			  f"input was {len(seq)} -- indices below are NOT trustworthy; "
			  f"inspect out.sequence before reading the distances.")
	if coords.shape[0] != len(out.sequence):
		print(f"[WARN] coordinates ({coords.shape[0]}) and sequence "
			  f"({len(out.sequence)}) disagree in length.")

	(c1, p1), (c2, p2) = DISULFIDE_BRIDGES[0]
	i1, i2 = chain_index(c1, p1), chain_index(c2, p2)
	for idx, chain, pos in ((i1, c1, p1), (i2, c2, p2)):
		got = out.sequence[idx] if idx < len(out.sequence) else "?"
		flag = "" if got == "C" else "   <-- NOT a cysteine, indices are off!"
		print(f"  {chain}:{pos} -> position {idx}: {got}{flag}")

	print(f"\natoms predicted at {c1}:{p1}: {' '.join(present_atoms(coords, i1))}")
	print(f"atoms predicted at {c2}:{p2}: {' '.join(present_atoms(coords, i2))}")

	sg = atom_distance(coords, i1, i2, "SG")
	cb = atom_distance(coords, i1, i2, "CB")
	ca = atom_distance(coords, i1, i2, "CA")
	mn = min_interatomic(coords, i1, i2)

	print(f"\n--- the interchain bridge: {c1}:Cys{p1} <-> {c2}:Cys{p2} ---")
	print(f"  SG-SG            {fmt(sg)} A   (bonded ~2.05, cutoff {SG_BONDED})")
	print(f"  CB-CB            {fmt(cb)} A   (bonded <{CB_BONDED})")
	print(f"  CA-CA            {fmt(ca)} A   (bonded ~5.5-6.8)")
	print(f"  closest approach {fmt(mn)} A")

	if sg is not None:
		formed = sg <= SG_BONDED
		basis = f"SG-SG={sg:.2f} A"
	elif cb is not None:
		formed = cb <= CB_BONDED
		basis = f"CB-CB={cb:.2f} A (no side-chain S predicted)"
	else:
		formed, basis = None, "neither SG nor CB predicted at both sites"
	verdict = "cannot tell" if formed is None else ("FORMED" if formed else "NOT formed")
	print(f"\n  verdict: bridge {verdict}  [{basis}]")

	# Rank the target pair against every other cysteine pair. Without this the
	# single number above has no scale: "18 A" only means something next to
	# what the model does with the cysteines it DID pair up.
	cys = [i for i, a in enumerate(out.sequence) if a == "C"]
	pairs = []
	for a in range(len(cys)):
		for b in range(a + 1, len(cys)):
			d = atom_distance(coords, cys[a], cys[b], "CB")
			if d is not None:
				pairs.append((d, cys[a], cys[b]))
	pairs.sort()

	def label(i: int) -> str:
		"""Position -> chain-local name, for output that reads in the same
		numbering references.py uses."""
		brk = out.sequence.find(CHAIN_BREAK)
		if brk == -1 or i < brk:
			return f"one:{i + 1}"
		return f"two:{i - brk}"

	print(f"\n--- all {len(pairs)} cysteine pairs by CB-CB, closest first ---")
	target = {i1, i2}
	for rank, (d, a, b) in enumerate(pairs, start=1):
		mark = "  <== the HA1-HA2 bridge" if {a, b} == target else ""
		bonded = "*" if d <= CB_BONDED else " "
		if rank <= 12 or {a, b} == target:
			print(f"  {rank:3d}. {bonded} {d:6.2f} A  "
				  f"{label(a):>9s} -- {label(b):<9s}{mark}")

	n_bonded = sum(1 for d, _, _ in pairs if d <= CB_BONDED)
	print(f"\n  {n_bonded} of {len(pairs)} pairs within {CB_BONDED} A "
		  f"(* above) -- these are the disulfides ESM3 actually built.")

	if pdb_out:
		out.to_pdb(pdb_out)
		print(f"\nwrote {pdb_out}")
		pc = out.to_protein_complex()
		print(f"to_protein_complex() split it into "
			  f"{len(list(pc.chain_iter()))} chains "
			  f"(expected 2 -- confirms the break survived the round trip)")


if __name__ == "__main__":
	main()
