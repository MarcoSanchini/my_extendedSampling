"""
Does ESM3 fold cleaved zero_polymer (HA1 | HA2) as ONE molecule, and does it
put Cys14(HA1) and Cys137(HA2) in disulfide-bonding geometry?

references.py RECORDS that bridge, never imposes it, so whether ESM3 finds it
unaided is what this measures. Run it on protein_g FIRST (see CONTROL below):
a fold number from this script means nothing until the harness is shown to
fold something.

WHY THIS GOES AROUND THE SAMPLER. The suite's usual route (ExtendedProtein +
predict_attention on sequence_probs) cannot carry a chainbreak: expand() maps
through C.SEQUENCE_USED_VOCAB (25 amino acids, no "|") and the encoder's
format_sequence_probs() zero-pads over token 31 (see CHAINBREAK_UNSUPPORTED
in references.py). The stock model.encode(ESMProtein(...)) route runs the
real sequence tokenizer, where "|" is a registered special token. CONFIRMED
working 2026-10-08: 551 positions in, 551 out, break intact,
to_protein_complex() recovers 2 chains, chain_index() lands on C at 13/466.

Two routes that look right and are not: chain_id= is consumed only by
geometric attention, which the soft path's all-NaN coords mask and zero out,
so it is a SILENT no-op; sequence_id= is an attention separator, so distinct
ids would make the chains ignore each other -- the opposite of a tether.

WHAT THE 2026-10-08 RUNS ESTABLISHED, and why this script now looks the way
it does:
  - num_steps MATTERS and init_structure_config() defaults to 1. The first
    run sampled all 551 structure tokens in one step: ptm=0.18, a 1.64 A
    CB-CB "contact" (shorter than a C-C bond), bridge CB-CB=38.55 A. That
    verdict was worthless. --steps is explicit now and the GEOMETRY SANITY
    GATE runs before any verdict.
  - Raising steps 1 -> 256 did NOT rescue it: ptm 0.18 -> 0.23 and the bridge
    distance got WORSE (38.6 -> 95.7 A). Step sweeps are not the answer.
  - THE CHAINBREAK IS EXONERATED. --ref mature (same 550 residues, no break)
    folds no better: ptm 0.261 vs 0.232, bridge 129.7 vs 95.7 A. Nothing
    indicts ESM3's multi-chain handling.
  - ESM3's decoder emits backbone + CB only; there is NO side-chain S at any
    residue, so SG-SG is permanently unavailable and CB-CB is the criterion.
  - 95-130 A separations across 550 residues mean the chain is EXTENDED, not
    merely misfolded (a compact 550-mer maxes out near 70-80 A). Leading
    explanation: HA is an obligate TRIMER whose HA2 central helix is an
    inter-protomer coiled-coil, so a lone protomer may have no self-contained
    fold. That would make low confidence the CORRECT answer here.

CB PROVENANCE, which decides whether the clash count is real. CB is not
predicted: ProteinChain.infer_cbeta() derives it from N/CA/C, and with its
default infer_cbeta_for_glycine=False it writes NaN at every glycine. So
glycines are excluded from every CB statistic here by construction, NOT by
accident -- two finite CBs 1.2 A apart therefore mean two nearly superimposed
BACKBONES, i.e. a real clash. The one position whose CB is meaningless is the
chainbreak itself ("|" is not G, so a CB gets inferred from nonsense
backbone): it is excluded explicitly below. Earlier runs did not exclude it.

Run from the softmax/code directory:
    python tests/test_cleaved_complex.py --ref protein_g --steps 56   # CONTROL
    python tests/test_cleaved_complex.py --ref cleaved --steps 256
    python tests/test_cleaved_complex.py --ref mature  --steps 256
    python tests/test_cleaved_complex.py --ref cleaved --steps 256 --pdb out.pdb
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
	PROTEIN_G,
	ZERO_POLYMER_CLEAVED,
	ZERO_POLYMER_MATURE,
	POLYMER_ONE,
	POLYMER_TWO,
	DISULFIDE_BRIDGES,
	CHAIN_BREAK,
	chain_index,
)

SG_BONDED = 2.5    # a disulfide is SG-SG ~2.05 A
CB_BONDED = 4.5    # with no side-chain S, the usual CB-CB stand-in
CB_CLASH = 3.0     # vdW contact ~3.4 A, C-C bond 1.54 A: under 3.0 is unphysical
PTM_FLOOR = 0.40   # below this the fold is not worth measuring
PTM_CONTROL = 0.70 # what protein_g must clear for the harness to be trusted
PLDDT_GOOD = 70.0  # per-residue confidence band, AlphaFold convention

REFS = {
	"cleaved": ZERO_POLYMER_CLEAVED,
	"mature": ZERO_POLYMER_MATURE,
	"one": POLYMER_ONE,
	"two": POLYMER_TWO,
	# POSITIVE CONTROL. 56 aa, the suite's long-standing reference, and a
	# protein ESM3 must be able to fold. If this does not clear PTM_CONTROL
	# with zero clashes, the harness is wrong -- note init_structure_config
	# carries condition_on_coordinates_only=True, written for contact-map
	# work, not de novo folding -- and every zero_polymer number from this
	# script is void rather than being a fact about ESM3. protein_g has NO
	# cysteines, so its pair scan is correctly empty.
	"protein_g": PROTEIN_G,
}


def arg(flag: str, default=None):
	return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def chain_spans(ref: str, seq: str) -> list[tuple[str, int, int]]:
	"""(name, start, end) per chain, as half-open coordinate ranges that
	EXCLUDE any chainbreak position."""
	brk = seq.find(CHAIN_BREAK)
	if brk != -1:
		return [("polymer_one", 0, brk), ("polymer_two", brk + 1, len(seq))]
	if ref == "mature":  # same two chains, no break occupying a position
		return [("polymer_one", 0, len(POLYMER_ONE)),
				("polymer_two", len(POLYMER_ONE), len(seq))]
	return [(ref, 0, len(seq))]


def make_label(ref: str, seq: str):
	"""Position -> chain-local name. Must be ref-aware: with no "|" in the
	sequence, find() returns -1 and a break-relative formula silently labels
	every HA2 residue as one:329+n, which made the first mature results
	uncomparable with the cleaved ones by eye."""
	spans = chain_spans(ref, seq)
	short = {"polymer_one": "one", "polymer_two": "two"}

	def label(i: int) -> str:
		for name, start, end in spans:
			if start <= i < end:
				return f"{short.get(name, name)}:{i - start + 1}"
		return f"BREAK:{i}"

	return label


def bridge_indices(ref: str) -> tuple[int, int] | None:
	"""Coordinate indices of the two disulfide anchors, or None where the pair
	does not exist. The cleaved and break-free forms differ by the position
	the break occupies."""
	(c1, p1), (c2, p2) = DISULFIDE_BRIDGES[0]
	if ref == "cleaved":
		return chain_index(c1, p1), chain_index(c2, p2)
	if ref == "mature":
		return p1 - 1, len(POLYMER_ONE) + p2 - 1
	return None


def atom_distance(coords: torch.Tensor, i: int, j: int, atom: str) -> float | None:
	a = atom_order[atom]
	xi, xj = coords[i, a], coords[j, a]
	if not (torch.isfinite(xi).all() and torch.isfinite(xj).all()):
		return None
	return torch.sqrt(((xi - xj) ** 2.).sum()).item()


def fmt(d: float | None) -> str:
	return "   n/a" if d is None else f"{d:6.2f}"


def main() -> None:
	ref = arg("--ref", "cleaved")
	if ref not in REFS:
		raise SystemExit(f"--ref must be one of {sorted(REFS)}, got {ref!r}")
	seq = REFS[ref]
	steps = int(arg("--steps", 64))
	if steps > len(seq):  # cannot decode more steps than there are masks
		print(f"# note: --steps {steps} exceeds {len(seq)} positions, clamping")
		steps = len(seq)
	pdb_out = arg("--pdb")
	is_control = ref == "protein_g"

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	nbreak = seq.count(CHAIN_BREAK)

	print(f"# reference: zero_polymer_{ref} (L={len(seq) - nbreak} residues"
		  + (f" + {nbreak} chainbreak, {len(seq)} positions)" if nbreak else ")"))
	print(f"# decoding steps: {steps}   device: {device}")
	if is_control:
		print(f"# POSITIVE CONTROL: must reach ptm >= {PTM_CONTROL} with 0 clashes,")
		print(f"# otherwise the harness is at fault and no other run from this")
		print(f"# script means anything.")
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
	if len(out.sequence) != len(seq):
		print(f"[WARN] {len(out.sequence)} positions returned, {len(seq)} sent "
			  f"-- indices below are NOT trustworthy.")

	label = make_label(ref, out.sequence)
	spans = chain_spans(ref, out.sequence)
	L = coords.shape[0]

	# Residue positions only: the chainbreak is not a residue, and "|" is not
	# glycine so infer_cbeta() hands it a CB derived from meaningless
	# backbone. Earlier runs let it into the CB matrix and the clash count.
	is_res = torch.zeros(L, dtype=torch.bool)
	for _, start, end in spans:
		is_res[start:min(end, L)] = True

	# ------------------------------------------------------------- plddt
	# The diagnostic that separates the two failure modes: domains that
	# folded but were placed wrongly relative to each other (good plddt,
	# bad ptm -- the obligate-trimer signature) from nothing folding at all.
	plddt = None if out.plddt is None else out.plddt.detach().to("cpu").float()
	print(f"\n--- confidence ---")
	print(f"  ptm   {'n/a' if ptm is None else f'{ptm:.4f}'}   (floor {PTM_FLOOR}"
		  + (f", control needs {PTM_CONTROL}" if is_control else "") + ")")
	if plddt is None:
		print(f"  plddt n/a -- decoder returned none")
	else:
		# ESM3 may hand back plddt on 0-1 or 0-100; normalise and say which.
		scale = 100.0 if float(plddt.max()) <= 1.0 else 1.0
		pl = plddt * scale
		print(f"  plddt reported on a 0-{'1' if scale == 100.0 else '100'} scale,"
			  f" shown here as 0-100")
		print(f"  plddt mean (all residues)   {float(pl[is_res].mean()):6.2f}")
		for name, start, end in spans:
			seg = pl[start:min(end, L)]
			if len(seg):
				frac = float((seg >= PLDDT_GOOD).float().mean()) * 100.
				print(f"  plddt mean {name:<12s}     {float(seg.mean()):6.2f}"
					  f"   ({frac:5.1f}% of residues >= {PLDDT_GOOD:.0f})")
		frac_all = float((pl[is_res] >= PLDDT_GOOD).float().mean()) * 100.
		print(f"  residues >= {PLDDT_GOOD:.0f}            {frac_all:5.1f}%")
		print(f"  READ: high plddt with low ptm => domains folded, their")
		print(f"        RELATIVE PLACEMENT failed. Low both => nothing folded.")

	# -------------------------------------------------------------- gate
	cb = coords[:, atom_order["CB"], :]
	ok = torch.isfinite(cb).all(dim=-1) & is_res
	safe = torch.where(ok[:, None], cb, torch.zeros_like(cb))
	d_cb = torch.cdist(safe, safe)
	sep = (torch.arange(L)[:, None] - torch.arange(L)[None, :]).abs()
	valid = ok[:, None] & ok[None, :] & (sep >= 2) & torch.triu(
		torch.ones(L, L, dtype=torch.bool), diagonal=1
	)
	clash_mask = (d_cb < CB_CLASH) & valid
	clashes = int(clash_mask.sum())
	min_cb = float(d_cb[valid].min()) if bool(valid.any()) else float("nan")

	print(f"\n--- geometry sanity ---")
	print(f"  min CB-CB (|i-j|>=2)      {min_cb:6.2f} A   (clash below {CB_CLASH})")
	print(f"  CB-CB clashes              {clashes}   (a physical fold has ~0)")

	# Provenance check, so the clash count can be trusted or dismissed.
	n_nan = int((~torch.isfinite(cb).all(dim=-1) & is_res).sum())
	n_gly = sum(1 for i, a in enumerate(out.sequence) if a == "G" and i < L)
	print(f"  CB absent (NaN) at {n_nan} residues; {n_gly} glycines in sequence")
	print(f"    (infer_cbeta writes NaN at every G by default, so G is excluded")
	print(f"     from all CB statistics by construction -- if these two numbers")
	print(f"     agree, the clashes above are between REAL inferred CBs.)")
	if clashes:
		idx = clash_mask.nonzero()
		print(f"  worst clashing pairs:")
		order = torch.argsort(d_cb[clash_mask])
		for k in order[:8].tolist():
			i, j = int(idx[k][0]), int(idx[k][1])
			print(f"    {float(d_cb[i, j]):5.2f} A  "
				  f"{label(i):>9s} {out.sequence[i]} -- "
				  f"{label(j):<9s} {out.sequence[j]}   |i-j|={abs(i - j)}")

	bad = []
	if ptm is not None and ptm < (PTM_CONTROL if is_control else PTM_FLOOR):
		bad.append(f"ptm {ptm:.3f} below {PTM_CONTROL if is_control else PTM_FLOOR}")
	if clashes:
		bad.append(f"{clashes} CB-CB clashes")

	if is_control:
		print(f"\n  CONTROL {'FAILED' if bad else 'PASSED'}"
			  + (f": {'; '.join(bad)}." if bad else "."))
		if bad:
			print(f"  The harness cannot fold protein_g, so it cannot be trusted")
			print(f"  on zero_polymer either. Treat every cleaved/mature number")
			print(f"  as void until this passes. First suspect:")
			print(f"  init_structure_config(condition_on_coordinates_only=True).")
		else:
			print(f"  Harness is sound; zero_polymer results describe the TARGET.")
	elif bad:
		print(f"\n  [UNTRUSTWORTHY] {'; '.join(bad)}.")
		print(f"  Distances below describe a structure the model does not")
		print(f"  believe in. Check --ref protein_g passes before reading them.")
	else:
		print(f"\n  [OK] fold passes the sanity gate; distances are meaningful.")

	# ------------------------------------------------------------ bridge
	idx = bridge_indices(ref)
	if idx is None:
		print(f"\n(ref {ref!r} is a single chain -- no interchain bridge)")
	else:
		i1, i2 = idx
		(c1, p1), (c2, p2) = DISULFIDE_BRIDGES[0]
		for i, chain, pos in ((i1, c1, p1), (i2, c2, p2)):
			got = out.sequence[i] if i < len(out.sequence) else "?"
			flag = "" if got == "C" else "   <-- NOT a cysteine, indices are off!"
			print(f"  {chain}:{pos} -> position {i}: {got}{flag}")
		sg = atom_distance(coords, i1, i2, "SG")
		cbd = atom_distance(coords, i1, i2, "CB")
		ca = atom_distance(coords, i1, i2, "CA")
		print(f"\n--- the bridge: {c1}:Cys{p1} <-> {c2}:Cys{p2} ---")
		print(f"  SG-SG  {fmt(sg)} A   (bonded ~2.05, cutoff {SG_BONDED})")
		print(f"  CB-CB  {fmt(cbd)} A   (bonded <{CB_BONDED})")
		print(f"  CA-CA  {fmt(ca)} A   (bonded ~5.5-6.8)")
		if plddt is not None:
			s = 100.0 if float(plddt.max()) <= 1.0 else 1.0
			print(f"  plddt at the two anchors: "
				  f"{float(plddt[i1]) * s:.1f} / {float(plddt[i2]) * s:.1f}")
		if sg is not None:
			formed, basis = sg <= SG_BONDED, f"SG-SG={sg:.2f} A"
		elif cbd is not None:
			formed, basis = cbd <= CB_BONDED, f"CB-CB={cbd:.2f} A, no SG predicted"
		else:
			formed, basis = None, "neither SG nor CB predicted"
		verdict = "cannot tell" if formed is None else ("FORMED" if formed else "NOT formed")
		print(f"\n  verdict: bridge {verdict}  [{basis}]"
			  + ("  (but see UNTRUSTWORTHY above)" if bad else ""))

	# --------------------------------------------------------- pair scan
	cys = [i for i, a in enumerate(out.sequence) if a == "C" and i < L and bool(ok[i])]
	pairs = sorted(
		(float(d_cb[a, b]), a, b)
		for n, a in enumerate(cys) for b in cys[n + 1:]
	)
	if not pairs:
		print(f"\n(no cysteine pairs with inferred CB -- "
			  f"{'protein_g has no cysteines, as expected' if is_control else 'unexpected'})")
	else:
		target = set(idx) if idx else set()
		print(f"\n--- all {len(pairs)} cysteine pairs by CB-CB, closest first ---")
		for rank, (d, a, b) in enumerate(pairs, start=1):
			if rank > 12 and {a, b} != target:
				continue
			mark = "  <== the HA1-HA2 bridge" if {a, b} == target else ""
			tag = "!" if d < CB_CLASH else ("*" if d <= CB_BONDED else " ")
			print(f"  {rank:3d}. {tag} {d:6.2f} A  "
				  f"{label(a):>9s} -- {label(b):<9s}{mark}")
		n_bond = sum(1 for d, _, _ in pairs if CB_CLASH <= d <= CB_BONDED)
		n_cl = sum(1 for d, _, _ in pairs if d < CB_CLASH)
		print(f"\n  {n_bond} pairs in plausible bonding range "
			  f"[{CB_CLASH}, {CB_BONDED}] A (*)")
		print(f"  {n_cl} pairs CLASHING below {CB_CLASH} A (!) -- broken geometry,")
		print(f"  not disulfides")

	if pdb_out:
		out.to_pdb(pdb_out)
		print(f"\nwrote {pdb_out}")
		if nbreak:
			pc = out.to_protein_complex()
			print(f"to_protein_complex() -> {len(list(pc.chain_iter()))} chains "
				  f"(expected {nbreak + 1})")


if __name__ == "__main__":
	main()
