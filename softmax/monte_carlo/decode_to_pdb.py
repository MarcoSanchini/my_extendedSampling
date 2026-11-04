"""
Decode one sequence with ESM3 and SAVE the predicted structure as a .pdb file,
with the per-residue pLDDT (0-100) in the B-factor column.

Why this exists: decode_controls.py and decode_trimer.py decode structures but
only keep summary numbers (Rg, mean pLDDT); they never write the coordinates.
To look at a prediction, or to compare it with an experimental structure
(compare_structures.py), the coordinates have to be on disk. This uses the
same model, the same generate() path (decode_structure_with_mask) and the same
step-count convention as those scripts, so the structure written here is the
one their numbers describe.

pLDDT goes in the B-factor column because that is where AlphaFold, ESMFold and
every viewer expect it: ChimeraX "color bfactor palette alphafold" and PyMOL
"spectrum b" colour the model by confidence straight away.

What to decode (pick one; default --ref zero_polymer):
  --ref NAME            a reference from code/tests/references.py
  --uniprot ACC[:S:E]   a UniProt entry (mature chain by default, as in
                        decode_controls.py); needs network access to UniProt
  --seq SEQUENCE        a raw sequence
  --residues S:E        keep only residues S..E (1-based, inclusive) of the
                        above; the PDB keeps the ORIGINAL numbering, so residue
                        17 is still called 17 in the file

Usage (on the GPU machine, from softmax/monte_carlo):
    python decode_to_pdb.py --ref zero_polymer --num-steps 32 --out structure_compare/ha0_pred.pdb
    python decode_to_pdb.py --uniprot P0AEX9 --num-steps 32 --out structure_compare/mbp_pred.pdb
"""
import argparse
import os
import sys

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
for p in (MONTE_CARLO_DIR, CODE_DIR, CUSTOMS_DIR, os.path.join(CODE_DIR, "tests")):
	if p not in sys.path:
		sys.path.insert(0, p)

import numpy as np

AA1TO3 = {
	"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
	"G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
	"P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def write_pdb(path, sequence, positions, mask, bfactor, atom_types, first_resseq=1, chain="A"):
	"""Plain PDB writer: one ATOM line per present atom (atom37 layout), the
	B-factor column set per residue. Written by hand so the B-factor is certain
	to be what we put there, whatever the ESM helper does with confidence."""
	lines, serial = [], 1
	for i, aa in enumerate(sequence):
		resname = AA1TO3.get(aa, "UNK")
		for a in np.where(mask[i])[0]:
			name = atom_types[a]
			x, y, z = positions[i, a]
			if not np.isfinite([x, y, z]).all():
				continue
			# PDB atom-name column: 4-letter names start in column 13, shorter in 14
			padded = name if len(name) == 4 else f" {name:<3s}"
			lines.append(f"ATOM  {serial:5d} {padded} {resname} {chain}{first_resseq + i:4d}    "
				f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfactor[i]:6.2f}          {name[0]:>2s}")
			serial += 1
	lines += ["TER", "END"]
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	with open(path, "w") as fh:
		fh.write("\n".join(lines) + "\n")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	src = ap.add_mutually_exclusive_group()
	src.add_argument("--ref", default=None, help="reference name (default zero_polymer)")
	src.add_argument("--uniprot", default=None, help="ACCESSION or ACCESSION:START:END")
	src.add_argument("--seq", default=None, help="raw amino-acid sequence")
	ap.add_argument("--residues", default=None, help="keep residues START:END (1-based, inclusive)")
	ap.add_argument("--no-mature", action="store_true", help="with --uniprot: the whole entry, not the mature chain")
	ap.add_argument("--num-steps", type=int, default=32, help="structure-generation steps (as in decode-ref)")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--out", required=True, help="output .pdb path")
	args = ap.parse_args()

	first = 1
	if args.uniprot:
		from decode_controls import resolve
		label, seq = resolve(args.uniprot, None, mature=not args.no_mature)
		# resolve() labels a slice as ACC[S:E]; keep S as the first residue number
		if "[" in label:
			first = int(label.split("[")[1].split(":")[0])
	elif args.seq:
		label, seq = "custom", args.seq.strip().upper()
	else:
		from references import get_reference
		label, seq = get_reference(args.ref or "zero_polymer")
	if args.residues:
		s, e = map(int, args.residues.split(":"))
		if not 1 <= s <= e <= len(seq):
			raise SystemExit(f"--residues {s}:{e} outside 1:{len(seq)}")
		seq, first = seq[s - 1:e], first + s - 1
		label = f"{label}[{s}:{e}]"
	print(f"Decoding {label}: {len(seq)} residues, num_steps={args.num_steps}", flush=True)

	import torch
	from compute_structural_metrics import decode_structure_with_mask, radius_of_gyration, CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from custom_esm.utils.residue_constants import atom_types

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	config = init_structure_config(num_steps=args.num_steps)
	torch.manual_seed(args.seed)
	eprot = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
	eprot.expand()
	positions, mask, plddt = decode_structure_with_mask(sampler.model, eprot, config)

	pos = positions.float().numpy()
	msk = mask.bool().numpy()
	if plddt is None:
		print("WARNING: the decoder returned no pLDDT; B-factors set to 0")
		pl = np.zeros(len(seq))
	else:
		pl = plddt.float().numpy()
		pl = pl * 100.0 if pl.max() <= 1.0 else pl   # store on the usual 0-100 scale
	write_pdb(args.out, seq, pos, msk, pl, atom_types, first_resseq=first)

	print(f"  mean pLDDT {pl.mean():.1f} / 100   Rg_ca {radius_of_gyration(positions[:, CA_IDX, :]):.2f} A")
	print(f"  residues with pLDDT >= 70: {np.mean(pl >= 70):.0%}   < 50: {np.mean(pl < 50):.0%}")
	print(f"Saved -> {args.out}  (pLDDT 0-100 in the B-factor column, residues numbered from {first})")


if __name__ == "__main__":
	main()
