"""
Calibration of the ESM3 structure decode on proteins that ARE known to fold:
what pLDDT and radius of gyration does this exact decode path give for
well-characterised, monomeric, single-domain-ish proteins of roughly the same
length as zero_polymer (566)?

Why: for zero_polymer (influenza HA0) the reference sequence itself decodes to
pLDDT 0.51-0.63 and Rg 45-70 A depending on the step count (DEVLOG,
2026-11-02), so Rg/pLDDT could not be read as "the sampled sequences keep the
native fold". That diagnosis assumed the decoder WOULD be confident on a
protein it can fold. Nothing in this repository has tested that assumption.
This script tests it, with the same model, the same generate() path and the
same metrics as compute_structural_metrics.py / structure_zero_polymer.py
decode-ref.

The defaults are monomers of 396-609 residues, each with experimental
structures in the PDB and no obligate partner:

  P04745  human salivary alpha-amylase
  P08659  firefly luciferase (Photinus pyralis)
  P02768  human serum albumin
  P0AEX9  E. coli maltose-binding protein

All four are around 400-600 residues, and each is decoded as its mature chain
(UniProt's first CHAIN feature, looked up per entry, so no signal peptide or
propeptide is included -- the mistake that would make this test look like the
HA0 one). Pass --no-mature for the whole entry.

plus, for scale, the zero_polymer reference itself and --n-random random
sequences of the same length (the floor: whatever pLDDT the decoder gives to
a sequence with no structure at all).

Sequences come from UniProt (rest.uniprot.org) unless --fasta is given, so the
machine running this needs network access to UniProt, or a local FASTA.

Usage (on the GPU machine):
    python decode_controls.py --num-steps 8 32 128 --out nsites_study/structure/controls.csv
    python decode_controls.py --entries P04745:16:511 P08659 --num-steps 32
    python decode_controls.py --fasta my_controls.fa --num-steps 8 32

An entry is ACCESSION, or ACCESSION:START:END to keep only residues
START..END (1-based, inclusive). With --fasta, an entry is a record id in that
file (a UniProt-style 'sp|P04745|...' id is also keyed by its accession), and
the mature-chain lookup is skipped, so give the range yourself if you need one.

Writes a CSV (label, Rg_ca, Rg_allatom, mean_plddt, num_steps, repeat,
length) next to whatever --out says, and prints a summary table.
"""
import argparse
import os
import sys
import urllib.request

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
for p in (MONTE_CARLO_DIR, CODE_DIR, CUSTOMS_DIR, os.path.join(CODE_DIR, "tests")):
	if p not in sys.path:
		sys.path.insert(0, p)

import numpy as np
import pandas as pd

DEFAULT_ENTRIES = ["P04745", "P08659", "P02768", "P0AEX9"]
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"
UNIPROT_JSON = "https://rest.uniprot.org/uniprotkb/{}.json?fields=ft_chain"
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
	"""{record id: sequence}; the id is the first whitespace-separated token,
	and a UniProt-style 'sp|P04745|AMY1_HUMAN' id is also keyed by P04745."""
	seqs, name, chunks = {}, None, []
	def flush():
		if name is not None:
			seqs[name] = "".join(chunks)
	with open(path) as fh:
		for line in fh:
			if line.startswith(">"):
				flush()
				name, chunks = line[1:].split()[0], []
			elif name is not None:
				chunks.append(line.strip())
	flush()
	for key in list(seqs):
		parts = key.split("|")
		if len(parts) >= 2 and parts[1] not in seqs:
			seqs[parts[1]] = seqs[key]
	return seqs


def fetch_uniprot(accession):
	with urllib.request.urlopen(UNIPROT_URL.format(accession), timeout=60) as r:
		text = r.read().decode()
	return "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))


def mature_range(accession):
	"""(start, end) of the entry's first CHAIN feature, i.e. the mature protein
	without signal peptide and propeptides, or None if UniProt annotates none.
	Looked up rather than hard-coded, so the numbering is UniProt's own."""
	import json
	with urllib.request.urlopen(UNIPROT_JSON.format(accession), timeout=60) as r:
		data = json.loads(r.read().decode())
	for f in data.get("features", []):
		if f.get("type") == "Chain":
			loc = f["location"]
			start, end = loc["start"].get("value"), loc["end"].get("value")
			if start and end:
				return int(start), int(end)
	return None


def resolve(entry, local, mature=False):
	"""'P04745' or 'P04745:16:511' -> (label, sequence), sliced and checked."""
	parts = entry.split(":")
	acc = parts[0]
	seq = local.get(acc) if local else None
	if seq is None:
		if local:
			raise SystemExit(f"{acc} is not in the FASTA file (ids: {', '.join(sorted(local))})")
		seq = fetch_uniprot(acc)
	label = acc
	if len(parts) == 1 and mature and not local:
		rng = mature_range(acc)
		if rng and rng != (1, len(seq)):
			parts = [acc, str(rng[0]), str(rng[1])]
	if len(parts) == 3:
		start, end = int(parts[1]), int(parts[2])
		if not 1 <= start <= end <= len(seq):
			raise SystemExit(f"{acc}: range {start}:{end} outside 1:{len(seq)}")
		seq, label = seq[start - 1:end], f"{acc}[{start}:{end}]"
	bad = sorted(set(seq) - set(CANONICAL))
	if bad:
		# U (selenocysteine), X and friends have no token here; the sampler's
		# alphabet is the 20 canonical residues.
		raise SystemExit(f"{label}: non-canonical residues {bad}; use --fasta with a cleaned sequence")
	return label, seq


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--entries", nargs="+", default=DEFAULT_ENTRIES,
		help="ACCESSION or ACCESSION:START:END (1-based, inclusive)")
	ap.add_argument("--no-mature", action="store_true",
		help="decode the whole UniProt entry instead of its first CHAIN feature "
			 "(by default the signal peptide and propeptides are dropped)")
	ap.add_argument("--fasta", help="read the sequences from this file instead of UniProt")
	ap.add_argument("--num-steps", type=int, nargs="+", default=[8, 32, 128],
		help="structure-generation steps per decode (as in decode-ref)")
	ap.add_argument("--repeats", type=int, default=1, help="decodes per sequence (the decode is deterministic)")
	ap.add_argument("--n-random", type=int, default=2, help="random sequences of length --random-length")
	ap.add_argument("--random-length", type=int, default=566)
	ap.add_argument("--ref", default="zero_polymer", help="reference to include; '' to skip")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--out", default="nsites_study/structure/controls.csv")
	args = ap.parse_args()

	import torch
	from compute_structural_metrics import decode_structure_with_mask, radius_of_gyration, CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from references import get_reference

	local = read_fasta(args.fasta) if args.fasta else None
	seqs = [resolve(e, local, mature=not args.no_mature) for e in args.entries]
	if args.ref:
		seqs.append((f"reference({args.ref})", get_reference(args.ref)[1]))
	rng = np.random.default_rng(args.seed)
	seqs += [(f"random{i}", "".join(rng.choice(list(CANONICAL), args.random_length)))
		for i in range(args.n_random)]
	print("Sequences:")
	for label, seq in seqs:
		print(f"  {label}: {len(seq)} residues")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)

	rows = []
	for num_steps in args.num_steps:
		config = init_structure_config(num_steps=num_steps)
		for label, seq in seqs:
			for r in range(args.repeats):
				torch.manual_seed(args.seed + r)
				eprot = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
				eprot.expand()
				positions, mask, plddt = decode_structure_with_mask(sampler.model, eprot, config)
				row = [label, radius_of_gyration(positions[:, CA_IDX, :]), radius_of_gyration(positions[mask]),
					plddt.mean().item() if plddt is not None else float("nan"), num_steps, r, len(seq)]
				rows.append(row)
				print(f"  {label} steps={num_steps} repeat={r}: "
					f"Rg_ca={row[1]:.3f}  Rg_allatom={row[2]:.3f}  mean_plddt={row[3]:.3f}", flush=True)

	df = pd.DataFrame(rows, columns=["label", "Rg_ca", "Rg_allatom", "mean_plddt", "num_steps", "repeat", "length"])
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	df.to_csv(args.out, index=False)
	print()
	print(df.groupby(["label", "num_steps"])[["Rg_ca", "Rg_allatom", "mean_plddt"]]
		.agg(["mean", "std"]).round(3).to_string())
	print(f"\nSaved -> {args.out}")
	# A folded globular protein of N residues has Rg about 2.2 * N**0.38 A
	# (Flory, compact globule); printed as a scale for the Rg column.
	for label, seq in seqs:
		print(f"  globule Rg estimate for {label} ({len(seq)} aa): {2.2 * len(seq) ** 0.38:.1f} A")


if __name__ == "__main__":
	main()
