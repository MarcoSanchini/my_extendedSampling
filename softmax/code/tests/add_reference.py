"""
Adds a UniProt entry to references.py as a new reference, sequence and all.

The four structure-decode controls (DEVLOG 2026-11-04) come from UniProt, and
the cloud container cannot reach rest.uniprot.org, so this script exists to add
them from a machine that can. It writes the sequence into references.py as a
literal, like every other reference there, so later runs need no network.

Each entry records where it came from: the accession, the mature-chain range it
was cut to (UniProt's first CHAIN feature, so no signal peptide or propeptide)
and the first few PDB entries UniProt cross-references, which is what makes it
easy to pull an experimental structure for the Rg comparison.

    cd softmax/code
    python tests/add_reference.py                 # the four controls
    python tests/add_reference.py --print         # show the block, change nothing
    python tests/add_reference.py --entries P69905:hemoglobin_alpha
    python tests/add_reference.py --entries P08659 --no-mature

An entry is ACCESSION, or ACCESSION:NAME to choose the reference key, or
ACCESSION:NAME:START:END to give the range yourself (1-based, inclusive).
Re-running is safe: an entry whose key is already in references.py is skipped.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCES_PY = os.path.join(HERE, "references.py")
FASTA_URL = "https://rest.uniprot.org/uniprotkb/{}.fasta"
JSON_URL = "https://rest.uniprot.org/uniprotkb/{}.json?fields=ft_chain,xref_pdb,protein_name"
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

# accession:key, the controls of decode_controls.py.
DEFAULT_ENTRIES = [
	"P0AEX9:maltose_binding_protein",
	"P0DUB6:salivary_amylase",
	"P08659:firefly_luciferase",
	"P02768:serum_albumin",
]


def fetch(accession):
	"""(sequence, protein name, (start, end) or None, [pdb ids])."""
	with urllib.request.urlopen(FASTA_URL.format(accession), timeout=60) as r:
		text = r.read().decode()
	seq = "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))
	if not seq:
		raise SystemExit(f"{accession}: UniProt returned no sequence; the accession is "
			f"probably obsolete or demerged. See https://www.uniprot.org/uniprotkb/{accession}")
	with urllib.request.urlopen(JSON_URL.format(accession), timeout=60) as r:
		data = json.loads(r.read().decode())
	chain = None
	for f in data.get("features", []):
		if f.get("type") == "Chain":
			loc = f["location"]
			if loc["start"].get("value") and loc["end"].get("value"):
				chain = (int(loc["start"]["value"]), int(loc["end"]["value"]))
			break
	name = (data.get("proteinDescription", {}).get("recommendedName", {})
		.get("fullName", {}).get("value") or accession)
	pdbs = [x["id"] for x in data.get("uniProtKBCrossReferences", []) if x.get("database") == "PDB"]
	return seq, name, chain, pdbs


def block_for(accession, key, seq, name, rng, pdbs):
	const = key.upper()
	lines = "".join(f'\t"{seq[i:i + 60]}"\n' for i in range(0, len(seq), 60))
	where = f", mature chain {rng[0]}:{rng[1]}" if rng else ""
	pdb = f" PDB: {', '.join(pdbs[:6])}{' ...' if len(pdbs) > 6 else ''}." if pdbs else ""
	return (f"\n# {name}, {len(seq)} aa. UniProt {accession}{where}.{pdb}\n"
		f"# Structure-decode control, added by tests/add_reference.py.\n"
		f"{const} = (\n{lines})\n"), const


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--entries", nargs="+", default=DEFAULT_ENTRIES,
		help="ACCESSION[:NAME[:START:END]]")
	ap.add_argument("--no-mature", action="store_true",
		help="keep the whole entry instead of its first CHAIN feature")
	ap.add_argument("--print", dest="show", action="store_true",
		help="print the block instead of editing references.py")
	args = ap.parse_args()

	src = open(REFERENCES_PY).read()
	blocks, dict_lines = [], []
	for entry in args.entries:
		parts = entry.split(":")
		accession = parts[0]
		key = parts[1] if len(parts) > 1 and parts[1] else accession.lower()
		if f'"{key}":' in src:
			print(f"{key}: already in references.py, skipped")
			continue
		seq, name, chain, pdbs = fetch(accession)
		if len(parts) == 4:
			chain = (int(parts[2]), int(parts[3]))
		elif args.no_mature:
			chain = None
		if chain and chain != (1, len(seq)):
			start, end = chain
			if not 1 <= start <= end <= len(seq):
				raise SystemExit(f"{accession}: range {start}:{end} outside 1:{len(seq)}")
			seq = seq[start - 1:end]
		else:
			chain = None
		bad = sorted(set(seq) - set(CANONICAL))
		if bad:
			raise SystemExit(f"{accession}: non-canonical residues {bad}; this alphabet is the 20 canonical ones")
		block, const = block_for(accession, key, seq, name, chain, pdbs)
		blocks.append(block)
		dict_lines.append(f'\t"{key}": {const},\n')
		print(f"{key}: {len(seq)} aa from {accession}" + (f", PDB {pdbs[0]}" if pdbs else ""))

	if not blocks:
		print("Nothing to add.")
		return
	if args.show:
		print("".join(blocks) + "".join(dict_lines))
		return

	anchor = "\nREFERENCES = {"
	if src.count(anchor) != 1:
		raise SystemExit("could not find the REFERENCES dict in references.py; add the block by hand")
	src = src.replace(anchor, "".join(blocks) + anchor)
	# The dict's own closing brace: the first line that is exactly "}" after it.
	i = src.index(anchor)
	close = src.index("\n}\n", i)
	src = src[:close + 1] + "".join(dict_lines) + src[close + 1:]
	open(REFERENCES_PY, "w").write(src)
	print(f"\nWrote {REFERENCES_PY}. Check it with:\n"
		f"  python -c \"import sys; sys.path.insert(0,'tests'); import references as r; "
		f"print({{k: len(v) for k, v in r.REFERENCES.items()}})\"")


if __name__ == "__main__":
	main()
