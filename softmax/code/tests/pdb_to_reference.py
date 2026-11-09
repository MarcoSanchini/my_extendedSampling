"""
Turns one chain of a local .pdb into a references.py block.

Why this exists next to add_reference.py: add_reference.py fetches a sequence
from UniProt, which the cloud container cannot reach, and a structure handed
over by a collaborator does not come with an accession anyway. This script
reads the coordinates that are actually in the file, so it needs no network.

What it refuses to hide: a crystal structure's ATOM records are the residues
that were OBSERVED. Disordered loops are simply absent, so the sequence read
off the coordinates is the native sequence MINUS those residues, and writing
it as one string silently joins across each gap. Every gap and every insertion
code is therefore reported, and the generated block records them in comments,
because a reference that quietly closes a 4-residue loop is a reference whose
decode and whose site numbering are both wrong in a way nothing downstream
would reveal.

    cd softmax/code
    python tests/pdb_to_reference.py path/to/file.pdb                # report only
    python tests/pdb_to_reference.py path/to/file.pdb --chain A --name na_head
    python tests/pdb_to_reference.py path/to/file.pdb --name na_head --write

--write inserts the block and its REFERENCES entry into references.py; without
it, the block goes to stdout and nothing is modified. A name already present
in references.py is never overwritten.

If SEQRES records are present (they often are not, in a file that has been
stripped to one chain), the chain's full deposited sequence is reported too
and --use-seqres writes that instead of the observed one: same construct, with
the disordered residues restored.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCES_PY = os.path.join(HERE, "references.py")

THREE_TO_ONE = {
	"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
	"GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
	"LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
	"SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
	# Selenomethionine is methionine as far as the sequence is concerned; it is
	# a crystallographic phasing trick, not a 21st amino acid.
	"MSE": "M",
}


def residue_table(path):
	"""THREE_TO_ONE plus whatever this file's MODRES records declare.

	A chemically modified residue (phosphoserine, an oxidised or
	arsenic-adducted cysteine, ...) is written under its own three-letter
	name and usually as HETATM, so a reader that knows only the 20 canonical
	names drops it. Dropping it is the worst of the options: the residue is
	part of the chain, so the sequence comes out one residue short and the
	numbering gap looks like a disordered loop. 1RA6 is the case in point --
	four of its cysteines are CAS, S-(dimethylarsenic)cysteine from the
	crystallisation buffer, and reading them as absent would have written a
	reference missing four cysteines without saying so.

	A PDB file declares the parent of each modified residue itself, in its
	MODRES records, so take the mapping from the file rather than from a
	hard-coded list that would always be missing the next one.
	"""
	table = dict(THREE_TO_ONE)
	for line in open(path):
		if not line.startswith("MODRES"):
			continue
		name, parent = line[12:15].strip(), line[24:27].strip()
		if parent in THREE_TO_ONE:
			table[name] = THREE_TO_ONE[parent]
	return table


def read_chain(path, chain=None):
	"""(residues, seqres, chains) for one chain.

	residues is a list of (resnum, icode, one-letter) in file order, seqres the
	deposited sequence for that chain if SEQRES records exist, chains the set of
	chains with ATOM records.
	"""
	table = residue_table(path)
	residues, seen, chains, seqres = [], set(), [], {}
	for line in open(path):
		if line.startswith("SEQRES"):
			ch = line[11]
			seqres.setdefault(ch, []).extend(line[19:].split())
			continue
		if not line.startswith(("ATOM", "HETATM")):
			continue
		name = line[17:20].strip()
		ch = line[21]
		if ch not in chains:
			chains.append(ch)
		if chain is not None and ch != chain:
			continue
		if name not in table:
			continue
		# An altloc duplicates a whole residue; keep whichever comes first.
		key = (int(line[22:26]), line[26].strip())
		if key in seen:
			continue
		seen.add(key)
		residues.append((key[0], key[1], table[name]))
	chain_seqres = None
	if chain in seqres or (chain is None and len(seqres) == 1):
		names = seqres[chain] if chain in seqres else next(iter(seqres.values()))
		chain_seqres = "".join(table.get(n, "X") for n in names)
	return residues, chain_seqres, chains


def gaps_of(residues):
	"""Breaks in the residue numbering: [(last_before, first_after, n_missing)]."""
	out = []
	for (a, _, _), (b, _, _) in zip(residues, residues[1:]):
		if b - a > 1:
			out.append((a, b, b - a - 1))
	return out


def block_for(name, seq, source, notes):
	const = name.upper()
	lines = "".join(f'\t"{seq[i:i + 60]}"\n' for i in range(0, len(seq), 60))
	comment = "".join(f"# {n}\n" for n in notes)
	return (f"\n# {source}\n{comment}{const} = (\n{lines})\n"), const


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("pdb")
	ap.add_argument("--chain", help="chain id; the only chain present by default")
	ap.add_argument("--name", help="references.py key, e.g. na_head; required with --write")
	ap.add_argument("--use-seqres", action="store_true",
		help="write the deposited SEQRES sequence instead of the observed residues")
	ap.add_argument("--write", action="store_true", help="edit references.py in place")
	args = ap.parse_args()

	residues, seqres, chains = read_chain(args.pdb, args.chain)
	if not chains:
		raise SystemExit(f"{args.pdb}: no ATOM records")
	if args.chain is None and len(chains) > 1:
		raise SystemExit(f"{args.pdb}: chains {' '.join(chains)}; pick one with --chain")
	chain = args.chain or chains[0]
	if not residues:
		raise SystemExit(f"{args.pdb}: chain {chain} has no standard amino acids")

	observed = "".join(r[2] for r in residues)
	gaps = gaps_of(residues)
	icodes = [(n, i) for n, i, _ in residues if i]
	first, last = residues[0][0], residues[-1][0]

	print(f"{os.path.basename(args.pdb)} chain {chain}: {len(observed)} residues observed, "
		f"numbered {first}-{last}")
	if gaps:
		missing = sum(g[2] for g in gaps)
		print(f"  {len(gaps)} gap(s), {missing} residue(s) unobserved: "
			+ ", ".join(f"{a}->{b} ({n})" for a, b, n in gaps))
		print("  the observed sequence JOINS across these, so it is not the native chain")
	else:
		print("  no gaps: the observed residues are numbered consecutively")
	if icodes:
		print("  insertion codes: " + ", ".join(f"{n}{i}" for n, i in icodes))
	if seqres:
		print(f"  SEQRES present: {len(seqres)} residues deposited for this chain")
	else:
		print("  no SEQRES records in this file")
	print(f"\n{observed}\n")

	if not args.name:
		print("Pass --name to generate a references.py block.")
		return

	seq, kind = observed, "observed residues only"
	if args.use_seqres:
		if not seqres:
			raise SystemExit("--use-seqres: this file has no SEQRES records for that chain")
		seq, kind = seqres, "deposited SEQRES sequence"
	bad = sorted(set(seq) - set("ACDEFGHIKLMNPQRSTVWY"))
	if bad:
		raise SystemExit(f"non-canonical residues {bad}; this alphabet is the 20 canonical ones")

	notes = [f"Read from {os.path.basename(args.pdb)} chain {chain} ({kind}), "
		f"numbered {first}-{last}.", "Added by tests/pdb_to_reference.py."]
	if gaps and not args.use_seqres:
		notes.insert(1, "UNOBSERVED residues, closed up in this string: "
			+ ", ".join(f"{a}->{b} ({n})" for a, b, n in gaps) + ".")
	if icodes:
		notes.insert(1, "Insertion codes in the file: "
			+ ", ".join(f"{n}{i}" for n, i in icodes) + ".")
	block, const = block_for(args.name, seq, f"{args.name}, {len(seq)} aa.", notes)

	if not args.write:
		print(block + f'\t"{args.name}": {const},')
		return

	src = open(REFERENCES_PY).read()
	if f'"{args.name}":' in src:
		raise SystemExit(f"{args.name} is already in references.py; pick another name")
	anchor = "\nREFERENCES = {"
	if src.count(anchor) != 1:
		raise SystemExit("could not find the REFERENCES dict; add the block by hand")
	src = src.replace(anchor, block + anchor)
	close = src.index("\n}\n", src.index(anchor))
	src = src[:close + 1] + f'\t"{args.name}": {const},\n' + src[close + 1:]
	open(REFERENCES_PY, "w").write(src)
	print(f"Wrote {REFERENCES_PY}: {args.name}, {len(seq)} aa")


if __name__ == "__main__":
	main()
