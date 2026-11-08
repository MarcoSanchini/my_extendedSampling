"""
Three diagnostics on a predicted structure that compare_structures.py does not do.

compare_structures.py answers "how far apart are these two structures" (RMSD,
TM, pLDDT). These three answer "in the terms this project actually cares about":

1. The CONTACT-MAP ENERGY. The effective energy of Zambon et al 2024, which is
   what the sampler minimises, is a difference of contact maps, not an RMSD:

       E = sum_ij |D_ij(pred) - D_ij(exp)| / sum_ij (D_ij(pred) + D_ij(exp))

   with D the contact map and |i-j| > 2. E = 0 means identical maps. Computing
   it between a predicted structure and the experimental one therefore says, in
   the sampler's own units, how far the predicted reference is from the real
   protein -- which is the number that decides whether a target is usable,
   because the sampled landscape is defined relative to the predicted map.

   The paper's map counts a contact when any two heavy atoms are within 4 A.
   decode_to_pdb.py writes backbone plus CB only, so the default here is CA
   within 8 A, the usual coarse equivalent. --heavy-atom switches to the paper's
   definition and requires both files to have side chains.

2. DISULFIDE BRIDGES. Found in the experimental file as CYS pairs whose SG atoms
   are within --ss-cutoff (2.5 A), then measured in the prediction as the CB-CB
   distance of the same pair (a native disulfide is about 3.8-4.1 A CB to CB).
   A predictor that gets the short-range bridges right and the long-range ones
   wrong is telling you which parts of the topology it failed to assemble, and
   in a repeat fold the long-range bridges are usually the ones that close it.

3. The PER-BLOCK RMSD table: deviation along the chain in blocks of --block
   residues, which localises the error without opening a viewer.

Residues are paired by the *_per_residue.csv that compare_structures.py writes,
which carries the correspondence explicitly (predicted number to experimental
chain and number) and so copes with a crystal that is missing residues or
numbers them differently. Without --csv the two files are paired by file order
instead, and the script stops if they do not have the same residue count.

Usage (anywhere, numpy not even required):
    python prediction_diagnostics.py --pred structure_compare/na_head_pred.pdb \
        --exp ../code/bridge_test/pdb/na_head_2HU0_monomer.pdb
    python prediction_diagnostics.py --pred p450cam_pred.pdb --exp 2CPP.pdb --heavy-atom
"""
import argparse
import itertools
import math

STD = {
	"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
	"LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
}


def read(path):
	"""[(reskey, resname, {atom name: xyz})] in file order, standard residues, first altloc."""
	res, order = {}, []
	for line in open(path):
		if not line.startswith(("ATOM", "HETATM")):
			continue
		name = line[17:20].strip()
		if name not in STD:
			continue
		if line.startswith("HETATM") and name != "MSE":
			continue
		key = (line[21], int(line[22:26]), line[26].strip())
		if key not in res:
			res[key] = {}
			order.append((key, name))
		atom = line[12:16].strip()
		if atom not in res[key]:
			res[key][atom] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
	return [(k, n, res[k]) for k, n in order]


def contact_map(residues, heavy_atom, ca_cutoff):
	"""{(i, j)} of residue indices in contact, |i-j| > 2, 0-based."""
	n = len(residues)
	if heavy_atom:
		coords = [[xyz for a, xyz in r[2].items() if not a.startswith("H")] for r in residues]
		cut = 4.0
	else:
		coords = [[r[2]["CA"]] if "CA" in r[2] else [] for r in residues]
		cut = ca_cutoff
	# Bin the atoms so this stays usable at 600 residues without numpy.
	cells = {}
	for i, atoms in enumerate(coords):
		for xyz in atoms:
			cells.setdefault(tuple(int(c // cut) for c in xyz), []).append((i, xyz))
	out = set()
	for cell, items in cells.items():
		neigh = [
			(cell[0] + dx, cell[1] + dy, cell[2] + dz)
			for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
		]
		for i, p in items:
			for c in neigh:
				for j, q in cells.get(c, ()):
					if j - i > 2 and math.dist(p, q) < cut:
						out.add((i, j))
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--pred", required=True)
	ap.add_argument("--exp", required=True)
	ap.add_argument("--heavy-atom", action="store_true",
		help="contacts as in the paper: any heavy atoms within 4 A (needs side chains in both files)")
	ap.add_argument("--ca-cutoff", type=float, default=8.0, help="A, CA-CA contact cutoff (default)")
	ap.add_argument("--ss-cutoff", type=float, default=2.5, help="A, SG-SG cutoff for a disulfide")
	ap.add_argument("--block", type=int, default=20, help="residues per row of the deviation table")
	ap.add_argument("--csv", help="a *_per_residue.csv from compare_structures.py: gives the "
		"residue pairing and the deviations for the block table")
	args = ap.parse_args()

	pred, exp = read(args.pred), read(args.exp)
	print(f"predicted:    {args.pred}  {len(pred)} residues")
	print(f"experimental: {args.exp}  {len(exp)} residues")
	dev = None
	if args.csv:
		import csv as _csv
		rows = [r for r in _csv.DictReader(open(args.csv))]
		pi = {(k[0], k[1], k[2]): i for i, (k, _, _) in enumerate(pred)}
		ei = {(k[0], k[1], k[2]): i for i, (k, _, _) in enumerate(exp)}
		pairs, dev = [], []
		def key(chain, resseq):
			"""('A', '169A') -> ('A', 169, 'A'); an unpaired row has an empty number."""
			s = (resseq or "").strip()
			if not s:
				return None
			num = s.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
			return (chain, int(num), s[len(num):].strip())

		for r in rows:
			pk, ek = key(r["pred_chain"], r["pred_resseq"]), key(r["exp_chain"], r["exp_resseq"])
			if pk and ek and pk in pi and ek in ei:
				pairs.append((pi[pk], ei[ek]))
				dev.append(float(r["ca_deviation"]))
		if not pairs:
			raise SystemExit(f"{args.csv}: none of its rows matched residues in both files")
		pred = [pred[a] for a, _ in pairs]
		exp = [exp[b] for _, b in pairs]
		print(f"paired by {args.csv}: {len(pairs)} residues")
	elif len(pred) != len(exp):
		raise SystemExit("the two files do not have the same number of residues; pass the "
			"--csv that compare_structures.py wrote, which carries the pairing")

	# 1. contact-map energy
	cp, ce = contact_map(pred, args.heavy_atom, args.ca_cutoff), contact_map(exp, args.heavy_atom, args.ca_cutoff)
	shared = cp & ce
	E = len(cp ^ ce) / (len(cp) + len(ce)) if cp or ce else 0.0
	kind = "heavy atoms < 4 A (paper)" if args.heavy_atom else f"CA-CA < {args.ca_cutoff} A"
	print(f"\n--- contact-map energy, contacts as {kind} ---")
	print(f"  contacts: predicted {len(cp)}, experimental {len(ce)}, shared {len(shared)}")
	print(f"  crystal contacts recovered      {len(shared) / len(ce):6.2f}" if ce else "")
	print(f"  E = sum|dP-dE| / sum(dP+dE)     {E:6.3f}   (0 = identical maps)")
	print("  This is the sampler's own energy between the predicted reference and the real")
	print("  protein. Everything the sampler then does is measured from the predicted map.")

	# 2. disulfides
	sg = {k: a["SG"] for k, _, a in exp if "SG" in a}
	bridges = [(i, j) for i, j in itertools.combinations(list(sg), 2)
		if math.dist(sg[i], sg[j]) < args.ss_cutoff]
	print(f"\n--- disulfide bridges: {len(bridges)} found in the experimental file ---")
	if bridges:
		pos = {k: n for n, (k, _, _) in enumerate(exp)}
		print("   exp pair        seq separation   crystal CB-CB   predicted CB-CB")
		ok = 0
		for i, j in sorted(bridges, key=lambda p: pos[p[0]]):
			a, b = pos[i], pos[j]
			ca_, cb_ = exp[a][2].get("CB"), exp[b][2].get("CB")
			pa, pb = pred[a][2].get("CB"), pred[b][2].get("CB")
			if not (ca_ and cb_ and pa and pb):
				continue
			dp = math.dist(pa, pb)
			ok += dp < 6.0
			print(f"   {i[1]:4d}-{j[1]:4d} {'':6s} {abs(b - a):6d} {'':8s} {math.dist(ca_, cb_):5.2f} "
				f"{'':9s} {dp:6.2f}{'' if dp < 6 else '   <-- not formed'}")
		print(f"   formed in the prediction (CB-CB < 6 A): {ok}/{len(bridges)}")

	# 3. per-block deviations
	if dev:
		print(f"\n--- CA deviation by block of {args.block} paired residues ---")
		for s in range(0, len(dev), args.block):
			v = dev[s:s + args.block]
			print(f"   {s + 1:4d}-{s + len(v):4d}  {math.sqrt(sum(x * x for x in v) / len(v)):7.2f} A")


if __name__ == "__main__":
	main()
