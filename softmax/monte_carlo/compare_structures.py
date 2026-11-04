"""
Compare a predicted structure (ESM3) with an experimental one (e.g. PDB 2HMG):
superimpose the CA atoms, report RMSD and friends, read the per-residue pLDDT
from the predicted file's B-factor column, and write files for a viewer.

Needs only numpy (matplotlib optional, for the plot); no GPU, no network.

WHY THE HA CASE NEEDS MORE THAN bridge_test/compare_to_pdb.py. That script
compares one chain with one chain. ESM3 decodes HA as ONE chain (HA0, the
precursor), while a crystal structure such as 2HMG holds the cleaved protein:
HA1 and HA2 as separate chains, three times over (a trimer). So here every
experimental chain is aligned by sequence to the predicted chain, and a chain is
used when it covers predicted residues that no earlier chain covered. For 2HMG
that picks HA1 (chain A) and HA2 (chain B) of the first protomer and skips the
other two copies, which would only map onto the same predicted residues again.
Use --exp-chains A,B to choose by hand.

WHAT IT REPORTS, and what each number means in plain words:

  sequence identity   of the aligned residues. Far below 100% means the
                      experimental protein is a different strain/variant;
                      the comparison still works, but say so.
  CA RMSD (all)       superimpose the two backbones as well as possible
                      (Kabsch: the best rotation + translation), then take the
                      typical distance between matching CA atoms, in angstrom.
                      ~1 A: same structure up to noise. 2-3 A: same fold, some
                      regions off. >5 A: large parts placed differently. For a
                      500-residue protein, >10 A means it is not the same shape.
                      One badly placed region inflates it, which is why the
                      next numbers exist.
  CA RMSD (core)      the same after repeatedly dropping residues more than
                      --cutoff A off and refitting (what PyMOL's "align" does
                      by default, 5 cycles of 2 A). It says how well the part
                      that DOES match matches; always read it together with
                      how many residues it kept.
  within 1/2/4/8 A    the fraction of residues whose CA lies within that
                      distance after the global fit (GDT-style). Easier to read
                      than RMSD: "60% of residues within 4 A".
  TM-score (approx.)  0-1, length-independent fold similarity, normalised by
                      the experimental length. >0.5 same fold, <0.17 like two
                      random structures. A lower bound (from compare_to_pdb.py).
  per-chain RMSD      each experimental chain (HA1, HA2) fitted on its own.
                      Low per-chain but high overall RMSD means the domains are
                      right but packed wrongly relative to each other.
  pLDDT               ESM3's own confidence per residue, 0-100, from the
                      B-factor column. >90 very confident, 70-90 confident,
                      50-70 low, <50 essentially a guess. Spearman(pLDDT,
                      deviation) < 0 means the model "knows" where it is wrong.

OUTPUTS (with --out-prefix P):
  P_superposed.pdb  the predicted model moved onto the experimental frame;
                    open it together with the experimental file and they
                    overlap in any viewer, no further fitting needed
  P_per_residue.csv one row per predicted residue: pLDDT, experimental chain
                    and number it matches, CA deviation after the fit
  P_view.pml        PyMOL script:    cd <folder>; pymol P_view.pml
  P_view.cxc        ChimeraX script: cd <folder>; chimerax P_view.cxc
                    (paths inside are relative, so copy the folder WITH the
                    experimental file in it, i.e. keep 2HMG.pdb next to them)
  P_per_residue.png deviation and pLDDT along the sequence (if matplotlib)

Usage (from softmax/monte_carlo):
    python compare_structures.py --pred structure_compare/ha0_pred.pdb \\
        --exp structure_compare/2HMG.pdb --out-prefix structure_compare/ha0_vs_2HMG
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../code/bridge_test")))
from compare_to_pdb import AA3, align, kabsch, tm_score, spearman  # noqa: E402


# ---------------------------------------------------------------- parsing
def read_chains(path):
	"""{chain id: [residue, ...]} from the first MODEL, residues that have a CA.
	A residue is a dict: aa, resseq, icode, ca (xyz), b (CA B-factor). First
	alternate location only; HETATM kept only for selenomethionine."""
	chains, seen = {}, set()
	with open(path) as fh:
		for line in fh:
			if line.startswith("ENDMDL"):
				break
			rec = line[:6].strip()
			if rec not in ("ATOM", "HETATM") or line[12:16].strip() != "CA":
				continue
			resname = line[17:20].strip()
			if rec == "HETATM" and resname != "MSE":
				continue
			if line[16] not in (" ", "A", "1"):
				continue
			ch, resseq, icode = line[21], int(line[22:26]), line[26]
			if (ch, resseq, icode) in seen:
				continue
			seen.add((ch, resseq, icode))
			b = float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else float("nan")
			chains.setdefault(ch, []).append({
				"aa": AA3.get(resname, "X"), "resseq": resseq, "icode": icode.strip(),
				"ca": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]), "b": b})
	if not chains:
		raise SystemExit(f"{path}: no CA atoms found")
	return chains


def seq_of(residues):
	return "".join(r["aa"] for r in residues)


def plddt_from_b(pred):
	"""Per-residue pLDDT on 0-100 from the B-factor column, or None when the
	column clearly is not pLDDT (all equal, e.g. 1.00 from a writer that had no
	confidence to put there)."""
	b = np.array([r["b"] for r in pred], dtype=float)
	if np.isnan(b).all() or np.nanmax(b) - np.nanmin(b) < 1e-6:
		return None
	return b * 100.0 if np.nanmax(b) <= 1.0 else b


# ------------------------------------------------------------- matching
def match_chains(pred_seq, exp_chains, wanted=None, min_identity=0.3, min_new=10):
	"""Pair predicted residues with experimental ones across several chains.
	Returns (pairs, used): pairs is a list of (pred index, exp chain, exp
	index), used a list of (chain, n paired, identity). A chain is accepted if
	its aligned identity is >= min_identity and it brings >= min_new predicted
	residues not already paired (so symmetry copies are skipped)."""
	order = wanted or sorted(exp_chains, key=lambda c: (-len(exp_chains[c]), c))
	taken, pairs, used, skipped = set(), [], [], []
	for ch in order:
		if ch not in exp_chains:
			raise SystemExit(f"chain {ch!r} not in the experimental file (has {sorted(exp_chains)})")
		eseq = seq_of(exp_chains[ch])
		mapping, n_match, _ = align(eseq, pred_seq)   # mapping[exp i] -> pred j
		aligned = [(j, i) for i, j in enumerate(mapping) if j is not None]
		ident = n_match / len(aligned) if aligned else 0.0
		new = [(j, i) for j, i in aligned if j not in taken]
		if not wanted and (ident < min_identity or len(new) < min_new):
			skipped.append((ch, len(eseq), ident, len(new)))
			continue
		for j, i in new:
			taken.add(j)
			pairs.append((j, ch, i))
		used.append((ch, len(new), sum(1 for j, i in new if pred_seq[j] == eseq[i]) / max(len(new), 1)))
	pairs.sort()
	return pairs, used, skipped


def core_fit(P, Q, cutoff, cycles):
	"""Outlier-rejection fit, as PyMOL 'align' does: fit, drop pairs farther
	than cutoff, refit on the rest, `cycles` times. Returns (R, t, kept idx)."""
	keep = np.arange(len(P))
	R, t = kabsch(P, Q)
	for _ in range(cycles):
		d = np.linalg.norm(P @ R.T + t - Q, axis=1)
		new = np.where(d <= cutoff)[0]
		if len(new) < 4 or len(new) == len(keep):
			break
		keep = new
		R, t = kabsch(P[keep], Q[keep])
	return R, t, keep


def transform_pdb(src, dst, R, t):
	"""Copy src with every ATOM/HETATM coordinate moved by x -> R x + t."""
	out = []
	with open(src) as fh:
		for line in fh:
			if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
				x = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) @ R.T + t
				line = f"{line[:30]}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{line[54:]}"
			out.append(line)
	with open(dst, "w") as fh:
		fh.writelines(out)


# ----------------------------------------------------------- view files
def write_view_scripts(prefix, exp_path, sup_path, used_chains, has_plddt):
	# Paths relative to the script's own folder, so the folder can be copied
	# to a laptop (scp) and the script still finds both files there. Run it
	# from that folder: cd <folder>; pymol <name>_view.pml
	here = os.path.dirname(os.path.abspath(prefix))
	exp_abs, sup_abs = os.path.relpath(exp_path, here), os.path.relpath(sup_path, here)
	chains_pml = "+".join(used_chains)
	chains_cx = ",".join(used_chains)
	pml = [
		f"load {exp_abs}, exp",
		f"load {sup_abs}, pred",
		"remove solvent",
		f"remove exp and not chain {chains_pml}",
		"hide everything",
		"show cartoon",
		"color grey70, exp",
		("spectrum b, red_white_blue, pred, minimum=30, maximum=90" if has_plddt else "color orange, pred"),
		"# already superposed by compare_structures.py. To redo the fit inside PyMOL:",
		"#   align pred and name CA, exp and name CA, cycles=0    (no outlier rejection)",
		"#   align pred and name CA, exp and name CA              (PyMOL default, drops outliers)",
		"orient",
		"bg_color white",
	]
	cxc = [
		f"open {exp_abs}",
		f"open {sup_abs}",
		"delete solvent",
		f"delete #1 & ~/{chains_cx}",
		"cartoon",
		"hide atoms",
		"color #1 lightgray",
		("color bfactor #2 palette alphafold" if has_plddt else "color #2 orange"),
		"# already superposed by compare_structures.py. To redo the fit inside ChimeraX",
		f"# against one experimental chain:  matchmaker #2 to #1/{used_chains[0]}",
		"view",
	]
	with open(f"{prefix}_view.pml", "w") as fh:
		fh.write("\n".join(pml) + "\n")
	with open(f"{prefix}_view.cxc", "w") as fh:
		fh.write("\n".join(cxc) + "\n")


def plot(prefix, idx, dev, pl_resseq, pl, chain_of):
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("  (matplotlib not installed: no plot)")
		return
	fig, ax = plt.subplots(figsize=(11, 3.6))
	ax.plot(idx, dev, lw=1, color="#3a5fa0", label="CA deviation after fit")
	ax.set_xlabel("predicted residue number")
	ax.set_ylabel("CA deviation (A)", color="#3a5fa0")
	if pl is not None:
		ax2 = ax.twinx()
		ax2.plot(pl_resseq, pl, lw=1, color="#c0504d", alpha=0.8)
		ax2.set_ylabel("pLDDT", color="#c0504d")
		ax2.set_ylim(0, 100)
	for k in range(1, len(idx)):
		if chain_of[k] != chain_of[k - 1]:
			ax.axvline(idx[k], color="k", lw=0.6, ls="--")
	ax.set_title("predicted vs experimental, per residue")
	fig.tight_layout()
	fig.savefig(f"{prefix}_per_residue.png", dpi=150)
	plt.close(fig)
	print(f"  wrote {prefix}_per_residue.png")


# ----------------------------------------------------------------- main
def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--pred", required=True, help="predicted PDB (e.g. from decode_to_pdb.py)")
	ap.add_argument("--exp", required=True, help="experimental PDB (e.g. 2HMG.pdb from rcsb.org)")
	ap.add_argument("--pred-chain", default=None, help="default: the longest chain")
	ap.add_argument("--exp-chains", default=None, help="comma-separated, e.g. A,B; default: automatic")
	ap.add_argument("--cutoff", type=float, default=2.0, help="A, outlier cutoff for the core RMSD")
	ap.add_argument("--cycles", type=int, default=5, help="outlier-rejection cycles for the core RMSD")
	ap.add_argument("--out-prefix", default=None, help="write superposed PDB, CSV, view scripts, plot")
	args = ap.parse_args()

	pchains = read_chains(args.pred)
	pch = args.pred_chain or max(pchains, key=lambda c: len(pchains[c]))
	pred = pchains[pch]
	pseq = seq_of(pred)
	echains = read_chains(args.exp)
	wanted = [c.strip() for c in args.exp_chains.split(",")] if args.exp_chains else None

	print(f"predicted:    {args.pred}  chain {pch}, {len(pred)} residues")
	print(f"experimental: {args.exp}  chains " +
		", ".join(f"{c} ({len(r)})" for c, r in sorted(echains.items())))

	pairs, used, skipped = match_chains(pseq, echains, wanted)
	if len(pairs) < 4:
		raise SystemExit("fewer than 4 residues could be paired; check the chains (--exp-chains)")
	print("\n--- which experimental chains were used ---")
	for ch, n, ident in used:
		js = [j for j, c, _ in pairs if c == ch]
		print(f"  chain {ch}: {n} residues paired with predicted {pred[min(js)]['resseq']}-"
			f"{pred[max(js)]['resseq']}, sequence identity {ident:.0%}")
	for ch, n, ident, new in skipped:
		why = "low identity" if ident < 0.3 else "covers residues already paired (a symmetry copy)"
		print(f"  chain {ch} ({n} residues) skipped: {why}")
	n_id = sum(1 for j, c, i in pairs if pseq[j] == echains[c][i]["aa"])
	print(f"  total: {len(pairs)} of {len(pred)} predicted residues paired "
		f"({len(pairs) / len(pred):.0%}), identity {n_id / len(pairs):.0%}")
	unpaired = len(pred) - len(pairs)
	if unpaired:
		print(f"  {unpaired} predicted residues have no experimental partner (not in the crystal, "
			"e.g. signal peptide, cleavage loop, transmembrane part); they are left out of every number")

	P = np.array([pred[j]["ca"] for j, _, _ in pairs])
	Q = np.array([echains[c][i]["ca"] for _, c, i in pairs])
	R, t = kabsch(P, Q)
	dev = np.linalg.norm(P @ R.T + t - Q, axis=1)
	rmsd = float(np.sqrt((dev ** 2).mean()))
	Rc, tc, keep = core_fit(P, Q, args.cutoff, args.cycles)
	dcore = np.linalg.norm(P[keep] @ Rc.T + tc - Q[keep], axis=1)
	tm = tm_score(P, Q, sum(len(echains[c]) for c, _, _ in used))

	print("\n--- global comparison (CA atoms, predicted fitted onto experimental) ---")
	print(f"  CA RMSD, all paired residues   {rmsd:7.2f} A   over {len(pairs)} residues")
	print(f"  CA RMSD, core                  {float(np.sqrt((dcore ** 2).mean())):7.2f} A   over "
		f"{len(keep)} residues ({len(keep) / len(pairs):.0%}) after {args.cycles} cycles at {args.cutoff} A")
	print("  residues within  " + "   ".join(f"{c:g} A: {np.mean(dev <= c):4.0%}" for c in (1, 2, 4, 8)))
	print(f"  TM-score (approx., lower bound) {tm:6.3f}   (>0.5 same fold, <0.17 random)")

	if len(used) > 1:
		print("\n--- each experimental chain fitted on its own ---")
		for ch, _, _ in used:
			k = np.array([n for n, (_, c, _) in enumerate(pairs) if c == ch])
			if len(k) < 4:
				continue
			Rk, tk = kabsch(P[k], Q[k])
			dk = np.linalg.norm(P[k] @ Rk.T + tk - Q[k], axis=1)
			print(f"  chain {ch}: CA RMSD {float(np.sqrt((dk ** 2).mean())):6.2f} A over {len(k)} residues "
				f"(vs {float(np.sqrt((dev[k] ** 2).mean())):6.2f} A for the same residues in the global fit)")

	pl_all = plddt_from_b(pred)
	print("\n--- pLDDT (predicted file's B-factor column) ---")
	if pl_all is None:
		print("  not present: every B-factor is the same value. Write the prediction with decode_to_pdb.py.")
		pl_pair = None
	else:
		pl_pair = np.array([pl_all[j] for j, _, _ in pairs])
		print(f"  mean pLDDT {np.nanmean(pl_all):5.1f} over all {len(pred)} residues, "
			f"{np.nanmean(pl_pair):5.1f} over the paired ones")
		print("  fraction  >=90: {:.0%}   70-90: {:.0%}   50-70: {:.0%}   <50: {:.0%}".format(
			np.mean(pl_all >= 90), np.mean((pl_all >= 70) & (pl_all < 90)),
			np.mean((pl_all >= 50) & (pl_all < 70)), np.mean(pl_all < 50)))
		print(f"  Spearman(pLDDT, CA deviation) {spearman(pl_pair, dev):+.2f}   "
			"(negative: low confidence where the prediction is wrong)")
		for lo, hi in ((0, 50), (50, 70), (70, 90), (90, 101)):
			m = (pl_pair >= lo) & (pl_pair < hi)
			if m.any():
				print(f"    pLDDT {lo:>2d}-{min(hi, 100):<3d}: {int(m.sum()):4d} residues, "
					f"mean CA deviation {dev[m].mean():6.2f} A")

	order = np.argsort(-dev)[:10]
	print("\n--- largest CA deviations ---")
	for o in order:
		j, c, i = pairs[o]
		pl = f"   pLDDT {pl_all[j]:5.1f}" if pl_all is not None else ""
		print(f"  predicted {pseq[j]}{pred[j]['resseq']:<4d} = exp chain {c} "
			f"{echains[c][i]['aa']}{echains[c][i]['resseq']:<4d} {dev[o]:7.2f} A{pl}")

	if args.out_prefix:
		os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
		sup = f"{args.out_prefix}_superposed.pdb"
		transform_pdb(args.pred, sup, R, t)
		dev_of = {j: dev[n] for n, (j, _, _) in enumerate(pairs)}
		exp_of = {j: (c, echains[c][i]) for j, c, i in pairs}
		with open(f"{args.out_prefix}_per_residue.csv", "w") as fh:
			fh.write("pred_resseq,aa,plddt,exp_chain,exp_resseq,exp_aa,ca_deviation\n")
			for j, r in enumerate(pred):
				c, e = exp_of.get(j, ("", None))
				fh.write(f"{r['resseq']}{r['icode']},{r['aa']},"
					f"{'' if pl_all is None else f'{pl_all[j]:.2f}'},{c},"
					f"{'' if e is None else str(e['resseq']) + e['icode']},{'' if e is None else e['aa']},"
					f"{'' if j not in dev_of else f'{dev_of[j]:.3f}'}\n")
		write_view_scripts(args.out_prefix, args.exp, sup, [c for c, _, _ in used], pl_all is not None)
		print(f"\n  wrote {sup}")
		print(f"  wrote {args.out_prefix}_per_residue.csv")
		print(f"  wrote {args.out_prefix}_view.pml   (PyMOL; run it from its own folder)")
		print(f"  wrote {args.out_prefix}_view.cxc   (ChimeraX; run it from its own folder)")
		idx = np.array([pred[j]["resseq"] for j, _, _ in pairs])
		plot(args.out_prefix, idx, dev, [r["resseq"] for r in pred], pl_all, [c for _, c, _ in pairs])


if __name__ == "__main__":
	main()
