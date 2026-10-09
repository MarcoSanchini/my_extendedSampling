"""
Compare a predicted structure with an experimental one: geometry, local
structure, contacts, and the disulfide bonds.

Written 2026-10-09. Every disulfide conclusion in this investigation rests on
the model's output and on distances nobody had checked against a real
structure. This script supplies that ground truth.

WHAT IT REPORTS, in order:
  1. Sequence check -- the reference aligned to each structure, with every
     mismatch listed (where a crambin isoform difference at 22/25 would show).
  2. Geometry -- CA RMSD after one optimal superposition, and an approximate
     TM-score, plus the residues that deviate most beside their pLDDT.
  3. Local structure, superposition-free -- the CA-CA distance error over all
     pairs, and the worst 7-residue windows. A global superposition can be
     dominated by one badly placed region; intra-chain distances cannot, because
     they do not change under rigid motion. This is the check that finds a
     confidently wrong loop.
  4. Contacts -- C-beta contact maps at 8 A (CA for glycine), precision, recall,
     F1, precision at the top L, L/2, L/5, and the distance error.
  5. Disulfides -- each KNOWN bond: experimental SG-SG (is the pairing a bond in
     the file? SG-SG <= 2.5 A), experimental C-beta, predicted C-beta, and the
     CA-CA distances. Then every cysteine pair, so a non-native pair that the
     prediction puts inside the criterion can be set against the experiment.

C-BETA CONVENTION. Both structures use the model's own convention: C-beta is
inferred from N, CA and C with ESM3's infer_CB (copied below), since the model
reports exactly that. The deposited experimental C-beta is printed alongside
for reference. The two differ by about 0.1-0.2 A on average, so this choice
moves experimental distances by a few tenths of an angstrom and no more.

APPROXIMATIONS, stated so they are not mistaken for more: the TM-score is the
score at a superposition found by fragment-seeded refinement, a lower bound on
the true TM-score.

    python bridge_test/compare_to_pdb.py --ref bpti \\
        --exp bridge_test/pdb/bpti_5PTI.pdb --pred bridge_test/pred/bpti.pdb \\
        [--plddt bridge_test/pred/bpti.pdb.plddt.tsv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

AA3 = {
	"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
	"GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
	"MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
	"TYR": "Y", "VAL": "V", "MSE": "M",
}

CONTACT = 8.0     # A, C-beta contact
BONDED_SG = 2.5   # A, SG-SG in the experimental file
CRITERION = 4.5   # A, the C-beta bond criterion under test
CA_WINDOW = 7     # residues, for the local-structure check


# ----------------------------------------------------------------- C-beta
def infer_CB(C, N, Ca, L=1.522, A=1.927, D=-2.143):
	"""C-beta from the backbone. Copied from custom_esm/utils/structure/protein_chain.py
	(infer_CB, after trDesign) so that this file runs without torch. It is the
	function ESM3 itself uses for the C-beta it reports, so applying it to both
	structures compares like with like. Kept in step with the original by hand;
	tests/test_compare_to_pdb.py checks the bond length and angle it must produce."""
	norm = lambda x: x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)
	with np.errstate(invalid="ignore"):
		vec_bc = N - Ca
		vec_ba = N - C
	bc = norm(vec_bc)
	n = norm(np.cross(vec_ba, bc))
	m = [bc, np.cross(n, bc), n]
	d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
	return Ca + sum([m * d for m, d in zip(m, d)])


# ---------------------------------------------------------------- parsing
def read_pdb(path, chain=None):
	"""Residues of one chain from the first model: a list of dicts with
	chain, resseq, icode, aa, atoms{name: xyz}. Alternate locations other than
	the first are ignored."""
	residues, index = [], {}
	with open(path) as fh:
		for line in fh:
			if line.startswith("ENDMDL"):
				break
			rec = line[:6].strip()
			if rec not in ("ATOM", "HETATM"):
				continue
			resname = line[17:20].strip()
			if rec == "HETATM" and resname != "MSE":
				continue
			if line[16] not in (" ", "A", "1"):
				continue
			ch = line[21]
			key = (ch, int(line[22:26]), line[26])
			if key not in index:
				index[key] = len(residues)
				residues.append({"chain": ch, "resseq": key[1], "icode": key[2],
								 "aa": AA3.get(resname, "X"), "atoms": {}})
			name = line[12:16].strip()
			residues[index[key]]["atoms"].setdefault(
				name, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
	if chain is None:
		counts = {}
		for r in residues:
			if "CA" in r["atoms"]:
				counts[r["chain"]] = counts.get(r["chain"], 0) + 1
		if not counts:
			raise SystemExit(f"{path}: no CA atoms found")
		chain = max(counts, key=counts.get)
	out = [r for r in residues if r["chain"] == chain and "CA" in r["atoms"]]
	if not out:
		raise SystemExit(f"{path}: chain {chain!r} has no residues with CA")
	return out, chain


def align(ref, other):
	"""Semi-global alignment of `ref` (string) onto `other` (string), free end
	gaps on both. Returns (map, n_match, mismatches): map[i] is the index in
	`other` aligned to ref[i], or None."""
	n, m = len(ref), len(other)
	# A mismatch must cost MORE than one gap (-3 vs -2), or a reference residue
	# missing from the structure gets forced onto an unrelated neighbour such as
	# an expression-tag residue. Interior point mutations still align as
	# mismatches: one costs -3, the two gaps needed to skip it cost -4.
	MATCH, MISM, GAP = 2, -3, -2
	H = np.zeros((n + 1, m + 1), dtype=int)
	T = np.zeros((n + 1, m + 1), dtype=int)  # 0 stop, 1 diag, 2 up, 3 left
	for i in range(1, n + 1):
		T[i, 0] = 2
	for j in range(1, m + 1):
		T[0, j] = 3
	for i in range(1, n + 1):
		for j in range(1, m + 1):
			d = H[i - 1, j - 1] + (MATCH if ref[i - 1] == other[j - 1] else MISM)
			u = H[i - 1, j] + GAP
			l = H[i, j - 1] + GAP
			best = max(d, u, l)
			H[i, j] = best
			T[i, j] = 1 if best == d else (2 if best == u else 3)
	# free trailing gaps: best score on the last row or last column
	cand = [(H[n, j], n, j) for j in range(m + 1)] + [(H[i, m], i, m) for i in range(n + 1)]
	_, i, j = max(cand)
	mapping = [None] * n
	while i > 0 and j > 0:
		t = T[i, j]
		if t == 1:
			mapping[i - 1] = j - 1
			i, j = i - 1, j - 1
		elif t == 2:
			i -= 1
		else:
			j -= 1
	mism = [(k, ref[k], other[mapping[k]]) for k in range(n)
			if mapping[k] is not None and ref[k] != other[mapping[k]]]
	n_match = sum(1 for k in range(n) if mapping[k] is not None and ref[k] == other[mapping[k]])
	return mapping, n_match, mism


# --------------------------------------------------------------- geometry
def kabsch(P, Q):
	"""Rotation R and translation t with P @ R.T + t closest to Q."""
	pc, qc = P.mean(0), Q.mean(0)
	U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
	d = np.sign(np.linalg.det(Vt.T @ U.T))
	R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
	return R, qc - pc @ R.T


def rmsd_after_fit(P, Q):
	R, t = kabsch(P, Q)
	diff = P @ R.T + t - Q
	return float(np.sqrt((diff ** 2).sum(1).mean())), np.sqrt((diff ** 2).sum(1))


def tm_score(P, Q, L_norm):
	"""Approximate TM-score of P (predicted CA) against Q (experimental CA),
	normalised by L_norm. Fragment-seeded iterative superposition; a lower
	bound on the optimum."""
	d0 = max(1.24 * np.cbrt(max(L_norm, 22) - 15) - 1.8, 0.5)
	n = len(P)
	best = 0.0
	for frag in sorted({n, max(n // 2, 4), max(n // 4, 4), 4}):
		if frag > n:
			continue
		for start in range(0, n - frag + 1, max(1, frag // 2)):
			idx = np.arange(start, start + frag)
			for _ in range(20):
				R, t = kabsch(P[idx], Q[idx])
				d = np.linalg.norm(P @ R.T + t - Q, axis=1)
				best = max(best, float((1.0 / (1.0 + (d / d0) ** 2)).sum() / L_norm))
				cut = d0 + 1.5
				new = np.where(d < cut)[0]
				while len(new) < 3 and cut < 50:
					cut += 0.5
					new = np.where(d < cut)[0]
				if len(new) == len(idx) and np.all(new == idx):
					break
				idx = new
	return best


def spearman(a, b):
	ra = np.argsort(np.argsort(a))
	rb = np.argsort(np.argsort(b))
	if len(a) < 3 or ra.std() == 0 or rb.std() == 0:
		return float("nan")
	return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------- reporting
def cb(res):
	"""C-beta as the MODEL defines it: inferred from N, CA, C. Glycine uses CA,
	as in standard contact maps. Applied to both structures."""
	if res is None:
		return None
	a = res["atoms"]
	if res["aa"] == "G":
		return a.get("CA")
	if not all(k in a for k in ("N", "CA", "C")):
		return None
	return infer_CB(a["C"], a["N"], a["CA"])


def cb_deposited(res):
	"""C-beta as deposited in the experimental file, for reference only."""
	if res is None:
		return None
	return res["atoms"].get("CB")


def dist(a, b):
	return float(np.linalg.norm(a - b)) if a is not None and b is not None else float("nan")


def fmt(x, w=6, p=2):
	return f"{'n/a':>{w}s}" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:{w}.{p}f}"


def read_plddt(path):
	vals = {}
	for line in Path(path).read_text().splitlines():
		parts = line.split("\t")
		if len(parts) >= 3 and parts[0].isdigit():
			vals[int(parts[0])] = float(parts[2])
	return vals


def parse_bonds(text):
	return [tuple(int(x) for x in b.split("-")) for b in text.split(",") if b.strip()]


def _ranges(nums):
	out, start, prev = [], nums[0], nums[0]
	for n in nums[1:]:
		if n != prev + 1:
			out.append((start, prev))
			start = n
		prev = n
	out.append((start, prev))
	return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", help="reference name from references.py (bpti, crambin, protein_g, ...)")
	ap.add_argument("--seq", help="reference sequence, if not using --ref")
	ap.add_argument("--exp", required=True, help="experimental PDB file")
	ap.add_argument("--pred", required=True, help="predicted PDB file")
	ap.add_argument("--exp-chain", default=None)
	ap.add_argument("--pred-chain", default=None)
	ap.add_argument("--bonds", default=None, help="known disulfides, e.g. 5-55,14-38,30-51 (1-based, reference numbering)")
	ap.add_argument("--plddt", default=None, help="per-residue pLDDT sidecar written by test_cleaved_complex.py --pdb")
	ap.add_argument("--min-sep", type=int, default=6, help="sequence separation for the headline contact metrics")
	args = ap.parse_args()

	known = []
	if args.ref:
		import references as R
		if args.ref not in R.REFERENCES:
			raise SystemExit(f"unknown --ref {args.ref!r}; available: {sorted(R.REFERENCES)}")
		seq = R.REFERENCES[args.ref]
		known = list(R.KNOWN_DISULFIDES.get(args.ref, ()))
	elif args.seq:
		seq = args.seq
	else:
		raise SystemExit("give --ref or --seq")
	if args.bonds:
		known = parse_bonds(args.bonds)
	L = len(seq)

	exp, exp_ch = read_pdb(args.exp, args.exp_chain)
	pred, pred_ch = read_pdb(args.pred, args.pred_chain)
	exp_map, exp_match, exp_mism = align(seq, "".join(r["aa"] for r in exp))
	pred_map, pred_match, pred_mism = align(seq, "".join(r["aa"] for r in pred))
	E = [exp[j] if j is not None else None for j in exp_map]
	P = [pred[j] if j is not None else None for j in pred_map]
	plddt = read_plddt(args.plddt) if args.plddt else {}

	print(f"reference: {args.ref or 'custom'}  (L={L})")
	print(f"experimental: {args.exp}  chain {exp_ch}: {len(exp)} residues with CA")
	print(f"predicted:    {args.pred}  chain {pred_ch}: {len(pred)} residues with CA")
	print("C-beta: inferred from N/CA/C on both sides (the model's convention); glycine uses CA")

	# ---- 1. sequence check
	print("\n--- sequence check (reference vs each structure) ---")
	for name, mp, match, mism in (("experimental", exp_map, exp_match, exp_mism),
								  ("predicted", pred_map, pred_match, pred_mism)):
		resolved = sum(1 for j in mp if j is not None)
		print(f"  {name:<13s} {resolved}/{L} reference positions present, "
			  f"{match} identical, {len(mism)} mismatched")
		for k, a, b in mism[:12]:
			print(f"      position {k + 1}: reference {a}, structure {b}")
		missing = [k + 1 for k, j in enumerate(mp) if j is None]
		if missing:
			print(f"      absent positions: {_ranges(missing)}")

	paired = [i for i in range(L) if E[i] is not None and P[i] is not None]
	if len(paired) < 4:
		raise SystemExit("fewer than 4 residues present in both structures; nothing to compare")
	Lnorm = sum(1 for e in E if e is not None)

	# ---- 2. geometry
	Pc = np.array([P[i]["atoms"]["CA"] for i in paired])
	Qc = np.array([E[i]["atoms"]["CA"] for i in paired])
	rmsd, per_res = rmsd_after_fit(Pc, Qc)
	tm = tm_score(Pc, Qc, Lnorm)
	print("\n--- geometry (CA, predicted onto experimental) ---")
	print(f"  residues compared      {len(paired)}")
	print(f"  CA RMSD                {rmsd:6.2f} A   (all compared residues, one superposition)")
	print(f"  TM-score (approx.)     {tm:6.3f}     (lower bound; >0.5 same fold, <0.17 random)")
	order = np.argsort(-per_res)[:8]
	has_pl = bool(plddt)
	print("  largest CA deviations:" + ("   (pLDDT in the last column)" if has_pl else ""))
	for o in order:
		i = paired[o]
		pl = f"  pLDDT {plddt[i + 1]:5.1f}" if has_pl and (i + 1) in plddt else ""
		print(f"      {seq[i]}{i + 1:<4d} {per_res[o]:6.2f} A{pl}")
	if has_pl:
		pls = np.array([plddt.get(i + 1, np.nan) for i in paired])
		ok = ~np.isnan(pls)
		if ok.sum() >= 5:
			print(f"  Spearman(pLDDT, CA deviation) = {spearman(pls[ok], per_res[ok]):+.2f}"
				  f"   (negative means low confidence marks the wrong residues)")

	# ---- 3. local structure, superposition-free
	De_ca = np.linalg.norm(Qc[:, None] - Qc[None], axis=-1)
	Dp_ca = np.linalg.norm(Pc[:, None] - Pc[None], axis=-1)
	iu = np.triu_indices(len(paired), 1)
	err = np.abs(Dp_ca - De_ca)[iu]
	print("\n--- local structure (superposition-free) ---")
	print(f"  CA-CA distance error, all pairs    MAE {err.mean():5.2f} A   max {err.max():5.2f} A")
	wins = []
	for s in range(len(paired) - CA_WINDOW + 1):
		if paired[s + CA_WINDOW - 1] - paired[s] != CA_WINDOW - 1:
			continue  # only windows of consecutive, resolved residues
		idx = np.arange(s, s + CA_WINDOW)
		sub = np.abs(Dp_ca[np.ix_(idx, idx)] - De_ca[np.ix_(idx, idx)])
		wins.append((float(sub[np.triu_indices(CA_WINDOW, 1)].mean()),
					 paired[s], paired[s + CA_WINDOW - 1]))
	if wins:
		wins.sort(reverse=True)
		print(f"  worst {CA_WINDOW}-residue windows (intra-window CA-CA error):")
		for m, a, b in wins[:3]:
			print(f"      {seq[a]}{a + 1}-{seq[b]}{b + 1}  {m:5.2f} A")

	# ---- 4. contacts
	De = np.full((L, L), np.nan)
	Dp = np.full((L, L), np.nan)
	for i in paired:
		for j in paired:
			if i < j:
				De[i, j] = dist(cb(E[i]), cb(E[j]))
				Dp[i, j] = dist(cb(P[i]), cb(P[j]))
	ii, jj = np.triu_indices(L, 1)
	print(f"\n--- contacts (C-beta-C-beta <= {CONTACT:.0f} A; CA for Gly) ---")
	for sep in sorted({3, args.min_sep, 12}):
		m = (jj - ii >= sep) & ~np.isnan(De[ii, jj]) & ~np.isnan(Dp[ii, jj])
		if not m.any():
			continue
		de, dp = De[ii, jj][m], Dp[ii, jj][m]
		ec, pc_ = de <= CONTACT, dp <= CONTACT
		tp, fp, fn = int((ec & pc_).sum()), int((~ec & pc_).sum()), int((ec & ~pc_).sum())
		prec = tp / (tp + fp) if tp + fp else float("nan")
		rec = tp / (tp + fn) if tp + fn else float("nan")
		f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
		print(f"  |i-j| >= {sep:<3d} experimental contacts {int(ec.sum()):4d}  predicted {int(pc_.sum()):4d}  "
			  f"TP {tp:4d}  precision {prec:5.2f}  recall {rec:5.2f}  F1 {f1:5.2f}")
	m = (jj - ii >= args.min_sep) & ~np.isnan(De[ii, jj]) & ~np.isnan(Dp[ii, jj])
	if m.any():
		de, dp = De[ii, jj][m], Dp[ii, jj][m]
		rank = np.argsort(dp)
		tops = []
		for label, k in (("L", len(paired)), ("L/2", len(paired) // 2), ("L/5", len(paired) // 5)):
			k = max(1, min(k, len(rank)))
			tops.append(f"top-{label} ({k}) precision {np.mean(de[rank[:k]] <= CONTACT):.2f}")
		print(f"  ranked by predicted distance, |i-j| >= {args.min_sep}: " + "; ".join(tops))
		near = de < 15.0
		if near.sum() >= 3:
			print(f"  distance error on pairs within 15 A in experiment ({int(near.sum())}): "
				  f"MAE {np.mean(np.abs(dp[near] - de[near])):.2f} A, Pearson r {np.corrcoef(dp[near], de[near])[0, 1]:.3f}")

	# ---- 5. disulfides
	cys = [i for i in range(L) if seq[i] == "C"]
	print("\n--- disulfides ---")
	if known:
		print(f"  known bonds (supplied): {len(known)}")
		print(f"  {'pair':<8s}{'exp SG-SG':>10s}{'exp CB':>8s}{'exp CB':>8s}{'exp CA':>8s}"
			  f"{'pred CB':>9s}{'pred CA':>9s}{'d(CB)':>8s}  exp bonded?   pLDDT")
		print(f"  {'':<8s}{'':>10s}{'(model)':>8s}{'(dep.)':>8s}{'-CA':>8s}{'':>9s}{'-CA':>9s}{'':>8s}")
		n_formed = 0
		for (p, q) in known:
			i, j = p - 1, q - 1
			ea = E[i]["atoms"] if E[i] else {}
			eb = E[j]["atoms"] if E[j] else {}
			sg = dist(ea.get("SG"), eb.get("SG"))
			ecb = dist(cb(E[i]), cb(E[j]))
			ecb_d = dist(cb_deposited(E[i]), cb_deposited(E[j]))
			eca = dist(ea.get("CA"), eb.get("CA"))
			pcb = dist(cb(P[i]), cb(P[j]))
			pca = dist(P[i]["atoms"].get("CA") if P[i] else None, P[j]["atoms"].get("CA") if P[j] else None)
			formed = (not np.isnan(sg)) and sg <= BONDED_SG
			n_formed += int(formed)
			status = "n/a (unresolved)" if np.isnan(sg) else ("YES" if formed else "NO")
			pl = (f"{plddt[p]:.0f}/{plddt[q]:.0f}" if p in plddt and q in plddt else "-")
			print(f"  {f'{p}-{q}':<8s}{fmt(sg, 10)}{fmt(ecb, 8)}{fmt(ecb_d, 8)}{fmt(eca, 8)}"
				  f"{fmt(pcb, 9)}{fmt(pca, 9)}{fmt(pcb - ecb, 8)}  {status:<14s}{pl}")
		print(f"  known bonds confirmed in the experimental file (SG-SG <= {BONDED_SG} A): {n_formed}/{len(known)}")
		if n_formed < len(known):
			print("  WARNING: not every supplied pairing is a bond in this structure; check the pairings or the entry "
				  "(reduced form, wrong chain, unresolved atoms).")
	else:
		print("  no known bonds supplied (--ref with KNOWN_DISULFIDES, or --bonds)")

	known_set = {frozenset((p - 1, q - 1)) for p, q in known}
	rows = []
	for a_i, a in enumerate(cys):
		for b in cys[a_i + 1:]:
			rows.append((dist(cb(P[a]), cb(P[b])), dist(cb(E[a]), cb(E[b])), a, b))
	rows.sort(key=lambda r: (np.isnan(r[0]), r[0]))
	if rows:
		pred_in = [r for r in rows if r[0] <= CRITERION]
		exp_in = [r for r in rows if r[1] <= CRITERION]
		pred_in_known = [r for r in pred_in if frozenset((r[2], r[3])) in known_set]
		print(f"\n  cysteine pairs: {len(rows)}   predicted within {CRITERION} A: {len(pred_in)} "
			  f"({len(pred_in_known)} known bonds, {len(pred_in) - len(pred_in_known)} non-native)   "
			  f"experimental within {CRITERION} A: {len(exp_in)}")
		print(f"  {'pair':<10s}{'pred CB':>9s}{'exp CB':>9s}  status")
		shown = 0
		for pcb, ecb, a, b in rows:
			native = frozenset((a, b)) in known_set
			interesting = native or (not np.isnan(pcb) and pcb <= CRITERION + 1.5) or \
						  (not np.isnan(ecb) and ecb <= CRITERION + 1.5)
			if not interesting and shown >= 10:
				continue
			tag = "KNOWN bond" if native else ("NON-NATIVE, inside criterion in prediction"
											   if (not np.isnan(pcb) and pcb <= CRITERION) else "")
			print(f"  {f'{a + 1}-{b + 1}':<10s}{fmt(pcb, 9)}{fmt(ecb, 9)}  {tag}")
			shown += 1


if __name__ == "__main__":
	main()
