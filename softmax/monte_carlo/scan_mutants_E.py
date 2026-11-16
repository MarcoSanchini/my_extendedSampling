"""
Single-mutant scan of the CONTACT-MAP energy on PTEN: every one of the
324 x 19 = 6156 point mutants is decoded by ESM3 and its contact map compared
with the reference's, the way Zambon et al 2024 define the energy:

    E = sum_ij |D_ij(mut) - D_ij(ref)| / sum_ij (D_ij(mut) + D_ij(ref)),  |i-j| > 2

with D the CA contact map at 8 A (the decoder writes backbone + CB only, so the
paper's 4 A heavy-atom map is not available). Both maps come from ESM3 at the
SAME number of generation steps, so E(reference) = 0 exactly (decoding is at
temperature 0 and was measured deterministic, bench_decode.py 2026-11-16).

The question it answers. bench_decode.py showed a 2-step decode costs 0.13 s,
about one U_am move, so a contact-map energy INSIDE the Monte Carlo loop is
affordable. What it did not show is whether such an energy can see a single
mutation at all: a structure predictor that returns the same contact map for
most point mutants gives dE = 0 for most proposals, and a sampler on a flat
landscape is a random walk. This scan measures that directly, and lines each
mutant up with the U_am scan (scan_single_mutants.py, same 6156 mutants) so
the two energies can be compared substitution by substitution.

Columns: site, ref_aa, mut_aa, E, n_diff (contacts gained + lost), rmsd (CA
RMSD to the reference's prediction, A), plddt (mean, 0-100), t (s).

Cost: one decode per mutant, 0.13 s at 2 steps -> ~14 min for all 6156 on
astaroth; 1.6 s at 32 steps, so use --sites for a subset there. Resumable like
scan_single_mutants.py: appended and flushed per site, complete sites skipped.

Usage (from softmax/monte_carlo, GPU idle):
    nohup python -u scan_mutants_E.py --num-steps 2 --out pten_natural/pten_mutants_E2.csv \\
        > scan_mutants_E2.log 2>&1 &
    # the same 20 sites at 32 steps, to check that 2 steps see the same mutants
    nohup python -u scan_mutants_E.py --num-steps 32 --sites $(seq 0 17 323) \\
        --out pten_natural/pten_mutants_E32.csv > scan_mutants_E32.log 2>&1 &

The summary at the end (also on its own, no GPU: --summary-only) gives the
share of mutants with E = 0, the distribution of E, and Spearman(E, dU_am)
pooled, within site and per site; with --compare, the agreement with another
E scan (e.g. 2 vs 32 steps) on the sites both contain.
"""
import argparse
import os
import sys
import time

import numpy as np

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
for p in (MONTE_CARLO_DIR, CODE_DIR, CUSTOMS_DIR, os.path.join(CODE_DIR, "tests")):
	if p not in sys.path:
		sys.path.insert(0, p)

from bench_decode import contact_map, kabsch_rmsd

AA = "ACDEFGHIKLMNPQRSTVWY"
COLUMNS = "site,ref_aa,mut_aa,E,n_diff,rmsd,plddt,t\n"


def done_sites(path):
	"""Sites with all 19 rows in an existing CSV, and those rows."""
	if not os.path.exists(path):
		return set(), []
	keep, counts = [], {}
	with open(path) as f:
		if f.readline() != COLUMNS:
			raise SystemExit(f"{path} has another header; move it aside or pass another --out")
		for line in f:
			s = line.split(",", 1)[0]
			if s.isdigit():
				counts[int(s)] = counts.get(int(s), 0) + 1
				keep.append(line)
	full = {s for s, n in counts.items() if n == 19}
	return full, [l for l in keep if int(l.split(",", 1)[0]) in full]


def summary(path, am_scan, compare):
	import pandas as pd
	m = pd.read_csv(path)
	print(f"\n# SUMMARY of {path}: {len(m)} mutants over {m.site.nunique()} sites")
	print(f"# time per mutant: median {m.t.median():.3f} s")
	zero = (m.n_diff == 0).mean()
	print(f"# E = 0 (identical contact map to the reference): {zero:.1%} of mutants; "
		f"n_diff <= 2: {(m.n_diff <= 2).mean():.1%}")
	q = m.E.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
	print("# E quantiles 10/25/50/75/90/max: " + " ".join(f"{v:.4f}" for v in q.values))
	print("# n_diff quantiles 10/25/50/75/90/max: " + " ".join(f"{v:.0f}" for v in m.n_diff.quantile([.1, .25, .5, .75, .9, 1.]).values))
	print(f"# RMSD to the reference's prediction: median {m.rmsd.median():.2f} A, 90% {m.rmsd.quantile(.9):.2f}, "
		f"max {m.rmsd.max():.2f}; pLDDT median {m.plddt.median():.1f}, min {m.plddt.min():.1f}")
	per_site_zero = m.groupby("site").n_diff.apply(lambda x: (x == 0).mean())
	print(f"# sites where EVERY mutant has E = 0: {(per_site_zero == 1).sum()} of {m.site.nunique()}; "
		f"where none does: {(per_site_zero == 0).sum()}")
	if am_scan and os.path.exists(am_scan):
		u = pd.read_csv(am_scan)
		j = m.merge(u, on=["site", "ref_aa", "mut_aa"], how="inner")
		print(f"# joined with {am_scan}: {len(j)} mutants")
		print(f"#   Spearman(E, dU_am) pooled: {j.E.corr(j.dU, method='spearman'):.3f}")
		ws = [g.E.corr(g.dU, method="spearman") for _, g in j.groupby("site") if g.E.nunique() > 2]
		print(f"#   within site (sites where E takes >2 values, n={len(ws)}): median {np.nanmedian(ws):.3f}, "
			f"mean {np.nanmean(ws):.3f}")
		g = j.groupby("site").agg(E=("E", "mean"), lU=("dU", lambda x: np.log10(x).mean()))
		print(f"#   per site, mean E vs mean log10 dU: {g.E.corr(g.lU, method='spearman'):.3f}")
		for lab, sub in (("E = 0", j[j.n_diff == 0]), ("E > 0", j[j.n_diff > 0])):
			if len(sub):
				print(f"#   dU_am where {lab}: median {sub.dU.median():.1f} (n={len(sub)})")
	if compare and os.path.exists(compare):
		c = pd.read_csv(compare)
		j = m.merge(c, on=["site", "ref_aa", "mut_aa"], suffixes=("", "_c"))
		print(f"# against {compare} on {len(j)} shared mutants: Spearman(E) {j.E.corr(j.E_c, method='spearman'):.3f}, "
			f"median |dE| {np.median(np.abs(j.E - j.E_c)):.4f}, "
			f"E = 0 in both {((j.n_diff == 0) & (j.n_diff_c == 0)).mean():.1%}, "
			f"in this only {((j.n_diff == 0) & (j.n_diff_c > 0)).mean():.1%}, "
			f"in the other only {((j.n_diff > 0) & (j.n_diff_c == 0)).mean():.1%}")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", default="pten_1d5r")
	ap.add_argument("--num-steps", type=int, default=2)
	ap.add_argument("--sites", type=int, nargs="+", help="0-based sites (default all)")
	ap.add_argument("--ca-cutoff", type=float, default=8.0)
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--out", required=True)
	ap.add_argument("--am-scan", default=os.path.join(MONTE_CARLO_DIR, "pten_natural", "pten_single_mutants.csv"),
		help="the U_am scan to compare with")
	ap.add_argument("--compare", default=None, help="another E scan (e.g. at 32 steps) to compare with")
	ap.add_argument("--summary-only", action="store_true")
	args = ap.parse_args()
	if args.summary_only:
		summary(args.out, args.am_scan, args.compare)
		return

	from references import get_reference
	_, ref = get_reference(args.ref)
	L = len(ref)

	import torch
	from compute_structural_metrics import CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from custom_esm.sdk.api import ESMProteinTensor

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	model, decoder = sampler.model, sampler.model.get_structure_decoder()
	cfg = init_structure_config(num_steps=args.num_steps)
	sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

	def predict(seq):
		torch.manual_seed(args.seed)
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		ep.expand()
		sync(); t0 = time.perf_counter()
		with torch.no_grad():
			out = model.generate(input=ESMProteinTensor(sequence=ep.tokens), config=cfg)
			tok = out.structure if out.structure.dim() > 1 else out.structure.unsqueeze(0)
			dec = decoder.decode(tok)
		sync()
		ca = dec["bb_pred"][0, 1:-1, CA_IDX].float().cpu().numpy()
		pl = dec["plddt"][0, 1:-1].float().cpu().numpy() if "plddt" in dec else np.full(L, np.nan)
		pl = pl * 100 if np.nanmax(pl) <= 1 else pl
		return ca, float(np.nanmean(pl)), time.perf_counter() - t0

	ref_ca, ref_pl, _ = predict(ref)
	ref_ca2, _, _ = predict(ref)
	ref_map = contact_map(ref_ca, args.ca_cutoff)
	n_ref = int(ref_map.sum())
	again = int((contact_map(ref_ca2, args.ca_cutoff) ^ ref_map).sum())
	print(f"# {args.ref}, L={L}, {args.num_steps} steps: reference pLDDT {ref_pl:.1f}, {n_ref} contacts; "
		f"same sequence twice differs in {again} contacts (must be 0)", flush=True)
	if again:
		print("# WARNING: the decode is not deterministic here, so E has a noise floor; see the line above.", flush=True)

	sites = args.sites if args.sites is not None else list(range(L))
	have, kept = done_sites(args.out)
	todo = [s for s in sites if s not in have]
	print(f"# {len(sites)} sites requested, {len(have & set(sites))} already in {args.out}, {len(todo)} to do "
		f"({19 * len(todo)} decodes)", flush=True)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		f.write(COLUMNS)
		f.writelines(kept)

	t0 = time.perf_counter()
	with open(args.out, "a") as out:
		for k, s in enumerate(todo, 1):
			cur = ref[s]
			Es = []
			for a in AA:
				if a == cur:
					continue
				ca, pl, dt = predict(ref[:s] + a + ref[s + 1:])
				cmap = contact_map(ca, args.ca_cutoff)
				n_diff = int((cmap ^ ref_map).sum())
				E = n_diff / (int(cmap.sum()) + n_ref)
				out.write(f"{s},{cur},{a},{E:.6f},{n_diff},{kabsch_rmsd(ca, ref_ca):.3f},{pl:.2f},{dt:.4f}\n")
				Es.append(E)
			out.flush()
			if k % 10 == 0 or k == len(todo):
				el = time.perf_counter() - t0
				print(f"# {k}/{len(todo)} sites, {el / 60:.1f} min, {el / k / 19:.3f} s per mutant, "
					f"{(len(todo) - k) * el / k / 60:.1f} min left; site {s} {cur}: E min {min(Es):.4f} "
					f"median {np.median(Es):.4f} max {max(Es):.4f}, zeros {sum(e == 0 for e in Es)}/19", flush=True)

	summary(args.out, args.am_scan, args.compare)


if __name__ == "__main__":
	main()
