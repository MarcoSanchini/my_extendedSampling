"""
Formal K-sites determination across MULTIPLE temperature-sweep runs, plus
a per-site entropy plot -- follow-up to analyze_ensemble.py's informal
"most/least conserved 10 sites" list (see DEVLOG.txt, 2026-09-23/24/25
entries: sites 8/'G' and 45/'D' were flagged as an early, informal signal
from just two temperatures; never formalized). Reuses analyze_ensemble.py's
own checkpoint-loading and site-entropy functions directly.

K-SITES DEFINITION -- now taken directly from Zambon et al 2024 (verified
by reading the paper, p.6): "there are some sites, i.e. 5, 14, 26, 30, 41,
43 and 54, that are COMPLETELY CONSERVED at the lowest temperature
T_s=8e-4 ... and are remarkably conserved at all T_s<T_s^c. We call them
K-sites." Exactly 7 sites for protein G, defined by near-ZERO entropy
(not a loose fraction of log(20)) that PERSISTS across every temperature
below the paper's structural transition T_s^c. The paper separately notes
9 more sites with entropy "lower than 1" at the same coldest T -- a
softer, secondary tier they do NOT call K-sites.

This script mirrors that two-tier structure:
  --k-site-threshold   (default 0.05, natural-log units, absolute -- NOT
                        a fraction of log(20) any more) -- a site is a
                        K-site if its entropy stays below this at EVERY
                        T passed in via --active-t.
  --secondary-threshold (default 1.0, matching the paper's own "lower
                        than 1" language) -- reported separately, not
                        merged into the K-sites list.
Caller is responsible for only passing "cold" T's (below wherever this
system's own T_s^c-analog sits) via --active-t -- passing a hot/disordered
T here would correctly show 0 K-sites (every site's entropy is high
there), which is expected, not a bug: this reproduces the paper's own
"K-sites is a cold-regime concept" framing, not a universal one.

An earlier version of this script used its own, much looser, invented
threshold (10% of log(20) at every T) before the paper was available in
this session's context -- see DEVLOG.txt, 2026-09-25 entries, both the
original and this correction.

Usage:
    python identify_k_sites.py \\
        --results-dirs tests_rate_sweep_temperature_refine/T0.3981/sim0 \\
                        tests_rate_sweep_temperature_refine/T0.6310/sim0 \\
        --active-t 0.3981 0.631

--results-dirs and --active-t must be the same length and in the same
order (one T label per results_dir). Writes per_site_entropy_by_T.csv and
per_site_entropy.png to --out-dir (default k_sites_plots/).
"""
import argparse
import math
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_ensemble import load_checkpoint_sequences, site_entropy, DEFAULT_REF_SEQ


DEFAULT_K_SITE_THRESHOLD = 0.05     # natural-log entropy units -- "completely conserved"
DEFAULT_SECONDARY_THRESHOLD = 1.0   # natural-log entropy units -- paper's own "lower than 1"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--results-dirs", nargs="+", required=True)
	parser.add_argument("--active-t", nargs="+", type=float, required=True,
						 help="one temperature label per --results-dirs entry, same order -- "
							  "pass only T's you believe are in the COLD regime (see module docstring)")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--burn-in-frac", type=float, default=0.5)
	parser.add_argument("--k-site-threshold", type=float, default=DEFAULT_K_SITE_THRESHOLD,
						 help="absolute entropy cutoff (natural log units) for a site to count "
							  "as a K-site, required at EVERY --active-t (default 0.05)")
	parser.add_argument("--secondary-threshold", type=float, default=DEFAULT_SECONDARY_THRESHOLD,
						 help="looser absolute entropy cutoff for the paper's secondary "
							  "'highly conserved' tier, required at EVERY --active-t (default 1.0)")
	parser.add_argument("--out-dir", default="k_sites_plots")
	args = parser.parse_args()

	if len(args.results_dirs) != len(args.active_t):
		raise ValueError(
			f"--results-dirs ({len(args.results_dirs)}) and --active-t ({len(args.active_t)}) "
			f"must have the same length -- one T per results_dir, same order."
		)

	ref_seq = args.ref_seq
	L = len(ref_seq)
	max_S = math.log(20)

	per_t_entropy = {}
	for results_dir, T in zip(args.results_dirs, args.active_t):
		sequences, max_move, n_total = load_checkpoint_sequences(results_dir, args.burn_in_frac)
		entropies, _ = site_entropy(sequences, L)
		per_t_entropy[T] = entropies
		mean_S = sum(entropies) / L
		print(f"T={T}: loaded {len(sequences)}/{n_total} checkpoints from {results_dir}, "
			  f"mean S(i)={mean_S:.4f} ({mean_S/max_S:.1%} of log(20))")

	row_labels = [f"{i}_{ref_seq[i]}" for i in range(L)]
	df = pd.DataFrame(per_t_entropy, index=row_labels)
	df.index.name = "site"

	is_k_site = (df < args.k_site_threshold).all(axis=1)
	is_secondary = (df < args.secondary_threshold).all(axis=1) & ~is_k_site
	k_sites = df.index[is_k_site].tolist()
	secondary_sites = df.index[is_secondary].tolist()

	print()
	print(f"=== K-sites (paper definition, p.6): entropy < {args.k_site_threshold} "
		  f"at EVERY T in {args.active_t} ===")
	print(f"{len(k_sites)}/{L} sites qualify (paper found 7/56 for protein G):")
	print(k_sites)
	print()
	print(f"=== secondary tier: entropy < {args.secondary_threshold} at every T, "
		  f"but not a K-site ===")
	print(f"{len(secondary_sites)}/{L} sites (paper found 9/56 more at this level):")
	print(secondary_sites)

	os.makedirs(args.out_dir, exist_ok=True)

	csv_path = os.path.join(args.out_dir, "per_site_entropy_by_T.csv")
	df.to_csv(csv_path)
	print(f"\nPer-site-per-T entropy table -> {csv_path}")

	# --- plot: per-site entropy, one line per T, K-sites/secondary shaded ---
	fig, ax = plt.subplots(figsize=(max(10, L * 0.18), 5))
	x = list(range(L))
	for T in args.active_t:
		ax.plot(x, df[T].values, marker='o', markersize=3, linewidth=1, label=f"T={T}")
	for i in x:
		if is_k_site.iloc[i]:
			ax.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.15)
		elif is_secondary.iloc[i]:
			ax.axvspan(i - 0.4, i + 0.4, color='orange', alpha=0.10)
	ax.axhline(args.k_site_threshold, color='red', linestyle='--', linewidth=1,
			   label=f"K-site threshold ({args.k_site_threshold})")
	ax.axhline(args.secondary_threshold, color='orange', linestyle='--', linewidth=1,
			   label=f"secondary threshold ({args.secondary_threshold})")
	ax.set_xticks(x)
	ax.set_xticklabels(df.index, rotation=90, fontsize=6)
	ax.set_xlabel("site (index_refAA)")
	ax.set_ylabel("site entropy S(i)")
	ax.set_title(f"Per-site entropy across T={args.active_t} -- "
				 f"{len(k_sites)} K-sites (red), {len(secondary_sites)} secondary (orange)")
	ax.legend(fontsize=8)
	fig.tight_layout()

	out_path = os.path.join(args.out_dir, "per_site_entropy.png")
	fig.savefig(out_path, dpi=120)
	plt.close(fig)
	print(f"Per-site entropy plot -> {out_path}")


if __name__ == "__main__":
	main()
