"""
Formal K-sites determination across MULTIPLE temperature-sweep runs, plus
a per-site entropy plot -- follow-up to analyze_ensemble.py's informal
"most/least conserved 10 sites" list (see DEVLOG.txt, 2026-09-23/24/25
entries: sites 8/'G' and 45/'D' were flagged as an early, informal signal
from just two temperatures; never formalized). Reuses analyze_ensemble.py's
own checkpoint-loading and site-entropy functions directly.

K-SITES DEFINITION USED HERE -- an explicit, documented choice, NOT a
literal transcription of Zambon et al 2024's own statistical procedure
(this repo doesn't have that procedure's exact form recorded anywhere;
revisit this if the paper's precise criterion is available to check
against): a site counts as a K-site if its entropy stays below
--threshold-frac (default 10%) of log(20) at EVERY one of the "active"
temperatures passed in via --results-dirs/--active-t. "Active" is left to
the caller to choose -- pass only T's where the ensemble is actually
sampling (see the 2026-09-24/25 sweep entries: T <~ 0.25 is completely
frozen and would trivially "pass" the threshold at every site, which
would make the criterion meaningless there).

Usage:
    python identify_k_sites.py \\
        --results-dirs tests_rate_sweep_temperature_refine/T0.3981/sim0 \\
                        tests_rate_sweep_temperature_refine/T0.6310/sim0 \\
                        tests_rate_sweep_temperature/T1e0/sim0 \\
                        tests_rate_sweep_temperature/T1e1/sim0 \\
                        tests_rate_sweep_temperature/T1e2/sim0 \\
        --active-t 0.3981 0.631 1.0 10.0 100.0

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


DEFAULT_THRESHOLD_FRAC = 0.10  # fraction of log(20)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--results-dirs", nargs="+", required=True)
	parser.add_argument("--active-t", nargs="+", type=float, required=True,
						 help="one temperature label per --results-dirs entry, same order")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--burn-in-frac", type=float, default=0.5)
	parser.add_argument("--threshold-frac", type=float, default=DEFAULT_THRESHOLD_FRAC,
						 help="K-site cutoff, as a fraction of log(20) (default 0.10)")
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
	threshold = args.threshold_frac * max_S

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

	is_k_site = (df < threshold).all(axis=1)
	k_sites = df.index[is_k_site].tolist()

	print()
	print(f"=== K-sites: entropy < {args.threshold_frac:.0%} of log(20)={max_S:.4f} "
		  f"(={threshold:.4f}) at EVERY T in {args.active_t} ===")
	print(f"{len(k_sites)}/{L} sites qualify:")
	print(k_sites)

	os.makedirs(args.out_dir, exist_ok=True)

	csv_path = os.path.join(args.out_dir, "per_site_entropy_by_T.csv")
	df.to_csv(csv_path)
	print(f"\nPer-site-per-T entropy table -> {csv_path}")

	# --- plot: per-site entropy, one line per T, K-sites shaded ---
	fig, ax = plt.subplots(figsize=(max(10, L * 0.18), 5))
	x = list(range(L))
	for T in args.active_t:
		ax.plot(x, df[T].values, marker='o', markersize=3, linewidth=1, label=f"T={T}")
	for i in x:
		if is_k_site.iloc[i]:
			ax.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.08)
	ax.axhline(threshold, color='gray', linestyle='--', linewidth=1,
			   label=f"K-site threshold ({args.threshold_frac:.0%} of log(20))")
	ax.set_xticks(x)
	ax.set_xticklabels(df.index, rotation=90, fontsize=6)
	ax.set_xlabel("site (index_refAA)")
	ax.set_ylabel("site entropy S(i)")
	ax.set_title(f"Per-site entropy across T={args.active_t} -- {len(k_sites)} K-sites shaded red")
	ax.legend(fontsize=8)
	fig.tight_layout()

	out_path = os.path.join(args.out_dir, "per_site_entropy.png")
	fig.savefig(out_path, dpi=120)
	plt.close(fig)
	print(f"Per-site entropy plot -> {out_path}")


if __name__ == "__main__":
	main()
