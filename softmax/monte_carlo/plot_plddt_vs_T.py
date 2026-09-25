"""
Mean pLDDT vs T_s across a temperature sweep -- reproduces the shape of
Zambon et al 2024's figure 2, lower panel (verified by reading the paper):
pLDDT roughly constant ~85 in the cold/foldable regime (T_s < T_s^n),
dropping toward ~40 (their cited "mark of disorder" threshold) as T rises
through the disordered regime; 70 is the paper's cited threshold for "a
good prediction."

Cheap: reads compute_structural_metrics.py's already-computed CSVs (no
model calls here) -- run that first for every T you want on this plot.

Usage:
    python plot_plddt_vs_T.py \\
        --results-dirs tests_rate_sweep_temperature/T1e-1/sim0 \\
                        tests_rate_sweep_temperature/T1e0/sim0 \\
                        tests_rate_sweep_temperature/T1e1/sim0 \\
                        tests_rate_sweep_temperature/T1e2/sim0 \\
        --active-t 0.1 1.0 10.0 100.0

Skips (with a warning) any results_dir whose structural_metrics.csv is
missing, rather than failing the whole plot.
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--results-dirs", nargs="+", required=True)
	parser.add_argument("--active-t", nargs="+", type=float, required=True,
						 help="one temperature label per --results-dirs entry, same order")
	parser.add_argument("--out-dir", default="k_sites_plots",
						 help="reuses the same default output dir as identify_k_sites.py "
							  "since both are 'across the sweep' summary plots")
	args = parser.parse_args()

	if len(args.results_dirs) != len(args.active_t):
		raise ValueError(
			f"--results-dirs ({len(args.results_dirs)}) and --active-t ({len(args.active_t)}) "
			f"must have the same length -- one T per results_dir, same order."
		)

	Ts, means, stds = [], [], []
	for results_dir, T in zip(args.results_dirs, args.active_t):
		csv_path = os.path.join(results_dir, "structural_metrics.csv")
		if not os.path.exists(csv_path):
			print(f"[skip] T={T}: {csv_path} not found -- run compute_structural_metrics.py first.")
			continue
		df = pd.read_csv(csv_path)
		Ts.append(T)
		means.append(df["mean_plddt"].mean())
		stds.append(df["mean_plddt"].std())
		print(f"T={T}: {len(df)} decoded checkpoints, mean_plddt={means[-1]:.2f} +/- {stds[-1]:.2f}")

	if not Ts:
		print("Nothing to plot -- no structural_metrics.csv found for any --results-dirs entry.")
		return

	os.makedirs(args.out_dir, exist_ok=True)

	fig, ax = plt.subplots(figsize=(8, 4))
	ax.errorbar(Ts, means, yerr=stds, marker='o', capsize=3)
	ax.axhline(70, color='gray', linestyle='--', linewidth=1, label="paper's 'good prediction' threshold (70)")
	ax.axhline(40, color='red', linestyle=':', linewidth=1, label="paper's 'mark of disorder' threshold (40)")
	ax.set_xscale("log")
	ax.set_xlabel("T_s")
	ax.set_ylabel("mean pLDDT")
	ax.set_title("Decode confidence (mean pLDDT) vs T_s -- cf Zambon et al 2024 fig. 2 lower panel")
	ax.legend(fontsize=8)
	fig.tight_layout()

	out_path = os.path.join(args.out_dir, "plddt_vs_T.png")
	fig.savefig(out_path, dpi=120)
	plt.close(fig)
	print(f"\npLDDT vs T -> {out_path}")


if __name__ == "__main__":
	main()
