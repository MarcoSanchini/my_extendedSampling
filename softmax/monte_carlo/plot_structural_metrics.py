"""
Plots for the structural side of the Zambon-replication analysis:
  - Rg_ca / Rg_allatom trajectories (raw + cumulative + rolling mean, same
    treatment as plot_run.py's data.dat metrics) from the CSV
    compute_structural_metrics.py writes -- this script does NOT call the
    model itself, so it's cheap and can be re-run freely to restyle plots.
  - pairwise Hamming distance distribution, i.e. the paper's p(q) target
    (see DEVLOG.txt, 2026-09-23 "temperature sweep scoped" entry) as a
    real histogram instead of analyze_ensemble.py's ASCII-bar one. Reuses
    analyze_ensemble.py's own q_distribution() directly (needs only the
    saved sequence strings, no model calls, unlike the Rg side above).

Usage:
    python plot_structural_metrics.py <results_dir> [--window 5] [--out-dir plots]
        # e.g. tests_rate_sweep_temperature/T1e0/sim0
    Expects <results_dir>/structural_metrics.csv to already exist (see
    compute_structural_metrics.py) for the Rg plots; the Hamming/q plot
    only needs the eprot/ checkpoints (same as analyze_ensemble.py) and
    is produced even if structural_metrics.csv is missing.
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_ensemble import load_checkpoint_sequences, q_distribution, DEFAULT_REF_SEQ


def plot_rg(results_dir, out_dir, window):
	csv_path = os.path.join(results_dir, "structural_metrics.csv")
	if not os.path.exists(csv_path):
		print(f"[skip] {csv_path} not found -- run compute_structural_metrics.py first.")
		return

	df = pd.read_csv(csv_path)
	for col in ["Rg_ca", "Rg_allatom"]:
		series = df[col]
		cumulative = series.expanding().mean()
		rolling = series.rolling(window, min_periods=1).mean()

		fig, ax = plt.subplots(figsize=(10, 4))
		ax.plot(df["move"], series, alpha=0.35, linewidth=0.8, marker='o', markersize=3,
				color="gray", label="raw (per decoded checkpoint)")
		ax.plot(df["move"], cumulative, linewidth=1.5, label="cumulative mean")
		ax.plot(df["move"], rolling, linewidth=1.5, label=f"rolling mean (window={window})")
		ax.set_xlabel("move")
		ax.set_ylabel(f"{col} [Å]")
		ax.set_title(f"{col} vs move -- {results_dir}")
		ax.legend(fontsize=8)
		fig.tight_layout()

		out_path = os.path.join(out_dir, f"{col}.png")
		fig.savefig(out_path, dpi=120)
		plt.close(fig)
		print(f"  {col}: raw + cumulative + rolling(window={window}) -> {out_path}")

	# quick plddt sanity plot too, since it's already in the same CSV and
	# is the standard "is this decode trustworthy" signal for this kind
	# of structure prediction
	if "mean_plddt" in df.columns:
		fig, ax = plt.subplots(figsize=(10, 3))
		ax.plot(df["move"], df["mean_plddt"], marker='o', markersize=3, linewidth=1)
		ax.set_xlabel("move")
		ax.set_ylabel("mean pLDDT")
		ax.set_title(f"decode confidence (mean pLDDT) vs move -- {results_dir}")
		fig.tight_layout()
		out_path = os.path.join(out_dir, "mean_plddt.png")
		fig.savefig(out_path, dpi=120)
		plt.close(fig)
		print(f"  mean_plddt -> {out_path}")


def plot_hamming(results_dir, out_dir, ref_seq, burn_in_frac):
	sequences, max_move, n_total = load_checkpoint_sequences(results_dir, burn_in_frac)
	print(f"Hamming/q distribution: {len(sequences)}/{n_total} post-burn-in checkpoints")

	qs = q_distribution(sequences)
	L = len(ref_seq)
	hamming = [round((1 - q) * L) for q in qs]

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))

	axes[0].hist(qs, bins=30, edgecolor="black", linewidth=0.3)
	axes[0].set_xlabel("pairwise sequence similarity q")
	axes[0].set_ylabel("pair count")
	axes[0].set_title(f"p(q) -- {results_dir}\nn_pairs={len(qs)}")

	axes[1].hist(hamming, bins=range(0, L + 2), edgecolor="black", linewidth=0.3)
	axes[1].set_xlabel("Hamming distance (sites differing)")
	axes[1].set_ylabel("pair count")
	axes[1].set_title(f"pairwise Hamming distance distribution (L={L})")

	fig.tight_layout()
	out_path = os.path.join(out_dir, "hamming_and_q_distribution.png")
	fig.savefig(out_path, dpi=120)
	plt.close(fig)
	print(f"  Hamming + q distribution -> {out_path}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_dir")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--burn-in-frac", type=float, default=0.5)
	parser.add_argument("--window", type=int, default=5,
						 help="rolling window size for Rg plots, in DECODED checkpoints "
							  "(not moves -- stride from compute_structural_metrics.py already thins these out)")
	parser.add_argument("--out-dir", default=None, help="defaults to <results_dir>/plots")
	args = parser.parse_args()

	out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
	os.makedirs(out_dir, exist_ok=True)

	plot_rg(args.results_dir, out_dir, args.window)
	plot_hamming(args.results_dir, out_dir, args.ref_seq, args.burn_in_frac)

	print("Done.")


if __name__ == "__main__":
	main()
