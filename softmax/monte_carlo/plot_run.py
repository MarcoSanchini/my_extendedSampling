"""
Pandas/matplotlib plots for a single run's data.dat trajectory -- companion
to analyze_run.py (which reports the same information as printed text/
block-averages, no plotting dependency). For every numeric column data.dat
logs, plots three views on one figure: the raw per-move value, its
cumulative (expanding) mean, and a rolling-window mean.

Usage:
    python plot_run.py <results_dir>                       # e.g. tests_rate_sweep_temperature/T1e0/sim0
    python plot_run.py <results_dir> --window 500 --out-dir plots

Writes one PNG per metric to <results_dir>/plots/ (or --out-dir). PNGs are
local-only (see .gitignore) -- this script is meant to be run on the
machine that has data.dat, not pushed/pulled as output.
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_run import load_rows


# Every numeric column data.dat's header documents (see
# classes/rate_sampler.py:_extend_buffer) except 'move' itself, which is
# the x-axis, not a metric to plot. 'accepted'/'acc_moves'/'acc_rate' are
# themselves already cumulative-ish quantities -- still plotted the same
# way as everything else for consistency; the rolling-window view is what
# adds real information for those specifically (a windowed acceptance
# rate isn't otherwise available anywhere in this project's tooling).
NUMERIC_COLUMNS = [
	"time", "U", "U_am", "U_structure_ce", "entropy", "Hd_to_ref", "dU",
	"log_a_AB", "log_a_BA", "log_ratio", "accepted", "acc_moves", "acc_rate",
]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_dir", help="e.g. tests_rate_sweep_temperature/T1e0/sim0")
	parser.add_argument("--window", type=int, default=200, help="rolling window size, in moves")
	parser.add_argument("--out-dir", default=None, help="defaults to <results_dir>/plots")
	args = parser.parse_args()

	data_dat = os.path.join(args.results_dir, "data.dat")
	rows = load_rows(data_dat)
	df = pd.DataFrame(rows)

	df["move"] = pd.to_numeric(df["move"], errors="coerce")
	present = [c for c in NUMERIC_COLUMNS if c in df.columns]
	for col in present:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
	os.makedirs(out_dir, exist_ok=True)

	print(f"Plotting {len(present)} metrics from {data_dat} ({len(df)} rows) -> {out_dir}/")

	for col in present:
		series = df[col]
		cumulative = series.expanding().mean()
		rolling = series.rolling(args.window, min_periods=1).mean()

		fig, ax = plt.subplots(figsize=(10, 4))
		ax.plot(df["move"], series, alpha=0.25, linewidth=0.5, color="gray", label="raw")
		ax.plot(df["move"], cumulative, linewidth=1.5, label="cumulative mean")
		ax.plot(df["move"], rolling, linewidth=1.5, label=f"rolling mean (window={args.window})")
		ax.set_xlabel("move")
		ax.set_ylabel(col)
		ax.set_title(f"{col} vs move -- {args.results_dir}")
		ax.legend(fontsize=8)
		fig.tight_layout()

		out_path = os.path.join(out_dir, f"{col}.png")
		fig.savefig(out_path, dpi=120)
		plt.close(fig)
		print(f"  {col}: raw + cumulative + rolling(window={args.window}) -> {out_path}")

	print("Done.")


if __name__ == "__main__":
	main()
