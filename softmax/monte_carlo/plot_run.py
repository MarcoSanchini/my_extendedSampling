"""
Pandas/matplotlib plots for a single run's data.dat trajectory -- companion
to analyze_run.py (which reports the same information as printed text/
block-averages, no plotting dependency). For most numeric columns data.dat
logs, plots three views on one figure: the raw per-move value, its
cumulative (expanding) mean, and a rolling-window mean. A few columns get a
different treatment -- see COLUMN_INFO's "mode" field and its comment.

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

# (y-axis label, one-line description, plot mode) per column -- lifted
# directly from data.dat's own header comment (_extend_buffer) so a plot
# alone explains itself instead of requiring you to remember or go look up
# what a column means. Added after "I can't remember what log_a_AB and
# log_a_BA was for and I can't discern it from the plots" -- same fix
# applied to every column, not just those two, and 'time' now carries its
# unit ([s]) in the axis label itself rather than needing a caption at all.
#
# mode:
#   "standard" -- raw (thin, alpha=0.25) + cumulative mean + rolling mean,
#                 the default treatment.
#   "raw_only" -- time/acc_moves/acc_rate are THEMSELVES already-cumulative
#                 bookkeeping quantities (data.dat defines acc_rate as
#                 acc_moves/move, i.e. already a running average). Taking a
#                 cumulative-mean-of-a-cumulative-value or a rolling-mean-
#                 of-a-running-average doesn't have a clean interpretation
#                 and looked exactly that confusing in practice (acc_rate's
#                 "cumulative mean" line is a doubly-smoothed, slower-
#                 converging shadow of its own raw curve; time's cumulative
#                 mean of a roughly-linear ramp is just a differently-
#                 scaled ramp) -- plot the raw trajectory alone instead,
#                 which is already the informative view for these.
#   "no_raw"   -- accepted is a dense 0/1 series; 300000 raw points at low
#                 alpha render as a solid gray block (confirmed by looking
#                 at the actual pushed plot), not an informative trace.
#                 Cumulative + rolling mean (i.e. cumulative/windowed
#                 acceptance rate) are still shown; the raw scatter is not.
COLUMN_INFO = {
	"time":            ("time [s]", "CPU time elapsed since simulation start", "raw_only"),
	"U":               ("U", "potential energy of the current accepted sequence", "standard"),
	"U_am":            ("U_am", "attention-map contribution to U", "standard"),
	"U_structure_ce":  ("U_structure_ce", "structure-token cross-entropy contribution to U (0 if unused)", "standard"),
	"entropy":         ("entropy", "Shannon entropy of the PROPOSED SITE's softmax probabilities (per-move, "
						 "single-site -- NOT the K-sites ensemble site entropy S(i) from identify_k_sites.py, "
						 "a different quantity computed over many sampled sequences)", "standard"),
	"Hd_to_ref":       ("Hd_to_ref", "Hamming distance, current accepted sequence vs reference", "standard"),
	"dU":              ("dU", "U(proposed) - U(current)", "standard"),
	"log_a_AB":        ("log_a_AB", "log-probability of proposing THIS substitution (forward, A->B)", "standard"),
	"log_a_BA":        ("log_a_BA", "log-probability of the REVERSE proposal (B->A), used in the M-H ratio", "standard"),
	"log_ratio":       ("log_ratio", "log Metropolis-Hastings ratio: -dU/T + log_a_BA - log_a_AB", "standard"),
	"accepted":        ("accepted (0/1)", "1 if this move's proposal was accepted, 0 if rejected", "no_raw"),
	"acc_moves":       ("acc_moves (count)", "cumulative number of accepted moves", "raw_only"),
	"acc_rate":        ("acc_rate", "cumulative acceptance rate, acc_moves / move (already a running average)", "raw_only"),
}


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
		ylabel, desc, mode = COLUMN_INFO.get(col, (col, "", "standard"))

		fig, ax = plt.subplots(figsize=(10, 4))

		if mode == "raw_only":
			ax.plot(df["move"], series, linewidth=1.2)
			final_val = series.dropna().iloc[-1] if series.notna().any() else float("nan")
			# raw_only columns are already-cumulative bookkeeping quantities
			# (see comment above COLUMN_INFO), so the single most useful
			# number for one of them IS its final value -- e.g. "what was
			# the overall acceptance rate" -- printed directly on the plot
			# instead of requiring a console re-run to find it.
			ax.annotate(f"final: {final_val:.4g}", xy=(0.98, 0.02), xycoords="axes fraction",
						ha="right", va="bottom", fontsize=9,
						bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
			summary = f"raw (already a cumulative/running quantity, see module docstring), final={final_val:.4g}"
		else:
			if mode != "no_raw":
				ax.plot(df["move"], series, alpha=0.25, linewidth=0.5, color="gray", label="raw")
			cumulative = series.expanding().mean()
			rolling = series.rolling(args.window, min_periods=1).mean()
			# alpha=0.7 (not 1.0): cumulative and rolling frequently sit
			# nearly on top of each other once a run has settled, and were
			# previously indistinguishable where they overlapped.
			ax.plot(df["move"], cumulative, linewidth=1.5, alpha=0.7, label="cumulative mean")
			ax.plot(df["move"], rolling, linewidth=1.5, alpha=0.7, label=f"rolling mean (window={args.window})")
			ax.legend(fontsize=8)
			summary = ("cumulative + rolling" if mode == "no_raw"
					   else f"raw + cumulative + rolling(window={args.window})")

		ax.set_xlabel("move")
		ax.set_ylabel(ylabel)
		title = f"{col} vs move -- {desc}" if desc else f"{col} vs move"
		ax.set_title(f"{title}\n{args.results_dir}", fontsize=9)
		fig.tight_layout()

		out_path = os.path.join(out_dir, f"{col}.png")
		fig.savefig(out_path, dpi=120)
		plt.close(fig)
		print(f"  {col}: {summary} -> {out_path}")

	print("Done.")


if __name__ == "__main__":
	main()
