"""
Single combined summary file across an entire T-sweep -- one row per T,
built straight from each T's data.dat (same field defs as analyze_run.py,
whose per-run text report this reuses the load_rows()/dU-accepted-vs-all
logic from) plus, where compute_structural_metrics.py has already been run
for that T, structural_metrics.csv's Rg/pLDDT averages. Answers "how does
T=0.25 compare to T=2.5" from one file instead of re-reading N separate
console outputs or N separate plot PNGs.

Same post-burn-in window convention as every other multi-T script in this
project (identify_k_sites.py, plot_plddt_vs_T.py, analyze_ensemble.py):
the first --burn-in-frac (default 0.5) of MOVES is dropped, and every
"mean X" column below is the mean over that kept window -- not a per-block
breakdown (see analyze_run.py for that instead, one run at a time).

Usage:
    python summarize_sweep.py \\
        --results-dirs tests_rate_sweep_temperature_paper_scale/T0.2512/sim0 \\
                        tests_rate_sweep_temperature_paper_scale/T1.0/sim0 \\
        --active-t 0.2512 1.0

--results-dirs and --active-t must be the same length and in the same
order (one T label per results_dir) -- same convention as
identify_k_sites.py/plot_plddt_vs_T.py. Writes summary_by_T.csv to
--out-dir (default k_sites_plots/) and prints the same table to stdout.
"""
import argparse
import csv
import os
import statistics

from analyze_run import load_rows, DEFAULT_REF_SEQ


def summarize_run(results_dir, T, burn_in_frac):
	rows = load_rows(os.path.join(results_dir, "data.dat"))
	n_total = len(rows)
	burn_in_n = int(n_total * burn_in_frac)
	post = rows[burn_in_n:]  # move 0's placeholder row (dU=0, accepted=0) is
	                          # always inside the discarded first half for any
	                          # run long enough to be worth summarizing, so no
	                          # separate skip-first-row step is needed here.

	moves = [int(r["move"]) for r in post]
	U = [float(r["U"]) for r in post]
	entropy = [float(r["entropy"]) for r in post]
	Hd = [int(r["Hd_to_ref"]) for r in post]
	dU_all = [float(r["dU"]) for r in post]
	accepted = [r["accepted"] for r in post]
	dU_acc = [d for d, a in zip(dU_all, accepted) if a == "1"]

	n_prop = len(post)
	n_acc = sum(1 for a in accepted if a == "1")
	acc_rate_window = n_acc / n_prop if n_prop else float("nan")
	acc_rate_final_cumulative = float(rows[-1]["acc_rate"])

	time_start, time_end = float(post[0]["time"]), float(rows[-1]["time"])
	move_start, move_end = moves[0], moves[-1]
	time_per_move_s = ((time_end - time_start) / (move_end - move_start)
						if move_end > move_start else float("nan"))

	row = {
		"T": T,
		"n_moves_analyzed": n_prop,
		"mean_dU_all_moves": statistics.mean(dU_all) if dU_all else float("nan"),
		"mean_dU_accepted_moves": statistics.mean(dU_acc) if dU_acc else float("nan"),
		"mean_dU_over_T_all_moves": (statistics.mean(dU_all) / T) if dU_all else float("nan"),
		"mean_dU_over_T_accepted_moves": (statistics.mean(dU_acc) / T) if dU_acc else float("nan"),
		"mean_U": statistics.mean(U) if U else float("nan"),
		"mean_U_over_T": (statistics.mean(U) / T) if U else float("nan"),
		"acceptance_rate_post_burn_in": acc_rate_window,
		"acceptance_rate_final_cumulative": acc_rate_final_cumulative,
		"time_per_move_s": time_per_move_s,
		"mean_entropy": statistics.mean(entropy) if entropy else float("nan"),
		"mean_Hd_to_ref": statistics.mean(Hd) if Hd else float("nan"),
	}

	# structural_metrics.csv is optional -- only exists once
	# compute_structural_metrics.py has been run for this results_dir.
	csv_path = os.path.join(results_dir, "structural_metrics.csv")
	if os.path.exists(csv_path):
		rg_ca, rg_allatom, plddt = [], [], []
		with open(csv_path) as f:
			for r in csv.DictReader(f):
				rg_ca.append(float(r["Rg_ca"]))
				rg_allatom.append(float(r["Rg_allatom"]))
				plddt.append(float(r["mean_plddt"]))
		row["n_structural_checkpoints"] = len(rg_ca)
		row["mean_Rg_ca"] = statistics.mean(rg_ca) if rg_ca else float("nan")
		row["mean_Rg_allatom"] = statistics.mean(rg_allatom) if rg_allatom else float("nan")
		row["mean_plddt"] = statistics.mean(plddt) if plddt else float("nan")
	else:
		row["n_structural_checkpoints"] = 0
		row["mean_Rg_ca"] = row["mean_Rg_allatom"] = row["mean_plddt"] = float("nan")

	return row


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--results-dirs", nargs="+", required=True)
	parser.add_argument("--active-t", nargs="+", type=float, required=True,
						 help="one temperature label per --results-dirs entry, same order")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--burn-in-frac", type=float, default=0.5)
	parser.add_argument("--out-dir", default="k_sites_plots")
	args = parser.parse_args()

	if len(args.results_dirs) != len(args.active_t):
		raise ValueError(
			f"--results-dirs ({len(args.results_dirs)}) and --active-t ({len(args.active_t)}) "
			f"must have the same length -- one T per results_dir, same order."
		)

	rows = []
	for results_dir, T in zip(args.results_dirs, args.active_t):
		row = summarize_run(results_dir, T, args.burn_in_frac)
		rows.append(row)
		print(f"T={T}: n_moves_analyzed={row['n_moves_analyzed']}  "
			  f"mean_dU(all/acc)={row['mean_dU_all_moves']:+.3f}/{row['mean_dU_accepted_moves']:+.3f}  "
			  f"mean_U={row['mean_U']:.3f}  acc_rate={row['acceptance_rate_post_burn_in']:.3f}  "
			  f"time/move={row['time_per_move_s']:.4f}s")

	os.makedirs(args.out_dir, exist_ok=True)
	out_path = os.path.join(args.out_dir, "summary_by_T.csv")
	with open(out_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)
	print(f"\nSummary table ({len(rows)} T's) -> {out_path}")


if __name__ == "__main__":
	main()
