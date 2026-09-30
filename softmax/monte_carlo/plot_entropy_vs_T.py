"""
Mean site entropy AND mean pairwise similarity q vs T, across however many
temperatures you pass in -- the "how fast does full disorder set in" view
flagged as outstanding after the paper-scale sweep's cold-subset K-sites
run (see DEVLOG.txt, 2026-09-30 entries): identify_k_sites.py plots
per-site entropy (one line per site, several T's), and analyze_ensemble.py
reports mean S(i)/mean q as text for ONE T at a time, but nothing so far
has plotted the SUMMARY curve (one point per T) the way this project's
own prose has been describing the transition all along. Also the closest
thing this project has to the paper's own figure 2 upper panel (Ē(T)/Cv(T))
shape-wise, though it plots a different quantity (entropy/q, not energy/
specific heat -- those still need the deferred multi-histogram reweighting
stage, not attempted here).

Cheap: only needs the already-saved sequence strings (same as
analyze_ensemble.py/identify_k_sites.py), no model calls.

Usage:
    python plot_entropy_vs_T.py \\
        --results-dirs tests_rate_sweep_temperature_paper_scale/T0.2512/sim0 \\
                        tests_rate_sweep_temperature_paper_scale/T0.3981/sim0 \\
                        tests_rate_sweep_temperature_paper_scale/T0.631/sim0 \\
                        tests_rate_sweep_temperature_paper_scale/T1.0/sim0 \\
                        tests_rate_sweep_temperature_paper_scale/T1.585/sim0 \\
                        tests_rate_sweep_temperature_paper_scale/T2.512/sim0 \\
        --active-t 0.2512 0.3981 0.631 1.0 1.585 2.512

Unlike identify_k_sites.py, this is meant to cover the FULL T range
(cold through hot) in one call -- there's no "only pass cold T's" caveat
here, since a summary curve showing the frozen->disordered shape is
exactly the point.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_ensemble import load_checkpoint_sequences, site_entropy, q_distribution, DEFAULT_REF_SEQ


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

	L = len(args.ref_seq)
	max_S = math.log(20)

	Ts, mean_S_vals, mean_q_vals = [], [], []
	for results_dir, T in zip(args.results_dirs, args.active_t):
		sequences, max_move, n_total = load_checkpoint_sequences(results_dir, args.burn_in_frac)
		entropies, _ = site_entropy(sequences, L)
		mean_S = sum(entropies) / L
		qs = q_distribution(sequences)
		mean_q = sum(qs) / len(qs)

		Ts.append(T)
		mean_S_vals.append(mean_S)
		mean_q_vals.append(mean_q)
		print(f"T={T}: {len(sequences)}/{n_total} checkpoints, mean S(i)={mean_S:.4f} "
			  f"({mean_S/max_S:.1%} of log(20)), mean q={mean_q:.4f}")

	os.makedirs(args.out_dir, exist_ok=True)

	fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

	axes[0].plot(Ts, [s / max_S for s in mean_S_vals], marker='o')
	axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=1, label='log(20) = fully uniform')
	axes[0].set_xscale("log")
	axes[0].set_xlabel("T")
	axes[0].set_ylabel("mean S(i) / log(20)")
	axes[0].set_title("Mean site entropy vs T")
	axes[0].legend(fontsize=8)

	axes[1].plot(Ts, mean_q_vals, marker='o', color='tab:orange')
	axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=1, label='q=1 (identical to reference)')
	axes[1].set_xscale("log")
	axes[1].set_xlabel("T")
	axes[1].set_ylabel("mean pairwise similarity q")
	axes[1].set_title("Mean q vs T")
	axes[1].legend(fontsize=8)

	fig.suptitle("Disorder onset across the sweep -- frozen (S~0, q~1) to fully random (S~log(20), q~1/20)")
	fig.tight_layout()

	out_path = os.path.join(args.out_dir, "entropy_and_q_vs_T.png")
	fig.savefig(out_path, dpi=120)
	plt.close(fig)
	print(f"\nentropy/q vs T -> {out_path}")


if __name__ == "__main__":
	main()
