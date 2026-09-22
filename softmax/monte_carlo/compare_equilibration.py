"""
Cross-condition comparison for run_equilibration.py / sweep_equilibration.sh:
does U/Hd_to_ref converge to the same regime regardless of starting point
(reference / moderate / far), or do dispersed starts stay in visibly
different, non-overlapping regimes? See run_equilibration.py's docstring
for the full rationale.

Reuses monte_carlo/analyze_run.py's load_rows() and block_stats() directly
rather than re-parsing data.dat itself.

Usage (from the monte_carlo directory, after sweep_equilibration.sh has
completed all 6 runs):
    python compare_equilibration.py
"""
import os
import statistics

from analyze_run import load_rows, block_stats


START_MODES = ["reference", "moderate", "far"]
SEEDS = [42, 43]
RESULTS_ROOT = "tests_rate_equilibration"
N_BLOCKS = 10
TAIL_FRACTION = 0.3  # last 30% of moves treated as the "settled" window


def main():
	per_mode_rows = {}
	for mode in START_MODES:
		rows_all = []
		for seed in SEEDS:
			path = os.path.join(RESULTS_ROOT, f"{mode}_seed{seed}", "data.dat")
			if not os.path.exists(path):
				print(f"WARNING: missing {path}, skipping")
				continue
			rows = load_rows(path)
			rows_all.append((seed, rows))
		per_mode_rows[mode] = rows_all

	# ================================================================
	# per-mode, per-seed block trajectories
	# ================================================================
	print("=== U trajectory by block, per run (mean U per block of the run) ===")
	for mode in START_MODES:
		for seed, rows in per_mode_rows[mode]:
			U = [float(r['U']) for r in rows]
			blocks = block_stats(U, n_blocks=N_BLOCKS)
			block_means = "  ".join(f"{avg:6.2f}" for _, _, avg in blocks)
			print(f"  {mode:9s} seed={seed}: {block_means}")

	print("\n=== Hd_to_ref trajectory by block, per run ===")
	for mode in START_MODES:
		for seed, rows in per_mode_rows[mode]:
			Hd = [int(r['Hd_to_ref']) for r in rows]
			blocks = block_stats(Hd, n_blocks=N_BLOCKS)
			block_means = "  ".join(f"{avg:5.1f}" for _, _, avg in blocks)
			print(f"  {mode:9s} seed={seed}: {block_means}")

	# ================================================================
	# tail-window comparison (pooled across seeds within each mode)
	# ================================================================
	print(f"\n=== tail window (last {TAIL_FRACTION:.0%} of moves), pooled across seeds per mode ===")
	tail_stats = {}
	for mode in START_MODES:
		U_tail, Hd_tail = [], []
		for seed, rows in per_mode_rows[mode]:
			n = len(rows)
			cutoff = int(n * (1 - TAIL_FRACTION))
			U_tail.extend(float(r['U']) for r in rows[cutoff:])
			Hd_tail.extend(int(r['Hd_to_ref']) for r in rows[cutoff:])
		if not U_tail:
			continue
		tail_stats[mode] = {
			"U_mean": statistics.mean(U_tail), "U_std": statistics.stdev(U_tail) if len(U_tail) > 1 else 0.,
			"Hd_mean": statistics.mean(Hd_tail), "Hd_std": statistics.stdev(Hd_tail) if len(Hd_tail) > 1 else 0.,
			"n": len(U_tail),
		}
		s = tail_stats[mode]
		print(f"  {mode:9s}: U={s['U_mean']:7.3f} +/- {s['U_std']:6.3f}   "
			  f"Hd={s['Hd_mean']:6.2f} +/- {s['Hd_std']:5.2f}   (n={s['n']})")

	print(f"\n(If all three modes' tail U/Hd substantially overlap -- means within a couple")
	print(f" std devs of each other -- that's evidence of genuine convergence to a common")
	print(f" equilibrium regime regardless of start. If 'reference' stays systematically")
	print(f" lower in U/Hd than 'moderate'/'far' even in the tail, with non-overlapping")
	print(f" bands, that supports a distant, barrier-separated basin instead. Check the")
	print(f" block trajectories above too -- a mode that's still visibly rising in its last")
	print(f" few blocks hasn't settled yet regardless of what the tail window says.)")


if __name__ == "__main__":
	main()
