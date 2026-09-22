"""
Pool the "gradient-informedness" signal (accept rate at initially-mutated
sites vs initially-reference-matching sites -- see analyze_run.py) across
several independent runs, typically several seeds at the same dt/T, into
one combined estimate with a real standard error, instead of eyeballing
several single-run printouts side by side.

Each seed draws its own init_muts pattern (not just a different downstream
random stream from an identical start), so pooling across seeds is a real
independent-replicate estimate, not double-counting one trajectory.

Usage:
    python aggregate_seeds.py <results_dir1> <results_dir2> ... [--ref-seq SEQ]

e.g. (bash brace expansion):
    python aggregate_seeds.py tests_rate_sweep_dt_seeds/dt2000_seed{42,43,44,45}/sim0
"""
import argparse
import os

from analyze_run import load_rows, reconstruct_initial_sequence, DEFAULT_REF_SEQ


def per_run_counts(results_dir, ref_seq):
	rows = load_rows(os.path.join(results_dir, "data.dat"))
	L = len(ref_seq)
	init_seq, unresolved = reconstruct_initial_sequence(rows, L)
	for s in unresolved:
		init_seq[s] = ref_seq[s]  # best guess, matches analyze_run.py
	mutated_set = set(s for s in range(L) if init_seq[s] != ref_seq[s])

	site_touches = {}   # site -> [n_proposed, n_accepted]
	for r in rows[1:]:
		s = int(r['site'])
		site_touches.setdefault(s, [0, 0])
		site_touches[s][0] += 1
		if r['accepted'] == '1':
			site_touches[s][1] += 1

	def group_counts(site_set):
		n_prop = sum(site_touches.get(s, [0, 0])[0] for s in site_set)
		n_acc = sum(site_touches.get(s, [0, 0])[1] for s in site_set)
		return n_prop, n_acc

	mp, ma = group_counts(mutated_set)
	rp, ra = group_counts(set(range(L)) - mutated_set)
	overall_acc = sum(1 for r in rows[1:] if r['accepted'] == '1')

	return {
		"n_mutated_sites": len(mutated_set),
		"mutated_prop": mp, "mutated_acc": ma,
		"ref_prop": rp, "ref_acc": ra,
		"overall_acc": overall_acc, "overall_n": len(rows) - 1,
	}


def binomial_se(k, n):
	if n == 0:
		return float('nan')
	p = k / n
	return (p * (1 - p) / n) ** 0.5


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_dirs", nargs='+',
						 help="e.g. tests_rate_sweep_dt_seeds/dt2000_seed42/sim0 ...")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	args = parser.parse_args()

	per_run = [(d, per_run_counts(d, args.ref_seq)) for d in args.results_dirs]

	print(f"=== per-run breakdown ({len(per_run)} runs) ===")
	for d, c in per_run:
		p_mut = c['mutated_acc'] / c['mutated_prop'] if c['mutated_prop'] else float('nan')
		p_ref = c['ref_acc'] / c['ref_prop'] if c['ref_prop'] else float('nan')
		print(f"  {d}: {c['n_mutated_sites']} initially-mutated sites  "
			  f"mutated_acc={p_mut:.3f} ({c['mutated_acc']:3d}/{c['mutated_prop']:4d})  "
			  f"ref_acc={p_ref:.3f} ({c['ref_acc']:3d}/{c['ref_prop']:4d})  "
			  f"gap={p_mut - p_ref:+.3f}  "
			  f"overall={c['overall_acc']}/{c['overall_n']}={c['overall_acc']/c['overall_n']:.3f}")

	mp = sum(c['mutated_prop'] for _, c in per_run)
	ma = sum(c['mutated_acc'] for _, c in per_run)
	rp = sum(c['ref_prop'] for _, c in per_run)
	ra = sum(c['ref_acc'] for _, c in per_run)
	oa = sum(c['overall_acc'] for _, c in per_run)
	on = sum(c['overall_n'] for _, c in per_run)

	p_mut = ma / mp if mp else float('nan')
	p_ref = ra / rp if rp else float('nan')
	se_mut = binomial_se(ma, mp)
	se_ref = binomial_se(ra, rp)
	se_gap = (se_mut**2 + se_ref**2) ** 0.5
	gap = p_mut - p_ref

	print(f"\n=== pooled across {len(per_run)} runs ===")
	print(f"  mutated-site accept rate: {p_mut:.4f}  ({ma}/{mp})  SE={se_mut:.4f}")
	print(f"  ref-matching accept rate: {p_ref:.4f}  ({ra}/{rp})  SE={se_ref:.4f}")
	print(f"  gap (mutated - ref):      {gap:+.4f}  SE={se_gap:.4f}  gap/SE={gap/se_gap:+.2f}")
	print(f"  overall accept rate:      {oa}/{on} = {oa/on:.4f}")


if __name__ == "__main__":
	main()
