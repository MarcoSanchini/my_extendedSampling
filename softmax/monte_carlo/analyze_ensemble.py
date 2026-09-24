"""
Per-temperature ensemble analysis for sweep_temperature.sh: site entropy
S(i) and the pairwise sequence-similarity distribution p(q), the two
target results from Zambon et al 2024 chosen as the first, cheapest ones
to try to reproduce (see DEVLOG.txt, 2026-09-23 entries).

Unlike everything analyze_run.py computes, this needs many FULL sampled
sequences along the run, not just the move/U/Hd_to_ref/site columns
data.dat has -- those come from the eprot checkpoint files
(<results_dir>/eprot/sequence_<move>.pt) that classes/rate_sampler.py:
_save_log writes every settings['log_step'] moves (sweep_temperature.sh
sets log_step=100, giving 500 checkpoints over a 50000-move run).

Definitions match the paper directly:
  site entropy S(i) = -sum_alpha p_i(alpha) log p_i(alpha), natural log,
    0 if site i is perfectly conserved across the (post-burn-in) sampled
    ensemble, log(20)=2.9957 if it's uniform over all 20 canonical amino
    acids (eq. in Zambon et al's section 3.1 -- "Structure of the space
    of sequences").
  q(sigma, sigma') = fraction of sites with the same amino acid identity
    between two sampled sequences (their eq., section 3.1) -- computed
    here over all pairs among the post-burn-in checkpoints (or a random
    subsample if there are many, see MAX_PAIRS below).

Usage:
    python analyze_ensemble.py <results_dir>       # e.g. tests_rate_sweep_temperature/T2.0/sim0
    python analyze_ensemble.py <results_dir> --burn-in-frac 0.5 --ref-seq <SEQUENCE>
"""
import argparse
import glob
import itertools
import math
import os
import random
import re
import statistics

import torch

from analyze_run import load_rows, block_stats


DEFAULT_REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
MAX_PAIRS = 200_000  # cap on pairwise q comparisons if there are many checkpoints


def load_checkpoint_sequences(results_dir, burn_in_frac):
	eprot_dir = os.path.join(results_dir, "eprot")
	paths = glob.glob(os.path.join(eprot_dir, "sequence_*.pt"))
	if not paths:
		raise FileNotFoundError(f"No sequence_*.pt checkpoints found in {eprot_dir} -- "
								 f"was this run launched with a finite settings['log_step']?")

	moves_and_paths = []
	for p in paths:
		m = re.search(r"sequence_(\d+)\.pt$", p)
		if m:
			moves_and_paths.append((int(m.group(1)), p))
	moves_and_paths.sort(key=lambda mp: mp[0])

	max_move = moves_and_paths[-1][0]
	cutoff = max_move * burn_in_frac
	kept = [(m, p) for m, p in moves_and_paths if m >= cutoff]

	sequences = [(m, torch.load(p, weights_only=False)) for m, p in kept]
	return sequences, max_move, len(moves_and_paths)


def site_entropy(sequences, L):
	# sequences: list of (move, seq_str)
	counts = [dict() for _ in range(L)]
	for _, seq in sequences:
		for i, aa in enumerate(seq):
			counts[i][aa] = counts[i].get(aa, 0) + 1

	n = len(sequences)
	entropies = []
	for i in range(L):
		S = 0.0
		for aa, c in counts[i].items():
			p = c / n
			S -= p * math.log(p)
		entropies.append(S)
	return entropies, counts


def q_distribution(sequences, seed=0):
	n = len(sequences)
	all_pairs = list(itertools.combinations(range(n), 2))
	if len(all_pairs) > MAX_PAIRS:
		rng = random.Random(seed)
		pairs = rng.sample(all_pairs, MAX_PAIRS)
	else:
		pairs = all_pairs

	qs = []
	for i, j in pairs:
		seq_i, seq_j = sequences[i][1], sequences[j][1]
		matches = sum(1 for a, b in zip(seq_i, seq_j) if a == b)
		qs.append(matches / len(seq_i))
	return qs


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_dir", help="e.g. tests_rate_sweep_temperature/T2.0/sim0")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--burn-in-frac", type=float, default=0.5,
						 help="discard checkpoints before this fraction of the run's final move number")
	args = parser.parse_args()

	ref_seq = args.ref_seq
	L = len(ref_seq)

	sequences, max_move, n_checkpoints_total = load_checkpoint_sequences(args.results_dir, args.burn_in_frac)
	print(f"Loaded {len(sequences)}/{n_checkpoints_total} checkpoints post burn-in "
		  f"(kept moves >= {args.burn_in_frac:.0%} of {max_move})")

	# --- context: U/Hd_to_ref trajectory, same as analyze_run.py ---
	data_dat = os.path.join(args.results_dir, "data.dat")
	if os.path.exists(data_dat):
		rows = load_rows(data_dat)
		U = [float(r['U']) for r in rows]
		Hd = [int(r['Hd_to_ref']) for r in rows]
		print(f"\nU trajectory: move 0={U[0]:.3f}  move {rows[-1]['move']}={U[-1]:.3f}")
		for start, end, avg in block_stats(U, n_blocks=10):
			print(f"  moves {start:6d}-{end-1:6d}: mean U = {avg:8.3f}")
		print(f"Hd_to_ref trajectory: move 0={Hd[0]}  move {rows[-1]['move']}={Hd[-1]}")

	# --- site entropy ---
	print(f"\n=== site entropy (natural log; 0=perfectly conserved, log(20)={math.log(20):.4f}=uniform) ===")
	entropies, counts = site_entropy(sequences, L)
	mean_S = statistics.mean(entropies)
	print(f"Mean site entropy: {mean_S:.4f}  (fraction of log(20): {mean_S/math.log(20):.3f})")

	# "K-sites" candidate: lowest-entropy sites, paper's own qualitative
	# marker (they report the sites that are essentially fixed/completely
	# conserved as the low end of this distribution -- print the extremes
	# for a first look, not a hard threshold).
	ranked = sorted(range(L), key=lambda i: entropies[i])
	print(f"\nMost conserved 10 sites (lowest S(i)): "
		  f"{[(i, ref_seq[i], round(entropies[i], 3)) for i in ranked[:10]]}")
	print(f"Least conserved 10 sites (highest S(i)): "
		  f"{[(i, ref_seq[i], round(entropies[i], 3)) for i in ranked[-10:]]}")

	print(f"\nPer-site entropy (site: ref_aa S(i)):")
	for i in range(L):
		print(f"  {i:3d} {ref_seq[i]}: {entropies[i]:.4f}")

	# --- q-distribution ---
	print(f"\n=== pairwise sequence similarity q ===")
	qs = q_distribution(sequences)
	print(f"n_pairs={len(qs)}  mean(q)={statistics.mean(qs):.4f}  "
		  f"median(q)={statistics.median(qs):.4f}  std(q)={statistics.stdev(qs):.4f}  "
		  f"min={min(qs):.4f}  max={max(qs):.4f}")

	# coarse histogram (10 bins), no plotting dependency needed
	n_bins = 10
	hist = [0]*n_bins
	for q in qs:
		b = min(int(q*n_bins), n_bins-1)
		hist[b] += 1
	print("q histogram (10 bins over [0,1]):")
	for b in range(n_bins):
		lo, hi = b/n_bins, (b+1)/n_bins
		bar = "#" * int(60*hist[b]/max(hist))
		print(f"  [{lo:.1f},{hi:.1f}): {hist[b]:6d}  {bar}")

	# --- burn-in sanity check: compare first half vs second half of the
	# post-burn-in window's mean entropy/q, flag if still trending ---
	half = len(sequences)//2
	if half >= 2:
		S1, _ = site_entropy(sequences[:half], L)
		S2, _ = site_entropy(sequences[half:], L)
		mean_S1, mean_S2 = statistics.mean(S1), statistics.mean(S2)
		denom = max(mean_S1, mean_S2, 1e-9)
		rel_gap = abs(mean_S2 - mean_S1) / denom
		# 10% relative change between halves is an arbitrary but reasonable
		# threshold for "still trending" vs "settled" -- not calibrated
		# against anything beyond eyeballing the 12-point T-sweep's own
		# results (T=10 showed ~3.5% and looked like the one point among
		# the active T's still trending; T=1/T=100 showed <1%).
		trending = rel_gap > 0.10
		note = (f"(gap is {rel_gap:.1%} of the mean -- still trending, "
				f"the {args.burn_in_frac:.0%} cutoff may not be enough yet)" if trending
				else f"(gap is only {rel_gap:.1%} of the mean -- looks settled)")
		print(f"\nBurn-in check: mean S(i) in first half of kept window={mean_S1:.4f}, "
			  f"second half={mean_S2:.4f}  {note}")


if __name__ == "__main__":
	main()
