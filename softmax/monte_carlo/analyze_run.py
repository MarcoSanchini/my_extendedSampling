"""
Analyze a classes/rate_sampler.py:ExtendedProteinRateSampler run's
data.dat log. Pure Python stdlib only (no torch/numpy/pandas needed), so it
runs anywhere -- including environments that don't have the ESM3/torch
stack installed, since everything it needs (site, cur_aa, proposed_aa,
accepted, U, dU, ...) is already recorded per move.

Usage:
    python analyze_run.py <results_dir>       # e.g. tests_rate/sim0
    python analyze_run.py <results_dir> --ref-seq <SEQUENCE>

What it reports:
  - U / Hd_to_ref trajectory summary (start vs end, min/max, first-vs-last
    block average, to see the drift direction over the run)
  - windowed (block) acceptance rate, to see if it changes over the course
    of the run rather than just the final cumulative number
  - dU statistics, split by accepted vs rejected
  - per-site accept/reject breakdown, split into "started already matching
    the reference" vs "started as one of the initial mutations" (the
    initial sequence is reconstructed purely from the log: for each site,
    the first time it's touched, cur_aa records what it was before that
    move -- no .pt files needed). This is the sanity check noted in
    DEVLOG.txt: do proposals at already-good (reference-matching) sites
    get rejected more often than proposals at genuinely-suboptimal ones?
"""
import argparse
import os
import statistics


DEFAULT_REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def load_rows(data_dat_path):
	with open(data_dat_path) as f:
		lines = [l for l in f if l.strip() and not l.startswith('#')]
	header = lines[0].rstrip('\n').split('\t')
	rows = []
	for line in lines[1:]:
		vals = line.rstrip('\n').split('\t')
		rows.append(dict(zip(header, vals)))
	return rows


def reconstruct_initial_sequence(rows, L):
	seq = [None]*L
	unresolved = set(range(L))
	for row in rows:
		if row['move'] == '0':
			continue
		site = int(row['site'])
		if site in unresolved:
			seq[site] = row['cur_aa']
			unresolved.discard(site)
	return seq, unresolved


def block_stats(values, n_blocks=10):
	n = len(values)
	size = max(1, n // n_blocks)
	blocks = []
	for i in range(0, n, size):
		chunk = values[i:i+size]
		if chunk:
			blocks.append((i, i+len(chunk), sum(chunk)/len(chunk)))
	return blocks


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_dir", help="e.g. tests_rate/sim0")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	args = parser.parse_args()

	data_dat = os.path.join(args.results_dir, "data.dat")
	rows = load_rows(data_dat)
	ref_seq = args.ref_seq
	L = len(ref_seq)

	print(f"Loaded {len(rows)} rows from {data_dat}")
	print(f"Reference length: {L}\n")

	# --- reconstruct initial sequence purely from the log ---
	init_seq, unresolved = reconstruct_initial_sequence(rows, L)
	if unresolved:
		print(f"WARNING: {len(unresolved)} site(s) never touched in this run, "
			  f"can't recover their initial identity from the log alone: {sorted(unresolved)}")
		for s in unresolved:
			init_seq[s] = ref_seq[s]  # assume unchanged (best guess, not verified)

	init_seq_str = "".join(init_seq)
	mutated_sites = sorted(s for s in range(L) if init_seq[s] != ref_seq[s])
	reconstructed_hd = len(mutated_sites)
	reported_hd0 = int(rows[0]['Hd_to_ref'])

	print(f"Reconstructed initial sequence: {init_seq_str}")
	print(f"Initially mutated sites (vs reference): {mutated_sites} "
		  f"({reconstructed_hd} sites)")
	print(f"Reconstructed Hd_to_ref at move 0: {reconstructed_hd}  "
		  f"(logged: {reported_hd0})  {'OK' if reconstructed_hd == reported_hd0 else 'MISMATCH -- check unresolved sites above'}\n")

	# --- U / Hd_to_ref trajectory ---
	moves = [int(r['move']) for r in rows]
	U = [float(r['U']) for r in rows]
	Hd = [int(r['Hd_to_ref']) for r in rows]
	accepted = [r['accepted'] for r in rows]  # '' at move 0, else '0'/'1'

	print("=== U trajectory ===")
	print(f"  move 0: U={U[0]:.3f}   move {moves[-1]}: U={U[-1]:.3f}   "
		  f"min={min(U):.3f}  max={max(U):.3f}")
	for start, end, avg in block_stats(U, n_blocks=10):
		print(f"  moves {start:4d}-{end-1:4d}: mean U = {avg:8.3f}")

	print("\n=== Hd_to_ref trajectory ===")
	print(f"  move 0: Hd={Hd[0]}   move {moves[-1]}: Hd={Hd[-1]}   "
		  f"min={min(Hd)}  max={max(Hd)}")
	for start, end, avg in block_stats(Hd, n_blocks=10):
		print(f"  moves {start:4d}-{end-1:4d}: mean Hd = {avg:6.2f}")

	# --- windowed acceptance rate ---
	print("\n=== windowed acceptance rate (block-local, not cumulative) ===")
	acc_binary = [1 if r['accepted'] == '1' else 0 for r in rows[1:]]  # skip move 0
	for start, end, avg in block_stats(acc_binary, n_blocks=10):
		print(f"  moves {start+1:4d}-{end:4d}: local acc_rate = {avg:.3f}")
	print(f"  overall: {sum(acc_binary)}/{len(acc_binary)} = {sum(acc_binary)/len(acc_binary):.3f}")

	# --- dU stats, accepted vs rejected ---
	dU_acc = [float(r['dU']) for r in rows[1:] if r['accepted'] == '1']
	dU_rej = [float(r['dU']) for r in rows[1:] if r['accepted'] == '0']
	print("\n=== dU statistics ===")
	if dU_acc:
		print(f"  accepted (n={len(dU_acc)}): mean={statistics.mean(dU_acc):+8.3f}  "
			  f"median={statistics.median(dU_acc):+8.3f}  min={min(dU_acc):+8.3f}  max={max(dU_acc):+8.3f}")
	if dU_rej:
		print(f"  rejected (n={len(dU_rej)}): mean={statistics.mean(dU_rej):+8.3f}  "
			  f"median={statistics.median(dU_rej):+8.3f}  min={min(dU_rej):+8.3f}  max={max(dU_rej):+8.3f}")

	# --- per-site: mutated-at-start vs reference-matching-at-start ---
	site_touches = {}   # site -> [n_proposed, n_accepted]
	for r in rows[1:]:
		s = int(r['site'])
		site_touches.setdefault(s, [0, 0])
		site_touches[s][0] += 1
		if r['accepted'] == '1':
			site_touches[s][1] += 1

	mutated_set = set(mutated_sites)
	def group_stats(site_set):
		n_prop = sum(site_touches.get(s, [0,0])[0] for s in site_set)
		n_acc = sum(site_touches.get(s, [0,0])[1] for s in site_set)
		return n_prop, n_acc, (n_acc/n_prop if n_prop else float('nan'))

	mp, ma, mr = group_stats(mutated_set)
	rp, ra, rr = group_stats(set(range(L)) - mutated_set)
	print("\n=== accept rate: initially-mutated sites vs initially-reference-matching sites ===")
	print(f"  initially-mutated     ({len(mutated_set):2d} sites): {ma:4d}/{mp:4d} proposals accepted = {mr:.3f}")
	print(f"  initially ref-matching({L-len(mutated_set):2d} sites): {ra:4d}/{rp:4d} proposals accepted = {rr:.3f}")
	print(f"  (if the energy is behaving sensibly, the reference-matching group's rate should be")
	print(f"   noticeably LOWER -- those sites are already at a local optimum and should mostly")
	print(f"   resist being proposed away from it.)")

	# --- final drift: how many sites differ from reference now that didn't at the start ---
	print(f"\n=== net drift ===")
	print(f"  Hd_to_ref: {Hd[0]} (start) -> {Hd[-1]} (end), net change of {Hd[-1]-Hd[0]:+d} over {moves[-1]} moves")


if __name__ == "__main__":
	main()
