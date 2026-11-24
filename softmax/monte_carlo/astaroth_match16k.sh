#!/usr/bin/env bash
# Bring every temperature worth matching to 16000 ACCEPTED moves past its own t_eq.
#
#   nohup bash softmax/monte_carlo/astaroth_match16k.sh > astaroth_match16k.log 2>&1 &
#
# WHY. Temperatures must be compared at a matched accepted-move count, never at a matched
# burn-in fraction: the per-site entropy is estimated from the accepted moves in the window,
# so a short window compresses the site ranking and attenuates the Spearman against every
# reference. 16000 is the largest N that T=4, 6, 7, 8 and 16 already clear, and
# compare_ladder.sh's second pass uses it. These three temperatures fall short of it, and
# their windows at 16000 currently start BEFORE t_eq, which makes their rows lower bounds.
#
# INVENTORY measured 2026-11-24 (accepted moves past t_eq, per chain; t_eq = the R-hat
# crossing on U and Hd_to_ref):
#
#   T     moves    t_eq   acc. rate  accepted now    target  accepted after
#   2    650000  440000      0.0485  10309 / 10046   790000  ~17000 / ~16700
#   3    180000  100000      0.0928   7435 /  7407   280000  ~16700 / ~16700
#   5    180000  100000      0.1540  12348 / 12299   210000  ~16900 / ~16800
#
# The targets carry only a few percent of margin on purpose: the acceptance rate is measured
# to three digits over >= 80000 moves, and buying more is buying nothing. T=1 and T=1.5 are
# deliberately ABSENT. They would cost 428000 and 1434000 extra moves per chain (18 h and
# 60 h) and neither would pay: their burial-removed family correlation is flat or zero
# already, and 222/211 of 324 sites at T=1 see no accepted move in the whole post-t_eq
# window, so the limit there is reachability, not sample size. See
# /mnt/project-files/ladder/equilibrated_move_inventory.md.
#
# COST. astaroth runs two concurrent 324-aa chains at 0.1516 s/move each (1.47x the
# throughput of one at 0.1118), so each temperature's two chains run together and the
# temperatures run in sequence: 5.9 h for T=2, then 4.2 h for T=3, then 1.3 h for T=5.
# About 11.4 h in total.
#
# T=2.5 IS NOT HERE, because it ran on urano and its resume set (data.dat, generator.npy,
# log.pt, start.txt, eprot/*.pt) is gitignored and exists only there. Two ways to finish it,
# both in the companion note; the script picks it up automatically if the files appear:
#   on urano     170000 more moves at 1.084 s/move = 51 h
#   on astaroth  the same 170000 at 0.1118 s/move = 5.3 h, after one rsync of its run dir,
#                and urano must then never resume it again.
#
# --resume continues a run to a larger --moves; run_nsites.py asserts every parameter
# against the run's own pars.txt, so the command must match the original except for --moves,
# and --moves must stay a multiple of --log-step. dt is left to the dt/sqrt(T) rule, which
# reproduces the recorded 0.0285708 (T=2), 0.034992 (T=3) and 0.0451744 (T=5) exactly.
# WITHOUT --resume the sampler CLEARS the run directory and starts over.
set -u
cd "$(dirname "$(readlink -f "$0")")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# T  target_moves  run dir prefix              seeds
PAIRS=(
	"2   790000 pten1d5r_study/equil_ref/T2 100 101"
	"3   280000 pten1d5r_study/equil/T3     100 101"
	"5   210000 pten1d5r_study/equil/T5     100 101"
	"2.5 310000 pten1d5r_study/warm/T2.5    300"
)

echo "== host"; hostname; date
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

# Two concurrent 324-aa chains need about 8.7 GB allocated plus CUDA context.
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mib:-0}" -lt 11000 ]; then
	echo "FATAL: only ${free_mib} MiB free on the GPU; a pair of PTEN chains needs ~11000."
	nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
	exit 1
fi

chain_dir() {  # prefix seed -> run dir
	case $2 in
		100|200|300) echo "$1/chain0";;
		*)           echo "$1/chain1";;
	esac
}

# Fail now, not after the first pair. A missing resume set is FATAL for the three astaroth
# temperatures and merely a skip for T=2.5, which may still be sitting on urano.
fatal=0
runnable=()
for spec in "${PAIRS[@]}"; do
	set -- $spec
	T=$1; MOVES=$2; PREFIX=$3; shift 3
	ok=1
	for seed in "$@"; do
		OUT=$(chain_dir "$PREFIX" "$seed")
		for f in data.dat pars.txt generator.npy log.pt start.txt; do
			if [ ! -f "$OUT/$f" ]; then
				echo "MISSING: $OUT/$f -- cannot --resume"
				ok=0
			fi
		done
		n=$(ls "$OUT"/eprot/*.pt 2>/dev/null | wc -l)
		[ "$n" -ge 4 ] || { echo "MISSING: $OUT/eprot has $n .pt files, --resume needs >= 4"; ok=0; }
		at=$(tail -1 "$OUT/data.dat" 2>/dev/null | awk '{print $1+0}')
		echo "T=$T $OUT at ${at:-?} moves -> $MOVES"
		[ "${at:-0}" -lt "$MOVES" ] || echo "  (already at target; it will be skipped)"
	done
	if [ "$ok" -eq 1 ]; then
		runnable+=("$spec")
	elif [ "$T" = "2.5" ]; then
		echo "  T=2.5 SKIPPED: its run directory is still on urano. Finish it there, or"
		echo "  rsync pten1d5r_study/warm/T2.5/chain0 here and re-run this script."
	else
		fatal=1
	fi
done
[ "$fatal" -eq 0 ] || { echo "FATAL: an astaroth chain is not restartable; nothing started."; exit 1; }

for spec in "${runnable[@]}"; do
	set -- $spec
	T=$1; MOVES=$2; PREFIX=$3; shift 3
	pids=()
	for seed in "$@"; do
		OUT=$(chain_dir "$PREFIX" "$seed")
		at=$(tail -1 "$OUT/data.dat" 2>/dev/null | awk '{print $1+0}')
		if [ "${at:-0}" -ge "$MOVES" ]; then
			echo "== $OUT already at $at moves -- skipping"
			continue
		fi
		echo "===== T=$T $OUT  $at -> $MOVES  seed $seed  $(date)"
		python -u run_nsites.py --ref pten_1d5r --T "$T" --moves "$MOVES" --seed "$seed" \
			--grad-mode one_pass --log-step 500 --print-step 5000 \
			--out "$OUT" --resume > "T${T}_$(basename "$OUT")_match16k.log" 2>&1 &
		pids+=($!)
	done
	rc=0
	for pid in "${pids[@]:-}"; do wait "$pid" || rc=1; done
	echo "===== T=$T done rc=$rc  $(date)"
	for seed in "$@"; do
		OUT=$(chain_dir "$PREFIX" "$seed")
		echo "--- $OUT/run_info.txt"; tail -14 "$OUT/run_info.txt"
		gzip -kf "$OUT/data.dat"
	done
	[ "$rc" -eq 0 ] || { echo "stopping: T=$T failed, not starting the next"; exit 1; }
done

echo; echo "matched-16000 top-up finished $(date)"
echo "Then, in the cloud clone or here:  bash compare_ladder.sh pten_natural"
echo "and check matched16k/'s log: every window start must now be past its t_eq"
echo "(440000 at T=2, 100000 at T=3 and T=5, 80000 at T=2.5)."
