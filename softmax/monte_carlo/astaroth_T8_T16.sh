#!/usr/bin/env bash
# Give T=8 and T=16 real margin past t_eq, on astaroth, two chains at a time.
#
#   nohup bash softmax/monte_carlo/astaroth_T8_T16.sh > astaroth_T8_T16.log 2>&1 &
#
# These are the ladder's two thinnest temperatures. With --burn-in-frac 0.8 the
# averaging window starts at 0.8*length, so against the measured t_eq:
#
#   now   T8  150000 moves, t_eq 120000 -> window starts at 120000, margin ZERO
#         T16  90000 moves, t_eq  60000 -> window starts at  72000, margin 12000
#   after T8  300000                    -> window starts at 240000, margin 120000, kept 60000
#         T16 200000                    -> window starts at 160000, margin 100000, kept 40000
#
# 300000 and 200000 and not more: those targets put T8 and T16 on the same
# footing as T1 and T1.5 (margin ~100000, kept window ~60000). Going further is
# cheap but pointless -- a sixfold change of post-t_eq window moved the
# correlations by 0.013 at T=4.
#
# At 324 aa astaroth runs two concurrent chains at 0.1516 s/move each (1.47x
# the throughput of one at 0.1118), so each pair runs together and the pairs run
# in sequence: about 6.3 h for T8, then 4.6 h for T16.
#
# --resume continues a run to a larger --moves. run_nsites.py asserts every
# parameter against the run's own pars.txt, so the command must match the
# original except for --moves; dt is left to the dt/sqrt(T) rule, which
# reproduces 0.0571417 at T=8 and 0.0808105 at T=16 exactly. WITHOUT --resume
# the sampler CLEARS the run directory and starts over.
set -u
cd "$(dirname "$(readlink -f "$0")")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# T  target_moves  run dir prefix        seeds
PAIRS=(
	"8  300000 pten1d5r_study/equil_ref/T8  200 201"
	"16 200000 pten1d5r_study/equil_ref/T16 200 201"
)

echo "== host"; hostname; date
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

# Two concurrent 324-aa chains need about 8.7 GB allocated plus CUDA context.
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mib:-0}" -lt 11000 ]; then
	echo "FATAL: only ${free_mib} MiB free on the GPU; a pair of PTEN chains needs ~11000."
	echo "       astaroth's GPU was full on 2026-11-20 -- check what is running first:"
	nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
	exit 1
fi

# Fail now, not after the first pair: every run dir must be restartable.
fatal=0
for spec in "${PAIRS[@]}"; do
	set -- $spec
	T=$1; MOVES=$2; PREFIX=$3; shift 3
	for seed in "$@"; do
		case $seed in 200) OUT="$PREFIX/chain0";; *) OUT="$PREFIX/chain1";; esac
		for f in data.dat pars.txt generator.npy log.pt start.txt; do
			[ -f "$OUT/$f" ] || { echo "FATAL: $OUT/$f missing -- cannot --resume"; fatal=1; }
		done
		n=$(ls "$OUT"/eprot/*.pt 2>/dev/null | wc -l)
		[ "$n" -ge 4 ] || { echo "FATAL: $OUT/eprot has $n .pt files, --resume needs >= 4"; fatal=1; }
		at=$(tail -1 "$OUT/data.dat" 2>/dev/null | awk '{print $1+0}')
		echo "T=$T $OUT at ${at:-?} moves -> $MOVES"
		[ "${at:-0}" -lt "$MOVES" ] || echo "  (already at target; it will be skipped)"
	done
done
[ "$fatal" -eq 0 ] || exit 1

for spec in "${PAIRS[@]}"; do
	set -- $spec
	T=$1; MOVES=$2; PREFIX=$3; shift 3
	pids=()
	for seed in "$@"; do
		case $seed in 200) OUT="$PREFIX/chain0";; *) OUT="$PREFIX/chain1";; esac
		at=$(tail -1 "$OUT/data.dat" 2>/dev/null | awk '{print $1+0}')
		if [ "${at:-0}" -ge "$MOVES" ]; then
			echo "== $OUT already at $at moves -- skipping"
			continue
		fi
		echo "===== T=$T $OUT  $at -> $MOVES  seed $seed  $(date)"
		python -u run_nsites.py --ref pten_1d5r --T "$T" --moves "$MOVES" --seed "$seed" \
			--grad-mode one_pass --log-step 500 --print-step 5000 \
			--out "$OUT" --resume > "$(basename "$PREFIX")_$(basename "$OUT")_resume.log" 2>&1 &
		pids+=($!)
	done
	# The pair runs together; the next pair waits, because a third concurrent
	# chain costs throughput rather than gaining it.
	rc=0
	for pid in "${pids[@]:-}"; do wait "$pid" || rc=1; done
	echo "===== T=$T pair done rc=$rc  $(date)"
	for seed in "$@"; do
		case $seed in 200) OUT="$PREFIX/chain0";; *) OUT="$PREFIX/chain1";; esac
		echo "--- $OUT/run_info.txt"; tail -14 "$OUT/run_info.txt"
		gzip -kf "$OUT/data.dat"
	done
	[ "$rc" -eq 0 ] || { echo "stopping: the T=$T pair failed, not starting the next"; exit 1; }
done

echo; echo "T8 and T16 extended $(date)"
echo "Re-run the comparison afterwards:  bash compare_ladder.sh pten_natural --cold T1 T2"
