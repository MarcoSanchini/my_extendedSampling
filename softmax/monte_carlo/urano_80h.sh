#!/usr/bin/env bash
# Two PTEN chains on urano, one after the other, about 79 h in total.
# Rationale and the ladder audit behind the choice of temperatures:
# /mnt/project-files/urano/urano_80h_plan.md
#
#   nohup bash softmax/monte_carlo/urano_80h.sh > urano_80h.log 2>&1 &
#   RESUME=1 nohup bash softmax/monte_carlo/urano_80h.sh > urano_80h_resume.log 2>&1 &
#
# urano has one GPU with room for exactly one 324-aa chain (4769 MB of 7.91 GB,
# with ~1.3 GB held by another process), so the chains are strictly sequential.
# Both are HOT-STARTED from an equilibrated neighbouring temperature: from the
# reference, t_eq at T=2 is 440000 moves, so a 140000-move chain at T=2.5 would
# never equilibrate. They therefore go under warm/ and NOT equil_ref/, which
# means "started from the reference" and is what compare_ladder.sh and the
# R-hat machinery key on. t_eq for a hot-started chain must be read off its own
# U trace; the ladder's t_eq values are relaxation times from the reference.
set -u
cd "$(dirname "$(readlink -f "$0")")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME=${RESUME:-0}

# T  moves   seed  start run dir (its highest eprot/sequence_*.pt is the start)
CHAINS=(
	"2.5 140000 300 pten1d5r_study/equil_ref/T2/chain0"
	"7   120000 301 pten1d5r_study/equil/T6/chain0"
)

echo "== host"; hostname; date
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Fail before burning 40 h: every start checkpoint must already be on this disk.
fatal=0
for spec in "${CHAINS[@]}"; do
	set -- $spec
	n=$(ls "$4"/eprot/sequence_*.pt 2>/dev/null | wc -l)
	if [ "$n" -eq 0 ]; then
		echo "FATAL: no eprot/sequence_*.pt under $4 -- rsync it from astaroth first (plan, step 1)"
		fatal=1
	else
		echo "start for T=$1: $(ls "$4"/eprot/sequence_*.pt | sed 's/.*sequence_//;s/\.pt//' | sort -n | tail -1) ($n checkpoint(s) present)"
	fi
done
[ "$fatal" -eq 0 ] || exit 1

for spec in "${CHAINS[@]}"; do
	set -- $spec
	T=$1; MOVES=$2; SEED=$3; START=$4
	OUT="pten1d5r_study/warm/T$T/chain0"

	extra=""
	if [ -f "$OUT/data.dat" ]; then
		done_moves=$(tail -1 "$OUT/data.dat" | awk '{print $1+0}')
		if [ "${done_moves:-0}" -ge "$MOVES" ]; then
			echo "== T=$T already at $done_moves moves of $MOVES -- skipping"
			continue
		fi
		if [ "$RESUME" = "1" ]; then
			echo "== T=$T resuming from $done_moves moves"
			extra="--resume"
		else
			echo "FATAL: $OUT exists at $done_moves moves. Without --resume the sampler CLEARS it"
			echo "       and starts over. Relaunch with RESUME=1 to continue it."
			exit 1
		fi
	fi

	echo; echo "===== T=$T  $MOVES moves  seed $SEED  start $START  $(date)"
	python -u run_nsites.py --ref pten_1d5r --T "$T" --moves "$MOVES" --seed "$SEED" \
		--grad-mode one_pass --start "$START" \
		--log-step 500 --print-step 5000 --out "$OUT" $extra
	rc=$?
	echo "===== T=$T exit $rc  $(date)"
	echo "--- run_info.txt (ref must be pten_1d5r, ref_len 324, T $T, seed $SEED)"
	cat "$OUT/run_info.txt"
	echo "--- start.txt (must name the checkpoint, not 'reference')"
	cat "$OUT/start.txt"
	[ "$rc" -eq 0 ] || { echo "stopping: T=$T failed, not starting the next chain"; exit "$rc"; }
	gzip -kf "$OUT/data.dat"
done

echo; echo "all chains done $(date)"
