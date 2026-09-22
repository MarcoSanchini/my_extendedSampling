#!/bin/bash

set -u

# ============================================================
# Dispersed-start equilibration check for run_equilibration.py.
#
# Question: does the U/Hd_to_ref drift seen in every sweep run so far (any
# dt, climbing toward Hd~40+/56 with no clear plateau within 5000 moves)
# reflect genuine T=2 equilibrium behavior (the far-from-reference bulk of
# sequence space is astronomically larger, so its entropy wins even though
# per-mutation costs are real -- see the reference-anchored scan and
# epistasis findings in DEVLOG.txt), or a starting-point artifact (the
# near-reference start getting stuck behind barriers in a distant basin)?
#
# Three starting conditions (reference/moderate/far, see
# run_equilibration.py's docstring), 2 seeds each = 6 runs. If the
# reference-start chains eventually reach the same U/Hd regime as the
# far-start chains (even later), that supports "genuine equilibrium
# behavior". If they stay in a visibly different, non-overlapping regime
# through the whole run, that supports "distant, barrier-separated basin".
#
# dt=2000, T=2.0 fixed across all runs (moderate/informed, already
# characterized -- see test_informedness_vs_dt.py's p_match curve) so
# any difference between conditions isn't confounded by different
# proposal sharpness. moves=15000 (3x a standard sweep run) to give a
# real chance of observing convergence or persistent divergence, not just
# extending the same still-transient regime a bit further.
# ============================================================

SEEDS=(42 43)
START_MODES=(reference moderate far)

DT=2000.0
T=2.0
MOVES=15000

mkdir -p logs_equilibration
mkdir -p tests_rate_equilibration

# Maximum number of simultaneous simulations (GPU-limited)
MAX_RUNNING=4

# Minimum free GPU memory required before launching a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


gpu_free_mb() {
    nvidia-smi \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits |
        head -1
}


cleanup_finished() {
    new_pids=()
    for pid in "${running_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        else
            echo "PID $pid finished."
        fi
    done
    running_pids=("${new_pids[@]}")
}


# ============================================================
# Launch simulations
# ============================================================

for MODE in "${START_MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        echo
        echo "============================================================"
        echo "Preparing start_mode=$MODE seed=$SEED"
        echo "============================================================"

        while true; do

            cleanup_finished
            FREE=$(gpu_free_mb)

            echo "GPU free memory: ${FREE} MB"
            echo "Running simulations: ${#running_pids[@]} / ${MAX_RUNNING}"

            if [ "${#running_pids[@]}" -lt "$MAX_RUNNING" ] && \
               [ "$FREE" -ge "$MIN_FREE_MB" ]; then
                break
            fi

            echo "Waiting ${CHECK_INTERVAL}s..."
            sleep "$CHECK_INTERVAL"

        done

        RESULTS_DIR="tests_rate_equilibration/${MODE}_seed${SEED}"
        mkdir -p "$RESULTS_DIR"

        echo "Launching start_mode=$MODE seed=$SEED"
        echo "  dt:       $DT"
        echo "  T:        $T"
        echo "  moves:    $MOVES"
        echo "  results:  $RESULTS_DIR"
        echo "  log:      logs_equilibration/${MODE}_seed${SEED}.log"
        echo "  GPU free: ${FREE} MB"

        nohup python run_equilibration.py \
            --start-mode "$MODE" \
            --seed "$SEED" \
            --moves "$MOVES" \
            --dt "$DT" \
            --T "$T" \
            --results-dir "$RESULTS_DIR" \
            > "logs_equilibration/${MODE}_seed${SEED}.log" 2>&1 &

        PID=$!
        running_pids+=("$PID")

        echo "  PID: $PID"
        echo "  Running now: ${#running_pids[@]} / ${MAX_RUNNING}"

    done
done


# ============================================================
# Wait for all simulations
# ============================================================

echo
echo "All simulations have been launched."
echo "Waiting for remaining simulations..."

while [ "${#running_pids[@]}" -gt 0 ]; do
    cleanup_finished
    if [ "${#running_pids[@]}" -gt 0 ]; then
        sleep "$CHECK_INTERVAL"
    fi
done


echo
echo "============================================================"
echo "All equilibration runs completed."
echo "Results in tests_rate_equilibration/<mode>_seed<seed>/data.dat"
echo
echo "Per-run detail (analyze_run.py works directly on these logs):"
echo "  for MODE in ${START_MODES[@]}; do for SEED in ${SEEDS[@]}; do"
echo "    echo \"--- \$MODE seed=\$SEED ---\""
echo "    python analyze_run.py tests_rate_equilibration/\${MODE}_seed\${SEED}"
echo "  done; done"
echo
echo "Cross-condition comparison:"
echo "  python compare_equilibration.py"
echo "============================================================"
