#!/bin/bash

set -u

# ============================================================
# Longer-window follow-up to sweep_temperature_refine.sh, targeting the
# 3 points (T=0.2512, 0.3981, 0.6310) that analyze_ensemble.py's burn-in
# check flagged as "still trending" at 50000 moves/50% burn-in -- see
# DEVLOG.txt, 2026-09-24 "refinement sweep results" entry. T=0.1585 is
# NOT re-run here: its burn-in gap (100% relative) looked large only
# because the underlying signal is still tiny in absolute terms
# (0.0005->0.0008), not because it's meaningfully unsettled the way
# 0.3981/0.6310 are (real, still-rising ABSOLUTE gaps there).
#
# moves=150000 (3x the original 50000) -- both because equilibration
# time is expected to grow near a transition (the same "critical slowing
# down" reasoning already used to explain the burn-in gap in the first
# place, not a new assumption) and because MORE total moves means the
# same 50% burn-in cutoff discards a longer absolute stretch while also
# keeping proportionally more post-cutoff data, both working in the same
# direction (fixed cutoff FRACTION, growing window -> shrinking relative
# contribution of any fixed-length initial transient).
#
# Same fixed dt=8000, init_muts=0, log_step=100, seed=42 as every sweep
# in this family -- one fixed proposal mechanism, no per-run retuning.
# log_step unchanged (not scaled with moves): 150000/100=1500 checkpoints
# per run x 3 runs, still small (~1-2KB/checkpoint based on the original
# refine sweep's files) and gitignored locally regardless.
# ============================================================

TS=(0.2512 0.3981 0.6310)

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

DT=8000.0
MOVES=150000
INIT_MUTS=0
LOG_STEP=100
SEED=42

mkdir -p sweep_temperature_refine_long_inputs
mkdir -p logs_temperature_refine_long
mkdir -p tests_rate_sweep_temperature_refine_long

# Maximum number of simultaneous simulations (GPU-limited)
MAX_RUNNING=4

# Minimum free GPU memory required before launching a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


# ============================================================
# Create parameter files (T, dt, moves, init_muts overridden; everything
# else comes through unchanged from BASE_PARS)
# ============================================================

for T in "${TS[@]}"; do

    PARS_FILE="sweep_temperature_refine_long_inputs/pars_T${T}.txt"

    # NOTE: substitution patterns require a whitespace boundary
    # ([[:space:]], not [[:space:]]*) so e.g. ^T doesn't also clobber
    # T_sftm -- see sweep_dt.sh's header for the bug this avoids.
    sed \
        -e "s/^T[[:space:]].*/T         ${T}            # sampled temperature/" \
        -e "s/^dt[[:space:]].*/dt        ${DT}            # single-site step size/" \
        -e "s/^moves[[:space:]].*/moves     ${MOVES}        # number of moves/" \
        -e "s/^init_muts[[:space:]].*/init_muts ${INIT_MUTS}            # initial random mutations/" \
        -e "s/^seed[[:space:]].*/seed      ${SEED}            # generator seed/" \
        "$BASE_PARS" > "$PARS_FILE"

    echo "T=${T}, dt=${DT}, moves=${MOVES}, init_muts=${INIT_MUTS}, seed=${SEED}"

done


# ============================================================
# GPU scheduler
# ============================================================

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

for T in "${TS[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing T=$T"
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


    # --------------------------------------------------------
    # Dedicated results/eprot directories (own raw_path -> own
    # counter.txt, so simultaneous runs never race on which sim<N> to
    # create)
    # --------------------------------------------------------

    RESULTS_DIR="tests_rate_sweep_temperature_refine_long/T${T}"
    mkdir -p "$RESULTS_DIR"


    RUN_SETTINGS="sweep_temperature_refine_long_inputs/settings_T${T}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        -e "s/^log_step[[:space:]].*/log_step    ${LOG_STEP}            # checkpoint every N moves, for site-entropy\/q-distribution analysis/" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    echo "Launching T=$T"
    echo "  dt:       $DT"
    echo "  moves:    $MOVES"
    echo "  log_step: $LOG_STEP"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_temperature_refine_long/T${T}.log"
    echo "  GPU free: ${FREE} MB"

    nohup python main_rate.py \
        --pars-file "sweep_temperature_refine_long_inputs/pars_T${T}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_temperature_refine_long/T${T}.log" 2>&1 &

    PID=$!
    running_pids+=("$PID")

    echo "  PID: $PID"
    echo "  Running now: ${#running_pids[@]} / ${MAX_RUNNING}"

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
echo "All refine-long sweep simulations completed."
echo "Results in tests_rate_sweep_temperature_refine_long/T<T>/sim0/{data.dat,eprot/}"
echo
echo "Per-T ensemble analysis (site entropy + q-distribution):"
echo "  for T in ${TS[@]}; do"
echo "    echo \"--- T=\$T ---\""
echo "    python analyze_ensemble.py tests_rate_sweep_temperature_refine_long/T\$T/sim0"
echo "  done"
echo "============================================================"
