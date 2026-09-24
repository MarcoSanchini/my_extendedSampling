#!/bin/bash

set -u

# ============================================================
# Refinement of sweep_temperature.sh's 12-point log sweep (T=1e-9..1e2),
# targeting the 0.1-1 gap specifically. See DEVLOG.txt, 2026-09-24 entry.
#
# The 12-point sweep found a sharp transition: T=1e-9 through T=1e-1 (8
# decades) are ALL completely frozen (zero acceptance, mean site entropy
# 0.0000 exactly at every one of the 56 sites, q-distribution mean=1.0000/
# std=0.0000), while T=1e0 already shows real activity (mean site entropy
# 1.7929, 59.9% of log(20); q mean=0.2634). The entire transition -- from
# "nothing moves" to "60% of max disorder" -- is therefore compressed into
# the single decade between T=0.1 and T=1.0, which the original 12-point
# sweep (one point per decade) cannot resolve any further.
#
# TS below: 4 points evenly spaced in LOG space strictly between 0.1 and
# 1.0 (10^(-1+i/5) for i=1..4), filling exactly one MAX_RUNNING=4 wave:
#   0.1585, 0.2512, 0.3981, 0.6310
# Endpoints T=0.1 (frozen) and T=1.0 (active) are NOT re-run here --
# already have both from the 12-point sweep, reused directly in any
# combined analysis/plot.
#
# Same fixed dt=8000, moves=50000, init_muts=0, log_step=100, seed=42 as
# the original sweep -- one fixed proposal mechanism across every T,
# consistent with both the original sweep and the paper's own approach.
# ============================================================

TS=(0.1585 0.2512 0.3981 0.6310)

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

DT=8000.0
MOVES=50000
INIT_MUTS=0
LOG_STEP=100
SEED=42

mkdir -p sweep_temperature_refine_inputs
mkdir -p logs_temperature_refine
mkdir -p tests_rate_sweep_temperature_refine

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

    PARS_FILE="sweep_temperature_refine_inputs/pars_T${T}.txt"

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

    RESULTS_DIR="tests_rate_sweep_temperature_refine/T${T}"
    mkdir -p "$RESULTS_DIR"


    RUN_SETTINGS="sweep_temperature_refine_inputs/settings_T${T}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        -e "s/^log_step[[:space:]].*/log_step    ${LOG_STEP}            # checkpoint every N moves, for site-entropy\/q-distribution analysis/" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    echo "Launching T=$T"
    echo "  dt:       $DT"
    echo "  moves:    $MOVES"
    echo "  log_step: $LOG_STEP"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_temperature_refine/T${T}.log"
    echo "  GPU free: ${FREE} MB"

    nohup python main_rate.py \
        --pars-file "sweep_temperature_refine_inputs/pars_T${T}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_temperature_refine/T${T}.log" 2>&1 &

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
echo "All refinement sweep simulations completed."
echo "Results in tests_rate_sweep_temperature_refine/T<T>/sim0/{data.dat,eprot/}"
echo
echo "Per-T ensemble analysis (site entropy + q-distribution):"
echo "  for T in ${TS[@]}; do"
echo "    echo \"--- T=\$T ---\""
echo "    python analyze_ensemble.py tests_rate_sweep_temperature_refine/T\$T/sim0"
echo "  done"
echo "============================================================"
