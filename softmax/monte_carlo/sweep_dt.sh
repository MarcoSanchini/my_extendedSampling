#!/bin/bash

set -u

# ============================================================
# dt sweep for classes/rate_sampler.py:ExtendedProteinRateSampler.
#
# Unlike the old ep_sampler.py sweep (fixed tau=dt*isteps), this sampler
# has no isteps/trajectory -- dt is a single, standalone knob controlling
# how sharply the ONE proposed site's substitution is chosen each move
# (see DEVLOG.txt: T=2 fixed U's equilibration but the mutated-site vs
# reference-matching-site accept-rate gap flattened, consistent with dt
# still being too small for the proposal to be meaningfully gradient-
# informed -- estimated dt~100-150 needed at T=2). These values bracket
# that estimate. T, moves, init_muts, seed are held fixed across the
# sweep (only dt varies) so the runs are directly comparable.
# ============================================================

DTS=(20 50 100 150)

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

T=2.0
MOVES=5000
INIT_MUTS=5

mkdir -p sweep_dt_inputs
mkdir -p logs_dt
mkdir -p tests_rate_sweep_dt

# Maximum number of simultaneous simulations (GPU-limited)
MAX_RUNNING=4

# Minimum free GPU memory required before launching a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


# ============================================================
# Create parameter files (dt, T, moves, init_muts overridden; everything
# else -- M, T_sftm, lambda_am, lambda_S, n_quad, eps, seed -- comes
# through unchanged from BASE_PARS)
# ============================================================

for DT in "${DTS[@]}"; do

    PARS_FILE="sweep_dt_inputs/pars_dt${DT}.txt"

    # NOTE: T's substitution pattern must require a whitespace boundary
    # (not [[:space:]]*, which also matches zero chars) or it clobbers
    # T_sftm too, since ^T[[:space:]]*.* matches "T_sftm ..." as well.
    sed \
        -e "s/^dt[[:space:]].*/dt        ${DT}            # single-site step size/" \
        -e "s/^T[[:space:]].*/T         ${T}            # sampled temperature/" \
        -e "s/^moves[[:space:]].*/moves     ${MOVES}        # number of moves/" \
        -e "s/^init_muts[[:space:]].*/init_muts ${INIT_MUTS}            # initial random mutations/" \
        "$BASE_PARS" > "$PARS_FILE"

    echo "dt=${DT}, T=${T}, moves=${MOVES}, init_muts=${INIT_MUTS}"

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

for DT in "${DTS[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing dt=$DT"
    echo "============================================================"


    # --------------------------------------------------------
    # Wait until:
    #   1. fewer than MAX_RUNNING simulations are active
    #   2. enough GPU memory is free
    # --------------------------------------------------------

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
    # Dedicated results directory (own raw_path -> own counter.txt,
    # so simultaneous runs never race on which sim<N> to create)
    # --------------------------------------------------------

    RESULTS_DIR="tests_rate_sweep_dt/dt${DT}"

    mkdir -p "$RESULTS_DIR"


    # --------------------------------------------------------
    # Dedicated settings file
    # --------------------------------------------------------

    RUN_SETTINGS="sweep_dt_inputs/settings_dt${DT}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    echo "Launching dt=$DT"
    echo "  T:        $T"
    echo "  moves:    $MOVES"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_dt/dt${DT}.log"
    echo "  GPU free: ${FREE} MB"


    # --------------------------------------------------------
    # Launch
    # --------------------------------------------------------

    nohup python main_rate.py \
        --pars-file "sweep_dt_inputs/pars_dt${DT}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_dt/dt${DT}.log" 2>&1 &

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
echo "All dt-sweep simulations completed."
echo "Results in tests_rate_sweep_dt/dt<DT>/sim0/data.dat -- analyze with:"
echo "  for DT in ${DTS[@]}; do"
echo "    echo \"--- dt=\$DT ---\""
echo "    python analyze_run.py tests_rate_sweep_dt/dt\$DT/sim0"
echo "  done"
echo "============================================================"
