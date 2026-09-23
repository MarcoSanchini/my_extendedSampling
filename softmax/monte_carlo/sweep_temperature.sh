#!/bin/bash

set -u

# ============================================================
# Temperature sweep for classes/rate_sampler.py:ExtendedProteinRateSampler,
# aimed at reproducing (with U_am, our faster sampler in place of Zambon
# et al 2024's plain Metropolis-Hastings) the paper's core structural
# findings for protein G: a small set of highly-conserved "K-sites" with
# the rest free to vary, and a unimodal sequence-similarity distribution
# p(q) that sharpens as T decreases. See DEVLOG.txt, 2026-09-23 entries.
#
# T=(0.1,0.2,0.4,0.7,1.2,2.0,3.5,6.0,10.0,17.0), log-spaced, anchored at
# T=2.0 (already characterized extensively: dispersed starts converge to
# U~75-80, Hd~43/56 at T=2, dt=2000-8000). Meant to bracket a cold,
# near-native regime (mean single-mutation cost from the reference is
# ~31 energy units per scan_U_am_landscape.py, so T~0.1-0.4 should
# strongly suppress drift) through a hot, disordered regime (T=2 already
# gives substantial divergence; 10-17 should push well past it).
#
# dt=8000 FIXED across the whole sweep -- characterized at T=2 (mean
# p_match~0.47 there); informedness scales roughly as 1/sqrt(T), so this
# stays informed (more so, in fact) at colder T and only gradually
# loosens at the hottest points, never fully uninformed. Same spirit as
# the paper itself: one fixed proposal mechanism across every T, no
# per-T retuning -- a pragmatic choice for a first pass, not a claim that
# dt=8000 is equally well-tuned at every T in this range.
#
# init_muts=0 (start exactly at the reference, per explicit request).
# moves=50000: comfortably past the ~27000+ moves the slowest chain
# needed to reach its shared equilibrium in the earlier dispersed-start
# test (see DEVLOG.txt, far/seed43 extension).
#
# log_step=100 (not the default 1): _save_log checkpoints the FULL
# discrete state (sequence_<move>.pt, tokens_<move>.pt, logits_<move>.pt,
# am_<move>.pt) every log_step moves via ExtendedProtein.save() -- this is
# the ONLY way to recover many sampled sequences along a run (data.dat
# itself only has move-0 and the trajectory's move/U/Hd_to_ref/site
# columns, not full intermediate sequences). log_step=100 over 50000
# moves gives 500 checkpoints/run, ~100-150MB total across the whole
# sweep -- doesn't lose anything from data.dat, which is flushed in
# batches regardless of log_step.
# ============================================================

TS=(0.1 0.2 0.4 0.7 1.2 2.0 3.5 6.0 10.0 17.0)

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

DT=8000.0
MOVES=50000
INIT_MUTS=0
LOG_STEP=100
SEED=42

mkdir -p sweep_temperature_inputs
mkdir -p logs_temperature
mkdir -p tests_rate_sweep_temperature

# Maximum number of simultaneous simulations (GPU-limited)
MAX_RUNNING=4

# Minimum free GPU memory required before launching a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


# ============================================================
# Create parameter files (T, dt, moves, init_muts overridden; everything
# else -- M, T_sftm, lambda_am, lambda_structure_ce, lambda_S, n_quad,
# eps, seed -- comes through unchanged from BASE_PARS)
# ============================================================

for T in "${TS[@]}"; do

    PARS_FILE="sweep_temperature_inputs/pars_T${T}.txt"

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

    RESULTS_DIR="tests_rate_sweep_temperature/T${T}"
    mkdir -p "$RESULTS_DIR"


    RUN_SETTINGS="sweep_temperature_inputs/settings_T${T}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        -e "s/^log_step[[:space:]].*/log_step    ${LOG_STEP}            # checkpoint every N moves, for site-entropy\/q-distribution analysis/" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    echo "Launching T=$T"
    echo "  dt:       $DT"
    echo "  moves:    $MOVES"
    echo "  log_step: $LOG_STEP"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_temperature/T${T}.log"
    echo "  GPU free: ${FREE} MB"

    nohup python main_rate.py \
        --pars-file "sweep_temperature_inputs/pars_T${T}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_temperature/T${T}.log" 2>&1 &

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
echo "All temperature sweep simulations completed."
echo "Results in tests_rate_sweep_temperature/T<T>/sim0/{data.dat,eprot/}"
echo
echo "Per-T ensemble analysis (site entropy + q-distribution):"
echo "  for T in ${TS[@]}; do"
echo "    echo \"--- T=\$T ---\""
echo "    python analyze_ensemble.py tests_rate_sweep_temperature/T\$T/sim0"
echo "  done"
echo "============================================================"
