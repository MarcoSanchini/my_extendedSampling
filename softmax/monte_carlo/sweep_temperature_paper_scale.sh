#!/bin/bash

set -u

# ============================================================
# "Proper" temperature sweep over the active/transition zone this project
# has already mapped empirically (T=0.2512 through 100 across the
# original 12-point sweep and the refine/refine-long follow-ups, see
# DEVLOG.txt 2026-09-24/25 entries), now at a move count matching Zambon
# et al 2024's own stated minimum: "each simulation lasted for at least
# ~3x10^5 steps." moves=300000 below matches that directly.
#
# TS deliberately does NOT reuse the paper's own T_s values (8e-4 to
# 1.71e-2) -- their effective energy is a normalized [0,1] contact-map
# difference from ESMFold-predicted structure, a completely different
# quantity from this project's U_am (log-squared attention-map distance);
# there is no shared unit to translate one T range into the other (see
# DEVLOG.txt 2026-09-23 "GAP" entry). TS instead covers the transition
# zone THIS system was already found to have, log-spaced by 0.2 decades
# continuing the same convention as sweep_temperature_refine.sh:
#   0.2512 (barely active, mostly frozen)
#   0.3981 (steepest part of the crossover, per the refine-long results)
#   0.631  (clearly active, not yet disordered)
#   1.0    (~60% of max entropy per the original 12-point sweep)
#   1.585, 2.512 (extending one/two more 0.2-decade steps past 1.0, into
#                 territory not covered by any prior sweep at this move
#                 count, to see how far the paper-scale budget shifts the
#                 already-mapped 1.0->10.0 trend)
# Also NOT re-sweeping the frozen regime (T<0.25): already fully
# characterized at 50000 moves (zero entropy, q=1.0 exactly across 8
# decades) -- nothing changes there with more moves at genuine
# equilibrium-zero, so it isn't worth the GPU time.
#
# dt=8000, init_muts=0, log_step=100, seed=42 unchanged from every sweep
# in this family -- one fixed proposal mechanism, no per-run retuning.
#
# First sweep script to source gpu_scheduler.sh (see DEVLOG.txt 2026-09-25
# "GPU scheduler: memory-gated instead of a fixed headcount" entry)
# instead of carrying its own copy of the launch-scheduling logic.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gpu_scheduler.sh"

TS=(0.2512 0.3981 0.631 1.0 1.585 2.512)

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

DT=8000.0
MOVES=300000
INIT_MUTS=0
LOG_STEP=100
SEED=42

mkdir -p sweep_temperature_paper_scale_inputs
mkdir -p logs_temperature_paper_scale
mkdir -p tests_rate_sweep_temperature_paper_scale


# ============================================================
# Create parameter files (T, dt, moves, init_muts overridden; everything
# else comes through unchanged from BASE_PARS)
# ============================================================

for T in "${TS[@]}"; do

    PARS_FILE="sweep_temperature_paper_scale_inputs/pars_T${T}.txt"

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
# Launch simulations
# ============================================================

for T in "${TS[@]}"; do

    echo
    echo "============================================================"
    echo "Preparing T=$T"
    echo "============================================================"

    wait_for_gpu_slot

    # --------------------------------------------------------
    # Dedicated results/eprot directories (own raw_path -> own
    # counter.txt, so simultaneous runs never race on which sim<N> to
    # create)
    # --------------------------------------------------------

    RESULTS_DIR="tests_rate_sweep_temperature_paper_scale/T${T}"
    mkdir -p "$RESULTS_DIR"


    RUN_SETTINGS="sweep_temperature_paper_scale_inputs/settings_T${T}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        -e "s/^log_step[[:space:]].*/log_step    ${LOG_STEP}            # checkpoint every N moves, for site-entropy\/q-distribution analysis/" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    echo "Launching T=$T"
    echo "  dt:       $DT"
    echo "  moves:    $MOVES"
    echo "  log_step: $LOG_STEP"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_temperature_paper_scale/T${T}.log"

    nohup python main_rate.py \
        --pars-file "sweep_temperature_paper_scale_inputs/pars_T${T}.txt" \
        --settings-file "$RUN_SETTINGS" \
        > "logs_temperature_paper_scale/T${T}.log" 2>&1 &

    PID=$!
    running_pids+=("$PID")

    echo "  PID: $PID"
    echo "  Running now: ${#running_pids[@]} / ${MAX_RUNNING} (soft cap)"

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
echo "All paper-scale temperature sweep simulations completed."
echo "Results in tests_rate_sweep_temperature_paper_scale/T<T>/sim0/{data.dat,eprot/}"
echo
echo "Per-T ensemble analysis (site entropy + q-distribution):"
echo "  for T in ${TS[@]}; do"
echo "    echo \"--- T=\$T ---\""
echo "    python analyze_ensemble.py tests_rate_sweep_temperature_paper_scale/T\$T/sim0"
echo "  done"
echo
echo "Plotting suite (run from softmax/monte_carlo):"
echo "  see smoke_test_plots.sh for the full per-T + cross-T command"
echo "  sequence -- point BASE_DIR/TS at tests_rate_sweep_temperature_paper_scale"
echo "  and (${TS[@]}) respectively."
echo "============================================================"
