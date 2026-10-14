#!/bin/bash

set -u

# ============================================================
# Temperature sweep on zero_polymer (566 aa, HA0), the counterpart of
# sweep_temperature.sh for protein G. Same sampler, moves, log_step, seed,
# output layout and GPU scheduler. Three deliberate differences:
#
# 1. REFERENCE. main_rate.py takes --ref (code/tests/references.py). It used
#    to hardcode protein G, so a zero_polymer sweep through it would have
#    silently run protein G. --ref is passed explicitly below, and every log
#    header names the reference.
#
# 2. dt IS RESCALED WITH sqrt(T). The informed proposal depends on
#    alpha = dt*grad_gap/(2*sqrt(M*T)), i.e. on dt/sqrt(T) (DEVLOG 2026-10-03
#    correction). zero_polymer's gradients are ~18x protein G's, so its
#    working point is dt~63 at T=2, and protein G's dt=8000 is ~127x larger.
#    DT(T) = DT_AT_T2 * (T/2)^DT_EXP with DT_EXP=0.5 holds alpha, and so the
#    proposal's informedness, fixed across the sweep. DT_EXP=0 gives back
#    protein G's fixed-dt design.
#
# 3. T GRID. Protein G's grid (1e-9..1e2) lies below zero_polymer's active
#    range: T=2 is already frozen here (DEVLOG 2026-10-02), and the 2-site
#    Boltzmann window is T~32-322, best ~107 (DEVLOG 2026-10-03). Scaling
#    protein G's active range (T~0.25-10) by the ~50x working-T ratio gives
#    ~12-500, so a half-decade grid from T=1 to T=1000 brackets it.
#
# Not checked for this sequence: whether 50000 moves equilibrates a 566-site
# chain (protein G's budget was set on its own 56-site chain), and whether
# dt=63 still gives p_match~0.15 at high drift (measured at 0.9% drift so
# far). Override per run with env vars, e.g.
#    MOVES=300000 bash sweep_temperature_zero_polymer.sh
#    DT_EXP=0 bash sweep_temperature_zero_polymer.sh      # protein G's fixed dt
#    DRY_RUN=1 bash sweep_temperature_zero_polymer.sh     # write inputs only
#    LOG_STEP=3000 PRINT_STEP=1000 MOVES=3000000 bash sweep_temperature_zero_polymer.sh
#                                                         # long runs: fewer checkpoints and status lines
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base inputs (rate_inputs/) and every output live in RUN_DIR.
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/zero_polymer_results}"
CONSTANTS="${SCRIPT_DIR}/../CONSTANTS.txt"

# Passed to main_rate.py as --ref. A literal, not read from REF, so a REF=...
# exported in the shell cannot redirect this sweep.
REF_NAME="zero_polymer"

# Full grid by default. TS_OVERRIDE="1e-1 31.6 ..." runs only the listed T
# values, e.g. to re-run failed points without re-running completed ones.
TS=(${TS_OVERRIDE:-1e-1 1e0 1e1 31.6 56.2 1e2 177 316 562 1e3 1e4})

DT_AT_T2="${DT_AT_T2:-63.0}"   # dt at T=2 where mean p_match~0.15 on zero_polymer
DT_EXP="${DT_EXP:-0.5}"        # 0.5: dt/sqrt(T) held fixed. 0: fixed dt
MOVES="${MOVES:-50000}"
INIT_MUTS=0
LOG_STEP="${LOG_STEP:-100}"
PRINT_STEP="${PRINT_STEP:-1}"
SEED=42

# Memory gate for the shared scheduler. One zero_polymer job used 4.6-8.7 GB
# in the 2026-10-13 test, and four at once filled the 23.5 GB GPU: 9 of 11 runs
# died with CUDA OOM on move 1 (DEVLOG 2026-10-13). Two at a time leaves
# headroom. Raise MAX_RUNNING only after measuring a larger job.
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
MAX_RUNNING="${MAX_RUNNING:-2}"

source "${SCRIPT_DIR}/gpu_scheduler.sh"

# DT(T) = DT_AT_T2 * (T/2)^DT_EXP. awk, because bash has no floats.
dt_for_T() {
    awk -v t="$1" -v c="$DT_AT_T2" -v e="$DT_EXP" \
        'BEGIN { printf "%.4f", c * (t / 2.0) ^ e }'
}

cd "$RUN_DIR" || exit 1

if [ ! -f rate_inputs/pars.txt ] || [ ! -f rate_inputs/settings.txt ]; then
    echo "rate_inputs/pars.txt and rate_inputs/settings.txt must exist in ${RUN_DIR}"
    exit 1
fi

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

mkdir -p sweep_temperature_zero_polymer_inputs
mkdir -p logs_temperature_zero_polymer
mkdir -p tests_rate_sweep_temperature_zero_polymer


# ============================================================
# Create parameter files (T, dt, moves, init_muts overridden; everything
# else comes through unchanged from BASE_PARS)
# ============================================================

for T in "${TS[@]}"; do

    DT=$(dt_for_T "$T")
    PARS_FILE="sweep_temperature_zero_polymer_inputs/pars_T${T}.txt"

    # Whitespace-anchored patterns, as in sweep_temperature.sh, so ^T does
    # not also clobber T_sftm.
    sed \
        -e "s/^T[[:space:]].*/T         ${T}            # sampled temperature/" \
        -e "s/^dt[[:space:]].*/dt        ${DT}            # single-site step size (rescaled with sqrt(T))/" \
        -e "s/^moves[[:space:]].*/moves     ${MOVES}        # number of moves/" \
        -e "s/^init_muts[[:space:]].*/init_muts ${INIT_MUTS}            # initial random mutations/" \
        -e "s/^seed[[:space:]].*/seed      ${SEED}            # generator seed/" \
        "$BASE_PARS" > "$PARS_FILE"

    echo "T=${T}, dt=${DT}, moves=${MOVES}, init_muts=${INIT_MUTS}, seed=${SEED}, ref=${REF_NAME}"

done

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: parameter files written to ${RUN_DIR}/sweep_temperature_zero_polymer_inputs, nothing launched."
    exit 0
fi


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

    RESULTS_DIR="tests_rate_sweep_temperature_zero_polymer/T${T}"
    mkdir -p "$RESULTS_DIR"


    RUN_SETTINGS="sweep_temperature_zero_polymer_inputs/settings_T${T}.txt"

    sed \
        -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
        -e "s/^log_step[[:space:]].*/log_step    ${LOG_STEP}            # checkpoint every N moves, for site-entropy\/q-distribution analysis/" \
        -e "s/^print_step[[:space:]].*/print_step  ${PRINT_STEP}            # number of moves between status printouts/" \
        "$BASE_SETTINGS" > "$RUN_SETTINGS"


    echo "Launching T=$T"
    echo "  dt:       $(dt_for_T "$T")"
    echo "  moves:    $MOVES"
    echo "  ref:      $REF_NAME"
    echo "  log_step: $LOG_STEP"
    echo "  results:  $RESULTS_DIR"
    echo "  log:      logs_temperature_zero_polymer/T${T}.log"

    nohup python -u "${SCRIPT_DIR}/main_rate.py" \
        --ref "$REF_NAME" \
        --pars-file "sweep_temperature_zero_polymer_inputs/pars_T${T}.txt" \
        --settings-file "$RUN_SETTINGS" \
        --cs-file "$CONSTANTS" \
        > "logs_temperature_zero_polymer/T${T}.log" 2>&1 &

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
echo "zero_polymer temperature sweep completed."
echo "Results in ${RUN_DIR}/tests_rate_sweep_temperature_zero_polymer/T<T>/sim0/{data.dat,eprot/}"
echo
echo "Per-T ensemble analysis. --ref-seq is REQUIRED: analyze_ensemble.py"
echo "defaults to protein G's sequence."
echo "  ZP=\$(python -c \"import sys; sys.path.insert(0, '${SCRIPT_DIR}/../code/tests'); from references import ZERO_POLYMER as Z; print(Z)\")"
echo "  for T in ${TS[@]}; do"
echo "    echo \"--- T=\$T ---\""
echo "    python ${SCRIPT_DIR}/analyze_ensemble.py ${RUN_DIR}/tests_rate_sweep_temperature_zero_polymer/T\$T/sim0 --ref-seq \"\$ZP\""
echo "  done"
echo "============================================================"
