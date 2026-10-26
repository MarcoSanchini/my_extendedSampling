#!/bin/bash

set -u

# ============================================================
# Mutations-per-step study on zero_polymer (DEVLOG 2026-10-21). Launches
# run_nsites.py chains through the shared GPU scheduler.
#
# PHASE=equil  NCHAINS chains per temperature at N_EQ sites per move. Chains
#              with an even index start at the reference, odd ones at a random
#              canonical sequence, so the starts are as far apart as possible.
#              Check with: python analyze_nsites.py equil $RUN_DIR
# PHASE=scan   for every T, every N in NS and every chain in SCAN_CHAINS: a
#              chain started from the last checkpoint of equil/T<T>/chain<c>.
#              Run it only after the equil analysis says the chains agree.
#              Then: python analyze_nsites.py scan $RUN_DIR
#
# GRAD_MODE=one_pass uses the one-pass move (2.4x faster per move, same per-move
#              statistics; DEVLOG 2026-10-24). Default site_soft, as in the first runs.
# EQ_DIR       directory under RUN_DIR for the equil chains (default equil); the
#              scan starts from EQ_DIR too.
# CONTINUE_FROM=equil  in PHASE=equil, chain c at T starts from the last
#              checkpoint of RUN_DIR/CONTINUE_FROM/T<T>/chain<c> when that run
#              exists (so it continues where it stopped, also with another
#              GRAD_MODE), and from the reference / a random sequence otherwise.
#
# RESUME=1 continues existing chains to a larger MOVES_EQ / MOVES_SCAN.
# DRY_RUN=1 prints the commands without launching them.
#
# Examples:
#   nohup bash study_nsites.sh > study_equil.log 2>&1 &
#   RESUME=1 MOVES_EQ=40000 nohup bash study_nsites.sh > study_equil_more.log 2>&1 &
#   PHASE=scan nohup bash study_nsites.sh > study_scan.log 2>&1 &
#   GRAD_MODE=one_pass EQ_DIR=equil_op CONTINUE_FROM=equil TS_OVERRIDE="18 32 100" MOVES_EQ=40000 \
#       nohup bash study_nsites.sh > study_equil_op.log 2>&1 &
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/nsites_study}"
PHASE="${PHASE:-equil}"

# Low, inside the 30-300 transition, high (DEVLOG 2026-10-14 acceptance table).
TS=(${TS_OVERRIDE:-32 100 316})

NCHAINS="${NCHAINS:-4}"
N_EQ="${N_EQ:-1}"
MOVES_EQ="${MOVES_EQ:-20000}"

NS=(${NS_OVERRIDE:-1 2 4 8 16 32})
SCAN_CHAINS=(${SCAN_CHAINS:-0 1})
MOVES_SCAN="${MOVES_SCAN:-5000}"

LOG_STEP="${LOG_STEP:-500}"
DT_FACTOR="${DT_FACTOR:-1.0}"
GRAD_MODE="${GRAD_MODE:-site_soft}"
EQ_DIR="${EQ_DIR:-equil}"
CONTINUE_FROM="${CONTINUE_FROM:-}"

# One job saturates the GPU: alone 0.364 s/move, two together 0.79 s/move each
# (2026-10-22), so concurrency is ~8% slower and makes s/move depend on
# neighbours. One at a time by default.
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
MAX_RUNNING="${MAX_RUNNING:-1}"
source "${SCRIPT_DIR}/gpu_scheduler.sh"

RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

mkdir -p "${RUN_DIR}/logs"

launch() {   # launch <log name> <run_nsites.py args...>
    local name="$1"; shift
    echo "python -u ${SCRIPT_DIR}/run_nsites.py $* --grad-mode ${GRAD_MODE} ${RESUME_FLAG}"
    if [ "${DRY_RUN:-0}" = "1" ]; then return; fi
    wait_for_gpu_slot
    nohup python -u "${SCRIPT_DIR}/run_nsites.py" "$@" --log-step "$LOG_STEP" --print-step "$LOG_STEP" \
        --dt-factor "$DT_FACTOR" --grad-mode "$GRAD_MODE" $RESUME_FLAG > "${RUN_DIR}/logs/${name}.log" 2>&1 &
    running_pids+=("$!")
    echo "  PID $!  log ${RUN_DIR}/logs/${name}.log"
}

# CONTINUE_FROM must find a checkpoint for every T it has runs for. A chain
# whose checkpoint is missing would otherwise start over from the reference or
# a random sequence without saying so (it happened on 2026-10-24).
if [ "$PHASE" = "equil" ] && [ -n "$CONTINUE_FROM" ]; then
    for T in "${TS[@]}"; do
        [ -d "${RUN_DIR}/${CONTINUE_FROM}/T${T}" ] || { echo "T=${T}: no ${CONTINUE_FROM}/T${T}, starts fresh"; continue; }
        for ((c = 0; c < NCHAINS; c++)); do
            if ! ls "${RUN_DIR}/${CONTINUE_FROM}/T${T}/chain${c}"/eprot/sequence_*.pt > /dev/null 2>&1; then
                echo "ERROR: CONTINUE_FROM=${CONTINUE_FROM}: no eprot/sequence_*.pt in ${RUN_DIR}/${CONTINUE_FROM}/T${T}/chain${c}. Nothing launched."
                exit 1
            fi
        done
    done
fi

if [ "$PHASE" = "equil" ]; then
    for T in "${TS[@]}"; do
        for ((c = 0; c < NCHAINS; c++)); do
            START="reference"
            [ $((c % 2)) -eq 1 ] && START="random"
            PREV="${RUN_DIR}/${CONTINUE_FROM}/T${T}/chain${c}"
            [ -n "$CONTINUE_FROM" ] && [ -d "${PREV}/eprot" ] && START="$PREV"
            launch "${EQ_DIR}_T${T}_chain${c}" --T "$T" --n-sites "$N_EQ" --moves "$MOVES_EQ" \
                --seed $((100 + c)) --start "$START" --out "${RUN_DIR}/${EQ_DIR}/T${T}/chain${c}"
        done
    done
elif [ "$PHASE" = "scan" ]; then
    # Chains outer, N inner, so a partial scan already covers every N.
    for c in "${SCAN_CHAINS[@]}"; do
        for T in "${TS[@]}"; do
            for N in "${NS[@]}"; do
                launch "scan_T${T}_N${N}_chain${c}" --T "$T" --n-sites "$N" --moves "$MOVES_SCAN" \
                    --seed $((200 + c)) --start "${RUN_DIR}/${EQ_DIR}/T${T}/chain${c}" \
                    --out "${RUN_DIR}/scan/T${T}/N${N}/chain${c}"
            done
        done
    done
else
    echo "PHASE must be equil or scan, got ${PHASE}"
    exit 1
fi

[ "${DRY_RUN:-0}" = "1" ] && exit 0

echo "All launched. Waiting..."
while [ "${#running_pids[@]}" -gt 0 ]; do
    cleanup_finished
    [ "${#running_pids[@]}" -gt 0 ] && sleep "$CHECK_INTERVAL"
done
echo "Phase ${PHASE} finished. Analyse with:"
echo "  python ${SCRIPT_DIR}/analyze_nsites.py ${PHASE} ${RUN_DIR} --equil-dir ${EQ_DIR}"
