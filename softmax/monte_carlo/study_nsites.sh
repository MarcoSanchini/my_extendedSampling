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
# RESUME=1 continues existing chains to a larger MOVES_EQ / MOVES_SCAN.
# DRY_RUN=1 prints the commands without launching them.
#
# Examples:
#   nohup bash study_nsites.sh > study_equil.log 2>&1 &
#   RESUME=1 MOVES_EQ=40000 nohup bash study_nsites.sh > study_equil_more.log 2>&1 &
#   PHASE=scan nohup bash study_nsites.sh > study_scan.log 2>&1 &
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
    echo "python -u ${SCRIPT_DIR}/run_nsites.py $* ${RESUME_FLAG}"
    if [ "${DRY_RUN:-0}" = "1" ]; then return; fi
    wait_for_gpu_slot
    nohup python -u "${SCRIPT_DIR}/run_nsites.py" "$@" --log-step "$LOG_STEP" --print-step "$LOG_STEP" \
        --dt-factor "$DT_FACTOR" $RESUME_FLAG > "${RUN_DIR}/logs/${name}.log" 2>&1 &
    running_pids+=("$!")
    echo "  PID $!  log ${RUN_DIR}/logs/${name}.log"
}

if [ "$PHASE" = "equil" ]; then
    for T in "${TS[@]}"; do
        for ((c = 0; c < NCHAINS; c++)); do
            START="reference"
            [ $((c % 2)) -eq 1 ] && START="random"
            launch "equil_T${T}_chain${c}" --T "$T" --n-sites "$N_EQ" --moves "$MOVES_EQ" \
                --seed $((100 + c)) --start "$START" --out "${RUN_DIR}/equil/T${T}/chain${c}"
        done
    done
elif [ "$PHASE" = "scan" ]; then
    # Chains outer, N inner, so a partial scan already covers every N.
    for c in "${SCAN_CHAINS[@]}"; do
        for T in "${TS[@]}"; do
            for N in "${NS[@]}"; do
                launch "scan_T${T}_N${N}_chain${c}" --T "$T" --n-sites "$N" --moves "$MOVES_SCAN" \
                    --seed $((200 + c)) --start "${RUN_DIR}/equil/T${T}/chain${c}" \
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
echo "  python ${SCRIPT_DIR}/analyze_nsites.py ${PHASE} ${RUN_DIR}"
