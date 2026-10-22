#!/bin/bash

set -u

# ============================================================
# One-pass move on zero_polymer (docs/one_pass_move.tex, tests T6-T8).
# Launches run_nsites.py chains, one at a time by default, for both
# grad_mode=site_soft (the current move) and grad_mode=one_pass. The one-pass
# dt is the site_soft rule times one_pass_dt_scale() (run_nsites.py does it).
#
# PHASE=bench      T6. 300 moves at T=100: site_soft alone, one_pass alone,
#                  then two one_pass jobs together, with nvidia-smi sampled
#                  every second. Summary: python analyze_onepass.py bench $RUN_DIR
# PHASE=transient  T7, first half. For every T and seed in SEEDS: both modes
#                  from the reference, MOVES_TR moves, same seed. Acceptance,
#                  Hd growth and U per move must agree within seed spread.
#                  python analyze_onepass.py transient $RUN_DIR
# PHASE=equil      T7 second half and T8. For every T and chain in EQ_CHAINS:
#                  both modes from the last checkpoint of the n_sites study's
#                  equilibrated chain (NSITES_DIR/equil/T<T>/chain<c>), MOVES_EQ
#                  moves. Equilibrium statistics and seconds per decorrelated
#                  sequence. Needs that study's equil phase finished and its
#                  analysis saying the chains are equilibrated.
#                  python analyze_onepass.py equil $RUN_DIR
#
# RESUME=1 continues existing chains to larger MOVES_*. DRY_RUN=1 prints the
# commands only. Examples:
#   PHASE=bench     nohup bash study_onepass.sh > onepass_bench.log 2>&1 &
#   PHASE=transient nohup bash study_onepass.sh > onepass_transient.log 2>&1 &
#   PHASE=equil     nohup bash study_onepass.sh > onepass_equil.log 2>&1 &
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/onepass_study}"
NSITES_DIR="${NSITES_DIR:-${SCRIPT_DIR}/nsites_study}"
PHASE="${PHASE:-transient}"

TS=(${TS_OVERRIDE:-32 100 316})
MODES=(${MODES_OVERRIDE:-site_soft one_pass})
SEEDS=(${SEEDS:-0 1})
MOVES_TR="${MOVES_TR:-5000}"
EQ_CHAINS=(${EQ_CHAINS:-0 1})
MOVES_EQ="${MOVES_EQ:-5000}"
BENCH_MOVES="${BENCH_MOVES:-300}"
LOG_STEP="${LOG_STEP:-100}"

# One job saturates the GPU (DEVLOG 2026-10-22); timings are only clean alone.
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
MAX_RUNNING="${MAX_RUNNING:-1}"
source "${SCRIPT_DIR}/gpu_scheduler.sh"

# Other GPU jobs (e.g. the n_sites study) would share the card and distort
# s/move, which T6 and T8 measure. EXCLUSIVE=1 (default) waits before every
# launch until no compute process outside this script is on the GPU.
EXCLUSIVE="${EXCLUSIVE:-1}"
wait_for_exclusive_gpu() {
    [ "$EXCLUSIVE" = "1" ] || return 0
    command -v nvidia-smi > /dev/null || return 0
    while true; do
        local others=0
        for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null); do
            local mine=0
            for p in "${running_pids[@]}"; do [ "$p" = "$pid" ] && mine=1; done
            [ "$mine" = "0" ] && others=$((others + 1))
        done
        [ "$others" = "0" ] && return 0
        echo "$(date '+%F %T')  ${others} other GPU process(es) running; waiting ${CHECK_INTERVAL}s (EXCLUSIVE=0 to skip)"
        sleep "$CHECK_INTERVAL"
    done
}

RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

mkdir -p "${RUN_DIR}/logs"

launch() {   # launch <log name> <run_nsites.py args...>
    local name="$1"; shift
    echo "python -u ${SCRIPT_DIR}/run_nsites.py $* ${RESUME_FLAG}"
    if [ "${DRY_RUN:-0}" = "1" ]; then return; fi
    wait_for_exclusive_gpu
    wait_for_gpu_slot
    nohup python -u "${SCRIPT_DIR}/run_nsites.py" "$@" --log-step "$LOG_STEP" --print-step "$LOG_STEP" \
        $RESUME_FLAG > "${RUN_DIR}/logs/${name}.log" 2>&1 &
    running_pids+=("$!")
    echo "  PID $!  log ${RUN_DIR}/logs/${name}.log"
}

wait_all() {
    while [ "${#running_pids[@]}" -gt 0 ]; do
        cleanup_finished
        [ "${#running_pids[@]}" -gt 0 ] && sleep "$CHECK_INTERVAL"
    done
}

if [ "$PHASE" = "bench" ]; then
    B="${RUN_DIR}/bench"
    mkdir -p "$B"
    smi() {   # smi <file>: GPU utilisation and memory every second, in the background
        if command -v nvidia-smi > /dev/null && [ "${DRY_RUN:-0}" != "1" ]; then
            nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$1" &
            SMI_PID=$!
        else
            SMI_PID=""
        fi
    }
    stop_smi() { [ -n "${SMI_PID}" ] && kill "$SMI_PID" 2> /dev/null; }
    for MODE in site_soft one_pass; do
        smi "$B/smi_${MODE}_alone.csv"
        launch "bench_${MODE}_alone" --T 100 --moves "$BENCH_MOVES" --seed 0 --start reference \
            --grad-mode "$MODE" --out "$B/${MODE}_alone"
        wait_all
        stop_smi
    done
    MAX_RUNNING=2
    smi "$B/smi_one_pass_pair.csv"
    for k in 0 1; do
        launch "bench_one_pass_pair${k}" --T 100 --moves "$BENCH_MOVES" --seed "$k" --start reference \
            --grad-mode one_pass --out "$B/one_pass_pair${k}"
    done
    wait_all
    stop_smi
elif [ "$PHASE" = "transient" ]; then
    for s in "${SEEDS[@]}"; do
        for T in "${TS[@]}"; do
            for MODE in "${MODES[@]}"; do
                launch "transient_T${T}_${MODE}_chain${s}" --T "$T" --moves "$MOVES_TR" --seed $((300 + s)) \
                    --start reference --grad-mode "$MODE" --out "${RUN_DIR}/transient/T${T}/${MODE}/chain${s}"
            done
        done
    done
elif [ "$PHASE" = "equil" ]; then
    for c in "${EQ_CHAINS[@]}"; do
        for T in "${TS[@]}"; do
            START="${NSITES_DIR}/equil/T${T}/chain${c}"
            if [ ! -d "$START/eprot" ]; then
                echo "missing equilibrated chain $START (run study_nsites.sh PHASE=equil first)"
                exit 1
            fi
            for MODE in "${MODES[@]}"; do
                launch "equil_T${T}_${MODE}_chain${c}" --T "$T" --moves "$MOVES_EQ" --seed $((400 + c)) \
                    --start "$START" --grad-mode "$MODE" --out "${RUN_DIR}/equil/T${T}/${MODE}/chain${c}"
            done
        done
    done
else
    echo "PHASE must be bench, transient or equil, got ${PHASE}"
    exit 1
fi

[ "${DRY_RUN:-0}" = "1" ] && exit 0

echo "All launched. Waiting..."
wait_all
echo "Phase ${PHASE} finished. Analyse with:"
echo "  python ${SCRIPT_DIR}/analyze_onepass.py ${PHASE} ${RUN_DIR}"
