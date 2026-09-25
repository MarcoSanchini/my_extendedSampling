#!/bin/bash

# ============================================================
# Shared GPU-memory-gated launch scheduler, meant to be `source`d by
# sweep scripts (sweep_temperature*.sh, sweep_dt*.sh, and any future
# ones) instead of each one re-implementing its own copy.
#
# Replaces the fixed MAX_RUNNING=4 headcount cap every sweep script in
# this project used until now (sweep_dt.sh, sweep_dt_seeds.sh,
# sweep_temperature*.sh) -- that number was never derived from measured
# GPU memory, just chosen conservatively; e.g. the refine-long sweep's
# own launch log showed 21451 MB still free with 3 jobs already running.
# See DEVLOG.txt, 2026-09-25 "GPU scheduler: memory-gated instead of a
# fixed headcount" entry.
#
# New policy: gate NEW launches primarily on live free GPU memory
# (MIN_FREE_MB, default 5000) rather than a small fixed process count.
# MAX_RUNNING still exists but only as a generous soft safety rail
# against a runaway loop, not the primary limiter -- memory is. This also
# directly answers the "what if a colleague starts a job on the same
# GPU while mine are running" case: nvidia-smi's free-memory number
# reflects EVERY process on the card, not just this script's own, so a
# colleague's job eating memory naturally slows down (not necessarily
# stops) new launches from this scheduler without any colleague-specific
# logic needed. Nothing already RUNNING is ever paused or killed by
# this -- only new launches are gated, which is the only point where
# gating is actually safe/meaningful (an already-running process can't
# be paused without its own checkpoint/resume support).
#
# Caveat, not resolved here: MIN_FREE_MB is only a real safety margin if
# it's >= a single job's actual peak memory footprint. That footprint
# has not been directly measured in this project (only the fact that the
# 4000-4999 MB gate never blocked a launch so far) -- if per-job peak
# usage is ever measured and found close to or above 5000 MB, raise
# MIN_FREE_MB accordingly before trusting a much higher MAX_RUNNING.
#
# Usage (see sweep_temperature_refine_long.sh's launch loop for the
# pre-refactor shape this replaces):
#   source gpu_scheduler.sh
#   for T in "${TS[@]}"; do
#       wait_for_gpu_slot
#       nohup python main_rate.py ... &
#       PID=$!
#       running_pids+=("$PID")
#   done
#   while [ "${#running_pids[@]}" -gt 0 ]; do
#       cleanup_finished
#       if [ "${#running_pids[@]}" -gt 0 ]; then sleep "$CHECK_INTERVAL"; fi
#   done
# ============================================================

MIN_FREE_MB="${MIN_FREE_MB:-5000}"
MAX_RUNNING="${MAX_RUNNING:-16}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

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

# Blocks until it's safe to launch one more job (memory clears
# MIN_FREE_MB AND the soft MAX_RUNNING cap isn't hit), then returns.
# Caller launches immediately after and appends the new PID to
# running_pids itself (this function doesn't launch anything, so the
# caller keeps full control of its own command line).
wait_for_gpu_slot() {
    while true; do
        cleanup_finished
        FREE=$(gpu_free_mb)
        echo "GPU free memory: ${FREE} MB   Running: ${#running_pids[@]} / ${MAX_RUNNING} (soft cap)"
        if [ "${#running_pids[@]}" -lt "$MAX_RUNNING" ] && [ "$FREE" -ge "$MIN_FREE_MB" ]; then
            return 0
        fi
        echo "Waiting ${CHECK_INTERVAL}s (GPU memory below ${MIN_FREE_MB} MB or soft cap reached -- may be a colleague's job)..."
        sleep "$CHECK_INTERVAL"
    done
}
