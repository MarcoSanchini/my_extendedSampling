#!/bin/bash

set -u

# ============================================================
# dt x seed sweep for classes/rate_sampler.py:ExtendedProteinRateSampler,
# following up on sweep_dt.sh's dt=(300,600,1000,1300) results (see
# DEVLOG.txt, 2026-09-22 entry): the mutated-site-vs-reference-matching
# accept-rate gap grew with dt but only reached gap/SE=+1.84 at dt=1300,
# still not decisive, and non-monotonically (dt=600 dipped negative,
# within 1-sigma noise). Two changes here address that:
#
# 1. dt pushed well past 1300. The old -1e6 additive exclusion penalty
#    (removed in the slicing refactor, see rate_sampler.py:
#    _competition_indices) used to cap how far dt could safely go before
#    step_coef*grad risked overwhelming it (~dt>1400). That ceiling no
#    longer exists: utils/proposals.py:log_pointing_prob is implemented
#    entirely in log-space (torch.special.log_ndtr for the tail -- stable
#    for very negative z, unlike log(Phi(z)) -- and torch.logsumexp for
#    the quadrature sum), with no additive constant anywhere. At these dt
#    values and T=2, the standardized separations (z = alpha+u_q) reach at
#    most the low thousands for the gradient magnitudes seen so far
#    (|grad|~0.01-0.08), nowhere near where log_ndtr would need z^2 to
#    overflow float32 (~1e19) -- confirmed safe before launching this.
#
#    WATCH FOR: at very large dt the *reverse* proposal probability
#    (log_a_BA in _step()) can legitimately become very small whenever the
#    return move isn't itself gradient-favored from the new state -- that's
#    correct Hastings-correction behavior, not a bug, and could mean
#    overall acceptance plateaus or even drops at the top of this range
#    rather than climbing forever. Worth reading as a real finding either
#    way, not assuming the trend must keep climbing.
#
# 2. Four seeds per dt instead of one. The limiting factor on the previous
#    sweep's statistics wasn't dt resolution -- it was sample size: only
#    ~342 proposals land on the (4 out of 56) initially-mutated sites per
#    5000-move run, since init_muts=5 only actually perturbs ~4 sites.
#    Four independent seeds per dt (each with its own independently-drawn
#    initial mutation pattern, not just a different downstream random
#    stream from the same start) pools to ~4x the proposals contributing
#    to the mutated-site accept rate at each dt, shrinking that SE
#    directly. Combine them with aggregate_seeds.py, not by eyeballing
#    four separate analyze_run.py printouts:
#      python aggregate_seeds.py tests_rate_sweep_dt_seeds/dt2000_seed4{2,3,4,5}
#
# T, moves, init_muts held fixed (same as the previous sweep) so this is
# directly comparable to the dt=(300,600,1000,1300) results already in
# DEVLOG.txt.
#
# Runtime note: 16 runs total (4 dt x 4 seeds) vs the previous sweep's 4 --
# at MAX_RUNNING=4 that's 4 launch waves instead of 1, so expect roughly
# 4x the previous sweep's wall-clock time.
# ============================================================

DTS=(2000 4000 8000 16000)
SEEDS=(42 43 44 45)

BASE_PARS="rate_inputs/pars.txt"
BASE_SETTINGS="rate_inputs/settings.txt"

T=2.0
MOVES=5000
INIT_MUTS=5

mkdir -p sweep_dt_seeds_inputs
mkdir -p logs_dt_seeds
mkdir -p tests_rate_sweep_dt_seeds

# Maximum number of simultaneous simulations (GPU-limited)
MAX_RUNNING=4

# Minimum free GPU memory required before launching a new simulation
MIN_FREE_MB=4000

CHECK_INTERVAL=30

running_pids=()


# ============================================================
# Create parameter files (dt, T, moves, init_muts, seed overridden;
# everything else -- M, T_sftm, lambda_am, lambda_S, n_quad, eps --
# comes through unchanged from BASE_PARS)
# ============================================================

for DT in "${DTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        PARS_FILE="sweep_dt_seeds_inputs/pars_dt${DT}_seed${SEED}.txt"

        # NOTE: substitution patterns require a whitespace boundary
        # ([[:space:]], not [[:space:]]*) so e.g. ^T doesn't also clobber
        # T_sftm -- see sweep_dt.sh's header for the bug this avoids.
        sed \
            -e "s/^dt[[:space:]].*/dt        ${DT}            # single-site step size/" \
            -e "s/^T[[:space:]].*/T         ${T}            # sampled temperature/" \
            -e "s/^moves[[:space:]].*/moves     ${MOVES}        # number of moves/" \
            -e "s/^init_muts[[:space:]].*/init_muts ${INIT_MUTS}            # initial random mutations/" \
            -e "s/^seed[[:space:]].*/seed      ${SEED}            # generator seed/" \
            "$BASE_PARS" > "$PARS_FILE"

        echo "dt=${DT}, T=${T}, moves=${MOVES}, init_muts=${INIT_MUTS}, seed=${SEED}"

    done
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
    for SEED in "${SEEDS[@]}"; do

        echo
        echo "============================================================"
        echo "Preparing dt=$DT seed=$SEED"
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

        RESULTS_DIR="tests_rate_sweep_dt_seeds/dt${DT}_seed${SEED}"

        mkdir -p "$RESULTS_DIR"


        # --------------------------------------------------------
        # Dedicated settings file
        # --------------------------------------------------------

        RUN_SETTINGS="sweep_dt_seeds_inputs/settings_dt${DT}_seed${SEED}.txt"

        sed \
            -e "s|^results_dir[[:space:]].*|results_dir ${RESULTS_DIR}|" \
            "$BASE_SETTINGS" > "$RUN_SETTINGS"


        echo "Launching dt=$DT seed=$SEED"
        echo "  T:        $T"
        echo "  moves:    $MOVES"
        echo "  results:  $RESULTS_DIR"
        echo "  log:      logs_dt_seeds/dt${DT}_seed${SEED}.log"
        echo "  GPU free: ${FREE} MB"


        # --------------------------------------------------------
        # Launch
        # --------------------------------------------------------

        nohup python main_rate.py \
            --pars-file "sweep_dt_seeds_inputs/pars_dt${DT}_seed${SEED}.txt" \
            --settings-file "$RUN_SETTINGS" \
            > "logs_dt_seeds/dt${DT}_seed${SEED}.log" 2>&1 &

        PID=$!

        running_pids+=("$PID")

        echo "  PID: $PID"
        echo "  Running now: ${#running_pids[@]} / ${MAX_RUNNING}"

    done
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
echo "All dt x seed sweep simulations completed."
echo "Results in tests_rate_sweep_dt_seeds/dt<DT>_seed<SEED>/sim0/data.dat"
echo
echo "Per-run detail:"
echo "  for DT in ${DTS[@]}; do for SEED in ${SEEDS[@]}; do"
echo "    echo \"--- dt=\$DT seed=\$SEED ---\""
echo "    python analyze_run.py tests_rate_sweep_dt_seeds/dt\${DT}_seed\${SEED}/sim0"
echo "  done; done"
echo
echo "Pooled per-dt stats (combined SE across the 4 seeds):"
echo "  for DT in ${DTS[@]}; do"
echo "    echo \"--- dt=\$DT, pooled over seeds ---\""
echo "    python aggregate_seeds.py tests_rate_sweep_dt_seeds/dt\${DT}_seed{42,43,44,45}/sim0"
echo "  done"
echo "============================================================"
