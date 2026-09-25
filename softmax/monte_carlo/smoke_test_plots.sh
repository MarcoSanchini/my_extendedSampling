#!/bin/bash

# ============================================================
# One-shot smoke test for plot_run.py / compute_structural_metrics.py /
# plot_structural_metrics.py / identify_k_sites.py / plot_plddt_vs_T.py,
# run against the already-completed refine-long sweep (T=0.2512, 0.3981,
# 0.6310, see sweep_temperature_refine_long.sh / DEVLOG.txt 2026-09-25
# entries). Bundled into one script (rather than backgrounding each
# command separately with its own &) because several steps depend on an
# earlier one's output file existing -- plot_structural_metrics.py and
# plot_plddt_vs_T.py both read structural_metrics.csv, which only exists
# after compute_structural_metrics.py has run for that T. Backgrounding
# them independently would race.
#
# --stride 100 below is a deliberately small smoke-test subsample (~7-8
# decodes per T instead of the default stride=10's ~75/T) -- this is
# purely to confirm the code path works and to time one decode before
# committing to a real stride/full sweep; NOT a value to reuse for real
# runs. See compute_structural_metrics.py's own cost-warning docstring.
#
# Run in the background so it's safe to close the terminal:
#   nohup bash smoke_test_plots.sh > smoke_test_plots.log 2>&1 &
# Then check progress any time with:
#   tail -f smoke_test_plots.log
# ============================================================

set -e   # stop on first real error rather than silently limping through the rest

TS=(0.2512 0.3981 0.6310)
BASE_DIR="tests_rate_sweep_temperature_refine_long"
STRIDE=100

for T in "${TS[@]}"; do
    RESULTS_DIR="${BASE_DIR}/T${T}/sim0"

    echo
    echo "============================================================"
    echo "T=$T -- $(date)"
    echo "============================================================"

    echo "--- plot_run.py ---"
    python plot_run.py "$RESULTS_DIR"

    echo "--- compute_structural_metrics.py (stride=$STRIDE) ---"
    python compute_structural_metrics.py "$RESULTS_DIR" --stride "$STRIDE"

    echo "--- plot_structural_metrics.py ---"
    python plot_structural_metrics.py "$RESULTS_DIR"
done

echo
echo "============================================================"
echo "Cross-T plots -- $(date)"
echo "============================================================"

RESULTS_DIRS=()
for T in "${TS[@]}"; do
    RESULTS_DIRS+=("${BASE_DIR}/T${T}/sim0")
done

echo "--- identify_k_sites.py ---"
python identify_k_sites.py --results-dirs "${RESULTS_DIRS[@]}" --active-t "${TS[@]}"

echo "--- plot_plddt_vs_T.py ---"
python plot_plddt_vs_T.py --results-dirs "${RESULTS_DIRS[@]}" --active-t "${TS[@]}"

echo
echo "============================================================"
echo "Smoke test complete -- $(date)"
echo "Plots: ${BASE_DIR}/T<T>/sim0/plots/*.png, k_sites_plots/*.png"
echo "============================================================"
