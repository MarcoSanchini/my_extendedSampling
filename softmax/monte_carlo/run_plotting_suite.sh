#!/bin/bash

# ============================================================
# Generalization of smoke_test_plots.sh (which stays as-is, hardcoded to
# the refine-long sweep it was first tested against -- see DEVLOG.txt
# 2026-09-25 "first real structural sweep" entry) -- runs the same 5-
# script sequence (plot_run.py, compute_structural_metrics.py,
# plot_structural_metrics.py, identify_k_sites.py, plot_plddt_vs_T.py)
# against ANY sweep's results directory, so a new sweep (e.g.
# sweep_temperature_paper_scale.sh) doesn't need its own hand-edited copy
# of this script.
#
# Same dependency reasoning as smoke_test_plots.sh: bundled into one
# script rather than backgrounded per-command, since plot_structural_
# metrics.py and plot_plddt_vs_T.py both read structural_metrics.csv,
# which only exists after compute_structural_metrics.py has run for that
# T -- backgrounding them independently would race.
#
# Usage:
#   bash run_plotting_suite.sh <base_dir> <T1> <T2> ... <TN>
#   # e.g.
#   bash run_plotting_suite.sh tests_rate_sweep_temperature_paper_scale \
#       0.2512 0.3981 0.631 1.0 1.585 2.512
#
# Safe to background:
#   nohup bash run_plotting_suite.sh <base_dir> <T...> > run_plotting_suite.log 2>&1 &
#
# --stride below (100) is the same smoke-test-scale subsample
# smoke_test_plots.sh used -- NOT necessarily right for a real analysis;
# override by editing STRIDE below once you know the real per-decode
# wall-clock time (see compute_structural_metrics.py's own cost-warning
# docstring).
# ============================================================

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: bash run_plotting_suite.sh <base_dir> <T1> [T2 ...]"
    exit 1
fi

BASE_DIR="$1"
shift
TS=("$@")
STRIDE=100

RESULTS_DIRS=()
for T in "${TS[@]}"; do
    RESULTS_DIRS+=("${BASE_DIR}/T${T}/sim0")
done

for i in "${!TS[@]}"; do
    T="${TS[$i]}"
    RESULTS_DIR="${RESULTS_DIRS[$i]}"

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

# Both cross-T scripts default to k_sites_plots/ -- namespaced per BASE_DIR
# here so running this against a second sweep doesn't silently overwrite
# the first sweep's per_site_entropy.png/csv and plddt_vs_T.png (this
# already happened once between the refine-long smoke test and this
# script's own first intended use against the paper-scale sweep).
OUT_DIR="k_sites_plots_$(basename "$BASE_DIR")"

echo "--- identify_k_sites.py ---"
python identify_k_sites.py --results-dirs "${RESULTS_DIRS[@]}" --active-t "${TS[@]}" --out-dir "$OUT_DIR"

echo "--- plot_plddt_vs_T.py ---"
python plot_plddt_vs_T.py --results-dirs "${RESULTS_DIRS[@]}" --active-t "${TS[@]}" --out-dir "$OUT_DIR"

echo
echo "============================================================"
echo "Plotting suite complete -- $(date)"
echo "Plots: ${BASE_DIR}/T<T>/sim0/plots/*.png, ${OUT_DIR}/*.png"
echo "============================================================"
