#!/bin/bash

set -u

# ============================================================
# Structural metrics, plots and the H3N2 site-conservation comparison for the
# zero_polymer chains of the n_sites study: the zero_polymer counterpart of
# run_plotting_suite.sh (protein G). Per chain: plot_run.py,
# compute_structural_metrics.py (ESM3 structure decode of every STRIDE-th
# post-burn-in checkpoint: the GPU step), plot_structural_metrics.py. Then
# structure_zero_polymer.py decode-ref (reference + random-sequence baselines)
# and structure_zero_polymer.py plot (box plots, metrics against T, site
# entropy vs H3N2). Needs the eprot/ checkpoints, so it runs on the GPU machine.
#
#   nohup bash structure_zero_polymer.sh > structure_zero_polymer.log 2>&1 &
#
# RUNS     LABEL=DIR list (a label may repeat; its chains are pooled). Default:
#          the reference-started chains at T5, T12 (equil_lowT), T32 (equil_op
#          0, 2) and all four T100 chains (mixed by 140k moves).
# COLD     labels used for the K-sites (default "T5 T12").
# H3N2     per-site natural entropy (default ../../FASTA/H3N2/site_entropy_566.csv,
#          from the repo root FASTA/: human H3N2). Point it at a cross-subtype
#          file from ha_site_entropy.py for the wider comparison, e.g.
#          H3N2=ha_natural/ha_group2_entropy_566.csv (see ha_natural_runbook.md).
# STRIDE   decode every STRIDE-th post-burn-in checkpoint (default 10).
# BURN_IN  fraction of each chain's checkpoints dropped (default 0.5).
# NUM_STEPS structure-generation steps per decode (default 8; 1 understates
#          Rg and pLDDT differences, DEVLOG 2026-09-30).
# RESIDUES START:END window also reported as *_win (e.g. 16:529, the ectodomain:
#          no signal peptide, no transmembrane/cytoplasmic tail). Default: none.
# SKIP_DECODE=1 reuses existing structural_metrics.csv files and baselines.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NS=nsites_study
RUNS=(${RUNS:-T5=$NS/equil_lowT/T5/chain0 T12=$NS/equil_lowT/T12/chain0 \
	T32=$NS/equil_op/T32/chain0 T32=$NS/equil_op/T32/chain2 \
	T100=$NS/equil_op/T100/chain0 T100=$NS/equil_op/T100/chain1 T100=$NS/equil_op/T100/chain2 T100=$NS/equil_op/T100/chain3})
COLD=(${COLD:-T5 T12})
H3N2="${H3N2:-$SCRIPT_DIR/../../FASTA/H3N2/site_entropy_566.csv}"
STRIDE="${STRIDE:-10}"
BURN_IN="${BURN_IN:-0.5}"
NUM_STEPS="${NUM_STEPS:-8}"
OUT="${OUT:-$NS/structure}"
RES_ARG=()
[ -n "${RESIDUES:-}" ] && RES_ARG=(--residues "$RESIDUES")
REF_SEQ=$(python -c "import sys; sys.path.append('../code/tests'); from references import get_reference; print(get_reference('zero_polymer')[1])")

mkdir -p "$OUT"
KEPT=()
for item in "${RUNS[@]}"; do
	DIR="${item#*=}"
	if ! ls "$DIR"/eprot/sequence_*.pt > /dev/null 2>&1; then
		echo "skip $item: no eprot checkpoints"
		continue
	fi
	KEPT+=("$item")
	echo; echo "=== $item -- $(date) ==="
	python plot_run.py "$DIR"
	if [ "${SKIP_DECODE:-0}" != "1" ] || [ ! -f "$DIR/structural_metrics.csv" ]; then
		python compute_structural_metrics.py "$DIR" --stride "$STRIDE" --num-steps "$NUM_STEPS" --burn-in-frac "$BURN_IN" "${RES_ARG[@]}"
	fi
	python plot_structural_metrics.py "$DIR" --ref-seq "$REF_SEQ" --burn-in-frac "$BURN_IN"
done

if [ "${SKIP_DECODE:-0}" != "1" ] || [ ! -f "$OUT/baseline_metrics.csv" ]; then
	echo; echo "=== baselines -- $(date) ==="
	python structure_zero_polymer.py decode-ref --out "$OUT/baseline_metrics.csv" --num-steps "$NUM_STEPS" "${RES_ARG[@]}"
fi

H3N2_ARG=()
if [ -f "$H3N2" ]; then H3N2_ARG=(--h3n2 "$H3N2"); else echo "no H3N2 entropy file at $H3N2: comparison skipped"; fi
echo; echo "=== plots -- $(date) ==="
python structure_zero_polymer.py plot --runs "${KEPT[@]}" --baseline "$OUT/baseline_metrics.csv" \
	"${H3N2_ARG[@]}" --cold "${COLD[@]}" --burn-in-frac "$BURN_IN" --out-dir "$OUT"
echo "Done -- $(date). Plots: $OUT/*.png and <chain>/plots/*.png"
