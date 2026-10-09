#!/usr/bin/env bash
# The experimental-comparison and "maintain / impose the bridge" runs, in order.
# Run from softmax/code. Phase 0 needs a machine that can reach the PDB; the
# others need the GPU and the ESM3 weights. Each step writes to its own file,
# because a generic filename has already overwritten a control once.
#
#   bash bridge_test/run_pdb_comparison.sh fetch      # experimental entries (needs PDB access)
#   bash bridge_test/run_pdb_comparison.sh predict    # the three controls at their default step count
#   bash bridge_test/run_pdb_comparison.sh compare    # those predictions against experiment
#   bash bridge_test/run_pdb_comparison.sh controls   # STEP SWEEP of the controls, then compare each
#   bash bridge_test/run_pdb_comparison.sh roundtrip  # can the representation hold the bonds?
#   bash bridge_test/run_pdb_comparison.sh prompt     # can a bridge be imposed?
#
# Run `controls` BEFORE trusting any fold result from `predict`. Each control was
# run at a single decoding step count (its length). The HA sweep showed step count
# can change a fold more than the effects under test, so a control's fold is a
# property of the protein only if it holds across step counts.
set -euo pipefail
cd "$(dirname "$0")/.."            # softmax/code
mkdir -p bridge_test/pred bridge_test/rt bridge_test/cmp bridge_test/cmp/v2 bridge_test/sweep_ctrl

exp() { ls bridge_test/pdb/"$1"_*.pdb | head -1; }   # the entry fetch_pdb.py kept
cmpv2() { # ref  predicted-pdb  output-name
  python bridge_test/compare_to_pdb.py --ref "$1" --exp "$(exp "$1")" --pred "$2" \
    --plddt "$2.plddt.tsv" | tee "bridge_test/cmp/v2/$3.txt"
}

case "${1:-}" in
fetch)
  python bridge_test/fetch_pdb.py --search | tee bridge_test/cmp/fetch_pdb.txt
  ;;
predict)
  python tests/test_cleaved_complex.py --ref protein_g --steps 56 --pdb bridge_test/pred/protein_g.pdb > bridge_test/pred/protein_g.txt
  python tests/test_cleaved_complex.py --ref bpti      --steps 58 --pdb bridge_test/pred/bpti.pdb      > bridge_test/pred/bpti.txt
  python tests/test_cleaved_complex.py --ref crambin   --steps 46 --pdb bridge_test/pred/crambin.pdb   > bridge_test/pred/crambin.txt
  ;;
compare)
  for r in protein_g bpti crambin; do
    cmpv2 "$r" "bridge_test/pred/$r.pdb" "compare_$r"
  done
  ;;
controls)
  # Step sweep of each control. Pre-registered reading (fixed before the run):
  #   a fold is CORRECT if CA RMSD <= 2.5 A and TM-score >= 0.7 against experiment;
  #   a control's fold is RELIABLE only if correct at every step count >= L/4;
  #   crambin's wrong fold is a property of the protein only if it is wrong at
  #   all of 11, 23 and 46 steps.
  for spec in "protein_g 14 28 56" "bpti 14 29 58" "crambin 11 23 46"; do
    set -- $spec; r=$1; shift
    for s in "$@"; do
      python tests/test_cleaved_complex.py --ref "$r" --steps "$s" --pdb "bridge_test/sweep_ctrl/${r}_${s}.pdb" \
        > "bridge_test/sweep_ctrl/${r}_${s}.txt"
      cmpv2 "$r" "bridge_test/sweep_ctrl/${r}_${s}.pdb" "ctrl_${r}_${s}"
    done
  done
  ;;
roundtrip)
  for r in bpti crambin; do
    python bridge_test/roundtrip_bridge.py --ref "$r" --exp "$(exp "$r")" --mode roundtrip \
      | tee "bridge_test/cmp/roundtrip_${r}.txt"
    cmpv2 "$r" "bridge_test/rt/${r}_roundtrip.pdb" "roundtrip_${r}"
  done
  ;;
prompt)
  # one bond, one bond with neighbours, then every bond
  python bridge_test/roundtrip_bridge.py --ref bpti --exp "$(exp bpti)" --mode prompt --residues 14,38 --window 0 \
    | tee bridge_test/cmp/prompt_bpti_14-38_w0.txt
  python bridge_test/roundtrip_bridge.py --ref bpti --exp "$(exp bpti)" --mode prompt --residues 14,38 --window 2 \
    | tee bridge_test/cmp/prompt_bpti_14-38_w2.txt
  python bridge_test/roundtrip_bridge.py --ref bpti --exp "$(exp bpti)" --mode prompt --all-bonds --window 0 \
    | tee bridge_test/cmp/prompt_bpti_allbonds_w0.txt
  python bridge_test/roundtrip_bridge.py --ref crambin --exp "$(exp crambin)" --mode prompt --residues 3,40 --window 0 \
    | tee bridge_test/cmp/prompt_crambin_3-40_w0.txt
  python bridge_test/roundtrip_bridge.py --ref crambin --exp "$(exp crambin)" --mode prompt --all-bonds --window 0 \
    | tee bridge_test/cmp/prompt_crambin_allbonds_w0.txt
  ;;
*)
  sed -n '2,15p' "$0"; exit 1
  ;;
esac
