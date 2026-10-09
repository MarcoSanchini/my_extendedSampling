#!/usr/bin/env bash
# The experimental-comparison and "maintain / impose the bridge" runs, in order.
# Run from softmax/code. Phase 1 needs a machine that can reach the PDB; phase 2
# needs the GPU and the ESM3 weights. Each step writes its output to its own
# file, because a generic filename has already overwritten a control once.
#
#   bash bridge_test/run_pdb_comparison.sh fetch       # phase 0: experimental entries
#   bash bridge_test/run_pdb_comparison.sh predict     # phase 1a: predictions with PDB + pLDDT
#   bash bridge_test/run_pdb_comparison.sh compare     # phase 1b: contacts / RMSD / disulfides
#   bash bridge_test/run_pdb_comparison.sh roundtrip   # phase 2a: can the representation hold the bonds?
#   bash bridge_test/run_pdb_comparison.sh prompt      # phase 2b: can the bridge be imposed?
set -euo pipefail
cd "$(dirname "$0")/.."            # softmax/code
mkdir -p bridge_test/pred bridge_test/rt bridge_test/cmp

exp() { ls bridge_test/pdb/"$1"_*.pdb | head -1; }   # the entry fetch_pdb.py kept

case "${1:-}" in
fetch)
  python bridge_test/fetch_pdb.py --search | tee bridge_test/cmp/fetch_pdb.txt
  ;;
predict)
  # same steps as the original controls; decoding is deterministic, so the numbers
  # should reproduce the earlier control files exactly
  python tests/test_cleaved_complex.py --ref protein_g --steps 56 --pdb bridge_test/pred/protein_g.pdb > bridge_test/pred/protein_g.txt
  python tests/test_cleaved_complex.py --ref bpti      --steps 58 --pdb bridge_test/pred/bpti.pdb      > bridge_test/pred/bpti.txt
  python tests/test_cleaved_complex.py --ref crambin   --steps 46 --pdb bridge_test/pred/crambin.pdb   > bridge_test/pred/crambin.txt
  ;;
compare)
  for r in protein_g bpti crambin; do
    python bridge_test/compare_to_pdb.py --ref "$r" --exp "$(exp $r)" --pred bridge_test/pred/$r.pdb \
      --plddt bridge_test/pred/$r.pdb.plddt.tsv | tee bridge_test/cmp/compare_$r.txt
  done
  ;;
roundtrip)
  for r in bpti crambin; do
    python bridge_test/roundtrip_bridge.py --ref $r --exp "$(exp $r)" --mode roundtrip | tee bridge_test/cmp/roundtrip_$r.txt
    python bridge_test/compare_to_pdb.py --ref $r --exp "$(exp $r)" --pred bridge_test/rt/${r}_roundtrip.pdb \
      --plddt bridge_test/rt/${r}_roundtrip.pdb.plddt.tsv > bridge_test/cmp/compare_${r}_roundtrip.txt
  done
  ;;
prompt)
  # one bond, one bond with neighbours, then every bond: how much telling does it take?
  python bridge_test/roundtrip_bridge.py --ref bpti --exp "$(exp bpti)" --mode prompt --residues 14,38 --window 0 | tee bridge_test/cmp/prompt_bpti_14-38_w0.txt
  python bridge_test/roundtrip_bridge.py --ref bpti --exp "$(exp bpti)" --mode prompt --residues 14,38 --window 2 | tee bridge_test/cmp/prompt_bpti_14-38_w2.txt
  python bridge_test/roundtrip_bridge.py --ref bpti --exp "$(exp bpti)" --mode prompt --all-bonds --window 0   | tee bridge_test/cmp/prompt_bpti_allbonds_w0.txt
  python bridge_test/roundtrip_bridge.py --ref crambin --exp "$(exp crambin)" --mode prompt --residues 3,40 --window 0 | tee bridge_test/cmp/prompt_crambin_3-40_w0.txt
  python bridge_test/roundtrip_bridge.py --ref crambin --exp "$(exp crambin)" --mode prompt --all-bonds --window 0 | tee bridge_test/cmp/prompt_crambin_allbonds_w0.txt
  ;;
*)
  sed -n '2,12p' "$0"; exit 1
  ;;
esac
