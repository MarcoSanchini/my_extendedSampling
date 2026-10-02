#!/usr/bin/env bash
set -u

python tests/test_proposals.py \
  > test_proposals.log 2>&1

REF=protein_g python -u tests/scan_U_am_landscape.py \
  2>&1 | tee protein_g/scan_U_am_CONTROL.txt

REF=protein_g python -u tests/test_steepest_descent.py \
  2>&1 | tee protein_g/steepest_descent_CONTROL.txt

python -u tests/scan_U_am_landscape.py \
  2>&1 | tee zero_polymer/scan_U_am_results.txt

python -u tests/test_steepest_descent.py \
  2>&1 | tee zero_polymer/steepest_descent_results.txt

python -u tests/compare_grad_methods.py \
  2>&1 | tee zero_polymer/compare_grad_results.txt

python -u tests/test_informedness_vs_dt.py \
  2>&1 | tee zero_polymer/informedness_vs_dt_results.txt

MOVES=2000 python -u tests/validate_boltzmann_toy.py \
  2>&1 | tee zero_polymer/boltzmann_Tcal.txt
