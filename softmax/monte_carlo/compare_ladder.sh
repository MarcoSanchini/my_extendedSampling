#!/usr/bin/env bash
# The nine-temperature ladder against the natural / abundance / activity references.
#
#   bash compare_ladder.sh [OUT_DIR] [EXTRA ARGS TO pten_k_sites.py compare]
#
# Why a script and not one --runs-dir: the ladder whose t_eq has been measured is split
# across two directories. equil_ref holds the long reference-started chains at T = 1, 1.5,
# 2, 4, 8 and 16; T = 3, 5 and 6 only ever ran in equil. Pointing --runs-dir at one of them
# silently drops temperatures, which is how the T = 6 maximum stayed hidden until
# 2026-11-19. The hot short chains in equil (T32, T64, T128, T316) are deliberately left
# out: they are the 30,000-move ladder of 2026-11-12 and have no measured t_eq.
#
# --burn-in-frac 0.8 is not optional: the 0.5 default does not clear t_eq at seven of the
# nine temperatures. See docs/pten_k_sites_explained.md section 5.2.
set -eu
cd "$(dirname "$(readlink -f "$0")")"
OUT=${1:-pten_natural}
[ $# -gt 0 ] && shift

RUNS=""
for T in 1 1.5 2 4 8 16; do
	RUNS="$RUNS T$T=pten1d5r_study/equil_ref/T$T/chain0 T$T.c1=pten1d5r_study/equil_ref/T$T/chain1"
done
for T in 3 5 6; do
	RUNS="$RUNS T$T=pten1d5r_study/equil/T$T/chain0 T$T.c1=pten1d5r_study/equil/T$T/chain1"
done

set -x
python -u pten_k_sites.py compare --burn-in-frac 0.8 --runs $RUNS --out-dir "$OUT" "$@"
