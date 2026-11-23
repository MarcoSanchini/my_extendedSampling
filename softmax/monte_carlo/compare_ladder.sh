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
#
# TWO passes are written, because they answer different questions and quoting the wrong one
# produced a spurious result on 2026-11-19:
#   OUT/            --burn-in-frac 0.8, every chain's window safely past its own t_eq. Use
#                   it for a statement about ONE temperature.
#   OUT/matched16k/ --match-accepted 16000, every window holding the same number of ACCEPTED
#                   moves. Use it for any statement COMPARING temperatures. A fixed fraction
#                   does not: the per-site entropy is attenuated when the window holds few
#                   accepted moves, so at 0.8 the hot chains' short windows scored lower than
#                   they should and invented a maximum at T=6 that is really a plateau.
#                   Caveat printed in its own log: at 16000 the windows of T=2, 3 and 5 start
#                   before their t_eq, so those rows are lower bounds, and T=1 and T=1.5
#                   never reach 16000 accepted moves at all (3000 and 8300 in 300000 moves).
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
mkdir -p "$OUT/matched16k"
python -u pten_k_sites.py compare --match-accepted 16000 --runs $RUNS \
	--out-dir "$OUT/matched16k" "$@"
