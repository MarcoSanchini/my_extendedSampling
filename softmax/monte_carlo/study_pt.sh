#!/bin/bash

set -u

# ============================================================
# Replica exchange on zero_polymer (run_pt.py, DEVLOG 2026-10-24): two
# independent ladders with the same temperatures, one with every walker at the
# reference and one with every walker at a random sequence, run one after the
# other (one job saturates the GPU). Then:
#     python analyze_pt.py $RUN_DIR/ref $RUN_DIR/rand
#
# Defaults are the pilot: 16 temperatures from 12 to 316, 2000 sweeps per
# ladder (about 1.4 h each at 0.153 s per one-pass move). Its swap acceptances
# and Cv(T) decide the production ladder: analyze_pt.py flags rungs below 0.1.
#
#   nohup bash study_pt.sh > study_pt_pilot.log 2>&1 &
#   RESUME=1 SWEEPS=10000 nohup bash study_pt.sh > study_pt_more.log 2>&1 &
#   TS="12 14 16 ..." RUN_DIR=$PWD/pt_prod SWEEPS=10000 nohup bash study_pt.sh > study_pt_prod.log 2>&1 &
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/pt_study}"
SWEEPS="${SWEEPS:-2000}"
LOG_EVERY="${LOG_EVERY:-100}"
T_MIN="${T_MIN:-12}"
T_MAX="${T_MAX:-316}"
N_TEMPS="${N_TEMPS:-16}"
TS="${TS:-}"
GRAD_MODE="${GRAD_MODE:-one_pass}"
N_SITES="${N_SITES:-1}"

LADDER=(--t-min "$T_MIN" --t-max "$T_MAX" --n-temps "$N_TEMPS")
[ -n "$TS" ] && LADDER=(--ts "$TS")
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

mkdir -p "${RUN_DIR}/logs"
for START in reference random; do
    NAME=ref; SEED=300
    [ "$START" = "random" ] && NAME=rand && SEED=400
    CMD=(python -u "${SCRIPT_DIR}/run_pt.py" "${LADDER[@]}" --sweeps "$SWEEPS" --log-every "$LOG_EVERY"
        --start "$START" --seed "$SEED" --grad-mode "$GRAD_MODE" --n-sites "$N_SITES"
        --out "${RUN_DIR}/${NAME}" $RESUME_FLAG)
    echo "${CMD[*]}"
    [ "${DRY_RUN:-0}" = "1" ] && continue
    "${CMD[@]}" > "${RUN_DIR}/logs/pt_${NAME}.log" 2>&1
    echo "ladder ${NAME} finished (exit $?)"
done

echo "Analyse with: python ${SCRIPT_DIR}/analyze_pt.py ${RUN_DIR}/ref ${RUN_DIR}/rand"
