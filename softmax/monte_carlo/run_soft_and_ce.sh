#!/usr/bin/env bash
# The soft-contact diagnostic (section 6.6e) and the U_structure_ce scan (6.6g), one
# GPU process at a time and in this order, then the correlations, which need no GPU.
#
#   cd ~/my_project/extendedSampling/softmax/monte_carlo
#   nohup ./run_soft_and_ce.sh > run_soft_and_ce.log 2>&1 &
#   tail -f run_soft_and_ce.log
#
# Why this cannot leak the card: every step is a separate python PROCESS, started only
# after the previous one has exited, and a process returns all of its CUDA memory to the
# driver when it exits. Nothing here is backgrounded with '&', and `set -e` stops the
# whole script at the first failure rather than starting the next model. The nvidia-smi
# line after each step is the proof: it should come back to roughly its starting value.
#
# Env knobs: STEPS (default "2 8 32"), SITES (default 0 17 34 ... 323, the 20-site
# subset every E scan so far has used), CE_SITES (default all 324), SKIP_CE=1.
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
STEPS=${STEPS:-"2 8 32"}
SITES=${SITES:-$(seq 0 17 323)}
UAM=pten_natural/pten_single_mutants.csv
SOFT2=pten_natural/pten_mutants_soft_E2.csv

say() { printf '\n== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
mem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"; }
fail() { say "FAILED at: $BASH_COMMAND"; mem; }
trap fail ERR

say "start; GPU now at $(mem)"
if [ ! -f "$UAM" ]; then
	say "$UAM is missing: the U_am scan has to exist first (step 6 of section 9). Stopping."
	exit 1
fi

# ---- 1. soft-contact E, one step count at a time -------------------------------------
# Hard and soft E (widths 0.5, 1, 2 A) come out of the SAME decode, so the soft columns
# are free; the cost is the decodes. 2 steps is ~0.14 s per mutant, 32 steps ~1.3 s.
first=1
for st in $STEPS; do
	out=pten_natural/pten_mutants_soft_E$st.csv
	if [ "$first" = 1 ]; then
		say "soft-contact E at $st steps -> $out (no --compare: it is the baseline)"
		python -u scan_mutants_E.py --num-steps "$st" --sites $SITES \
			--out "$out" --am-scan "$UAM" > "scan_mutants_soft_E$st.log" 2>&1
		first=0
	else
		say "soft-contact E at $st steps -> $out, compared with $SOFT2"
		python -u scan_mutants_E.py --num-steps "$st" --sites $SITES \
			--out "$out" --am-scan "$UAM" --compare "$SOFT2" > "scan_mutants_soft_E$st.log" 2>&1
	fi
	tail -14 "scan_mutants_soft_E$st.log"
	say "GPU after $st steps: $(mem)"
done

# ---- 2. U_structure_ce ---------------------------------------------------------------
# The other structural energy in energies.py: the cross-entropy of the predicted
# structure-token distribution against the reference's own structure tokens. One extra
# forward through the structure head per mutant, no generate(), no decoding schedule.
# Five sites first, to see the cost before committing the card; the scan is resumable,
# so those five are kept and the full run carries on from site 5.
if [ "${SKIP_CE:-0}" != 1 ]; then
	CE=pten_natural/pten_mutants_ce.csv
	say "U_structure_ce, 5 sites first -> $CE"
	python -u scan_single_mutants.py --structure-ce --sites 0 1 2 3 4 --out "$CE" > scan_mutants_ce_trial.log 2>&1
	grep -E "U_structure_ce\(reference\)|noise floor|s per mutant|sites," scan_mutants_ce_trial.log | tail -6
	say "GPU after the trial: $(mem)"
	say "U_structure_ce, all ${CE_SITES:+listed }sites (resumes, the 5 above are kept)"
	python -u scan_single_mutants.py --structure-ce ${CE_SITES:+--sites $CE_SITES} \
		--out "$CE" > scan_mutants_ce.log 2>&1
	tail -4 scan_mutants_ce.log
	say "GPU after the CE scan: $(mem)"
fi

# ---- 3. correlations, no GPU ---------------------------------------------------------
say "correlations (CPU only)"
python pten_k_sites.py mutants --mutants "$SOFT2" --energy-column E_w1 --tag soft_E2_w1 \
	--versus "$UAM" > pten_natural/mutants_soft_E2_w1.log 2>&1
if [ "${SKIP_CE:-0}" != 1 ]; then
	python pten_k_sites.py mutants --mutants pten_natural/pten_mutants_ce.csv --energy-column dU_ce \
		--tag ce --versus "$UAM" > pten_natural/mutants_ce.log 2>&1
	sed -n '/--- against the other energy/,$p' pten_natural/mutants_ce.log | head -30
fi

say "done; GPU back to $(mem)"
echo
echo "Read these:"
echo "  scan_mutants_soft_E*.log   'agreement between the two' block: hard E vs each soft width"
echo "  pten_natural/mutants_ce.log        U_structure_ce against both experiments"
echo "  pten_natural/mutants_soft_E2_w1.log   the soft E at width 1 A against both experiments"
