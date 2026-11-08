#!/usr/bin/env bash
# The decode gate for a candidate sampling target, end to end, on the GPU machine.
#
# Why: since HA0 turned out not to be predictable (DEVLOG 2026-11-04) every new
# candidate has to pass the same test before any sampling time is spent on it,
# and the test is always the same four steps. This script is those four steps,
# so a candidate costs one command instead of an afternoon of remembering.
#
#   1. add the sequence to references.py from UniProt   (needs network)
#   2. download the experimental structure from RCSB     (needs network)
#   3. decode it at 8, 32 and 128 generation steps       (needs the GPU)
#   4. compare each decode with the experiment, and run prediction_diagnostics
#
# Everything lands in structure_compare/ and in logs named after the reference.
#
# Usage, from softmax/monte_carlo:
#   ./run_target_gate.sh p450cam P00183 2CPP
#   ./run_target_gate.sh p450bm3 P14779 1BU7 1:455      # keep residues 1-455
#   REF_ONLY=1 ./run_target_gate.sh p450cam P00183 2CPP # skip step 1 (already added)
#   STEPS="32" ./run_target_gate.sh p450cam P00183 2CPP # fewer step counts
#
# Arguments: <reference name> <UniProt accession> <PDB id> [START:END]
#
# Read the gate as: mean pLDDT around 90 and a low RMSD means the candidate is
# usable; pLDDT near 0.5-0.7 with a large RMSD means it is not, whatever else
# about the protein is attractive. prediction_diagnostics.py additionally prints
# the contact-map energy between the prediction and the crystal, which is the
# sampler's own units -- the number to quote when saying a target is unusable.
set -euo pipefail

NAME=${1:?reference name, e.g. p450cam}
ACC=${2:?UniProt accession, e.g. P00183}
PDBID=${3:?PDB id, e.g. 2CPP}
RANGE=${4:-}
STEPS=${STEPS:-8 32 128}
OUT=structure_compare
mkdir -p "$OUT"

echo "=== gate for $NAME (UniProt $ACC, PDB $PDBID${RANGE:+, residues $RANGE})"

if [ -z "${REF_ONLY:-}" ]; then
	echo "--- 1. sequence into references.py"
	entry="$ACC:$NAME"
	[ -n "$RANGE" ] && entry="$ACC:$NAME:${RANGE%%:*}:${RANGE##*:}"
	(cd ../code && python tests/add_reference.py --entries "$entry")
	# Fail here rather than three steps later with a confusing KeyError.
	(cd ../code && python -c "
import sys; sys.path.insert(0, 'tests'); import references as r
assert '$NAME' in r.REFERENCES, '$NAME is not in references.py'
print('   references.py: $NAME, %d aa' % len(r.REFERENCES['$NAME']))")
fi

echo "--- 2. experimental structure"
if [ ! -s "$OUT/$PDBID.pdb" ]; then
	wget -q -O "$OUT/$PDBID.pdb" "https://files.rcsb.org/download/$PDBID.pdb"
fi
grep -c '^ATOM' "$OUT/$PDBID.pdb" | sed 's/^/   ATOM records: /'
# A deposited entry is usually the whole assembly; compare_structures.py picks
# the chains it can pair, but say how many there are, because a multi-chain
# entry is exactly the case that made HA ambiguous.
awk '$1=="ATOM"{print substr($0,22,1)}' "$OUT/$PDBID.pdb" | sort -u | tr '\n' ' ' | sed 's/^/   chains: /'
echo

for s in $STEPS; do
	tag=""; [ "$s" != 32 ] && tag="_${s}steps"
	echo "--- 3. decode at $s steps"
	python -u decode_to_pdb.py --ref "$NAME" --num-steps "$s" \
		--out "$OUT/${NAME}_pred${tag}.pdb" 2>&1 | tee "${NAME}_decode${tag}.log"

	echo "--- 4. compare at $s steps"
	python -u compare_structures.py \
		--pred "$OUT/${NAME}_pred${tag}.pdb" \
		--exp "$OUT/$PDBID.pdb" \
		--out-prefix "$OUT/${NAME}_vs_${PDBID}${tag}" 2>&1 | tee "${NAME}_cmp${tag}.log"

	python -u prediction_diagnostics.py \
		--pred "$OUT/${NAME}_pred${tag}.pdb" --exp "$OUT/$PDBID.pdb" \
		--csv "$OUT/${NAME}_vs_${PDBID}${tag}_per_residue.csv" \
		2>&1 | tee "${NAME}_diag${tag}.log" || \
		echo "   (diagnostics skipped: the two files probably differ in residue count)"
done

echo
echo "=== summary"
grep -h "mean pLDDT" "${NAME}"_decode*.log || true
grep -h "CA RMSD\|TM-score" "${NAME}"_cmp*.log || true
grep -h "E = sum" "${NAME}"_diag*.log || true
echo "Commit the logs and structure_compare/ so the next session can read them:"
echo "  git add ${NAME}_*.log $OUT/${NAME}_* && git commit -m '$NAME gate' && git push"
