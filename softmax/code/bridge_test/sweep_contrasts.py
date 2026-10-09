"""
Paired contrasts across the decoding-step sweep.

Written 2026-10-09 after the step sweep, to apply the rule the previous devlog
entry set in advance: a claim survives only if its SIGN holds at every step
count tested. Every contrast below is a difference between two constructs at the
SAME step count, so each row is a matched comparison; the last columns count how
many step counts agree in sign and give the mean and spread of the differences.

With three step counts, a mean and a spread are descriptive only -- there is no
honest p-value to be had from n=3, and none is claimed. The useful reading is
the sign pattern, and whether the spread is larger than the mean.

HA2 alone cannot run past 221 steps (a chain cannot have more steps than
positions), so its 256-step slot is its 221-step run, which is what the script
itself does when asked for 256.

    python bridge_test/sweep_contrasts.py          # from softmax/code
"""
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_runs import parse  # noqa: E402

STEPS = (64, 128, 256)


def load() -> dict:
	# Repeats of a run are byte-identical, so which copy is kept does not matter
	# -- EXCEPT for files written by older script versions. ha_c256.txt and
	# ha_mature.txt predate the pLDDT block (no per-chain values, no anchors) and
	# the exclusion of the chain-break position from the clash count (16, where
	# the corrected count is 15). Keeping whichever file sorts first silently
	# selected those, so prefer the row with the most parsed chains; ties keep
	# the first.
	runs = {}
	for path in sorted(HERE.glob("*.txt")):
		row = parse(path)
		if not (row and row.get("steps")):
			continue
		key = (row["ref"], row["steps"])
		if key not in runs or len(row["chains"]) > len(runs[key]["chains"]):
			runs[key] = row
	return runs


def chain_plddt(row, chain):
	c = row["chains"].get(chain) if row else None
	return c[0] if c else None


def frac70(row, chain):
	c = row["chains"].get(chain) if row else None
	return c[1] if c else None


def anchor14(row):
	a = row.get("anchors") if row else None
	return float(a.split("/")[0]) if a else None


def get(runs, ref, steps):
	# HA2 alone is capped at its length.
	if ref == "two":
		steps = min(steps, 221)
	return runs.get((ref, steps))


def contrast(runs, label, fa, fb, fmt="{:+7.2f}"):
	"""fa(steps) - fb(steps) at each step count, with a sign summary."""
	diffs = []
	cells = []
	for s in STEPS:
		a, b = fa(runs, s), fb(runs, s)
		if a is None or b is None:
			cells.append("      n/a")
			continue
		d = a - b
		diffs.append(d)
		cells.append(fmt.format(d))
	if diffs:
		pos = sum(1 for d in diffs if d > 0)
		neg = sum(1 for d in diffs if d < 0)
		sign = "+ at all" if neg == 0 and pos == len(diffs) else (
			"- at all" if pos == 0 and neg == len(diffs) else f"MIXED ({pos}+ / {neg}-)")
		mean = statistics.mean(diffs)
		sd = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
		tail = f"  {sign:<14s} mean {mean:+7.2f}  sd {sd:6.2f}"
	else:
		tail = ""
	print(f"  {label:<52s}" + "".join(f"{c:>10s}" for c in cells) + tail)


def main() -> None:
	runs = load()
	need = [("cleaved", s) for s in STEPS] + [("mature", s) for s in STEPS] + \
		   [("one", s) for s in STEPS] + [("two", 64), ("two", 128), ("two", 221)]
	missing = [k for k in need if k not in runs]
	if missing:
		print(f"WARNING: missing runs {missing}; affected contrasts show n/a\n")

	hdr = f"  {'contrast (difference, same steps)':<52s}" + "".join(f"{s:>10d}" for s in STEPS)
	print(hdr)
	print("  " + "-" * (len(hdr) - 2) + "-" * 40)

	C = lambda r, s: get(r, "cleaved", s)
	M = lambda r, s: get(r, "mature", s)
	O = lambda r, s: get(r, "one", s)
	T = lambda r, s: get(r, "two", s)

	print("HA1 pLDDT")
	contrast(runs, "break - fused        (cleaved - mature)",
			 lambda r, s: chain_plddt(C(r, s), "polymer_one"),
			 lambda r, s: chain_plddt(M(r, s), "polymer_one"))
	contrast(runs, "alone - in complex   (HA1 alone - cleaved)",
			 lambda r, s: chain_plddt(O(r, s), "one"),
			 lambda r, s: chain_plddt(C(r, s), "polymer_one"))
	contrast(runs, "alone - fused        (HA1 alone - mature)",
			 lambda r, s: chain_plddt(O(r, s), "one"),
			 lambda r, s: chain_plddt(M(r, s), "polymer_one"))
	print("HA1 % residues >= 70")
	contrast(runs, "break - fused",
			 lambda r, s: frac70(C(r, s), "polymer_one"),
			 lambda r, s: frac70(M(r, s), "polymer_one"), fmt="{:+7.1f}")
	print("HA1 Cys14 anchor pLDDT")
	contrast(runs, "break - fused",
			 lambda r, s: anchor14(C(r, s)),
			 lambda r, s: anchor14(M(r, s)), fmt="{:+7.1f}")
	print("HA2 pLDDT")
	contrast(runs, "alone - in cleaved complex",
			 lambda r, s: chain_plddt(T(r, s), "two"),
			 lambda r, s: chain_plddt(C(r, s), "polymer_two"))
	contrast(runs, "alone - in mature (fused) complex",
			 lambda r, s: chain_plddt(T(r, s), "two"),
			 lambda r, s: chain_plddt(M(r, s), "polymer_two"))
	contrast(runs, "break - fused        (cleaved - mature)",
			 lambda r, s: chain_plddt(C(r, s), "polymer_two"),
			 lambda r, s: chain_plddt(M(r, s), "polymer_two"))
	print("whole construct")
	contrast(runs, "overall pLDDT, break - fused",
			 lambda r, s: C(r, s)["plddt"] if C(r, s) and "plddt" in C(r, s) else None,
			 lambda r, s: M(r, s)["plddt"] if M(r, s) and "plddt" in M(r, s) else None)
	contrast(runs, "pTM, break - fused",
			 lambda r, s: C(r, s)["ptm"] if C(r, s) else None,
			 lambda r, s: M(r, s)["ptm"] if M(r, s) else None, fmt="{:+7.4f}")

	print("\nper-run values (the contrasts above are differences of these):")
	cols = [("HA1 cleaved", lambda r, s: chain_plddt(C(r, s), "polymer_one")),
			("HA1 fused", lambda r, s: chain_plddt(M(r, s), "polymer_one")),
			("HA1 alone", lambda r, s: chain_plddt(O(r, s), "one")),
			("HA2 cleaved", lambda r, s: chain_plddt(C(r, s), "polymer_two")),
			("HA2 fused", lambda r, s: chain_plddt(M(r, s), "polymer_two")),
			("HA2 alone", lambda r, s: chain_plddt(T(r, s), "two"))]
	print(f"  {'pLDDT':<14s}" + "".join(f"{s:>9d}" for s in STEPS))
	for name, f in cols:
		print(f"  {name:<14s}" + "".join(
			f"{(f(runs, s) if f(runs, s) is not None else float('nan')):>9.2f}" for s in STEPS))

	print("\ngate and bridge, per step count:")
	print(f"  {'':<14s}" + "".join(f"{s:>9d}" for s in STEPS))
	for name, ref, key, fmtx in (("clashes cleaved", "cleaved", "clashes", "{:>9.0f}"),
								 ("clashes fused", "mature", "clashes", "{:>9.0f}"),
								 ("clashes HA1", "one", "clashes", "{:>9.0f}"),
								 ("clashes HA2", "two", "clashes", "{:>9.0f}"),
								 ("bridge cleaved", "cleaved", "bridge", "{:>9.1f}"),
								 ("bridge fused", "mature", "bridge", "{:>9.1f}")):
		vals = []
		for s in STEPS:
			r = get(runs, ref, s)
			vals.append(fmtx.format(r[key]) if r and key in r else "      n/a")
		print(f"  {name:<14s}" + "".join(vals))


if __name__ == "__main__":
	main()
