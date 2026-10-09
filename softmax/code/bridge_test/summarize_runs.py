"""
Tabulate every test_cleaved_complex.py result file in this directory.

Written 2026-10-09 after the step-matched HA1/HA2 runs, for two reasons. The
numbers in the devlogs and in docs/bridge_test_report.tex were being copied by
hand from result files, which is exactly how a transcription error gets in;
and a step sweep produces too many files to compare by eye.

It reads the CONTENT of each file, not its name -- the filenames are
inconsistent (test_cleaved_complex_results.txt has held three different runs)
-- and prints one row per run. Files that are not result files (the devlogs,
a run that crashed before printing a header) are skipped, and listed so a
silent skip cannot hide a missing result.

    python bridge_test/summarize_runs.py            # from softmax/code
    python bridge_test/summarize_runs.py --csv      # machine-readable
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RE_REF = re.compile(r"#\s*reference:\s*(?:zero_polymer_)?(\S+)\s*\(L=(\d+)")
RE_STEPS = re.compile(r"#\s*decoding steps:\s*(\d+)")
RE_PTM = re.compile(r"^\s*ptm\s+([\d.]+)")
RE_PL_ALL = re.compile(r"plddt mean \(all residues\)\s+([\d.]+)")
RE_PL_CHAIN = re.compile(r"plddt mean (\S+)\s+([\d.]+)\s+\(\s*([\d.]+)% of residues")
RE_FRAC = re.compile(r"residues >= 70\s+([\d.]+)%")
RE_MINCB = re.compile(r"min CB-CB \(\|i-j\|>=2\)\s+([\d.]+)")
RE_CLASH = re.compile(r"CB-CB clashes\s+(\d+)")
RE_BRIDGE = re.compile(r"^\s*CB-CB\s+([\d.]+) A\s+\(bonded")
RE_ANCHOR = re.compile(r"plddt at the two anchors:\s*([\d.]+)\s*/\s*([\d.]+)")
RE_PAIR = re.compile(r"^\s*1\.\s*[*! ]\s*([\d.]+) A\s+(\S+)\s+--\s+(\S+)")
RE_CLAMP = re.compile(r"#\s*note:\s*--steps\s+(\d+)\s+exceeds")


def parse(path: Path) -> dict | None:
	text = path.read_text(errors="replace")
	ref = RE_REF.search(text)
	if not ref:
		return None
	row = {"file": path.name, "ref": ref.group(1), "L": int(ref.group(2))}
	steps = RE_STEPS.search(text)
	row["steps"] = int(steps.group(1)) if steps else None
	clamp = RE_CLAMP.search(text)
	row["asked"] = int(clamp.group(1)) if clamp else row["steps"]
	chains = {}
	for line in text.splitlines():
		for key, rx in (("ptm", RE_PTM), ("plddt", RE_PL_ALL), ("frac70", RE_FRAC),
						("mincb", RE_MINCB), ("clashes", RE_CLASH)):
			m = rx.search(line)
			if m and key not in row:
				row[key] = float(m.group(1))
		m = RE_PL_CHAIN.search(line)
		if m:
			chains[m.group(1)] = (float(m.group(2)), float(m.group(3)))
		m = RE_BRIDGE.search(line)
		if m and "bridge" not in row:
			row["bridge"] = float(m.group(1))
		m = RE_ANCHOR.search(line)
		if m:
			row["anchors"] = f"{m.group(1)}/{m.group(2)}"
		m = RE_PAIR.search(line)
		if m and "closest" not in row:
			row["closest"] = f"{m.group(1)} {m.group(2)}--{m.group(3)}"
	row["chains"] = chains
	return row


def chain_cell(row: dict, name: str) -> str:
	c = row["chains"].get(name)
	return f"{c[0]:.2f} ({c[1]:.1f}%)" if c else "-"


def main() -> None:
	rows, skipped = [], []
	for path in sorted(HERE.glob("*.txt")):
		row = parse(path)
		(rows if row else skipped).append(row or path.name)

	rows.sort(key=lambda r: (r["ref"], r["steps"] or 0, r["file"]))
	cols = ["ref", "L", "steps", "ptm", "plddt", "frac70", "mincb", "clashes", "bridge"]

	if "--csv" in sys.argv:
		print(",".join(cols + ["one", "two", "anchors", "closest", "file"]))
		for r in rows:
			print(",".join([str(r.get(c, "")) for c in cols]
						   + [chain_cell(r, "polymer_one"), chain_cell(r, "polymer_two"),
							  r.get("anchors", ""), r.get("closest", ""), r["file"]]))
		return

	hdr = (f"{'ref':<10}{'L':>4}{'steps':>6}{'ptm':>8}{'plddt':>7}{'%>=70':>7}"
		   f"{'minCB':>7}{'clash':>6}{'bridge':>8}  {'HA1 plddt (%>=70)':<19}"
		   f"{'HA2 plddt (%>=70)':<19}closest Cys pair")
	print(hdr)
	print("-" * len(hdr))
	for r in rows:
		one = chain_cell(r, "polymer_one") if r["ref"] in ("cleaved", "mature") else (
			chain_cell(r, "one") if r["ref"] == "one" else "-")
		two = chain_cell(r, "polymer_two") if r["ref"] in ("cleaved", "mature") else (
			chain_cell(r, "two") if r["ref"] == "two" else "-")
		clamp = "" if r["asked"] == r["steps"] else f"  [asked {r['asked']}, clamped]"
		print(f"{r['ref']:<10}{r['L']:>4}{r['steps'] or 0:>6}"
			  f"{r.get('ptm', float('nan')):>8.4f}{r.get('plddt', float('nan')):>7.2f}"
			  f"{r.get('frac70', float('nan')):>7.1f}{r.get('mincb', float('nan')):>7.2f}"
			  f"{int(r.get('clashes', -1)):>6}"
			  f"{r.get('bridge', float('nan')):>8.2f}  {one:<19}{two:<19}"
			  f"{r.get('closest', '-')}{clamp}")

	if skipped:
		print(f"\nskipped (no '# reference:' header): {', '.join(skipped)}")


if __name__ == "__main__":
	main()
