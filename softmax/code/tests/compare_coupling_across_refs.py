"""
Cross-reference comparison of test_joint_coupling_N.py results.

Built when test_joint_coupling_N.py was re-run on zero_polymer (566 aa)
after protein G (56 aa) -- see DEVLOG.txt, 2026-10-02 entries. The raw
energies in the two results files are NOT comparable: compute_U_am sums
over the upper triangle of the LxL attention map, and a single-site change
perturbs one row+column of it (~L entries), so dU per changed site grows
roughly linearly in L (measured: 7.9x between L=56 and L=566, vs L ratio
10.1x). Comparing "is coupling worse on the longer protein" therefore
needs scale-free ratios, which the test's own summary table does not
print.

Three ratios, all per N, each dividing mean |coupling| by a different
measure of the move's own size:
  |c|/dU_j     -- what DEVLOG's 2026-09-25 entries used. Inflated when
                  dU_joint has mixed signs across trials (protein G N=2:
                  mean dU_j=15.1 but mean|dU_j|=39.8), so read the next
                  two in preference.
  |c|/|dU_j|   -- same, sign-insensitive.
  |c|/|iso|    -- against the sum of the true isolated single-site
                  effects, i.e. "how much of the joint move is NOT
                  explained by its parts".
plus frac(|c|>|iso|), the fraction of individual trials where coupling
exceeded the isolated sum (a per-trial count, not a ratio of means).

Also reports, since the 2026-10-02 results showed the ratios above
falling mostly because their DENOMINATOR grew rather than because
coupling shrank:
  - the single-site energy scale (mean |dU| at N=1, where coupling is 0
    by construction), which is what sets each protein's dU units;
  - mean dU_joint / (N * single-site scale): the N-site move's cost in
    single-site units, where 1.0 means it costs exactly N independent
    single-site moves and <1 means cancellation between sites helps;
  - mean coupling and frac(coupling<0): whether coupling helps or hurts
    on average (it hurt on protein G, helps on zero_polymer);
  - |mean c|/std c: whether coupling is a systematic, subtractable bias
    (>1) or idiosyncratic per site-set (<1 -- true for both proteins).

Pure stdlib, parses the results files' own per-trial rows -- no torch, no
GPU, no model. Runs anywhere, including a sandbox. From softmax/code:
    python tests/compare_coupling_across_refs.py
    python tests/compare_coupling_across_refs.py A_results.txt B_results.txt
"""
import os
import re
import statistics as st
import sys

ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)"
    r"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)

DEFAULT_FILES = [
    "protein_g/test_joint_coupling_N_results.txt",
    "zero_polymer/test_joint_coupling_N_results.txt",
]


def load(path):
    """Per-trial rows of a test_joint_coupling_N.py results file, plus the
    reference/current sequences from its header. The [N=1 sanity check]
    lines interleaved between rows don't match ROW, so they're skipped."""
    rows, ref, cur = [], None, None
    with open(path) as f:
        for line in f:
            if line.startswith("Reference: "):
                ref = line.split()[1]
            elif line.startswith("Current A: "):
                cur = line.split()[2]
            m = ROW.match(line)
            if m:
                rows.append({
                    "N": int(m[1]), "trial": int(m[2]),
                    "linear": float(m[3]), "isolated": float(m[4]),
                    "joint": float(m[5]), "nonlin": float(m[6]),
                    "coupling": float(m[7]),
                })
    return rows, ref, cur


def per_N(rows):
    out = {}
    for N in sorted({r["N"] for r in rows}):
        R = [r for r in rows if r["N"] == N]
        c = [r["coupling"] for r in R]
        out[N] = {
            "n": len(R),
            "mean_c": st.mean(c),
            "std_c": st.stdev(c) if len(c) > 1 else 0.,
            "mean_abs_c": st.mean(abs(x) for x in c),
            "mean_joint": st.mean(r["joint"] for r in R),
            "mean_abs_joint": st.mean(abs(r["joint"]) for r in R),
            "mean_abs_iso": st.mean(abs(r["isolated"]) for r in R),
            "frac_c_gt_iso": sum(abs(r["coupling"]) > abs(r["isolated"]) for r in R)/len(R),
            "frac_c_neg": sum(x < 0 for x in c)/len(c),
        }
    return out


def safe_div(a, b):
    return a/b if b else float("nan")


def report_one(label, rows, ref, cur):
    s = per_N(rows)
    L = len(ref) if ref else None
    print(f"\n=== {label} ===")
    if L:
        hd = sum(a != b for a, b in zip(ref, cur)) if cur else None
        noncanon = sorted(set(cur) - set("ACDEFGHIKLMNPQRSTVWY")) if cur else []
        print(f"L={L}, upper-triangle attention pairs={L*(L-1)//2}", end="")
        if hd is not None:
            print(f", seq_A Hd={hd}/{L} ({100.*hd/L:.1f}%)", end="")
            if noncanon:
                print(f", non-canonical in seq_A: {noncanon}", end="")
        print()
    print(f"{'N':>3} {'trials':>7} {'mean|c|':>11} {'mean dU_j':>11} {'mean|dU_j|':>11} "
          f"{'mean|iso|':>11} {'|c|/dU_j':>9} {'|c|/|dU_j|':>11} {'|c|/|iso|':>10} {'frac|c|>|iso|':>14}")
    for N, v in s.items():
        print(f"{N:>3} {v['n']:>7} {v['mean_abs_c']:>11.2f} {v['mean_joint']:>11.2f} "
              f"{v['mean_abs_joint']:>11.2f} {v['mean_abs_iso']:>11.2f} "
              f"{safe_div(v['mean_abs_c'], v['mean_joint']):>9.2f} "
              f"{safe_div(v['mean_abs_c'], v['mean_abs_joint']):>11.2f} "
              f"{safe_div(v['mean_abs_c'], v['mean_abs_iso']):>10.2f} "
              f"{v['frac_c_gt_iso']:>14.2f}")

    # N=1 has coupling=0 by construction, so its mean |dU| is a pure
    # single-site energy scale for this protein -- the unit everything
    # below is expressed in.
    scale = s[1]["mean_abs_joint"] if 1 in s else None
    if scale:
        print(f"\nsingle-site energy scale (N=1 mean|dU|) = {scale:.2f}")
        print("cost of an N-site move in single-site units "
              "(mean dU_joint / (N * scale); 1.0 = N independent single-site moves):")
        for N, v in s.items():
            if N == 1:
                continue
            print(f"   N={N:>3}  {v['mean_joint']/(N*scale):>6.2f}")

    print("\ncoupling sign and consistency "
          "(frac c<0 => how often the joint move is LESS uphill than its parts;")
    print(" |mean|/std > 1 would mean a systematic, subtractable bias):")
    for N, v in s.items():
        if N == 1:
            continue
        print(f"   N={N:>3}  mean c={v['mean_c']:>10.2f}  std={v['std_c']:>9.2f}  "
              f"frac c<0={v['frac_c_neg']:>5.2f}  |mean|/std={safe_div(abs(v['mean_c']), v['std_c']):>5.2f}")
    return s


def main():
    paths = sys.argv[1:] or DEFAULT_FILES
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Results file(s) not found: {missing}\n"
              f"Run from softmax/code (paths are relative to it), or pass paths explicitly.",
              file=sys.stderr)
        return 1

    loaded = []
    for p in paths:
        rows, ref, cur = load(p)
        if not rows:
            print(f"No per-trial rows parsed from {p} -- is it a "
                  f"test_joint_coupling_N.py results file?", file=sys.stderr)
            return 1
        label = os.path.basename(os.path.dirname(os.path.abspath(p))) or p
        loaded.append((label, per_N(rows), report_one(label, rows, ref, cur),
                       len(ref) if ref else None))

    if len(loaded) < 2:
        return 0

    # Side-by-side on the two sign-insensitive ratios only: these are the
    # ones meant to be compared across proteins (see module docstring).
    print("\n" + "="*78)
    print("=== side-by-side: the scale-free ratios (lower = coupling matters less) ===")
    labels = [l for l, _, _, _ in loaded]
    width = max(13, max(len(l) for l in labels) + 2)
    for key, title in (("mean_abs_joint", "mean|c| / mean|dU_joint|"),
                       ("mean_abs_iso", "mean|c| / mean|sum_isolated|"),
                       (None, "frac of trials with |c| > |sum_isolated|")):
        print(f"\n{title}:")
        print(f"{'N':>3} " + " ".join(f"{l:>{width}}" for l in labels))
        allN = sorted(set().union(*[set(s) for _, s, _, _ in loaded]))
        for N in allN:
            cells = []
            for _, s, _, _ in loaded:
                if N not in s:
                    cells.append(f"{'--':>{width}}")
                elif N == 1:
                    cells.append(f"{'0 (by constr)':>{width}}")
                elif key is None:
                    cells.append(f"{s[N]['frac_c_gt_iso']:>{width}.2f}")
                else:
                    cells.append(f"{safe_div(s[N]['mean_abs_c'], s[N][key]):>{width}.2f}")
            print(f"{N:>3} " + " ".join(cells))

    print("\nRead these DOWN a column for the coupling-vs-N trend within one protein,")
    print("and ACROSS a row for whether coupling matters more or less on the other")
    print("reference at the same N. Raw energies are not comparable across columns")
    print("(U_am's per-site scale grows ~linearly in L); these ratios are.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
