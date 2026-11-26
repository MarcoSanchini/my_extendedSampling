"""
The figure of docs/energy_rmsd.tex, from energy_rmsd/scores.csv and, when it exists,
energy_rmsd/controls_scores.csv.

  A  RMSD to 1D5R against the number of mutations: the sampled frames against random
     mutants carrying the SAME number of mutations. This is the result.
  B  the same comparison in mean pLDDT.
  C  the sampled frames alone, on a linear scale, with the decode's own noise drawn to
     scale: inside the ensemble there is almost nothing left to resolve.
  D  the soft contact energy against the RMSD to the reference's own decode -- the
     coordinate that does have range inside the ensemble.

Usage:  python plot_energy_rmsd.py [--scores ...] [--controls ...] [--out ...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_vs_rmsd import spearman, partial_spearman

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e8e8e8"
# Two series that must never be confused: one cool (what the sampler produced), one warm
# (what chance produces). The sampled series additionally carries temperature as a
# single-hue sequential ramp; the control series is one flat accent, never a ramp.
CONTROL = "#c2410c"


def main():
	ap = argparse.ArgumentParser()
	here = os.path.dirname(os.path.abspath(__file__))
	ap.add_argument("--scores", default=os.path.join(here, "energy_rmsd", "scores.csv"))
	ap.add_argument("--controls", default=os.path.join(here, "energy_rmsd", "controls_scores.csv"))
	ap.add_argument("--out", default=os.path.join(here, "energy_rmsd", "energy_rmsd_fig.png"))
	a = ap.parse_args()

	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm, ListedColormap
	from matplotlib.lines import Line2D

	s = pd.read_csv(a.scores)
	ref = s[s.frame_id == "reference"].iloc[0]
	d = s[s.frame_id != "reference"].copy()
	d["T"] = d["T"].astype(float)

	c = None
	if os.path.exists(a.controls):
		c = pd.read_csv(a.controls)
		c = c[c.frame_id != "reference"]
		mut = c[c.label.str.startswith("mut")].copy()
		mut["n"] = mut.label.str[3:].astype(int)
		mut = mut.sort_values("n")
		other = c[~c.label.str.startswith("mut")]

	cmap = ListedColormap(matplotlib.colormaps["Blues"](np.linspace(0.30, 0.95, 256)))
	norm = LogNorm(vmin=d["T"].min(), vmax=d["T"].max())
	dot = dict(c=d["T"], cmap=cmap, norm=norm, s=5, linewidths=0, alpha=0.65)

	fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.8), layout="constrained")
	for x in ax.ravel():
		x.tick_params(labelsize=8.5, colors=MUTED)
		for sp in ("top", "right"):
			x.spines[sp].set_visible(False)
		for sp in ("left", "bottom"):
			x.spines[sp].set_color(MUTED)
		x.grid(axis="y", color=GRID, lw=0.6)
		x.set_axisbelow(True)

	def control_curve(axis, col, label):
		"""Median of the random mutants at each mutation level, plus the composition and
		uniform-random points, drawn as one warm series against the cool sampled cloud."""
		if c is None:
			return
		g = mut.groupby("n")[col].median()
		axis.plot(g.index, g.values, color=CONTROL, lw=2, marker="o", ms=5,
			markeredgecolor="white", markeredgewidth=0.8, zorder=5, label=label)
		for lab, mark in (("shuf", "s"), ("rand", "^")):
			sub = other[other.label == lab]
			if len(sub):
				axis.scatter(sub.Hd.median(), sub[col].median(), color=CONTROL, marker=mark, s=55,
					edgecolors="white", linewidths=0.8, zorder=6)

	# ---------------------------------------------------------------- A
	A = ax[0, 0]
	sc = A.scatter(d.Hd, d.rmsd_xtal, **dot)
	control_curve(A, "rmsd_xtal", "random mutants")
	A.set_yscale("log")
	A.set_yticks([2, 3, 5, 10, 20])
	A.set_yticklabels(["2", "3", "5", "10", "20"])
	A.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())   # else log minor ticks print 6x10^0
	A.axhline(ref.rmsd_xtal, color=INK, lw=1, ls="--", zorder=1)
	A.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	A.set_ylabel("CA RMSD to 1D5R  [Å]   (log)", fontsize=9.5, color=INK)
	A.set_title("A. The sampler keeps the fold where chance destroys it", fontsize=10.5,
		color=INK, loc="left")
	if c is not None:
		A.legend(handles=[
			Line2D([], [], marker="o", color=CONTROL, lw=2, ms=5, label="random mutants, same count"),
			Line2D([], [], marker="s", color=CONTROL, lw=0, ms=7, label="shuffled reference"),
			Line2D([], [], marker="^", color=CONTROL, lw=0, ms=7, label="uniform random"),
			Line2D([], [], marker="o", color="#4a90c2", lw=0, ms=5, label="sampled frames"),
		], fontsize=8, frameon=False, loc="center right", labelcolor=INK)

	# ---------------------------------------------------------------- B
	B = ax[0, 1]
	B.scatter(d.Hd, d.plddt, **dot)
	control_curve(B, "plddt", "random mutants")
	B.axhline(ref.plddt, color=INK, lw=1, ls="--")
	B.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	B.set_ylabel("mean pLDDT of the decode", fontsize=9.5, color=INK)
	B.set_title("B. The model's own confidence says the same", fontsize=10.5, color=INK, loc="left")

	# ---------------------------------------------------------------- C
	C = ax[1, 0]
	C.scatter(d.Hd, d.rmsd_xtal, **dot)
	C.axhline(ref.rmsd_xtal, color=INK, lw=1, ls="--")
	C.annotate(f"the reference itself, {ref.rmsd_xtal:.2f} Å", (d.Hd.min(), ref.rmsd_xtal),
		fontsize=8.5, color=INK, xytext=(2, -12), textcoords="offset points")
	n = d[d.rmsd_xtal_n.notna()] if "rmsd_xtal_n" in d else d.iloc[:0]
	if len(n):
		noise = float(np.median(abs(n.rmsd_xtal - n.rmsd_xtal_n)))
		x0 = d.Hd.min() + 0.06 * (d.Hd.max() - d.Hd.min())
		y0 = d.rmsd_xtal.quantile(0.93)
		C.errorbar([x0], [y0], yerr=[noise / 2], color=CONTROL, capsize=4, lw=2)
		C.annotate(f"decode noise, ±{noise / 2:.3f} Å", (x0, y0), fontsize=8.5,
			color=CONTROL, ha="left", xytext=(9, -4), textcoords="offset points")
	C.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	C.set_ylabel("CA RMSD to 1D5R  [Å]", fontsize=9.5, color=INK)
	C.set_title("C. Inside the ensemble, almost nothing left to resolve", fontsize=10.5,
		color=INK, loc="left")

	# ---------------------------------------------------------------- D
	D = ax[1, 1]
	D.scatter(d.Es_ref, d.rmsd_ref, **dot)
	D.set_xlabel("soft contact energy against the reference's decode", fontsize=9.5, color=INK)
	D.set_ylabel("CA RMSD to the reference's decode  [Å]", fontsize=9.5, color=INK)
	D.set_title(f"D. Where the energy does rank inside the ensemble\nrho "
		f"{spearman(d.Es_ref, d.rmsd_ref):+.2f}, {partial_spearman(d.Es_ref, d.rmsd_ref, d.Hd):+.2f} "
		f"with the mutation count removed", fontsize=10.5, color=INK, loc="left")

	cb = fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.55, pad=0.015, ticks=[1, 2, 4, 8, 16])
	cb.ax.set_yticklabels(["1", "2", "4", "8", "16"], fontsize=8.5, color=MUTED)
	cb.set_label("sampling temperature T", fontsize=9, color=INK)
	cb.outline.set_visible(False)
	fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
	print("Saved ->", a.out, "(with controls)" if c is not None else "(no controls file found)")


if __name__ == "__main__":
	main()
