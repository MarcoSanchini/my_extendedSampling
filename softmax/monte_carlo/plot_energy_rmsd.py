"""
The figure of docs/energy_rmsd.tex, from energy_rmsd/scores.csv and, when it exists,
energy_rmsd/controls_scores.csv.

  A  RMSD to 1D5R against the number of mutations: the sampled frames against random
     mutants carrying the SAME number of mutations. This is the result.
  B  the same comparison in mean pLDDT.
  C  the share of random mutants still folded, with the fitted midpoint: where chance
     loses the fold, and how broad the crossover is.
  D  the sampled frames alone, on a linear scale, with the decode's own noise drawn to
     scale: inside the ensemble there is almost nothing left to resolve.
  E  the soft contact energy against the RMSD to the reference's own decode -- the
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


def logistic_midpoint(x, y):
	"""Coarse maximum-likelihood fit of P(folded) = 1 / (1 + exp((x - n50) / w)). A grid is
	enough and has no convergence failure to hide: the likelihood is smooth and the grid is
	finer than the 5-mutation spacing of the levels."""
	best = None
	for n50 in np.arange(10.0, 200.0, 0.5):
		for w in np.arange(2.0, 60.0, 0.5):
			p = np.clip(1 / (1 + np.exp((x - n50) / w)), 1e-9, 1 - 1e-9)
			ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
			if best is None or ll > best[0]:
				best = (ll, n50, w)
	return best[1], best[2]


def main():
	ap = argparse.ArgumentParser()
	here = os.path.dirname(os.path.abspath(__file__))
	ap.add_argument("--scores", default=os.path.join(here, "energy_rmsd", "scores.csv"))
	ap.add_argument("--controls", default=os.path.join(here, "energy_rmsd", "controls_scores.csv"))
	ap.add_argument("--out", default=os.path.join(here, "energy_rmsd", "energy_rmsd_fig.png"))
	ap.add_argument("--folded-threshold", type=float, default=3.0,
		help="CA RMSD to 1D5R below which a decode counts as still on the fold, for panel C")
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

	fig, ax = plt.subplots(2, 3, figsize=(15.2, 7.8), layout="constrained")
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
		q = mut.groupby("n")[col].quantile([0.25, 0.75]).unstack()
		axis.fill_between(g.index, q[0.25], q[0.75], color=CONTROL, alpha=0.15, lw=0, zorder=4)
		axis.plot(g.index, g.values, color=CONTROL, lw=2, marker="o", ms=4,
			markeredgecolor="white", markeredgewidth=0.7, zorder=5, label=label)
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
	C = ax[0, 2]
	if c is not None:
		thr = a.folded_threshold
		frac = mut.groupby("n").rmsd_xtal.apply(lambda v: (v < thr).mean())
		k = mut.groupby("n").size()
		# Wilson interval, so a level at 0 or 1 still shows its uncertainty.
		z = 1.96
		ph = frac.values
		n_ = k.values
		centre = (ph + z * z / (2 * n_)) / (1 + z * z / n_)
		half = z / (1 + z * z / n_) * np.sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_))
		C.fill_between(frac.index, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1),
			color=CONTROL, alpha=0.15, lw=0)
		C.plot(frac.index, frac.values, color=CONTROL, lw=2, marker="o", ms=4,
			markeredgecolor="white", markeredgewidth=0.7)
		n50, w = logistic_midpoint(mut.n.values.astype(float), (mut.rmsd_xtal.values < thr).astype(float))
		xs = np.linspace(0, mut.n.max(), 400)
		C.plot(xs, 1 / (1 + np.exp((xs - n50) / w)), color=CONTROL, lw=1, ls=":")
		C.axvline(n50, color=MUTED, lw=1, ls="--")
		C.annotate(f"half the fold lost\nat {n50:.0f} mutations", (n50, 0.55), fontsize=8.5, color=INK,
			xytext=(8, 0), textcoords="offset points")
		C.axhline(1.0, color="#4a90c2", lw=2)
		C.annotate(f"the sampler, to {int(d.Hd.max())} mutations", (d.Hd.max(), 1.0), fontsize=8.5,
			color="#2a6a99", ha="right", xytext=(0, -14), textcoords="offset points")
		C.set_ylim(-0.04, 1.12)
	C.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	C.set_ylabel(f"share with RMSD < {a.folded_threshold:g} Å", fontsize=9.5, color=INK)
	C.set_title("C. Where chance loses the fold", fontsize=10.5, color=INK, loc="left")

	# ---------------------------------------------------------------- D
	D = ax[1, 0]
	D.scatter(d.Hd, d.rmsd_xtal, **dot)
	D.axhline(ref.rmsd_xtal, color=INK, lw=1, ls="--")
	D.annotate(f"the reference itself, {ref.rmsd_xtal:.2f} Å", (d.Hd.min(), ref.rmsd_xtal),
		fontsize=8.5, color=INK, xytext=(2, -12), textcoords="offset points")
	n = d[d.rmsd_xtal_n.notna()] if "rmsd_xtal_n" in d else d.iloc[:0]
	if len(n):
		noise = float(np.median(abs(n.rmsd_xtal - n.rmsd_xtal_n)))
		x0 = d.Hd.min() + 0.06 * (d.Hd.max() - d.Hd.min())
		y0 = d.rmsd_xtal.quantile(0.93)
		D.errorbar([x0], [y0], yerr=[noise / 2], color=CONTROL, capsize=4, lw=2)
		D.annotate(f"decode noise, ±{noise / 2:.3f} Å", (x0, y0), fontsize=8.5,
			color=CONTROL, ha="left", xytext=(9, -4), textcoords="offset points")
	D.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	D.set_ylabel("CA RMSD to 1D5R  [Å]", fontsize=9.5, color=INK)
	D.set_title("D. Inside the ensemble, almost nothing left to resolve", fontsize=10.5,
		color=INK, loc="left")

	# ---------------------------------------------------------------- D
	E = ax[1, 1]
	E.scatter(d.Es_ref, d.rmsd_ref, **dot)
	E.set_xlabel("soft contact energy against the reference's decode", fontsize=9.5, color=INK)
	E.set_ylabel("CA RMSD to the reference's decode  [Å]", fontsize=9.5, color=INK)
	E.set_title(f"E. Where the energy does rank inside the ensemble\nrho "
		f"{spearman(d.Es_ref, d.rmsd_ref):+.2f}, {partial_spearman(d.Es_ref, d.rmsd_ref, d.Hd):+.2f} "
		f"with the mutation count removed", fontsize=10.5, color=INK, loc="left")

	ax[1, 2].axis("off")
	cb = fig.colorbar(sc, ax=ax[1, 2], shrink=0.75, location="left", ticks=[1, 2, 4, 8, 16])
	cb.ax.set_yticklabels(["1", "2", "4", "8", "16"], fontsize=8.5, color=MUTED)
	cb.set_label("sampling temperature T", fontsize=9, color=INK)
	cb.outline.set_visible(False)
	fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
	print("Saved ->", a.out, "(with controls)" if c is not None else "(no controls file found)")


if __name__ == "__main__":
	main()
