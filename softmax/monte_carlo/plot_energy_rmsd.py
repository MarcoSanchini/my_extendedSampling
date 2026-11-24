"""
The four panels of docs/energy_rmsd.tex, from energy_rmsd/scores.csv.

  A  RMSD to 1D5R against the number of mutations. The headline: it barely moves.
  B  mean pLDDT of the same decodes, which does move, smoothly.
  C  the soft contact energy against the RMSD to the REFERENCE's own decode -- the
     structural coordinate that has range, and the one the energy is defined against.
  D  the decode-noise floor: the same 50 frames decoded at 2 and at 8 steps.

Usage:  python plot_energy_rmsd.py [--scores energy_rmsd/scores.csv] [--out energy_rmsd/energy_rmsd_fig.png]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_vs_rmsd import spearman, partial_spearman

INK, MUTED = "#1a1a1a", "#6b6b6b"


def main():
	ap = argparse.ArgumentParser()
	here = os.path.dirname(os.path.abspath(__file__))
	ap.add_argument("--scores", default=os.path.join(here, "energy_rmsd", "scores.csv"))
	ap.add_argument("--out", default=os.path.join(here, "energy_rmsd", "energy_rmsd_fig.png"))
	a = ap.parse_args()

	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.colors import LogNorm, ListedColormap

	s = pd.read_csv(a.scores)
	ref = s[s.frame_id == "reference"].iloc[0]
	d = s[s.frame_id != "reference"].copy()
	d["T"] = d["T"].astype(float)
	# One hue, light to dark, for an ordered quantity: T. Never a rainbow.
	cmap = ListedColormap(matplotlib.colormaps["Blues"](np.linspace(0.30, 0.95, 256)))
	norm = LogNorm(vmin=d["T"].min(), vmax=d["T"].max())
	dot = dict(c=d["T"], cmap=cmap, norm=norm, s=5, linewidths=0, alpha=0.65)

	fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6), layout="constrained")
	for x in ax.ravel():
		x.tick_params(labelsize=8.5, colors=MUTED)
		for sp in ("top", "right"):
			x.spines[sp].set_visible(False)
		for sp in ("left", "bottom"):
			x.spines[sp].set_color(MUTED)
		x.grid(axis="y", color="#e8e8e8", lw=0.6)
		x.set_axisbelow(True)

	A = ax[0, 0]
	sc = A.scatter(d.Hd, d.rmsd_xtal, **dot)
	A.axhline(ref.rmsd_xtal, color=INK, lw=1.2, ls="--")
	A.annotate(f"the reference itself, {ref.rmsd_xtal:.2f} Å", (d.Hd.min(), ref.rmsd_xtal), fontsize=8.5,
		color=INK, xytext=(2, -11), textcoords="offset points")
	A.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	A.set_ylabel("CA RMSD to 1D5R  [Å]", fontsize=9.5, color=INK)
	A.set_title(f"A. The decode never leaves the fold\nmedian {d.rmsd_xtal.median():.2f} Å, "
		f"95% below {d.rmsd_xtal.quantile(.95):.2f}, over 10–{d.Hd.max()} mutations",
		fontsize=10, color=INK, loc="left")

	B = ax[0, 1]
	B.scatter(d.Hd, d.plddt, **dot)
	B.axhline(ref.plddt, color=INK, lw=1.2, ls="--")
	B.set_xlabel("mutations from the reference", fontsize=9.5, color=INK)
	B.set_ylabel("mean pLDDT of the decode", fontsize=9.5, color=INK)
	B.set_title("B. Confidence does fall, smoothly\n"
		f"{d.groupby('T').plddt.median().iloc[0]:.0f} at T=1 down to "
		f"{d.groupby('T').plddt.median().iloc[-1]:.0f} at T=16", fontsize=10, color=INK, loc="left")

	C = ax[1, 0]
	C.scatter(d.Es_ref, d.rmsd_ref, **dot)
	C.set_xlabel("soft contact energy against the reference's decode", fontsize=9.5, color=INK)
	C.set_ylabel("CA RMSD to the reference's decode  [Å]", fontsize=9.5, color=INK)
	C.set_title(f"C. Against a coordinate that has range\nrho {spearman(d.Es_ref, d.rmsd_ref):+.2f}, "
		f"{partial_spearman(d.Es_ref, d.rmsd_ref, d.Hd):+.2f} with the mutation count removed",
		fontsize=10, color=INK, loc="left")

	D = ax[1, 1]
	n = d[d.rmsd_xtal_n.notna()]
	lim = [min(n.rmsd_xtal.min(), n.rmsd_xtal_n.min()) - 0.03, max(n.rmsd_xtal.max(), n.rmsd_xtal_n.max()) + 0.03]
	D.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=0)
	D.scatter(n.rmsd_xtal, n.rmsd_xtal_n, c=n["T"], cmap=cmap, norm=norm, s=22, linewidths=0.5,
		edgecolors="white")
	D.set_xlim(lim); D.set_ylim(lim)
	D.set_xlabel("RMSD to 1D5R, 2 decode steps  [Å]", fontsize=9.5, color=INK)
	D.set_ylabel("the same frame at 8 steps  [Å]", fontsize=9.5, color=INK)
	iqr = n.rmsd_xtal.quantile(.75) - n.rmsd_xtal.quantile(.25)
	D.set_title(f"D. And the noise eats much of what range is left\nmedian shift "
		f"{np.median(abs(n.rmsd_xtal - n.rmsd_xtal_n)):.3f} Å = "
		f"{np.median(abs(n.rmsd_xtal - n.rmsd_xtal_n)) / iqr:.0%} of the spread it must resolve",
		fontsize=10, color=INK, loc="left")

	cb = fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.6, pad=0.015, ticks=[1, 2, 4, 8, 16])
	cb.ax.set_yticklabels(["1", "2", "4", "8", "16"], fontsize=8.5, color=MUTED)
	cb.set_label("sampling temperature T", fontsize=9, color=INK)
	cb.outline.set_visible(False)
	fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
	print("Saved ->", a.out)


if __name__ == "__main__":
	main()
