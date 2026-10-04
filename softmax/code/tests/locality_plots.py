"""
Figures for tests/test_attention_locality.py.

Kept in its own module, free of torch and of the ESM3 model, so the plotting
can be exercised on synthetic arrays without a GPU -- a plotting bug is
otherwise only discovered after a model run has already been paid for. Every
function here takes plain numpy arrays.

What gets drawn, and why these panels specifically:

  attention_panels()  one figure per probe site, four panels:
      am BEFORE          the attention map of seq_A
      am AFTER           the attention map of seq_B (= seq_A with ONE site
                         changed). Visually near-identical to BEFORE -- that
                         is the point, and the reason the difference panel is
                         needed at all.
      |delta am|         the difference, on a LOG colour scale. Linear would
                         be dominated by the row/column of the mutated site
                         and would show the rest as uniform black, which is
                         precisely the question being asked, so the scale has
                         to be log for the panel to be informative.
      per-pair dU_am     the change in each pair's energy contribution,
                         (log am_ij - log ref_ij)^2, on a SIGNED diverging
                         scale. This is the panel that matters: U_am is a sum
                         over these, so this shows where the energy
                         consequence of a single mutation actually comes from.
      The mutated site is marked with crosshairs on every panel.

  decay_profile()     mean |delta am| against sequence distance from the
                      mutated site, log y. Turns "is it local?" into a curve:
                      a true locality would fall to the numerical floor
                      immediately outside distance 0.

Colour-scale choices are deliberate and are stated in each figure's title, so
a reader is never guessing whether a feature is real or a scaling artifact.
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # headless: no display on the GPU box
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

DPI = 120                        # matches monte_carlo/plot_run.py's convention


def _safe_log_norm(a, floor_pct=50.0):
    """LogNorm needs strictly positive bounds, and a difference array contains
    exact zeros wherever the two maps agree bit-for-bit. Clip to a low
    percentile of the POSITIVE entries so the colour scale is set by real
    structure rather than by the zeros."""
    pos = a[a > 0]
    if pos.size == 0:
        return None, a
    vmin = float(np.percentile(pos, floor_pct))
    vmax = float(a.max())
    if not np.isfinite(vmin) or vmin <= 0 or vmax <= vmin:
        vmin = float(pos.min())
        vmax = max(float(a.max()), vmin*10)
    return LogNorm(vmin=vmin, vmax=vmax), np.clip(a, vmin, None)


def _crosshair(ax, s, L):
    ax.axhline(s, color="lime", linewidth=0.6, alpha=0.9)
    ax.axvline(s, color="lime", linewidth=0.6, alpha=0.9)


def attention_panels(am_before, am_after, dpair, site, out_path,
                     ref_label="reference", title_extra=""):
    """am_before/am_after: (L,L) attention maps. dpair: (L,L) signed per-pair
    change in the energy contribution (upper triangle populated). site: the
    one position that differs."""
    L = am_before.shape[0]
    diff = np.abs(am_after - am_before)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    norm_a, a_before = _safe_log_norm(am_before, floor_pct=0.5)
    im = axes[0, 0].imshow(a_before, norm=norm_a, cmap="viridis", interpolation="nearest")
    axes[0, 0].set_title(f"am BEFORE (seq_A)\nlog colour scale")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    norm_b, a_after = _safe_log_norm(am_after, floor_pct=0.5)
    im = axes[0, 1].imshow(a_after, norm=norm_b, cmap="viridis", interpolation="nearest")
    axes[0, 1].set_title(f"am AFTER (site {site} substituted)\nlog colour scale")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    norm_d, d_clipped = _safe_log_norm(diff, floor_pct=50.0)
    if norm_d is None:
        axes[1, 0].text(0.5, 0.5, "delta am is identically zero",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
    else:
        im = axes[1, 0].imshow(d_clipped, norm=norm_d, cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_title("|delta am|  (LOG scale -- linear would show only the\n"
                         f"row/col of site {site} and hide everything else)")

    # signed per-pair energy change, symmetric about 0 with robust limits so a
    # couple of extreme pairs do not flatten the rest
    full = dpair + dpair.T
    lim = float(np.percentile(np.abs(full[full != 0]), 99)) if np.any(full != 0) else 1.0
    lim = lim if lim > 0 else 1.0
    im = axes[1, 1].imshow(full, cmap="RdBu_r",
                           norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                           interpolation="nearest")
    axes[1, 1].set_title("per-pair change in energy contribution\n"
                         "(log am - log ref)^2, signed; U_am is the sum of this")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    for ax in axes.ravel():
        _crosshair(ax, site, L)
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")

    fig.suptitle(f"Effect of changing ONE site (s={site}) on the attention map"
                 f"{title_extra}\ngreen crosshairs mark site {site}; "
                 f"L={L}, vs {ref_label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def decay_profile(profile_df, site, out_path):
    """profile_df: DataFrame with columns distance, mean_abs_delta, n_entries."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(profile_df["distance"], profile_df["mean_abs_delta"],
            marker="o", linewidth=1.5, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(f"sequence distance from mutated site s={site}"
                  "   (min(|i-s|, |j-s|))")
    ax.set_ylabel("mean |delta am| over entries at that distance")
    ax.set_title(f"How far does a single substitution reach? (site {site})\n"
                 "strict locality would collapse to the numerical floor "
                 "immediately past distance 0")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def local_vs_distal_bar(summary_df, out_path):
    """One bar pair per probe site: share of |delta am| and share of dU_am
    attributable to DISTAL entries (those not involving the mutated site).
    This is the single figure that answers the question outright."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_df))
    w = 0.38
    ax.bar(x - w/2, 100*summary_df["distal_share_abs"], w,
           label="share of total |delta am|", alpha=0.85)
    ax.bar(x + w/2, 100*summary_df["distal_share_dU"], w,
           label="share of total dU_am", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={int(s)}" for s in summary_df["site"]])
    ax.set_ylabel("% contributed by DISTAL entries\n(pairs NOT involving the mutated site)")
    # DISTAL has ~L/2 times more entries than LOCAL, so a large distal SHARE is
    # not by itself surprising. The per-entry ratio is the honest statistic and
    # is annotated on top of each pair.
    if "mean_abs_local" in summary_df and "mean_abs_distal" in summary_df:
        for xi, (ml, md) in enumerate(zip(summary_df["mean_abs_local"],
                                          summary_df["mean_abs_distal"])):
            if md:
                ax.annotate(f"per-entry\nlocal/distal\n={ml/md:.0f}x",
                            (xi, 2), ha="center", va="bottom", fontsize=8,
                            color="dimgray")
    ax.set_title("If a single substitution only changed its own row/column,\n"
                 "both bars would be ~0.  (DISTAL has ~L/2 x more entries than\n"
                 "LOCAL, so also read the per-entry ratio annotated below.)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def write_tables(summary_df, profiles, out_dir):
    """Machine-readable companions to the figures, so the numbers can be
    re-analysed without re-running the model."""
    s_path = os.path.join(out_dir, "locality_summary.csv")
    summary_df.to_csv(s_path, index=False)
    p_path = os.path.join(out_dir, "locality_decay_profiles.csv")
    pd.concat(profiles, ignore_index=True).to_csv(p_path, index=False)
    return s_path, p_path
