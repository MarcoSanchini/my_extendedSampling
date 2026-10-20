#!/usr/bin/env python
"""
geodesic_esm3.py -- energy along the 19 linear geodesics at one site of protein G, in ESM3.

Companion code to the paper "geodesic_aa_ranking". It uses the custom ESM3 of the
extendedSampling repository (customs/custom_esm), imported as in
softmax/monte_carlo/main_rate.py and loaded as the samplers do
(softmax/code/classes/rate_sampler.py):

    model = ESM3.from_pretrained("esm3-open")

custom_esm/pretrained.py casts the weights to float16 (and from_pretrained to bfloat16 on
cuda); this script then casts the model to float32 and, by default, runs without autocast.
The weights are therefore the same as the sampler's, but the arithmetic is float32, whereas
model.predict_attention (what the sampler calls) runs under bf16 autocast on cuda. Pass
--autocast to reproduce the sampler's arithmetic.

Setting (notation of the paper)
  reference sigma_ref  protein G, 56 aa, the target of the extendedSampling runs;
  start sigma^(1)      sigma_ref with --muts random substitutions, always including --site,
                       so the wild-type residue of the site is one of the 19 targets;
  path to mutant beta  pi(gamma) = (1 - gamma) pi^(1) + gamma pi^(beta): only the row of
                       the site moves (Section 3, eq. 3);
  energy (--energy)    on the residue block of A (BOS and EOS removed, as clean_and_symm does):
                       log2  eq. (10), the sampler's U_am: with S = A + A^T,
                             U = sum_{i<=j} (log S_ij - log S^ref_ij)^2, computed by
                             softmax/code/utils/energies.py:compute_U_am on clean_and_symm(A) = S/2
                             (the factor 1/2 cancels in the log-ratio);
                       ce    cross entropy relative to the reference, U = sum_{i!=j} A^ref_ij
                             log(A^ref_ij / A_ij), on the raw (not symmetrized) attention.
                             It differs from -sum A^ref log A by a constant, so slopes and Delta U
                             are those of the cross entropy. The residue block without its
                             diagonal is not normalized (BOS, EOS and A_ii take part of each row),
                             so U is not minimal at the reference and can be negative.

Attention maps
  full        the map the sampler uses, model.predict_attention: mean over the heads of the
              LAST layer, after the soft sequence has gone through the whole transformer.
  head        one head (--head) of the FIRST layer. Its input, the sum of the track
              embeddings, is affine in pi; the LayerNorm before the q/k/v projection and the
              q/k LayerNorms are not. With their statistics frozen at sigma^(1) -- exact at
              every position except the mutated one -- q and k are affine in pi, and the
              attention along the path is given exactly by the closed form of Section 3
              (eq. 7, factors f_ia and F). The script uses that closed form and checks it
              against a direct softmax. This is the first-layer head "before linearity is
              disrupted".
  head_exact  the same head with the true LayerNorms, for comparison.

Outputs, in --outdir (default geodesic_esm3_out/ for log2, geodesic_esm3_out_ce/ for ce)
  geodesics_<map>.csv      U(gamma) on the 19 paths (rows: gamma, columns: mutants)
  geodesics_<map>.png      the same, plotted with pandas
  ranking_<map>.csv        slope g'(0), Delta U, ranks and monotonicity, per mutant
  slope_vs_dU_<map>.png    g'(0) against Delta U, one point per mutant
  summary.txt              ranking statistics and the discordant pairs, per map

Run (a GPU is strongly recommended; the model is cast to float32, about 6 GB):
  python geodesic_esm3.py --selftest                  # checks against custom_esm, then exits
  python geodesic_esm3.py --site 43 --muts 3 --seed 0
  python geodesic_esm3.py --site 43 --muts 3 --seed 0 --energy ce
custom_esm is taken from <checkout>/customs and utils.energies from <checkout>/softmax/code,
where <checkout> is found by find_repo_root (or set with EXTSAMPLING_ROOT), so the script
runs from any folder and also as a copy outside the checkout.
"""
import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def find_repo_root():
    """The extendedSampling checkout: the folder holding customs/custom_esm and softmax/code.

    The paths are not hard-wired relative to this file, so a copy of the script placed
    elsewhere (e.g. next to the checkout) still finds them. Looked up, in order:
      $EXTSAMPLING_ROOT; this file's folder and its parents; the current folder and its
      parents; and, at each of those levels, the subfolders one level down (so
      ~/my_project/geodesic_esm3.py finds ~/my_project/extendedSampling).
    """
    def ok(d):
        return (d / "customs" / "custom_esm" / "__init__.py").is_file() and \
               (d / "softmax" / "code" / "utils" / "energies.py").is_file()

    if os.environ.get("EXTSAMPLING_ROOT"):
        d = Path(os.environ["EXTSAMPLING_ROOT"]).expanduser().resolve()
        if ok(d):
            return d
        raise SystemExit(f"EXTSAMPLING_ROOT={d} has no customs/custom_esm or softmax/code")
    seen = []
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if d in seen:
                continue
            seen.append(d)
            if ok(d):
                return d
            with contextlib.suppress(OSError):
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and not sub.name.startswith(".") and ok(sub):
                        return sub
    raise SystemExit(
        "geodesic_esm3.py: cannot find the extendedSampling checkout (customs/custom_esm and "
        "softmax/code) above this script or the current folder; set EXTSAMPLING_ROOT to it."
    )


REPO_ROOT = find_repo_root()
CODE_DIR = str(REPO_ROOT / "softmax" / "code")
CUSTOMS_DIR = str(REPO_ROOT / "customs")

sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CUSTOMS_DIR)

from custom_esm.models.esm3 import ESM3  # noqa: E402
import custom_esm.utils.constants.esm3 as C  # noqa: E402
from custom_esm.utils.custom import clean_and_symm  # noqa: E402
from utils.energies import compute_U_am  # noqa: E402

PROTEIN_G = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
STANDARD = "LAGVSERTIDPKQNFYMHWC"     # the 20 standard residues, in SEQUENCE_USED_VOCAB order


# --------------------------------------------------------------------------- helpers
def symm(A):
    """(A + A^T)/2 of a residue block: clean_and_symm without the trimming."""
    return (A + A.transpose(-1, -2)) / 2


def U_log2(A, A_ref):
    """Eq. (10), the sampler's U_am: utils/energies.py:compute_U_am on clean_and_symm(A)."""
    return compute_U_am(symm(A), symm(A_ref))


def U_ce(A, A_ref):
    """sum_{i!=j} A^ref_ij log(A^ref_ij / A_ij) on the raw residue block."""
    off = ~torch.eye(A.shape[-1], dtype=torch.bool, device=A.device)
    return (A_ref * (torch.log(A_ref) - torch.log(A)))[off].sum()


ENERGIES = {"log2": U_log2, "ce": U_ce}


def layer_norm(mod, x, stats=None):
    """nn.LayerNorm, optionally with frozen statistics (mean, var) -> an affine map of x."""
    if isinstance(mod, torch.nn.Identity):
        return x
    if stats is None:
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
    else:
        mu, var = stats
    y = (x - mu) / torch.sqrt(var + mod.eps)
    if mod.weight is not None:
        y = y * mod.weight
    if mod.bias is not None:
        y = y + mod.bias
    return y


def stats_of(x):
    return (x.mean(-1, keepdim=True).detach(), x.var(-1, keepdim=True, unbiased=False).detach())


def is_monotone(curve, slope0):
    """Monotone on [0,1]: grid differences of one sign, and the exact slope at 0 the same way."""
    d = np.diff(curve)
    return bool((np.all(d >= 0) and slope0 > 0) or (np.all(d <= 0) and slope0 < 0))


def spearman(x, y):
    rx = pd.Series(np.asarray(x)).rank().to_numpy(); ry = pd.Series(np.asarray(y)).rank().to_numpy()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


def make_start(ref, site0, muts, seed):
    """sigma^(1): ref with `muts` random substitutions, the first one at site0."""
    rng = np.random.default_rng(seed)
    if muts <= 0:
        return ref, []
    others = [i for i in range(len(ref)) if i != site0]
    sites = [site0] + [int(i) for i in rng.choice(others, size=muts - 1, replace=False)]
    seq = list(ref)
    for i in sites:
        choices = [aa for aa in STANDARD if aa != ref[i]]
        seq[i] = choices[int(rng.integers(len(choices)))]
    return "".join(seq), sorted(sites)


# --------------------------------------------------------------------------- model wrapper
class Esm3Maps:
    """Attention maps of ESM3 as functions of the soft sequence pi (shape L x K)."""

    def __init__(self, model, C, clean_and_symm, layer, head, autocast):
        self.m, self.C, self.cs = model, C, clean_and_symm
        self.layer, self.head = layer, head
        self.attn = model.transformer.blocks[layer].attn
        self.autocast = autocast
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.K = len(C.SEQUENCE_USED_VOCAB)

    def ctx(self):
        if self.autocast and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def one_hot(self, seq):
        idx = torch.tensor([self.C.SEQUENCE_USED_VOCAB.index(aa) for aa in seq], device=self.device)
        return torch.nn.functional.one_hot(idx, num_classes=self.K).to(self.dtype)

    # -- whole transformer: the map of model.predict_attention (last layer, mean over heads),
    #    as the raw residue block (BOS and EOS removed, not symmetrized)
    def full_map(self, probs):
        with self.ctx():
            A = self.m._last_attention(sequence_probs=probs.unsqueeze(0))
        return A.float()[0, 1:-1, 1:-1]

    # -- input of the chosen block (layer 0: the sum of the track embeddings, affine in pi)
    def block_input(self, probs):
        L = probs.shape[0] + 2                                    # BOS and EOS
        default_tokens, affine, affine_mask = self.m._default(L, probs.device)
        x = self.m.encoder.custom_forward(probs.unsqueeze(0), *default_tokens)
        if self.layer > 0:
            chain_id = torch.ones(x.shape[:-1], dtype=torch.int64, device=x.device)
            for block in self.m.transformer.blocks[: self.layer]:
                x = block(x, None, affine, affine_mask, chain_id)
        return x

    def frozen_stats(self, x):
        """LayerNorm statistics at sigma^(1), per position: makes q and k affine in x."""
        ln, lin = self.attn.layernorm_qkv[0], self.attn.layernorm_qkv[1]
        st = {"qkv": stats_of(x)}
        q, k, _ = torch.chunk(lin(layer_norm(ln, x)), 3, dim=-1)
        st["q"], st["k"] = stats_of(q), stats_of(k)
        return st

    def qk(self, x, frozen=None):
        """Queries and keys of every head, (1, L+2, H, d_head); q carries the 1/sqrt(d_head)."""
        ln, lin = self.attn.layernorm_qkv[0], self.attn.layernorm_qkv[1]
        h = layer_norm(ln, x, None if frozen is None else frozen["qkv"])
        q, k, _ = torch.chunk(lin(h), 3, dim=-1)
        q = layer_norm(self.attn.q_ln, q, None if frozen is None else frozen["q"])
        k = layer_norm(self.attn.k_ln, k, None if frozen is None else frozen["k"])
        q, k = self.attn._apply_rotary(q, k)
        H, dh = self.attn.n_heads, self.attn.d_head
        return q.unflatten(-1, (H, dh)) / math.sqrt(dh), k.unflatten(-1, (H, dh))

    @staticmethod
    def softmax_attention(q, k):
        """(1, L+2, H, d) -> attention of every head, (1, H, L+2, L+2)."""
        return torch.softmax(torch.einsum("blhd,bmhd->bhlm", q, k), dim=-1)

    def head_map(self, probs, frozen=None):
        with self.ctx():
            q, k = self.qk(self.block_input(probs), frozen)
            A = self.softmax_attention(q, k)[:, self.head]
        return A.float()[0, 1:-1, 1:-1]


def closed_form_path(q1, k1, q2, k2, a, gammas):
    """Eq. (7) of Section 3 for one head: q*, k* are (n, d) at sigma^(1) and at the mutant,
    with q already scaled, a the row of the mutated site (BOS included). Returns (G, n, n)."""
    A1 = torch.softmax(q1 @ k1.T, dim=-1)
    A2 = torch.softmax(q2 @ k2.T, dim=-1)
    dq, dk = q2[a] - q1[a], k2[a] - k1[a]
    v = q1 @ dk                                          # q_i^(1) . Delta k
    c = float(dq @ dk)                                   # Delta q . Delta k
    out = []
    for g in gammas:
        g = float(g)
        W = A1.clone()
        W[:, a] = W[:, a] * torch.exp(g * v)             # f_ia(gamma), rows i != a
        row = A1[a] ** (1 - g) * A2[a] ** g              # row a: A1^(1-g) A2^g
        row[a] = row[a] * math.exp(-g * (1 - g) * c)     # F(gamma)
        W[a] = row
        out.append(W / W.sum(-1, keepdim=True))
    return torch.stack(out)


# --------------------------------------------------------------------------- experiment
def path_probs(p0, site0, beta_idx, gamma):
    p = p0.clone()
    p[site0] = (1 - gamma) * p0[site0]
    p[site0, beta_idx] = p[site0, beta_idx] + gamma
    return p


def slopes_by_backward(maps, p0, site0, alpha_idx, targets, energy_fn):
    """All 19 first-order terms from ONE backward pass: g'_beta(0) = h_a,beta - h_a,alpha."""
    p = p0.clone().requires_grad_(True)
    U = energy_fn(p)
    U.backward()
    h = p.grad[site0].double().cpu()
    return {t: float(h[i] - h[alpha_idx]) for t, i in targets}, float(U.detach())


def run_map(name, maps, p0, site0, alpha_idx, targets, gammas, energy_fn, curve_fn):
    t0 = time.time()
    slopes, U0 = slopes_by_backward(maps, p0, site0, alpha_idx, targets, energy_fn)
    curves = {}
    with torch.no_grad():
        for t, i in targets:
            curves[t] = curve_fn(i)
    df = pd.DataFrame(curves, index=pd.Index(np.round(gammas, 6), name="gamma"))
    print(f"  [{name}] U(sigma1) = {U0:.4f}, {len(targets)} paths in {time.time() - t0:.1f} s")
    return df, pd.Series(slopes)


def ranking_table(df, slopes):
    dU = df.iloc[-1] - df.iloc[0]
    tab = pd.DataFrame({"slope": slopes, "dU": dU, "U_end": df.iloc[-1]})
    tab["rank_by_slope"] = tab["slope"].rank(method="min").astype(int)   # 1 = steepest descent
    tab["rank_by_dU"] = tab["dU"].rank(method="min").astype(int)         # 1 = largest decrease
    tab["monotone"] = pd.Series({c: is_monotone(df[c].to_numpy(), slopes[c]) for c in df.columns})
    return tab.sort_values("rank_by_dU")


def ranking_summary(name, df, tab, wt_label):
    s, d = tab["slope"], tab["dU"]
    labels = list(tab.index)
    tot = ok = 0
    discordant, disc_mono, diffmono = [], [], [0, 0]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            if d[a] == d[b]:
                continue
            tot += 1
            good = bool(np.sign(s[a] - s[b]) == np.sign(d[a] - d[b]))
            ok += good
            phi = df[a].to_numpy() - df[b].to_numpy()
            if is_monotone(phi, s[a] - s[b]):
                diffmono[0] += 1; diffmono[1] += good
            if not good:
                discordant.append((a, b))
                if tab.loc[a, "monotone"] and tab.loc[b, "monotone"]:
                    disc_mono.append((a, b))
    lines = [f"== {name} ==",
             f"pairs ordered correctly by g'(0): {ok}/{tot} = {ok / max(tot, 1):.3f} (0.5 is chance)",
             f"Spearman(g'(0), Delta U) = {spearman(s, d):+.3f}",
             f"best mutant by Delta U: {tab.index[0]}; by g'(0): {s.idxmin()}",
             f"paths monotone: {int(tab['monotone'].sum())}/{len(tab)}",
             f"pairs whose difference is monotone: {diffmono[0]}, ordered correctly: {diffmono[1]}"
             " (Lemma 2(a): all of them)",
             f"discordant pairs with BOTH paths monotone (counterexamples to the 1-D argument): "
             f"{len(disc_mono)}" + (f", e.g. {disc_mono[:5]}" if disc_mono else "")]
    if wt_label in tab.index:
        lines.append(f"reverting mutation {wt_label}: rank {tab.loc[wt_label, 'rank_by_dU']} by Delta U,"
                     f" rank {tab.loc[wt_label, 'rank_by_slope']} by g'(0)")
    return "\n".join(lines)


def plot_curves(df, title, path, highlight=None):
    ax = df.plot(figsize=(8.0, 4.8), colormap="tab20", lw=1.3)
    if highlight in df.columns:
        line = ax.get_lines()[list(df.columns).index(highlight)]
        line.set_color("black"); line.set_linewidth(2.6)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$U(A(\gamma))$")
    ax.set_title(title, fontsize=10)
    ax.legend(ncol=2, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=200)
    plt.close(ax.figure)


def plot_scatter(tab, title, path, highlight=None):
    ax = tab.plot.scatter(x="slope", y="dU", figsize=(5.2, 4.6), s=28, color="tab:blue")
    for lab, r in tab.iterrows():
        ax.annotate(lab[-1], (r["slope"], r["dU"]), fontsize=8, xytext=(3, 2),
                    textcoords="offset points", weight="bold" if lab == highlight else "normal")
    ax.axhline(0, lw=0.6, color="gray"); ax.axvline(0, lw=0.6, color="gray")
    ax.set_xlabel(r"$g'(0)$"); ax.set_ylabel(r"$\Delta U$")
    ax.set_title(title, fontsize=10)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=200)
    plt.close(ax.figure)


# --------------------------------------------------------------------------- self-test
def selftest(maps, p0, site0, alpha_idx, targets, am_ref_full, am_ref_head, U, energy):
    """Checks this script's re-implementations against custom_esm itself."""
    print("self-test")
    tol = 1e-4 if maps.dtype == torch.float32 and not maps.autocast else 3e-2
    with torch.no_grad():
        x = maps.block_input(p0)
        q, k = maps.qk(x)
        mine = maps.softmax_attention(q, k).mean(1)
        theirs = maps.attn.attention(x, None)                 # custom_esm MultiHeadAttention
        e1 = float((mine - theirs).abs().max())
        print(f"  per-head attention, mean over heads vs custom_esm attention(): {e1:.1e}")
        e2 = float((symm(maps.full_map(p0)) - maps.m.predict_attention(p0).float()).abs().max())
        print(f"  full_map vs model.predict_attention:                          {e2:.1e}"
              + ("  (predict_attention runs in bf16 on cuda, full_map in fp32 unless --autocast)"
                 if maps.device.type == "cuda" else ""))
        x2 = maps.block_input(path_probs(p0, site0, targets[0][1], 1.0))
        moved = (x2 - x).abs().amax(-1).squeeze(0)
        e0 = float(torch.cat([moved[: site0 + 1], moved[site0 + 2:]]).max())
        print(f"  layer-0 input moves only at the site (other rows, max change): {e0:.1e}")
        A_raw = maps.m._last_attention(sequence_probs=p0.unsqueeze(0)).float()
        mine_U = float(U_log2(maps.full_map(p0), am_ref_full))
        repo_U = float(compute_U_am(clean_and_symm(A_raw).squeeze(0), symm(am_ref_full)))
        e7 = abs(mine_U - repo_U) / max(1.0, abs(repo_U))
        print(f"  log2 energy vs utils.energies.compute_U_am(clean_and_symm(A)), rel.: {e7:.1e}")
        print(f"  energy for the slope checks below: {energy}")
        fr = maps.frozen_stats(maps.block_input(p0))
        e3 = float((maps.head_map(p0, fr) - maps.head_map(p0)).abs().max())
        print(f"  frozen-LayerNorm head equals exact head at sigma1:            {e3:.1e}")
        e4 = 0.0
        for t, i in targets[:3]:
            x1 = maps.block_input(p0); x2 = maps.block_input(path_probs(p0, site0, i, 1.0))
            q1, k1 = maps.qk(x1, fr); q2, k2 = maps.qk(x2, fr)
            h, a = maps.head, site0 + 1
            cf = closed_form_path(q1[0, :, h].double(), k1[0, :, h].double(),
                                  q2[0, :, h].double(), k2[0, :, h].double(), a, [0.25, 0.5, 0.75])
            for n, g in enumerate([0.25, 0.5, 0.75]):
                qg, kg = maps.qk(maps.block_input(path_probs(p0, site0, i, g)), fr)
                direct = maps.softmax_attention(qg, kg)[0, h].double()
                e4 = max(e4, float((cf[n] - direct).abs().max()))
        print(f"  closed form eq. (7) vs direct softmax, affine head:            {e4:.1e}")
    # g'(0) from one backward pass against five-point finite differences. U is computed in
    # float32, so it carries rounding noise (~1e-7 relative per operation, more through the
    # 48 layers of the full map) that a difference amplifies by 1/eps, while the truncation
    # error grows as eps^4. A single eps can be dominated by either, so scan eps: if the
    # backward slope is right, the differences converge to it between the two regimes.
    steps = [8e-2, 4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3]

    def check_slopes(name, energy, tgts):
        sl, U0 = slopes_by_backward(maps, p0, site0, alpha_idx, tgts, energy)
        print(f"  {name}: U(sigma1) = {U0:.4f}; five-point differences at eps = "
              + ", ".join(f"{e:g}" for e in steps))
        worst = 0.0
        with torch.no_grad():
            for t, i in tgts:
                Ug = lambda g: float(energy(path_probs(p0, site0, i, g)))
                fds = [(-Ug(2 * e) + 8 * Ug(e) - 8 * Ug(-e) + Ug(-2 * e)) / (12 * e) for e in steps]
                worst = max(worst, min(abs(fd - sl[t]) / max(1.0, abs(fd)) for fd in fds))
                print(f"    {t}: backward {sl[t]:+.5f} | " + " ".join(f"{fd:+.5f}" for fd in fds))
        print(f"  {name}: slope, backward vs best finite difference, rel. err: {worst:.1e}")
        return worst

    e5 = check_slopes("head (frozen LN)", lambda p: U(maps.head_map(p, fr), am_ref_head),
                      targets[:3])
    e6 = check_slopes("full map", lambda p: U(maps.full_map(p), am_ref_full), targets[:3])
    bad = [e for e in (e0, e1, e3, e4, e7) if e > tol]
    if bad or max(e5, e6) > max(50 * tol, 5e-3):
        raise SystemExit("SELF-TEST FAILED: the re-implementation disagrees with custom_esm")
    print("self-test passed")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=int, default=43, help="1-based residue index of the site")
    ap.add_argument("--muts", type=int, default=3,
                    help="substitutions between sigma1 and protein G, the first at --site (0: sigma1 = ref)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=str, default=None, help="explicit start sequence sigma1 (overrides --muts)")
    ap.add_argument("--ref", type=str, default=PROTEIN_G, help="reference sequence (default protein G)")
    ap.add_argument("--layer", type=int, default=0, help="layer of the single head (must be 0)")
    ap.add_argument("--head", type=int, default=0, help="head index within the layer")
    ap.add_argument("--npoints", type=int, default=21, help="gamma points for the full transformer")
    ap.add_argument("--npoints-head", type=int, default=101, help="gamma points for the head")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autocast", action="store_true", help="bf16 autocast on cuda (faster, noisier)")
    ap.add_argument("--energy", choices=sorted(ENERGIES), default="log2",
                    help="log2: eq. (10), the sampler's U_am; ce: cross entropy to the reference")
    ap.add_argument("--outdir", type=str, default=None,
                    help="default geodesic_esm3_out (log2) or geodesic_esm3_out_<energy>")
    ap.add_argument("--no-exact-head", action="store_true", help="skip the head with true LayerNorms")
    ap.add_argument("--selftest", action="store_true", help="check against custom_esm and exit")
    args = ap.parse_args()

    if args.layer != 0:
        raise SystemExit("the closed form of Section 3 (eq. 7) needs the first layer (--layer 0): "
                         "only there is the input of the attention affine in pi")
    torch.manual_seed(args.seed)
    print("loading ESM3 (esm3-open) ...")
    model = ESM3.from_pretrained("esm3-open", device=torch.device(args.device))
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device=torch.device(args.device), dtype=torch.float32).eval()

    ref = args.ref
    site0 = args.site - 1
    if not 0 <= site0 < len(ref):
        raise SystemExit(f"--site must be in 1..{len(ref)}")
    if args.start is not None:
        if len(args.start) != len(ref):
            raise SystemExit(f"--start must have length {len(ref)}")
        start, mut_sites = args.start, [i for i in range(len(ref)) if args.start[i] != ref[i]]
    else:
        start, mut_sites = make_start(ref, site0, args.muts, args.seed)
    alpha = start[site0]
    targets_aa = [aa for aa in STANDARD if aa != alpha]
    wt_label = f"{alpha}{args.site}{ref[site0]}" if ref[site0] != alpha else None
    print(f"reference : {ref}")
    print(f"start     : {start}")
    print(f"differs at: {[i + 1 for i in mut_sites]}; site {args.site}: {alpha} -> 19 targets")
    U = ENERGIES[args.energy]
    print(f"energy    : {args.energy}")
    if start == ref and args.energy == "log2":
        print("NOTE: start == reference, so every slope vanishes (Proposition 4).")

    maps = Esm3Maps(model, C, clean_and_symm, args.layer, args.head, args.autocast)
    if not 0 <= args.head < maps.attn.n_heads:
        raise SystemExit(f"--head must be in 0..{maps.attn.n_heads - 1}")
    p0, pref = maps.one_hot(start), maps.one_hot(ref)
    idx = {aa: C.SEQUENCE_USED_VOCAB.index(aa) for aa in STANDARD}
    alpha_idx = idx[alpha]
    labels = [f"{alpha}{args.site}{t}" for t in targets_aa]
    targets = [(lab, idx[t]) for lab, t in zip(labels, targets_aa)]

    with torch.no_grad():
        am_ref_full = maps.full_map(pref)
        am_ref_head = maps.head_map(pref)                    # reference head, true LayerNorms

    if args.selftest:
        selftest(maps, p0, site0, alpha_idx, targets, am_ref_full, am_ref_head, U, args.energy)
        return

    if args.outdir is None:
        args.outdir = "geodesic_esm3_out" + ("" if args.energy == "log2" else f"_{args.energy}")
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    g_full = np.linspace(0, 1, args.npoints)
    g_head = np.linspace(0, 1, args.npoints_head)
    results, report = {}, []

    # 1) whole transformer: the map of model.predict_attention
    e_full = lambda p: U(maps.full_map(p), am_ref_full)
    curve_full = lambda i: [float(e_full(path_probs(p0, site0, i, g))) for g in g_full]
    results["full"] = run_map("full", maps, p0, site0, alpha_idx, targets, g_full, e_full, curve_full)

    # 2) first-layer head, LayerNorm statistics frozen at sigma1: closed form of Section 3 (eq. 7)
    with torch.no_grad():
        fr = maps.frozen_stats(maps.block_input(p0))
        q1, k1 = maps.qk(maps.block_input(p0), fr)
    h, a = args.head, site0 + 1                               # +1: BOS is position 0

    def curve_head(i):
        with torch.no_grad():
            q2, k2 = maps.qk(maps.block_input(path_probs(p0, site0, i, 1.0)), fr)
            A = closed_form_path(q1[0, :, h].double(), k1[0, :, h].double(),
                                 q2[0, :, h].double(), k2[0, :, h].double(), a, g_head)
            am = A[:, 1:-1, 1:-1]                             # (G, L, L), the residue block
            return [float(U(am[n], am_ref_head.double())) for n in range(len(g_head))]

    e_head = lambda p: U(maps.head_map(p, fr), am_ref_head)
    results["head"] = run_map("head", maps, p0, site0, alpha_idx, targets, g_head, e_head, curve_head)

    # 3) the same head with the true LayerNorms
    if not args.no_exact_head:
        e_hex = lambda p: U(maps.head_map(p), am_ref_head)
        curve_hex = lambda i: [float(e_hex(path_probs(p0, site0, i, g))) for g in g_head]
        results["head_exact"] = run_map("head_exact", maps, p0, site0, alpha_idx, targets, g_head, e_hex, curve_hex)

    names = {"full": "whole transformer (last layer, mean over heads)",
             "head": f"layer 0, head {h}, affine (closed form, eq. 7)",
             "head_exact": f"layer 0, head {h}, true LayerNorms"}
    header = (f"protein G, site {args.site} ({alpha} in sigma1, {ref[site0]} in the reference); "
              f"sigma1 differs from the reference at {[i + 1 for i in mut_sites]}; seed {args.seed}; "
              f"energy {args.energy}")
    report.append(header)
    for key, (df, slopes) in results.items():
        tab = ranking_table(df, slopes)
        df.to_csv(out / f"geodesics_{key}.csv")
        tab.to_csv(out / f"ranking_{key}.csv")
        title = f"{names[key]}, energy {args.energy}\nsite {args.site}, 19 substitutions of {alpha}"
        plot_curves(df, title, out / f"geodesics_{key}.png", highlight=wt_label)
        plot_scatter(tab, title, out / f"slope_vs_dU_{key}.png", highlight=wt_label)
        report.append(ranking_summary(names[key], df, tab, wt_label))
    text = "\n\n".join(report)
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()