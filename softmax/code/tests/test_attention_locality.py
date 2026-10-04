"""
Settles two questions about the gradient empirically, rather than by argument:

  Q1  "for N=1 the attention map for the backward pass should be the same
       except for that site, yes?"
  Q2  "for N=1 the gradient is computed on a single row back and forth, so I
       assume it's cheaper than computing the gradient on the whole sequence?"

Both are reasonable intuitions. This script measures them directly instead of
reasoning about them. Sections:

  (0) prints the SOURCE of every function involved, so the results file is
      self-contained and the code being discussed can be read without opening
      the repository.

  (1) ATTENTION-MAP LOCALITY (answers Q1). Builds seq_B = seq_A with exactly
      ONE site s changed, computes both attention maps, and splits the
      difference into
         LOCAL  entries in row s or column s   (2L-1 of them)
         DISTAL every other entry              (the remaining ~L^2-2L)
      and reports how much of the total change, and how much of the actual
      change in U_am, each accounts for. If the map really were "the same
      except for that site", DISTAL would be ~0. Also profiles |delta am|
      against sequence distance from s, so any decay is visible rather than
      assumed.

  (2) GRADIENT COST vs NUMBER OF RELAXED SITES (answers Q2). Times a gradient
      pass in which k sites are softmax-relaxed and the other L-k are held at
      their exact one-hot values, for k = 1, 2, 5, 20, 100, L. k=1 is exactly
      what the production sampler does (_grad_pass_site); k=L is exactly the
      whole-sequence gradient. If the single-site gradient were cheaper, the
      time would rise with k. Peak GPU memory is reported per k for the same
      reason: to check whether relaxing one row stores fewer activations.
      Forward-only time is measured separately so the forward/backward split
      is visible.

  (3) EQUIVALENCE CHECK. The k=1 gradient from _grad_pass_site against row s
      of the k=L whole-sequence gradient: cosine similarity and max absolute
      difference. (compare_grad_methods.py already does this; it is repeated
      here so this file stands alone.)

Needs ESM3 + weights + ideally a GPU. Cheap: a few dozen model calls, ~1 min.
Run from softmax/code:
    python -u tests/test_attention_locality.py 2>&1 | tee zero_polymer/attention_locality.txt
Environment:
    REF=zero_polymer|protein_g   SITES=<comma list>   MUTS=<int>   SEED=<int>
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import inspect
import math
import statistics as st
import time

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.energies import compute_U_am, compute_entropy
from utils.operations import mutate
import custom_esm.utils.constants.esm3 as C

from references import get_reference

# Plotting lives in tests/locality_plots.py, deliberately free of torch so the
# figures can be exercised on synthetic arrays without a GPU. If matplotlib or
# pandas are missing the numeric sections still run and only the figures are
# skipped -- the diagnostic must not become unrunnable because of a plotting
# dependency.
try:
    import locality_plots as LP
    _PLOTS = True
    _PLOT_ERR = ""
except Exception as exc:                      # pragma: no cover
    _PLOTS = False
    _PLOT_ERR = f"{type(exc).__name__}: {exc}"

REF_NAME, REF_SEQ = get_reference()
SEED = int(os.environ.get("SEED", 0))
N_INIT_MUTS = int(os.environ.get("MUTS", 5))
PROBE_SITES = [int(x) for x in os.environ.get("SITES", "").split(",") if x.strip()]
# Where figures and CSVs go. Default sits next to this reference's other
# results. PNGs are gitignored project-wide (see .gitignore: regenerable, and
# they bloat the repo), so these stay local unless deliberately un-ignored.
PLOT_DIR = os.environ.get("PLOT_DIR", f"{REF_NAME}/attention_locality_plots")
NO_PLOTS = os.environ.get("NO_PLOTS", "") not in ("", "0", "false", "False")

PARS = {
    "T": 2.0, "dt": 2.0, "M": 1.0, "T_sftm": 0.1,
    "lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0,
    "eps": 1.0e-9, "n_quad": 40,
}

LEGEND = """
================================================================================
LEGEND -- symbols used in this file.
================================================================================
  L          sequence length (number of residues).
  K          size of SEQUENCE_USED_VOCAB = 25 (the 20 standard residues plus
             X, B, U, Z, O). Note the sampler's PROPOSAL only ever targets the
             20 standard ones minus the current residue = 19 competitors; the
             25 here is the width of the probability rows fed to the model.
  d_model    transformer width (1536 for esm3-open), n_layers = 48.
  s          the single site that differs between seq_A and seq_B.
  am         attention map, shape (L, L), the mean over heads of the last
             block's attention, with BOS/EOS stripped and symmetrised.
  delta am   am(seq_B) - am(seq_A), i.e. the effect of changing ONLY site s.
  LOCAL      entries of delta am lying in row s or column s: 2L-1 entries.
             These are the ones a "the map only changes at that site"
             intuition would predict to be the only nonzero ones.
  DISTAL     every other entry of delta am: entries (i,j) with i != s and
             j != s. If the intuition held, these would all be ~0.
  dU_am      the change in the energy U_am = sum_{i<j} (log am_ij -
             log ref_am_ij)^2 caused by that one substitution, split into the
             contribution of LOCAL pairs and of DISTAL pairs. This is the
             decomposition that matters: it says how much of the ENERGY
             consequence of a single mutation comes from parts of the map
             that do not involve the mutated site at all.
  k          number of sites simultaneously softmax-relaxed in a gradient
             pass. k=1 is what the production sampler does
             (_grad_pass_site); k=L is the whole-sequence gradient.
  t_fwd      wall-clock of the forward pass alone (no autograd graph).
  t_fwd_bwd  wall-clock of forward + backward for a given k.
  peak MB    peak GPU memory allocated during that pass.
  cosine     cosine similarity between two gradient vectors over the K
             components at a site; 1.0 means identical direction.
================================================================================
"""


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def grad_k_sites(sampler, eprot, sites, pars, backward=True):
    """Gradient with EXACTLY len(sites) rows softmax-relaxed and every other
    row held at its exact one-hot value.

    k=1 reproduces classes/rate_sampler.py:_grad_pass_site; k=L reproduces the
    whole-sequence gradient used by the multi-site diagnostics. Keeping both in
    one function is the point: the only thing that varies across the timing
    sweep is how many rows carry a gradient, so any cost difference must come
    from that and nothing else.

    Note the leaf is ONLY the k relaxed rows, so the softmax is computed on a
    (k, K) tensor, not on all L rows -- otherwise the comparison would be
    confounded by doing L softmaxes at every k."""
    hard = eprot.get_probs()                                   # (L,K) one-hot, no grad
    idx = torch.as_tensor(sites, dtype=torch.long, device=hard.device)
    leaf = eprot.logits[idx].detach().clone().requires_grad_(True)   # (k,K)
    soft = torch.softmax(leaf/pars['T_sftm'], dim=-1)                # (k,K)
    probs = hard.index_copy(0, idx, soft)                            # (L,K)

    am = sampler.model.predict_attention(sequence_probs=probs)
    U = pars['lambda_am']*compute_U_am(am, sampler.ref_eprot.am)
    if pars['lambda_S'] > 0.:
        U = U + pars['lambda_S']*compute_entropy(soft, pars['eps'])
    if backward:
        U.backward()
        return leaf.grad.detach().clone(), U.item()
    return None, U.item()


def _emit_outputs(site_records, profile_frames, plot_payload, want_plots):
    """Figures and CSVs. Wrapped so a plotting failure degrades to a warning
    rather than discarding the model run that produced the numbers."""
    if not want_plots:
        return
    try:
        import pandas as pd
        os.makedirs(PLOT_DIR, exist_ok=True)
        summary = pd.DataFrame(site_records)
        profiles = [pd.DataFrame(p) for p in profile_frames]
        c1, c2 = LP.write_tables(summary, profiles, PLOT_DIR)
        made = [c1, c2]
        for s_, am_a, am_b, dp, lab in plot_payload:
            made.append(LP.attention_panels(
                am_a, am_b, dp, s_, os.path.join(PLOT_DIR, f"panels_site{s_}.png"),
                ref_label=f"reference {REF_NAME}", title_extra=f"  [{lab}]"))
        for p in profiles:
            s_ = int(p["site"].iloc[0])
            made.append(LP.decay_profile(p, s_, os.path.join(PLOT_DIR, f"decay_site{s_}.png")))
        made.append(LP.local_vs_distal_bar(summary, os.path.join(PLOT_DIR, "distal_share.png")))
        print(f"\n  wrote {len(made)} files to {PLOT_DIR}/ :")
        for p in made:
            print(f"    {os.path.getsize(p)/1024:9.1f} KB  {os.path.basename(p)}")
        print("  (PNGs are gitignored project-wide -- regenerable, and they would bloat")
        print("   the repo. The two CSVs carry the same numbers in machine-readable form.)")
    except Exception as exc:
        print(f"\n  [figure/CSV generation FAILED: {type(exc).__name__}: {exc}]")
        print("  The numbers above are unaffected.")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    L = len(REF_SEQ)
    vocab = C.SEQUENCE_USED_VOCAB

    print(f"# reference: {REF_NAME} (L={L} residues)   device={device}")
    print(LEGEND)

    # ===================================================================== #
    print("=== (0) the code under discussion, verbatim ===")
    print("--- classes/rate_sampler.py: _grad_pass_site "
          "(what the production sampler runs, k=1) ---")
    print(inspect.getsource(ExtendedProteinRateSampler._grad_pass_site))
    print("--- this file's grad_k_sites (the generalisation timed in section 2) ---")
    print(inspect.getsource(grad_k_sites))
    print("--- utils/energies.py: compute_U_am ---")
    print(inspect.getsource(compute_U_am))
    print("Key structural point to check against the source above: `probs` is a")
    print("FULL (L,K) tensor in every case. predict_attention therefore runs over")
    print("all L residues whatever k is, and the backward traverses the same")
    print("transformer graph. Only the leaf differs: (k,K) instead of (L,K).")
    print()

    sampler = ExtendedProteinRateSampler(config_settings={})
    sampler.model.to(device)
    sampler._canonical_idx = sampler._canonical_idx.to(device)

    from generator.custom_generator import CustomGenerator
    sampler.generator = CustomGenerator(seed=SEED, device=device)

    sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
    sampler.ref_eprot.expand()
    sampler.ref_eprot.am = sampler.model.predict_attention(
        sequence_probs=sampler.ref_eprot.get_probs())

    seq_A = mutate(REF_SEQ, N_INIT_MUTS, sampler.generator.get()) if N_INIT_MUTS else REF_SEQ
    eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
    eprot_A.expand()

    sites = PROBE_SITES or [L//7, L//2, (5*L)//6]
    sites = [s for s in sites if 0 <= s < L]

    # ===================================================================== #
    # (1) attention-map locality                                            #
    # ===================================================================== #
    n_pairs = L*(L-1)//2
    print("=== (1) does changing ONE site change the attention map only at that site? ===")
    print(f"Scale first: U_am sums over {n_pairs} upper-triangle pairs, and only")
    print(f"{L-1} of them involve site s at all -- {100.*(L-1)/n_pairs:.3f}% of the terms.")
    print("So if the map really changed only in row/column s, a single substitution")
    print("could only ever move that fraction of the energy. Measured single-site dU")
    print("is in the hundreds against a U_A in the thousands, which already suggests")
    print("otherwise; the split below settles it exactly.")
    print("For each probe site s: seq_B = seq_A with s substituted, then")
    print("delta am = am(seq_B) - am(seq_A), split into LOCAL (row/col s) and")
    print("DISTAL (everything else). See LEGEND.")

    want_plots = _PLOTS and not NO_PLOTS
    if NO_PLOTS:
        print("  [figures disabled via NO_PLOTS]")
    elif not _PLOTS:
        print(f"  [figures skipped -- plotting import failed: {_PLOT_ERR}]")
    site_records, profile_frames, plot_payload = [], [], []

    with torch.no_grad():
        am_A = sampler.model.predict_attention(sequence_probs=eprot_A.get_probs())
    ref_am = sampler.ref_eprot.am
    log_ref = torch.log(ref_am)
    log_A = torch.log(am_A)

    for s in sites:
        cur = vocab.index(seq_A[s])
        comp = sampler._competition_indices(cur)
        tgt = int(comp[0].item()) if vocab[int(comp[0].item())] != seq_A[s] else int(comp[1].item())
        seq_B = seq_A[:s] + vocab[tgt] + seq_A[s+1:]
        eprot_B = ExtendedProtein(sequence=seq_B, requires_grad=False, device=device)
        eprot_B.expand()
        with torch.no_grad():
            am_B = sampler.model.predict_attention(sequence_probs=eprot_B.get_probs())

        D = (am_B - am_A).abs()
        mask_local = torch.zeros_like(D, dtype=torch.bool)
        mask_local[s, :] = True
        mask_local[:, s] = True
        n_local = int(mask_local.sum().item())
        n_distal = D.numel() - n_local

        tot = D.sum().item()
        loc = D[mask_local].sum().item()
        dis = tot - loc

        # energy-side decomposition: how much of the ACTUAL dU_am comes from
        # pairs that do not involve site s at all
        log_B = torch.log(am_B)
        per_pair_A = torch.triu((log_A - log_ref)**2., diagonal=1)
        per_pair_B = torch.triu((log_B - log_ref)**2., diagonal=1)
        dpair = per_pair_B - per_pair_A
        triu_local = mask_local & (torch.triu(torch.ones_like(D), diagonal=1) > 0)
        dU_total = dpair.sum().item()
        dU_local = dpair[triu_local].sum().item()
        dU_distal = dU_total - dU_local

        print(f"\n  site s={s}  ({seq_A[s]} -> {vocab[tgt]})")
        print(f"    |delta am|:  total={tot:.4e}   LOCAL={loc:.4e} ({100*loc/tot:5.1f}% of total, "
              f"{n_local} entries)   DISTAL={dis:.4e} ({100*dis/tot:5.1f}%, {n_distal} entries)")
        print(f"    mean |delta am| per entry:  LOCAL={loc/n_local:.4e}   "
              f"DISTAL={dis/n_distal:.4e}   ratio local/distal={(loc/n_local)/(dis/n_distal):.1f}")
        print(f"    max |delta am| among DISTAL entries = {D[~mask_local].max().item():.4e}")
        print(f"    dU_am = {dU_total:+.4f}   from LOCAL pairs {dU_local:+.4f} "
              f"({100*dU_local/dU_total if dU_total else float('nan'):5.1f}%)   "
              f"from DISTAL pairs {dU_distal:+.4f} "
              f"({100*dU_distal/dU_total if dU_total else float('nan'):5.1f}%)")

        # decay profile: mean |delta am| by sequence distance from s
        ii = torch.arange(L, device=D.device).view(-1, 1).expand(L, L)
        jj = torch.arange(L, device=D.device).view(1, -1).expand(L, L)
        dist = torch.minimum((ii - s).abs(), (jj - s).abs())
        bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 10**6)]
        prof, prof_rows = [], []
        for lo, hi in bins:
            m = (dist >= lo) & (dist <= hi)
            if m.any():
                mv = D[m].mean().item()
                prof.append(f"{lo}-{hi if hi < 10**6 else 'inf'}:{mv:.2e}")
                prof_rows.append(dict(site=s, distance=lo, distance_hi=hi,
                                      mean_abs_delta=mv, n_entries=int(m.sum().item())))
        print(f"    mean |delta am| by distance from s:  " + "  ".join(prof))

        site_records.append(dict(
            site=s, from_aa=seq_A[s], to_aa=vocab[tgt],
            n_local=n_local, n_distal=n_distal,
            abs_total=tot, abs_local=loc, abs_distal=dis,
            distal_share_abs=(dis/tot if tot else float('nan')),
            mean_abs_local=loc/n_local, mean_abs_distal=dis/n_distal,
            max_abs_distal=D[~mask_local].max().item(),
            dU_total=dU_total, dU_local=dU_local, dU_distal=dU_distal,
            distal_share_dU=(dU_distal/dU_total if dU_total else float('nan'))))
        profile_frames.append(prof_rows)
        if want_plots:
            plot_payload.append((s, am_A.float().cpu().numpy(),
                                 am_B.float().cpu().numpy(),
                                 dpair.float().cpu().numpy(),
                                 f"{seq_A[s]}->{vocab[tgt]}"))

    if site_records:
        _emit_outputs(site_records, profile_frames, plot_payload, want_plots)

    print("\n  How to read (1): if the attention map changed only at the mutated site,")
    print("  DISTAL would be ~0 in every column above. The DISTAL share of dU_am is")
    print("  the decisive number -- it is the part of a single mutation's ENERGY")
    print("  consequence that comes from pairs not involving that site at all, and")
    print("  it is also exactly the mechanism that produces cross-site coupling.")
    print()

    # ===================================================================== #
    # (2) gradient cost vs number of relaxed sites                          #
    # ===================================================================== #
    print("=== (2) does a gradient over FEWER sites cost less? ===")
    print("k rows softmax-relaxed, L-k rows held at their exact one-hot value.")
    print("k=1 is the production sampler's _grad_pass_site; k=L is the")
    print("whole-sequence gradient. If one row were cheaper, t_fwd_bwd would rise")
    print("with k.")

    def timeit(fn, repeats=5, warmup=2):
        for _ in range(warmup):
            fn()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        sync(device)
        return (time.perf_counter() - t0)/repeats

    g = torch.Generator().manual_seed(SEED)
    k_values = [k for k in (1, 2, 5, 20, 100, L) if k <= L]

    print(f"\n  {'k':>6} {'t_fwd (ms)':>12} {'t_fwd_bwd (ms)':>16} "
          f"{'backward (ms)':>15} {'peak MB':>10}")
    base = None
    for k in k_values:
        ks = torch.randperm(L, generator=g)[:k].tolist()
        t_f = timeit(lambda: grad_k_sites(sampler, eprot_A, ks, PARS, backward=False))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_fb = timeit(lambda: grad_k_sites(sampler, eprot_A, ks, PARS, backward=True))
        peak = (torch.cuda.max_memory_allocated()/2**20) if device.type == "cuda" else float('nan')
        if base is None:
            base = t_fb
        print(f"  {k:>6} {t_f*1e3:>12.2f} {t_fb*1e3:>16.2f} {(t_fb-t_f)*1e3:>15.2f} "
              f"{peak:>10.1f}")
    print(f"\n  k=1 vs k=L ratio of t_fwd_bwd: "
          f"{base/timeit(lambda: grad_k_sites(sampler, eprot_A, list(range(L)), PARS)):.4f}")
    print("  A flat column means the cost is set by the forward over all L residues")
    print("  and the backward through all 48 layers, neither of which depends on k.")
    print()

    # ===================================================================== #
    # (3) equivalence of the k=1 gradient with row s of the k=L gradient    #
    # ===================================================================== #
    print("=== (3) is the k=1 gradient the same vector as row s of the k=L gradient? ===")
    print("They are NOT required to be identical -- k=1 holds the other sites at")
    print("their exact one-hot values while k=L softmax-relaxes them all, so the")
    print("two evaluate the derivative at slightly different points. The question")
    print("is whether that difference matters in practice.")
    grad_all, _ = grad_k_sites(sampler, eprot_A, list(range(L)), PARS)
    print(f"\n  {'site':>6} {'cosine':>10} {'|g_k1|':>12} {'|g_kL|':>12} "
          f"{'max|diff|':>12} {'same argmax?':>13}")
    for s in sites:
        g1, _ = grad_k_sites(sampler, eprot_A, [s], PARS)
        g1 = g1[0]
        gL = grad_all[s]
        cos = torch.nn.functional.cosine_similarity(g1.float(), gL.float(), dim=0).item()
        comp = sampler._competition_indices(int(eprot_A.logits[s].argmax(-1).item()))
        same = int((-g1[comp]).argmax().item()) == int((-gL[comp]).argmax().item())
        print(f"  {s:>6} {cos:>10.6f} {g1.norm().item():>12.4e} {gL.norm().item():>12.4e} "
              f"{(g1-gL).abs().max().item():>12.4e} {str(same):>13}")
    print("\n  'same argmax?' is the only thing the sampler actually consumes: the")
    print("  gradient's preferred substitution among the 19 legal competitors.")


if __name__ == "__main__":
    main()
