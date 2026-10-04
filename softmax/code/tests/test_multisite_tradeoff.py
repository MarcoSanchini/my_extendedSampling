"""
Deep multi-site (N-mutations-per-move) analysis: coupling vs. starting point,
energy decomposition, wall-clock cost per move, and the actual throughput
question.

WHY THIS EXISTS ALONGSIDE test_joint_coupling_N.py
--------------------------------------------------
test_joint_coupling_N.py answered one question -- "how non-separable is U_am
across simultaneously-changed sites?" -- from ONE starting point
(mutate(REF,5)), with no timing and no acceptance model. Two things have since
made that insufficient (see DEVLOG.txt, 2026-10-03/04):

  1. MUTS=5 on a 566-residue reference is 0.9% drift, i.e. a measurement taken
     essentially AT U_am's floor. That was shown to be actively misleading
     about proposal quality: the same gradient that looked bad at the floor
     (mean dU at its pick +474) looks good in the bulk (-360). Every coupling
     number on record inherits that caveat, so the first job here is to vary
     the starting point.
  2. Coupling magnitude was never the decision-relevant quantity anyway. What
     decides whether multi-site moves are worth building is THROUGHPUT:
     accepted mutations per unit wall-clock. An N-site move costs about the
     same as a single-site move (the dominant cost is a whole-sequence
     forward pass through predict_attention, which runs over all L residues
     regardless of how many are being changed -- DEVLOG 2026-09-24), but
     delivers N mutations instead of 1 when accepted. So multi-site wins iff

         a_N * N / t_N  >  a_1 * 1 / t_1,     and if t_N ~ t_1,  a_N * N > a_1

     where a_N is the acceptance probability of an N-site move. Nothing has
     ever measured a_N, or t_N, or checked whether t_N really does equal t_1.
     This script measures all three.

WHAT IT MEASURES
----------------
Per starting point (drift level x seed), per site-selection mode, per N:

  energy decomposition (as before, but with better statistics)
      sum_linear   first-order Taylor prediction (no coupling, no within-site
                   nonlinearity)
      sum_isolated true single-site effects applied one at a time vs. the
                   ORIGINAL sequence (within-site nonlinearity, no coupling)
      dU_joint     true effect with all N sites changed at once
      nonlin       = sum_isolated - sum_linear
      coupling     = dU_joint - sum_isolated
    reported with MEDIAN and IQR alongside mean/std, because these
    distributions were already shown to be heavy-tailed and sign-flipping, so
    a mean +- std misrepresents them.

  per-site normalisation
      dU_joint / N -- the cost per mutation actually delivered, which is what
      must be compared against a single-site move's own dU.

  acceptance and throughput, over a grid of T
      a_N          estimated acceptance of the N-site move
      a_N * N      accepted mutations per move
      vs. a_1      the single-site baseline measured in the same run
    This is the figure of merit, and it is the only part of this script that
    answers the design question directly.

  wall-clock timing
      t_exact      one exact-energy forward pass
      t_grad_site  one single-site relaxed gradient (what the production
                   sampler does, 2x per move)
      t_grad_whole one whole-sequence relaxed gradient (what a multi-site
                   sampler would do, 2x per move)
    measured with proper CUDA synchronisation. Whether t_grad_whole ~
    t_grad_site is an ASSUMPTION the multi-site case rests on and has never
    been checked; it is checked here.

  site-separation dependence (the "is it just distance?" control)
      DISPERSED sites are drawn uniformly over L; CLUSTERED sites are drawn
      within a window of CLUSTER_WINDOW residues. On a 566-residue sequence,
      N random sites are far apart, which is a candidate explanation for why
      coupling looked weaker there than on 56-residue protein G. If clustered
      coupling is much stronger than dispersed at the same N, the length
      effect is largely a distance effect and the earlier cross-protein
      comparison was confounded.

COST, and why this is affordable
--------------------------------
The isolated single-site energy of site s depends only on (starting point, s,
target), and the target is deterministic given the gradient -- so it is the
SAME for every trial and every N that touches s. test_joint_coupling_N.py
recomputed it every trial; this caches it per starting point, which is where
most of the saving comes from. Per starting point the cost is then

    1 backward (whole-sequence gradient)
  + |site pool| forwards (isolated energies, computed once)
  + (#modes x #N x trials) forwards (one joint energy per trial)

The script prints its own forward-pass budget before doing any work, so an
expensive configuration can be aborted rather than discovered.

Configuration, all via environment variables (defaults give the full run):
    DRIFTS=5,51,150     initial-mutation counts defining the starting points
    SEEDS=0,1           one starting point per (drift, seed) pair
    NVALS=1,2,3,5,8,12,20
    TRIALS=10           trials per (start, mode, N)
    MODES=dispersed,clustered
    CLUSTER_WINDOW=30   window width for clustered site selection
    TGRID=32,107,322    temperatures for the acceptance/throughput model
    POOL=60             random sites per starting point in the isolated cache
    REF=zero_polymer    (via tests/references.py)

Needs a working ESM3 install + weights + GPU. Run from softmax/code:
    python -u tests/test_multisite_tradeoff.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

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

REF_NAME, REF_SEQ = get_reference()

def _ints(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]
def _floats(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",") if x.strip()]

DRIFTS         = _ints("DRIFTS", "5,51,150")
SEEDS          = _ints("SEEDS", "0,1")
N_VALUES       = _ints("NVALS", "1,2,3,5,8,12,20")
TRIALS         = int(os.environ.get("TRIALS", 10))
MODES          = [m.strip() for m in os.environ.get("MODES", "dispersed,clustered").split(",") if m.strip()]
CLUSTER_WINDOW = int(os.environ.get("CLUSTER_WINDOW", 30))
T_GRID         = _floats("TGRID", "32,107,322")
POOL           = int(os.environ.get("POOL", 60))

# Energy parameters. lambda_am=1 and no entropy term, matching _exact_energy's
# own convention and every other diagnostic in this suite. NOTE the "T" used
# for acceptance is T_GRID, not this one -- pars['T'] only enters the
# (unused here) proposal width.
PARS = {
    "T": 2.0, "dt": 2.0, "M": 1.0, "T_sftm": 0.1,
    "lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0,
    "eps": 1.0e-9, "n_quad": 40,
}


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def sync(device):
    """CUDA kernels are asynchronous, so any wall-clock measurement that does
    not synchronise is measuring queue-submission time, not compute."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def whole_sequence_grad(sampler, eprot, pars):
    """Whole-sequence-relaxed gradient: every site softmax-relaxed at once, one
    backward pass yields d U / d logits for ALL sites. This is the
    (now-removed) method the original JOINT design used, reimplemented
    identically in tests/compare_grad_methods.py and
    tests/test_joint_coupling_N.py -- see DEVLOG.txt 2026-09-24 for why it is
    the historically correct one to use for a multi-site question."""
    probs = eprot.get_probs(pars['T_sftm'])
    am = sampler.model.predict_attention(sequence_probs=probs)
    U_am = compute_U_am(am, sampler.ref_eprot.am)
    entropy = compute_entropy(probs, pars['eps'])
    U = pars['lambda_am']*U_am + pars['lambda_S']*entropy
    eprot.logits.grad = None
    U.backward()
    grad = eprot.logits.grad.detach().clone()
    eprot.logits.grad = None
    return grad


def energy_of(sampler, seq, pars, device):
    eprot = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
    eprot.expand()
    U, _, _ = sampler._exact_energy(eprot, pars)
    return U


def quartiles(xs):
    """(median, q1, q3). Used instead of mean+-std throughout the per-N
    reporting: these distributions are heavy-tailed and change sign between
    trials, so a mean and a standard deviation together describe them badly."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    med = st.median(s)
    lo = s[:n//2]
    hi = s[(n+1)//2:]
    q1 = st.median(lo) if lo else med
    q3 = st.median(hi) if hi else med
    return med, q1, q3


def draw_sites(L, N, mode, rng, window):
    """Dispersed: uniform over the whole sequence. Clustered: uniform within a
    randomly placed window of `window` residues, which is the control for
    whether coupling is really a sequence-distance effect.

    The window is clamped to L, and clustered selection degenerates to
    dispersed when N >= window (there is no room to cluster) -- both guards
    matter for a short reference such as protein_g, where an unclamped window
    would index past the end of the sequence."""
    window = min(window, L)
    if mode == "dispersed" or N >= window:
        return torch.randperm(L, generator=rng)[:N].tolist()
    start = int(torch.randint(0, L - window + 1, (1,), generator=rng).item())
    local = torch.randperm(window, generator=rng)[:N].tolist()
    return [start + x for x in local]


def site_spread(sites):
    """Mean pairwise sequence separation of the chosen sites; reported so the
    dispersed/clustered contrast is quantified rather than asserted."""
    if len(sites) < 2:
        return float('nan')
    ds = [abs(a-b) for i, a in enumerate(sites) for b in sites[i+1:]]
    return sum(ds)/len(ds)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    L = len(REF_SEQ)
    vocab = C.SEQUENCE_USED_VOCAB

    n_starts = len(DRIFTS)*len(SEEDS)
    joint_per_start = len(MODES)*len(N_VALUES)*TRIALS
    # Isolated-energy cost is the number of DISTINCT sites touched, not the
    # number of draws: a site's isolated energy is computed once and reused.
    # For dispersed draws the expected distinct count after d draws over L
    # sites is L*(1-(1-1/L)^d) (coupon-collector); clustered draws concentrate
    # into windows so they touch no more, which makes this an upper bound.
    draws_per_mode = TRIALS*sum(N_VALUES)
    exp_distinct = L*(1.0 - (1.0 - 1.0/L)**draws_per_mode)
    iso_per_start = min(L, len(MODES)*exp_distinct)
    budget = n_starts*(iso_per_start + joint_per_start)

    print(f"# reference: {REF_NAME} (L={L} residues)")
    print(f"# starting points: {n_starts}  (drifts={DRIFTS} x seeds={SEEDS})")
    print(f"# N values: {N_VALUES}   trials/cell: {TRIALS}   modes: {MODES}")
    print(f"# acceptance temperatures: {T_GRID}")
    print(f"# estimated budget: ~{budget:.0f} forward passes + {n_starts} backward passes")
    print(f"#   ({iso_per_start:.0f} isolated-energy forwards per starting point, bounded by the")
    print(f"#    number of DISTINCT sites touched, plus {joint_per_start} joint-energy forwards)")
    print(f"# (isolated single-site energies are cached per starting point -- they depend")
    print(f"#  only on (start, site, target), so they are reused across every N and trial)")
    print()

    sampler = ExtendedProteinRateSampler(config_settings={})
    sampler.model.to(device)
    sampler._canonical_idx = sampler._canonical_idx.to(device)

    from generator.custom_generator import CustomGenerator
    sampler.generator = CustomGenerator(seed=0, device=device)

    sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
    sampler.ref_eprot.expand()
    sampler.ref_eprot.am = sampler.model.predict_attention(
        sequence_probs=sampler.ref_eprot.get_probs())

    # ===================================================================== #
    # (0) wall-clock cost per operation                                      #
    # ===================================================================== #
    print("=== (0) wall-clock cost per model operation ===")
    print("The multi-site case rests on t_grad_whole ~ t_grad_site (one backward")
    print("covering all sites costs no more than one covering a single site, because")
    print("the forward pass runs over all L residues either way). That is an")
    print("assumption; this measures it.")

    probe_seq = mutate(REF_SEQ, 5, sampler.generator.get())
    probe = ExtendedProtein(sequence=probe_seq, requires_grad=True, device=device)
    probe.expand()

    def timeit(fn, repeats=5, warmup=2):
        for _ in range(warmup):
            fn()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        sync(device)
        return (time.perf_counter() - t0)/repeats

    t_exact = timeit(lambda: sampler._exact_energy(probe.copy(), PARS))
    t_grad_site = timeit(lambda: sampler._grad_pass_site(probe, 0, PARS))
    t_grad_whole = timeit(lambda: whole_sequence_grad(sampler, probe, PARS))

    # One move of each kind: 2 gradient passes (forward proposal + reverse
    # probability at the candidate) + 1 exact energy. Same structure either way.
    t_move_single = 2*t_grad_site + t_exact
    t_move_multi = 2*t_grad_whole + t_exact

    print(f"\n  t_exact       (1 forward)              = {t_exact*1e3:8.2f} ms")
    print(f"  t_grad_site   (1 single-site backward) = {t_grad_site*1e3:8.2f} ms")
    print(f"  t_grad_whole  (1 whole-seq backward)   = {t_grad_whole*1e3:8.2f} ms"
          f"   ratio to site: {t_grad_whole/t_grad_site:.3f}")
    print(f"\n  modelled move cost (2 grads + 1 exact):")
    print(f"    single-site move = {t_move_single*1e3:8.2f} ms")
    print(f"    N-site move      = {t_move_multi*1e3:8.2f} ms"
          f"   ratio: {t_move_multi/t_move_single:.3f}")
    if t_move_multi/t_move_single > 1.25:
        print("  *** t_N is materially larger than t_1: the 'same cost, N mutations'")
        print("      premise does NOT hold here and the throughput model below must be")
        print("      read with the measured ratio, not assumed parity. ***")
    else:
        print("  -> cost parity holds: an N-site move costs about the same as a")
        print("     single-site one, so throughput is decided by a_N*N vs a_1.")

    eta = budget*t_exact + n_starts*t_grad_whole
    print(f"\n  ETA for the rest of this run: ~{eta/60:.1f} min "
          f"({budget:.0f} forwards at the {t_exact*1e3:.1f} ms just measured). "
          f"Ctrl-C now if that is too long;\n  reduce it with TRIALS, NVALS, DRIFTS, SEEDS or MODES.")
    print()

    # ===================================================================== #
    # per starting point                                                     #
    # ===================================================================== #
    # results[(drift, seed, mode, N)] = list of per-trial dicts
    results = {}
    starts_meta = {}

    for drift in DRIFTS:
        for seed in SEEDS:
            gen = CustomGenerator(seed=seed, device=device)
            seq_A = mutate(REF_SEQ, drift, gen.get()) if drift > 0 else REF_SEQ
            hd = sum(1 for a, b in zip(seq_A, REF_SEQ) if a != b)

            eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
            eprot_A.expand()
            U_A = energy_of(sampler, seq_A, PARS, device)
            grad_whole = whole_sequence_grad(sampler, eprot_A, PARS)

            starts_meta[(drift, seed)] = dict(hd=hd, U_A=U_A, seq=seq_A)
            print(f"=== starting point: drift={drift} seed={seed} -> "
                  f"Hd={hd}/{L} ({100.*hd/L:.1f}%)  U_A={U_A:.2f} ===")

            # ---- deterministic gradient-greedy target per site ----
            tgt_cache = {}

            def target_of(s):
                """Deterministic gradient-greedy target for site s. Cached: each
                .item() forces a GPU synchronisation, and this is called once per
                site per trial, so recomputing it would cost real time for a
                value that cannot change (grad_whole and seq_A are both fixed
                for this starting point)."""
                if s not in tgt_cache:
                    cur = int(eprot_A.logits[s].argmax(dim=-1).item())
                    comp = sampler._competition_indices(cur)
                    best = (-grad_whole[s][comp]).argmax(dim=-1)
                    tgt_cache[s] = (cur, int(comp[best].item()))
                return tgt_cache[s]

            # ---- isolated single-site energies, computed once per site ----
            iso_cache = {}
            lin_cache = {}

            def isolated(s):
                if s not in iso_cache:
                    cur, tgt = target_of(s)
                    cand = seq_A[:s] + vocab[tgt] + seq_A[s+1:]
                    iso_cache[s] = energy_of(sampler, cand, PARS, device) - U_A
                    oh = torch.zeros(len(vocab), device=device)
                    oh[tgt] += 1.
                    oh[cur] -= 1.
                    lin_cache[s] = (grad_whole[s]*oh).sum().item()
                return iso_cache[s], lin_cache[s]

            for mode in MODES:
                for N in N_VALUES:
                    rows = []
                    for trial in range(TRIALS):
                        rng = torch.Generator().manual_seed(
                            1000003*drift + 10007*seed + 1009*N + 13*trial
                            + (7 if mode == "clustered" else 0))
                        sites = draw_sites(L, N, mode, rng, CLUSTER_WINDOW)

                        sum_lin = sum_iso = 0.
                        partial = list(seq_A)
                        for s in sites:
                            iso, lin = isolated(s)
                            sum_iso += iso
                            sum_lin += lin
                            partial[s] = vocab[target_of(s)[1]]

                        U_joint = energy_of(sampler, "".join(partial), PARS, device)
                        dU = U_joint - U_A
                        rows.append(dict(
                            sites=sites, dU=dU, iso=sum_iso, lin=sum_lin,
                            nonlin=sum_iso - sum_lin, coup=dU - sum_iso,
                            spread=site_spread(sites)))
                    results[(drift, seed, mode, N)] = rows
                    med, q1, q3 = quartiles([r['dU'] for r in rows])
                    cmed, _, _ = quartiles([r['coup'] for r in rows])
                    print(f"  {mode:<10} N={N:>3}  dU_joint med={med:>10.1f} "
                          f"[{q1:>9.1f},{q3:>9.1f}]   coupling med={cmed:>10.1f}   "
                          f"dU/site={med/N:>9.1f}")
            # The isolated cache ends up covering most or all of the sequence,
            # so it is a complete single-site landscape at this starting point
            # -- already paid for, and the cleanest view of how drift changes
            # the energy surface the proposal sees.
            iso_vals = list(iso_cache.values())
            med, q1, q3 = quartiles(iso_vals)
            n_dn = sum(1 for v in iso_vals if v < 0)
            print(f"  isolated single-site dU over {len(iso_vals)} distinct sites "
                  f"({100.*len(iso_vals)/L:.0f}% of the sequence):")
            print(f"    median={med:>10.1f}  IQR=[{q1:>9.1f},{q3:>9.1f}]  "
                  f"min={min(iso_vals):>10.1f}  max={max(iso_vals):>10.1f}")
            print(f"    downhill (dU<0): {n_dn}/{len(iso_vals)} = {100.*n_dn/len(iso_vals):.0f}%"
                  f"   mean={st.mean(iso_vals):>+10.1f}")
            starts_meta[(drift, seed)]['iso_median'] = med
            starts_meta[(drift, seed)]['iso_frac_down'] = n_dn/len(iso_vals)
            starts_meta[(drift, seed)]['iso_n'] = len(iso_vals)
            print()

    # ===================================================================== #
    # (1) N=1 consistency check                                             #
    # ===================================================================== #
    print("=== (1) consistency: coupling must be exactly 0 at N=1 ===")
    if 1 in N_VALUES:
        bad = [k for k in results if k[3] == 1
               for r in results[k] if abs(r['coup']) > 1e-2]
        print(f"  N=1 cells checked: {sum(1 for k in results if k[3]==1)}; "
              f"violations: {len(bad)}  -> {'OK' if not bad else 'MISMATCH'}")
    else:
        print("  N=1 not in NVALS; check skipped.")
    print()

    # ===================================================================== #
    # (2) does the starting point change the coupling picture?              #
    # ===================================================================== #
    print("=== (1b) the single-site energy landscape, by starting point ===")
    print("This is the baseline every multi-site move competes against: what ONE")
    print("mutation costs, and how often one is downhill at all. It also shows the")
    print("floor effect directly -- at low drift the sequence sits at U_am's minimum")
    print("and almost nothing is downhill, which is what made the earlier MUTS=5")
    print("coupling numbers unrepresentative (DEVLOG 2026-10-03).")
    print(f"\n{'drift':>6} {'seed':>5} {'Hd':>6} {'U_A':>12} {'median dU':>11} "
          f"{'% downhill':>11} {'sites':>7}")
    for d in DRIFTS:
        for sd in SEEDS:
            m = starts_meta.get((d, sd))
            if not m or 'iso_median' not in m:
                continue
            print(f"{d:>6} {sd:>5} {m['hd']:>6} {m['U_A']:>12.1f} "
                  f"{m['iso_median']:>11.1f} {100*m['iso_frac_down']:>10.0f}% {m['iso_n']:>7}")
    print()

    BASE_MODE = "dispersed" if "dispersed" in MODES else MODES[0]
    print(f"=== (2) coupling vs N, by drift ({BASE_MODE} sites, pooled over seeds) ===")
    print("Scale-free ratio: median|coupling| / median|sum_isolated|. Raw energies")
    print("are not comparable across drift levels (U_A itself moves by orders of")
    print("magnitude), so the ratio is what carries meaning here.")
    hdr = f"{'N':>4} " + " ".join(f"{'drift='+str(d):>14}" for d in DRIFTS)
    print(hdr); print("-"*len(hdr))
    for N in N_VALUES:
        cells = []
        for d in DRIFTS:
            rows = [r for s in SEEDS for r in results.get((d, s, BASE_MODE, N), [])]
            if not rows:
                cells.append(f"{'--':>14}"); continue
            mc = st.median([abs(r['coup']) for r in rows])
            mi = st.median([abs(r['iso']) for r in rows])
            cells.append(f"{(mc/mi if mi else float('nan')):>14.3f}")
        print(f"{N:>4} " + " ".join(cells))
    print()

    # ===================================================================== #
    # (3) is coupling a distance effect?                                    #
    # ===================================================================== #
    if "clustered" in MODES and "dispersed" in MODES:
        print("=== (3) dispersed vs clustered: is coupling a sequence-distance effect? ===")
        print(f"Clustered sites are drawn within a {CLUSTER_WINDOW}-residue window.")
        print("If clustered coupling is much larger at the same N, then the weaker")
        print("coupling seen on the 566-residue reference is substantially a")
        print("consequence of random sites being far apart, and the earlier")
        print("protein_g-vs-zero_polymer comparison was confounded by length.")
        print(f"\n{'N':>4} {'disp spread':>12} {'clus spread':>12} "
              f"{'disp |c|/|iso|':>15} {'clus |c|/|iso|':>15} {'ratio':>8}")
        for N in N_VALUES:
            if N < 2:
                continue
            out = {}
            for mode in ("dispersed", "clustered"):
                rows = [r for d in DRIFTS for s in SEEDS
                        for r in results.get((d, s, mode, N), [])]
                if not rows:
                    out[mode] = None; continue
                mc = st.median([abs(r['coup']) for r in rows])
                mi = st.median([abs(r['iso']) for r in rows])
                out[mode] = (st.median([r['spread'] for r in rows]),
                             mc/mi if mi else float('nan'))
            if out.get("dispersed") and out.get("clustered"):
                ds, dr = out["dispersed"]; cs, cr = out["clustered"]
                print(f"{N:>4} {ds:>12.1f} {cs:>12.1f} {dr:>15.3f} {cr:>15.3f} "
                      f"{(cr/dr if dr else float('nan')):>8.2f}")
        print()

    # ===================================================================== #
    # (4) THE DESIGN QUESTION: throughput                                   #
    # ===================================================================== #
    print("=== (4) throughput: accepted mutations per move, vs the single-site baseline ===")
    print("Acceptance is estimated from the energy term alone,")
    print("    a_N ~ mean_trials min(1, exp(-dU_joint/T)),")
    print("which OMITS the Hastings proposal-ratio term (log a_BA - log a_AB). That")
    print("term is O(1) while exp(-dU/T) spans orders of magnitude, so this is the")
    print("right leading-order estimate, but it is an estimate and can err in either")
    print("direction -- it is not a substitute for running the real sampler.")
    print("Multi-site is worth building only where a_N*N exceeds a_1, shown as 'gain'.")

    for T in T_GRID:
        print(f"\n--- T = {T:g} ---")
        base_rows = [r for d in DRIFTS for s in SEEDS
                     for r in results.get((d, s, BASE_MODE, 1), [])]
        if not base_rows:
            print("  no N=1 cells in this run, so there is no single-site baseline to")
            print("  compare against -- add 1 to NVALS to enable section (4).")
            break
        a1 = st.mean(min(1., math.exp(min(0., -r['dU']/T))) for r in base_rows)
        print(f"  single-site baseline a_1 = {a1:.4g}  (from the N=1 cells of this run)")
        print(f"  {'N':>4} {'a_N':>12} {'a_N*N':>12} {'gain vs a_1':>13} {'verdict':>12}")
        for N in N_VALUES:
            rows = [r for d in DRIFTS for s in SEEDS
                    for r in results.get((d, s, BASE_MODE, N), [])]
            if not rows:
                continue
            aN = st.mean(min(1., math.exp(min(0., -r['dU']/T))) for r in rows)
            thr = aN*N
            gain = thr/a1 if a1 else float('nan')
            verdict = "WINS" if gain > 1.0 else "loses"
            if N == 1:
                verdict = "baseline"
            print(f"  {N:>4} {aN:>12.4g} {thr:>12.4g} {gain:>13.3g} {verdict:>12}")
    print()

    print("Reading (4): a_N collapses roughly geometrically in N whenever dU per")
    print("mutated site is larger than T, because the N-site move must pay every")
    print("site's cost at once with no opportunity to reject them individually.")
    print("A single-site chain pays the same costs but gets N independent")
    print("accept/reject decisions, which is why it is hard to beat. Multi-site")
    print("can only win where dU/site is small relative to T -- look for that")
    print("regime in the per-drift table of section (2) and the T sweep above,")
    print("rather than assuming it exists.")


if __name__ == "__main__":
    main()
