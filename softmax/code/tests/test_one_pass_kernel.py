"""Checks for the one-pass move of ExtendedProteinRateSampler (grad_mode="one_pass",
docs/one_pass_move.tex). The CPU part of the test plan: T3 (exact kernel on a toy
energy, with and without gradient noise, restart) and T9 (compatibility), plus the
toy-energy versions of T1 (same energy) and T2 (same ranking, predicted scale c).

Run from softmax/code/tests:

    python -u test_one_pass_kernel.py

Needs what test_nsite_kernel.py needs, and nothing more: the toy model and toy
energy come from there (the "attention map" is the probability matrix itself and
the energy is a random Potts-type function of it), so the sampler's one-pass code
runs as written, without ESM3.

  A. Gradient. _energy_and_grad returns dU/dp at the one-hot sequence (checked
     against central finite differences along p_ik - p_ij), its U equals
     _exact_energy's bit for bit (T1), and the site_soft gradient over the
     competitors is c*(g_k - mean) with c = p_off/T_sftm, cosine >= 0.999 (T2).
  B. Detailed balance and stationarity of the one-pass kernel, enumerated exactly
     over every sequence of a 3-letter alphabet, for several (L, n_sites, T).
  C. _step in one-pass mode against the exact kernel rows, from fixed starts.
  D. Long chains of _step: the visit histogram matches exp(-U/T)
       D1 deterministic gradient,
       D2 Gaussian noise added to every gradient evaluation (the non-reproducible
          backward of the GPU): exact on the extended state, so the sequence
          marginal is still exp(-U/T),
       D3 negative control: a chain that keeps g(B) even when B is rejected must
          FAIL the same comparison, so D1/D2 are sensitive enough to matter.
  E. Cost. Model calls per move: one-pass makes 1 forward + 1 backward, site_soft
     3 forwards + 2 backwards (counted on the toy model).
  F. sample() in one-pass mode: data.dat rows, and a restart reproduces the straight
     run exactly, both with the saved gradient cache and with it deleted
     (the cache is then recomputed from the restored sequence).
  G. Compatibility (T9): the default grad_mode replays main's sampler row for row
     (needs git; skipped without it), lambda_S != 0 and an unknown grad_mode are
     refused, a pars dict without grad_mode runs site_soft, and _step in one-pass
     mode works without _setup (run_equilibration.py, validate_boltzmann_toy.py).
"""
import importlib.util
import itertools
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_nsite_kernel as nk                                   # noqa: E402  (stubs ESM3, sets up paths)
from test_nsite_kernel import (                                  # noqa: E402
    C, ExtendedProtein, ExtendedProteinRateSampler, log_pointing_prob,
    make_pars, make_sampler, set_reference, set_toy_energy, read_data, toy_energy, TOY,
)
from classes import rate_sampler as rs                           # noqa: E402

LETTERS = ("A", "G", "K")


def one_pass_pars(n_sites, **extra):
    return make_pars(n_sites, grad_mode="one_pass", **extra)


def eprot_of(seq):
    e = ExtendedProtein(sequence=seq, requires_grad=True, device="cpu")
    e.expand()
    return e


def toy_system(L, seed=3, letters=LETTERS):
    set_toy_energy(L, seed=seed)
    sampler = make_sampler(seed=0, letters=list(letters))
    set_reference(sampler, letters[0]*L)
    return sampler


# ------------------------------------------------------------------------ #
# A. The gradient at the exact sequence.                                   #
# ------------------------------------------------------------------------ #
def check_gradient():
    L = 6
    sampler = toy_system(L, seed=8, letters=nk.CANONICAL)
    pars = one_pass_pars(1)
    torch.manual_seed(1)
    worst_fd, worst_cos, scales = 0., 1., []
    for trial in range(10):
        seq = "".join(nk.CANONICAL[i] for i in torch.randint(0, 20, (L,)).tolist())
        e = eprot_of(seq)
        U, U_am, g, e = sampler._energy_and_grad(e, pars)
        U_ex, U_am_ex, _ = sampler._exact_energy(eprot_of(seq), pars)
        assert U == U_ex and U_am == U_am_ex, f"one-pass U {U!r} != exact U {U_ex!r} at {seq}"
        assert g.shape == (L, len(C.SEQUENCE_USED_VOCAB))

        # finite differences of U along p_ik - p_ij, in float64
        P0 = e.get_probs().double()
        f = lambda P: (P*TOY["h"].double()).sum() + torch.einsum("ia,ijab,jb->", P, TOY["J"].double(), P)
        for i in range(L):
            cur = C.SEQUENCE_USED_VOCAB.index(seq[i])
            comps = [c for c in sampler._competition_indices(cur).tolist()]
            for k, j in [(comps[0], comps[1]), (comps[2], cur), (comps[5], comps[9])]:
                d = torch.zeros_like(P0)
                d[i, k], d[i, j] = 1., -1.
                h = 1e-4
                fd = ((f(P0 + h*d) - f(P0 - h*d))/(2*h)).item()
                an = (g[i, k] - g[i, j]).item()
                worst_fd = max(worst_fd, abs(fd - an)/max(1e-6, abs(fd)))

        # site_soft gradient over the competitors = c * centred g (T2, toy version)
        for i in range(L):
            cur = C.SEQUENCE_USED_VOCAB.index(seq[i])
            comps = sampler._competition_indices(cur)
            old = sampler._grad_pass_sites(e, [i], make_pars(1))[0][comps]
            new = g[i][comps]
            old_c, new_c = old - old.mean(), new - new.mean()
            worst_cos = min(worst_cos, (old_c @ new_c / (old_c.norm()*new_c.norm())).item())
            scales.append((old_c @ new_c / (new_c @ new_c)).item())

    K = len(C.SEQUENCE_USED_VOCAB)
    p_cur = 1./(1. + (K - 1)*math.exp(-10.))
    c = p_cur*math.exp(-10.)/0.1
    s_med = float(np.median(scales))
    assert worst_fd < 1e-3, f"gradient differs from finite differences (worst relative error {worst_fd:.2e})"
    assert worst_cos >= 0.999, f"site_soft and one-pass gradients rank competitors differently (cosine {worst_cos:.5f})"
    assert abs(s_med/c - 1.) < 0.01, f"fitted scale {s_med:.4e}, predicted c = {c:.4e}"
    print(f"  10 sequences x {L} sites: U bit-identical to _exact_energy, worst finite-difference error {worst_fd:.1e}, "
          f"worst centred cosine {worst_cos:.6f}, fitted scale {s_med:.4e} vs c = {c:.4e} "
          f"(spread {min(scales)/c:.4f}..{max(scales)/c:.4f} of c)")
    return c


# ------------------------------------------------------------------------ #
# B. Exact one-pass kernel.                                                #
# ------------------------------------------------------------------------ #
def exact_kernel_one_pass(sampler, pars, L, letters):
    states = ["".join(s) for s in itertools.product(letters, repeat=L)]
    index = {s: k for k, s in enumerate(states)}
    n = pars["n_sites"]
    sigma = pars["dt"]*math.sqrt(pars["T"]/pars["M"])
    step_coef = pars["dt"]**2/(2*pars["M"])
    U, G = np.zeros(len(states)), {}
    for a, seq in enumerate(states):
        U[a], _, G[seq], _ = sampler._energy_and_grad(eprot_of(seq), pars)

    P = np.zeros((len(states), len(states)))
    subsets = list(itertools.combinations(range(L), n))
    spread = 0.
    for a, seq_a in enumerate(states):
        eA = eprot_of(seq_a)
        for S in subsets:
            sites = list(S)
            muA = -step_coef*G[seq_a][sites]
            cur = [int(eA.logits[s].argmax().item()) for s in sites]
            comps = [sampler._competition_indices(c) for c in cur]
            total, logs = 0., []
            for local in itertools.product(*[range(len(c)) for c in comps]):
                log_qf, tgt = 0., []
                for k in range(n):
                    log_qf += log_pointing_prob(muA[k][comps[k]], sigma, torch.tensor(local[k]), pars["n_quad"]).item()
                    tgt.append(int(comps[k][local[k]].item()))
                total += math.exp(log_qf)
                logs.append(log_qf)
                seq_b = sampler._substitute_sites(eA, sites, tgt).sequence
                b = index[seq_b]
                log_qr = sampler._reverse_logprob(-step_coef*G[seq_b][sites], cur, tgt, sigma, pars)
                log_ratio = -(U[b] - U[a])/pars["T"] + (log_qr - log_qf)
                P[a, b] += math.exp(log_qf)*math.exp(min(0., log_ratio))/math.comb(L, n)
            spread = max(spread, max(logs) - min(logs))
            assert abs(total - 1.) < 1e-5, f"forward proposal not normalized: {total}"
        P[a, a] = 1. - P[a].sum()
    return states, U, P, spread


def boltzmann(U, T):
    pi = np.exp(-(U - U.min())/T)
    return pi/pi.sum()


def check_detailed_balance(L, n, T, dt=2.0):
    sampler = toy_system(L)
    pars = one_pass_pars(n, T=T, dt=dt)
    states, U, P, spread = exact_kernel_one_pass(sampler, pars, L, list(LETTERS))
    assert spread > 0.01, f"proposal nearly uniform (spread {spread:.4f}): the gradient is not exercised"
    pi = boltzmann(U, T)
    flow = pi[:, None]*P
    assert np.all(P >= -1e-6), f"negative transition probability (min {P.min():.2e})"
    assert np.allclose(flow, flow.T, rtol=1e-6, atol=1e-12), f"detailed balance violated (L={L}, n={n}, T={T})"
    assert np.allclose(pi @ P, pi, rtol=1e-6, atol=1e-12), f"stationarity violated (L={L}, n={n}, T={T})"
    print(f"  L={L} n={n} T={T}: {len(states)} states, max |flow - flow^T| = {np.max(np.abs(flow - flow.T)):.2e}, "
          f"log-proposal spread {spread:.2f}, mean acceptance {float(np.mean(1. - np.diag(P))):.3f}")


# ------------------------------------------------------------------------ #
# C. _step itself against the kernel rows.                                 #
# ------------------------------------------------------------------------ #
def check_step_matches_kernel(L=4, n=2, T=1.0, dt=2.0, draws=4000):
    sampler = toy_system(L)
    pars = one_pass_pars(n, T=T, dt=dt)
    states, _, P, _ = exact_kernel_one_pass(sampler, pars, L, list(LETTERS))
    index = {s: k for k, s in enumerate(states)}
    for a in (0, len(states)//2, len(states) - 1):
        eA = eprot_of(states[a])
        counts, accepted = np.zeros(len(states)), 0
        for _ in range(draws):
            eNew, info = sampler._step(eA, None, pars)
            counts[index[eNew.sequence]] += 1
            accepted += info["accepted"]
        row = np.clip(P[a], 0., 1.)   # float32 quadrature leaves entries of order -1e-8
        se = np.sqrt(row*(1. - row)/draws)
        assert np.all(np.abs(counts/draws - row) < 5*se + 2e-3), f"_step frequencies differ from the kernel at {states[a]}"
        acc = 1. - row[a]
        assert abs(accepted/draws - acc) < 5*np.sqrt(acc*(1. - acc)/draws) + 2e-3, \
            f"_step accepts {accepted/draws:.4f}, kernel {acc:.4f} at {states[a]}"
    print(f"  3 fixed starts, {draws} one-pass _step calls each: frequencies and acceptance match the kernel rows")


# ------------------------------------------------------------------------ #
# D. Long chains.                                                          #
# ------------------------------------------------------------------------ #
class NoisyGradSampler(ExtendedProteinRateSampler):
    """Adds N(0, noise^2) to every gradient evaluation, from its own generator,
    as the GPU's non-reproducible backward does. U is left exact."""
    noise = 0.

    def _energy_and_grad(self, eprot, pars):
        U, U_am, g, eprot = super()._energy_and_grad(eprot, pars)
        if self.noise > 0.:
            g = g + self.noise*torch.randn(g.shape, generator=self.noise_gen)
        return U, U_am, g, eprot


class AlwaysCacheBSampler(NoisyGradSampler):
    """The bug the cache must avoid: g(B) kept as the next g(A) even when B is
    rejected, so the forward proposal from A uses B's gradient."""

    def _step_one_pass(self, eprot, pars, sites=None):
        out = super()._step_one_pass(eprot, pars, sites)
        if not out[1]["accepted"]:
            self._grad_cache["grad"] = self._last_grad_B
        return out

    def _energy_and_grad(self, eprot, pars):
        out = super()._energy_and_grad(eprot, pars)
        self._last_grad_B = out[2]
        return out


def make_chain_sampler(cls, noise, seed):
    sampler = object.__new__(cls)
    sampler.name = cls.__name__
    sampler._init_attributes()
    sampler.model = nk.ToyModel()
    sampler.generator = nk.CustomGenerator(seed=seed, device="cpu")
    sampler._canonical_idx = torch.tensor([C.SEQUENCE_USED_VOCAB.index(a) for a in LETTERS], dtype=torch.long)
    sampler.noise = noise
    sampler.noise_gen = torch.Generator().manual_seed(1000 + seed)
    return sampler


def run_chain(cls, noise, L, pars, moves, seed):
    sampler = make_chain_sampler(cls, noise, seed)
    set_reference(sampler, LETTERS[0]*L)
    states = ["".join(s) for s in itertools.product(LETTERS, repeat=L)]
    index = {s: k for k, s in enumerate(states)}
    e = eprot_of(LETTERS[-1]*L)
    visits = np.empty(moves, dtype=np.int64)
    acc = 0
    for m in range(moves):
        e, info = sampler._step(e, None, pars)
        visits[m] = index[e.sequence]
        acc += info["accepted"]
    return visits, acc/moves


def compare_to_boltzmann(visits, pi, burn, batches=40):
    """Max |z| over states of (frequency - pi), with batch-means standard errors,
    and the total variation distance."""
    v = visits[burn:]
    n = len(v)//batches
    freq = np.stack([np.bincount(v[b*n:(b + 1)*n], minlength=len(pi))/n for b in range(batches)])
    mean = freq.mean(axis=0)
    se = freq.std(axis=0, ddof=1)/np.sqrt(batches)
    z = np.abs(mean - pi)/np.maximum(se, 1e-4)
    return float(z.max()), 0.5*float(np.abs(mean - pi).sum())


def check_long_chains(L=3, T=1.0, dt=2.0, moves=60000, chains=3):
    set_toy_energy(L, seed=5, scale=0.8)
    ref = make_chain_sampler(NoisyGradSampler, 0., 0)
    set_reference(ref, LETTERS[0]*L)
    states = ["".join(s) for s in itertools.product(LETTERS, repeat=L)]
    U = np.array([ref._energy_and_grad(eprot_of(s), one_pass_pars(1))[0] for s in states])
    pi = boltzmann(U, T)
    pars = one_pass_pars(1, T=T, dt=dt)
    gaps = []
    for s in states:
        g = ref._energy_and_grad(eprot_of(s), pars)[2]
        for i in range(L):
            col = g[i][ref._canonical_idx]
            gaps.append(float(col.max() - col.min()))
    noise = float(np.median(gaps))   # noise as large as the typical gradient gap
    results = {}
    for label, cls, nz in [("D1 deterministic", NoisyGradSampler, 0.),
                           ("D2 noisy gradient", NoisyGradSampler, noise),
                           ("D3 keeps rejected g(B)", AlwaysCacheBSampler, 0.)]:
        zs, tvs, accs = [], [], []
        for c in range(chains):
            visits, a = run_chain(cls, nz, L, pars, moves, seed=c)
            z, tv = compare_to_boltzmann(visits, pi, burn=moves//20)
            zs.append(z); tvs.append(tv); accs.append(a)
        results[label] = (max(zs), float(np.mean(tvs)), float(np.mean(accs)))
        print(f"  {label:<24} {chains} chains x {moves} moves: max |z| = {max(zs):5.1f}, "
              f"mean TV = {np.mean(tvs):.4f}, acceptance {np.mean(accs):.3f}" + (f" (noise sd {nz:.2f})" if nz else ""))
    # 27 states x 3 chains: |z| under 4.5 is a 1e-4-level false alarm per comparison.
    assert results["D1 deterministic"][0] < 4.5, "deterministic one-pass chain does not sample exp(-U/T)"
    assert results["D2 noisy gradient"][0] < 4.5, "one-pass chain with gradient noise does not sample exp(-U/T)"
    assert results["D3 keeps rejected g(B)"][0] > 6., "the negative control was not detected: D1/D2 are not sensitive"
    print(f"  pi over {len(states)} states, max pi {pi.max():.3f}; the negative control fails as it should")


# ------------------------------------------------------------------------ #
# E. Model calls per move.                                                 #
# ------------------------------------------------------------------------ #
class CountingModel(nk.ToyModel):
    def __init__(self):
        self.forward = 0
        self.backward = 0

    def predict_attention(self, sequence_probs):
        self.forward += 1
        out = sequence_probs*1.
        if out.requires_grad:
            out.register_hook(self._count)
        return out

    def _count(self, grad):
        self.backward += 1


def check_cost(moves=200):
    L = 5
    for mode, want in [("one_pass", (1, 1)), ("site_soft", (3, 2))]:
        sampler = toy_system(L)
        model = CountingModel()
        sampler.model = model
        set_reference(sampler, "A"*L)
        pars = make_pars(2, T=1.0, dt=2.0 if mode == "one_pass" else 20.0, grad_mode=mode)
        e = eprot_of("AGKAG")
        U, _, e = sampler._exact_energy(e, pars)
        if mode == "one_pass":
            e, info = sampler._step(e, U, pars)  # fills the cache: one extra pass at the chain start
            U = info["U_B"] if info["accepted"] else U
        model.forward = model.backward = 0
        for _ in range(moves):
            e, info = sampler._step(e, U, pars)
            if info["accepted"]:
                U = info["U_B"]
        per = (model.forward/moves, model.backward/moves)
        assert per == want, f"{mode}: {per} forward/backward per move, expected {want}"
        print(f"  {mode:<9}: {per[0]:.0f} forward + {per[1]:.0f} backward per move over {moves} moves")


# ------------------------------------------------------------------------ #
# F. sample(): rows and restart.                                           #
# ------------------------------------------------------------------------ #
def check_sample_and_restart():
    REF = "MKTAYIAK"
    set_toy_energy(len(REF), seed=4)
    base = dict(T=1.0, dt=0.5, M=1.0, init_muts=4, n_sites=2, seed=7, lambda_am=1.0, grad_mode="one_pass")

    def settings(path, restart=False):
        os.makedirs(os.path.join(path, "eprot"), exist_ok=True)
        return dict(results_dir=path, eprot_dir=os.path.join(path, "eprot"), log_step=20,
                    print_step=1000, restart=restart, device="cpu", num_threads=1)

    drop_time = lambda rows: [[v for k, v in enumerate(r) if k != 1] for r in rows]
    with tempfile.TemporaryDirectory() as tmp:
        straight_dir = os.path.join(tmp, "straight")
        make_sampler(seed=7).sample(REF, dict(base, moves=100), settings(straight_dir))
        _, straight = read_data(os.path.join(straight_dir, "data.dat"))
        assert [int(r[0]) for r in straight] == list(range(101))
        acc = np.mean([int(r[13]) for r in straight[1:]])
        assert 0.05 < acc < 0.95, f"acceptance {acc:.2f}: pick a dt that exercises both outcomes"
        for r in straight[1:]:
            assert len(r[6].split(",")) == 2 and set(r[8].split(",")) <= set(nk.CANONICAL), r
        assert os.path.isfile(os.path.join(straight_dir, "grad_cache.pt"))

        for keep_cache in (True, False):
            run_dir = os.path.join(tmp, f"run_{keep_cache}")
            make_sampler(seed=7).sample(REF, dict(base, moves=60), settings(run_dir))
            open(os.path.join(run_dir, "pars.txt"), "w").close()   # restart checks for it
            if not keep_cache:
                os.remove(os.path.join(run_dir, "grad_cache.pt"))
            make_sampler(seed=7).sample(REF, dict(base, moves=100), settings(run_dir, restart=True))
            _, restarted = read_data(os.path.join(run_dir, "data.dat"))
            assert drop_time(restarted) == drop_time(straight), f"restart (cache kept: {keep_cache}) differs from the straight run"
    print(f"  100 one-pass moves with n_sites=2 (acceptance {acc:.2f}); restart at 60 reproduces the straight run "
          f"with the saved cache and with it deleted")


# ------------------------------------------------------------------------ #
# G. Compatibility.                                                        #
# ------------------------------------------------------------------------ #
def load_main_sampler():
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        for ref in ("origin/main", "main"):
            res = subprocess.run(["git", "show", f"{ref}:softmax/code/classes/rate_sampler.py"], cwd=root, capture_output=True, text=True)
            if res.returncode == 0:
                break
        else:
            return None, None
    except (OSError, subprocess.CalledProcessError):
        return None, None
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(res.stdout)
    tmp.close()
    spec = importlib.util.spec_from_file_location("classes.rate_sampler_main", tmp.name)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "classes"
    spec.loader.exec_module(module)
    module.compute_U_am = rs.compute_U_am
    os.unlink(tmp.name)
    return module, ref


def check_compatibility():
    REF = "MKTAYIAK"
    set_toy_energy(len(REF), seed=4)
    base = dict(T=1.0, dt=1.0, M=1.0, init_muts=4, n_sites=3, seed=7, lambda_am=1.0, moves=60)

    def settings(path):
        os.makedirs(os.path.join(path, "eprot"), exist_ok=True)
        return dict(results_dir=path, eprot_dir=os.path.join(path, "eprot"), log_step=20,
                    print_step=1000, restart=False, device="cpu", num_threads=1)

    def sampler_of(cls):
        s = object.__new__(cls)
        s.name = "ExtendedProteinRateSampler"
        s._init_attributes()
        s.model = nk.ToyModel()
        s.generator = nk.CustomGenerator(seed=7, device="cpu")
        return s

    drop_time = lambda rows: [[v for k, v in enumerate(r) if k != 1] for r in rows]
    main_mod, ref = load_main_sampler()
    with tempfile.TemporaryDirectory() as tmp:
        sampler_of(ExtendedProteinRateSampler).sample(REF, dict(base), settings(f"{tmp}/new"))
        sampler_of(ExtendedProteinRateSampler).sample(REF, dict(base, grad_mode="site_soft"), settings(f"{tmp}/explicit"))
        _, new = read_data(f"{tmp}/new/data.dat")
        _, explicit = read_data(f"{tmp}/explicit/data.dat")
        assert drop_time(new) == drop_time(explicit), "grad_mode='site_soft' differs from the default"
        assert not os.path.exists(f"{tmp}/new/grad_cache.pt"), "site_soft wrote a gradient cache"
        if main_mod is not None:
            sampler_of(main_mod.ExtendedProteinRateSampler).sample(REF, dict(base), settings(f"{tmp}/main"))
            _, old = read_data(f"{tmp}/main/data.dat")
            assert drop_time(new) == drop_time(old), f"default grad_mode does not replay {ref}'s sampler"
            print(f"  default grad_mode: 60 moves, n_sites=3, identical to {ref}'s rate_sampler.py row for row")
        else:
            print("  (git not available: replay against main skipped)")

        for bad, why in [(dict(grad_mode="one_pass", lambda_S=0.1), "lambda_S"), (dict(grad_mode="onepass"), "grad_mode")]:
            try:
                sampler_of(ExtendedProteinRateSampler).sample(REF, dict(base, **bad), settings(f"{tmp}/bad"))
            except AssertionError as err:
                assert why in str(err), str(err)
            else:
                raise AssertionError(f"{bad} was accepted")
        print("  lambda_S=0.1 with one_pass and grad_mode='onepass' refused")

    # Without _setup, as run_equilibration.py and validate_boltzmann_toy.py call it.
    sampler = toy_system(4)
    pars = make_pars(1, T=1.0, dt=2.0)
    for key in ("n_sites",):
        del pars[key]
    e = eprot_of("AGKA")
    U, _, e = sampler._exact_energy(e, pars)
    sampler._step(e, U, pars)          # no grad_mode key: site_soft
    assert sampler._grad_cache is None
    pars["grad_mode"] = "one_pass"
    for _ in range(50):
        e, info = sampler._step(e, U, pars, sites=[2])
        assert info["site"] == "2", info["site"]
        if info["accepted"]:
            U = info["U_B"]
    assert sampler._grad_cache["key"][0] == e.sequence, "cache does not hold the current state"
    U_chk, _, g_chk, _ = sampler._energy_and_grad(eprot_of(e.sequence), pars)
    assert U_chk == sampler._grad_cache["U"] and torch.equal(g_chk, sampler._grad_cache["grad"])
    try:
        sampler._step(e, U, dict(pars, lambda_S=0.5))
    except AssertionError as err:
        assert "lambda_S" in str(err)
    else:
        raise AssertionError("lambda_S != 0 accepted by _step in one-pass mode")
    print("  without _setup: no grad_mode runs site_soft; one_pass fills and keeps the cache; sites= fixes the sites")


if __name__ == "__main__":
    print("A. gradient at the exact sequence (T1, T2 on the toy)")
    check_gradient()
    print("B. detailed balance and stationarity of the one-pass kernel (exact enumeration)")
    for L, n, T in [(4, 1, 1.0), (4, 2, 1.0), (4, 2, 0.5), (5, 2, 1.0), (5, 3, 2.0)]:
        check_detailed_balance(L, n, T)
    print("C. one-pass _step against the exact kernel rows")
    check_step_matches_kernel()
    print("D. long chains against exp(-U/T)")
    check_long_chains()
    print("E. model calls per move")
    check_cost()
    print("F. sample(), restart")
    check_sample_and_restart()
    print("G. compatibility")
    check_compatibility()
    print("ALL ONE-PASS CHECKS PASSED")
