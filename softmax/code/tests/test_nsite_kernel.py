"""Checks for the n_sites move of ExtendedProteinRateSampler (classes/rate_sampler.py).

Run from softmax/code/tests:

    python -u test_nsite_kernel.py

Needs what the sampler itself needs: torch, numpy, pandas, huggingface_hub and the
customs package. No ESM3 weights: the "attention map" is the probability matrix
itself, and the energy is a random pairwise (Potts-type) function of it, so the
sampler's relaxed gradient, proposal, reverse proposal and output code run as
written. One piece is re-implemented here: the Metropolis ratio inside
exact_kernel. Check F runs _step itself against those exact rows.

  A. Detailed balance and stationarity. exact_kernel enumerates the kernel on every
     sequence of length L over a 3-letter canonical alphabet, for several
     (L, n_sites, T), from _grad_pass_sites, _substitute_sites, _reverse_logprob,
     _exact_energy and log_pointing_prob. Detailed balance and stationarity of
     exp(-U/T) must hold.
  B. Sampling. _draw_targets must realize the pointing probabilities that
     log_pointing_prob reports (Monte Carlo over the momentum draws).
  C. Starts. _mutate_canonical writes canonical residues only.
  D. Guard. _setup refuses a reference that holds a non-canonical residue.
  E. Output and restart. sample() with n_sites=3 writes the expected data.dat rows,
     and a restarted run reproduces the straight run exactly.
  F. _step itself. From fixed starts, its transition frequencies and acceptance
     match the exact kernel rows.
  G. A pars dict without n_sites, as run_equilibration.py passes, runs single-site.
"""
import itertools
import math
import os
import sys
import tempfile
import types
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
CUSTOMS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "customs"))
sys.path[:0] = [CODE, CUSTOMS]


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


# The sampler imports the real ESM3 model and structure config. This test replaces
# the model with a toy, so those imports are stubbed out.
_stub("custom_esm.models.esm3", ESM3=object)
_stub("custom_esm.sdk.api", ESM3InferenceClient=object)
_stub("utils.predictions", init_structure_config=lambda **kwargs: None)

import custom_esm.utils.constants.esm3 as C                     # noqa: E402
from classes import rate_sampler as rs                           # noqa: E402
from classes.ExtendedProtein import ExtendedProtein              # noqa: E402
from generator.custom_generator import CustomGenerator           # noqa: E402
from utils.proposals import log_pointing_prob                    # noqa: E402

ExtendedProteinRateSampler = rs.ExtendedProteinRateSampler
CANONICAL = [a for a in C.SEQUENCE_USED_VOCAB if a not in rs._NONCANONICAL_RESIDUES]


class ToyModel:
    """Stands in for ESM3: the 'attention map' is the probability matrix itself."""

    def to(self, device):
        return self

    def parameters(self):
        return []

    def predict_attention(self, sequence_probs):
        return sequence_probs


TOY = {}


def toy_energy(P):
    return (P*TOY["h"]).sum() + torch.einsum("ia,ijab,jb->", P, TOY["J"], P)


# The attention-map term of the energy, replaced by the toy energy relative to the reference.
rs.compute_U_am = lambda am, ref_am: toy_energy(am) - toy_energy(ref_am)


def set_toy_energy(L, seed, scale=0.6):
    g = torch.Generator().manual_seed(seed)
    K = len(C.SEQUENCE_USED_VOCAB)
    TOY["h"] = (scale*torch.randn(L, K, generator=g)).float()
    J = torch.zeros(L, L, K, K)
    for i in range(L):
        for j in range(i + 1, L):
            J[i, j] = scale*torch.randn(K, K, generator=g)
    TOY["J"] = J.float()


def make_sampler(seed=0, letters=None):
    sampler = object.__new__(ExtendedProteinRateSampler)
    sampler.name = "ExtendedProteinRateSampler"
    sampler._init_attributes()
    sampler.model = ToyModel()
    sampler.generator = CustomGenerator(seed=seed, device="cpu")
    if letters is not None:
        sampler._canonical_idx = torch.tensor([C.SEQUENCE_USED_VOCAB.index(a) for a in letters], dtype=torch.long)
    return sampler


def set_reference(sampler, ref_seq):
    sampler.ref_eprot = ExtendedProtein(sequence=ref_seq, requires_grad=False, device="cpu")
    sampler.ref_eprot.expand()
    sampler.ref_eprot.am = sampler.model.predict_attention(sampler.ref_eprot.get_probs())


def make_pars(n_sites, T=1.0, dt=1.0, M=1.0, **extra):
    pars = {
        "moves": 1, "T": T, "dt": dt, "M": M, "T_sftm": 0.1,
        "lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0,
        "n_quad": 40, "init_muts": 0, "n_sites": n_sites, "eps": 1e-9, "seed": 0,
    }
    pars.update(extra)
    return pars


# ------------------------------------------------------------------------ #
# A. Exact kernel on a small system, built from the sampler's own pieces.  #
# ------------------------------------------------------------------------ #
def exact_kernel(sampler, pars, L, letters):
    states = ["".join(s) for s in itertools.product(letters, repeat=L)]
    index = {s: k for k, s in enumerate(states)}
    n = pars["n_sites"]
    sigma = pars["dt"]*math.sqrt(pars["T"]/pars["M"])
    step_coef = pars["dt"]**2/(2*pars["M"])
    P = np.zeros((len(states), len(states)))
    U = np.zeros(len(states))
    subsets = list(itertools.combinations(range(L), n))

    for a, seq_a in enumerate(states):
        eA = ExtendedProtein(sequence=seq_a, requires_grad=True, device="cpu")
        eA.expand()
        U[a], _, _ = sampler._exact_energy(eA, pars)
        for S in subsets:
            sites = list(S)
            muA = -step_coef*sampler._grad_pass_sites(eA, sites, pars)
            cur = [int(eA.logits[s].argmax().item()) for s in sites]
            comps = [sampler._competition_indices(c) for c in cur]
            total = 0.
            for local in itertools.product(*[range(len(c)) for c in comps]):
                log_qf = 0.
                tgt = []
                for k in range(n):
                    log_qf += log_pointing_prob(muA[k][comps[k]], sigma, torch.tensor(local[k]), pars["n_quad"]).item()
                    tgt.append(int(comps[k][local[k]].item()))
                total += math.exp(log_qf)

                eB = sampler._substitute_sites(eA, sites, tgt)
                muB = -step_coef*sampler._grad_pass_sites(eB, sites, pars)
                log_qr = sampler._reverse_logprob(muB, cur, tgt, sigma, pars)
                UB, _, _ = sampler._exact_energy(eB, pars)

                log_ratio = -(UB - U[a])/pars["T"] + (log_qr - log_qf)
                assert not math.isnan(log_ratio), (seq_a, sites, tgt)
                accept = math.exp(min(0., log_ratio))
                P[a, index["".join(eB.sequence)]] += math.exp(log_qf)*accept/math.comb(L, n)
            assert abs(total - 1.) < 1e-5, f"forward proposal not normalized: {total} at {seq_a}, sites {sites}"
        P[a, a] = 1. - P[a].sum()
    return states, U, P


def check_detailed_balance(L, n, T, letters=("A", "G", "K"), seed=3):
    set_toy_energy(L, seed=seed)
    sampler = make_sampler(seed=0, letters=list(letters))
    set_reference(sampler, letters[0]*L)
    pars = make_pars(n, T=T)
    states, U, P = exact_kernel(sampler, pars, L, list(letters))

    pi = np.exp(-(U - U.min())/T)
    pi /= pi.sum()
    flow = pi[:, None]*P
    assert np.all(P >= -1e-12), "negative transition probability"
    assert np.allclose(flow, flow.T, rtol=1e-6, atol=1e-12), f"detailed balance violated (L={L}, n={n}, T={T})"
    assert np.allclose(pi @ P, pi, rtol=1e-6, atol=1e-12), f"stationarity violated (L={L}, n={n}, T={T})"
    moved = P[np.arange(len(states)), np.arange(len(states))]
    print(f"  L={L} n={n} T={T}: {len(states)} states, max |flow - flow^T| = {np.max(np.abs(flow - flow.T)):.2e}, "
          f"mean acceptance of a move = {float(np.mean(1. - moved)):.3f}")


# ------------------------------------------------------------------------ #
# B. Targets drawn by _draw_targets follow the pointing probabilities.      #
# ------------------------------------------------------------------------ #
def check_draw_targets(draws=30000):
    sampler = make_sampler(seed=11)
    n = 2
    pars = make_pars(n)
    K = len(C.SEQUENCE_USED_VOCAB)
    g = torch.Generator().manual_seed(5)
    mu = (0.8*torch.randn(n, K, generator=g)).float()
    cur = [C.SEQUENCE_USED_VOCAB.index("A"), C.SEQUENCE_USED_VOCAB.index("L")]
    sigma = 1.0

    comps = [sampler._competition_indices(c) for c in cur]
    exact = [np.exp([log_pointing_prob(mu[k][comps[k]], sigma, torch.tensor(j), 40).item()
                     for j in range(len(comps[k]))]) for k in range(n)]
    counts = [np.zeros(len(comps[k])) for k in range(n)]
    joint = np.zeros((len(comps[0]), len(comps[1])))

    for draw in range(draws):
        tgt, log_a = sampler._draw_targets(mu, cur, sigma, pars)
        local = [int((comps[k] == tgt[k]).nonzero()[0, 0]) for k in range(n)]
        for k in range(n):
            counts[k][local[k]] += 1
        joint[local[0], local[1]] += 1
        if draw < 300:
            expected = sum(log_pointing_prob(mu[k][comps[k]], sigma, torch.tensor(local[k]), 40).item() for k in range(n))
            assert abs(log_a - expected) < 1e-4, (log_a, expected)

    for k in range(n):
        freq = counts[k]/draws
        se = np.sqrt(exact[k]*(1. - exact[k])/draws)
        assert np.all(np.abs(freq - exact[k]) < 5*se + 1e-3), f"site {k}: sampled {freq} vs pointing {exact[k]}"
    product = np.outer(exact[0], exact[1])
    se = np.sqrt(product*(1. - product)/draws)
    assert np.all(np.abs(joint/draws - product) < 5*se + 1e-3), "joint targets are not the product of the per-site proposals"
    print(f"  {draws} draws: per-site and joint target frequencies match the pointing probabilities")


# ------------------------------------------------------------------------ #
# C. Canonical starts.                                                     #
# ------------------------------------------------------------------------ #
def check_mutate_canonical():
    sampler = make_sampler(seed=2)
    counts = Counter()
    for _ in range(300):
        seq = sampler._mutate_canonical("A"*30, 20)
        assert set(seq) <= set(CANONICAL), f"non-canonical residue in start: {seq}"
        counts.update(seq)
    assert set(counts) == set(CANONICAL), "some canonical residues never drawn"
    print(f"  300 starts of 20 mutations: only canonical residues, all {len(CANONICAL)} drawn")


# ------------------------------------------------------------------------ #
# D. Reference guard.                                                      #
# ------------------------------------------------------------------------ #
def check_reference_guard():
    with tempfile.TemporaryDirectory() as tmp:
        settings = dict(results_dir=tmp, eprot_dir=tmp, log_step=1, print_step=1,
                        restart=False, device="cpu", num_threads=1)
        sampler = make_sampler(seed=0)
        try:
            sampler._setup("AAAXAAAA", make_pars(1, moves=5), settings)
        except AssertionError as err:
            assert "non-canonical" in str(err), str(err)
        else:
            raise AssertionError("a reference holding X was accepted")
    print("  reference with X refused")


# ------------------------------------------------------------------------ #
# E. sample(): data.dat rows, and restart reproduces the straight run.     #
# ------------------------------------------------------------------------ #
def read_data(path):
    rows = [line.rstrip("\n").split("\t") for line in open(path) if line.strip() and not line.startswith("#")]
    return rows[0], rows[1:]


def check_sample_and_restart():
    REF = "MKTAYIAK"
    set_toy_energy(len(REF), seed=4)
    base = dict(T=1.0, dt=1.0, M=1.0, init_muts=4, n_sites=3, seed=7, lambda_am=1.0)

    def settings(path, restart=False):
        os.makedirs(os.path.join(path, "eprot"), exist_ok=True)
        return dict(results_dir=path, eprot_dir=os.path.join(path, "eprot"), log_step=20,
                    print_step=1000, restart=restart, device="cpu", num_threads=1)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        straight_dir = os.path.join(tmp, "straight")

        make_sampler(seed=7).sample(REF, dict(base, moves=60), settings(run_dir))
        open(os.path.join(run_dir, "pars.txt"), "w").close()   # restart checks for it
        header, rows = read_data(os.path.join(run_dir, "data.dat"))
        assert header[:9] == ["move", "time", "U", "U_am", "U_structure_ce", "Hd_to_ref", "site", "cur_aa", "proposed_aa"], header
        assert [int(r[0]) for r in rows] == list(range(61)), "data.dat rows must be moves 0..60, each once"
        for r in rows[1:]:
            sites = [int(s) for s in r[6].split(",")]
            assert len(sites) == 3 and len(set(sites)) == 3 and all(0 <= s < len(REF) for s in sites), r
            assert len(r[7].split(",")) == 3 and len(r[8].split(",")) == 3, r
            assert set(r[8].split(",")) <= set(CANONICAL), r
            assert r[13] in ("0", "1"), r

        make_sampler(seed=7).sample(REF, dict(base, moves=100), settings(run_dir, restart=True))
        make_sampler(seed=7).sample(REF, dict(base, moves=100), settings(straight_dir))
        _, restarted = read_data(os.path.join(run_dir, "data.dat"))
        _, straight = read_data(os.path.join(straight_dir, "data.dat"))
        drop_time = lambda rs_: [[v for k, v in enumerate(r) if k != 1] for r in rs_]
        assert drop_time(restarted) == drop_time(straight), "restarted run does not reproduce the straight run"
    print("  60 moves with n_sites=3, then restart to 100: rows match the straight run exactly")


# ------------------------------------------------------------------------ #
# F. _step itself: from fixed starts, its transition frequencies and       #
#    acceptance must match the exact kernel rows. exact_kernel re-implements #
#    the accept rule, so this is the check of _step's own ratio and draws.  #
# ------------------------------------------------------------------------ #
def check_step_matches_kernel(L=4, n=2, T=1.0, letters=("A", "G", "K"), seed=3, draws=4000):
    set_toy_energy(L, seed=seed)
    sampler = make_sampler(seed=0, letters=list(letters))
    set_reference(sampler, letters[0]*L)
    pars = make_pars(n, T=T)
    states, _, P = exact_kernel(sampler, pars, L, list(letters))
    index = {s: k for k, s in enumerate(states)}

    for a in (0, len(states)//2, len(states) - 1):
        eA = ExtendedProtein(sequence=states[a], requires_grad=True, device="cpu")
        eA.expand()
        U_A, _, eA = sampler._exact_energy(eA, pars)
        counts = np.zeros(len(states))
        accepted = 0
        for _ in range(draws):
            eNew, info = sampler._step(eA, U_A, pars)
            counts[index[eNew.sequence]] += 1
            accepted += info["accepted"]
        row = P[a]
        se = np.sqrt(row*(1. - row)/draws)
        assert np.all(np.abs(counts/draws - row) < 5*se + 2e-3), f"_step frequencies differ from the kernel at {states[a]}"
        acc = 1. - row[a]
        assert abs(accepted/draws - acc) < 5*np.sqrt(acc*(1. - acc)/draws) + 2e-3, \
            f"_step accepts {accepted/draws:.4f}, kernel {acc:.4f} at {states[a]}"
    print(f"  3 fixed starts, {draws} _step calls each: frequencies and acceptance match the kernel rows")


# ------------------------------------------------------------------------ #
# G. A pars dict without n_sites (run_equilibration.py builds one and       #
#    bypasses _setup) must run as a single-site move.                       #
# ------------------------------------------------------------------------ #
def check_step_without_n_sites(draws=20):
    set_toy_energy(4, seed=3)
    sampler = make_sampler(seed=0, letters=["A", "G", "K"])
    set_reference(sampler, "AAAA")
    pars = make_pars(1)
    del pars["n_sites"]
    eA = ExtendedProtein(sequence="AGKA", requires_grad=True, device="cpu")
    eA.expand()
    U_A, _, eA = sampler._exact_energy(eA, pars)
    for _ in range(draws):
        _, info = sampler._step(eA, U_A, pars)
        assert len(info["site"].split(",")) == 1, info["site"]
    print("  pars without n_sites: _step makes single-site moves")


if __name__ == "__main__":
    print("A. detailed balance and stationarity (exact enumeration)")
    for L, n, T in [(4, 1, 1.0), (4, 2, 1.0), (4, 2, 0.5), (5, 2, 1.0), (5, 3, 2.0)]:
        check_detailed_balance(L, n, T)
    print("B. sampled targets follow the pointing probabilities")
    check_draw_targets()
    print("C. canonical starts")
    check_mutate_canonical()
    print("D. reference guard")
    check_reference_guard()
    print("E. output, restart")
    check_sample_and_restart()
    print("F. _step against the exact kernel rows")
    check_step_matches_kernel()
    print("G. pars without n_sites")
    check_step_without_n_sites()
    print("ALL N-SITE CHECKS PASSED")
