"""
Direct test of whether classes/rate_sampler.py:ExtendedProteinRateSampler's
core per-move mechanism actually converges to the correct Boltzmann target
exp(-U/T)/Z -- something nothing so far in this project has tested. Every
prior check (test_steepest_descent.py, compare_grad_methods.py,
test_informedness_vs_dt.py, scan_U_am_landscape.py) validated pieces of the
MACHINERY (gradient correctness, dt-informedness, the energy floor,
epistasis) but never the actual statistical claim: that running this
sampler for many moves produces samples distributed according to the target
density.

Why this needs testing, not just trusting: utils/proposals.py:
log_pointing_prob's own docstring says the K proposal classes are treated
as INDEPENDENT Gaussians, when the real displacement (mean-centered for
gauge-fixing) has a small but real induced correlation (~-1/(K-1) pairwise,
~-5% for K=20). Metropolis-Hastings is only exactly correct when the
acceptance ratio uses the EXACT probability of the mechanism that actually
generated the proposal -- here it uses a documented approximation instead.
Per-move this is a small effect; whether it meaningfully biases the
STATIONARY distribution after many moves is untested. Separately, every
sweep run so far shows U still drifting after 5000 moves with no clear
plateau, so there's no existing evidence of equilibration at all, at any
dt.

The honest scope of this test: exactly enumerating the Boltzmann
distribution over the FULL 56-site, ~20-letter sequence space is
impossible (~20^56 states). Restricting the sampler to propose moves at
only FREE_SITES (2 sites here, 20^2=400 states) while every other site
stays fixed at the reference identity makes the TARGET distribution exactly
enumerable, AND (since uniform site selection over a fixed set is still
symmetric/state-independent, exactly as classes/rate_sampler.py's own
class docstring argues for the full 56-site case) makes the resulting
Markov chain's correct stationary distribution EXACTLY the conditional
Boltzmann distribution over those 2 sites given that fixed background. This
tests the real acceptance-ratio code path (log_pointing_prob, the
independence approximation and all) against a known-exact target -- the
most decisive test available without a fundamentally intractable
enumeration. It does NOT by itself prove the full 56-site joint sampler is
unbiased (that target is one nobody can enumerate to check against), but
the acceptance-ratio math doesn't change based on how many other sites
exist, so a clean pass here is real evidence for the core mechanism, and a
failure here would be decisive evidence against it.

restricted_step() below is a near-verbatim copy of
classes/rate_sampler.py:_step() (as of the commit this was written against)
with exactly ONE change: the site is drawn uniformly from FREE_SITES
instead of from range(L). Everything else -- grad_A, momentum, gauge-
fixing, competitors, log_a_AB, the substitution, grad_B, log_a_BA, the
exact energy, the acceptance test -- calls the SAME sampler methods the
real _step() calls, unmodified. If rate_sampler.py's _step() changes later,
re-diff this against it.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/validate_boltzmann_toy.py

Expect real GPU time: 400 enumeration passes + MOVES*3 model calls for the
MCMC run (2 grad passes + 1 exact pass per move, same per-move cost as a
real sweep run). MOVES=20000 below is ~4x a standard 5000-move sweep run;
lower it for a faster, coarser first look (this trades statistical
resolution for speed -- it will still catch a gross discrepancy, e.g. the
sampler concentrating on the wrong states entirely, just not a subtle bias).
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import math
import itertools
import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.proposals import log_pointing_prob
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SEED = 0

# Two of the sites already characterized in test_informedness_vs_dt.py /
# scan_U_am_landscape.py (both were gradient-top1 HITs there) -- arbitrary
# but fixed for continuity with earlier results. 20 canonical letters each
# -> 20^2=400 states, small enough to enumerate exactly.
FREE_SITES = [30, 52]

# Moderate dt: low enough to explore broadly across all 20 classes per site
# (very high dt concentrates proposals on the gradient's fixed favorite,
# per test_informedness_vs_dt.py's p_match curve, which is good for speed
# but risks under-exploring the other ~19 classes within a limited move
# budget -- the opposite of what THIS test wants). dt=1300 gives mean
# p_match~0.15 (mild informedness, not near-uniform, not near-greedy).
pars = {
	"T": 2.0, "dt": 1300.0, "M": 1.0, "T_sftm": 0.1,
	"lambda_am": 1.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": 40,
}

MOVES = 20000
BURN_IN = 4000  # moves discarded before collecting the empirical histogram


def restricted_step(sampler, eprot, U_A, pars, free_sites):
	"""Near-verbatim copy of classes/rate_sampler.py:_step(); the ONLY
	change is the site draw (uniform over free_sites instead of range(L)).
	See module docstring for why this still targets the exact conditional
	Boltzmann distribution over free_sites given everything else fixed."""
	site = free_sites[int(torch.randint(0, len(free_sites), (1,), generator=sampler.generator.get(), device=sampler.generator.device).item())]

	grad_A = sampler._grad_pass_site(eprot, site, pars)
	cur_idx = eprot.logits[site].argmax(dim=-1)

	sigma = pars['dt'] * math.sqrt(pars['T']/pars['M'])
	step_coef = pars['dt']**2. / (2.*pars['M'])
	mu_A = -step_coef*grad_A

	momentum = sampler._extract_momenta(grad_A.shape, pars['T'], pars['M'])
	delta_x = pars['dt']*momentum/pars['M'] - step_coef*grad_A
	delta_x = delta_x - delta_x.mean(dim=-1, keepdim=True)

	competitors_A = sampler._competition_indices(int(cur_idx.item()))
	local_tgt = delta_x[competitors_A].argmax(dim=-1)
	tgt_idx = competitors_A[local_tgt]

	log_a_AB = log_pointing_prob(mu_A[competitors_A], sigma, local_tgt, pars['n_quad']).item()

	eprot_B = sampler._substitute(eprot, site, int(tgt_idx.item()))

	grad_B = sampler._grad_pass_site(eprot_B, site, pars)
	mu_B = -step_coef*grad_B

	competitors_B = sampler._competition_indices(int(tgt_idx.item()))
	local_cur_matches = (competitors_B == cur_idx).nonzero(as_tuple=True)[0]
	if local_cur_matches.numel() == 0:
		log_a_BA = float('-inf')
	else:
		local_cur = local_cur_matches[0]
		log_a_BA = log_pointing_prob(mu_B[competitors_B], sigma, local_cur, pars['n_quad']).item()

	U_B, U_am_B, eprot_B = sampler._exact_energy(eprot_B, pars)

	dU = U_B - U_A
	log_ratio = -dU/pars['T'] + (log_a_BA - log_a_AB)

	u = torch.rand(1, device=sampler.generator.device, generator=sampler.generator.get()).item()
	accepted = math.log(max(u, 1e-300)) <= min(0., log_ratio)

	return (eprot_B, U_B, accepted) if accepted else (eprot, U_A, accepted)


def main():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	# Needed here, unlike in the other test scripts: this is the first one
	# that actually runs _step()'s reverse-direction logic
	# (competitors_B == cur_idx), which compares self._canonical_idx
	# against a raw CUDA tensor (cur_idx isn't .item()'d there). The other
	# scripts only ever pass _competition_indices() a plain Python int, so
	# a CPU _canonical_idx never mismatched a CUDA tensor. Production runs
	# never hit this either, since _setup() (which this bypasses, same as
	# every other test script here) does this move already.
	sampler._canonical_idx = sampler._canonical_idx.to(device)

	from generator.custom_generator import CustomGenerator
	sampler.generator = CustomGenerator(seed=SEED, device=device)

	sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())

	vocab = C.SEQUENCE_USED_VOCAB
	canonical = [vocab[i] for i in sampler._canonical_idx.tolist()]
	s1, s2 = FREE_SITES

	print(f"Reference: {REF_SEQ}")
	print(f"Free sites: {s1} (ref={REF_SEQ[s1]}), {s2} (ref={REF_SEQ[s2]})  "
		  f"-- {len(canonical)}^2={len(canonical)**2} states, background fixed at reference elsewhere")
	print(f"T={pars['T']}, dt={pars['dt']}, MOVES={MOVES}, BURN_IN={BURN_IN}\n")

	# ================================================================
	# exact enumeration
	# ================================================================
	print("=== exact enumeration ===")
	U_exact = {}
	for a1, a2 in itertools.product(canonical, repeat=2):
		seq = list(REF_SEQ)
		seq[s1] = a1
		seq[s2] = a2
		seq = "".join(seq)
		eprot_c = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		eprot_c.expand()
		U_c, _, _ = sampler._exact_energy(eprot_c, pars)
		U_exact[(a1, a2)] = U_c

	U_ref_ref = U_exact[(REF_SEQ[s1], REF_SEQ[s2])]
	print(f"U at (ref,ref) = {U_ref_ref:.6f}  (should be exactly/near-exactly 0)")

	# logsumexp over all 400 states for a numerically stable Z
	neg_U_over_T = torch.tensor([-u/pars['T'] for u in U_exact.values()])
	logZ = torch.logsumexp(neg_U_over_T, dim=0).item()
	p_exact = {k: math.exp(-u/pars['T'] - logZ) for k, u in U_exact.items()}
	print(f"sum(p_exact) = {sum(p_exact.values()):.6f}  (should be 1.0)")

	top_exact = sorted(p_exact.items(), key=lambda kv: -kv[1])[:10]
	print("\nTop 10 states by exact Boltzmann probability:")
	for (a1, a2), p in top_exact:
		print(f"  ({a1},{a2}): p_exact={p:.4f}  U={U_exact[(a1,a2)]:+.4f}")

	# ================================================================
	# restricted MCMC run
	# ================================================================
	print(f"\n=== running {MOVES} restricted moves (site in {FREE_SITES} only) ===")
	eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=True, device=device)
	eprot.expand()
	U_A, _, eprot = sampler._exact_energy(eprot, pars)

	visit_counts = {}
	n_accepted = 0
	for move in range(1, MOVES+1):
		eprot, U_A, accepted = restricted_step(sampler, eprot, U_A, pars, FREE_SITES)
		n_accepted += int(accepted)

		if move > BURN_IN:
			state = (eprot.sequence[s1], eprot.sequence[s2])
			visit_counts[state] = visit_counts.get(state, 0) + 1

		if move % max(1, MOVES//20) == 0:
			print(f"  move {move:6d}/{MOVES}  acc_rate so far={n_accepted/move:.3f}  "
				  f"distinct states visited post-burn-in={len(visit_counts)}")

	n_counted = sum(visit_counts.values())
	p_empirical = {k: v/n_counted for k, v in visit_counts.items()}

	print(f"\nOverall acceptance rate: {n_accepted/MOVES:.4f}")
	print(f"Moves counted toward the histogram (post burn-in): {n_counted}")
	print(f"Distinct states visited: {len(visit_counts)} / {len(canonical)**2}")

	# ================================================================
	# comparison
	# ================================================================
	print("\n=== exact vs. empirical ===")
	print("Top 10 exact states, with empirical count/frequency alongside:")
	for (a1, a2), p in top_exact:
		emp_n = visit_counts.get((a1, a2), 0)
		emp_p = emp_n / n_counted if n_counted else 0.
		print(f"  ({a1},{a2}): p_exact={p:.4f}  p_empirical={emp_p:.4f}  (n={emp_n})")

	top_empirical = sorted(visit_counts.items(), key=lambda kv: -kv[1])[:5]
	print("\nTop 5 empirical states (for comparison, in case they differ from the exact top 5):")
	for (a1, a2), n in top_empirical:
		print(f"  ({a1},{a2}): p_empirical={n/n_counted:.4f} (n={n})  p_exact={p_exact[(a1,a2)]:.4f}")

	# KL(empirical || exact): well-defined everywhere since p_exact>0 for
	# every state (the floor argument -- U is finite everywhere, so no
	# state has exactly zero Boltzmann weight). The reverse KL would blow
	# up wherever a low-probability state was never visited.
	kl = sum(p*math.log(p/p_exact[k]) for k, p in p_empirical.items() if p > 0)
	tv = 0.5 * sum(abs(p_exact[k] - p_empirical.get(k, 0.)) for k in p_exact)

	mean_U_exact = sum(p*U_exact[k] for k, p in p_exact.items())
	mean_U_empirical = sum(p*U_exact[k] for k, p in p_empirical.items())

	print(f"\nKL(empirical || exact):     {kl:.4f}  (0 = perfect match; compare to log(400)={math.log(400):.2f} as a rough scale)")
	print(f"Total variation distance:   {tv:.4f}  (0 = perfect match, 1 = no overlap at all)")
	print(f"Mean U under p_exact:       {mean_U_exact:+.4f}")
	print(f"Mean U under p_empirical:   {mean_U_empirical:+.4f}")
	print(f"\n(With only {MOVES} moves and {n_counted} post-burn-in samples over 400 states, expect real")
	print(f" Monte Carlo noise even for a perfectly correct sampler -- this is a first-pass, not a")
	print(f" high-precision bias measurement. A large, systematic gap (e.g. empirical concentrated on")
	print(f" states the exact distribution says are unlikely, or missing the true top few states")
	print(f" entirely) would be the decisive signal; run longer for a finer read.)")


if __name__ == "__main__":
	main()
