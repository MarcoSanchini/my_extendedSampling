"""
Standalone checks for utils/proposals.py:log_pointing_prob. Pure torch, no
ESM3/model needed. Run with:
    cd softmax/code && python tests/test_proposals.py

These checks were independently validated (numpy/scipy re-implementation,
cross-checked against brute-force Monte Carlo) before being ported here; see
the derivation in the conversation this sampler was designed in. Running
them here re-confirms the same formula against its actual torch/production
implementation.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch

from utils.proposals import log_pointing_prob


def mc_pointing_probs(mu, sigma, n_samples=2_000_000, seed=0):
	g = torch.Generator().manual_seed(seed)
	K = mu.shape[-1]
	X = mu + sigma*torch.randn(n_samples, K, generator=g)
	winners = X.argmax(dim=-1)
	return torch.bincount(winners, minlength=K).float() / n_samples


def check_normalization_and_mc(K, sigma, scale, n_nodes=40, seed=1, tol_sum=1e-4, verbose=True):
	torch.manual_seed(seed)
	mu = torch.randn(K) * scale

	log_a = torch.stack([
		log_pointing_prob(mu, sigma, torch.tensor(j), n_nodes) for j in range(K)
	])
	a = log_a.exp()
	total = a.sum().item()
	assert abs(total - 1.0) < tol_sum, f"a_ij does not sum to 1 (got {total}) for K={K}, sigma={sigma}, scale={scale}"

	a_mc = mc_pointing_probs(mu, sigma, n_samples=2_000_000, seed=seed+100)
	mc_se = torch.sqrt(a_mc*(1-a_mc)/2_000_000) + 1e-6
	n_bad = int((torch.abs(a - a_mc) > 6*mc_se).sum().item())
	assert n_bad == 0, f"Gauss-Hermite disagrees with Monte Carlo beyond 6 std errs for {n_bad}/{K} classes (K={K}, sigma={sigma}, scale={scale})"

	assert a.argmax().item() == mu.argmax().item(), "argmax(a) should match argmax(mu) (monotonicity)"

	if verbose:
		print(f"  K={K:3d} sigma={sigma:.3g} scale={scale:.3g}: sum(a)={total:.6f}, max|a_GH-a_MC|={(a-a_mc).abs().max().item():.5f}, OK")


def check_dt_regime():
	"""
	Confirms the sampler's operating-regime note: as dt -> 0, mu and sigma
	both shrink (mu ~ dt^2, sigma ~ dt) so the standardized gap -> 0 and the
	per-site choice becomes UNIFORM over all K classes (not concentrated on
	"stay") -- while as dt grows, the choice concentrates on the true
	gradient-favored class. This is the opposite of typical
	small-step-size intuition and is the main tuning knob for this sampler.
	"""
	grad = torch.tensor([-2.0, -0.3, 0.1, 0.05, 1.5])  # class 0 has most negative grad
	M, T, K = 1.0, 1.0, len(grad)

	dt_small = 1e-4
	step_coef, sigma = dt_small**2/(2*M), dt_small*math.sqrt(T/M)
	mu = -step_coef*grad
	a_small = torch.stack([log_pointing_prob(mu, sigma, torch.tensor(j), 40).exp() for j in range(K)])
	assert torch.allclose(a_small, torch.full((K,), 1./K), atol=1e-3), \
		f"expected ~uniform proposal at very small dt, got {a_small.tolist()}"

	dt_large = 5.0
	step_coef, sigma = dt_large**2/(2*M), dt_large*math.sqrt(T/M)
	mu = -step_coef*grad
	a_large = torch.stack([log_pointing_prob(mu, sigma, torch.tensor(j), 40).exp() for j in range(K)])
	assert a_large[0].item() > 0.99, \
		f"expected proposal to concentrate on the steepest-descent class at large dt, got {a_large.tolist()}"

	print(f"  dt={dt_small}: a={a_small.tolist()} (uniform, as expected)")
	print(f"  dt={dt_large}: a={a_large.tolist()} (concentrated on class 0, as expected)")


def check_deterministic_argmax_matches_pointing_argmax(n_trials=200, K=20, seed=2):
	"""
	The "p=0" sanity check, at the level of the proposal math (not the real
	ESM3 energy -- see tests/test_steepest_descent.py for that): the class
	favored with near-certainty as dt grows (argmax of a_ij) must equal the
	deterministic delta_x argmax (mu.argmax()), since mu IS the p=0 delta_x.
	This is guaranteed by construction, so failure here would indicate a
	bug in log_pointing_prob rather than a modeling issue.
	"""
	torch.manual_seed(seed)
	dt, M, T = 5.0, 1.0, 1.0
	step_coef, sigma = dt**2/(2*M), dt*math.sqrt(T/M)

	for _ in range(n_trials):
		grad = torch.randn(K)
		mu = -step_coef*grad
		j_star = mu.argmax().item()
		a_star = log_pointing_prob(mu, sigma, torch.tensor(j_star), 40).exp().item()
		assert a_star == max(
			log_pointing_prob(mu, sigma, torch.tensor(j), 40).exp().item() for j in range(K)
		), "argmax(a_ij) should coincide with argmax(mu) = argmax(-grad), i.e. the p=0 steepest-descent direction"
	print(f"  {n_trials}/{n_trials} trials: argmax(a_ij) == argmax(-grad) (steepest-descent direction), as expected")


if __name__ == "__main__":
	print("=== normalization + Monte Carlo cross-check ===")
	for K in [3, 5, 10, 20, 25]:
		for scale in [0.1, 1.0, 5.0]:
			check_normalization_and_mc(K=K, sigma=1.0, scale=scale)

	print("\n=== dt regime (uniform at small dt, informed at large dt) ===")
	check_dt_regime()

	print("\n=== p=0 / steepest-descent consistency (proposal math only) ===")
	check_deterministic_argmax_matches_pointing_argmax()

	print("\nAll checks passed.")
