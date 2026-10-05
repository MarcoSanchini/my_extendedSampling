import math

import numpy as np
import torch


# ------------------------------------------------------------------ #
# Gauss-Hermite quadrature nodes/weights, cached per (n_nodes,device,dtype).
# Rescaled so that sum_q exp(log_w_q) * f(u_q) approximates E_{u~N(0,1)}[f(u)].
# ------------------------------------------------------------------ #
_GH_CACHE = {}

def _gh_nodes(n_nodes: int, device, dtype):
	key = (n_nodes, device, dtype)
	if key not in _GH_CACHE:
		t, w = np.polynomial.hermite.hermgauss(n_nodes)
		u = torch.tensor(t * math.sqrt(2.), device=device, dtype=dtype)
		log_w = torch.tensor(np.log(w) - 0.5*np.log(np.pi), device=device, dtype=dtype)
		_GH_CACHE[key] = (u, log_w)
	return _GH_CACHE[key]


# ------------------------------------------------------------------ #
# log a_{i->j}, the log-probability that the (independent-Gaussian
# approximation of the) per-site displacement dx points at class j,
# i.e. that dx_j = max_k dx_k, over ALL K classes (including j==i,
# i.e. "staying" is one of the competing outcomes, not excluded).
#
# dx_k ~ N(mu_k, sigma^2), independent across k (mean-centering-induced
# correlation across k is neglected; see note in the module docstring
# below / the accompanying derivation).
# ------------------------------------------------------------------ #
def log_pointing_prob(
		mu: torch.Tensor,
		sigma: float,
		target: torch.Tensor,
		n_nodes: int = 40,
) -> torch.Tensor:
	"""
	mu:     (..., L, K) per-site, per-class means of the displacement dx
	        (e.g. mu = -dt**2/(2*M) * grad_U, the deterministic/drift part only).
	sigma:  scalar std of dx, common to every site and class
	        (sigma = dt*sqrt(T/M); holds as long as dt, T, M are global scalars).
	target: (..., L) long tensor, the class index j whose pointing-probability
	        a_{i->j} is being evaluated at each site (i is implicit in mu:
	        mu[..., s, i_s] is just another entry of mu, not special-cased).
	n_nodes: number of Gauss-Hermite quadrature nodes.

	Returns:
	        (..., L) log a_{i->j} = log P(dx_j = max_k dx_k).
	        exp(.) sums to 1 over all K choices of `target` at a fixed site
	        (up to quadrature error), since it's the exact "which independent
	        Gaussian is the max" probability, not merely proportional to it.
	"""
	u_q, log_w_q = _gh_nodes(n_nodes, mu.device, mu.dtype)          # (Q,)

	mu_t = torch.gather(mu, -1, target.unsqueeze(-1))                # (..., L, 1)
	alpha = (mu_t - mu) / sigma                                      # (..., L, K); alpha[...,target]==0

	z = alpha.unsqueeze(-1) + u_q                                    # (..., L, K, Q)
	log_phi = torch.special.log_ndtr(z)                              # (..., L, K, Q)

	sum_all = log_phi.sum(dim=-2)                                    # (..., L, Q)      sum over K (incl. k==target)
	self_term = torch.special.log_ndtr(u_q)                          # (Q,)             the k==target term (alpha=0)
	sum_excl_target = sum_all - self_term                            # (..., L, Q)      sum over k != target

	log_a = torch.logsumexp(log_w_q + sum_excl_target, dim=-1)       # (..., L)
	return log_a


"""
Note on the independence assumption -- it is EXACT here, not an approximation.
This corrects what this file said until 2026-10-05.

The concern was: the gauge-fixing that keeps momenta and the displacement
mean-centered across the class axis (removing the softmax translation
invariance) induces a covariance Sigma = sigma^2*(I - J/K) rather than
sigma^2*I, so treating the K classes as independent N(mu_k, sigma^2) looked
like a deliberate approximation.

It is not, for the quantity this function computes. Both centerings --
_extract_momenta's (p -= p.mean()) and _step's (delta_x -= delta_x.mean()) --
subtract the SAME SCALAR from every component, and argmax is invariant under
a common shift. The realized proposal is therefore the argmax of the
UNCENTERED displacement, whose components are genuinely independent
N(mu_k, sigma^2). The induced correlation never reaches the functional being
evaluated. Slicing to the competitor subset afterwards does not disturb this,
since the same scalar was subtracted from those components too.

Verified by simulating the exact pipeline (centered momentum, centered
delta_x, argmax over the competitor slice, 2e6 draws): the formula with the
uncentered sigma matches empirical argmax frequencies to 3.3e-4 against a
Monte Carlo standard error of 2.6e-4, while the "corrected" marginal
sigma*sqrt(1-1/K) -- the natural thing to reach for once one notices the
centering -- is wrong by 2.5e-3, about ten standard errors. Drawing
uncentered momenta directly reproduces the centered pipeline's argmax
frequencies to 3.6e-4, confirming the shift-invariance argument rather than
just asserting it.

So the only error left in a_ij is quadrature error, which is separately
verified (n_nodes=40 converged to ~6 significant figures even in the steepest
case tested).

Note on the general formula: this function assumes a COMMON sigma across
classes, which holds here because dt, T and M are global scalars, so the
momentum is isotropic. The general "which independent Gaussian is the max"
formula carries a second coefficient,
    a_ij = int phi(u) prod_{k!=i,j} Phi(alpha_kj + beta_kj*u) du
    alpha_kj = (mu_j - mu_k)/sigma_k,    beta_kj = sigma_j/sigma_k
and reduces to what is implemented here only because beta_kj == 1. A variant
with a per-class or per-site step size would have to reinstate beta; using a
single sigma when the sigma_k genuinely differ was measured wrong by up to
0.135 in absolute probability on a K=6 test. See softmax/docs/informedness.tex.
"""
