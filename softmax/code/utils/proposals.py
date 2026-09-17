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
Note on the independence approximation:
mu/sigma above come from a displacement dx whose K components are, strictly,
NOT independent: the gauge-fixing that keeps logits (and momenta) mean-centered
across the class axis (removing the softmax translation invariance) induces a
covariance Sigma = sigma^2*(I - J/K) rather than sigma^2*I. The formula here
treats the K classes as independent N(mu_k, sigma^2) instead, which is a
deliberate, small, controlled approximation (pairwise correlation -1/(K-1),
e.g. ~-0.05 for K=20) made for tractability -- computing a_ij under the exact
projected covariance would require a (K-1)-dimensional orthant integral with
no closed form.
"""
