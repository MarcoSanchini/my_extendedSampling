"""
Live-model sanity check for the gradient-informed proposal used by
classes/rate_sampler.py:ExtendedProteinRateSampler.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/test_steepest_descent.py

What it checks, per tested site s (current amino acid i_s):
  1. (trivial by construction, still worth asserting) the deterministic
     p=0 displacement mu_A[s] = -dt^2/(2M)*grad_A[s] is a steepest-descent
     direction: mu_A[s] . grad_A[s] <= 0.
  2. (the real question) whether j*_s = argmax(mu_A[s]) -- the class the
     gradient-informed proposal favors -- is actually a good single-site
     substitution, by brute-force scanning the TRUE (exact, recomputed)
     energy of every one of the K candidates at site s and comparing:
       - is j*_s the true arg-minimum (top-1)?
       - how does the true delta_U at j*_s compare to the true delta_U
         averaged over the other K-1 candidates?
     This is the linearization-quality question flagged during design:
     the gradient is only exactly right to first order, and U_am runs
     through several nonlinear self-attention layers, so j*_s need not
     always be the true optimum -- but it should beat a random guess by
     a wide margin, or the proposal isn't earning its keep.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import math
import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.proposals import log_pointing_prob
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
N_SITES_TESTED = 10
SEED = 0

# dt sweep for the exploration-budget report below. Includes 2.0, the
# current default in monte_carlo/rate_inputs/pars.txt.
DT_SWEEP = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]


def report_exploration_budget(sampler, grad_A, sites_info, pars, dt_values=DT_SWEEP):
	"""
	Uses ONLY the gradient already computed by main() (no new ESM3 calls) to
	answer: at a given dt, how likely is the actual stochastic proposal
	a_{cur->j} to land on "stay" vs. on the true best target identified by
	main()'s brute-force scan, and how many independent moves would it take
	(in expectation) for this site's own target alone to land on the true
	best purely by chance?

	This bounds how fast the *proposal* mechanism could plausibly discover
	a known-missed improvement at a given site; it says nothing about
	whether the resulting JOINT move (every site proposes simultaneously)
	would actually get accepted -- that needs a real run.
	"""
	M = pars['M']
	print("\n=== exploration budget vs dt (T fixed at {:.1f}) ===".format(pars['T']))
	print("a[stay]      : probability the proposal re-picks the current amino acid")
	print("a[true_best] : probability the proposal picks the site's true-best amino acid")
	print("E[#moves]    : expected number of independent moves before this site's own")
	print("               target lands on true_best at least once (~1/a[true_best])\n")

	header = "".join(f"{dt:>9.2f}" for dt in dt_values)
	for s, info in sites_info.items():
		cur, true_best, true_dU_best = info['cur'], info['true_best'], info['true_dU_best']
		print(f"site {s:3d} (cur={C.SEQUENCE_USED_VOCAB[cur]}, true best={C.SEQUENCE_USED_VOCAB[true_best]}, true dU={true_dU_best:+.4f}):")
		print(f"   dt:        {header}")

		a_stay_row, a_best_row, exp_row = [], [], []
		for dt in dt_values:
			sigma = dt*math.sqrt(pars['T']/M)
			step_coef = dt**2 / (2.*M)
			mu = -step_coef*grad_A[s]
			mu = sampler._penalize_noncanonical(mu)

			a_stay = log_pointing_prob(mu, sigma, torch.tensor(cur, device=mu.device), pars['n_quad']).exp().item()
			a_best = log_pointing_prob(mu, sigma, torch.tensor(true_best, device=mu.device), pars['n_quad']).exp().item()

			a_stay_row.append(a_stay)
			a_best_row.append(a_best)
			exp_row.append(1./a_best if a_best > 1e-12 else float('inf'))

		print("   a[stay]:   " + "".join(f"{v:>9.4f}" for v in a_stay_row))
		print("   a[best]:   " + "".join(f"{v:>9.4f}" for v in a_best_row))
		print("   E[#moves]: " + "".join(f"{v:>9.1f}" if v != float('inf') else f"{'inf':>9}" for v in exp_row))
		print()


def main():
	pars = {
		"T": 20.0, "dt": 1.0, "M": 1.0, "T_sftm": 0.1,
		"lambda_am": 1.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": 40,
	}
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)

	from generator.custom_generator import CustomGenerator
	sampler.generator = CustomGenerator(seed=SEED, device=device)

	sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())

	# Start a few mutations away from the reference so gradients are non-trivial
	from utils.operations import mutate
	seq_A = mutate(REF_SEQ, 5, sampler.generator.get())
	eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
	eprot_A.expand()

	print(f"Reference: {REF_SEQ}")
	print(f"Current A: {seq_A}")

	grad_A = sampler._grad_pass(eprot_A, pars)
	step_coef = pars['dt']**2 / (2.*pars['M'])
	mu_A = -step_coef*grad_A

	# check 1: steepest-descent direction, by construction (before the
	# non-canonical penalty, which is a proposal-shaping choice, not physics)
	dot = (mu_A * grad_A).sum(dim=-1)
	assert (dot <= 1e-8).all(), f"mu_A should be a descent direction everywhere; max dot={dot.max().item()}"
	print("[OK] mu_A . grad_A <= 0 at every site (deterministic displacement is a descent direction)")

	# check 2 (below) compares against what the sampler can actually propose,
	# so apply the same non-canonical exclusion it uses.
	mu_A = sampler._penalize_noncanonical(mu_A)
	canonical_idx = [i for i in range(len(C.SEQUENCE_USED_VOCAB)) if i not in sampler._noncanonical_idx.tolist()]

	vocab = C.SEQUENCE_USED_VOCAB
	L = len(seq_A)
	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

	top1_hits = 0
	dU_star_list, dU_other_list = [], []
	sites_info = {}

	for s in test_sites:
		cur = vocab.index(seq_A[s])
		j_star = mu_A[s].argmax().item()

		true_dU = {}
		for j in canonical_idx:
			cand_seq = seq_A[:s] + vocab[j] + seq_A[s+1:]
			eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
			eprot_c.expand()
			U_c, _, _ = sampler._exact_energy(eprot_c, pars)
			true_dU[j] = U_c
		if cur not in true_dU:
			eprot_c = ExtendedProtein(sequence=seq_A, requires_grad=False, device=device)
			eprot_c.expand()
			true_dU[cur], _, _ = sampler._exact_energy(eprot_c, pars)

		U_A_exact = true_dU[cur]
		true_best = min(true_dU, key=true_dU.get)

		dU_star = true_dU[j_star] - U_A_exact
		dU_others = [true_dU[j]-U_A_exact for j in true_dU if j != j_star]

		hit = int(j_star == true_best)
		top1_hits += hit
		dU_star_list.append(dU_star)
		dU_other_list.extend(dU_others)
		sites_info[s] = {'cur': cur, 'true_best': true_best, 'true_dU_best': true_dU[true_best]-U_A_exact}

		print(f"  site {s:3d} (cur={vocab[cur]}): gradient-predicted={vocab[j_star]} "
			  f"(true dU={dU_star:+.4f})  true best={vocab[true_best]} (true dU={true_dU[true_best]-U_A_exact:+.4f})  "
			  f"top1={'HIT' if hit else 'miss'}")

	mean_dU_star = sum(dU_star_list)/len(dU_star_list)
	mean_dU_other = sum(dU_other_list)/len(dU_other_list)

	print()
	print(f"Top-1 accuracy (gradient pick == true best): {top1_hits}/{N_SITES_TESTED}")
	print(f"Mean true dU at gradient-predicted class:     {mean_dU_star:+.4f}")
	print(f"Mean true dU at all other candidate classes:  {mean_dU_other:+.4f}")
	print(f"(gradient-predicted class should be markedly lower than the random/other average;")
	print(f" it need not hit the true optimum every time -- that's the linearization-quality tradeoff.)")

	report_exploration_budget(sampler, grad_A, sites_info, pars)


if __name__ == "__main__":
	main()
