"""
Live-model sanity check for the single-site gradient-informed proposal used
by classes/rate_sampler.py:ExtendedProteinRateSampler._step().

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/test_steepest_descent.py

What it checks, per tested site s (current amino acid cur):
  1. p=0 check: with the stochastic (momentum) term removed, the
     deterministic displacement mu_A = -dt^2/(2M)*grad_A is a steepest-
     descent direction (mu_A . grad_A <= 0) -- true by construction, so a
     failure here means a bug (sign error, wrong gauge-fixing, wrong
     token<->index mapping), not a modeling limitation.
  2. The real question: is j* = argmax(mu_A, cur and non-canonical
     excluded) -- the class the gradient-informed proposal favors among the
     candidates it's actually allowed to pick from -- a good single-site
     substitution? Brute-force scans the TRUE (exact, recomputed) energy of
     every canonical candidate != cur and compares:
       - is j* the true arg-minimum among those candidates (top-1)?
       - how does the true delta_U at j* compare to the true delta_U
         averaged over the other candidates?
     "cur" is excluded from both the prediction and the comparison set,
     matching what _step() actually does now (staying is decided by site
     selection, not by this per-site competition -- see DEVLOG.txt).

Unlike the archived (joint-design) version of this test, the gradient here
is computed with classes/rate_sampler.py:_grad_pass_site -- only the tested
site is relaxed through softmax(./T_sftm); every other site is held at its
exact discrete value, matching the sampler's actual per-move computation.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
N_SITES_TESTED = 10
SEED = 0


def main():
	pars = {
		"T": 20.0, "dt": 2.0, "M": 1.0, "T_sftm": 0.1,
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

	vocab = C.SEQUENCE_USED_VOCAB
	step_coef = pars['dt']**2 / (2.*pars['M'])
	L = len(seq_A)

	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

	top1_hits = 0
	dU_star_list, dU_other_list = [], []
	max_descent_dot = -float('inf')

	for s in test_sites:
		grad_A = sampler._grad_pass_site(eprot_A, s, pars)
		cur = int(eprot_A.logits[s].argmax(dim=-1).item())
		mu_A = -step_coef*grad_A

		# check 1: steepest-descent direction, by construction (before exclusion masking)
		dot = (mu_A * grad_A).sum().item()
		max_descent_dot = max(max_descent_dot, dot)

		excl = torch.unique(torch.cat([sampler._noncanonical_idx, torch.tensor([cur])]))
		mu_A_masked = sampler._penalize_indices(mu_A, excl)
		j_star = int(mu_A_masked.argmax(dim=-1).item())

		canonical_idx = [i for i in range(len(vocab)) if i not in sampler._noncanonical_idx.tolist() and i != cur]

		true_dU = {}
		for j in canonical_idx:
			cand_seq = seq_A[:s] + vocab[j] + seq_A[s+1:]
			eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
			eprot_c.expand()
			U_c, _, _ = sampler._exact_energy(eprot_c, pars)
			true_dU[j] = U_c

		eprot_cur = ExtendedProtein(sequence=seq_A, requires_grad=False, device=device)
		eprot_cur.expand()
		U_A_exact, _, _ = sampler._exact_energy(eprot_cur, pars)

		true_best = min(true_dU, key=true_dU.get)
		dU_star = true_dU[j_star] - U_A_exact
		dU_others = [true_dU[j]-U_A_exact for j in true_dU if j != j_star]

		hit = int(j_star == true_best)
		top1_hits += hit
		dU_star_list.append(dU_star)
		dU_other_list.extend(dU_others)

		print(f"  site {s:3d} (cur={vocab[cur]}): gradient-predicted={vocab[j_star]} "
			  f"(true dU={dU_star:+.4f})  true best={vocab[true_best]} (true dU={true_dU[true_best]-U_A_exact:+.4f})  "
			  f"top1={'HIT' if hit else 'miss'}")

	assert max_descent_dot <= 1e-8, f"mu_A should be a descent direction at every tested site; max dot={max_descent_dot}"
	print(f"\n[OK] mu_A . grad_A <= 0 at every tested site (p=0 displacement is a descent direction)")

	mean_dU_star = sum(dU_star_list)/len(dU_star_list)
	mean_dU_other = sum(dU_other_list)/len(dU_other_list)

	print()
	print(f"Top-1 accuracy (gradient pick == true best, among candidates excluding 'stay'): {top1_hits}/{N_SITES_TESTED}")
	print(f"Mean true dU at gradient-predicted class:     {mean_dU_star:+.4f}")
	print(f"Mean true dU at all other candidate classes:  {mean_dU_other:+.4f}")
	print(f"(gradient-predicted class should be markedly lower than the random/other average;")
	print(f" it need not hit the true optimum every time -- that's the linearization-quality tradeoff.)")


if __name__ == "__main__":
	main()
