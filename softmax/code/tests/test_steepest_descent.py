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

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
N_SITES_TESTED = 10
SEED = 0


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

	# check 1: steepest-descent direction, by construction
	dot = (mu_A * grad_A).sum(dim=-1)
	assert (dot <= 1e-8).all(), f"mu_A should be a descent direction everywhere; max dot={dot.max().item()}"
	print("[OK] mu_A . grad_A <= 0 at every site (deterministic displacement is a descent direction)")

	vocab = C.SEQUENCE_USED_VOCAB
	L = len(seq_A)
	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

	top1_hits = 0
	dU_star_list, dU_other_list = [], []

	for s in test_sites:
		cur = vocab.index(seq_A[s])
		j_star = mu_A[s].argmax().item()

		true_dU = {}
		for j in range(len(vocab)):
			cand_seq = seq_A[:s] + vocab[j] + seq_A[s+1:]
			eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
			eprot_c.expand()
			U_c, _, _ = sampler._exact_energy(eprot_c, pars)
			true_dU[j] = U_c

		U_A_exact = true_dU[cur]
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

	mean_dU_star = sum(dU_star_list)/len(dU_star_list)
	mean_dU_other = sum(dU_other_list)/len(dU_other_list)

	print()
	print(f"Top-1 accuracy (gradient pick == true best): {top1_hits}/{N_SITES_TESTED}")
	print(f"Mean true dU at gradient-predicted class:     {mean_dU_star:+.4f}")
	print(f"Mean true dU at all other candidate classes:  {mean_dU_other:+.4f}")
	print(f"(gradient-predicted class should be markedly lower than the random/other average;")
	print(f" it need not hit the true optimum every time -- that's the linearization-quality tradeoff.)")


if __name__ == "__main__":
	main()
