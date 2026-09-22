"""
Side-by-side comparison of the two gradient-computation methods discussed
during the M=1 pivot, to isolate whether the site-4 outlier seen in a real
tests/test_steepest_descent.py run (gradient-predicted dU=+356.5, far
larger in magnitude than anything seen before -- worst previously ~-93) is
a genuine property of that site, or an artifact specific to relaxing only
one site.

  - _grad_pass_site (current production code, classes/rate_sampler.py):
    only the tested site is relaxed through softmax(logits/T_sftm); every
    OTHER site is held at its exact discrete (hard one-hot) value.
  - whole_sequence_grad (below; a local reimplementation of the gradient
    computation the earlier joint-design sampler used -- removed from
    production code when the design changed to single-site, see
    DEVLOG.txt): every site, including the one being tested, is relaxed
    through softmax(logits/T_sftm) SIMULTANEOUSLY; the tested site's
    gradient is one row of the resulting (L,K) tensor. Only needs to be
    computed once per sequence (one backward pass covers every site), then
    indexed per site.

If the two methods roughly agree in direction (high cosine similarity,
same argmax pick excluding cur+non-canonical) at site 4 but differ in
confidence, that's method (a) from the discussion: a real, if extreme,
instance of the known linearization-quality limit, and the site-only
gradient is just less "diluted" by the other 57 sites also being softened.
If they disagree sharply (low/negative cosine similarity, or a different
argmax pick), that points to something specific to the site-only
relaxation worth investigating further.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/compare_grad_methods.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.energies import compute_U_am, compute_entropy
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
N_SITES_TESTED = 10
SEED = 0
COSINE_FLAG_THRESHOLD = 0.5  # below this, flag as a sharp disagreement


def whole_sequence_grad(sampler, eprot, pars):
	"""Reimplementation of the earlier (now-removed) whole-sequence-relaxed
	_grad_pass: every site is softmax(logits/T_sftm)-relaxed simultaneously.
	Returns the full (L,K) gradient; the caller indexes whichever site(s)
	it needs. NOT used by the current sampler -- diagnostic only."""
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


def cosine_sim(a, b):
	return (a*b).sum().item() / (a.norm().item()*b.norm().item() + 1e-12)


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

	from utils.operations import mutate
	seq_A = mutate(REF_SEQ, 5, sampler.generator.get())
	eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
	eprot_A.expand()

	print(f"Reference: {REF_SEQ}")
	print(f"Current A: {seq_A}\n")

	# one backward pass covers every site for the whole-sequence method
	grad_whole_full = whole_sequence_grad(sampler, eprot_A, pars)  # (L,K)

	vocab = C.SEQUENCE_USED_VOCAB
	step_coef = pars['dt']**2 / (2.*pars['M'])
	L = len(seq_A)

	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()
	# make sure the flagged outlier site is included even if a re-run of
	# randperm would not have picked it (kept here for reproducibility of
	# the specific finding being investigated; harmless if already present)
	if 4 not in test_sites:
		test_sites = [4] + test_sites

	flagged = []

	for s in test_sites:
		cur = int(eprot_A.logits[s].argmax(dim=-1).item())
		competitors = sampler._competition_indices(cur)  # global vocab indices: canonical minus cur

		grad_site = sampler._grad_pass_site(eprot_A, s, pars)          # (K,), current production method
		grad_whole = grad_whole_full[s]                                 # (K,), old (removed) method

		cos = cosine_sim(grad_site, grad_whole)

		mu_site = -step_coef*grad_site
		mu_whole = -step_coef*grad_whole
		j_site = int(competitors[mu_site[competitors].argmax(dim=-1)].item())
		j_whole = int(competitors[mu_whole[competitors].argmax(dim=-1)].item())
		agree = (j_site == j_whole)

		# true dU for whichever candidate(s) the two methods actually picked
		def true_dU(j):
			cand_seq = seq_A[:s] + vocab[j] + seq_A[s+1:]
			eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
			eprot_c.expand()
			U_c, _, _ = sampler._exact_energy(eprot_c, pars)
			eprot_cur = ExtendedProtein(sequence=seq_A, requires_grad=False, device=device)
			eprot_cur.expand()
			U_A, _, _ = sampler._exact_energy(eprot_cur, pars)
			return U_c - U_A

		dU_site = true_dU(j_site)
		dU_whole = true_dU(j_whole) if not agree else dU_site

		flag = (cos < COSINE_FLAG_THRESHOLD) or (not agree and abs(dU_site - dU_whole) > 50)
		if flag:
			flagged.append(s)

		print(f"site {s:3d} (cur={vocab[cur]}): cosine(grad_site, grad_whole)={cos:+.4f}  "
			  f"|grad_site|={grad_site.norm().item():.4f}  |grad_whole|={grad_whole.norm().item():.4f}")
		if agree:
			print(f"    both methods pick {vocab[j_site]}  (true dU={dU_site:+.4f})")
		else:
			print(f"    site-only picks {vocab[j_site]} (true dU={dU_site:+.4f})   "
				  f"whole-sequence picks {vocab[j_whole]} (true dU={dU_whole:+.4f})")
		if flag:
			print(f"    [FLAGGED: low agreement between methods]")

	print()
	if flagged:
		print(f"Flagged sites (cosine<{COSINE_FLAG_THRESHOLD} or disagreeing picks with |dU diff|>50): {flagged}")
		print("-> worth a closer look at those specific sites before trusting the site-only gradient there.")
	else:
		print(f"No sites flagged: the two gradient methods agree closely enough at every tested site")
		print("(including the flagged outlier, if present in this run) that the site-only relaxation")
		print("does not look like an artifact -- any large true dU is a genuine property of that site's")
		print("energy landscape, not a computation issue introduced by relaxing only one position.")


if __name__ == "__main__":
	main()
