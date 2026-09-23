"""
First real-model sanity check for the NEW structure-token cross-entropy
energy (compute_U_structure_ce, wired into classes/rate_sampler.py via
lambda_structure_ce -- see that file's class docstring and DEVLOG.txt).
Mirrors the validation pattern already applied to U_am (scan_U_am_landscape.py,
test_steepest_descent.py's check 1), so the two energies get a directly
comparable first look, per the explicit "support both, see how they behave"
decision.

Run with lambda_structure_ce=1.0, lambda_am=0.0 -- PURE structure-token
energy, isolated from any attention-map contribution, so everything
measured here is attributable to U_structure_ce alone.

IMPORTANT, unlike U_am: compute_U_structure_ce has NO proven floor. U_am(ref)=0
is a mathematical certainty (a non-negative squared distance from a
reference compared to itself). U_structure_ce is a cross-entropy between a
CONTINUOUS predicted distribution and a DISCRETE (argmax-derived) target --
even evaluated at the reference against its own argmax tokens, this is only
small if the model's structure-token predictions are confidently peaked
(softmax outputs are never exactly one-hot), and there is no guarantee some
OTHER sequence couldn't score lower. This script does not assume a floor;
it measures whether the empirical behavior looks reasonable (small
self-distance, single mutations cost something, not wildly negative) and
reports numbers plainly rather than asserting a specific shape.

What this checks:
  (1) Plumbing sanity: does predict_structure_logits run without dtype/shape
      errors, and is U_structure_ce(ref, ref) small and finite?
  (2) Gradient sanity: same p=0 descent-direction check as
      test_steepest_descent.py's check 1 (mu_A . grad_A <= 0, true by
      construction regardless of which energy is behind grad_A) -- a
      failure here would mean a real bug in the NEW pathway, not a
      modeling limitation.
  (3) Reference-anchored scan: at a few sites, single-mutant U_structure_ce
      relative to the reference itself (same style as
      scan_U_am_landscape.py's check 2), to see whether this energy shows a
      similarly-shaped "cost of one mutation" landscape or something
      qualitatively different.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/scan_U_structure_ce_landscape.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
N_SITES_TESTED = 5
SEED = 0


def main():
	pars = {
		"T": 2.0, "dt": 1300.0, "M": 1.0, "T_sftm": 0.1,
		"lambda_am": 0.0, "lambda_structure_ce": 1.0, "lambda_S": 0.0,
		"eps": 1.0e-9, "n_quad": 40,
	}
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)

	from generator.custom_generator import CustomGenerator
	sampler.generator = CustomGenerator(seed=SEED, device=device)

	sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())

	with torch.no_grad():
		ref_structure_logits = sampler.model.predict_structure_logits(sequence_probs=sampler.ref_eprot.get_probs())
		sampler.ref_structure_tokens = ref_structure_logits.argmax(dim=-1)[1:-1]

	vocab = C.SEQUENCE_USED_VOCAB
	L = len(REF_SEQ)

	print(f"Reference: {REF_SEQ}")
	print(f"ref_structure_tokens shape={tuple(sampler.ref_structure_tokens.shape)}  "
		  f"dtype={sampler.ref_structure_tokens.dtype}  "
		  f"min={sampler.ref_structure_tokens.min().item()}  max={sampler.ref_structure_tokens.max().item()}\n")

	# ================================================================
	# (1) plumbing + self-distance sanity check
	# ================================================================
	print("=== (1) plumbing sanity: U_structure_ce(ref, ref) ===")
	eprot_ref = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
	eprot_ref.expand()
	U_ref_exact, _, eprot_ref = sampler._exact_energy(eprot_ref, pars)
	U_ce_ref = eprot_ref.U_structure_ce.item()  # eprot.U_structure_ce is a CPU tensor (see rate_sampler.py's _exact_energy)
	print(f"U_structure_ce(ref, ref) = {U_ce_ref:.6f}  "
		  f"(no proven floor -- small-but-nonzero is expected and fine, see module docstring)")
	print(f"finite: {torch.isfinite(eprot_ref.U_structure_ce).item()}\n")

	# ================================================================
	# (2) p=0 descent-direction check (same construction as
	# test_steepest_descent.py's check 1), now exercising the NEW pathway
	# ================================================================
	print("=== (2) gradient sanity: p=0 descent direction ===")
	from utils.operations import mutate
	seq_A = mutate(REF_SEQ, 5, sampler.generator.get())
	eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
	eprot_A.expand()
	print(f"Current A: {seq_A}")

	step_coef = pars['dt']**2 / (2.*pars['M'])
	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

	max_descent_dot = -float('inf')
	for s in test_sites:
		grad_A = sampler._grad_pass_site(eprot_A, s, pars)
		mu_A = -step_coef*grad_A
		dot = (mu_A * grad_A).sum().item()
		max_descent_dot = max(max_descent_dot, dot)
		print(f"  site {s:3d}: mu_A.grad_A = {dot:+.6e}  |grad_A|={grad_A.norm().item():.4f}")

	assert max_descent_dot <= 1e-8, f"mu_A should be a descent direction at every tested site; max dot={max_descent_dot}"
	print(f"[OK] mu_A . grad_A <= 0 at every tested site\n")

	# ================================================================
	# (3) reference-anchored single-mutant scan
	# ================================================================
	print("=== (3) reference-anchored scan (Hd=1 from ref, no prior drift) ===")
	all_dU_from_ref = []
	for s in test_sites:
		cur_ref = REF_SEQ[s]
		cur_idx = vocab.index(cur_ref)
		competitors = sampler._competition_indices(cur_idx)

		dU_from_ref = {}
		for j in competitors.tolist():
			cand_seq = REF_SEQ[:s] + vocab[j] + REF_SEQ[s+1:]
			eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
			eprot_c.expand()
			_, _, eprot_c = sampler._exact_energy(eprot_c, pars)
			dU_from_ref[j] = eprot_c.U_structure_ce.item() - U_ce_ref

		best_j = min(dU_from_ref, key=dU_from_ref.get)
		worst_j = max(dU_from_ref, key=dU_from_ref.get)
		mean_dU = sum(dU_from_ref.values()) / len(dU_from_ref)
		n_negative = sum(1 for v in dU_from_ref.values() if v < 0)

		all_dU_from_ref.extend(dU_from_ref.values())

		print(f"  site {s:3d} (ref={cur_ref}): best={vocab[best_j]} (dU={dU_from_ref[best_j]:+.4f})  "
			  f"worst={vocab[worst_j]} (dU={dU_from_ref[worst_j]:+.4f})  mean={mean_dU:+.4f}  "
			  f"n_negative={n_negative}/{len(dU_from_ref)}")

	n_neg_total = sum(1 for v in all_dU_from_ref if v < 0)
	print(f"\nAcross all {len(all_dU_from_ref)} single mutants of the reference tested: "
		  f"{n_neg_total} negative (unlike U_am, this is NOT guaranteed to be 0 -- report only, no assertion).")
	print(f"Mean dU-from-ref: {sum(all_dU_from_ref)/len(all_dU_from_ref):+.4f}")
	print(f"(Compare by eye to scan_U_am_landscape.py's own reference-anchored scan at the same sites,")
	print(f" if run with the same test_sites, to see whether the two energies agree on which mutations")
	print(f" are cheap/costly, or diverge -- that comparison is the point of supporting both.)")


if __name__ == "__main__":
	main()
