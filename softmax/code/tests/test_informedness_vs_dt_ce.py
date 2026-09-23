"""
U_structure_ce analog of test_informedness_vs_dt.py -- same two-question
decomposition (dt-independent gradient quality vs. dt-dependent realized-
proposal reliability), now isolating the structure-token cross-entropy
energy (lambda_am=0.0, lambda_structure_ce=1.0) instead of U_am. See that
file's module docstring for the full derivation of why this decomposition
is valid regardless of which energy is behind grad_A -- nothing about the
argument is U_am-specific, so it carries over unchanged.

Why this needs its OWN dt sweep, not just reusing U_am's dt=(300..32000)
table: scan_U_structure_ce_landscape.py's first real run measured
|grad_A| ~ 0.0000-0.0003 for U_structure_ce, roughly 100-1000x smaller
than U_am's (~0.01-0.08 in earlier work). Since informedness depends on
step_coef*grad_gap/sigma ~ (dt/2)*grad_gap*sqrt(1/(M*T)) -- linear in dt
for fixed grad_gap -- reaching the same standardized separation needs a
proportionally larger dt. DTS_TO_SWEEP below is deliberately wide (5
orders of magnitude) to bracket the real crossover empirically rather
than guess a single scale factor from one gradient-magnitude estimate.

No new numerical-safety concern from going this high: log_pointing_prob
(utils/proposals.py) is fully log-space (torch.special.log_ndtr +
torch.logsumexp, verified for U_am up to dt=128000 via a standalone numpy
reimplementation -- see DEVLOG.txt) and is completely agnostic to which
energy produced mu_A/grad_A, so that verification carries over. The one
thing checked freshly here: step_coef=dt^2/(2M) reaches ~5e15 at the top
of this sweep, multiplied by grad_A~1e-4 gives mu_A~1e11-1e12 -- nowhere
near float32 overflow (~3.4e38).

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/test_informedness_vs_dt_ce.py
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

T = 2.0
M = 1.0
N_QUAD = 40

# ~5 orders of magnitude, to bracket the real crossover rather than assume
# a single scale factor from one gradient-magnitude estimate (see module
# docstring). Adjust once real numbers come back if the crossover turns
# out to sit outside this range.
DTS_TO_SWEEP = [1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7, 1.0e8]


def main():
	pars = {
		"T": T, "dt": 1.0, "M": M, "T_sftm": 0.1,  # pars['dt'] unused below (see module docstring)
		"lambda_am": 0.0, "lambda_structure_ce": 1.0, "lambda_S": 0.0,
		"eps": 1.0e-9, "n_quad": N_QUAD,
	}
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)

	from generator.custom_generator import CustomGenerator
	sampler.generator = CustomGenerator(seed=SEED, device=device)

	sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	# Still needed even with lambda_am=0: _grad_pass_site/_exact_energy compute
	# U_am unconditionally (only its CONTRIBUTION to the total is lambda-gated),
	# so self.ref_eprot.am must exist or compute_U_am crashes on None.
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())

	with torch.no_grad():
		ref_structure_logits = sampler.model.predict_structure_logits(sequence_probs=sampler.ref_eprot.get_probs())
		sampler.ref_structure_tokens = ref_structure_logits.argmax(dim=-1)[1:-1]

	from utils.operations import mutate
	seq_A = mutate(REF_SEQ, 5, sampler.generator.get())
	eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=True, device=device)
	eprot_A.expand()

	print(f"Reference: {REF_SEQ}")
	print(f"Current A: {seq_A}")
	print(f"T={T}, M={M}  (fixed; dt is the swept axis)  [lambda_am=0, lambda_structure_ce=1]\n")

	vocab = C.SEQUENCE_USED_VOCAB
	L = len(seq_A)

	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

	top1_hits = 0
	dU_star_list, dU_other_list = [], []
	p_match_by_dt = {dt: [] for dt in DTS_TO_SWEEP}

	for s in test_sites:
		grad_A = sampler._grad_pass_site(eprot_A, s, pars)
		cur = int(eprot_A.logits[s].argmax(dim=-1).item())
		competitors = sampler._competition_indices(cur)

		local_j_star = (-grad_A[competitors]).argmax(dim=-1)
		j_star = int(competitors[local_j_star].item())

		true_dU = {}
		for j in competitors.tolist():
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

		p_match_row = {}
		for dt in DTS_TO_SWEEP:
			step_coef = dt**2 / (2.*M)
			sigma = dt * math.sqrt(T/M)
			mu_A = -step_coef*grad_A
			log_p = log_pointing_prob(mu_A[competitors], sigma, local_j_star, N_QUAD).item()
			p_match = math.exp(log_p)
			p_match_row[dt] = p_match
			p_match_by_dt[dt].append(p_match)

		print(f"  site {s:3d} (cur={vocab[cur]}): gradient-predicted={vocab[j_star]} "
			  f"(true dU={dU_star:+.4f})  true best={vocab[true_best]} (true dU={true_dU[true_best]-U_A_exact:+.4f})  "
			  f"top1={'HIT' if hit else 'miss'}")
		print("      p_match(dt): " + "  ".join(f"dt={dt:.0e}:{p_match_row[dt]:.3f}" for dt in DTS_TO_SWEEP))

	mean_dU_star = sum(dU_star_list)/len(dU_star_list)
	mean_dU_other = sum(dU_other_list)/len(dU_other_list)

	print()
	print("=== (A) dt-independent: is the gradient's fixed pick any good? ===")
	print(f"Top-1 accuracy (gradient pick == true best, among candidates excluding 'stay'): {top1_hits}/{N_SITES_TESTED}")
	print(f"Mean true dU at gradient-predicted class:     {mean_dU_star:+.4f}")
	print(f"Mean true dU at all other candidate classes:  {mean_dU_other:+.4f}")
	print("(This number does NOT change with dt -- dt only affects how reliably a single")
	print(" realized move follows this fixed pick, see (B) below.)")

	print()
	print("=== (B) dt-dependent, closed form: mean P(realized proposal == gradient's pick), pooled over sites ===")
	for dt in DTS_TO_SWEEP:
		vals = p_match_by_dt[dt]
		mean_p = sum(vals)/len(vals)
		print(f"  dt={dt:.1e}: mean p_match={mean_p:.4f}  "
			  f"(min={min(vals):.4f}  max={max(vals):.4f} across the {N_SITES_TESTED} tested sites)")
	print("(As dt -> infinity this should -> 1.0 at every site. If the whole swept range still")
	print(" reads near 0 or near 1 with no visible transition, DTS_TO_SWEEP needs to shift further")
	print(" down or up respectively -- rerun with an adjusted range centered on wherever the")
	print(" transition actually showed signs of starting.)")


if __name__ == "__main__":
	main()
