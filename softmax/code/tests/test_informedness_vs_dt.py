"""
Replaces the mutated-vs-reference-matching accept-rate gap (monte_carlo/
sweep_dt.sh, sweep_dt_seeds.sh) as the "is the dt-based proposal actually
informed" diagnostic. See DEVLOG.txt, 2026-09-22 "gap reversal" entry: that
diagnostic's premise -- that the reference sequence sits at a local
U-minimum at each site, so mutated sites should be more improvable -- does
NOT hold for this attention-map energy once the sampler is genuinely
informed (dt=2000-8000 gave a growing NEGATIVE, not positive, gap). The
sampler mechanism itself was already independently verified correct
(test_steepest_descent.py's p=0 descent-direction check,
compare_grad_methods.py's cross-method agreement) -- the accept-rate-gap
metric was just measuring the wrong thing.

Key realization this script is built around: mu_A = -step_coef*grad_A,
step_coef = dt^2/(2M) is a POSITIVE SCALAR, so scaling grad_A by it never
changes which competitor is the argmax. The deterministic, p=0 proposal
direction -- and therefore whether it matches the TRUE energetically-best
single-site substitution -- is completely INDEPENDENT of dt. dt does not
make the gradient's pick any better or worse; it only controls how
reliably the REALIZED stochastic proposal (delta_x = dt*momentum/M -
step_coef*grad_A, i.e. mu_A plus momentum noise) follows that fixed pick.
This script separates the two questions instead of conflating them:

  (A) dt-INDEPENDENT: does the gradient's fixed argmax pick (over
      canonical competitors, cur excluded) match the true brute-force-best
      substitution? Computed once per site -- same brute-force scan
      test_steepest_descent.py already does (its dt=2.0 was an arbitrary
      placeholder that changes nothing about this answer, per the above).

  (B) dt-DEPENDENT, closed form, no brute force needed: given the
      gradient's fixed pick j*, what is the analytical probability that
      one realized proposal actually lands on j*, as a function of dt?
      This is exactly log_pointing_prob(mu_A[competitors], sigma(dt),
      target=j*) -- the same quadrature utils/proposals.py:_step() uses
      for log_a_AB, evaluated at mu_A's own argmax instead of whatever a
      single Monte Carlo draw happened to realize. Once grad_A is known
      (one real forward+backward pass per site, same cost as the p=0
      check), this sweeps arbitrarily many dt values for free -- no
      further model calls, no new main_rate.py runs, no waiting on the
      GPU sweep scheduler.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/test_informedness_vs_dt.py
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

# T, M held fixed at the values the real sweeps used (see DEVLOG.txt); dt is
# the axis being probed, so it's the only thing varied across the table.
T = 2.0
M = 1.0
N_QUAD = 40

# Matches the dt values actually run in sweep_dt.sh / sweep_dt_seeds.sh,
# plus a couple points further out to see where p_match saturates.
DTS_TO_SWEEP = [300, 600, 1000, 1300, 2000, 4000, 8000, 16000, 32000]


def main():
	pars = {
		"T": T, "dt": 1.0, "M": M, "T_sftm": 0.1,  # pars['dt'] unused below (see module docstring)
		"lambda_am": 1.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": N_QUAD,
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
	print(f"Current A: {seq_A}")
	print(f"T={T}, M={M}  (fixed; dt is the swept axis)\n")

	vocab = C.SEQUENCE_USED_VOCAB
	L = len(seq_A)

	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

	top1_hits = 0
	dU_star_list, dU_other_list = [], []
	# per-dt p_match, pooled across sites for the summary table
	p_match_by_dt = {dt: [] for dt in DTS_TO_SWEEP}

	for s in test_sites:
		# --- one real forward+backward pass per site; everything below is free ---
		grad_A = sampler._grad_pass_site(eprot_A, s, pars)
		cur = int(eprot_A.logits[s].argmax(dim=-1).item())
		competitors = sampler._competition_indices(cur)  # global vocab indices: canonical minus cur

		# (A) dt-independent: argmax of -grad_A over competitors. step_coef
		# is a positive scalar, so this is the SAME class for every dt.
		local_j_star = (-grad_A[competitors]).argmax(dim=-1)
		j_star = int(competitors[local_j_star].item())

		# --- brute-force ground truth (once per site, dt has no bearing on this) ---
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

		# (B) dt-dependent, closed form: P(realized proposal == j_star) at each dt
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
		print("      p_match(dt): " + "  ".join(f"dt={dt}:{p_match_row[dt]:.3f}" for dt in DTS_TO_SWEEP))

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
		print(f"  dt={dt:6d}: mean p_match={mean_p:.4f}  "
			  f"(min={min(vals):.4f}  max={max(vals):.4f} across the {N_SITES_TESTED} tested sites)")
	print("(As dt -> infinity this should -> 1.0 at every site; the crossover dt where it")
	print(" leaves the momentum-dominated ~uniform regime is exactly the number the old")
	print(" accept-rate-gap sweep was trying to locate indirectly and noisily.)")


if __name__ == "__main__":
	main()
