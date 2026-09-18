"""
Tests whether an ACTUAL joint multi-site move (as
classes/rate_sampler.py:ExtendedProteinRateSampler proposes it -- many sites
changing simultaneously) behaves like the sum of its individual single-site
effects, or whether coupling/epistasis between the simultaneously-changed
sites makes the true joint effect meaningfully different.

Motivation: tests/test_steepest_descent.py compares the gradient's
single-site suggestion against a brute-force single-site scan, holding
every OTHER site fixed at the CURRENT sequence. That is a fair test of the
gradient itself (a genuine partial derivative -- other sites truly held
fixed when it's computed) against within-site nonlinearity, but it is NOT a
fair proxy for how the sampler actually behaves: the sampler proposes many
sites changing AT ONCE, and U_am runs through self-attention, so U need not
be separable across sites. The single-site "true best", found holding every
other site fixed at its ORIGINAL value, is not a stable target if the other
proposed changes shift it once they're also applied.

For each of several actual joint proposals (drawn exactly as _step draws
them, repeatedly from the same starting sequence A), this decomposes the
true energy change into:

  sum_linear      = sum over touched sites of grad_A[s] . (target_s - cur_s)
                    (first-order Taylor prediction, no coupling, no
                    within-site nonlinearity -- what the gradient alone
                    "thinks" will happen)
  sum_isolated    = sum over touched sites of [U(A with ONLY site s changed)
                    - U(A)], i.e. the TRUE single-site effect of each
                    change, applied alone against the ORIGINAL sequence
                    (captures within-site nonlinearity; still assumes zero
                    coupling between sites)
  dU_joint        = U(A with ALL analyzed sites changed AT ONCE) - U(A)
                    (the real number an accept/reject step would use)

  sum_isolated - sum_linear  -> within-site linearization error (the
                                 quantity test_steepest_descent.py already
                                 probes, aggregated here)
  dU_joint - sum_isolated    -> coupling/epistasis between the
                                 simultaneously-changed sites (the new
                                 question raised in the design conversation)

If a move touches more than MAX_SITES_ANALYZED sites, a random subset is
analyzed and dU_joint is recomputed for THAT SUBSET applied alone (not the
full move), so the three numbers stay apples-to-apples; the full move's
real dU is also reported for reference. When nothing is capped, dU_joint
should match the full move's dU almost exactly -- printed as a sanity
check on this script itself, not just on the sampler.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/test_joint_coupling.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SEED = 0
DT_VALUES = [0.5, 2.0, 10.0]
TRIALS_PER_DT = 3
MAX_SITES_ANALYZED = 15  # bounds brute-force cost if a move touches many sites
CONSISTENCY_TOL = 1.0e-2


def analyze_move(sampler, seq_A, grad_A, U_A_exact, proposed_seq, pars, device):
	vocab = C.SEQUENCE_USED_VOCAB
	touched = [s for s in range(len(seq_A)) if seq_A[s] != proposed_seq[s]]
	capped = len(touched) > MAX_SITES_ANALYZED

	if capped:
		g = torch.Generator().manual_seed(SEED)
		perm = torch.randperm(len(touched), generator=g)[:MAX_SITES_ANALYZED].tolist()
		touched = [touched[i] for i in perm]

	sum_linear, sum_isolated = 0., 0.
	partial_seq = list(seq_A)

	for s in touched:
		cur = vocab.index(seq_A[s])
		tgt = vocab.index(proposed_seq[s])

		onehot_diff = torch.zeros(len(vocab), device=device)
		onehot_diff[tgt] += 1.
		onehot_diff[cur] -= 1.
		sum_linear += (grad_A[s]*onehot_diff).sum().item()

		cand_seq = seq_A[:s] + proposed_seq[s] + seq_A[s+1:]
		eprot_c = ExtendedProtein(sequence=cand_seq, requires_grad=False, device=device)
		eprot_c.expand()
		U_c, _, _ = sampler._exact_energy(eprot_c, pars)
		sum_isolated += U_c - U_A_exact

		partial_seq[s] = proposed_seq[s]

	eprot_p = ExtendedProtein(sequence="".join(partial_seq), requires_grad=False, device=device)
	eprot_p.expand()
	U_p, _, _ = sampler._exact_energy(eprot_p, pars)
	dU_joint = U_p - U_A_exact

	return len(touched), capped, sum_linear, sum_isolated, dU_joint


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

	U_A_exact, _, _ = sampler._exact_energy(eprot_A.copy(), pars)
	# grad_A doesn't depend on dt (dt is only used downstream of it in _step),
	# so it's the same for every trial/dt below -- compute it once.
	grad_A = sampler._grad_pass(eprot_A, pars)

	header = f"{'dt':>6} {'trial':>6} {'#sites':>7} {'sum_linear':>12} {'sum_isolated':>13} {'dU_joint':>10} {'nonlin':>10} {'coupling':>10}"
	print(header)
	print("-"*len(header))

	for dt in DT_VALUES:
		trial_pars = dict(pars)
		trial_pars['dt'] = dt
		for trial in range(TRIALS_PER_DT):
			_, info = sampler._step(eprot_A, U_A_exact, trial_pars)

			if info['muts_move'] == 0:
				print(f"{dt:>6.2f} {trial:>6} {'--':>7}  (no move proposed at this dt)")
				continue

			n_sites, capped, sum_linear, sum_isolated, dU_joint = analyze_move(
				sampler, seq_A, grad_A, U_A_exact, info['proposed_sequence'], trial_pars, device
			)
			nonlin = sum_isolated - sum_linear
			coupling = dU_joint - sum_isolated

			tag = f"{n_sites}*" if capped else f"{n_sites}"
			print(f"{dt:>6.2f} {trial:>6} {tag:>7} {sum_linear:>12.4f} {sum_isolated:>13.4f} {dU_joint:>10.4f} {nonlin:>10.4f} {coupling:>10.4f}")

			if not capped:
				diff = abs(dU_joint - info['dU'])
				status = "OK" if diff < CONSISTENCY_TOL else "MISMATCH"
				print(f"         [consistency check: dU_joint vs full-move dU ({info['dU']:+.4f}), diff={diff:.5f} -- {status}]")
			else:
				print(f"         (move touched {info['muts_move']} sites total; analyzed a random {n_sites}-site subset; "
					  f"full move's real dU was {info['dU']:+.4f})")

	print()
	print("* = move touched more sites than MAX_SITES_ANALYZED; a random subset was analyzed, so its")
	print("    dU_joint is for THAT SUBSET applied alone, not the full move (see full dU printed alongside).")
	print("nonlin   = sum_isolated - sum_linear  (within-site linearization error)")
	print("coupling = dU_joint - sum_isolated    (0 => U separable across the touched sites; large |value|")
	print("           => the isolated single-site 'best' targets shift once applied together)")


if __name__ == "__main__":
	main()
