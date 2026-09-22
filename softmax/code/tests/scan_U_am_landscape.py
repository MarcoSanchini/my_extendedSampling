"""
Investigates utils/energies.py:compute_U_am directly, independent of the
sampler's proposal mechanism -- follow-up to the accept-rate-gap sign
reversal (DEVLOG.txt, 2026-09-22 entries) and test_informedness_vs_dt.py's
result that the reversal shows up well before the sampler is fully
informed. That pointed at the ENERGY, not the sampler, as the source, and
this checks it two ways.

compute_U_am(am, ref_am) = sum_{i<j} (log(am_ij) - log(ref_am_ij))^2 (upper
triangle only) is a squared log-space distance: PROVABLY >= 0 everywhere,
with equality (essentially) only when am == ref_am exactly, i.e. at the
reference sequence itself. classes/rate_sampler.py:_exact_energy computes
U = lambda_am * compute_U_am(am, ref_am) with no entropy term (a hard,
non-relaxed sequence has ~0 entropy anyway, so this is equivalent, not a
bug) -- with the lambda_am=1.0 used everywhere so far, U IS this distance,
exactly. That makes U(reference sequence) = 0 the (essentially unique)
GLOBAL minimum over the entire sequence space: no sequence, single mutant
or otherwise, can have U below 0.

  (1) FLOOR CONSISTENCY CHECK: confirms compute_U_am(ref_am, ref_am) == 0,
      then recomputes the EXACT SAME seq_A used in test_steepest_descent.py
      / test_informedness_vs_dt.py (mutate(REF_SEQ, 5, ...), SEED=0) and
      prints its own U_A_exact directly. This settles, rather than
      assumes, whether test_informedness_vs_dt.py's reported dU values
      (down to -92.99 at site 55) are mathematically consistent with the
      U>=0 floor (they require U_A_exact >= 93) -- if U_A_exact turns out
      to be much smaller than that, it means something is inconsistent
      somewhere in the pipeline, a real bug to chase, not just a landscape
      interpretation question.

  (2) REFERENCE-ANCHORED SCAN: at the SAME 10 sites already tested, scans
      every canonical single-point mutant DIRECTLY FROM THE REFERENCE
      SEQUENCE itself (Hd=1 from ref, no prior drift at all -- unlike
      test_informedness_vs_dt.py's scan, which was from seq_A, Hd=4).
      Every one of these MUST have U > 0 by the floor argument above; the
      question is how much headroom they have, and specifically whether a
      site that looks "expensive" to mutate directly from the reference
      becomes cheap/improving once evaluated from seq_A instead -- the
      direct site-by-site test of the epistasis explanation for the
      accept-rate-gap reversal.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in. Run it from the softmax/code directory:
    python tests/scan_U_am_landscape.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.energies import compute_U_am
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
N_SITES_TESTED = 10
SEED = 0

# The same 10 sites test_steepest_descent.py / test_informedness_vs_dt.py
# tested, reproduced identically below via the same seeded randperm call.


def main():
	pars = {
		"T": 2.0, "M": 1.0, "T_sftm": 0.1,
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

	vocab = C.SEQUENCE_USED_VOCAB
	L = len(REF_SEQ)

	# ================================================================
	# (1) floor consistency check
	# ================================================================
	print("=== (1) floor consistency check ===")

	U_am_ref_self = compute_U_am(sampler.ref_eprot.am, sampler.ref_eprot.am).item()
	print(f"compute_U_am(ref_am, ref_am) = {U_am_ref_self:.8e}  (should be exactly/near-exactly 0)")

	from utils.operations import mutate
	seq_A = mutate(REF_SEQ, 5, sampler.generator.get())
	eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=False, device=device)
	eprot_A.expand()
	U_A_exact, U_am_A, _ = sampler._exact_energy(eprot_A, pars)
	hd_A = sum(1 for a, b in zip(seq_A, REF_SEQ) if a != b)

	print(f"Current A (Hd={hd_A} from ref): {seq_A}")
	print(f"U_A_exact = {U_A_exact:.4f}")
	print(f"(test_informedness_vs_dt.py's most negative reported dU at this seq_A")
	print(f" was -92.9918 at site 55; that requires U_A_exact >= 92.9918 to be possible.")
	print(f" {'CONSISTENT' if U_A_exact >= 92.9918 else 'INCONSISTENT -- see note below'}: "
		  f"U_A_exact={U_A_exact:.4f} {'>=' if U_A_exact >= 92.9918 else '<'} 92.9918)")
	if U_A_exact < 92.9918:
		print("  NOTE: if this prints INCONSISTENT, U_A_exact here and the U_A_exact used")
		print("  inside test_informedness_vs_dt.py's own true_dU computation are somehow")
		print("  different despite using the same seq_A -- worth comparing the two code")
		print("  paths line by line (_exact_energy is called identically in both).")

	# ================================================================
	# (2) reference-anchored single-mutant scan, same 10 sites as before
	# ================================================================
	print("\n=== (2) reference-anchored scan (Hd=1 from ref, no prior drift) ===")

	torch.manual_seed(SEED)
	test_sites = torch.randperm(L)[:N_SITES_TESTED].tolist()

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
			U_c, _, _ = sampler._exact_energy(eprot_c, pars)
			dU_from_ref[j] = U_c  # U_ref == 0, so U_c IS the dU

		best_j = min(dU_from_ref, key=dU_from_ref.get)
		worst_j = max(dU_from_ref, key=dU_from_ref.get)
		mean_dU = sum(dU_from_ref.values()) / len(dU_from_ref)
		n_negative = sum(1 for v in dU_from_ref.values() if v < 0)

		all_dU_from_ref.extend(dU_from_ref.values())

		print(f"  site {s:3d} (ref={cur_ref}): best={vocab[best_j]} (dU={dU_from_ref[best_j]:+.4f})  "
			  f"worst={vocab[worst_j]} (dU={dU_from_ref[worst_j]:+.4f})  mean={mean_dU:+.4f}  "
			  f"n_negative={n_negative}/{len(dU_from_ref)}"
			  f"{'  *** FLOOR VIOLATION ***' if dU_from_ref[best_j] < -1e-6 else ''}")

	n_neg_total = sum(1 for v in all_dU_from_ref if v < 0)
	print(f"\nAcross all {len(all_dU_from_ref)} single mutants of the reference tested: "
		  f"{n_neg_total} negative (should be 0, up to float precision -- the floor argument")
	print(f"guarantees every one of these is >= 0 since U(ref)=0 is the global minimum).")
	print(f"Mean dU-from-ref across all tested single mutants: {sum(all_dU_from_ref)/len(all_dU_from_ref):+.4f}")
	print(f"(This is the typical 'cost' of one mutation evaluated with NO other drift --")
	print(f" compare by eye to test_informedness_vs_dt.py's true_dU values at the SAME sites,")
	print(f" which were evaluated from seq_A (Hd=4). A site whose cost-from-ref is clearly")
	print(f" positive but whose cost-from-seq_A went negative is direct, site-matched evidence")
	print(f" of epistasis/context-dependence -- not a landscape-optimality violation, since")
	print(f" seq_A's own baseline U_A_exact already accounts for its own Hd=4 distance from ref.)")


if __name__ == "__main__":
	main()
