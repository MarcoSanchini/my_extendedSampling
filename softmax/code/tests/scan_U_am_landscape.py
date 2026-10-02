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


from references import get_reference

# Reference sequence now comes from tests/references.py so the whole
# diagnostic suite can be re-pointed in one place (and run on protein_g as
# a control) -- see that module's docstring. Override per run with e.g.
#     REF=protein_g python tests/<this script>
REF_NAME, REF_SEQ = get_reference()
# Both overridable per run, so a verdict at a new sequence length can be
# tightened (or matched to another reference's drift fraction) without
# editing this file -- defaults reproduce every recorded protein G result:
#   SITES  how many sites to test. 10 of protein G's 56 is 18% coverage but
#          only 1.8% of zero_polymer's 566, and the top-1 count out of 10 has
#          a wide error bar; raise it for a firmer answer (cost is linear).
#   MUTS   initial mutations in seq_A. 5 is 9% of protein G but 0.9% of
#          zero_polymer, which leaves seq_A almost exactly at U_am's floor
#          where nearly every move is uphill. ~51 matches protein G's
#          fraction on zero_polymer.
# e.g.  SITES=30 MUTS=51 python tests/<this script>
N_SITES_TESTED = int(os.environ.get("SITES", 10))
N_INIT_MUTS = int(os.environ.get("MUTS", 5))
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

	print(f"# reference: {REF_NAME} (L={L} residues)")

	# ================================================================
	# (0) numerical health of log(am) -- length-sensitive, added 2026-10-02
	# ================================================================
	# compute_U_am takes a RAW torch.log(am), with no eps guard (unlike
	# compute_entropy's log(p+eps)). am is a softmax over L+2 positions, so
	# its entries shrink as the sequence gets longer (mean ~1/(L+2): ~0.017
	# at L=56, ~0.0018 at L=566) and its small tail shrinks faster. Two
	# distinct failure modes, neither of which any existing check would
	# notice, and both of which get worse with L:
	#   - an entry reaching exactly 0 makes log(am) = -inf and U_am inf/nan;
	#   - log amplifies RELATIVE error (d log x = dx/x), and on GPU this runs
	#     under torch.autocast(bfloat16) (~8 mantissa bits, ~0.4% relative
	#     precision), so every one of the L(L-1)/2 summed pairs carries
	#     log-amplified rounding noise. The pair count grows ~L^2 (1540 at
	#     L=56, 159895 at L=566), so the NOISE FLOOR of U_am itself grows
	#     with length -- and dU values below that floor are not meaningful
	#     regardless of how the sampler behaves.
	# The recompute below measures that floor directly: predict_attention on
	# the SAME sequence twice should be bit-identical (deterministic model),
	# giving exactly 0. Anything above 0 is run-to-run numerical noise, and
	# is the resolution limit every dU in every other diagnostic is subject
	# to.
	print("\n=== (0) log(am) numerical health (length-sensitive) ===")

	ref_am = sampler.ref_eprot.am
	n_entries = ref_am.numel()
	n_zero = int((ref_am == 0).sum().item())
	n_nonfinite = int((~torch.isfinite(ref_am)).sum().item())
	min_am = ref_am.min().item()
	print(f"am: shape={tuple(ref_am.shape)}  min={min_am:.6e}  max={ref_am.max().item():.6e}  "
		  f"mean={ref_am.mean().item():.6e}  (uniform would be ~{1./(L+2):.6e})")
	print(f"am entries exactly 0: {n_zero}/{n_entries}   non-finite: {n_nonfinite}/{n_entries}")
	if n_zero or n_nonfinite:
		print("  *** log(am) IS -inf/nan FOR THESE ENTRIES -- compute_U_am has no eps guard,")
		print("      so U_am is not trustworthy at this length until this is fixed ***")
	else:
		log_ref = torch.log(ref_am)
		print(f"log(am) range: [{log_ref.min().item():.4f}, {log_ref.max().item():.4f}]  "
			  f"(all finite, so compute_U_am's raw log is safe on this reference)")

	am_again = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())
	noise_floor = compute_U_am(am_again.to(ref_am.device), ref_am).item()
	max_abs_dev = (am_again.to(ref_am.device) - ref_am).abs().max().item()
	print(f"U_am NOISE FLOOR (same sequence, two separate forward passes) = {noise_floor:.6e}")
	print(f"  max |am - am_recomputed| = {max_abs_dev:.6e}")
	print("  (0 => the forward pass is bit-reproducible and U_am has no run-to-run noise.")
	print("   Non-zero => every dU smaller in magnitude than this is numerical noise, not signal.)")

	# ================================================================
	# (1) floor consistency check
	# ================================================================
	print("\n=== (1) floor consistency check ===")

	U_am_ref_self = compute_U_am(sampler.ref_eprot.am, sampler.ref_eprot.am).item()
	print(f"compute_U_am(ref_am, ref_am) = {U_am_ref_self:.8e}  (should be exactly/near-exactly 0)")

	from utils.operations import mutate
	seq_A = mutate(REF_SEQ, N_INIT_MUTS, sampler.generator.get())
	eprot_A = ExtendedProtein(sequence=seq_A, requires_grad=False, device=device)
	eprot_A.expand()
	U_A_exact, U_am_A, _ = sampler._exact_energy(eprot_A, pars)
	hd_A = sum(1 for a, b in zip(seq_A, REF_SEQ) if a != b)

	print(f"Current A (Hd={hd_A} from ref): {seq_A}")
	print(f"U_A_exact = {U_A_exact:.4f}")

	# U >= 0 holds for every sequence (squared log-distance), so U_A_exact is
	# itself the hard ceiling on how negative any dU measured FROM seq_A can
	# be: dU = U_cand - U_A_exact >= -U_A_exact. That part is
	# reference-independent and always checkable.
	print(f"Floor bound: no dU measured from this seq_A can be below "
		  f"-U_A_exact = {-U_A_exact:.4f} (since U_cand >= 0 always).")
	if U_A_exact < 0:
		print("  *** U_A_exact < 0 -- IMPOSSIBLE for a squared log-distance. Real bug. ***")

	# The sharper, historical form of this check needs a known most-negative
	# dU from a test_informedness_vs_dt.py run on the SAME reference and
	# seq_A. Recorded per reference; None until that run exists.
	MOST_NEGATIVE_DU = {
		# protein G: site 55, from the 2026-09-22 run (code/protein_g/
		# informedness_vs_dt_results.txt). Requires U_A_exact >= 92.9918.
		"protein_g": (92.9918, "site 55"),
		# zero_polymer: site 189, from the 2026-10-02 run (code/zero_polymer/
		# informedness_vs_dt_results.txt). Requires U_A_exact >= 835.2300;
		# the same run's scan measured U_A_exact=2557.0425, so consistent.
		# (Note that dU is the TRUE BEST at that site, and it is a proline --
		# see the DEVLOG entry on proline picks.)
		"zero_polymer": (835.2300, "site 189"),
	}
	known = MOST_NEGATIVE_DU.get(REF_NAME)
	if known is None:
		print(f"(No recorded most-negative dU for reference {REF_NAME!r} yet -- run")
		print(" test_informedness_vs_dt.py on it, then record its value in this script's")
		print(" MOST_NEGATIVE_DU to enable the sharper cross-script consistency check.)")
	else:
		need, where = known
		ok = U_A_exact >= need
		print(f"(test_informedness_vs_dt.py's most negative reported dU at this seq_A")
		print(f" was -{need} at {where}; that requires U_A_exact >= {need} to be possible.")
		print(f" {'CONSISTENT' if ok else 'INCONSISTENT -- see note below'}: "
			  f"U_A_exact={U_A_exact:.4f} {'>=' if ok else '<'} {need})")
		if not ok:
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
