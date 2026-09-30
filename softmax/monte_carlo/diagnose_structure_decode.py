"""
Direct diagnostic for the Rg/pLDDT-flat-across-T finding (see DEVLOG.txt,
2026-09-30 "full paper-scale plotting suite run" entry): decodes a small,
deliberately extreme pair of sequences -- the exact reference (expected
highly foldable) and a fully-scrambled random sequence of the same
length (expected NOT foldable) -- directly, outside any sampler/sweep
context, at several num_steps values for the structure GenerationConfig.

This isolates the question from the paper-scale sweep's own T range,
--stride choice, and burn-in/equilibration state entirely:
  - If reference and scrambled DON'T separate in mean pLDDT/Rg at ANY
    num_steps tested here, that points at explanation (1) or (2) from
    the DEVLOG entry -- either the one-step decode setup itself, or
    ESM3's general-purpose structure track not being well-calibrated for
    foldability the way the paper's dedicated ESMFold is -- rather than
    anything about the sweep.
  - If they DO separate clearly (especially at higher num_steps), that
    instead points at (3): the sweep's own num_steps=1 default understating
    real differences that a properly-configured decode would show, in
    which case the T-sweep's Rg/pLDDT numbers should be treated as
    provisional and worth re-running with the working num_steps.

Reuses compute_structural_metrics.py's decode_structure_with_mask/
radius_of_gyration/CA_IDX directly rather than reimplementing them.

Note on trials: init_structure_config()'s default temperature=0.0 makes
generation close to deterministic -- repeated trials at the SAME
num_steps are a sanity check (should be near-identical, not a source of
real variation), not an attempt to sample real stochasticity. Real
variation, if any shows up, should come from varying num_steps itself.
There's no generator/seed argument threaded through model.generate()
anywhere in this codebase, so torch.manual_seed() (the global RNG) is
the only practical reproducibility hook available here.

Usage:
    python diagnose_structure_decode.py [--num-steps 1 4 8] [--trials 3]
"""
import argparse
import os
import statistics
import sys

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CUSTOMS_DIR)

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.predictions import init_structure_config
import custom_esm.utils.constants.esm3 as C

from compute_structural_metrics import decode_structure_with_mask, radius_of_gyration, CA_IDX
from analyze_ensemble import DEFAULT_REF_SEQ


def scramble_sequence(ref_seq, seed):
	# Canonical residues only, same exclusion set _NONCANONICAL_RESIDUES
	# uses elsewhere in this project (classes/rate_sampler.py) -- a
	# scrambled sequence landing on X/B/U/Z/O would be an unfair,
	# different kind of "bad" sequence (ambiguity codes, not just a
	# shuffled-but-still-canonical one).
	noncanonical = ('X', 'B', 'U', 'Z', 'O')
	vocab = [a for a in C.SEQUENCE_USED_VOCAB if a not in noncanonical]
	g = torch.Generator().manual_seed(seed)
	idx = torch.randint(0, len(vocab), (len(ref_seq),), generator=g)
	return "".join(vocab[i] for i in idx.tolist())


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--num-steps", nargs="+", type=int, default=[1, 4, 8])
	parser.add_argument("--trials", type=int, default=3,
						 help="repeat each (sequence, num_steps) combo this many times "
							  "(sanity check, see module docstring -- not expected to vary much)")
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)

	ref_seq = args.ref_seq
	scrambled_seq = scramble_sequence(ref_seq, seed=0)
	n_shared = sum(a == b for a, b in zip(ref_seq, scrambled_seq))

	print(f"Reference:  {ref_seq}")
	print(f"Scrambled:  {scrambled_seq}")
	print(f"(scrambled shares {n_shared}/{len(ref_seq)} sites with reference, by chance)")
	print()

	sequences = {"reference": ref_seq, "scrambled": scrambled_seq}

	header = f"{'label':>10} {'num_steps':>10} {'trial':>6} {'Rg_ca':>8} {'Rg_allatom':>11} {'mean_plddt':>11}"
	print(header)
	print("-" * len(header))

	results = {}
	for label, seq in sequences.items():
		eprot = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		eprot.expand()

		for num_steps in args.num_steps:
			config = init_structure_config(num_steps=num_steps)

			for trial in range(args.trials):
				torch.manual_seed(1000 * num_steps + trial)
				positions, mask, plddt = decode_structure_with_mask(sampler.model, eprot, config)

				ca_points = positions[:, CA_IDX, :]
				rg_ca = radius_of_gyration(ca_points)
				allatom_points = positions[mask]
				rg_allatom = radius_of_gyration(allatom_points)
				mean_plddt = plddt.mean().item() if plddt is not None else float('nan')

				results.setdefault((label, num_steps), []).append(mean_plddt)
				print(f"{label:>10} {num_steps:>10} {trial:>6} {rg_ca:>8.3f} {rg_allatom:>11.3f} {mean_plddt:>11.4f}")

	print()
	print("=== summary: mean pLDDT by (label, num_steps), pooled over trials ===")
	print(f"{'label':>10} {'num_steps':>10} {'mean_plddt':>12} {'std':>8}")
	for (label, num_steps), vals in results.items():
		std = statistics.stdev(vals) if len(vals) > 1 else 0.0
		print(f"{label:>10} {num_steps:>10} {statistics.mean(vals):>12.4f} {std:>8.4f}")

	print()
	print("Read this looking for whether 'reference' and 'scrambled' mean_plddt separate at ANY")
	print("num_steps -- if they stay close together even at num_steps=8, the flatness seen in the")
	print("T-sweep is likely NOT fixable by more decode steps (points at explanation 2: ESM3's")
	print("structure track itself, not the GenerationConfig). If they separate clearly once")
	print("num_steps is high enough, the T-sweep should be re-run with that higher num_steps")
	print("before trusting its existing Rg/pLDDT numbers.")


if __name__ == "__main__":
	main()
