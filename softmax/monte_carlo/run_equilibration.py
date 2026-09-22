"""
Tests whether the U/Hd_to_ref drift seen in every sweep run so far (any dt,
climbing from its start toward Hd~40+/56 with no clear plateau within 5000
moves -- see DEVLOG.txt) reflects genuine convergence toward wherever the
T=2 equilibrium distribution actually lives, or an artifact of always
starting near the reference and getting stuck behind large barriers in a
distant, disconnected basin.

Standard MCMC diagnostic: run independent chains from DISPERSED starting
points and check whether their long-run U/Hd_to_ref statistics converge to
the same regime. If they do -- even if the near-reference start takes
longer to "catch up" -- that's evidence the drift is genuine equilibration
(an entropy-vs-energy fact about compute_U_am at T=2: the bulk of sequence
space is astronomically larger than the near-reference neighborhood, so
even modest per-mutation costs get outweighed), not a starting-point
artifact. If dispersed starts settle into visibly different, non-
overlapping U/Hd regimes and stay there, that supports real barrier-
separated basins instead.

Three starting conditions:
  reference : Hd=0, exactly the reference sequence.
  moderate  : Hd~5, matching every prior run in this project.
  far       : a fully independent random canonical sequence
              (Hd~53/56 expected), maximally dispersed from the reference.

NOTE on non-canonical residues: utils/operations.py:mutate() draws its
replacement amino acid from the FULL 25-symbol vocab, including the 5
non-canonical ambiguity codes (X/B/U/Z/O), not canonical-only. Checked the
actual sequences already used throughout this project: BOTH
test_steepest_descent.py's seq_A (position 36: N->O) AND the real
sweep_dt.sh/sweep_dt_seeds.sh production runs' seed=42 starting sequence
(position 34: N->O) landed a non-canonical residue this way -- confirmed,
not hypothetical. Per classes/rate_sampler.py:_step()'s own reverse-
probability logic, competitors_B is always canonical-only, so log_a_BA is
always -inf whenever the CURRENT state at a proposed site isn't canonical
-- meaning a site holding a non-canonical residue can never be moved away
from again, ever, for the rest of that run. Minor for one frozen site (only
~1/56 of proposals touch it), but the "far" condition here implies many
more mutations, which would risk freezing several sites at once, distorting
exactly the comparison this script exists to make. mutate_canonical() /
random_canonical_sequence() below sidestep this locally (canonical-only
draws); utils/operations.py:mutate() itself is left untouched (shared with
the older ep_sampler.py -- a separate, deliberate fix if wanted later, not
bundled into this diagnostic).

_step() itself is called completely unmodified here (the real, full
56-site version, not a restricted one like validate_boltzmann_toy.py's) --
this is meant to be directly comparable to every other production run in
this project, just with a controllable starting sequence main_rate.py's
pipeline doesn't support.

Logs to <results-dir>/data.dat in a format deliberately compatible with
monte_carlo/analyze_run.py (same move/U/Hd_to_ref/site/cur_aa/dU/accepted
columns), so that script can be reused directly on the output.

Cannot be run without a working ESM3 install + downloaded weights + network
access for the first `from_pretrained` call -- this was NOT runnable in the
sandbox this was written in.

Usage:
    python run_equilibration.py --start-mode {reference,moderate,far} \
        --seed 42 --moves 15000 --results-dir tests_rate_equilibration/far_seed42
"""
import os
import sys
import argparse

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../customs"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CUSTOMS_DIR)

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from utils.operations import compute_Hd
import custom_esm.utils.constants.esm3 as C


REF_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def mutate_canonical(ref_sequence, n_mutations, generator, canonical_letters):
	"""Like utils.operations.mutate, but draws replacement amino acids only
	from canonical_letters, not the full 25-symbol vocab -- see module
	docstring. Sites can still repeat (matching mutate()'s own behavior),
	so n_mutations is an upper bound on the resulting Hamming distance,
	not a guarantee."""
	L = len(ref_sequence)
	sites = torch.randint(0, L, (n_mutations,), device=generator.device, generator=generator).tolist()
	idxs = torch.randint(0, len(canonical_letters), (n_mutations,), device=generator.device, generator=generator).tolist()
	seq = list(ref_sequence)
	for s, i in zip(sites, idxs):
		seq[s] = canonical_letters[i]
	return "".join(seq)


def random_canonical_sequence(length, generator, canonical_letters):
	idxs = torch.randint(0, len(canonical_letters), (length,), device=generator.device, generator=generator).tolist()
	return "".join(canonical_letters[i] for i in idxs)


def main(args):
	os.makedirs(args.results_dir, exist_ok=True)

	pars = {
		"T": args.T, "dt": args.dt, "M": 1.0, "T_sftm": 0.1,
		"lambda_am": 1.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": 40,
	}
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	# Needed here for the same reason as validate_boltzmann_toy.py: a full
	# many-move run inevitably exercises _step()'s reverse-direction logic
	# (competitors_B == cur_idx), which needs _canonical_idx on the same
	# device as the CUDA tensors it's compared against. _setup() would
	# normally do this; this script bypasses _setup() to control the
	# starting sequence directly.
	sampler._canonical_idx = sampler._canonical_idx.to(device)

	from generator.custom_generator import CustomGenerator
	sampler.generator = CustomGenerator(seed=args.seed, device=device)

	sampler.ref_eprot = ExtendedProtein(sequence=REF_SEQ, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())

	vocab = C.SEQUENCE_USED_VOCAB
	canonical = [vocab[i] for i in sampler._canonical_idx.tolist()]

	if args.start_mode == "reference":
		start_seq = REF_SEQ
	elif args.start_mode == "moderate":
		start_seq = mutate_canonical(REF_SEQ, 5, sampler.generator.get(), canonical)
	elif args.start_mode == "far":
		start_seq = random_canonical_sequence(len(REF_SEQ), sampler.generator.get(), canonical)
	else:
		raise ValueError(f"unknown start_mode {args.start_mode}")

	hd0 = sum(1 for a, b in zip(start_seq, REF_SEQ) if a != b)
	print(f"PID: {os.getpid()}")
	print(f"start_mode={args.start_mode}  seed={args.seed}  dt={args.dt}  T={args.T}  moves={args.moves}")
	print(f"start_seq (Hd={hd0} from ref): {start_seq}")

	eprot = ExtendedProtein(sequence=start_seq, requires_grad=True, device=device)
	eprot.expand()
	U, _, eprot = sampler._exact_energy(eprot, pars)

	log_path = os.path.join(args.results_dir, "data.dat")
	with open(log_path, "w") as f:
		f.write("move\tU\tHd_to_ref\tsite\tcur_aa\tdU\taccepted\n")
		f.write(f"0\t{U}\t{hd0}\t0\t\t0.0\t\n")

		n_accepted = 0
		for move in range(1, args.moves+1):
			eprot, info = sampler._step(eprot, U, pars)
			if info["accepted"]:
				U = info["U_B"]
			n_accepted += info["accepted"]
			hd = compute_Hd(eprot.logits, sampler.ref_eprot.logits)
			f.write(f"{move}\t{U}\t{hd}\t{info['site']}\t{info['cur_aa']}\t{info['dU']}\t{info['accepted']}\n")

			if move % max(1, args.moves//20) == 0:
				f.flush()
				print(f"  move {move:6d}/{args.moves}  U={U:.3f}  Hd={hd}  acc_rate={n_accepted/move:.3f}")

	print(f"\nDone. Log at {log_path}")


def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--start-mode", choices=["reference", "moderate", "far"], required=True)
	parser.add_argument("--seed", type=int, required=True)
	parser.add_argument("--moves", type=int, default=15000)
	parser.add_argument("--dt", type=float, default=2000.0)
	parser.add_argument("--T", type=float, default=2.0)
	parser.add_argument("--results-dir", type=str, required=True)
	return parser


if __name__ == "__main__":
	parser = create_parser()
	args = parser.parse_args()
	main(args)
