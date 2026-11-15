"""
Exhaustive single-mutant scan of U_am on the PTEN reference: every one of the
324 x 19 = 6156 point mutants, each evaluated DIRECTLY from the reference
sequence (Hamming distance 1, no drift). The deep-mutational-scanning
counterpart of the sampler's energy, and so the thing that can be put next to
VAMP-seq abundance (Matreyek 2018/2021) and Mighell's activity variant by
variant rather than site by site.

Why it is interpretable at all: U = lambda_am * compute_U_am(am, ref_am) =
sum_{i<j} [log am_ij - log ref_am_ij]^2 with lambda_am = 1 and
lambda_structure_ce = 0 everywhere in this project. That is a squared
log-distance, so U >= 0 for every sequence and U(reference) = 0 is the global
minimum. Hence for a mutant of the reference

    dU = U(mutant) - U(reference) = U(mutant) >= 0,

i.e. the number this script writes is both the mutant's energy and its cost
relative to the reference, with no subtraction and no reference run needed.
Small dU = the sampler finds that substitution cheap = "tolerant"; large dU =
expensive = "the model thinks this residue cannot change". The sign convention
is therefore the OPPOSITE of the experimental scores (high VAMP-seq score =
tolerated), so pten_k_sites.py mutants correlates -dU with them.

Needs ESM3 and a GPU: 6156 forward passes. On astaroth (RTX 4500 Ada) one
L=324 no-grad forward is ~0.1-0.2 s, so expect roughly 20-45 min at
--batch 19, a few hours at --batch 1. It is NOT runnable in the cloud
session that wrote it (no torch there).

The run is resumable and crash-safe: the CSV is appended site by site and
flushed after each site, and re-running the same command skips every site
already complete in it (19 rows). Kill it and restart it freely.

Usage (from softmax/monte_carlo):

    nohup python -u scan_single_mutants.py --out pten_natural/pten_single_mutants.csv \
        > scan_single_mutants.log 2>&1 &

    # quick sanity run: 5 sites, and check the batched energy against the
    # sampler's own _exact_energy on the first of them
    python -u scan_single_mutants.py --sites 0 1 2 3 4 --check --out /tmp/scan_test.csv

Then compare with the papers and the natural alignments:

    python pten_k_sites.py mutants --mutants pten_natural/pten_single_mutants.csv
"""
import argparse
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../customs"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CUSTOMS_DIR)
sys.path.append(os.path.join(CODE_DIR, "tests"))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler
from classes.ExtendedProtein import ExtendedProtein
from generator.custom_generator import CustomGenerator
from utils.energies import compute_U_am
from custom_esm.utils.custom import clean_and_symm
from custom_esm.utils.structure.affine3d import build_affine3d_from_coordinates
import custom_esm.utils.constants.esm3 as C

from references import get_reference

# The pars the whole PTEN study runs with (run_nsites.py). Only lambda_am and
# lambda_structure_ce enter _exact_energy; T and the proposal parameters are
# irrelevant here, since nothing is proposed or accepted.
PARS = {"T": 1.0, "M": 1.0, "T_sftm": 0.1, "lambda_am": 1.0,
	"lambda_structure_ce": 0.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": 40}
COLUMNS = "site,ref_aa,mut_aa,dU\n"


def batched_U(model, probs, ref_am):
	"""U_am for a (B, L, K) batch of sequence-probability tensors, in one forward
	pass. Lifted from code/tests/bench_batch.py, which checked it against
	_exact_energy sequence by sequence: ESM3's own predict_attention asserts
	batch 1, so the (1, L) default inputs are expanded to B by hand. Same
	bfloat16 autocast as predict_attention, so the numbers are the sampler's."""
	B, L = probs.shape[0], probs.shape[1] + 2
	tokens, _, _ = model._default(L, probs.device)
	tokens = tuple(t.expand(B, *t.shape[1:]) for t in tokens)
	coords = torch.full((B, L, 3, 3), float("nan"), dtype=torch.float, device=probs.device)
	affine, affine_mask = build_affine3d_from_coordinates(coords)
	with torch.no_grad():
		with torch.autocast(enabled=probs.device.type == "cuda", device_type=probs.device.type, dtype=torch.bfloat16):
			x = model.encoder.custom_forward(probs, *tokens)
			attention = model.transformer.last_attention(x, None, affine, affine_mask, None)
		am = clean_and_symm(attention)
		return torch.triu((torch.log(am) - torch.log(ref_am))**2.).sum(dim=(-2, -1))


def probs_of(seqs, device):
	out = []
	for s in seqs:
		e = ExtendedProtein(sequence=s, requires_grad=False, device=device)
		e.expand()
		out.append(e.get_probs())
	return torch.stack(out)


def done_sites(path, n_expected):
	"""Sites already complete in an existing CSV, and whether it is well formed.
	A site is complete when it has all n_expected rows, so a run killed in the
	middle of a site simply redoes that site (its partial rows are dropped)."""
	if not os.path.exists(path):
		return set(), []
	keep, counts = [], {}
	with open(path) as f:
		head = f.readline()
		if head != COLUMNS:
			raise SystemExit(f"{path} does not start with {COLUMNS.strip()!r}; move it aside or pass another --out")
		for line in f:
			s = line.split(",", 1)[0]
			if s.isdigit():
				counts[int(s)] = counts.get(int(s), 0) + 1
				keep.append(line)
	full = {s for s, n in counts.items() if n == n_expected}
	return full, [l for l in keep if int(l.split(",", 1)[0]) in full]


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", default="pten_1d5r", help="reference name in code/tests/references.py (default pten_1d5r)")
	ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
		"pten_natural", "pten_single_mutants.csv"))
	ap.add_argument("--batch", type=int, default=19,
		help="mutants per forward pass (default 19, one whole site). 1 reproduces the sampler's own call exactly; "
			"on an out-of-memory error the batch is halved automatically")
	ap.add_argument("--sites", type=int, nargs="+", help="only these 0-based sites (default: all of them)")
	ap.add_argument("--device", default="cuda:0")
	ap.add_argument("--check", action="store_true",
		help="also compute the first scanned site with the sampler's _exact_energy, one sequence at a time, and print "
			"the largest disagreement with the batched path")
	ap.add_argument("--restart", action="store_true", help="ignore an existing --out and scan everything again")
	args = ap.parse_args()

	ref_name, ref_seq = get_reference(args.ref)
	L = len(ref_seq)
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	vocab = C.SEQUENCE_USED_VOCAB

	print(f"# reference {ref_name}, L={L}", flush=True)
	print(f"# device {device}", flush=True)

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	sampler.generator = CustomGenerator(seed=0, device=device)
	sampler._canonical_idx = sampler._canonical_idx.to(device)
	sampler.ref_eprot = ExtendedProtein(sequence=ref_seq, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())
	ref_am = sampler.ref_eprot.am

	# The resolution limit of every dU below: two forward passes on the SAME
	# sequence should give exactly 0, and anything above that is run-to-run
	# numerical noise in the bfloat16 attention (see the long comment in
	# code/tests/scan_U_am_landscape.py). dU values under this floor mean
	# nothing, whatever they are compared with.
	again = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())
	floor = compute_U_am(again.to(ref_am.device), ref_am).item()
	print(f"# U_am(ref, ref) = {compute_U_am(ref_am, ref_am).item():.3e}   "
		f"noise floor (same sequence twice) = {floor:.3e}", flush=True)

	# 19 candidates per site: the 20 canonical residues minus the site's own.
	# _competition_indices is the sampler's own eligibility rule, so the scan
	# covers exactly the substitutions a move could ever propose (X, B, U, Z, O
	# are structurally absent from it).
	n_per_site = len(sampler._competition_indices(vocab.index(ref_seq[0])))
	sites = args.sites if args.sites is not None else list(range(L))
	if args.restart and os.path.exists(args.out):
		os.remove(args.out)
	have, kept = done_sites(args.out, n_per_site)
	todo = [s for s in sites if s not in have]
	print(f"# {n_per_site} mutants per site, {len(sites)} sites requested, "
		f"{len(have & set(sites))} already in {args.out}, {len(todo)} to do "
		f"({len(todo) * n_per_site} forward passes)", flush=True)

	os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
	with open(args.out, "w") as f:               # rewritten without the rows of any partial site
		f.write(COLUMNS)
		f.writelines(kept)

	batch, t0, done = max(1, args.batch), time.perf_counter(), 0
	with open(args.out, "a") as out:
		for s in todo:
			cur = ref_seq[s]
			cand_idx = sampler._competition_indices(vocab.index(cur)).tolist()
			seqs = [ref_seq[:s] + vocab[j] + ref_seq[s + 1:] for j in cand_idx]
			dU = []
			i = 0
			while i < len(seqs):
				chunk = seqs[i:i + batch]
				try:
					if batch == 1:
						e = ExtendedProtein(sequence=chunk[0], requires_grad=False, device=device)
						e.expand()
						dU.append(sampler._exact_energy(e, PARS)[0])
					else:
						dU += batched_U(sampler.model, probs_of(chunk, device), ref_am).float().tolist()
				except torch.cuda.OutOfMemoryError:
					torch.cuda.empty_cache()
					if batch == 1:
						raise
					batch = max(1, batch // 2)
					print(f"# out of memory, batch -> {batch}", flush=True)
					continue
				i += len(chunk)
			for j, u in zip(cand_idx, dU):
				out.write(f"{s},{cur},{vocab[j]},{u:.6f}\n")
			out.flush()
			done += 1
			if args.check and done == 1:
				ex = []
				for seq in seqs:
					e = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
					e.expand()
					ex.append(sampler._exact_energy(e, PARS)[0])
				d = np.abs(np.array(ex) - np.array(dU))
				print(f"# check at site {s}: max |batched - _exact_energy| = {d.max():.3e} "
					f"(dU range {min(dU):.3f} to {max(dU):.3f}; the noise floor above is the scale to judge it against)",
					flush=True)
			if done % 10 == 0 or done == len(todo):
				el = time.perf_counter() - t0
				print(f"# {done}/{len(todo)} sites, {el / 60:.1f} min, "
					f"{el / done / n_per_site:.3f} s per mutant, "
					f"{(len(todo) - done) * el / done / 60:.1f} min left", flush=True)
			if done <= 3 or done % 50 == 0:
				print(f"site {s:3d} {cur}: dU min {min(dU):10.3f} ({vocab[cand_idx[int(np.argmin(dU))]]}) "
					f"median {np.median(dU):10.3f} max {max(dU):10.3f} "
					f"({vocab[cand_idx[int(np.argmax(dU))]]})", flush=True)

	print(f"\n# done. {args.out}", flush=True)
	print("# next: python pten_k_sites.py mutants --mutants " + args.out, flush=True)


if __name__ == "__main__":
	main()
