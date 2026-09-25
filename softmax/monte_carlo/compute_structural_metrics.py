"""
Decodes real 3D coordinates for sampled sequences (via ESM3's structure
generation + token decoder -- customs/custom_esm/models/esm3.py:
predict_protein/custom_decode, the SAME capability the old, now-abandoned
ep_sampler.py used via utils/predictions.py:predict_structure; unused by
the current single-site sampler until now) and computes two radius-of-
gyration variants per decoded checkpoint:

  Rg_ca      : "residue-residue" Rg -- each residue reduced to one point
               (its C-alpha), the standard coarse-grained protein Rg.
  Rg_allatom : "atom-atom" Rg -- every RESOLVED heavy atom in the decoded
               atom37 representation (customs/custom_esm/utils/
               residue_constants.py:atom_types -- N, CA, C, CB, O, ...,
               OXT). No hydrogens exist anywhere in atom37 by construction
               (same convention AlphaFold/ESMFold use) -- "excluding
               hydrogens" therefore needs no special filtering here; only
               atom37_mask is used, to exclude genuinely-ABSENT slots for
               a given residue's type (e.g. glycine has no CB), which is
               a different thing from excluding an element.

Both are UNWEIGHTED (geometric) Rg: Rg^2 = mean_i |r_i - r_mean|^2, not
mass-weighted.

Reimplements the last step of custom_esm.utils.decoding.custom_decode_structure
(decode_structure_with_mask below) rather than calling it directly, ONLY
because that function discards atom37_mask on return (positions only) --
everything else (generate() on the "structure" track -> structure_decoder.
decode() -> ProteinChain.from_backbone_atom_coordinates().infer_oxygen()
[.infer_cbeta()]) is copied verbatim from it. Same spirit as this
project's other small, documented reimplementations of an existing method
that drops something a diagnostic needs (e.g. tests/compare_grad_methods.py:
whole_sequence_grad, reimplementing the removed _grad_pass).

COST WARNING -- NOT established empirically, no GPU in the sandbox this
was written in: this calls the model's real structure-generation path
(model.generate on the "structure" track, GenerationConfig(num_steps=1) by
default via utils/predictions.py:init_structure_config) once per sequence.
This is meaningfully more expensive than every other analysis in this
project (site entropy / q-distribution need no model calls at all, just
the already-saved sequence strings). Decoding every checkpoint at every T
already collected (500-1500 per run) could be slow -- --stride subsamples
instead of decoding all of them (default 10, i.e. 1 in 10 post-burn-in
checkpoints).

Usage:
    python compute_structural_metrics.py <results_dir> [--stride 10] [--burn-in-frac 0.5]
        # e.g. tests_rate_sweep_temperature/T1e0/sim0

Writes <results_dir>/structural_metrics.csv (move, Rg_ca, Rg_allatom,
mean_plddt) -- the expensive step is decoupled from plotting (see
plot_structural_metrics.py), so a slow decode only ever needs to run once.
"""
import argparse
import os
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
from custom_esm.sdk.api import ESMProteinTensor
from custom_esm.utils.structure.protein_chain import ProteinChain

from analyze_ensemble import load_checkpoint_sequences, DEFAULT_REF_SEQ


CA_IDX = 1  # atom_types.index("CA") in customs/custom_esm/utils/residue_constants.py


def decode_structure_with_mask(model, eprot, config, infer_cbeta=True):
	"""Reimplements custom_esm.utils.decoding.custom_decode_structure, ONLY
	to additionally keep atom37_mask (the existing wrapper discards it) --
	see module docstring."""
	tensor = ESMProteinTensor(sequence=eprot.tokens)
	output = model.generate(input=tensor, config=config)
	decoder_output = model.get_structure_decoder().decode(output.structure)
	bb_coords = decoder_output["bb_pred"][0, 1:-1, ...].detach().cpu()
	plddt = decoder_output["plddt"][0, 1:-1].detach().cpu() if "plddt" in decoder_output else None

	chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=eprot.sequence)
	chain = chain.infer_oxygen()
	if infer_cbeta:
		chain = chain.infer_cbeta()
	return torch.tensor(chain.atom37_positions), torch.tensor(chain.atom37_mask), plddt


def radius_of_gyration(points: torch.Tensor) -> float:
	"""Unweighted (geometric) Rg over an (N,3) point set."""
	center = points.mean(dim=0)
	return torch.sqrt(((points - center) ** 2).sum(dim=-1).mean()).item()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_dir", help="e.g. tests_rate_sweep_temperature/T1e0/sim0")
	parser.add_argument("--ref-seq", default=DEFAULT_REF_SEQ)
	parser.add_argument("--burn-in-frac", type=float, default=0.5)
	parser.add_argument("--stride", type=int, default=10,
						 help="decode every Nth post-burn-in checkpoint (cost control, see module docstring)")
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	config = init_structure_config()

	sequences, max_move, n_total = load_checkpoint_sequences(args.results_dir, args.burn_in_frac)
	sequences = sequences[::args.stride]
	print(f"Decoding {len(sequences)}/{n_total} checkpoints (stride={args.stride}) from {args.results_dir}")

	rows = []
	for move, seq in sequences:
		eprot = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		eprot.expand()
		positions, mask, plddt = decode_structure_with_mask(sampler.model, eprot, config)

		ca_points = positions[:, CA_IDX, :]          # (L, 3) -- CA is always resolved
		rg_ca = radius_of_gyration(ca_points)

		allatom_points = positions[mask]             # (N_valid_heavy_atoms, 3)
		rg_allatom = radius_of_gyration(allatom_points)

		mean_plddt = plddt.mean().item() if plddt is not None else float('nan')

		rows.append((move, rg_ca, rg_allatom, mean_plddt))
		print(f"  move {move:>7}: Rg_ca={rg_ca:.3f}  Rg_allatom={rg_allatom:.3f}  mean_plddt={mean_plddt:.3f}")

	out_path = os.path.join(args.results_dir, "structural_metrics.csv")
	with open(out_path, "w") as f:
		f.write("move,Rg_ca,Rg_allatom,mean_plddt\n")
		for move, rg_ca, rg_allatom, mean_plddt in rows:
			f.write(f"{move},{rg_ca:.6f},{rg_allatom:.6f},{mean_plddt:.6f}\n")
	print(f"Saved {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
	main()
