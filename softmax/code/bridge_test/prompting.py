"""
Coordinate helpers for roundtrip_bridge.py, in plain numpy so they can be tested
without torch or the ESM3 weights.

Three jobs:
  * turn an experimental PDB chain into an atom37 array that lines up with the
    reference sequence in references.py (missing residues and atoms stay NaN);
  * build a PARTIAL coordinate prompt from it -- only the backbone N/CA/C of the
    chosen residues, everything else NaN -- which is how ESM3 is told where
    something is without being told what the rest looks like;
  * measure a pair distance in an atom37 array.

NaN means "not given". build_affine3d_from_coordinates drops any non-finite
coordinate when it builds the frames the geometric attention uses, so NaN rows
are simply residues the model has no geometry for.

`atom_order` is passed in rather than imported, because the real table lives in
custom_esm.utils.residue_constants and importing that package pulls in torch.
"""
from pathlib import Path

import numpy as np

import compare_to_pdb as C

BACKBONE = ("N", "CA", "C")


def experimental_coords37(ref_seq, exp_path, chain, atom_order):
	"""atom37 coordinates (L, 37, 3) for `ref_seq`, from an experimental chain.

	Returns (coords, info). Reference positions with no experimental residue stay
	NaN. A residue whose type differs from the reference keeps its coordinates
	but is listed in info["mismatches"], so the caller can say so rather than
	silently using a different residue's geometry."""
	residues, used_chain = C.read_pdb(exp_path, chain)
	mapping, n_match, mism = C.align(ref_seq, "".join(r["aa"] for r in residues))
	coords = np.full((len(ref_seq), 37, 3), np.nan)
	for i, j in enumerate(mapping):
		if j is None:
			continue
		for name, xyz in residues[j]["atoms"].items():
			if name in atom_order:
				coords[i, atom_order[name]] = xyz
	info = {
		"chain": used_chain,
		"resolved": sum(1 for j in mapping if j is not None),
		"mismatches": [(k + 1, a, b) for k, a, b in mism],
		"absent": [k + 1 for k, j in enumerate(mapping) if j is None],
	}
	return coords, info


def partial_prompt(coords37, residues, window, atom_order):
	"""A prompt that carries only backbone N/CA/C for the given residues.

	`residues` are 1-based positions in the reference numbering; `window` adds
	that many neighbours on each side. Everything not selected is NaN."""
	n = coords37.shape[0]
	chosen = set()
	for r in residues:
		for k in range(r - 1 - window, r + window):
			if 0 <= k < n:
				chosen.add(k)
	prompt = np.full_like(coords37, np.nan)
	for k in sorted(chosen):
		for name in BACKBONE:
			prompt[k, atom_order[name]] = coords37[k, atom_order[name]]
	return prompt, sorted(k + 1 for k in chosen)


def pair_distance(coords37, i, j, atom_order, atom="CB"):
	"""Distance between `atom` of residues i and j (0-based), NaN if either is
	absent. Glycine has no CB; callers asking for CB get NaN there."""
	a = coords37[i, atom_order[atom]]
	b = coords37[j, atom_order[atom]]
	if not (np.isfinite(a).all() and np.isfinite(b).all()):
		return float("nan")
	return float(np.linalg.norm(a - b))


def to_numpy(x):
	"""A torch tensor or array as a float numpy array, without importing torch."""
	if hasattr(x, "detach"):
		x = x.detach().cpu().numpy()
	return np.asarray(x, dtype=float)
