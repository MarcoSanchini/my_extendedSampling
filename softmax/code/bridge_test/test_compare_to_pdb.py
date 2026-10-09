"""
Logic tests for compare_to_pdb.py and prompting.py, on SYNTHETIC structures.

These check the code, not any protein: alignment with unresolved residues and a
mismatch, the superposition, the contact counts, the SG-SG test that decides
whether a supplied disulfide is really a bond, the C-beta convention, the
superposition-free local-structure check (which must find a deliberately
stretched segment), and the coordinate helpers used for prompts. A scrambled
structure must score badly.

Synthetic structures say nothing about real proteins, so passing here means the
arithmetic is right and nothing more. Run from softmax/code:

    python bridge_test/test_compare_to_pdb.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))

import compare_to_pdb as C  # noqa: E402
import prompting as PR  # noqa: E402
import references as R  # noqa: E402

THREE = {v: k for k, v in C.AA3.items() if k != "MSE"}
SEQ = R.BPTI
BONDS = R.BPTI_DISULFIDES
ATOM_ORDER = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "SG": 10}


def synthetic_backbone(n):
	"""A compact curve in a ~20 A box, so there are many long-range contacts."""
	i = np.arange(n)
	return np.stack([10 * np.sin(0.35 * i), 10 * np.cos(0.23 * i), 10 * np.sin(0.17 * i + 1.0)], axis=1)


def build_atoms(seq, ca):
	"""Per-residue atom dicts around each CA. Offsets are arbitrary but fixed;
	glycine gets no CB, cysteine gets an SG."""
	atoms = []
	for aa, c in zip(seq, ca):
		a = {"N": c + [-1.2, 0.5, 0.0], "CA": c.copy(), "C": c + [1.2, 0.5, 0.0], "O": c + [1.7, 1.0, 0.0]}
		if aa != "G":
			a["CB"] = c + [0.0, -1.0, 1.2]
		if aa == "C":
			a["SG"] = c + [0.0, -2.0, 2.2]
		atoms.append(a)
	return atoms


def write_pdb(path, seq, atoms, keep, first_number=1, mutate=None):
	"""Write the residues in `keep` (0-based indices). `mutate` maps index->aa to
	change the residue name without moving atoms."""
	lines, serial = [], 1
	for i in keep:
		aa = (mutate or {}).get(i, seq[i])
		for name, xyz in atoms[i].items():
			lines.append(
				f"ATOM  {serial:5d} {name:<4s} {THREE[aa]:>3s} A{first_number + i:4d}    "
				f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]:>2s}")
			serial += 1
	lines.append("END")
	Path(path).write_text("\n".join(lines) + "\n")


def random_rotation(rng):
	q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
	return q * np.sign(np.linalg.det(q))


def move(atoms, rot, shift, rng, noise=0.3):
	"""Rigidly move every atom, then add per-atom Gaussian noise."""
	return [{k: v @ rot.T + shift + rng.normal(scale=noise, size=3) for k, v in a.items()} for a in atoms]


def run_compare(exp, pred, extra=()):
	out = subprocess.run(
		[sys.executable, str(HERE / "compare_to_pdb.py"), "--ref", "bpti", "--exp", str(exp), "--pred", str(pred), *extra],
		capture_output=True, text=True)
	assert out.returncode == 0, out.stderr + out.stdout
	return out.stdout


def grab(text, pattern, group=1):
	m = re.search(pattern, text)
	assert m, f"pattern {pattern!r} not found in:\n{text}"
	return float(m.group(group))


def main():
	rng = np.random.default_rng(1)
	n = len(SEQ)
	full = build_atoms(SEQ, synthetic_backbone(n))
	for p, q in BONDS:  # bonded cysteines: SG atoms 2.04 A apart, as in a real bond
		full[q - 1]["SG"] = full[p - 1]["SG"] + np.array([2.04, 0.0, 0.0])

	# --- unit checks
	P = rng.normal(size=(30, 3))
	R3 = random_rotation(rng)
	rmsd, _ = C.rmsd_after_fit(P @ R3.T + 5.0, P)
	assert rmsd < 1e-6, f"Kabsch should recover a rigid motion exactly, got {rmsd}"
	mp, match, mism = C.align("ACDEFGHIK", "XXCDEFGHIKYY")
	assert mp[0] is None and mp[1] == 2 and mp[8] == 9 and match == 8 and not mism, (mp, match, mism)
	mp, match, mism = C.align("ACDEFGHIK", "ACDEWGHIK")
	assert len(mism) == 1 and mism[0][0] == 4, mism

	# C-beta convention: the model's infer_CB must give the bond length and the
	# N-CA-CB angle it is parameterised with, for any N, CA, C. The tolerance is
	# 1e-5, not tighter: the 1e-8 regulariser in its normalisation leaves ~1e-6 A.
	for _ in range(20):
		Nn, CAx, Cx = rng.normal(size=(3, 3)) * 2
		cbx = C.infer_CB(Cx, Nn, CAx)
		assert abs(np.linalg.norm(cbx - CAx) - 1.522) < 1e-5, "CA-CB bond length must be 1.522 A"
		cosang = np.dot(Nn - CAx, cbx - CAx) / (np.linalg.norm(Nn - CAx) * np.linalg.norm(cbx - CAx))
		assert abs(np.arccos(np.clip(cosang, -1, 1)) - 1.927) < 1e-5, "N-CA-CB angle must be 1.927 rad"
	print("[ok] C-beta convention: CA-CB 1.522 A and N-CA-CB 1.927 rad for arbitrary backbone")

	with tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)

		# experimental: residues 1-3 and 20 unresolved, numbering offset by 9
		keep = [i for i in range(n) if i not in (0, 1, 2, 19)]
		write_pdb(tmp / "exp.pdb", SEQ, full, keep, first_number=10)
		write_pdb(tmp / "exp_all.pdb", SEQ, full, range(n))

		# predicted: every residue, rotated and translated, with 0.3 A noise on each atom
		rot, shift = random_rotation(rng), np.array([12.0, -7.0, 3.0])
		moved = move(full, rot, shift, rng)
		write_pdb(tmp / "pred.pdb", SEQ, moved, range(n))

		text = run_compare(tmp / "exp.pdb", tmp / "pred.pdb")
		assert "absent positions: 1-3, 20" in text, text
		assert "0 mismatched" in text
		r = grab(text, r"CA RMSD\s+([\d.]+)")
		tm = grab(text, r"TM-score \(approx\.\)\s+([\d.]+)")
		f1 = grab(text, r"\|i-j\| >= 6\s.*?F1\s+([\d.]+)")
		assert r < 0.8, f"RMSD should be about the 0.3 A noise, got {r}"
		assert tm > 0.9, f"TM-score should be near 1, got {tm}"
		assert f1 > 0.85, f"contact F1 should be near 1, got {f1}"
		assert "known bonds confirmed in the experimental file (SG-SG <= 2.5 A): 3/3" in text, text
		print(f"[ok] aligned, superposed and scored a noisy copy: RMSD {r:.2f}, TM {tm:.3f}, F1 {f1:.2f}")

		# a supplied pairing that is NOT a bond in the experimental file must be flagged
		text = run_compare(tmp / "exp.pdb", tmp / "pred.pdb", extra=["--bonds", "5-55,14-38,5-30"])
		assert "confirmed in the experimental file (SG-SG <= 2.5 A): 2/3" in text, text
		assert "WARNING: not every supplied pairing is a bond" in text
		print("[ok] a wrong pairing is reported as not a bond in the experimental file")

		# a sequence mismatch must be listed with its position
		write_pdb(tmp / "exp_mut.pdb", SEQ, full, keep, first_number=10, mutate={21: "A"})
		text = run_compare(tmp / "exp_mut.pdb", tmp / "pred.pdb")
		assert "1 mismatched" in text and "position 22: reference" in text, text
		print("[ok] a sequence mismatch is listed with its position")

		# local structure, superposition-free: a noisy copy must show small
		# intra-chain errors; a copy with residues 10-16 stretched by 30% must have
		# larger errors, and the worst window must overlap 10-16
		text = run_compare(tmp / "exp_all.pdb", tmp / "pred.pdb")
		noise_text = text
		noise_mae = grab(text, r"CA-CA distance error, all pairs\s+MAE\s+([\d.]+) A")
		assert noise_mae < 0.5, f"noise-only local error should be small, got {noise_mae}"
		seg = list(range(9, 16))
		stretched = [dict(a) for a in full]
		cen = np.mean([full[i]["CA"] for i in seg], axis=0)
		for i in seg:
			stretched[i] = {k: cen + 1.3 * (v - cen) for k, v in full[i].items()}
		write_pdb(tmp / "pred_stretch.pdb", SEQ, move(stretched, rot, shift, rng), range(n))
		text = run_compare(tmp / "exp_all.pdb", tmp / "pred_stretch.pdb")
		stretch_mae = grab(text, r"CA-CA distance error, all pairs\s+MAE\s+([\d.]+) A")
		# The all-pairs error is dominated by pairs outside the segment, so it barely
		# moves. The localising measure is the worst window, compared with the same
		# window in the noise-only run.
		win_pat = r"worst 7-residue windows.*?\n\s+[A-Z](\d+)-[A-Z](\d+)\s+([\d.]+) A"
		m = re.search(win_pat, text, re.S)
		assert m, text
		start, end, win_stretch = int(m.group(1)), int(m.group(2)), float(m.group(3))
		m0 = re.search(win_pat, noise_text, re.S)
		assert m0, noise_text
		win_noise = float(m0.group(3))
		assert start <= 16 and end >= 10, f"worst window {start}-{end} should overlap the stretched 10-16"
		assert win_stretch > 3 * win_noise, (win_noise, win_stretch)
		print(f"[ok] local check: worst window {start}-{end} at {win_stretch:.2f} A when 10-16 is stretched, "
			  f"vs {win_noise:.2f} A for the noise-only copy")

		# negative control: scrambled coordinates must score badly
		perm = rng.permutation(n)
		scrambled = [full[k] for k in perm]
		write_pdb(tmp / "scr.pdb", SEQ, scrambled, range(n))
		text = run_compare(tmp / "exp.pdb", tmp / "scr.pdb")
		r2 = grab(text, r"CA RMSD\s+([\d.]+)")
		tm2 = grab(text, r"TM-score \(approx\.\)\s+([\d.]+)")
		f12 = grab(text, r"\|i-j\| >= 6\s.*?F1\s+([\d.]+)")
		assert r2 > 5.0 and tm2 < 0.5 and f12 < 0.6, (r2, tm2, f12)
		print(f"[ok] a scrambled structure scores badly: RMSD {r2:.1f}, TM {tm2:.3f}, F1 {f12:.2f}")

		# --- prompting helpers (used by roundtrip_bridge.py)
		coords, info = PR.experimental_coords37(SEQ, tmp / "exp.pdb", None, ATOM_ORDER)
		assert coords.shape == (n, 37, 3)
		assert info["absent"] == [1, 2, 3, 20] and info["resolved"] == n - 4, info
		assert np.isnan(coords[0]).all() and np.isnan(coords[19]).all(), "unresolved residues must stay NaN"
		assert np.isfinite(coords[4, ATOM_ORDER["SG"]]).all(), "a resolved cysteine keeps its SG"

		gly = [i for i, a in enumerate(SEQ) if a == "G"]
		inf = PR.with_inferred_cb(coords, SEQ, ATOM_ORDER)
		for i in range(n):
			if i in gly or not np.isfinite(coords[i, [0, 1, 2]]).all():
				assert np.isnan(inf[i, ATOM_ORDER["CB"]]).all(), f"residue {i + 1} should have no C-beta"
			else:
				want = C.infer_CB(coords[i, 2], coords[i, 0], coords[i, 1])
				assert np.allclose(inf[i, ATOM_ORDER["CB"]], want), f"residue {i + 1} C-beta not on the model convention"
		assert np.allclose(inf[:, [0, 1, 2]], coords[:, [0, 1, 2]], equal_nan=True), "backbone must be untouched"

		prompt, given = PR.partial_prompt(coords, [5, 55], 0, ATOM_ORDER)
		assert given == [5, 55], given
		for name in ("N", "CA", "C"):
			assert np.isfinite(prompt[4, ATOM_ORDER[name]]).all() and np.isfinite(prompt[54, ATOM_ORDER[name]]).all()
		assert np.isnan(prompt[4, ATOM_ORDER["CB"]]).all(), "a prompt carries backbone only, never side-chain atoms"
		assert np.isnan(np.delete(prompt, [4, 54], axis=0)).all(), "every other residue must stay unknown"
		_, given_w = PR.partial_prompt(coords, [5, 55], 1, ATOM_ORDER)
		assert given_w == [4, 5, 6, 54, 55, 56], given_w
		_, given_edge = PR.partial_prompt(coords, [1], 2, ATOM_ORDER)
		assert given_edge == [1, 2, 3], given_edge
		d = PR.pair_distance(coords, 4, 54, ATOM_ORDER, "SG")
		assert abs(d - 2.04) < 1e-6, d
		assert np.isnan(PR.pair_distance(coords, 0, 54, ATOM_ORDER)), "an unresolved residue has no distance"
		print("[ok] coordinate helpers: alignment, C-beta convention, backbone-only prompts, windows, distances")

	print("\nall logic tests passed (synthetic data: the arithmetic is right, nothing more)")


if __name__ == "__main__":
	main()
