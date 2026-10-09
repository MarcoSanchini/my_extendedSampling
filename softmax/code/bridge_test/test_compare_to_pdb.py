"""
Logic tests for compare_to_pdb.py, on SYNTHETIC structures.

These check the code, not any protein. They cover: alignment with unresolved
residues and with a sequence mismatch, the superposition, the contact counts,
the SG-SG test that decides whether a supplied disulfide is really a bond, and
a scrambled negative control that must score badly.

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
import references as R  # noqa: E402

THREE = {v: k for k, v in C.AA3.items() if k != "MSE"}
SEQ = R.BPTI
BONDS = R.BPTI_DISULFIDES


def synthetic_backbone(n, seed=0):
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

	# bonded cysteines: place the two SG atoms 2.04 A apart, as in a real bond
	for p, q in BONDS:
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

	with tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)

		# experimental: residues 1-3 and 20 unresolved (20 is not a cysteine), numbering offset by 9
		keep = [i for i in range(n) if i not in (0, 1, 2, 19)]
		write_pdb(tmp / "exp.pdb", SEQ, full, keep, first_number=10)

		# predicted: every residue, rotated and translated, 0.3 A noise on each atom
		rot, shift = random_rotation(rng), np.array([12.0, -7.0, 3.0])
		moved = [{k: v @ rot.T + shift + rng.normal(scale=0.3, size=3) for k, v in a.items()} for a in full]
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
		import prompting as PR
		atom_order = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "SG": 10}
		coords, info = PR.experimental_coords37(SEQ, tmp / "exp.pdb", None, atom_order)
		assert coords.shape == (n, 37, 3)
		assert info["absent"] == [1, 2, 3, 20] and info["resolved"] == n - 4, info
		assert np.isnan(coords[0]).all() and np.isnan(coords[19]).all(), "unresolved residues must stay NaN"
		assert np.isfinite(coords[4, atom_order["SG"]]).all(), "a resolved cysteine keeps its SG"
		prompt, given = PR.partial_prompt(coords, [5, 55], 0, atom_order)
		assert given == [5, 55], given
		for name in ("N", "CA", "C"):
			assert np.isfinite(prompt[4, atom_order[name]]).all() and np.isfinite(prompt[54, atom_order[name]]).all()
		assert np.isnan(prompt[4, atom_order["CB"]]).all(), "a prompt carries backbone only, never side-chain atoms"
		assert np.isnan(np.delete(prompt, [4, 54], axis=0)).all(), "every other residue must stay unknown"
		_, given_w = PR.partial_prompt(coords, [5, 55], 1, atom_order)
		assert given_w == [4, 5, 6, 54, 55, 56], given_w
		_, given_edge = PR.partial_prompt(coords, [1], 2, atom_order)
		assert given_edge == [1, 2, 3], given_edge
		d = PR.pair_distance(coords, 4, 54, atom_order, "SG")
		assert abs(d - 2.04) < 1e-6, d
		assert np.isnan(PR.pair_distance(coords, 0, 54, atom_order)), "an unresolved residue has no distance"
		print("[ok] coordinate helpers: alignment to the reference, backbone-only prompts, windows, pair distances")

	print("\nall logic tests passed (synthetic data: the arithmetic is right, nothing more)")


if __name__ == "__main__":
	main()
