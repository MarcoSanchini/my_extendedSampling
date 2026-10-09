"""
Can ESM3 MAINTAIN a disulfide bridge, and what does it take to IMPOSE one?

Written 2026-10-09 as the follow-up to the step sweep, which showed that nothing
about the HA1|HA2 bridge is formed at any step count and that the chain-break
makes no consistent difference. Those runs asked ESM3 to DISCOVER the bridge from
sequence. This script asks the two questions that come after:

  roundtrip  (can the representation hold a bond at all?)
      Take the EXPERIMENTAL coordinates, encode them to ESM3's discrete structure
      tokens, decode the tokens back to coordinates. Nothing is predicted: the
      model is given the whole answer. If the known disulfides come back at
      their experimental distances, the tokenizer and decoder can express them
      and any failure to form a bridge lies in prediction. If they come back
      loosened, the representation itself cannot hold the bond, and no amount of
      prompting or conditioning can fix that.

  prompt     (can we tell it where the bridge is?)
      Give the model ONLY the backbone N/CA/C of chosen residues (default: the two
      cysteines of one bond), leave everything else unknown, and generate the
      structure. This is ESM3's own way of imposing geometry: the coordinates
      feed the geometric attention (condition_on_coordinates_only is the
      project default), so the model sees where those two frames are. Questions:
      does the output keep the prompted pair where it was put, does the rest of
      the protein fold around them, and do the OTHER known bonds form as a side
      effect? --window adds neighbouring residues; --all-bonds prompts every
      known cysteine, an upper bound on what imposition can do here.

BOTH ARE ORACLE TESTS ON PURPOSE. They use experimental geometry of proteins
whose bonds are known (BPTI, crambin), so the right answer is known and a failure
is informative. For HA the geometry would have to come from an experimental HA
entry (fetch_pdb.py, run where the PDB is reachable) -- do that only after these
show imposition works at all.

Outputs go to bridge_test/rt/: <ref>_roundtrip.pdb and <ref>_prompt_<tag>.pdb,
each with a .plddt.tsv beside it. Score them with compare_to_pdb.py. This script
also prints the known-bond distances itself, as a quick read.

NOT TESTED where it was written: the cloud session had no torch and no weights.
The coordinate helpers (prompting.py) are tested; the model calls below follow
the SDK path the other scripts already use (encode -> custom_decode / predict_protein)
and are the thing to check first if this fails.

    python bridge_test/roundtrip_bridge.py --ref bpti --exp bridge_test/pdb/bpti_5PTI.pdb --mode roundtrip
    python bridge_test/roundtrip_bridge.py --ref bpti --exp bridge_test/pdb/bpti_5PTI.pdb --mode prompt --residues 14,38
    python bridge_test/roundtrip_bridge.py --ref bpti --exp bridge_test/pdb/bpti_5PTI.pdb --mode prompt --all-bonds
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
sys.path.append(os.path.abspath(os.path.join(HERE, "../../../customs")))
sys.path.insert(0, str(HERE.parent / "tests"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from custom_esm.models.esm3 import ESM3
from custom_esm.sdk.api import ESMProtein
from custom_esm.utils.residue_constants import atom_order

from utils.predictions import init_structure_config

import prompting as P
import references as R

CRITERION = 4.5  # A, the CB-CB bond criterion


def save(out, path):
	"""PDB plus a per-residue pLDDT sidecar (the PDB itself does not carry it)."""
	path = Path(path)
	path.parent.mkdir(exist_ok=True)
	out.to_pdb(str(path))
	if out.plddt is not None:
		pl = P.to_numpy(out.plddt)
		scale = 100.0 if pl.max() <= 1.0 else 1.0
		rows = [f"{k + 1}\t{a}\t{pl[k] * scale:.2f}" for k, a in enumerate(out.sequence)]
		Path(f"{path}.plddt.tsv").write_text("\n".join(rows) + "\n")
	print(f"  wrote {path}")


def bond_report(title, out, exp37, bonds, prompted=()):
	"""Known-bond CB-CB: experimental against this output, plus its pLDDT."""
	coords = P.to_numpy(out.coordinates)
	pl = P.to_numpy(out.plddt) if out.plddt is not None else None
	scale = 100.0 if pl is not None and pl.max() <= 1.0 else 1.0
	print(f"\n--- {title}: known disulfides (CB-CB, A) ---")
	print(f"  {'pair':<8s}{'experiment':>11s}{'this output':>13s}{'change':>9s}  within {CRITERION}?  pLDDT      prompted")
	within = 0
	for p, q in bonds:
		i, j = p - 1, q - 1
		e = P.pair_distance(exp37, i, j, atom_order)
		o = P.pair_distance(coords, i, j, atom_order)
		ok = np.isfinite(o) and o <= CRITERION
		within += int(ok)
		pls = f"{pl[i] * scale:.0f}/{pl[j] * scale:.0f}" if pl is not None else "-"
		flag = "yes" if (p in prompted and q in prompted) else ""
		print(f"  {f'{p}-{q}':<8s}{e:11.2f}{o:13.2f}{o - e:+9.2f}  {'yes' if ok else 'NO':<12s}{pls:<11s}{flag}")
	print(f"  {within}/{len(bonds)} inside the criterion")
	return within


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", required=True, choices=sorted(R.KNOWN_DISULFIDES))
	ap.add_argument("--exp", required=True, help="experimental PDB (from fetch_pdb.py)")
	ap.add_argument("--exp-chain", default=None)
	ap.add_argument("--mode", required=True, choices=["roundtrip", "prompt"])
	ap.add_argument("--residues", default=None, help="prompt residues, 1-based, e.g. 14,38")
	ap.add_argument("--all-bonds", action="store_true", help="prompt every cysteine in a known bond")
	ap.add_argument("--window", type=int, default=0, help="also prompt this many neighbours each side")
	ap.add_argument("--steps", type=int, default=None, help="decoding steps (default: the chain length)")
	ap.add_argument("--out-dir", default=str(HERE / "rt"))
	args = ap.parse_args()

	seq = R.REFERENCES[args.ref]
	bonds = list(R.KNOWN_DISULFIDES[args.ref])
	L = len(seq)
	steps = min(args.steps or L, L)

	exp37, info = P.experimental_coords37(seq, args.exp, args.exp_chain, atom_order)
	# C-beta on the model's convention, so experiment and output are compared like with like
	exp37i = P.with_inferred_cb(exp37, seq, atom_order)
	print(f"# {args.ref} (L={L}): experimental chain {info['chain']}, {info['resolved']}/{L} residues resolved")
	if info["mismatches"]:
		print(f"# WARNING: {len(info['mismatches'])} residue types differ from the reference: {info['mismatches']}")
	if info["absent"]:
		print(f"# unresolved reference positions: {info['absent']}")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = ESM3.from_pretrained("esm3-open").to(device)
	for prm in model.parameters():
		prm.requires_grad = False
	outdir = Path(args.out_dir)

	if args.mode == "roundtrip":
		# Everything the experiment gives, encoded to structure tokens and decoded
		# back. No generation, so no step count and no sampling.
		protein = ESMProtein(sequence=seq, coordinates=torch.tensor(exp37, dtype=torch.float32))
		tensor = model.encode(protein)
		out = model.custom_decode(tensor)
		save(out, outdir / f"{args.ref}_roundtrip.pdb")
		bond_report("round trip (experiment -> tokens -> coordinates)", out, exp37i, bonds)
		print("\n  Reading: bonds that come back near their experimental distance mean the")
		print("  representation can hold them. Bonds that loosen here mean it cannot, whatever")
		print("  the prediction does.")
		return

	# ---- prompt mode
	if args.all_bonds:
		residues = sorted({r for pair in bonds for r in pair})
	elif args.residues:
		residues = [int(x) for x in args.residues.split(",")]
	else:
		residues = list(bonds[0])
		print(f"# no --residues given; prompting the first known bond {bonds[0]}")
	bad = [r for r in residues if not 1 <= r <= L]
	if bad:
		raise SystemExit(f"--residues out of range for L={L}: {bad}")
	print(f"# prompting residues {residues} ({''.join(seq[r - 1] for r in residues)}), window {args.window}")

	prompt37, given = P.partial_prompt(exp37, residues, args.window, atom_order)
	print(f"# backbone N/CA/C given for {len(given)} residues: {given}")
	protein = ESMProtein(sequence=seq, coordinates=torch.tensor(prompt37, dtype=torch.float32))
	tensor = model.encode(protein)
	out = model.predict_protein(tensor, init_structure_config(num_steps=steps))
	tag = ("allbonds" if args.all_bonds else "-".join(str(r) for r in residues)) + f"_w{args.window}"
	save(out, outdir / f"{args.ref}_prompt_{tag}.pdb")

	ptm = float(out.ptm) if out.ptm is not None else float("nan")
	print(f"\n  pTM {ptm:.4f}, mean pLDDT "
		  f"{(P.to_numpy(out.plddt).mean() * (100 if P.to_numpy(out.plddt).max() <= 1 else 1)):.1f}")
	bond_report(f"prompted with {residues}", out, exp37i, bonds, prompted=set(residues))
	print("\n  Reading: the prompted pair is the direct test of whether imposed geometry survives")
	print("  decoding. The OTHER bonds are the side effect: if they form too, telling the model")
	print("  about one bridge helps the fold; if not, the prompt held its own pair and nothing more.")
	print("  Run compare_to_pdb.py on the output for RMSD, contacts and the full cysteine table.")


if __name__ == "__main__":
	main()
