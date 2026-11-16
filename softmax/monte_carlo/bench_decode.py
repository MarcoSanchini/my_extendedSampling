"""
How long ESM3 takes to predict a structure, and how accurate it is, as a
function of the number of generation steps -- the numbers needed to decide
whether a contact-map energy (Zambon et al 2024) can be evaluated at every
Monte Carlo move, or only on a subsample of a finished U_am chain.

Rows: "attention" is the U_am path the sampler uses (one ESM3 forward pass to
the last attention map, no structure), then one row per --steps value.
For every sequence and method it records

  P@L     top-L long-range (|i-j| >= 6) contact precision against the crystal:
          attention pairs ranked by attention weight, decodes by CA distance
  t_fwd_bwd (attention only) forward + backward, the core of a one-pass move

  t_gen   wall time of model.generate() (the iterative unmasking, one ESM3
          forward pass per step), CUDA-synchronised
  t_dec   wall time of the structure-token decoder (once per prediction)
  pLDDT   mean, 0-100
  RMSD    CA RMSD to the crystal over the residues paired by --pair-csv
  E_xtal  contact-map energy to the crystal (paired residues)
  E_ref   contact-map energy to the REFERENCE's prediction at the same number
          of steps: this is the Zambon-type energy E(seq) a sampler would use
  dE_128  |E_ref(n) - E_ref(128)|: the error an n-step energy makes relative
          to the most converged decode (needs 128 in --steps)

E = sum_ij |D_ij(a) - D_ij(b)| / sum_ij (D_ij(a) + D_ij(b)), |i-j| > 2, as in
prediction_diagnostics.py. The decoder writes backbone + inferred CB only, so
the map is CA within 8 A (--ca-cutoff), not the paper's 4 A heavy-atom map.

Sequences decoded: the reference, --random random sequences of the same
length, and --per-chain snapshots from each --chains run directory, spaced
evenly over the second half of the chain (the equilibrated part), each with its
U_am from data.dat. The Spearman correlation between E_ref and U_am over those
snapshots says how much a U_am-sampled ensemble already knows about the
contact-map energy, which is what a subsampling strategy relies on.

init_structure_config() decodes at temperature 0, so a prediction is a
deterministic function of the sequence: --repeats only re-times it (and
reports the largest E between repeats, which should be 0 up to GPU noise).

Timings are only meaningful on an otherwise idle GPU. With sampling chains
running on the same card, the accuracy columns are still exact but the times
are inflated; re-time with --no-samples once the card is free.

Usage, on the GPU machine from softmax/monte_carlo:
    nohup python -u bench_decode.py --ref pten_1d5r \\
        --exp structure_compare/1D5R.pdb \\
        --pair-csv structure_compare/pten_1d5r_vs_1D5R_per_residue.csv \\
        --steps 2 4 16 32 128 \\
        --chains pten1d5r_study/equil_ref/T2/chain0 pten1d5r_study/equil/T3/chain0 \\
                 pten1d5r_study/equil/T5/chain0 pten1d5r_study/equil_ref/T8/chain0 \\
                 pten1d5r_study/equil_ref/T16/chain0 \\
        --out decode_bench/pten > decode_bench_pten.log 2>&1 &
"""
import argparse
import csv
import math
import os
import sys
import time

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
for p in (MONTE_CARLO_DIR, CODE_DIR, CUSTOMS_DIR, os.path.join(CODE_DIR, "tests")):
	if p not in sys.path:
		sys.path.insert(0, p)

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


def contact_map(ca, cutoff):
	"""Boolean (L, L) CA contact map with |i-j| > 2; rows with no CA are False."""
	d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
	c = d < cutoff
	c &= np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :]) > 2
	c &= np.isfinite(ca).all(1)[:, None] & np.isfinite(ca).all(1)[None, :]
	return np.triu(c, 3)


def energy(a, b):
	s = a.sum() + b.sum()
	return float((a ^ b).sum() / s) if s else float("nan")


def kabsch_rmsd(P, Q):
	P, Q = P - P.mean(0), Q - Q.mean(0)
	U, S, Vt = np.linalg.svd(P.T @ Q)
	d = np.sign(np.linalg.det(U @ Vt))
	R = U @ np.diag([1, 1, d]) @ Vt
	return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def read_crystal(pdb, pair_csv, L):
	"""(L, 3) crystal CA coordinates in PREDICTED numbering, NaN where unpaired."""
	from prediction_diagnostics import read
	ca = {(k[0], k[1]): atoms["CA"] for k, _, atoms in read(pdb) if "CA" in atoms}
	out = np.full((L, 3), np.nan)
	with open(pair_csv) as f:
		for r in csv.DictReader(f):
			if r["exp_chain"] and r["exp_resseq"]:
				xyz = ca.get((r["exp_chain"], int(r["exp_resseq"])))
				if xyz is not None:
					out[int(r["pred_resseq"]) - 1] = xyz
	return out


def chain_snapshots(run, n, ref):
	"""n (label, sequence, U, Hd) evenly spaced over the second half of a chain."""
	from analyze_nsites import data_file, read_data, read_start, reconstruct
	d = read_data(data_file(run))
	last = int(d["move"][-1])
	want = (np.linspace(last // 2, last, n) / 1000).round().astype(int) * 1000   # data is logged every 500
	stride = math.gcd(*[int(w) for w in want]) or 1
	snaps, moves, bad = reconstruct(d, read_start(run), stride, ref)
	if bad:
		print(f"WARNING: {run}: {bad} inconsistencies while rebuilding sequences")
	U = dict(zip(d["move"].tolist(), d["U"].tolist()))
	Hd = dict(zip(d["move"].tolist(), d["Hd"].tolist()))
	out = []
	where = {int(m): i for i, m in enumerate(moves)}
	tag = os.path.relpath(run, MONTE_CARLO_DIR) if os.path.isabs(run) else run
	for w in want:
		s = snaps[where[int(w)]].tobytes().decode()
		out.append((f"{tag}@{w}", s, U.get(int(w), float("nan")), Hd.get(int(w), -1)))
	return out


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", default="pten_1d5r")
	ap.add_argument("--exp", default=None, help="crystal .pdb (needs --pair-csv)")
	ap.add_argument("--pair-csv", default=None, help="*_per_residue.csv from compare_structures.py")
	ap.add_argument("--steps", type=int, nargs="+", default=[2, 4, 16, 32, 128])
	ap.add_argument("--repeats", type=int, default=3, help="timed decodes of the reference per step count")
	ap.add_argument("--chains", nargs="*", default=[], help="run directories to take snapshots from")
	ap.add_argument("--per-chain", type=int, default=3)
	ap.add_argument("--random", type=int, default=2, help="random sequences of the reference's length")
	ap.add_argument("--no-samples", action="store_true", help="reference only: a quick re-timing")
	ap.add_argument("--ca-cutoff", type=float, default=8.0)
	ap.add_argument("--min-sep", type=int, default=6, help="|i-j| for long-range contact precision")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--out", required=True, help="output prefix; writes PREFIX_decodes.csv")
	args = ap.parse_args()

	from references import get_reference
	_, ref = get_reference(args.ref)
	L = len(ref)
	xtal = read_crystal(args.exp, args.pair_csv, L) if args.exp and args.pair_csv else None
	paired = np.isfinite(xtal).all(1) if xtal is not None else None
	map_x = contact_map(xtal, args.ca_cutoff) if xtal is not None else None

	seqs = [("reference", ref, 0.0, 0)]
	if not args.no_samples:
		rng = np.random.default_rng(args.seed)
		for k in range(args.random):
			s = "".join(rng.choice(list(AA), L))
			seqs.append((f"random{k}", s, float("nan"), sum(a != b for a, b in zip(s, ref))))
		for run in args.chains:
			seqs += chain_snapshots(run, args.per_chain, ref)
	print(f"{args.ref}: {L} residues; {len(seqs)} sequences x steps {args.steps}"
		+ (f"; crystal paired at {int(paired.sum())} residues" if xtal is not None else ""), flush=True)

	import torch
	from compute_structural_metrics import CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from custom_esm.sdk.api import ESMProteinTensor

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	model = sampler.model
	decoder = model.get_structure_decoder()
	sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

	def predict(seq, n):
		torch.manual_seed(args.seed)
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		ep.expand()
		cfg = init_structure_config(num_steps=n)
		sync(); t0 = time.perf_counter()
		with torch.no_grad():
			out = model.generate(input=ESMProteinTensor(sequence=ep.tokens), config=cfg)
			tok = out.structure if out.structure.dim() > 1 else out.structure.unsqueeze(0)
			sync(); t1 = time.perf_counter()
			dec = decoder.decode(tok)
			sync(); t2 = time.perf_counter()
		ca = dec["bb_pred"][0, 1:-1, CA_IDX].float().cpu().numpy()
		pl = dec["plddt"][0, 1:-1].float().cpu().numpy() if "plddt" in dec else np.full(L, np.nan)
		pl = pl * 100 if np.nanmax(pl) <= 1 else pl
		return ca, float(np.nanmean(pl)), t1 - t0, t2 - t1

	from utils.energies import compute_U_am

	def attention(seq, grad):
		"""U_am of seq and its symmetrised attention map, timed: a forward pass alone (the cost
		of scoring one proposal) or forward + backward (the cost of a gradient-informed move)."""
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		sync(); t0 = time.perf_counter()
		if grad:
			probs = ep.get_probs().clone().requires_grad_(True)   # gradient w.r.t. the input, as in a move
			am = model.predict_attention(sequence_probs=probs)
			U = compute_U_am(am, ref_am)
			U.backward()
		else:
			with torch.no_grad():
				am = model.predict_attention(sequence_probs=ep.get_probs())
				U = compute_U_am(am, ref_am)
		sync()
		return am.detach().float().cpu().numpy(), float(U), time.perf_counter() - t0

	with torch.no_grad():
		ref_am = model.predict_attention(sequence_probs=ExtendedProtein(
			sequence=ref, requires_grad=False, device=device).get_probs())
	predict(ref, min(args.steps))                    # warm-up: kernels, allocator, lazy loads
	attention(ref, True)
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats()

	long_range = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= args.min_sep
	def precision(score):
		"""Share of the top-L scoring pairs (|i-j| >= --min-sep, both residues in the crystal)
		that are crystal contacts: the usual top-L long-range contact precision."""
		if xtal is None:
			return float("nan")
		ok = np.triu(long_range & paired[:, None] & paired[None, :])
		iu = np.flatnonzero(ok)
		top = iu[np.argsort(-score.ravel()[iu])[:L]]
		xt = map_x | map_x.T
		return float(xt.ravel()[top].mean())

	methods = ["attention"] + [f"{n} steps" for n in args.steps]
	ref_map, rows = {}, []
	for label, seq, U, Hd in seqs:
		reps = args.repeats if label == "reference" else 1
		for meth in methods:
			row = {"label": label, "method": meth, "U_am": U, "Hd": Hd, "t": np.nan, "t_gen": np.nan,
				"t_dec": np.nan, "t_fwd_bwd": np.nan, "U_calc": np.nan, "plddt": np.nan, "precision": np.nan,
				"rmsd_xtal": np.nan, "E_xtal": np.nan, "E_ref": np.nan, "rep_E_max": np.nan}
			if meth == "attention":
				res = [attention(seq, False) for _ in range(reps)]
				row["t"] = row["t_gen"] = float(np.median([r[2] for r in res]))
				row["t_fwd_bwd"] = float(np.median([attention(seq, True)[2] for _ in range(reps)]))
				row["U_calc"] = res[0][1]
				row["precision"] = precision(res[0][0])
			else:
				n = int(meth.split()[0])
				res = [predict(seq, n) for _ in range(reps)]
				ca, pl = res[0][0], res[0][1]
				cmap = contact_map(ca, args.ca_cutoff)
				if label == "reference":
					ref_map[n] = cmap
				row.update({
					"t_gen": float(np.median([r[2] for r in res])), "t_dec": float(np.median([r[3] for r in res])),
					"plddt": pl, "E_ref": energy(cmap, ref_map[n]),
					"rep_E_max": max(energy(cmap, contact_map(r[0], args.ca_cutoff)) for r in res),
				})
				row["t"] = row["t_gen"] + row["t_dec"]
				if xtal is not None:
					d = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
					row["precision"] = precision(-d)
					row["rmsd_xtal"] = kabsch_rmsd(ca[paired], xtal[paired])
					pm = paired[:, None] & paired[None, :]
					row["E_xtal"] = energy(cmap & pm, map_x)
			rows.append(row)
			print(f"{label:42s} {meth:10s} t {row['t']:6.3f}s  P@L {row['precision']:.3f}  pLDDT {row['plddt']:5.1f}  "
				f"RMSD {row['rmsd_xtal']:6.2f}  E_xtal {row['E_xtal']:.3f}  E_ref {row['E_ref']:.3f}  "
				f"U_am {U:8.1f} (recomputed {row['U_calc']:8.1f})  Hd {Hd}", flush=True)

	top = f"{max(args.steps)} steps"
	E_top = {r["label"]: r["E_ref"] for r in rows if r["method"] == top}
	for r in rows:
		r["dE_top"] = abs(r["E_ref"] - E_top[r["label"]])
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out + "_decodes.csv", "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=list(rows[0]))
		w.writeheader()
		w.writerows(rows)

	def spearman(a, b):
		r = lambda v: np.argsort(np.argsort(v))
		return float(np.corrcoef(r(a), r(b))[0, 1])
	print(f"\n# summary ({len(seqs)} sequences; times are medians over all of them, in seconds)")
	print("# t = scoring one sequence (attention: forward only; decode: generate + decoder); t_fwd_bwd = one-pass move")
	print("# P@L = top-L long-range contact precision vs the crystal; samp_* = medians over chain snapshots")
	print(f"# dE = |E_ref(n) - E_ref({top})| over snapshots; rho = Spearman(E_ref or recomputed U_am, U_am from data.dat)")
	print("method\tt\tt_gen\tt_dec\tt_fwd_bwd\tref_P@L\tref_pLDDT\tref_RMSD\tref_E_xtal\tsamp_P@L\tsamp_pLDDT\tsamp_E_ref\tdE_med\tdE_max\trho\trep_E_max")
	for meth in methods:
		R = [r for r in rows if r["method"] == meth]
		ref_r = next(r for r in R if r["label"] == "reference")
		S = [r for r in R if r["label"] != "reference" and not r["label"].startswith("random")]
		med = lambda key, rr=R: float(np.nanmedian([r[key] for r in rr])) if rr else np.nan
		key = "U_calc" if meth == "attention" else "E_ref"
		rho = spearman([r[key] for r in S], [r["U_am"] for r in S]) if len(S) > 2 else np.nan
		dmax = float(np.nanmax([r["dE_top"] for r in S])) if S and meth != "attention" else np.nan
		print(f"{meth}\t{med('t'):.3f}\t{med('t_gen'):.3f}\t{med('t_dec'):.3f}\t{med('t_fwd_bwd'):.3f}\t"
			f"{ref_r['precision']:.3f}\t{ref_r['plddt']:.1f}\t{ref_r['rmsd_xtal']:.2f}\t{ref_r['E_xtal']:.3f}\t"
			f"{med('precision', S):.3f}\t{med('plddt', S):.1f}\t{med('E_ref', S):.3f}\t{med('dE_top', S):.3f}\t"
			f"{dmax:.3f}\t{rho:.2f}\t{np.nanmax([r['rep_E_max'] for r in R]) if meth != 'attention' else np.nan:.3f}")
	rnd = [r for r in rows if r["label"].startswith("random") and r["method"] == top]
	if rnd:
		print(f"# random sequences at {top}: E_ref {np.mean([r['E_ref'] for r in rnd]):.3f}, "
			f"pLDDT {np.mean([r['plddt'] for r in rnd]):.1f}, P@L {np.mean([r['precision'] for r in rnd]):.3f} (the floor)")
	if device.type == "cuda":
		print(f"# peak GPU memory allocated by this process: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
	print(f"Saved -> {args.out}_decodes.csv")


if __name__ == "__main__":
	main()
