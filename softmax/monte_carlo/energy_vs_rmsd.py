"""
Does a low energy mean a sequence folds like the crystal? Equilibrated frames of the PTEN
ladder, scored by every energy we have, against the RMSD of their decoded structure to 1D5R.

Strategy B of the 2026-11-20 thread: score a DENSE, evenly spaced set of frames once on the
GPU, then let the CPU analysis pick subsets from it (decorrelated, energy-stratified), so no
selection ever needs the GPU again.

Subcommands
  frames   CPU. Replays data.dat from start.txt and writes --per-chain sequences evenly spaced
           over the last (1 - burn-in-frac) of every chain of the ladder, plus each chain's
           integrated autocorrelation time of U in that window (tau_U, in moves).
           --burn-in-frac defaults to 0.8 because 0.5 does NOT clear t_eq on this ladder.
  score    GPU. One row per frame:
             U_am        from data.dat (the sampler's own number; lambda_structure_ce = 0)
             U_am_calc   recomputed for the first --recheck-am frames, must equal U_am
             U_ce_ref    cross-entropy of the structure head against the REFERENCE's own argmax
                         structure tokens, exactly the sampler's U_structure_ce (fp32)
             U_ce_xtal   the same against the 1D5R crystal's VQ structure tokens, over the
                         residues the crystal resolves (unresolved ones are ignored)
             E_ref       Zambon contact-map energy (CA, 8 A) against the reference's decode at
                         the same number of steps
             E_xtal      the same against the crystal's CA map, on residues the crystal resolves
             Es_ref/_xtal  the soft-contact versions (sigmoid of width --soft-width), free
             rmsd_xtal   CA RMSD of the decode to 1D5R after Kabsch, paired residues  <- target
             tm_xtal     TM-score at that same Kabsch superposition (a lower bound on the real
                         TM-score, which optimises its own superposition)
             rmsd_ref    CA RMSD to the reference's own decode
             plddt       mean pLDDT of the decode
             *_n         rmsd_xtal / E_ref / E_xtal re-decoded at --noise-steps, for an evenly
                         spread subset of --noise-frames frames: the decode-noise floor
           Identical sequences (frequent at T=1, where 1% of moves are accepted) are decoded
           once. Resumable: frames already in --out are skipped.
  analyze  CPU. Spearman of every energy with rmsd_xtal (and tm_xtal):
             B   all frames, pooled and within each T, with a block bootstrap whose block is
                 tau_U long (neighbouring frames are correlated, so plain n would lie)
             A   one frame per tau_U per chain: independent frames, honest but small
             C   frames stratified evenly over log U across the pooled ladder
           and, for pooled numbers, the partial Spearman given Hd_to_ref: across the ladder
           everything grows with the number of mutations, so the raw pooled rho is mostly
           "more mutations, worse fold". The partial is the number that says the energy knows
           something Hd does not.

CAUTION, read before quoting: E_xtal and rmsd_xtal are two readings of the SAME decode against
the SAME crystal, so their correlation is partly built in. The tests that mean something are
the energies that never see the crystal (U_am, U_ce_ref, E_ref); the crystal-side ones are a
ceiling to compare them with.

Usage, on astaroth from softmax/monte_carlo:
    python -u energy_vs_rmsd.py frames --out energy_rmsd/frames.tsv
    nohup python -u energy_vs_rmsd.py score --ref pten_1d5r --frames energy_rmsd/frames.tsv \\
        --out energy_rmsd/scores.csv > energy_rmsd_score.log 2>&1 &
    python -u energy_vs_rmsd.py analyze --scores energy_rmsd/scores.csv --out-dir energy_rmsd
"""
import argparse
import csv
import math
import os
import re
import sys
import time

import numpy as np

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
for p in (MONTE_CARLO_DIR, CODE_DIR, CUSTOMS_DIR, os.path.join(CODE_DIR, "tests")):
	if p not in sys.path:
		sys.path.insert(0, p)

STUDY = "pten1d5r_study"
# The ladder whose t_eq has been measured, exactly as compare_ladder.sh names it: equil_ref
# holds T = 1, 1.5, 2, 4, 8, 16; T = 3, 5, 6 only ever ran in equil. The hot 30000-move chains
# of equil (T32..T316) have no measured t_eq and are left out on purpose.
LADDER = [(T, f"{STUDY}/equil_ref/T{T}") for T in ("1", "1.5", "2", "4", "8", "16")] + \
	[(T, f"{STUDY}/equil/T{T}") for T in ("3", "5", "6")] + \
	[(T, f"{STUDY}/warm/T{T}") for T in ("2.5", "7")]
# The warm/ chains are HOT-STARTED from an equilibrated neighbour (T2.5 from equil_ref/T2,
# T7 from equil/T6) and ran one at a time on urano, so there is only a chain0 and their t_eq
# is not the ladder's relaxation time from the reference. Taking the last 20% is safe either
# way, and `frames` prints each chain's own tau_U next to it.

FRAME_COLS = ["frame_id", "label", "T", "chain", "move", "U", "Hd", "tau_U", "spacing", "seq"]
SCORE_COLS = ["frame_id", "label", "T", "chain", "move", "Hd", "tau_U", "spacing", "U_am", "U_am_calc",
	"U_ce_ref", "U_ce_xtal", "E_ref", "E_xtal", "Es_ref", "Es_xtal", "rmsd_xtal", "tm_xtal", "rmsd_ref",
	"plddt", "rmsd_xtal_n", "E_ref_n", "E_xtal_n", "t"]
ENERGIES = ["U_am", "U_ce_ref", "E_ref", "Es_ref", "U_ce_xtal", "E_xtal", "Es_xtal"]
SEES_CRYSTAL = {"U_ce_xtal", "E_xtal", "Es_xtal"}


# ----------------------------------------------------------------------------- frames (CPU)

def default_runs():
	runs = []
	for T, base in LADDER:
		for c in (0, 1):
			path = os.path.join(MONTE_CARLO_DIR, base, f"chain{c}")
			if c and not os.path.isdir(path):
				continue                      # the warm/ chains have no second chain
			runs.append((f"T{T}" + ("" if c == 0 else ".c1"), path))
	return runs


def parse_runs(items):
	"""LABEL=PATH pairs, as pten_k_sites.py compare takes them."""
	out = []
	for it in items:
		if "=" not in it:
			raise SystemExit(f"--runs wants LABEL=PATH, got {it!r}")
		lab, path = it.split("=", 1)
		out.append((lab, path if os.path.isabs(path) else os.path.join(MONTE_CARLO_DIR, path)))
	return out


def read_pars(run):
	pars = {}
	path = os.path.join(run, "pars.txt")
	if os.path.isfile(path):
		for line in open(path):
			parts = line.split()
			if len(parts) == 2 and not line.startswith("#"):
				pars[parts[0]] = parts[1]
	return pars


def tau_int(x):
	"""Integrated autocorrelation time (in samples) with Sokal's window: stop at the first
	lag k >= 5 tau. Same estimator used for the numbers quoted in the 2026-11-20 plan."""
	x = np.asarray(x, float) - np.mean(x)
	n = len(x)
	if n < 10 or not np.any(x):
		return float("nan")
	f = np.fft.rfft(x, 2 * n)
	ac = np.fft.irfft(f * np.conj(f))[:n]
	ac /= ac[0]
	tau = 1.0
	for k in range(1, n):
		tau += 2 * ac[k]
		if k >= 5 * tau:
			break
	return float(tau)


def replay(d, start, want):
	"""Sequences at the moves in `want` (sorted), replaying accepted moves from `start`, and
	the number of rows where data.dat's cur_aa disagrees with the replayed state (must be 0)."""
	state = np.frombuffer(start.encode(), dtype=np.uint8).copy()
	want = list(want)
	out, bad, k = {}, 0, 0
	for i, m in enumerate(d["move"]):
		if m > 0:
			sites = [int(s) for s in d["site"][i].split(",")]
			if any(state[s] != ord(c) for s, c in zip(sites, d["cur"][i].split(","))):
				bad += 1
			if d["acc"][i]:
				for s, a in zip(sites, d["prop"][i].split(",")):
					state[s] = ord(a)
		while k < len(want) and want[k] == m:
			out[m] = state.tobytes().decode()
			k += 1
		if k == len(want):
			break
	if k != len(want):
		raise SystemExit(f"moves {want[k:k + 3]}... are not in data.dat")
	return out, bad


def cmd_frames(args):
	from analyze_nsites import data_file, read_data, read_start
	from references import get_reference
	_, ref = get_reference(args.ref)
	if args.burn_in_frac < 0.8:
		print(f"WARNING: --burn-in-frac {args.burn_in_frac} < 0.8. On this ladder 0.5 does not clear t_eq at "
			"seven of nine temperatures (docs/pten_k_sites_explained.md 5.2).", flush=True)
	runs = parse_runs(args.runs) if args.runs else default_runs()
	os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
	n_rows = 0
	with open(args.out, "w", newline="") as f:
		w = csv.writer(f, delimiter="\t")
		w.writerow(FRAME_COLS)
		for label, run in runs:
			pars = read_pars(run)
			if float(pars.get("lambda_structure_ce", 0)) != 0:
				print(f"WARNING: {run} sampled with lambda_structure_ce = {pars['lambda_structure_ce']}: "
					"its U is not U_am alone", flush=True)
			T = float(pars["T"]) if "T" in pars else float(re.search(r"T([0-9.]+)", label).group(1))
			d = read_data(data_file(run))
			moves = d["move"]
			last = int(moves[-1])
			first = int(math.ceil(args.burn_in_frac * last))
			win = moves >= first
			# data.dat holds every move, so tau is in moves.
			tau = tau_int(d["U"][win])
			want = np.unique(np.linspace(first, last, args.per_chain).round().astype(int))
			have = set(moves[win].tolist())
			want = [int(m) for m in want if int(m) in have]     # data.dat logs every move; this only guards gaps
			seqs, bad = replay(d, read_start(run), want)
			Uof = dict(zip(moves.tolist(), d["U"].tolist()))
			Hdof = dict(zip(moves.tolist(), d["Hd"].tolist()))
			spacing = (last - first) / max(len(want) - 1, 1)
			nhd = 0
			for m in want:
				s = seqs[m]
				if len(s) != len(ref):
					raise SystemExit(f"{run}: sequence length {len(s)} but --ref {args.ref} is {len(ref)}")
				if sum(a != b for a, b in zip(s, ref)) != Hdof[m]:
					nhd += 1
				w.writerow([f"{label}@{m}", label, f"{T:g}", run.rstrip("/").split("/")[-1], m, f"{Uof[m]:.4f}",
					Hdof[m], f"{tau:.0f}", f"{spacing:.1f}", s])
				n_rows += 1
			n_uniq = len(set(seqs.values()))
			print(f"{label:8s} T={T:<5g} moves {first}-{last}  tau_U {tau:6.0f}  N_eff {(last - first) / tau:5.1f}  "
				f"{len(want)} frames every {spacing:.0f} moves ({spacing / tau:.2f} tau), {n_uniq} distinct  "
				f"Hd {min(Hdof[m] for m in want)}-{max(Hdof[m] for m in want)}"
				+ (f"  WARNING {bad} replay mismatches" if bad else "")
				+ (f"  WARNING {nhd} Hd mismatches vs --ref" if nhd else ""), flush=True)
	print(f"Saved {n_rows} frames -> {args.out}")


# ----------------------------------------------------------------------------- geometry

def kabsch(P, Q):
	"""Rotation-and-translation superposing P onto Q; returns the moved P."""
	pc, qc = P.mean(0), Q.mean(0)
	U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
	dd = np.sign(np.linalg.det(U @ Vt))
	R = U @ np.diag([1, 1, dd]) @ Vt
	return (P - pc) @ R + qc


def rmsd_tm(P, Q):
	"""CA RMSD after Kabsch, and the TM-score at that superposition (normalised by len(Q))."""
	Pm = kabsch(P, Q)
	d = np.linalg.norm(Pm - Q, axis=1)
	n = len(Q)
	d0 = max(1.24 * (n - 15) ** (1 / 3) - 1.8, 0.5)
	return float(np.sqrt((d ** 2).mean())), float((1 / (1 + (d / d0) ** 2)).mean())


def read_crystal_backbone(pdb, pair_csv, L):
	"""(L, 3, 3) N/CA/C crystal coordinates in PREDICTED numbering, NaN where the residue is
	unpaired or one of the three atoms is missing."""
	from prediction_diagnostics import read
	atoms = {(k[0], k[1]): a for k, _, a in read(pdb)}
	out = np.full((L, 3, 3), np.nan)
	with open(pair_csv) as f:
		for r in csv.DictReader(f):
			if r["exp_chain"] and r["exp_resseq"]:
				a = atoms.get((r["exp_chain"], int(r["exp_resseq"])))
				if a is not None and all(x in a for x in ("N", "CA", "C")):
					out[int(r["pred_resseq"]) - 1] = [a["N"], a["CA"], a["C"]]
	return out


# ----------------------------------------------------------------------------- score (GPU)

def cmd_score(args):
	import pandas as pd
	from bench_decode import contact_map, energy
	from scan_mutants_E import soft_contact, soft_energy
	from references import get_reference
	_, ref = get_reference(args.ref)
	L = len(ref)
	frames = pd.read_csv(args.frames, sep="\t", dtype={"T": str})
	if (frames.seq.str.len() != L).any():
		raise SystemExit(f"{args.frames} holds sequences that are not {L} long: wrong --ref?")

	bb_x = read_crystal_backbone(args.exp, args.pair_csv, L)
	paired = np.isfinite(bb_x).all((1, 2))
	ca_x = bb_x[:, 1]
	pm = paired[:, None] & paired[None, :]
	map_x = contact_map(np.where(paired[:, None], ca_x, np.nan), args.ca_cutoff)
	soft_x = soft_contact(np.where(paired[:, None], ca_x, np.nan), args.ca_cutoff, args.soft_width)
	print(f"# {args.ref}, L={L}; crystal {os.path.basename(args.exp)} resolves N/CA/C at {paired.sum()} of {L} "
		f"residues, {int(map_x.sum())} CA contacts at {args.ca_cutoff} A", flush=True)

	import torch
	import torch.nn.functional as F
	from compute_structural_metrics import CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from utils.energies import compute_U_am, compute_U_structure_ce
	from custom_esm.sdk.api import ESMProteinTensor
	from custom_esm.utils.structure.normalize_coordinates import normalize_coordinates

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if device.type != "cuda":
		print("WARNING: no GPU; this will be very slow", flush=True)
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	model = sampler.model
	decoder, encoder = model.get_structure_decoder(), model.get_structure_encoder()
	cfgs = {n: init_structure_config(num_steps=n) for n in {args.num_steps, args.noise_steps}}

	def probs(seq):
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		ep.expand()
		return ep.get_probs()

	def decode(seq, n):
		torch.manual_seed(args.seed)
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		ep.expand()
		with torch.no_grad():
			out = model.generate(input=ESMProteinTensor(sequence=ep.tokens), config=cfgs[n])
			tok = out.structure if out.structure.dim() > 1 else out.structure.unsqueeze(0)
			dec = decoder.decode(tok)
		bb = dec["bb_pred"][0, 1:-1, :3].float().cpu().numpy()
		pl = dec["plddt"][0, 1:-1].float().cpu().numpy() if "plddt" in dec else np.full(L, np.nan)
		pl = pl * 100 if np.nanmax(pl) <= 1 else pl
		return bb, float(np.nanmean(pl)), tok[0, 1:-1].detach().cpu()

	def vq_tokens(bb):
		"""Structure tokens of an (L, 3, 3) N/CA/C backbone, NaN rows allowed (they get
		affine_mask False inside the encoder and are ignored by the caller)."""
		c = torch.tensor(bb, dtype=torch.float32, device=device).unsqueeze(0)
		c = normalize_coordinates(c)
		with torch.no_grad():
			_, tok = encoder.encode(c, residue_index=torch.arange(1, L + 1, device=device).unsqueeze(0))
		return tok[0]

	def structure_logits(seq):
		with torch.no_grad():
			return model.predict_structure_logits(sequence_probs=probs(seq))[1:-1].float()

	# targets of the two cross-entropies
	ref_logits = structure_logits(ref)
	tok_ref = ref_logits.argmax(-1)                                  # exactly rate_sampler._setup
	tok_x = vq_tokens(bb_x).to(device)
	mask_x = torch.tensor(paired, device=device)
	tgt_x = torch.where(mask_x, tok_x, torch.full_like(tok_x, -100))

	def U_ce(logits):
		u_ref = compute_U_structure_ce(logits.unsqueeze(0), tok_ref.unsqueeze(0)).item()
		u_x = F.cross_entropy(logits, tgt_x, ignore_index=-100).item()
		return u_ref, u_x

	with torch.no_grad():
		ref_am = model.predict_attention(sequence_probs=probs(ref))

	def U_am_of(seq):
		with torch.no_grad():
			return compute_U_am(model.predict_attention(sequence_probs=probs(seq)), ref_am).item()

	# The reference: its decode is what E_ref is measured against, and its row is the baseline.
	ref_dec = {}
	for n in sorted(cfgs):
		bb, pl, tok = decode(ref, n)
		ref_dec[n] = (bb[:, 1], contact_map(bb[:, 1], args.ca_cutoff), soft_contact(bb[:, 1], args.ca_cutoff,
			args.soft_width), pl, bb, tok)
	bb2 = decode(ref, args.num_steps)[0]
	u_ref0 = U_ce(ref_logits)
	print(f"# reference: U_ce_ref {u_ref0[0]:.6f} (the sampler's U_structure_ce(reference), 2.397 on 2026-11-18), "
		f"U_ce_xtal {u_ref0[1]:.6f}", flush=True)
	same = (tok_x[mask_x] == tok_ref[mask_x]).float().mean().item()
	print(f"# crystal VQ tokens equal the reference's argmax tokens at {same:.1%} of resolved residues", flush=True)
	bb_r, tok_r = ref_dec[args.num_steps][4], ref_dec[args.num_steps][5]
	rt = (vq_tokens(bb_r).cpu() == tok_r).float().mean().item()
	print(f"# encoder round trip: re-encoding the reference's own {args.num_steps}-step decode recovers {rt:.1%} "
		"of the tokens it was decoded from (a sanity check of the encoder call: far above 1/4096 = 0.02%)",
		flush=True)
	again = int((contact_map(bb2[:, 1], args.ca_cutoff) ^ ref_dec[args.num_steps][1]).sum())
	print(f"# same sequence decoded twice differs in {again} contacts (must be 0)", flush=True)

	def score(seq, n):
		ca_r, map_r, soft_r, _, _, _ = ref_dec[n]
		bb, pl, _ = decode(seq, n)
		ca = bb[:, 1]
		cmap = contact_map(ca, args.ca_cutoff)
		r_x, tm_x = rmsd_tm(ca[paired], ca_x[paired])
		return {"E_ref": energy(cmap, map_r), "E_xtal": energy(cmap & pm, map_x),
			"Es_ref": soft_energy(soft_contact(ca, args.ca_cutoff, args.soft_width), soft_r),
			"Es_xtal": soft_energy(np.where(pm, soft_contact(ca, args.ca_cutoff, args.soft_width), 0), soft_x),
			"rmsd_xtal": r_x, "tm_xtal": tm_x, "rmsd_ref": rmsd_tm(ca, ca_r)[0], "plddt": pl}

	done = set()
	if os.path.exists(args.out):
		with open(args.out) as f:
			head = f.readline().rstrip("\n").split(",")
			if head != SCORE_COLS:
				raise SystemExit(f"{args.out} has another header; move it aside or pass another --out")
			done = {line.split(",", 1)[0] for line in f}
	else:
		os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
		with open(args.out, "w") as f:
			f.write(",".join(SCORE_COLS) + "\n")

	n_noise = min(args.noise_frames, len(frames))
	noise_ids = set(frames.frame_id.iloc[np.linspace(0, len(frames) - 1, n_noise).round().astype(int)]) \
		if n_noise else set()
	todo = frames[~frames.frame_id.isin(done)]
	print(f"# {len(frames)} frames, {len(done)} already in {args.out}, {len(todo)} to do; "
		f"{frames.seq.nunique()} distinct sequences; {n_noise} re-decoded at {args.noise_steps} steps", flush=True)
	fmt = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else (
		f"{v:.6g}" if isinstance(v, float) else str(v))

	with open(args.out, "a") as out:
		if "reference" not in done:
			s = score(ref, args.num_steps)
			row = {"frame_id": "reference", "label": "reference", "T": 0, "chain": "", "move": 0, "Hd": 0,
				"tau_U": "", "spacing": "", "U_am": 0.0, "U_am_calc": U_am_of(ref), "U_ce_ref": u_ref0[0],
				"U_ce_xtal": u_ref0[1], **s, "t": 0.0}
			out.write(",".join(fmt(row.get(c)) for c in SCORE_COLS) + "\n")
			print(f"# reference decode: RMSD to 1D5R {s['rmsd_xtal']:.2f} A, TM {s['tm_xtal']:.3f}, "
				f"E_xtal {s['E_xtal']:.3f}, pLDDT {s['plddt']:.1f}", flush=True)
		cache, t0, k_am = {}, time.perf_counter(), len(done)
		for k, fr in enumerate(todo.itertuples(index=False), 1):
			t1 = time.perf_counter()
			if fr.seq not in cache:
				s = score(fr.seq, args.num_steps)
				s["U_ce_ref"], s["U_ce_xtal"] = U_ce(structure_logits(fr.seq))
				cache[fr.seq] = s
			s = dict(cache[fr.seq])
			if fr.frame_id in noise_ids:
				sn = score(fr.seq, args.noise_steps)
				s.update(rmsd_xtal_n=sn["rmsd_xtal"], E_ref_n=sn["E_ref"], E_xtal_n=sn["E_xtal"])
			if k_am < args.recheck_am:
				s["U_am_calc"] = U_am_of(fr.seq)
				k_am += 1
				if abs(s["U_am_calc"] - fr.U) > 1e-3 * max(1.0, abs(fr.U)):
					print(f"WARNING: {fr.frame_id}: recomputed U_am {s['U_am_calc']:.3f} vs data.dat {fr.U:.3f}",
						flush=True)
			row = {"frame_id": fr.frame_id, "label": fr.label, "T": fr.T, "chain": fr.chain, "move": fr.move,
				"Hd": fr.Hd, "tau_U": fr.tau_U, "spacing": fr.spacing, "U_am": float(fr.U), **s,
				"t": time.perf_counter() - t1}
			out.write(",".join(fmt(row.get(c)) for c in SCORE_COLS) + "\n")
			if k % 50 == 0 or k == len(todo):
				out.flush()
				el = time.perf_counter() - t0
				print(f"# {k}/{len(todo)}  {el / 60:.1f} min, {el / k:.3f} s/frame, {(len(todo) - k) * el / k / 60:.1f} "
					f"min left; {fr.frame_id}: RMSD {s['rmsd_xtal']:.2f} TM {s['tm_xtal']:.3f} U_ce_ref "
					f"{s['U_ce_ref']:.4f} E_ref {s['E_ref']:.3f}", flush=True)
	if device.type == "cuda":
		print(f"# peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
	print(f"Saved -> {args.out}. Next: python -u energy_vs_rmsd.py analyze --scores {args.out}")


# ----------------------------------------------------------------------------- analyze (CPU)

def spearman(x, y):
	x, y = np.asarray(x, float), np.asarray(y, float)
	ok = np.isfinite(x) & np.isfinite(y)
	if ok.sum() < 5:
		return np.nan
	rx, ry = rank(x[ok]), rank(y[ok])
	if rx.std() == 0 or ry.std() == 0:
		return np.nan
	return float(np.corrcoef(rx, ry)[0, 1])


def rank(v):
	"""Average ranks (ties share their mean rank), as scipy.stats.rankdata."""
	order = np.argsort(v, kind="mergesort")
	r = np.empty(len(v))
	r[order] = np.arange(len(v), dtype=float)
	sv = v[order]
	b = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1], True])
	for i, j in zip(b[:-1], b[1:]):
		if j - i > 1:
			r[order[i:j]] = (i + j - 1) / 2
	return r


def partial_spearman(x, y, z):
	"""Spearman of x and y after the rank of z has been regressed out of both ranks (the same
	definition as pten_k_sites.py)."""
	x, y, z = (np.asarray(v, float) for v in (x, y, z))
	ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
	if ok.sum() < 5:
		return np.nan
	rx, ry, rz = rank(x[ok]), rank(y[ok]), rank(z[ok])
	ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
	ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
	if ex.std() == 0 or ey.std() == 0:
		return np.nan
	return float(np.corrcoef(ex, ey)[0, 1])


def block_resample(df, rng):
	"""Moving-block bootstrap within every chain, blocks tau_U long: the frames are spaced
	closer than tau, so resampling them one at a time would treat correlated frames as
	independent and shrink the interval."""
	idx = []
	for _, g in df.groupby(["label"]):
		ii = g.sort_values("move").index.to_numpy()
		n = len(ii)
		b = int(max(1, min(n, round(float(g.tau_U.iloc[0]) / max(float(g.spacing.iloc[0]), 1)))))
		starts = rng.integers(0, n - b + 1, size=int(math.ceil(n / b)))
		idx.append(np.concatenate([ii[s:s + b] for s in starts])[:n])
	return df.loc[np.concatenate(idx)]


def with_ci(df, fn, n_boot, rng):
	est = fn(df)
	if n_boot <= 0 or not np.isfinite(est):
		return est, np.nan, np.nan
	bs = np.array([fn(block_resample(df, rng)) for _ in range(n_boot)])
	bs = bs[np.isfinite(bs)]
	if len(bs) < 10:
		return est, np.nan, np.nan
	return est, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def select_decorrelated(df):
	"""Selection A: in every chain, frames at least tau_U apart."""
	keep = []
	for _, g in df.groupby("label"):
		g = g.sort_values("move")
		last = -np.inf
		tau = float(g.tau_U.iloc[0])
		for i, m in zip(g.index, g.move):
			if m - last >= tau:
				keep.append(i)
				last = m
	return df.loc[keep]


def select_stratified(df, n_bins, per_bin, rng):
	"""Selection C: equal-width bins of log10 U_am over the pooled ladder, up to per_bin frames
	from each, drawn without replacement among frames at least tau_U apart within a chain."""
	d = select_decorrelated(df) if per_bin else df
	lu = np.log10(np.maximum(d.U_am.astype(float), 1e-9))
	edges = np.linspace(lu.min(), lu.max() + 1e-9, n_bins + 1)
	b = np.digitize(lu, edges) - 1
	keep = []
	for k in range(n_bins):
		ii = d.index[b == k].to_numpy()
		if len(ii):
			keep.extend(rng.choice(ii, size=min(per_bin, len(ii)), replace=False))
	return d.loc[keep]


def match_window(d, mode, n):
	"""Trim every chain to a comparable window before correlating.

	WHY this is needed for the within-T column, and why it is NOT the attenuation that
	faked the T=6 entropy peak (docs/pten_k_sites_explained.md 5.2, --match-accepted).
	There, the per-site entropy was ESTIMATED from the accepted moves in the window, so a
	short window compressed the estimate itself. Here each frame's energy and RMSD are
	exact for that sequence, so nothing is mis-estimated. What a short window does is
	RANGE RESTRICTION: the chain explores less of its equilibrium spread of U, and a
	Spearman over a narrower range is pulled toward zero. Both shrink rho at the hot end,
	for different reasons, and both make a raw temperature trend untrustworthy.

	mode "moves": the last n moves of every chain -- equal clock time.
	mode "tau":   the last n * tau_U moves -- equal INDEPENDENT sampling, which is what
	              range restriction actually scales with, since a chain with a long tau
	              wanders less in the same number of moves.
	n None means the largest value every chain can supply, so nothing is thrown away
	beyond what matching demands.
	"""
	if mode == "none":
		return d, None
	span = d.groupby("label").move.agg(lambda m: m.max() - m.min())
	if mode == "moves":
		avail = span
	else:
		avail = span / d.groupby("label").tau_U.first()
	n = float(avail.min()) if n is None else float(n)
	keep = []
	for lab, g in d.groupby("label"):
		width = n if mode == "moves" else n * float(g.tau_U.iloc[0])
		keep.append(g[g.move > g.move.max() - width])
	import pandas as pd
	return pd.concat(keep), n


def cmd_analyze(args):
	import pandas as pd
	rng = np.random.default_rng(args.seed)
	s = pd.read_csv(args.scores)
	ref_row = s[s.frame_id == "reference"]
	d = s[s.frame_id != "reference"].copy()
	if args.frames:
		# scores.csv is resumable, so rows from an OLDER frames.tsv (e.g. before T8/T16 were
		# extended on 2026-11-23) would otherwise stay in the analysis unnoticed.
		ids = set(pd.read_csv(args.frames, sep="\t", usecols=["frame_id"]).frame_id)
		stale = ~d.frame_id.isin(ids)
		if stale.any():
			print(f"# dropping {stale.sum()} scored frames not in {args.frames} (stale rows); "
				f"{(~pd.Series(list(ids)).isin(d.frame_id)).sum()} frames of it are not scored yet", flush=True)
		d = d[~stale]
	d["T"] = d["T"].astype(float)
	os.makedirs(args.out_dir, exist_ok=True)
	energies = [e for e in ENERGIES if e in d and d[e].notna().any()]
	target = args.target
	lines = []

	def say(t=""):
		print(t, flush=True)
		lines.append(t)

	say(f"# {args.scores}: {len(d)} frames from {d.label.nunique()} chains at T = "
		+ ", ".join(f"{t:g}" for t in sorted(d["T"].unique())) + f"; {d.Hd.min()}-{d.Hd.max()} mutations")
	if len(ref_row):
		r = ref_row.iloc[0]
		say(f"# reference decode: rmsd_xtal {r.rmsd_xtal:.2f} A, tm_xtal {r.tm_xtal:.3f}, U_ce_ref {r.U_ce_ref:.4f}, "
			f"U_ce_xtal {r.U_ce_xtal:.4f}, E_xtal {r.E_xtal:.3f}")
	say(f"# target: {target}. Spearman rho; [95% block-bootstrap interval, blocks tau_U long, {args.n_boot} reps].")
	say("# * = the energy itself looks at the crystal, so part of its correlation with the target is built in.")
	if "U_am_calc" in d and d.U_am_calc.notna().any():
		c = d[d.U_am_calc.notna()]
		say(f"# U_am recheck on {len(c)} frames: max |recomputed - data.dat| = {np.abs(c.U_am_calc - c.U_am).max():.3g}")

	rows = []

	def block(name, sub, boot):
		say(f"\n## {name}: {len(sub)} frames")
		say(f"{'energy':12s} {'rho':>24s} {'partial | Hd':>24s} {'rho Hd':>8s}")
		for e in energies:
			r = with_ci(sub, lambda x: spearman(x[e], x[target]), args.n_boot if boot else 0, rng)
			p = with_ci(sub, lambda x: partial_spearman(x[e], x[target], x.Hd), args.n_boot if boot else 0, rng)
			star = "*" if e in SEES_CRYSTAL else " "
			ci = lambda v: f"{v[0]:+.3f} [{v[1]:+.2f},{v[2]:+.2f}]" if np.isfinite(v[1]) else f"{v[0]:+.3f}"
			say(f"{e + star:12s} {ci(r):>24s} {ci(p):>24s} {spearman(sub[e], sub.Hd):+8.3f}")
			rows.append({"selection": name, "energy": e, "n": len(sub), "rho": r[0], "rho_lo": r[1], "rho_hi": r[2],
				"partial_Hd": p[0], "partial_lo": p[1], "partial_hi": p[2]})
		say(f"{'Hd itself':12s} {spearman(sub.Hd, sub[target]):+24.3f}")

	block("B pooled, all frames", d, True)
	A = select_decorrelated(d)
	block("A pooled, one frame per tau_U", A, False)
	C = select_stratified(d, args.n_bins, args.per_bin, rng)
	block(f"C pooled, {args.n_bins} log-U strata x up to {args.per_bin}", C, False)

	say(f"\n## spread inside each chain's window: the interquartile range of every column. A chain that "
		f"wanders less has a smaller spread and its rho is pulled toward zero by range restriction alone, "
		f"so read the within-T rows below against this table, not against each other.")
	say(f"{'T':>5s} {'moves':>8s} {'tau_U':>7s} {'win/tau':>8s} {target + ' IQR':>14s} "
		+ " ".join(f"{e:>10s}" for e in energies))
	for T, g in sorted(d.groupby("T"), key=lambda x: x[0]):
		span = g.groupby("label").move.agg(lambda m: m.max() - m.min()).mean()
		tau = g.groupby("label").tau_U.first().mean()
		iqr = lambda c: g[c].quantile(.75) - g[c].quantile(.25)
		say(f"{T:5g} {span:8.0f} {tau:7.0f} {span / tau:8.2f} {iqr(target):14.3g} "
			+ " ".join(f"{iqr(e):10.3g}" for e in energies))

	say(f"\n## within each temperature (both chains together), rho with {target}; B with interval, A point estimate")
	say(f"{'T':>5s} {'n_B':>5s} {'n_A':>4s} {target + ' median':>14s} " + " ".join(f"{e:>22s}" for e in energies))
	wt = {e: [] for e in energies}
	for T, g in sorted(d.groupby("T"), key=lambda x: x[0]):
		a = A[A["T"] == T]
		cells = []
		for e in energies:
			r = with_ci(g, lambda x: spearman(x[e], x[target]), args.n_boot, rng)
			wt[e].append(r[0])
			cells.append(f"{r[0]:+.2f} [{r[1]:+.2f},{r[2]:+.2f}] {spearman(a[e], a[target]):+.2f}"
				if np.isfinite(r[1]) else f"{r[0]:+.2f}")
			rows.append({"selection": f"B within T={T:g}", "energy": e, "n": len(g), "rho": r[0], "rho_lo": r[1],
				"rho_hi": r[2], "rho_A": spearman(a[e], a[target]), "n_A": len(a)})
		say(f"{T:5g} {len(g):5d} {len(a):4d} {g[target].median():14.2f} " + " ".join(f"{c:>22s}" for c in cells))
	say(f"{'mean':>5s} {'':5s} {'':4s} {'':14s} " + " ".join(f"{np.nanmean(wt[e]):>+22.2f}" for e in energies))

	m, width = match_window(d, args.match_window, args.match_n)
	if width is not None:
		unit = "moves" if args.match_window == "moves" else "x tau_U"
		say(f"\n## within each temperature, MATCHED: every chain cut to its last {width:g} {unit} "
			f"({len(m)} frames). THIS is the column to read as a trend across T; the raw one above is not, "
			f"because its windows differ in how much of the equilibrium spread they cover.")
		say(f"{'T':>5s} {'n':>5s} {target + ' IQR':>14s} " + " ".join(f"{e:>10s}" for e in energies))
		for T, g in sorted(m.groupby("T"), key=lambda x: x[0]):
			cells = []
			for e in energies:
				r = spearman(g[e], g[target])
				cells.append(f"{r:+.2f}")
				rows.append({"selection": f"matched within T={T:g}", "energy": e, "n": len(g), "rho": r})
			say(f"{T:5g} {len(g):5d} {g[target].quantile(.75) - g[target].quantile(.25):14.3g} "
				+ " ".join(f"{c:>10s}" for c in cells))

	if "rmsd_xtal_n" in d and d.rmsd_xtal_n.notna().any():
		n = d[d.rmsd_xtal_n.notna()]
		say(f"\n## decode-noise floor: {len(n)} frames re-decoded at the alternative step count")
		for a_, b_ in (("rmsd_xtal", "rmsd_xtal_n"), ("E_ref", "E_ref_n"), ("E_xtal", "E_xtal_n")):
			iqr = n[a_].quantile(.75) - n[a_].quantile(.25)
			say(f"{a_:10s} rho between step counts {spearman(n[a_], n[b_]):+.3f}; median |diff| "
				f"{np.median(np.abs(n[a_] - n[b_])):.3f} = {np.median(np.abs(n[a_] - n[b_])) / iqr:.2f} of its IQR")

	pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"correlations_{target}.csv"), index=False)
	with open(os.path.join(args.out_dir, f"summary_{target}.txt"), "w") as f:
		f.write("\n".join(lines) + "\n")
	figure(d, energies, target, os.path.join(args.out_dir, f"energy_vs_{target}.png"))
	print(f"Saved -> {args.out_dir}/summary_{target}.txt, correlations_{target}.csv, energy_vs_{target}.png")


def figure(d, energies, target, path):
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		from matplotlib.colors import LogNorm
	except ImportError:
		print("# matplotlib not installed: no figure")
		return
	n = len(energies)
	cols = min(4, n)
	rows_ = int(math.ceil(n / cols))
	fig, axes = plt.subplots(rows_, cols, figsize=(3.6 * cols, 3.4 * rows_), squeeze=False, layout="constrained")
	norm = LogNorm(vmin=d["T"].min(), vmax=d["T"].max())
	cmap = matplotlib.colormaps["Blues"]
	cmap = matplotlib.colors.ListedColormap(cmap(np.linspace(0.35, 1.0, 256)))   # one hue, light -> dark
	for ax, e in zip(axes.ravel(), energies):
		sc = ax.scatter(d[e], d[target], c=d["T"], cmap=cmap, norm=norm, s=6, linewidths=0, alpha=0.7)
		ax.set_xlabel(e, fontsize=9)
		ax.set_ylabel(target, fontsize=9)
		ax.set_title(f"{e}{' *' if e in SEES_CRYSTAL else ''}\nrho {spearman(d[e], d[target]):+.2f}, "
			f"partial|Hd {partial_spearman(d[e], d[target], d.Hd):+.2f}", fontsize=9)
		if e == "U_am":
			ax.set_xscale("log")
		ax.tick_params(labelsize=8)
		for sp in ("top", "right"):
			ax.spines[sp].set_visible(False)
	for ax in axes.ravel()[n:]:
		ax.axis("off")
	cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8)
	cb.set_label("T")
	fig.suptitle(f"equilibrated PTEN frames: energy vs {target}  (* = the energy looks at the crystal)", fontsize=10)
	fig.savefig(path, dpi=150, bbox_inches="tight")
	plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = ap.add_subparsers(dest="cmd", required=True)

	f = sub.add_parser("frames", help="CPU: replay the ladder and write the dense frame set")
	f.add_argument("--ref", default="pten_1d5r")
	f.add_argument("--runs", nargs="*", help="LABEL=PATH run directories (default: the 18-chain ladder)")
	f.add_argument("--burn-in-frac", type=float, default=0.8)
	f.add_argument("--per-chain", type=int, default=200)
	f.add_argument("--out", required=True)

	s = sub.add_parser("score", help="GPU: every energy and the RMSD to the crystal, per frame")
	s.add_argument("--ref", required=True, help="pass pten_1d5r explicitly (no silent default reference)")
	s.add_argument("--frames", required=True)
	s.add_argument("--out", required=True)
	s.add_argument("--exp", default=os.path.join(MONTE_CARLO_DIR, "structure_compare", "1D5R.pdb"))
	s.add_argument("--pair-csv", default=os.path.join(MONTE_CARLO_DIR, "structure_compare",
		"pten_1d5r_vs_1D5R_per_residue.csv"))
	s.add_argument("--num-steps", type=int, default=2,
		help="decode steps (2 matched 128 against the crystal on 2026-11-16: P@L 0.98, RMSD 2.0)")
	s.add_argument("--noise-steps", type=int, default=8)
	s.add_argument("--noise-frames", type=int, default=50)
	s.add_argument("--recheck-am", type=int, default=5, help="frames whose U_am is recomputed against data.dat")
	s.add_argument("--ca-cutoff", type=float, default=8.0)
	s.add_argument("--soft-width", type=float, default=1.0)
	s.add_argument("--seed", type=int, default=0)

	a = sub.add_parser("analyze", help="CPU: correlations, subsets A/B/C, figure")
	a.add_argument("--scores", required=True)
	a.add_argument("--out-dir", required=True)
	a.add_argument("--frames", default=os.path.join(MONTE_CARLO_DIR, "energy_rmsd", "frames.tsv"),
		help="keep only scored frames listed here (pass '' to keep every row of --scores)")
	a.add_argument("--target", default="rmsd_xtal", choices=["rmsd_xtal", "tm_xtal", "rmsd_ref", "plddt"],
		help="what the energies are correlated against. rmsd_xtal and rmsd_ref grow as the structure gets "
			"worse, so a POSITIVE rho means the energy predicts it; tm_xtal and plddt grow as it gets "
			"BETTER, so there the informative sign is negative")
	a.add_argument("--n-boot", type=int, default=500)
	a.add_argument("--n-bins", type=int, default=25)
	a.add_argument("--per-bin", type=int, default=20)
	a.add_argument("--seed", type=int, default=0)
	a.add_argument("--match-window", default="tau", choices=["none", "moves", "tau"],
		help="trim every chain to a comparable window for the matched within-T table: equal moves, or "
			"equal multiples of its own tau_U (the default, since range restriction scales with the "
			"number of independent samples, not with clock time)")
	a.add_argument("--match-n", type=float, default=None,
		help="the window width, in moves or in tau_U (default: the largest every chain can supply)")

	args = ap.parse_args()
	{"frames": cmd_frames, "score": cmd_score, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
	main()
