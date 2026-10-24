"""
T0, T1, T2 of the one-pass test plan (docs/one_pass_move.tex, Section 7), on
ESM3. Needs the GPU machine. Run from softmax/code:

    nohup python -u tests/test_one_pass_gradient.py > one_pass_gradient.log 2>&1 &

  (T1 and T2 run first, T0 last, so a T0 problem cannot hide T2.)
  T0 Gradient correctness. The one-pass gradient g = dU/dp at the one-hot
     sequence, at fp32 without autocast, against a central finite difference
     of U at FD_DTYPE (default float64, weights cast to it) along p_ik - p_ij,
     at FD_SITES sites x 3 pairs, for every h in FD_H. Pass: relative error
     < 1% for every pair at its best h. Also reported: how far the production
     gradient (bfloat16 autocast, what the sampler uses) is from the fp32 one,
     and "T2 at fp32": the T2 comparison with both gradients at fp32, which
     separates bfloat16 rounding from the method itself.
  T1 Same energy. U from the gradient-enabled forward (_energy_and_grad) equals
     _exact_energy bit for bit, at N_T1 random sequences.
  T2 Same ranking, predicted scale. At T2_SITES sites: the one-pass g against
     _grad_pass_sites (site_soft) over the 19 competitors. Pass: cosine of the
     centred vectors >= 0.999, same argmax except at near-ties (top two closer
     than the site_soft gradient's own run-to-run noise), median fitted scale
     within 1% of c = one_pass_dt_scale(). The site_soft gradient is computed
     twice per site to measure that noise floor.

The exact reference is a degenerate point: U_am is a sum of squares that all
vanish there, so g = 0 exactly (and the site_soft gradient is O(1e-3) of
its usual size). The default sequences are therefore MUTS=5 (near the floor,
as in the earlier informedness runs) and MUTS=51 (the bulk); the reference
is only checked for |g| ~ 0.

Environment overrides:  MUTS=5,51  FD_SITES=20  FD_H=1e-2,1e-3,1e-4  FD_DTYPE=float64
                        N_T1=50  T2_SITES=30  SKIP_T0=1  REF=zero_polymer
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))

import math
import time
import contextlib

import numpy as np
import torch

from classes.rate_sampler import ExtendedProteinRateSampler, one_pass_dt_scale
from classes.ExtendedProtein import ExtendedProtein
from generator.custom_generator import CustomGenerator
from utils.energies import compute_U_am
from custom_esm.utils.custom import clean_and_symm
import custom_esm.utils.constants.esm3 as C

from references import get_reference

REF_NAME, REF_SEQ = get_reference()
MUTS = [int(x) for x in os.environ.get("MUTS", "5,51").split(",")]
FD_SITES = int(os.environ.get("FD_SITES", 20))
FD_H = [float(x) for x in os.environ.get("FD_H", "1e-2,1e-3,1e-4").split(",")]
FD_DTYPE = os.environ.get("FD_DTYPE", "float64")
N_T1 = int(os.environ.get("N_T1", 50))
T2_SITES = int(os.environ.get("T2_SITES", 30))
SKIP_T0 = os.environ.get("SKIP_T0", "") not in ("", "0")
SEED = 0

PARS = {
	"T": 2.0, "dt": 1.0, "M": 1.0, "T_sftm": 0.1,
	"lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": 40,
	"grad_mode": "one_pass",
}


def eprot_of(seq, device):
	e = ExtendedProtein(sequence=seq, requires_grad=True, device=device)
	e.expand()
	return e


def no_autocast(device):
	return torch.autocast(device_type=device.type, enabled=False) if device.type == "cuda" else contextlib.nullcontext()


def attention_full(sampler, probs):
	"""predict_attention without its bfloat16 autocast, in the dtype of the
	weights and of probs (the caller casts both). ESM3._default builds the
	plddt inputs and the affine in float32 whatever the weights are, so they
	are cast to probs.dtype here (no-op at fp32)."""
	model = sampler.model
	default = type(model)._default

	def default_in_dtype(L, device):
		tokens, affine, affine_mask = default(model, L, device)
		tokens = tuple(t.to(probs.dtype) if t.is_floating_point() else t for t in tokens)
		return tokens, affine.to(dtype=probs.dtype), affine_mask

	model._default = default_in_dtype
	try:
		with no_autocast(probs.device):
			return clean_and_symm(model._last_attention(sequence_probs=probs.unsqueeze(0))).squeeze(0)
	finally:
		del model._default


def centred_cos(a, b):
	a, b = a - a.mean(), b - b.mean()
	return (a @ b / (a.norm()*b.norm() + 1e-30)).item()


def main():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	sampler._canonical_idx = sampler._canonical_idx.to(device)
	sampler.generator = CustomGenerator(seed=SEED, device=device)
	sampler.ref_eprot = eprot_of(REF_SEQ, device)
	sampler.ref_eprot.requires_grad = False
	with torch.no_grad():
		sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())
	vocab = C.SEQUENCE_USED_VOCAB
	L = len(REF_SEQ)
	c = one_pass_dt_scale(PARS["T_sftm"])

	print(f"# reference: {REF_NAME} (L={L})")
	print(f"device {device}, c = one_pass_dt_scale({PARS['T_sftm']}) = {c:.6e}\n")

	seqs = {m: sampler._mutate_canonical(REF_SEQ, m) for m in MUTS}
	verdicts = {}

	# ------------------------------------------------------------ reference --- #
	U0, _, g0, _ = sampler._energy_and_grad(eprot_of(REF_SEQ, device), PARS)
	print(f"reference: U = {U0:.3e}, max |g| = {g0.abs().max().item():.3e}  (expected 0: U_am is a sum of squares that vanish here)\n")

	# ------------------------------------------------------------------- T1 --- #
	print(f"=== T1: U from the gradient-enabled forward vs _exact_energy, {N_T1} sequences ===")
	equal, worst = 0, 0.
	for k in range(N_T1):
		m = int(torch.randint(1, 120, (1,), generator=sampler.generator.get(), device=device).item())
		seq = sampler._mutate_canonical(REF_SEQ, m)
		U1, Uam1, _, _ = sampler._energy_and_grad(eprot_of(seq, device), PARS)
		U2, Uam2, _ = sampler._exact_energy(eprot_of(seq, device), PARS)
		equal += int(U1 == U2 and Uam1 == Uam2)
		worst = max(worst, abs(U1 - U2))
		if k < 5 or U1 != U2:
			print(f"  muts={m:3d}: one-pass U={U1!r}  exact U={U2!r}  diff={U1 - U2:+.3e}")
	verdicts["T1"] = equal == N_T1
	print(f"T1: {equal}/{N_T1} bit-identical, worst |diff| = {worst:.3e}  -> {'PASS' if verdicts['T1'] else 'FAIL'}\n")

	# ------------------------------------------------------------------- T2 --- #
	print(f"=== T2: one-pass g vs site_soft gradient over the competitors, {T2_SITES} sites ===")
	t2_ok = True
	for m, seq in seqs.items():
		e = eprot_of(seq, device)
		_, _, g, _ = sampler._energy_and_grad(eprot_of(seq, device), PARS)
		torch.manual_seed(SEED + 1000 + m)
		sites = torch.randperm(L)[:T2_SITES].tolist()
		cos, cos_noise, scales, mism, ties = [], [], [], 0, 0
		for s in sites:
			cur = vocab.index(seq[s])
			comps = sampler._competition_indices(cur)
			old1 = sampler._grad_pass_sites(e, [s], PARS)[0][comps]
			old2 = sampler._grad_pass_sites(e, [s], PARS)[0][comps]
			new = g[s][comps]
			cos.append(centred_cos(old1, new))
			cos_noise.append(centred_cos(old1, old2))
			oc, nc = old1 - old1.mean(), new - new.mean()
			scales.append((oc @ nc / (nc @ nc)).item())
			# the proposal favours the most negative gradient
			a_old, a_new = int((-old1).argmax()), int((-new).argmax())
			if a_old != a_new:
				srt = torch.sort(-old1, descending=True).values
				gap = (srt[0] - srt[1]).item()
				noise = (old1 - old2).abs().max().item()
				tie = gap <= 2*noise or gap <= 1e-3*(srt[0] - srt[-1]).abs().item()
				ties += int(tie)
				mism += int(not tie)
				print(f"  site {s:3d}: argmax differs (site_soft {vocab[int(comps[a_old])]}, one-pass {vocab[int(comps[a_new])]}), "
					  f"site_soft top-2 gap {gap:.3e}, its noise {noise:.3e} -> {'near-tie' if tie else 'REAL MISMATCH'}")
		s_med = float(np.median(scales))
		ok = min(cos) >= 0.999 and mism == 0 and abs(s_med/c - 1.) < 0.01
		t2_ok &= ok
		print(f"MUTS={m}: centred cosine min {min(cos):.6f} median {np.median(cos):.6f} "
			  f"(site_soft vs itself: min {min(cos_noise):.6f}); argmax mismatches {mism} (+{ties} near-ties); "
			  f"fitted scale median {s_med:.5e} = {s_med/c:.4f} c (range {min(scales)/c:.4f}..{max(scales)/c:.4f}) -> {'PASS' if ok else 'FAIL'}\n")
	verdicts["T2"] = t2_ok

	# ------------------------------------------------------------------- T0 --- #
	if not SKIP_T0:
		print(f"=== T0: fp32 gradient vs central finite differences in {FD_DTYPE}, {FD_SITES} sites x 3 pairs, h in {FD_H} ===")
		# ESM3 is stored in bfloat16 (esm3.py:334) and autocast casts inputs to
		# it. The reference path casts the weights instead: fp32 for the
		# gradient (an fp64 backward would not fit next to fp64 weights) and
		# FD_DTYPE for the finite differences. The 2026-10-23 run did both in
		# fp32: the error then grew as h shrank (1/h), the signature of fp32
		# round-off in U ~ 1e3-1e4, and pairs with small gaps failed. bf16 ->
		# fp32/fp64 -> bf16 is lossless, so the weights are restored exactly.
		weight_dtype = next(sampler.model.parameters()).dtype
		fd_dtype = getattr(torch, FD_DTYPE)

		def energy_at(P, dtype, ref_am):
			return compute_U_am(attention_full(sampler, P.to(dtype)), ref_am)

		def site_soft_full(seq_e, s):
			"""_grad_pass_sites at full precision: site s relaxed, others one-hot."""
			hard = seq_e.get_probs()
			x = seq_e.logits[[s]].detach().clone().requires_grad_(True)
			probs = hard.clone()
			probs[[s]] = torch.softmax(x/PARS["T_sftm"], dim=-1)
			energy_at(probs, torch.float32, ref_am32).backward()
			return x.grad[0].detach()

		sampler.model.float()
		with torch.no_grad():
			ref_am32 = attention_full(sampler, sampler.ref_eprot.get_probs())
		all_ok, t2f_ok = True, True
		for m, seq in seqs.items():
			e = eprot_of(seq, device)
			P0 = e.get_probs().detach().clone().requires_grad_(True)
			t0 = time.perf_counter()
			U = energy_at(P0, torch.float32, ref_am32)
			U.backward()
			g32 = P0.grad.detach()
			print(f"MUTS={m}: U_fp32={U.item():.4f}  (fwd+bwd {time.perf_counter() - t0:.2f}s)")

			# T2 again, both gradients at fp32: separates the bf16 effect from the method.
			torch.manual_seed(SEED + 1000 + m)
			t2_sites = torch.randperm(L)[:T2_SITES].tolist()
			cos_f, sc_f = [], []
			for s in t2_sites:
				comps = sampler._competition_indices(vocab.index(seq[s]))
				old = site_soft_full(e, s)[comps]
				new = g32[s][comps]
				cos_f.append(centred_cos(old, new))
				oc, nc = old - old.mean(), new - new.mean()
				sc_f.append((oc @ nc / (nc @ nc)).item())
			ok_f = min(cos_f) >= 0.999 and abs(np.median(sc_f)/c - 1.) < 0.01
			t2f_ok &= ok_f
			print(f"MUTS={m}: T2 at fp32 ({T2_SITES} sites): centred cosine min {min(cos_f):.6f} median {np.median(cos_f):.6f}; "
				  f"scale median {np.median(sc_f)/c:.4f} c (range {min(sc_f)/c:.4f}..{max(sc_f)/c:.4f}) -> {'PASS' if ok_f else 'FAIL'}")

			sampler.model.to(weight_dtype)
			_, _, g16, _ = sampler._energy_and_grad(eprot_of(seq, device), PARS)
			sampler.model.to(fd_dtype)
			with torch.no_grad():
				ref_am_fd = attention_full(sampler, sampler.ref_eprot.get_probs().to(fd_dtype))
			P0d = P0.detach().to(fd_dtype)
			torch.manual_seed(SEED + m)
			sites = torch.randperm(L)[:FD_SITES].tolist()
			errs, gaps, cos16 = [], [], []
			with torch.no_grad():
				for s in sites:
					cur = vocab.index(seq[s])
					comps = sampler._competition_indices(cur).tolist()
					cos16.append(centred_cos(g16[s][comps], g32[s][comps]))
					picks = torch.randperm(len(comps))[:4].tolist()
					for kk, jj in [(comps[picks[0]], comps[picks[1]]), (comps[picks[2]], cur), (comps[picks[3]], comps[picks[0]])]:
						d = torch.zeros_like(P0d)
						d[s, kk], d[s, jj] = 1., -1.
						an = (g32[s, kk] - g32[s, jj]).item()
						row = []
						for h in FD_H:
							fd = ((energy_at(P0d + h*d, fd_dtype, ref_am_fd) - energy_at(P0d - h*d, fd_dtype, ref_am_fd))/(2*h)).item()
							row.append(abs(fd - an)/max(abs(fd), 1e-12))
						errs.append(row)
						gaps.append(abs(an))
						print(f"  site {s:3d} ({vocab[cur]}) {vocab[kk]}-{vocab[jj]}: g diff {an:+.4e}  "
							  + "  ".join(f"h={h:g}: rel err {r:.2e}" for h, r in zip(FD_H, row)))
			sampler.model.float()
			errs, gaps = np.array(errs), np.array(gaps)
			best = errs.min(axis=1)
			ok = bool(np.all(best < 0.01))
			all_ok &= ok
			print(f"MUTS={m}: relative error at the best h: median {np.median(best):.2e}, max {best.max():.2e} "
				  f"(absolute: median {np.median(best*gaps):.3g}, max {np.max(best*gaps):.3g}; median |g diff| {np.median(gaps):.3g}); "
				  f"per h, median: " + ", ".join(f"{h:g}: {np.median(errs[:, i]):.2e}" for i, h in enumerate(FD_H)))
			print(f"MUTS={m}: bf16 production gradient vs fp32, centred cosine over competitors: "
				  f"min {min(cos16):.5f}, median {np.median(cos16):.5f}\n")
		sampler.model.to(weight_dtype)
		verdicts["T0"] = all_ok
		verdicts["T2 at fp32"] = t2f_ok
		print(f"T0: {'PASS' if all_ok else 'FAIL'} (criterion: every pair < 1% at its best h)\n")



	print("=== summary ===")
	for k, v in verdicts.items():
		print(f"{k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
	main()
