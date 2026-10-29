"""
Batching benchmark and exact-neighbourhood diagnostic, for the sampler
speed-up candidates (docs: the "speed up the sampler" analysis, 2026-10-29).
Needs the GPU machine. Run from softmax/code:

    nohup python -u tests/bench_batch.py > bench_batch.log 2>&1 &

About 5-10 min on a free card. Peak memory is set by the fwd+bwd part
(~6.4 GB per sequence in the batch); it stops at the first OOM and goes on.

  (A) Cost vs batch size. ESM3's forward runs one sequence at a time
      (_last_attention asserts batch 1), so the GPU may be mostly idle at
      L=566. Times a batched no-grad forward (U for B sequences at once) and
      a batched forward+backward (the one-pass gradient for B sequences) for
      each B, against B single calls. Also checks that the batched U equals
      _exact_energy, sequence by sequence.
  (B) GPU busy fraction of one single-sequence forward (torch.profiler): the
      kernel time over the wall time. Low means launch/CPU bound, which
      CUDA graphs or torch.compile could recover.
  (C) The exact single-site neighbourhood of one state: at S sites, U of all
      19 single mutants (batched). From it:
        - how well the one-pass gradient ranks those mutants (Spearman, top-1),
          which bounds any gradient-informed site or residue selection;
        - per T, the current move's acceptance in its near-uniform limit
          (mean min(1, exp(-dU/T))) against the change probability of a
          Metropolized-Gibbs move at a uniform site (propose j != cur with
          prob pi_j/(1-pi_cur), accept min(1, (1-pi_cur)/(1-pi_j)), pi the
          exact 20-state conditional at that site). Its reverse move uses the
          same 20 energies, so it costs one batched forward of 19 sequences
          and no gradient.
      Use an equilibrated state (START_RUN): a random MUTS start sits far
      above equilibrium and measures relaxation, not sampling.

Environment overrides:  BATCHES=1,2,4,8,19,32  GRAD_BATCHES=1,2,3  REPS=5
                        START_RUN=<run dir with eprot/>  MUTS=51  S=20
                        TS=8,12,15,32,100  CHUNK=19  REF=zero_polymer
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../customs")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../monte_carlo")))

import math
import time

import numpy as np
import torch

from classes.rate_sampler import ExtendedProteinRateSampler, _NONCANONICAL_RESIDUES
from classes.ExtendedProtein import ExtendedProtein
from generator.custom_generator import CustomGenerator
from custom_esm.utils.custom import clean_and_symm
from custom_esm.utils.structure.affine3d import build_affine3d_from_coordinates
import custom_esm.utils.constants.esm3 as C

from references import get_reference

REF_NAME, REF_SEQ = get_reference()
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,2,4,8,19,32").split(",")]
GRAD_BATCHES = [int(x) for x in os.environ.get("GRAD_BATCHES", "1,2,3").split(",")]
REPS = int(os.environ.get("REPS", 5))
START_RUN = os.environ.get("START_RUN", "")
MUTS = int(os.environ.get("MUTS", 51))
S = int(os.environ.get("S", 20))
TS = [float(x) for x in os.environ.get("TS", "8,12,15,32,100").split(",")]
CHUNK = int(os.environ.get("CHUNK", 19))
SEED = 0

PARS = {
	"T": 32.0, "dt": 1.0, "M": 1.0, "T_sftm": 0.1,
	"lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0, "eps": 1.0e-9, "n_quad": 40,
	"grad_mode": "one_pass",
}


def eprot_of(seq, device):
	e = ExtendedProtein(sequence=seq, requires_grad=True, device=device)
	e.expand()
	return e


def batched_attention(model, probs):
	"""predict_attention for a (B, L, K) batch: _default's (1, L) inputs are
	expanded to B and the (empty) affine is built at batch B. Same bfloat16
	autocast as predict_attention. Returns (B, L, L)."""
	B, L = probs.shape[0], probs.shape[1] + 2
	tokens, _, _ = model._default(L, probs.device)
	tokens = tuple(t.expand(B, *t.shape[1:]) for t in tokens)
	coords = torch.full((B, L, 3, 3), float("nan"), dtype=torch.float, device=probs.device)
	affine, affine_mask = build_affine3d_from_coordinates(coords)
	with torch.autocast(enabled=probs.device.type == "cuda", device_type=probs.device.type, dtype=torch.bfloat16):
		x = model.encoder.custom_forward(probs, *tokens)
		attention = model.transformer.last_attention(x, None, affine, affine_mask, None)
	return clean_and_symm(attention)


def batched_U(model, probs, ref_am):
	am = batched_attention(model, probs)
	return torch.triu((torch.log(am) - torch.log(ref_am))**2.).sum(dim=(-2, -1))


def probs_of(seqs, device):
	return torch.stack([eprot_of(s, device).get_probs() for s in seqs])


def timed(fn, reps, device):
	fn()   # warm-up
	ts = []
	for _ in range(reps):
		if device.type == "cuda": torch.cuda.synchronize()
		t = time.perf_counter()
		fn()
		if device.type == "cuda": torch.cuda.synchronize()
		ts.append(time.perf_counter() - t)
	return float(np.median(ts))


def ranks(x):
	r = np.empty(len(x))
	r[np.argsort(x)] = np.arange(len(x))
	return r


def spearman(a, b):
	return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def start_sequence(sampler):
	if START_RUN:
		from run_nsites import last_checkpoint_sequence
		seq, move = last_checkpoint_sequence(START_RUN)
		return seq, f"{START_RUN} at move {move}"
	return sampler._mutate_canonical(REF_SEQ, MUTS), f"MUTS={MUTS} random mutations of the reference (NOT equilibrated: relaxation numbers)"


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
	ref_am = sampler.ref_eprot.am
	model = sampler.model
	vocab = C.SEQUENCE_USED_VOCAB
	canonical = [i for i, c in enumerate(vocab) if c not in _NONCANONICAL_RESIDUES]
	L = len(REF_SEQ)
	if device.type == "cuda":
		print(f"# device {torch.cuda.get_device_name(device)}, free {torch.cuda.mem_get_info(device)[0]/2**20:.0f} MiB at start")
	print(f"# reference: {REF_NAME} (L={L})\n")

	seq_A, origin = start_sequence(sampler)
	print(f"state A: {origin}")
	pool = [sampler._mutate_canonical(seq_A, 1) for _ in range(max(BATCHES + GRAD_BATCHES))]

	# ------------------------------------------------------------------ (A) --- #
	print("\n=== (A) cost vs batch size ===")
	with torch.no_grad():
		U_b = batched_U(model, probs_of(pool[:4], device), ref_am)
		U_1 = [sampler._exact_energy(eprot_of(s, device), PARS)[0] for s in pool[:4]]
	rel = max(abs(a - b)/max(abs(b), 1.) for a, b in zip(U_b.tolist(), U_1))
	print(f"batched U vs _exact_energy, 4 sequences: max relative difference {rel:.2e}  ({'OK' if rel < 1e-2 else 'CHECK'}; bf16 batched kernels need not be bit-identical)")

	def fwd(B):
		probs = probs_of(pool[:B], device)
		def run():
			with torch.no_grad():
				batched_U(model, probs, ref_am)
		return run

	def fwd_bwd(B):
		probs = probs_of(pool[:B], device)
		def run():
			p = probs.detach().clone().requires_grad_(True)
			with torch.enable_grad():
				batched_U(model, p, ref_am).sum().backward()
		return run

	t_fwd1 = timed(fwd(1), REPS, device)
	print(f"\nno-grad forward (exact U), B sequences at once:")
	print(f"   B    t [ms]   per seq [ms]   vs B single calls")
	t_fwd = {}
	for B in BATCHES:
		try:
			t_fwd[B] = timed(fwd(B), REPS, device)
		except torch.cuda.OutOfMemoryError:
			print(f"  {B:>3}    OOM"); torch.cuda.empty_cache(); break
		print(f"  {B:>3}  {1e3*t_fwd[B]:8.1f}  {1e3*t_fwd[B]/B:12.1f}  {B*t_fwd1/t_fwd[B]:10.2f}x")

	t_fb1 = timed(fwd_bwd(1), REPS, device)
	print(f"\nforward+backward (one-pass gradient), B sequences at once:")
	print(f"   B    t [ms]   per seq [ms]   vs B single calls   peak MiB")
	for B in GRAD_BATCHES:
		if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
		try:
			t = timed(fwd_bwd(B), REPS, device)
		except torch.cuda.OutOfMemoryError:
			print(f"  {B:>3}    OOM"); torch.cuda.empty_cache(); break
		peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else float("nan")
		print(f"  {B:>3}  {1e3*t:8.1f}  {1e3*t/B:12.1f}  {B*t_fb1/t:12.2f}x  {peak:13.0f}")

	# ------------------------------------------------------------------ (B) --- #
	print("\n=== (B) GPU busy fraction, single-sequence forward ===")
	if device.type == "cuda":
		run = fwd(1)
		run()
		torch.cuda.synchronize()
		with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
			t = time.perf_counter()
			for _ in range(REPS): run()
			torch.cuda.synchronize()
			wall = time.perf_counter() - t
		kernel_us = 0.
		for ev in prof.key_averages():
			kernel_us += getattr(ev, "self_device_time_total", getattr(ev, "self_cuda_time_total", 0.))
		print(f"wall {1e3*wall/REPS:.1f} ms/forward, GPU kernels {1e-3*kernel_us/REPS:.1f} ms/forward: busy {1e-6*kernel_us/wall:.0%}")
		print("(under the profiler; other jobs on the card make both numbers larger)")
	else:
		print("skipped (no CUDA)")

	# ------------------------------------------------------------------ (C) --- #
	print(f"\n=== (C) exact single-site neighbourhood of A, {S} sites x 19 mutants ===")
	gen = torch.Generator().manual_seed(SEED)
	sites = torch.randperm(L, generator=gen)[:S].tolist()
	U_A, _, g, _ = sampler._energy_and_grad(eprot_of(seq_A, device), PARS)
	g = g.float().cpu()
	cur, mutants, pred = [], [], []
	for s in sites:
		c = vocab.index(seq_A[s])
		cur.append(c)
		for j in canonical:
			if j == c: continue
			mutants.append(seq_A[:s] + vocab[j] + seq_A[s+1:])
			pred.append((g[s, j] - g[s, c]).item())
	t = time.perf_counter()
	dU = []
	with torch.no_grad():
		for k in range(0, len(mutants), CHUNK):
			dU += (batched_U(model, probs_of(mutants[k:k+CHUNK], device), ref_am) - U_A).tolist()
	print(f"U_A = {U_A:.1f}; {len(mutants)} mutants in {time.perf_counter() - t:.1f} s")
	dU, pred = np.array(dU).reshape(S, 19), np.array(pred).reshape(S, 19)

	print(f"\nsingle-mutant dU: median {np.median(dU):.1f}, IQR [{np.percentile(dU, 25):.1f}, {np.percentile(dU, 75):.1f}], downhill {np.mean(dU < 0):.0%}")
	print(f"gradient ranking (pred dU = g[s,j] - g[s,cur]):")
	print(f"  Spearman, all {dU.size} pairs pooled:   {spearman(pred.ravel(), dU.ravel()):+.3f}")
	print(f"  Spearman within a site, mean:     {np.mean([spearman(p, d) for p, d in zip(pred, dU)]):+.3f}")
	print(f"  top-1 (pred argmin == true argmin): {np.mean(pred.argmin(1) == dU.argmin(1)):.0%}  (chance 5%)")
	print(f"  sign agreement:                   {np.mean(np.sign(pred) == np.sign(dU)):.0%}")
	print(f"  Spearman of site ranking at T=32 (log sum_j exp(-dU/T), pred vs true): "
		f"{spearman(np.log(np.exp(-pred/32.).sum(1) + 1e-300), np.log(np.exp(-dU/32.).sum(1) + 1e-300)):+.3f}")

	t_mg = t_fwd.get(19, float("nan"))
	print(f"\nper T: current move (near-uniform limit) vs Metropolized Gibbs at a uniform site")
	print(f"  cost per move: one-pass fwd+bwd {1e3*t_fb1:.0f} ms, MG one batched forward of 19 {1e3*t_mg:.0f} ms")
	print(f"     T    a_MH    P_change_MG   ratio   changes/s MH   changes/s MG   gain")
	for T in TS:
		a_mh = np.mean(np.minimum(1., np.exp(-dU/T)))
		E = np.concatenate([np.zeros((S, 1)), dU], axis=1)/T   # column 0 = stay
		pi = np.exp(-(E - E.min(1, keepdims=True)))
		pi /= pi.sum(1, keepdims=True)
		pc = []
		with np.errstate(divide="ignore", invalid="ignore"):
			for p in pi:
				q = p[1:]/(1. - p[0])
				pc.append(np.sum(q*np.minimum(1., (1. - p[0])/(1. - p[1:]))) if p[0] < 1. else 0.)
		p_mg = float(np.mean(pc))
		r_mh, r_mg = a_mh/t_fb1, p_mg/t_mg
		print(f"  {T:5.0f}  {a_mh:6.3f}  {p_mg:12.3f}  {p_mg/max(a_mh, 1e-300):6.1f}x  {r_mh:13.2f}  {r_mg:13.2f}  {r_mg/max(r_mh, 1e-300):5.2f}x")
	print("\n(a_MH ignores the Hastings term and the gradient's tilt; compare it with the chains' own acceptance at that T.)")


if __name__ == "__main__":
	main()
