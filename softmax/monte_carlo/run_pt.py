"""
Replica exchange (parallel tempering) on zero_polymer, with the one-pass move
(DEVLOG 2026-10-24).

Why. Single chains at fixed T did not equilibrate at T=32 and T=100 within
20000 moves (DEVLOG 2026-10-23): at T=100 the reference- and random-started
chains sit on two flat U levels about 4500 apart. Replica exchange runs one
walker per temperature of a ladder and periodically proposes to swap the
states of neighbouring temperatures, so a state stuck at a low T can travel
to a high T, decorrelate there and come back. It also gives <U>, <Hd_to_ref>
and Var(U)/T^2 on the whole ladder in one run, which is what locating the
transition needs.

Kernel. A sweep moves every walker once with the sampler's own _step (the
one-pass move by default) at the temperature of its slot. Every --swap-every
sweeps one swap round follows: pairs (k, k+1) with k even on even rounds and
k odd on odd rounds, each accepted with probability
    min(1, exp[(1/T_k - 1/T_{k+1}) (U_k - U_{k+1})]),
the standard exact replica-exchange rule. U is the exact energy, which the
one-pass cache already holds, so a swap costs no network pass. The cache is
keyed by sequence and energy weights, not T, so it stays valid when a state
changes temperature; each walker keeps its own.

dt. Each slot uses the dt rule of run_nsites.py at its own T (dt/sqrt(T)
fixed, times one_pass_dt_scale() for the one-pass move), so the move at every
rung is the one already tested.

Output (--out):
  ladder.dat         one row per sweep: sweep, wall seconds of the sweep, then
                     U, Hd_to_ref and walker id at every slot (T order).
  swaps.dat          one row per swap attempt: sweep, k, T_k, T_k+1, accepted.
  walkers/w<i>/      the trajectory of walker i (it changes T at swaps), in
                     run_nsites.py's data.dat format plus a T column, and its
                     start.txt, so analyze_nsites.py's reconstruction works.
  state.pt           checkpoint: sweep, slot permutation, sequences, generator
                     state. --resume continues from it to a larger --sweeps.
  pars.txt, run_info.txt
Files are flushed at checkpoints only, so they always end at the same sweep.

Not run in the cloud session that wrote it (no torch there). The swap and
slot bookkeeping below run without torch and were tested on a toy energy.

Usage:
    python -u run_pt.py --ts "10 12.2 14.9 ..." --sweeps 2000 --start reference --seed 300 --out pt_study/ref
    python -u run_pt.py --t-min 10 --t-max 200 --n-temps 16 --sweeps 2000 --start random --seed 400 --out pt_study/rand
"""
import os
import sys
import math
import time
import argparse

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../customs"))

DATA_KEYS = ["move", "time", "U", "U_am", "U_structure_ce", "Hd_to_ref", "site", "cur_aa", "proposed_aa",
	"dU", "log_a_AB", "log_a_BA", "log_ratio", "accepted", "acc_moves", "acc_rate", "T"]
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------ ladder bookkeeping (no torch) --- #

def geometric_ladder(t_min, t_max, n):
	return [float(f"{t:.4g}") for t in np.geomspace(t_min, t_max, n)]


def swap_log_prob(T_lo, T_hi, U_lo, U_hi):
	"""log acceptance of exchanging the states at T_lo and T_hi."""
	return (1. / T_lo - 1. / T_hi) * (U_lo - U_hi)


def swap_round(perm, U_of_walker, Ts, parity, uniform):
	"""One round of neighbour swaps on pairs (k, k+1) with k % 2 == parity.
	perm[k] is the walker at slot k; it is changed in place. uniform() draws
	from [0, 1). Returns [(k, accepted)]."""
	out = []
	for k in range(parity, len(Ts) - 1, 2):
		a, b = perm[k], perm[k + 1]
		la = swap_log_prob(Ts[k], Ts[k + 1], U_of_walker[a], U_of_walker[b])
		acc = math.log(max(uniform(), 1e-300)) <= min(0., la)
		if acc:
			perm[k], perm[k + 1] = b, a
		out.append((k, int(acc)))
	return out


class Walker:
	def __init__(self, wid, sequence, U, U_am, ref_seq, out_dir, start_note):
		self.wid, self.sequence, self.U, self.U_am = wid, sequence, U, U_am
		self.eprot, self.cache = None, None
		self.acc_moves, self.moves = 0, 0
		self.ref_seq = ref_seq
		self.dir = f"{out_dir}/walkers/w{wid}"
		self.rows = []
		self.start_note = start_note

	def hd(self):
		return sum(a != b for a, b in zip(self.sequence, self.ref_seq))

	def write_start(self):
		os.makedirs(self.dir, exist_ok=True)
		with open(f"{self.dir}/start.txt", "w") as f:
			f.write(f"# start: {self.start_note}\n# Hd_to_ref: {self.hd()}\n{self.sequence}\n")
		with open(f"{self.dir}/data.dat", "w") as f:
			f.write("# replica-exchange walker trajectory (run_pt.py); columns as in run_nsites.py data.dat, plus T\n")
			f.write("\t".join(DATA_KEYS) + "\n")
			f.write("\t".join(str(x) for x in [0, 0., self.U, self.U_am, 0., self.hd(), 0, "", "", 0., 0., 0., 0., 0, 0, 1., ""]) + "\n")

	def log(self, sweep, wall, T, info):
		self.moves += 1
		self.acc_moves += info["accepted"]
		self.rows.append("\t".join(str(x) for x in [
			sweep, f"{wall:.3f}", self.U, self.U_am, 0., self.hd(), info["site"], info["cur_aa"], info["proposed_aa"],
			info["dU"], info["log_a_AB"], info["log_a_BA"], info["log_ratio"], info["accepted"],
			self.acc_moves, self.acc_moves / self.moves, T]))

	def flush(self):
		if self.rows:
			with open(f"{self.dir}/data.dat", "a") as f:
				f.write("\n".join(self.rows) + "\n")
			self.rows = []


class Ladder:
	"""Walkers, the slot permutation and the output files. step(walker, k)
	must move the walker once at slot k's temperature and return the info
	dict of the sampler's _step (with 'accepted', 'U_B', 'U_am_B',
	'proposed_sequence')."""

	def __init__(self, Ts, walkers, out_dir, swap_every, uniform):
		self.Ts, self.walkers, self.out = Ts, walkers, out_dir
		self.perm = list(range(len(Ts)))
		self.swap_every, self.uniform = swap_every, uniform
		self.sweep, self.rounds = 0, 0
		self.ladder_rows, self.swap_rows = [], []

	def header(self):
		K = len(self.Ts)
		with open(f"{self.out}/ladder.dat", "w") as f:
			f.write("# T per slot: " + " ".join(str(t) for t in self.Ts) + "\n")
			f.write("\t".join(["sweep", "wall_s"] + [f"U_{k}" for k in range(K)] + [f"Hd_{k}" for k in range(K)] + [f"w_{k}" for k in range(K)]) + "\n")
		with open(f"{self.out}/swaps.dat", "w") as f:
			f.write("sweep\tk\tT_k\tT_k1\taccepted\n")
		self._ladder_row(0.)

	def _ladder_row(self, wall):
		ws = [self.walkers[w] for w in self.perm]
		self.ladder_rows.append("\t".join([str(self.sweep), f"{wall:.3f}"] + [f"{w.U}" for w in ws] + [str(w.hd()) for w in ws] + [str(w.wid) for w in ws]))

	def do_sweep(self, step):
		t0 = time.perf_counter()
		self.sweep += 1
		for k, wid in enumerate(self.perm):
			w = self.walkers[wid]
			info = step(w, k)
			if info["accepted"]:
				w.sequence, w.U, w.U_am = info["proposed_sequence"], info["U_B"], info["U_am_B"]
			w.log(self.sweep, time.perf_counter() - t0, self.Ts[k], info)
		if self.sweep % self.swap_every == 0:
			U = {w.wid: w.U for w in self.walkers}
			for k, acc in swap_round(self.perm, U, self.Ts, self.rounds % 2, self.uniform):
				self.swap_rows.append(f"{self.sweep}\t{k}\t{self.Ts[k]}\t{self.Ts[k + 1]}\t{acc}")
			self.rounds += 1
		self._ladder_row(time.perf_counter() - t0)

	def flush(self):
		for w in self.walkers:
			w.flush()
		with open(f"{self.out}/ladder.dat", "a") as f:
			f.write("\n".join(self.ladder_rows) + "\n")
		if self.swap_rows:
			with open(f"{self.out}/swaps.dat", "a") as f:
				f.write("\n".join(self.swap_rows) + "\n")
		self.ladder_rows, self.swap_rows = [], []


# ------------------------------------------------------------- the real run --- #

def main(args):
	sys.path.insert(0, CODE_DIR)
	sys.path.insert(0, CUSTOMS_DIR)
	sys.path.append(os.path.join(CODE_DIR, "tests"))
	import torch
	from classes.rate_sampler import ExtendedProteinRateSampler, _NONCANONICAL_RESIDUES, one_pass_dt_scale
	from classes.ExtendedProtein import ExtendedProtein
	from generator.custom_generator import CustomGenerator
	from utils.general import load_inputs
	from references import get_reference, header

	print(f"PID: {os.getpid()}\n")
	ref_name, ref_seq = get_reference(args.ref)
	print(header(ref_name, ref_seq))
	out = args.out.rstrip("/")
	os.makedirs(out, exist_ok=True)
	assert args.sweeps % args.log_every == 0, "--sweeps must be a multiple of --log-every (files are flushed at checkpoints)"

	Ts = [float(t) for t in args.ts.split()] if args.ts else geometric_ladder(args.t_min, args.t_max, args.n_temps)
	assert Ts == sorted(Ts) and len(set(Ts)) == len(Ts), f"the ladder must be strictly increasing: {Ts}"
	scale = one_pass_dt_scale(0.1) if args.grad_mode == "one_pass" else 1.
	dts = [args.dt_factor * args.dt_at_t2 * (T / 2.) ** args.dt_exp * scale for T in Ts]
	base = {"M": 1.0, "T_sftm": 0.1, "lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0,
		"n_quad": 40, "n_sites": args.n_sites, "eps": 1.0e-9, "grad_mode": args.grad_mode}
	slot_pars = [dict(base, T=T, dt=dt) for T, dt in zip(Ts, dts)]

	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings=load_inputs(args.cs_file))
	sampler.model.to(device)
	sampler._canonical_idx = sampler._canonical_idx.to(device)
	sampler.generator = CustomGenerator(seed=args.seed, device=device)
	sampler.ref_eprot = ExtendedProtein(sequence=ref_seq, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	with torch.no_grad():
		sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())
	frozen = sorted(set(ref_seq) & set(_NONCANONICAL_RESIDUES))
	assert not frozen, f"reference contains non-canonical residues {frozen}"
	sampler._grad_cache = None

	def make_eprot(seq):
		e = ExtendedProtein(sequence=seq, requires_grad=True, device=device)
		e.expand()
		return e

	def energy(seq):
		e = make_eprot(seq)
		if args.grad_mode == "one_pass":
			U, U_am, grad, e = sampler._energy_and_grad(e, slot_pars[0])
			sampler._set_cache(e, U, U_am, grad, slot_pars[0])
			return U, U_am, e, sampler._grad_cache
		U, U_am, e = sampler._exact_energy(e, slot_pars[0])
		return U, U_am, e, None

	uniform = lambda: torch.rand(1, device=device, generator=sampler.generator.get()).item()

	if args.resume:
		state = torch.load(f"{out}/state.pt", weights_only=False)
		assert state["Ts"] == Ts, f"--resume: ladder {Ts} differs from the saved one {state['Ts']}"
		sampler.generator.load(f"{out}/generator.npy")
		walkers = []
		for wid, (seq, am, n) in enumerate(zip(state["seqs"], state["acc_moves"], state["moves"])):
			U, U_am, e, cache = energy(seq)   # the cache is refilled, which is exact (one_pass_move.tex, Section 4)
			w = Walker(wid, seq, U, U_am, ref_seq, out, "resume")
			w.eprot, w.cache, w.acc_moves, w.moves = e, cache, am, n
			walkers.append(w)
		ladder = Ladder(Ts, walkers, out, args.swap_every, uniform)
		ladder.perm, ladder.sweep, ladder.rounds = list(state["perm"]), state["sweep"], state["rounds"]
		print(f"resumed at sweep {ladder.sweep}")
	else:
		rng = np.random.default_rng(args.seed)
		walkers = []
		for wid in range(len(Ts)):
			if args.start == "reference":
				seq, note = ref_seq, "reference"
			else:
				seq = "".join(CANONICAL[i] for i in rng.integers(0, 20, len(ref_seq)))
				note = f"random (numpy seed {args.seed}, walker {wid})"
			U, U_am, e, cache = energy(seq)
			w = Walker(wid, seq, U, U_am, ref_seq, out, note)
			w.eprot, w.cache = e, cache
			w.write_start()
			walkers.append(w)
		ladder = Ladder(Ts, walkers, out, args.swap_every, uniform)
		ladder.header()
		with open(f"{out}/pars.txt", "w") as f:
			f.write("# replica exchange (run_pt.py)\n")
			for k, v in sorted(dict(base, seed=args.seed, swap_every=args.swap_every, start=args.start).items()):
				f.write(f"{k:<20}{v}\n")
			f.write("T_ladder            " + " ".join(str(t) for t in Ts) + "\n")
			f.write("dt_ladder           " + " ".join(f"{d:.6g}" for d in dts) + "\n")

	def step(w, k):
		sampler._grad_cache = w.cache
		eprot, info = sampler._step(w.eprot, w.U, slot_pars[k])
		w.eprot, w.cache = eprot, sampler._grad_cache
		return info

	def checkpoint():
		ladder.flush()
		sampler.generator.save(f"{out}/generator.npy")
		torch.save({"sweep": ladder.sweep, "rounds": ladder.rounds, "perm": ladder.perm, "Ts": Ts,
			"seqs": [w.sequence for w in walkers], "acc_moves": [w.acc_moves for w in walkers],
			"moves": [w.moves for w in walkers]}, f"{out}/state.pt")

	if not args.resume:
		checkpoint()
	t_start = time.perf_counter()
	while ladder.sweep < args.sweeps:
		ladder.do_sweep(step)
		if ladder.sweep % args.log_every == 0:
			checkpoint()
			Us = " ".join(f"{walkers[w].U:.0f}" for w in ladder.perm)
			print(f"sweep {ladder.sweep:6d}  {(time.perf_counter() - t_start) / 3600:.2f} h  U by T: {Us}", flush=True)

	with open(f"{out}/run_info.txt", "a") as f:
		f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  sweeps={args.sweeps}  resume={bool(args.resume)}\n")
		f.write(f"temps {len(Ts)}\ngrad_mode {args.grad_mode}\nn_sites {args.n_sites}\nwall_total_s {time.perf_counter() - t_start:.1f}\n")
		if torch.cuda.is_available():
			f.write(f"gpu {torch.cuda.get_device_name(0)}\npeak_allocated_MB {torch.cuda.max_memory_allocated() / 2**20:.0f}\n")
		f.write("\n")
	print("\nSimulation completed!")


def create_parser():
	p = argparse.ArgumentParser()
	p.add_argument("--ts", type=str, default=None, help='explicit ladder, e.g. "10 15 22"; default geometric from the next three')
	p.add_argument("--t-min", type=float, default=10.)
	p.add_argument("--t-max", type=float, default=200.)
	p.add_argument("--n-temps", type=int, default=16)
	p.add_argument("--sweeps", type=int, required=True, help="moves per walker")
	p.add_argument("--swap-every", type=int, default=1, help="sweeps between swap rounds")
	p.add_argument("--start", choices=["reference", "random"], default="reference")
	p.add_argument("--seed", type=int, required=True)
	p.add_argument("--out", type=str, required=True)
	p.add_argument("--resume", action="store_true")
	p.add_argument("--log-every", type=int, default=100, help="sweeps between checkpoints")
	p.add_argument("--grad-mode", choices=["one_pass", "site_soft"], default="one_pass")
	p.add_argument("--n-sites", type=int, default=1)
	p.add_argument("--dt-at-t2", type=float, default=63.0)
	p.add_argument("--dt-exp", type=float, default=0.5)
	p.add_argument("--dt-factor", type=float, default=1.0)
	p.add_argument("--ref", type=str, default="zero_polymer")
	p.add_argument("--device", type=str, default="cuda:0")
	p.add_argument("--cs-file", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../CONSTANTS.txt"))
	return p


if __name__ == "__main__":
	main(create_parser().parse_args())
