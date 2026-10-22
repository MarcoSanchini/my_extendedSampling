"""
One chain of the mutations-per-step study on zero_polymer (DEVLOG 2026-10-21).

The study has two phases, both run through this script (study_nsites.sh
launches them):

  equil  several chains per temperature, from the reference and from random
         sequences, long enough that they agree (analyze_nsites.py equil).
  scan   for each n_sites value, short chains started from the final state of
         the equilibrated chains, so every n_sites begins from the same
         equilibrium sequences.

This is main_rate.py with three additions:

1. Start sequence. --start reference | random | <run dir>. A run dir means
   "the last checkpoint of that run" (eprot/sequence_<move>.pt, or the one at
   --start-move). The sampler only knows ref + init_muts, so the start goes in
   through _mutate_canonical, which _setup calls only when init_muts > 0. The
   reference is still the energy reference. start.txt records the start.
2. Wall-clock time per move. data.dat's time column is process_time(), CPU
   time, which is not the GPU's time. _step is timed with perf_counter()
   between two torch.cuda.synchronize() calls and written to wall.dat
   (move, seconds), flushed at the same checkpoints as data.dat.
3. GPU memory. Peak allocated and reserved memory go to run_info.txt.

pars.txt is written by this script, since the sampler's restart check needs
it. --resume continues the run in --out to a larger --moves, from its last
checkpoint, as main_rate.py --extend does.

Not runnable without ESM3 (the cloud session that wrote it has no torch).

Usage:
    python -u run_nsites.py --T 100 --n-sites 1 --moves 20000 --seed 100 \
        --start reference --out nsites_study/equil/T100/chain0
    python -u run_nsites.py --T 100 --n-sites 8 --moves 5000 --seed 200 \
        --start nsites_study/equil/T100/chain0 --out nsites_study/scan/T100/N8/chain0
"""
import os
import sys
import glob
import time
import argparse

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../customs"))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, CUSTOMS_DIR)
sys.path.append(os.path.join(CODE_DIR, "tests"))

import torch

from classes.rate_sampler import ExtendedProteinRateSampler, _NONCANONICAL_RESIDUES
from utils.general import load_inputs
from references import get_reference, header
import custom_esm.utils.constants.esm3 as C


class TimedSampler(ExtendedProteinRateSampler):
	"""The rate sampler with a chosen start sequence and wall-clock timing.
	_step itself is unchanged; it is only wrapped."""

	start_seq = None
	start_note = ""

	def _mutate_canonical(self, ref_seq: str, mutations: int) -> str:
		return self.start_seq if self.start_seq is not None else super()._mutate_canonical(ref_seq, mutations)

	def _step(self, eprot, U_A, pars):
		cuda = self.generator.device.type == "cuda"
		if cuda:
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		out = super()._step(eprot, U_A, pars)
		if cuda:
			torch.cuda.synchronize()
		self.wall_rows.append(time.perf_counter() - t0)
		return out

	# start.txt is written here, after _setup has cleared the directory of a
	# fresh run (it removes every file but pars.txt).
	def _setup(self, ref_seq, pars, settings=None):
		out = super()._setup(ref_seq, pars, settings)
		if not settings["restart"]:
			eprot = out[0]
			hd0 = sum(a != b for a, b in zip(eprot.sequence, ref_seq))
			with open(f'{settings["results_dir"]}/start.txt', "w") as f:
				f.write(f"# start: {self.start_note}\n# Hd_to_ref: {hd0}\n{eprot.sequence}\n")
			print(f"start: {self.start_note}  (Hd to ref {hd0})")
		return out

	# Checkpoints flush data.dat; wall.dat is flushed with them, so the two
	# files always end at the same move, also after a resume.
	def _save_log(self, eprot, data, settings):
		super()._save_log(eprot, data, settings)
		if self.wall_rows:
			first = data["move"] - len(self.wall_rows) + 1
			with open(f'{settings["results_dir"]}/wall.dat', "a") as f:
				for i, seconds in enumerate(self.wall_rows):
					f.write(f"{first + i}\t{seconds:.6f}\n")
			self.wall_rows = []


def dt_for_T(T, dt_at_t2, dt_exp, factor):
	# Same rule as sweep_temperature_zero_polymer.sh: dt/sqrt(T) fixed.
	return factor * dt_at_t2 * (T / 2.0) ** dt_exp


def last_checkpoint_sequence(run_dir, move=None):
	eprot_dir = os.path.join(run_dir, "eprot")
	if move is None:
		moves = [int(os.path.basename(f)[len("sequence_"):-3]) for f in glob.glob(f"{eprot_dir}/sequence_*.pt")]
		if not moves:
			raise FileNotFoundError(f"no eprot/sequence_*.pt checkpoint in {run_dir}")
		move = max(moves)
	seq = torch.load(f"{eprot_dir}/sequence_{move}.pt", weights_only=False)
	return seq, move


def random_canonical(L, seed):
	canonical = [c for c in C.SEQUENCE_USED_VOCAB if c not in _NONCANONICAL_RESIDUES]
	rng = np.random.default_rng(seed)
	return "".join(canonical[i] for i in rng.integers(0, len(canonical), L))


def write_pars(path, pars):
	with open(path, "w") as f:
		f.write("# parameters (written by run_nsites.py)\n\n## sampler\n")
		for k, v in pars.items():
			f.write(f"{k:<20}{v}\n")


def main(args):
	print(f"PID: {os.getpid()}\n")
	ref_name, ref_seq = get_reference(args.ref)
	print(header(ref_name, ref_seq))

	out = args.out.rstrip("/")
	os.makedirs(f"{out}/eprot", exist_ok=True)
	assert args.moves % args.log_step == 0, (
		f"--moves ({args.moves}) must be a multiple of --log-step ({args.log_step}): "
		"data.dat and wall.dat are only flushed at checkpoints.")

	dt = args.dt if args.dt is not None else dt_for_T(args.T, args.dt_at_t2, args.dt_exp, args.dt_factor)

	if args.resume:
		start_seq, start_note = None, "resume"
	elif args.start == "reference":
		start_seq, start_note = ref_seq, "reference"
	elif args.start == "random":
		start_seq, start_note = random_canonical(len(ref_seq), args.seed), f"random (numpy seed {args.seed})"
	else:
		start_seq, move = last_checkpoint_sequence(args.start, args.start_move)
		start_note = f"{os.path.abspath(args.start)} move {move}"

	pars = {
		"moves": args.moves, "T": args.T, "dt": round(dt, 4), "M": 1.0, "T_sftm": 0.1,
		"lambda_am": 1.0, "lambda_structure_ce": 0.0, "lambda_S": 0.0, "n_quad": 40,
		# init_muts=1 only routes start_seq through _mutate_canonical; no
		# random mutation is drawn. 0 starts at the reference itself.
		"init_muts": 0 if start_seq is None or start_seq == ref_seq else 1,
		"n_sites": args.n_sites, "eps": 1.0e-9, "seed": args.seed,
	}
	if args.resume:
		# The sampler silently starts over (and clears the directory) when a
		# restart file is missing, so check them here, as main_rate.py --extend does.
		missing = [f for f in ["data.dat", "pars.txt", "generator.npy", "log.pt", "start.txt"] if not os.path.isfile(f"{out}/{f}")]
		n_eprot = len(glob.glob(f"{out}/eprot/*.pt"))
		assert not missing and n_eprot >= 4, f"--resume: {out} cannot be restarted (missing {missing}, {n_eprot} eprot files)."
		saved = load_inputs(f"{out}/pars.txt", start="## sampler", end="##")
		pars["init_muts"] = int(saved["init_muts"])
		changed = {k: (saved[k], v) for k, v in pars.items() if k != "moves" and float(saved[k]) != float(v)}
		assert not changed, f"--resume: parameters differ from {out}/pars.txt: {changed}"
	write_pars(f"{out}/pars.txt", pars)

	settings = {
		"results_dir": out, "eprot_dir": f"{out}/eprot",
		"log_step": args.log_step, "print_step": args.print_step,
		"restart": bool(args.resume), "device": args.device, "num_threads": 1,
	}
	sampler = TimedSampler(config_settings=load_inputs(args.cs_file))
	sampler.start_seq = start_seq
	sampler.start_note = start_note
	sampler.wall_rows = []

	t0 = time.perf_counter()
	sampler.sample(ref_seq=ref_seq, pars=pars, settings=settings)
	wall_total = time.perf_counter() - t0

	with open(f"{out}/run_info.txt", "a") as f:
		f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  moves={args.moves}  resume={bool(args.resume)}\n")
		f.write(f"T {args.T}\nn_sites {args.n_sites}\ndt {pars['dt']}\nseed {args.seed}\nstart {start_note}\n")
		f.write(f"wall_total_s {wall_total:.1f}\n")
		if torch.cuda.is_available():
			f.write(f"gpu {torch.cuda.get_device_name(0)}\n")
			f.write(f"peak_allocated_MB {torch.cuda.max_memory_allocated() / 2**20:.0f}\n")
			f.write(f"peak_reserved_MB {torch.cuda.max_memory_reserved() / 2**20:.0f}\n")
		f.write("\n")
	print("\nSimulation completed!")


def create_parser():
	p = argparse.ArgumentParser()
	p.add_argument("--T", type=float, required=True)
	p.add_argument("--n-sites", type=int, default=1)
	p.add_argument("--moves", type=int, required=True)
	p.add_argument("--seed", type=int, required=True)
	p.add_argument("--start", type=str, default="reference",
		help="reference, random, or a run directory whose last checkpoint is the start")
	p.add_argument("--start-move", type=int, default=None, help="checkpoint move to start from, with a run directory")
	p.add_argument("--out", type=str, required=True)
	p.add_argument("--resume", action="store_true", help="continue the run in --out to a larger --moves")
	p.add_argument("--dt", type=float, default=None, help="fixed dt; default is the sweep rule below")
	p.add_argument("--dt-at-t2", type=float, default=63.0)
	p.add_argument("--dt-exp", type=float, default=0.5)
	p.add_argument("--dt-factor", type=float, default=1.0)
	p.add_argument("--log-step", type=int, default=500)
	p.add_argument("--print-step", type=int, default=500)
	p.add_argument("--ref", type=str, default="zero_polymer")
	p.add_argument("--device", type=str, default="cuda:0")
	p.add_argument("--cs-file", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../CONSTANTS.txt"))
	return p


if __name__ == "__main__":
	main(create_parser().parse_args())
