"""
Analysis of replica-exchange runs (run_pt.py, DEVLOG 2026-10-24). numpy only;
matplotlib optional.

    python analyze_pt.py pt_study/ref pt_study/rand

Give two or more independent ladders with the same temperatures (for example
one started at the reference and one at random sequences); the convergence
check compares them slot by slot. Tables and plots go to the parent directory
of the first run, under analysis/.

Per slot (temperature):
  swap acc    fraction of accepted swaps between this T and the next one,
              pooled over the runs. Below ~0.1 the ladder is too sparse there
              (add a temperature in between); above ~0.6 a rung could go.
  t_eq        first sweep t from which, in the window (t/2, t], R-hat across
              the runs of U and of Hd_to_ref stays below 1.1 to the end. The
              ladder is equilibrated at t_eq = the largest slot t_eq.
  <U>, <Hd>   means over sweeps after the burn-in (t_eq, or --burn, or half
              the run when t_eq is not reached, which the table flags).
  Cv          Var(U)/T^2 (k_B = 1), pooled over the runs, with the spread
              between runs. Its peak locates the transition.
  acc         move acceptance at this T, from the walkers' trajectories.
Mixing:
  round trips walker trips from the lowest T to the highest and back. Few or
              none means states are not crossing the ladder, whatever R-hat says.
Equilibrium sequences for later runs: analysis/final_T<T>_<run>.txt holds the
sequence at each T at the last sweep (run_nsites.py --start takes the file).
"""
import os
import sys
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_nsites import rhat, read_data, read_start, reconstruct, write

try:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
except ImportError:
	plt = None


def read_ladder(run):
	with open(f"{run}/ladder.dat") as f:
		Ts = [float(t) for t in f.readline().split(":")[1].split()]
		keys = f.readline().split()
		rows = {}
		for line in f:
			if line.strip():
				p = line.split()
				rows[int(p[0])] = p
	sweeps = sorted(rows)
	a = np.array([[float(x) for x in rows[s]] for s in sweeps])
	K = len(Ts)
	return {"Ts": Ts, "sweep": a[:, 0].astype(int), "wall": a[:, 1],
		"U": a[:, 2:2 + K], "Hd": a[:, 2 + K:2 + 2 * K], "w": a[:, 2 + 2 * K:2 + 3 * K].astype(int), "run": run}


def read_swaps(run, K):
	tries, accs = np.zeros(K - 1), np.zeros(K - 1)
	path = f"{run}/swaps.dat"
	if os.path.isfile(path):
		d = np.loadtxt(path, skiprows=1, ndmin=2)
		if d.size:
			for k in range(K - 1):
				m = d[:, 1] == k
				tries[k] += m.sum()
				accs[k] += d[m, 4].sum()
	return tries, accs


def round_trips(w):
	"""Completed lowest -> highest -> lowest trips, summed over walkers."""
	K = w.shape[1]
	slot_of = {}
	for t in range(len(w)):
		for k, wid in enumerate(w[t]):
			slot_of.setdefault(wid, []).append(k)
	trips = 0
	for path in slot_of.values():
		state = None   # last extreme visited: 'lo' or 'hi'
		half = 0
		for k in path:
			if k == 0:
				if state == "hi":
					half += 1
				state = "lo"
			elif k == K - 1:
				if state == "lo":
					half += 1
				state = "hi"
		trips += half // 2
	return trips


def acceptance_by_T(run, Ts, burn):
	acc = {T: [] for T in Ts}
	for wdir in glob.glob(f"{run}/walkers/w*"):
		with open(f"{wdir}/data.dat") as f:
			lines = [l.rstrip("\n").split("\t") for l in f if not l.startswith("#")]
		keys, rows = lines[0], lines[2:]
		im, ia, iT = keys.index("move"), keys.index("accepted"), keys.index("T")
		for r in rows:
			if int(r[im]) > burn:
				acc[float(r[iT])].append(int(r[ia]))
	return {T: (np.mean(v) if v else np.nan) for T, v in acc.items()}


def main():
	p = argparse.ArgumentParser()
	p.add_argument("runs", nargs="+")
	p.add_argument("--every", type=int, default=100, help="spacing of the R-hat time grid, in sweeps")
	p.add_argument("--rhat", type=float, default=1.1)
	p.add_argument("--burn", type=int, default=None, help="sweeps to drop; default t_eq, or half the run")
	p.add_argument("--ref", type=str, default="zero_polymer")
	args = p.parse_args()
	runs = [r.rstrip("/") for r in args.runs]
	out_dir = os.path.join(os.path.dirname(runs[0]) or ".", "analysis")

	L = [read_ladder(r) for r in runs]
	Ts = L[0]["Ts"]
	assert all(l["Ts"] == Ts for l in L), "all runs must use the same ladder"
	K, end = len(Ts), min(l["sweep"][-1] for l in L)
	names = [os.path.basename(r) for r in runs]

	# ---- convergence per slot
	grid = list(range(args.every, end + 1, args.every))
	R = {k: [] for k in range(K)}
	for t in grid:
		for k in range(K):
			sel = [(l["sweep"] > t // 2) & (l["sweep"] <= t) for l in L]
			n = min(s.sum() for s in sel)
			RU = rhat([l["U"][s, k][:n] for l, s in zip(L, sel)]) if len(L) > 1 else np.nan
			RH = rhat([l["Hd"][s, k][:n] for l, s in zip(L, sel)]) if len(L) > 1 else np.nan
			R[k].append((t, RU, RH))
	t_eq = {}
	for k in range(K):
		t_eq[k] = None
		for i, (t, _, _) in enumerate(R[k]):
			if all(ru < args.rhat and rh < args.rhat for _, ru, rh in R[k][i:]):
				t_eq[k] = t
				break
	all_eq = all(t_eq[k] for k in range(K)) and len(L) > 1
	t_ladder = max(t_eq.values()) if all_eq else None
	burn = args.burn if args.burn is not None else (t_ladder if t_ladder else end // 2)

	# ---- swaps and mixing
	tries, accs = np.zeros(K - 1), np.zeros(K - 1)
	for r in runs:
		a, b = read_swaps(r, K)
		tries += a
		accs += b
	swap_acc = np.where(tries > 0, accs / np.maximum(tries, 1), np.nan)
	trips = [round_trips(l["w"][l["sweep"] <= end]) for l in L]
	acc_T = [acceptance_by_T(r, Ts, burn) for r in runs]

	# ---- equilibrium table
	lines = [f"# replica exchange: runs {', '.join(names)}; {K} temperatures; {end} sweeps in common",
		f"# burn-in {burn} sweeps ({'ladder t_eq' if args.burn is None and t_ladder else 'given' if args.burn is not None else 'NOT EQUILIBRATED: half the run'})",
		f"# round trips per run: {dict(zip(names, trips))}",
		f"# median wall seconds per sweep: {np.median(np.concatenate([l['wall'][1:] for l in L])):.2f}",
		"k\tT\tt_eq\tRhat_U\tRhat_Hd\t<U>\t<Hd>\tCv\tCv_spread\tacc\tswap_acc(k,k+1)"]
	table = []
	for k, T in enumerate(Ts):
		post = [l["sweep"] > burn for l in L]
		U = np.concatenate([l["U"][m, k] for l, m in zip(L, post)])
		H = np.concatenate([l["Hd"][m, k] for l, m in zip(L, post)])
		cvs = [l["U"][m, k].var() / T ** 2 for l, m in zip(L, post)]
		row = dict(T=T, U=U.mean(), H=H.mean(), Cv=U.var() / T ** 2, Cv_sd=np.std(cvs), acc=np.nanmean([a[T] for a in acc_T]))
		table.append(row)
		_, RU, RH = R[k][-1] if R[k] else (0, np.nan, np.nan)
		lines.append(f"{k}\t{T:g}\t{t_eq[k] or '-'}\t{RU:.3f}\t{RH:.3f}\t{row['U']:.1f}\t{row['H']:.1f}\t{row['Cv']:.2f}\t{row['Cv_sd']:.2f}\t{row['acc']:.3f}\t"
			+ (f"{swap_acc[k]:.3f}" if k < K - 1 else "-"))
	kmax = int(np.argmax([r["Cv"] for r in table]))
	lines.append(f"# Cv peak at T = {Ts[kmax]:g} (Cv {table[kmax]['Cv']:.1f})" + (" -- at the edge of the ladder, extend it" if kmax in (0, K - 1) else ""))
	lines.append(f"# ladder equilibrated: {'yes, t_eq = %d sweeps' % t_ladder if t_ladder else 'no'}")
	for k in range(K - 1):
		if swap_acc[k] < 0.1:
			lines.append(f"# sparse rung: swap acceptance {swap_acc[k]:.3f} between T={Ts[k]:g} and {Ts[k + 1]:g}; add a temperature in between")
	write(out_dir, "pt_summary.tsv", lines)

	# ---- equilibrium sequences at the last sweep, per T and run
	from references import get_reference
	_, ref = get_reference(args.ref)
	for l, r, name in zip(L, runs, names):
		last = np.nonzero(l["sweep"] == end)[0][0]
		for k, T in enumerate(Ts):
			wdir = f"{r}/walkers/w{l['w'][last, k]}"
			d = read_data(f"{wdir}/data.dat")
			d = {key: (v[d["move"] <= end] if isinstance(v, np.ndarray) else [x for x, m in zip(v, d["move"] <= end) if m]) for key, v in d.items()}
			S, moves, bad = reconstruct(d, read_start(wdir), end, ref)
			seq = S[-1].tobytes().decode()
			with open(f"{out_dir}/final_T{T:g}_{name}.txt", "w") as f:
				f.write(f"# T={T:g}, {r}, walker {l['w'][last, k]}, sweep {end}, reconstruction mismatches {bad}\n{seq}\n")

	if plt:
		fig, ax = plt.subplots(2, 2, figsize=(12, 8))
		for l, name in zip(L, names):
			m = l["sweep"] > burn
			ax[0, 0].plot(Ts, l["U"][m].mean(0), "o-", label=name)
			ax[0, 1].plot(Ts, l["Hd"][m].mean(0), "o-", label=name)
			ax[1, 0].plot(Ts, l["U"][m].var(0) / np.array(Ts) ** 2, "o-", label=name)
		ax[1, 1].plot(np.sqrt(np.array(Ts[:-1]) * np.array(Ts[1:])), swap_acc, "o-")
		ax[1, 1].axhline(0.1, color="k", ls=":")
		for a, lab in zip(ax.flat, ["<U>", "<Hd_to_ref>", "Var(U)/T^2", "swap acceptance"]):
			a.set_xscale("log"); a.set_xlabel("T"); a.set_ylabel(lab)
		ax[0, 0].legend(fontsize=8)
		fig.suptitle(f"zero_polymer replica exchange, burn-in {burn} sweeps")
		fig.tight_layout()
		fig.savefig(f"{out_dir}/pt_equilibrium.png", dpi=130)
		plt.close(fig)

		fig, axs = plt.subplots(1, 2, figsize=(14, 4))
		for k in range(K):
			for l, ls in zip(L, ["-", "--", ":", "-."]):
				axs[0].plot(l["sweep"], l["U"][:, k], ls, lw=.5, color=plt.cm.viridis(k / max(1, K - 1)))
			axs[1].plot([t for t, _, _ in R[k]], [ru for _, ru, _ in R[k]], color=plt.cm.viridis(k / max(1, K - 1)), label=f"T={Ts[k]:g}")
		axs[0].set_xlabel("sweep"); axs[0].set_ylabel("U at slot (colour = T, line style = run)")
		axs[1].axhline(args.rhat, color="k", ls=":"); axs[1].set_yscale("log"); axs[1].set_xlabel("sweep"); axs[1].set_ylabel("R-hat U")
		axs[1].legend(fontsize=6, ncol=2)
		fig.tight_layout()
		fig.savefig(f"{out_dir}/pt_traces.png", dpi=130)
		plt.close(fig)


if __name__ == "__main__":
	sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/tests")))
	main()
