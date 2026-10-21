"""
Analysis of the mutations-per-step study (study_nsites.sh, DEVLOG 2026-10-21).
Reads only data.dat, start.txt, wall.dat and run_info.txt, so it needs numpy
but not torch or ESM3. matplotlib is optional (plots are skipped without it).

    python analyze_nsites.py equil nsites_study     # are the chains equilibrated?
    python analyze_nsites.py scan  nsites_study     # which n_sites is fastest?

Both write tables to stdout and to <run dir>/analysis/.

Trajectories. data.dat logs every move: the site(s), the residues before
(cur_aa) and the proposed ones (proposed_aa), and whether it was accepted.
With start.txt that gives the full sequence at every move, so Hamming distances
between any two times are exact, without needing a checkpoint at each time.
The reconstruction is checked against cur_aa (each move's "before" residues)
and against the logged Hd_to_ref.

EQUIL. For each T, on a grid of times t, the window (t/2, t] of every chain:
  Rhat_U, Rhat_Hd  Gelman-Rubin potential scale reduction across chains. 1 when
                   the chains sample the same distribution. Threshold 1.1.
  dHd/dt           slope of Hd_to_ref in the window, per 1000 moves, mean over
                   chains, with its standard error from the spread over chains.
  rms_Hd0          RMS over chains of Hd(x_0, x_t), the distance of each chain
                   from its own start, and v_rms, its rate per 1000 moves. The
                   "velocity of divergence"; it goes to 0 when the chains have
                   forgotten their starts, but also when they are frozen, which
                   is why it is not used alone.
  t_eq             first t from which Rhat_U and Rhat_Hd stay below 1.1 to the
                   end, with |dHd/dt| within 2 standard errors of 0 at t_eq.

SCAN. For each T and n_sites (N), pooled over the chains:
  a           acceptance, fraction of moves accepted.
  N*a         site changes accepted per move (each accepted move changes
              exactly N sites, since the current residue is never a target).
  D(tau)      mean Hd(x_t, x_{t+tau}) over t. Grows from 0 and saturates at
  D_inf       the Hd between independent equilibrium sequences, estimated from
              pairs of snapshots of chains with different starts (different
              equilibrated chains), pooled over N, since all N sample the same
              distribution.
  tau_q       lag at which D(tau) = (1 - 1/e) D_inf: the moves needed to
              decorrelate the sequence. '>' marks a lower bound (not reached).
  tau_U       integrated autocorrelation time of U, in moves (Sokal window,
              c=5), mean over chains. '>' when the window did not close.
  s/move      median wall-clock seconds per move (wall.dat).
  s/decorr    tau_q * s/move: wall-clock seconds per independent sequence. The
              figure of merit; smaller is better.
  speedup     s/decorr of N=1 divided by that of N.
"""
import os
import re
import sys
import glob
import argparse

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/tests")))

try:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
except ImportError:
	plt = None

E_FOLD = 1. - np.exp(-1.)


# ---------------------------------------------------------------- reading --- #

def read_data(path):
	"""data.dat as a dict of columns, one entry per move. A move logged twice
	(possible around a restart) keeps its last row."""
	rows, keys = {}, None
	with open(path) as f:
		for line in f:
			if line.startswith("#") or not line.strip():
				continue
			parts = line.rstrip("\n").split("\t")
			if keys is None:
				keys = parts
				continue
			rows[int(parts[0])] = parts
	moves = sorted(rows)
	col = {k: [rows[m][i] for m in moves] for i, k in enumerate(keys)}
	return {
		"move": np.array(moves),
		"U": np.array(col["U"], dtype=float),
		"Hd": np.array(col["Hd_to_ref"], dtype=int),
		"site": col["site"], "cur": col["cur_aa"], "prop": col["proposed_aa"],
		"acc": np.array([int(x) if x else 0 for x in col["accepted"]]),
	}


def read_start(run):
	with open(f"{run}/start.txt") as f:
		return [l.strip() for l in f if l.strip() and not l.startswith("#")][-1]


def read_wall(run):
	path = f"{run}/wall.dat"
	if not os.path.isfile(path):
		return np.array([])
	w = {}
	with open(path) as f:
		for line in f:
			m, s = line.split()
			w[int(m)] = float(s)
	return np.array([w[m] for m in sorted(w)])


def read_info(run):
	info = {}
	path = f"{run}/run_info.txt"
	if os.path.isfile(path):
		with open(path) as f:
			for line in f:
				if line.startswith("#") or not line.strip():
					continue
				k, _, v = line.strip().partition(" ")
				info[k] = v   # the last entry wins, i.e. the latest (resumed) segment
	return info


def reconstruct(d, start, stride, ref=None):
	"""Sequences (as uint8 codes) at moves 0, stride, 2*stride, ... and the
	number of inconsistencies found on the way (should be 0)."""
	state = np.frombuffer(start.encode(), dtype=np.uint8).copy()
	ref_arr = np.frombuffer(ref.encode(), dtype=np.uint8) if ref else None
	snaps, snap_moves, bad = [], [], 0
	for i, m in enumerate(d["move"]):
		if m > 0:
			sites = [int(s) for s in d["site"][i].split(",")]
			cur = d["cur"][i].split(",")
			if any(state[s] != ord(c) for s, c in zip(sites, cur)):
				bad += 1
			if d["acc"][i]:
				for s, a in zip(sites, d["prop"][i].split(",")):
					state[s] = ord(a)
		if m % stride == 0:
			snaps.append(state.copy())
			snap_moves.append(m)
			if ref_arr is not None and int((state != ref_arr).sum()) != d["Hd"][i]:
				bad += 1
	return np.array(snaps), np.array(snap_moves), bad


def load_run(run, stride, ref):
	d = read_data(f"{run}/data.dat")
	d["run"] = run
	d["snaps"], d["snap_moves"], d["bad"] = reconstruct(d, read_start(run), stride, ref)
	d["wall"] = read_wall(run)
	d["info"] = read_info(run)
	if d["bad"]:
		print(f"WARNING: {run}: {d['bad']} moves where the reconstructed sequence disagrees with data.dat")
	return d


def T_of(path):
	return float(re.search(r"T([0-9.eE+-]+)", os.path.basename(path)).group(1))


# --------------------------------------------------------------- statistics --- #

def rhat(chains):
	"""Gelman-Rubin R-hat for equal-length chains (rows)."""
	x = np.asarray(chains, dtype=float)
	m, n = x.shape
	if n < 2 or m < 2:
		return np.nan
	W = x.var(axis=1, ddof=1).mean()
	B = n * x.mean(axis=1).var(ddof=1)
	if W == 0:
		return 1. if B == 0 else np.inf
	return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def autocorr(x):
	x = np.asarray(x, dtype=float) - np.mean(x)
	n = len(x)
	f = np.fft.rfft(x, 2 * n)
	acf = np.fft.irfft(f * np.conj(f))[:n]
	return acf / acf[0] if acf[0] > 0 else np.ones(n)


def tau_int(x, c=5.):
	"""Integrated autocorrelation time with Sokal's window. Returns (tau,
	closed); closed is False when the window reached the end of the series,
	so tau is a lower bound."""
	rho = autocorr(x)
	if np.allclose(rho, 1.):
		return np.inf, False
	taus = 2. * np.cumsum(rho) - 1.
	for W in range(1, len(rho)):
		if W >= c * taus[W]:
			return float(taus[W]), True
	return float(taus[-1]), False


def lag_curve(snaps, max_lag=None):
	"""D(k) = mean Hd between snapshots k apart, for k on a log grid."""
	n = len(snaps)
	max_lag = max_lag or n // 2
	lags = np.unique(np.round(np.geomspace(1, max(1, max_lag), 40)).astype(int))
	return lags, np.array([(snaps[:-k] != snaps[k:]).sum(axis=1).mean() for k in lags])


def cross_hd(groups, n_sub=150, seed=0):
	"""Mean Hd between snapshots of different groups (independent chains)."""
	rng = np.random.default_rng(seed)
	subs = [g[rng.choice(len(g), min(n_sub, len(g)), replace=False)] for g in groups if len(g)]
	vals = []
	for i in range(len(subs)):
		for j in range(i + 1, len(subs)):
			vals.append((subs[i][:, None, :] != subs[j][None, :, :]).sum(axis=2).mean())
	return float(np.mean(vals)) if vals else np.nan


def crossing(lags, D, level):
	"""First lag where D reaches level, interpolated in log-lag. (lag, reached)."""
	above = np.nonzero(D >= level)[0]
	if not len(above):
		return float(lags[-1]), False
	i = above[0]
	if i == 0:
		return float(lags[0]), True
	l0, l1 = np.log(lags[i - 1]), np.log(lags[i])
	f = (level - D[i - 1]) / (D[i] - D[i - 1])
	return float(np.exp(l0 + f * (l1 - l0))), True


def fmt(x, closed=True, p=1):
	if x is None or (isinstance(x, float) and not np.isfinite(x)):
		return "inf" if x == np.inf else "-"
	return ("" if closed else ">") + f"{x:.{p}f}"


def write(out_dir, name, lines):
	os.makedirs(out_dir, exist_ok=True)
	with open(f"{out_dir}/{name}", "w") as f:
		f.write("\n".join(lines) + "\n")
	print("\n".join(lines))


# ------------------------------------------------------------------- equil --- #

def analyze_equil(args, ref):
	out_dir = f"{args.run_dir}/analysis"
	summary = ["# equilibration summary (analyze_nsites.py equil)",
		"T\tchains\tmoves\tt_eq\t<U>\t<Hd_to_ref>\tVar(U)/T^2\tacc\ts/move\tpeak_MB"]
	for Tdir in sorted(glob.glob(f"{args.run_dir}/equil/T*"), key=T_of):
		T = T_of(Tdir)
		runs = [load_run(r, args.stride, ref) for r in sorted(glob.glob(f"{Tdir}/chain*")) if os.path.isfile(f"{r}/data.dat")]
		if len(runs) < 2:
			print(f"T={T}: fewer than 2 chains with data, skipped")
			continue
		end = min(r["move"][-1] for r in runs)
		every = args.every
		lines = [f"# T={T}: {len(runs)} chains, common length {end} moves",
			"# starts: " + ", ".join(f"{os.path.basename(r['run'])}={'ref' if r['Hd'][0] == 0 else 'Hd%d' % r['Hd'][0]}" for r in runs),
			"t\tRhat_U\tRhat_Hd\tdHd/dt\tse\trms_Hd0\tv_rms\tmean_U\tmean_Hd"]
		table, prev_rms = [], None
		for t in range(every, end + 1, every):
			lo = t // 2
			Us = [r["U"][(r["move"] > lo) & (r["move"] <= t)] for r in runs]
			Hs = [r["Hd"][(r["move"] > lo) & (r["move"] <= t)] for r in runs]
			n = min(len(u) for u in Us)
			Us, Hs = [u[:n] for u in Us], [h[:n] for h in Hs]
			ms = (np.arange(n) + lo + 1) / 1000.
			slopes = [np.polyfit(ms, h, 1)[0] if n > 2 else np.nan for h in Hs]
			slope, se = float(np.mean(slopes)), float(np.std(slopes, ddof=1) / np.sqrt(len(slopes)))
			k = t // args.stride
			rms = float(np.sqrt(np.mean([((r["snaps"][k] != r["snaps"][0]).sum()) ** 2 for r in runs])))
			v_rms = (rms - prev_rms) / (every / 1000.) if prev_rms is not None else np.nan
			prev_rms = rms
			row = dict(t=t, RU=rhat(Us), RH=rhat(Hs), slope=slope, se=se, rms=rms, v=v_rms,
				U=float(np.mean(Us)), H=float(np.mean(Hs)))
			table.append(row)
			lines.append(f"{t}\t{row['RU']:.3f}\t{row['RH']:.3f}\t{slope:.2f}\t{se:.2f}\t{rms:.1f}\t{fmt(v_rms, p=2)}\t{row['U']:.1f}\t{row['H']:.1f}")

		t_eq = None
		for i, row in enumerate(table):
			ok_after = all(r["RU"] < args.rhat and r["RH"] < args.rhat for r in table[i:])
			if ok_after and abs(row["slope"]) <= 2 * row["se"]:
				t_eq = row["t"]
				break
		lines.append(f"# t_eq = {t_eq if t_eq else 'not reached (extend with RESUME=1 and a larger MOVES_EQ)'}")
		write(out_dir, f"equil_T{T:g}.tsv", lines)

		if t_eq:
			U = np.concatenate([r["U"][(r["move"] > t_eq) & (r["move"] <= end)] for r in runs])
			H = np.concatenate([r["Hd"][(r["move"] > t_eq) & (r["move"] <= end)] for r in runs])
			acc = np.mean(np.concatenate([r["acc"][(r["move"] > t_eq) & (r["move"] <= end)] for r in runs]))
			stats = f"{U.mean():.1f}\t{H.mean():.1f}\t{U.var() / T**2:.4f}\t{acc:.3f}"
		else:
			stats = "-\t-\t-\t-"
		walls = np.concatenate([r["wall"] for r in runs]) if any(len(r["wall"]) for r in runs) else np.array([np.nan])
		peak = max((float(r["info"].get("peak_allocated_MB", "nan")) for r in runs), default=np.nan)
		summary.append(f"{T:g}\t{len(runs)}\t{end}\t{t_eq or '-'}\t{stats}\t{np.median(walls):.3f}\t{peak:.0f}")

		if plt:
			fig, ax = plt.subplots(1, 3, figsize=(15, 4))
			for r in runs:
				ax[0].plot(r["move"], r["Hd"], lw=.7, label=os.path.basename(r["run"]))
				ax[1].plot(r["move"], r["U"], lw=.7)
			ts = [row["t"] for row in table]
			ax[2].plot(ts, [row["RU"] for row in table], label="R-hat U")
			ax[2].plot(ts, [row["RH"] for row in table], label="R-hat Hd")
			ax[2].axhline(args.rhat, color="k", ls=":")
			ax[2].set_yscale("log")
			for a in ax:
				if t_eq:
					a.axvline(t_eq, color="k", ls="--", lw=.8)
				a.set_xlabel("move")
			ax[0].set_ylabel("Hd_to_ref"); ax[1].set_ylabel("U"); ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
			fig.suptitle(f"zero_polymer equilibration, T={T:g}")
			fig.tight_layout()
			fig.savefig(f"{out_dir}/equil_T{T:g}.png", dpi=130)
			plt.close(fig)
	print()
	write(out_dir, "equil_summary.tsv", summary)


# -------------------------------------------------------------------- scan --- #

def analyze_scan(args, ref):
	out_dir = f"{args.run_dir}/analysis"
	header = "T\tN\tchains\tmoves\ta\tN*a\ttau_U\ttau_Hd\tD_inf\ttau_q\ts/move\ts/decorr\tspeedup\tN*a/s\tpeak_MB"
	summary = ["# n_sites scan summary (analyze_nsites.py scan); see the module docstring for the columns", header]
	for Tdir in sorted(glob.glob(f"{args.run_dir}/scan/T*"), key=T_of):
		T = T_of(Tdir)
		byN = {}
		for Ndir in glob.glob(f"{Tdir}/N*"):
			N = int(os.path.basename(Ndir)[1:])
			runs = [load_run(r, args.stride, ref) for r in sorted(glob.glob(f"{Ndir}/chain*")) if os.path.isfile(f"{r}/data.dat")]
			for r in runs:
				r["chain"] = os.path.basename(r["run"])
				keep = r["move"] > args.burn
				for k in ("U", "Hd", "acc"):
					r[k + "_eq"] = r[k][keep]
				r["snaps_eq"] = r["snaps"][r["snap_moves"] >= args.burn]
			if runs:
				byN[N] = runs
		if not byN:
			continue

		# D_inf from different chains (different equilibrated starts), all N pooled.
		chains = sorted({r["chain"] for runs in byN.values() for r in runs})
		groups = [np.concatenate([r["snaps_eq"] for runs in byN.values() for r in runs if r["chain"] == c]) for c in chains]
		D_inf = cross_hd(groups)
		if not np.isfinite(D_inf):
			print(f"T={T}: D_inf needs at least two chains with different starts; tau_q not computed")

		rows, curves = [], {}
		for N in sorted(byN):
			runs = byN[N]
			a = float(np.mean(np.concatenate([r["acc_eq"] for r in runs])))
			tU = [tau_int(r["U_eq"]) for r in runs]
			tH = [tau_int(r["Hd_eq"]) for r in runs]
			n_snap = min(len(r["snaps_eq"]) for r in runs)
			lags, D = None, []
			for r in runs:
				lags, d = lag_curve(r["snaps_eq"][:n_snap])
				D.append(d)
			D = np.mean(D, axis=0)
			lags_moves = lags * args.stride
			curves[N] = (lags_moves, D)
			tq, tq_ok = crossing(lags_moves, D, E_FOLD * D_inf) if np.isfinite(D_inf) else (np.nan, False)
			walls = np.concatenate([r["wall"] for r in runs]) if any(len(r["wall"]) for r in runs) else np.array([np.nan])
			spm = float(np.median(walls))
			peak = max((float(r["info"].get("peak_allocated_MB", "nan")) for r in runs), default=np.nan)
			rows.append(dict(N=N, chains=len(runs), moves=min(r["move"][-1] for r in runs), a=a,
				tU=np.mean([t for t, _ in tU]), tU_ok=all(ok for _, ok in tU),
				tH=np.mean([t for t, _ in tH]), tH_ok=all(ok for _, ok in tH),
				tq=tq, tq_ok=tq_ok, spm=spm, sdec=tq * spm, peak=peak))

		base = next((r["sdec"] for r in rows if r["N"] == 1), np.nan)
		for r in rows:
			speed = base / r["sdec"] if r["tq_ok"] else np.nan
			summary.append("\t".join([f"{T:g}", str(r["N"]), str(r["chains"]), str(r["moves"]),
				f"{r['a']:.3f}", f"{r['N'] * r['a']:.3f}",
				fmt(r["tU"], r["tU_ok"]), fmt(r["tH"], r["tH_ok"]), f"{D_inf:.1f}",
				fmt(r["tq"], r["tq_ok"]), f"{r['spm']:.3f}", fmt(r["sdec"], r["tq_ok"]),
				fmt(speed, p=2), f"{r['N'] * r['a'] / r['spm']:.2f}", f"{r['peak']:.0f}"]))
		ok = [r for r in rows if r["tq_ok"]]
		if ok:
			best = min(ok, key=lambda r: r["sdec"])
			summary.append(f"# T={T:g}: fastest N = {best['N']} ({best['sdec']:.0f} s per decorrelated sequence, acceptance {best['a']:.2f})")
		best_na = max(rows, key=lambda r: r["N"] * r["a"] / r["spm"])
		summary.append(f"# T={T:g}: N with most accepted site changes per second = {best_na['N']}")

		if plt:
			fig, ax = plt.subplots(1, 3, figsize=(15, 4))
			Ns = [r["N"] for r in rows]
			for N, (lm, D) in curves.items():
				spm = next(r["spm"] for r in rows if r["N"] == N)
				ax[0].plot(lm * spm, D / D_inf, label=f"N={N}")
			ax[0].axhline(E_FOLD, color="k", ls=":")
			ax[0].set_xscale("log"); ax[0].set_xlabel("wall-clock lag [s]"); ax[0].set_ylabel("D(lag) / D_inf"); ax[0].legend(fontsize=7)
			ax[1].plot(Ns, [r["a"] for r in rows], "o-", label="acceptance a")
			ax[1].plot(Ns, [r["N"] * r["a"] for r in rows], "s-", label="N*a")
			ax[1].set_xscale("log", base=2); ax[1].set_yscale("log"); ax[1].set_xlabel("N (sites per move)"); ax[1].legend(fontsize=7)
			ax[2].plot(Ns, [r["sdec"] for r in rows], "o-")
			for r in rows:
				if not r["tq_ok"]:
					ax[2].annotate("lower bound", (r["N"], r["sdec"]), fontsize=7)
			ax[2].set_xscale("log", base=2); ax[2].set_yscale("log"); ax[2].set_xlabel("N"); ax[2].set_ylabel("s per decorrelated sequence")
			fig.suptitle(f"zero_polymer n_sites scan, T={T:g}")
			fig.tight_layout()
			os.makedirs(out_dir, exist_ok=True)
			fig.savefig(f"{out_dir}/scan_T{T:g}.png", dpi=130)
			plt.close(fig)
	write(out_dir, "scan_summary.tsv", summary)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("phase", choices=["equil", "scan"])
	p.add_argument("run_dir", help="RUN_DIR of study_nsites.sh (default there: monte_carlo/nsites_study)")
	p.add_argument("--stride", type=int, default=10, help="moves between reconstructed snapshots")
	p.add_argument("--every", type=int, default=500, help="equil: spacing of the time grid, in moves")
	p.add_argument("--rhat", type=float, default=1.1, help="equil: R-hat threshold")
	p.add_argument("--burn", type=int, default=0, help="scan: moves dropped at the start of each chain")
	p.add_argument("--ref", type=str, default="zero_polymer", help="reference name, for the Hd_to_ref check")
	args = p.parse_args()
	args.run_dir = args.run_dir.rstrip("/")

	from references import get_reference
	_, ref = get_reference(args.ref)
	if args.every % args.stride:
		sys.exit("--every must be a multiple of --stride")
	(analyze_equil if args.phase == "equil" else analyze_scan)(args, ref)


if __name__ == "__main__":
	main()
