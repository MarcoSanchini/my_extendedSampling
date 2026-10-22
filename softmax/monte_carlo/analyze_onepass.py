"""
Analysis of the one-pass tests T6-T8 (study_onepass.sh, docs/one_pass_move.tex
Section 7). Reads only run_nsites.py's output (data.dat, start.txt, wall.dat,
run_info.txt) plus nvidia-smi logs, so it needs numpy but not torch or ESM3. The
trajectory and decorrelation code is analyze_nsites.py's.

    python analyze_onepass.py bench     onepass_study   # T6: cost per move
    python analyze_onepass.py transient onepass_study   # T7: same dynamics from the reference
    python analyze_onepass.py equil     onepass_study   # T7/T8: equilibrium and speed

Tables go to stdout and to <run dir>/analysis/.

BENCH (T6). Per run: median wall-clock s/move (wall.dat), CPU s/move (data.dat's
  time column over moves), peak allocated GPU memory, and the mean GPU
  utilisation over the samples where it was above 0. Pass: one_pass alone
  <= 0.170 s/move.

TRANSIENT (T7, first half). Same starts (the reference), same seeds, both modes.
  At each checkpoint move: mean and spread over seeds of Hd_to_ref and U, and the
  acceptance up to there. z = (one_pass - site_soft) / sqrt(se1^2 + se2^2), se
  from the spread over seeds. Agreement: |z| < 2 everywhere (a guide, not a test:
  the checkpoints are correlated).

EQUIL (T7 second half, T8). Both modes from the same equilibrated states.
  a, <U>, <Hd_to_ref>, Var(U)/T^2, site entropy S (mean over sites of the entropy
  of the residue histogram, pooled over chains), q (mean fraction of identical
  sites between snapshots of different chains), each with z between the modes;
  the error of a mean is sd*sqrt(tau_int/n). Then T8: tau_q (analyze_nsites.py's
  decorrelation lag, with D_inf from snapshots of different chains of both modes
  pooled, since both sample the same distribution), s/move, s/decorr = tau_q *
  s/move, and speedup = site_soft s/decorr / one_pass s/decorr.
"""
import os
import sys
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "../code/tests"))

from analyze_nsites import (load_run, tau_int, lag_curve, cross_hd, crossing, E_FOLD,  # noqa: E402
                            T_of, fmt, write, read_data, read_info)

MODES = ("site_soft", "one_pass")


def runs_of(path, stride, ref):
	return [load_run(r, stride, ref) for r in sorted(glob.glob(f"{path}/chain*")) if os.path.isfile(f"{r}/data.dat")]


def zscore(m1, s1, m2, s2):
	d = np.sqrt(s1**2 + s2**2)
	return (m2 - m1)/d if d > 0 else (0. if m1 == m2 else np.inf)


def mean_se(x):
	"""Mean of a correlated series and its error, sd*sqrt(tau_int/n)."""
	x = np.asarray(x, dtype=float)
	if len(x) < 3 or np.allclose(x, x[0]):
		return float(np.mean(x)), 0.
	tau, _ = tau_int(x)
	tau = max(1., tau) if np.isfinite(tau) else len(x)
	return float(x.mean()), float(x.std(ddof=1)*np.sqrt(tau/len(x)))


def site_entropy(snaps):
	"""Mean over sites of the entropy of the residue histogram (nats)."""
	S = []
	for i in range(snaps.shape[1]):
		_, counts = np.unique(snaps[:, i], return_counts=True)
		p = counts/counts.sum()
		S.append(-(p*np.log(p)).sum())
	return float(np.mean(S))


# -------------------------------------------------------------------- bench --- #

def smi_util(path):
	if not os.path.isfile(path):
		return np.nan, np.nan
	util, mem = [], []
	for line in open(path):
		parts = [p.strip() for p in line.split(",")]
		if len(parts) >= 3:
			try:
				util.append(float(parts[1])); mem.append(float(parts[2]))
			except ValueError:
				pass
	util = np.array(util)
	busy = util[util > 0]
	return (float(busy.mean()) if len(busy) else np.nan), (float(max(mem)) if mem else np.nan)


def cpu_per_move(run):
	d = {}
	with open(f"{run}/data.dat") as f:
		keys = None
		for line in f:
			if line.startswith("#") or not line.strip():
				continue
			parts = line.rstrip("\n").split("\t")
			if keys is None:
				keys = parts
				continue
			d[int(parts[0])] = float(parts[keys.index("time")])
	moves = sorted(d)
	return (d[moves[-1]] - d[moves[0]])/(moves[-1] - moves[0]) if len(moves) > 1 else np.nan


def analyze_bench(args, ref):
	B = f"{args.run_dir}/bench"
	lines = ["# T6 cost per move (analyze_onepass.py bench)",
		"run\tmoves\twall_s/move\tcpu_s/move\tpeak_alloc_MB\tsmi_mem_MB\tgpu_util_%"]
	res = {}
	for run in sorted(glob.glob(f"{B}/*")):
		if not os.path.isfile(f"{run}/data.dat"):
			continue
		name = os.path.basename(run)
		wall = load_run(run, 10, ref)["wall"]
		w = float(np.median(wall[1:])) if len(wall) > 1 else np.nan   # the first move may refill the cache
		info = read_info(run)
		smi = f"{B}/smi_{name}.csv" if "pair" not in name else f"{B}/smi_one_pass_pair.csv"
		util, mem = smi_util(smi)
		res[name] = w
		lines.append(f"{name}\t{len(wall)}\t{w:.4f}\t{cpu_per_move(run):.4f}\t{info.get('peak_allocated_MB', '-')}\t{mem:.0f}\t{util:.0f}")
	if "site_soft_alone" in res and "one_pass_alone" in res:
		lines.append(f"# speed-up per move, alone: {res['site_soft_alone']/res['one_pass_alone']:.2f}x "
			f"(predicted 361/150 = 2.4x)")
	if "one_pass_alone" in res:
		ok = res["one_pass_alone"] <= 0.170
		lines.append(f"# T6: one_pass alone {res['one_pass_alone']:.3f} s/move -> {'PASS' if ok else 'FAIL'} (criterion <= 0.170)")
	pair = [v for k, v in res.items() if "pair" in k]
	if pair and "one_pass_alone" in res:
		lines.append(f"# two one_pass jobs together: {np.mean(pair):.3f} s/move each, i.e. a move every "
			f"{np.mean(pair)/2:.3f} s vs {res['one_pass_alone']:.3f} s alone")
	write(f"{args.run_dir}/analysis", "bench.tsv", lines)


# ---------------------------------------------------------------- transient --- #

def analyze_transient(args, ref):
	lines = ["# T7 transient from the reference (analyze_onepass.py transient)",
		"T\tmove\tquantity\tsite_soft\tsd\tone_pass\tsd\tz"]
	worst = {}
	for Tdir in sorted(glob.glob(f"{args.run_dir}/transient/T*"), key=T_of):
		T = T_of(Tdir)
		byM = {m: runs_of(f"{Tdir}/{m}", args.stride, ref) for m in MODES}
		if not all(byM.values()):
			print(f"T={T}: a mode has no runs, skipped")
			continue
		end = min(r["move"][-1] for rs in byM.values() for r in rs)
		checkpoints = [m for m in args.checkpoints if m <= end] or [end]
		for m in checkpoints:
			for q in ("Hd", "U", "acc"):
				stats = []
				for mode in MODES:
					vals = []
					for r in byM[mode]:
						i = int(np.searchsorted(r["move"], m))
						vals.append(r[q][i] if q != "acc" else r["acc"][1:i + 1].mean())
					vals = np.array(vals, dtype=float)
					sd = vals.std(ddof=1) if len(vals) > 1 else np.nan
					stats.append((vals.mean(), sd, sd/np.sqrt(len(vals)) if len(vals) > 1 else np.nan))
				z = zscore(stats[0][0], stats[0][2], stats[1][0], stats[1][2])
				worst[(T, q)] = max(worst.get((T, q), 0.), abs(z) if np.isfinite(z) else 0.)
				lines.append(f"{T:g}\t{m}\t{q}\t{stats[0][0]:.3f}\t{stats[0][1]:.3f}\t{stats[1][0]:.3f}\t{stats[1][1]:.3f}\t{z:+.2f}")
		spm = {mode: np.median(np.concatenate([r["wall"] for r in byM[mode]])) for mode in MODES}
		lines.append(f"# T={T:g}: {len(byM['site_soft'])} vs {len(byM['one_pass'])} seeds, s/move {spm['site_soft']:.3f} vs "
			f"{spm['one_pass']:.3f} ({spm['site_soft']/spm['one_pass']:.2f}x); "
			+ ", ".join(f"max |z| {q} {worst[(T, q)]:.1f}" for q in ("Hd", "U", "acc")))
	write(f"{args.run_dir}/analysis", "transient.tsv", lines)


# -------------------------------------------------------------------- equil --- #

def analyze_equil(args, ref):
	lines = ["# T7 equilibrium statistics and T8 speed (analyze_onepass.py equil)"]
	speed = ["T\tmode\tchains\tmoves\ta\ttau_U\tD_inf\ttau_q\ts/move\ts/decorr\tspeedup"]
	for Tdir in sorted(glob.glob(f"{args.run_dir}/equil/T*"), key=T_of):
		T = T_of(Tdir)
		byM = {m: runs_of(f"{Tdir}/{m}", args.stride, ref) for m in MODES}
		if not all(byM.values()):
			print(f"T={T}: a mode has no runs, skipped")
			continue
		for runs in byM.values():
			for r in runs:
				keep = r["move"] > args.burn
				r["U_eq"], r["Hd_eq"], r["acc_eq"] = r["U"][keep], r["Hd"][keep], r["acc"][keep]
				r["snaps_eq"] = r["snaps"][r["snap_moves"] >= args.burn]
				r["chain"] = os.path.basename(r["run"])

		lines.append(f"# T={T:g}")
		lines.append("quantity\tsite_soft\terr\tone_pass\terr\tz")
		stats = {}
		for mode, runs in byM.items():
			U = [mean_se(r["U_eq"]) for r in runs]
			H = [mean_se(r["Hd_eq"]) for r in runs]
			comb = lambda xs: (float(np.mean([m for m, _ in xs])), float(np.sqrt(np.sum([s**2 for _, s in xs]))/len(xs)))
			allU = np.concatenate([r["U_eq"] for r in runs])
			v = allU.var()/T**2
			v_err = np.std([r["U_eq"].var()/T**2 for r in runs], ddof=1)/np.sqrt(len(runs)) if len(runs) > 1 else np.nan
			snaps = np.concatenate([r["snaps_eq"] for r in runs])
			S = site_entropy(snaps)
			S_err = np.std([site_entropy(r["snaps_eq"]) for r in runs], ddof=1)/np.sqrt(len(runs)) if len(runs) > 1 else np.nan
			chains = [r["snaps_eq"] for r in runs]
			q = 1. - cross_hd(chains)/snaps.shape[1] if len(chains) > 1 else np.nan
			a = float(np.mean(np.concatenate([r["acc_eq"] for r in runs])))
			n = sum(len(r["acc_eq"]) for r in runs)
			stats[mode] = {"a": (a, np.sqrt(a*(1 - a)/n)), "<U>": comb(U), "<Hd_to_ref>": comb(H),
				"Var(U)/T^2": (v, v_err), "site entropy": (S, S_err), "q": (q, np.nan)}
		for k in stats["site_soft"]:
			(m1, e1), (m2, e2) = stats["site_soft"][k], stats["one_pass"][k]
			z = zscore(m1, e1, m2, e2) if np.isfinite(e1) and np.isfinite(e2) else np.nan
			lines.append(f"{k}\t{m1:.4f}\t{fmt(e1, p=4)}\t{m2:.4f}\t{fmt(e2, p=4)}\t{fmt(z, p=2)}")

		groups = [r["snaps_eq"] for runs in byM.values() for r in runs]
		# D_inf from different starts only: chains with the same index share a start.
		by_chain = {}
		for runs in byM.values():
			for r in runs:
				by_chain.setdefault(r["chain"], []).append(r["snaps_eq"])
		D_inf = cross_hd([np.concatenate(v) for v in by_chain.values()]) if len(by_chain) > 1 else np.nan
		rows = {}
		for mode, runs in byM.items():
			n_snap = min(len(r["snaps_eq"]) for r in runs)
			D = []
			for r in runs:
				lags, d = lag_curve(r["snaps_eq"][:n_snap])
				D.append(d)
			D = np.mean(D, axis=0)
			tq, ok = crossing(lags*args.stride, D, E_FOLD*D_inf) if np.isfinite(D_inf) else (np.nan, False)
			spm = float(np.median(np.concatenate([r["wall"] for r in runs])))
			tU = np.mean([tau_int(r["U_eq"])[0] for r in runs])
			rows[mode] = dict(chains=len(runs), moves=min(r["move"][-1] for r in runs), a=stats[mode]["a"][0],
				tU=tU, tq=tq, ok=ok, spm=spm, sdec=tq*spm)
		for mode in MODES:
			r = rows[mode]
			sp = rows["site_soft"]["sdec"]/r["sdec"] if r["ok"] and rows["site_soft"]["ok"] else np.nan
			speed.append(f"{T:g}\t{mode}\t{r['chains']}\t{r['moves']}\t{r['a']:.3f}\t{r['tU']:.1f}\t{D_inf:.1f}\t"
				f"{fmt(r['tq'], r['ok'])}\t{r['spm']:.3f}\t{fmt(r['sdec'], r['ok'])}\t{fmt(sp, p=2)}")
	write(f"{args.run_dir}/analysis", "equil_stats.tsv", lines)
	print()
	speed.append("# speedup = site_soft s/decorr / one_pass s/decorr; predicted ~2.4 (docs/one_pass_move.tex Section 6)")
	write(f"{args.run_dir}/analysis", "equil_speed.tsv", speed)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("phase", choices=["bench", "transient", "equil"])
	p.add_argument("run_dir", help="RUN_DIR of study_onepass.sh (default there: monte_carlo/onepass_study)")
	p.add_argument("--stride", type=int, default=10, help="moves between reconstructed snapshots")
	p.add_argument("--checkpoints", type=int, nargs="+", default=[250, 500, 1000, 2000, 3000, 5000],
		help="transient: moves at which the modes are compared")
	p.add_argument("--burn", type=int, default=0, help="equil: moves dropped at the start of each chain")
	p.add_argument("--ref", type=str, default="zero_polymer")
	args = p.parse_args()
	args.run_dir = args.run_dir.rstrip("/")
	from references import get_reference
	_, ref = get_reference(args.ref)
	{"bench": analyze_bench, "transient": analyze_transient, "equil": analyze_equil}[args.phase](args, ref)


if __name__ == "__main__":
	main()
