"""
Structural metrics and site conservation for the zero_polymer chains of the
n_sites study (nsites_study/equil_op, equil_lowT), the zero_polymer version of
what run_plotting_suite.sh did for the protein G sweeps. Driven by
structure_zero_polymer.sh, which first runs compute_structural_metrics.py on
every chain (the expensive, GPU step) and then the two subcommands below.

  decode-ref  decodes the reference and --n-random uniformly random sequences
              once, as baselines for the box plots (GPU, same decode as
              compute_structural_metrics.py).
  plot        no model calls. Per T label:
                - box plots of Rg_ca, Rg_allatom, mean pLDDT (from each chain's
                  structural_metrics.csv) and U, Hd_to_ref (data.dat at the
                  post-burn-in checkpoint moves), with the baselines;
                - the same metrics against T (median and interquartile range);
                - per-site entropy over the post-burn-in checkpoints, K-sites
                  as in identify_k_sites.py (entropy < --k-site-threshold at
                  every --cold label), and the comparison with the natural
                  per-site entropy of the H3N2 alignment (--h3n2): Spearman
                  rank correlation, and the overlap between the most
                  conserved sites of the two, with its hypergeometric p-value.

zero_polymer is an H3N2 hemagglutinin HA0 precursor (code/tests/references.py),
so the H3N2 entropy is the natural counterpart of the sampled one. Its units
(log2 or ln) do not matter here: every comparison is by rank. One caveat for
the single-chain low-T runs: a site with zero entropy there is a site the
chain never changed in the kept window, which is evidence of constraint only
in proportion to how many mutations the chain accepted (printed per label).

Usage (see structure_zero_polymer.sh for the full pipeline):
    python structure_zero_polymer.py decode-ref --out nsites_study/structure/baseline_metrics.csv
    python structure_zero_polymer.py plot \\
        --runs T12=nsites_study/equil_lowT/T12/chain0 T32=nsites_study/equil_op/T32/chain0 ... \\
        --baseline nsites_study/structure/baseline_metrics.csv \\
        --h3n2 ../../FASTA/H3N2/site_entropy_566.csv --cold T5 T12 \\
        --out-dir nsites_study/structure

A label may be given several times (T32=chain0 T32=chain2): its chains are pooled.
"""
import argparse
import gzip
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(MONTE_CARLO_DIR, "../code/tests"))
from references import get_reference

STRUCT_METRICS = ["Rg_ca", "Rg_allatom", "mean_plddt"]
CHAIN_METRICS = ["U", "Hd_to_ref"]
YLABELS = {"Rg_ca": "Rg, C-alpha [A]", "Rg_allatom": "Rg, heavy atoms [A]", "mean_plddt": "mean pLDDT",
	"U": "U", "Hd_to_ref": "Hd to reference"}


# ------------------------------------------------------------------ input --- #

def label_T(label):
	m = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", label)
	return float(m.group()) if m else math.nan


def parse_runs(items):
	runs = {}
	for item in items:
		label, _, path = item.partition("=")
		if not path:
			raise ValueError(f"--runs entries are LABEL=DIR, got {item!r}")
		runs.setdefault(label, []).append(path)
	return dict(sorted(runs.items(), key=lambda kv: label_T(kv[0])))


def load_chain(run_dir):
	# data.dat (or data.dat.gz, as pushed for the long low-T chains): move, U, Hd_to_ref, accepted.
	path = os.path.join(run_dir, "data.dat")
	opener = open
	if not os.path.isfile(path) and os.path.isfile(path + ".gz"):
		path, opener = path + ".gz", gzip.open
	rows = []
	with opener(path, "rt") as f:
		for line in f:
			s = line.split()
			if not s or s[0][0] in "#m":
				continue
			try:
				rows.append((int(s[0]), float(s[2]), float(s[5]), int(s[-3])))
			except (ValueError, IndexError):
				pass
	return pd.DataFrame(rows, columns=["move", "U", "Hd_to_ref", "accepted"]).drop_duplicates("move").set_index("move")


def kept_sequences(run_dir, burn_in_frac):
	from analyze_ensemble import load_checkpoint_sequences   # needs torch, so only here
	sequences, _, _ = load_checkpoint_sequences(run_dir, burn_in_frac)
	return sequences


def load_h3n2(path, L):
	# Site and entropy columns by name when they are recognisable, else the
	# first column is the site and the last numeric one the entropy. 1-based
	# site numbers are shifted to 0-based.
	df = pd.read_csv(path, sep=None, engine="python")
	cols = {c.lower(): c for c in df.columns}
	site_col = next((cols[k] for k in cols if k.startswith(("site", "pos", "index", "resid"))), df.columns[0])
	ent_col = next((cols[k] for k in cols if "entropy" in k), None)
	if ent_col is None:
		numeric = [c for c in df.columns if c != site_col and pd.api.types.is_numeric_dtype(df[c])]
		ent_col = numeric[-1]
	sites = pd.to_numeric(df[site_col], errors="coerce")
	if sites.isna().any():   # no usable site column: rows are taken in order
		sites = pd.Series(range(len(df)))
	if sites.min() == 1 and sites.max() == L:
		sites = sites - 1
	ent = pd.Series(df[ent_col].astype(float).values, index=sites.astype(int).values).sort_index()
	print(f"H3N2 entropy: {path}, columns site={site_col!r} entropy={ent_col!r}, {len(ent)} sites "
		f"(range {ent.min():.3f}-{ent.max():.3f})")
	if len(ent) != L or ent.index.min() != 0 or ent.index.max() != L - 1:
		raise ValueError(f"{path}: expected one entropy per site 0..{L-1}, got {len(ent)} rows, "
			f"sites {ent.index.min()}..{ent.index.max()}")
	return ent


# --------------------------------------------------------------- analysis --- #

def site_entropy(sequences, L):
	counts = np.zeros((L, 26))
	for _, seq in sequences:
		for i, aa in enumerate(seq):
			counts[i, ord(aa) - 65] += 1
	p = counts / counts.sum(axis=1, keepdims=True)
	with np.errstate(divide="ignore", invalid="ignore"):
		return -np.nansum(np.where(p > 0, p * np.log(p), 0.), axis=1)


def spearman(a, b):
	return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def hypergeom_sf(k, M, n, N):
	# P(X >= k), X ~ Hypergeometric(population M, n successes, N draws)
	return sum(math.comb(n, i) * math.comb(M - n, N - i) for i in range(k, min(n, N) + 1)) / math.comb(M, N)


def conserved_overlap(model, natural, frac):
	# The frac most conserved sites of each (lowest entropy; ties broken by
	# site order, stable), their overlap, and its chance probability.
	L = len(model)
	n = max(1, round(frac * L))
	top_m = set(np.argsort(model, kind="stable")[:n])
	top_n = set(np.argsort(natural, kind="stable")[:n])
	k = len(top_m & top_n)
	return n, k, n * n / L, hypergeom_sf(k, L, n, n)


# ------------------------------------------------------------------ plots --- #

def box_panel(ax, data, labels, metric, baseline):
	ax.boxplot(data, showfliers=False)
	ax.set_xticks(range(1, len(labels) + 1), labels)
	if baseline is not None and metric in baseline.columns:
		ref = baseline[baseline["label"] == "reference"][metric]
		rnd = baseline[baseline["label"].str.startswith("random")][metric]
		if len(ref):
			ax.axhline(ref.mean(), color="tab:green", ls="--", lw=1, label="reference")
		if len(rnd):
			ax.axhspan(rnd.min(), rnd.max(), color="tab:red", alpha=.12, label="random sequences")
	ax.set_ylabel(YLABELS.get(metric, metric))


def plot_metrics(per_label, baseline, out_dir):
	metrics = [m for m in STRUCT_METRICS + CHAIN_METRICS if any(m in d.columns for d in per_label.values())]
	labels = list(per_label)
	fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.8))
	for ax, metric in zip(np.atleast_1d(axes), metrics):
		data = [per_label[l][metric].dropna().values if metric in per_label[l] else [] for l in labels]
		box_panel(ax, data, labels, metric, baseline)
	np.atleast_1d(axes)[0].legend(fontsize=7)
	fig.suptitle("zero_polymer, post-burn-in checkpoints per T")
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "metrics_boxplots.png"), dpi=130)
	plt.close(fig)

	Ts = [label_T(l) for l in labels]
	fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.4))
	for ax, metric in zip(np.atleast_1d(axes), metrics):
		q = [np.nanpercentile(per_label[l][metric], [25, 50, 75]) if metric in per_label[l] and per_label[l][metric].notna().any()
			else [np.nan] * 3 for l in labels]
		q = np.array(q)
		ax.fill_between(Ts, q[:, 0], q[:, 2], alpha=.25)
		ax.plot(Ts, q[:, 1], "o-")
		ax.set_xscale("log"); ax.set_xlabel("T"); ax.set_ylabel(YLABELS.get(metric, metric))
	fig.suptitle("median and interquartile range against T")
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "metrics_vs_T.png"), dpi=130)
	plt.close(fig)


def plot_entropy(ent, h3n2, ref_seq, k_sites, cold, out_dir):
	L = len(ref_seq)
	labels = list(ent.columns)
	fig, axes = plt.subplots(len(labels) + (h3n2 is not None), 1, figsize=(14, 1.9 * (len(labels) + 1)), sharex=True)
	axes = np.atleast_1d(axes)
	if h3n2 is not None:
		axes[0].bar(range(L), h3n2.values, width=1., color="0.3")
		axes[0].set_ylabel("H3N2\nentropy", fontsize=8)
	for ax, label in zip(axes[int(h3n2 is not None):], labels):
		ax.bar(range(L), ent[label].values, width=1.)
		ax.set_ylabel(f"{label}\nentropy", fontsize=8)
	for ax in axes:
		for i in k_sites:
			ax.axvline(i, color="tab:red", alpha=.25, lw=.8)
	axes[-1].set_xlabel(f"site (0-based); red: K-sites at {', '.join(cold)}")
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "site_entropy_profile.png"), dpi=130)
	plt.close(fig)

	if h3n2 is None:
		return
	fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.4), squeeze=False)
	for ax, label in zip(axes[0], labels):
		ax.scatter(h3n2.values, ent[label].values, s=5, alpha=.5)
		ax.set_xlabel("H3N2 entropy"); ax.set_ylabel(f"sampled entropy, {label}")
		ax.set_title(f"{label}: Spearman {spearman(ent[label].values, h3n2.values):.2f}", fontsize=9)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "site_entropy_vs_h3n2.png"), dpi=130)
	plt.close(fig)

	# Natural variability of the sites the sampler conserves, against the rest.
	groups, names = [], []
	for label in labels:
		zero = ent[label].values < 0.05
		groups += [h3n2.values[zero], h3n2.values[~zero]]
		names += [f"{label}\nconserved\n(n={zero.sum()})", f"{label}\nother\n(n={(~zero).sum()})"]
	fig, ax = plt.subplots(figsize=(1.3 * len(names) + 1, 4))
	ax.boxplot(groups, showfliers=False)
	ax.set_xticks(range(1, len(names) + 1), names, fontsize=7)
	ax.set_ylabel("H3N2 entropy")
	ax.set_title("natural variability of sites the sampler keeps (entropy < 0.05) vs the rest", fontsize=9)
	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "h3n2_entropy_by_sampled_class.png"), dpi=130)
	plt.close(fig)


# ------------------------------------------------------------ subcommands --- #

def cmd_decode_ref(args):
	import torch
	from compute_structural_metrics import decode_structure_with_mask, radius_of_gyration, CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config

	_, ref_seq = get_reference(args.ref)
	canonical = "ACDEFGHIKLMNPQRSTVWY"
	rng = np.random.default_rng(args.seed)
	seqs = [("reference", ref_seq)] + [(f"random{i}", "".join(rng.choice(list(canonical), len(ref_seq))))
		for i in range(args.n_random)]

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	config = init_structure_config(num_steps=args.num_steps)
	rows = []
	for label, seq in seqs:
		eprot = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		eprot.expand()
		positions, mask, plddt = decode_structure_with_mask(sampler.model, eprot, config)
		rows.append((label, radius_of_gyration(positions[:, CA_IDX, :]), radius_of_gyration(positions[mask]),
			plddt.mean().item() if plddt is not None else float("nan")))
		print(f"  {label}: Rg_ca={rows[-1][1]:.3f}  Rg_allatom={rows[-1][2]:.3f}  mean_plddt={rows[-1][3]:.3f}")
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	pd.DataFrame(rows, columns=["label"] + STRUCT_METRICS).to_csv(args.out, index=False)
	print(f"Saved -> {args.out}")


def cmd_plot(args):
	_, ref_seq = get_reference(args.ref)
	L = len(ref_seq)
	runs = parse_runs(args.runs)
	os.makedirs(args.out_dir, exist_ok=True)
	baseline = pd.read_csv(args.baseline) if args.baseline and os.path.isfile(args.baseline) else None
	h3n2 = load_h3n2(args.h3n2, L) if args.h3n2 else None

	per_label, ent, summary = {}, {}, []
	for label, dirs in runs.items():
		frames, seqs, n_acc = [], [], 0
		for d in dirs:
			chain = load_chain(d)
			sequences = kept_sequences(d, args.burn_in_frac)
			seqs += sequences
			moves = [m for m, _ in sequences]
			frame = chain.reindex(moves)[CHAIN_METRICS].reset_index(drop=True)
			frame["move"] = moves
			sm = os.path.join(d, "structural_metrics.csv")
			if os.path.isfile(sm):
				frame = frame.merge(pd.read_csv(sm), on="move", how="left")
			else:
				print(f"  {d}: no structural_metrics.csv, structural metrics skipped")
			frames.append(frame)
			n_acc += int(chain.loc[chain.index >= moves[0], "accepted"].sum())
		per_label[label] = pd.concat(frames, ignore_index=True)
		ent[label] = site_entropy(seqs, L)
		row = {"label": label, "T": label_T(label), "chains": len(dirs), "checkpoints": len(seqs),
			"accepted_mutations": n_acc, "mean_site_entropy": ent[label].mean(),
			"sites_entropy_lt_0.05": int((ent[label] < 0.05).sum())}
		for metric in STRUCT_METRICS + CHAIN_METRICS:
			if metric in per_label[label]:
				row[f"median_{metric}"] = per_label[label][metric].median()
		if h3n2 is not None:
			row["spearman_vs_h3n2"] = spearman(ent[label], h3n2.values)
			n, k, expected, p = conserved_overlap(ent[label], h3n2.values, args.top_frac)
			row.update({f"top{args.top_frac:g}_n": n, "overlap": k, "overlap_expected": round(expected, 1), "overlap_p": p})
		summary.append(row)

	ent = pd.DataFrame(ent, index=[f"{i}_{ref_seq[i]}" for i in range(L)])
	ent.index.name = "site"
	cold = [c for c in args.cold if c in ent.columns]
	k_sites = np.where((ent[cold] < args.k_site_threshold).all(axis=1))[0] if cold else np.array([], int)

	summary = pd.DataFrame(summary)
	summary.to_csv(os.path.join(args.out_dir, "summary_by_T.tsv"), sep="\t", index=False)
	if h3n2 is not None:
		ent.insert(0, "H3N2", h3n2.values)
	ent.to_csv(os.path.join(args.out_dir, "site_entropy_by_T.csv"))
	if h3n2 is not None:
		ent = ent.drop(columns="H3N2")

	plot_metrics(per_label, baseline, args.out_dir)
	plot_entropy(ent, h3n2, ref_seq, k_sites, cold, args.out_dir)

	with pd.option_context("display.width", 200, "display.max_columns", 30):
		print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
	print(f"\nK-sites (entropy < {args.k_site_threshold} at every one of {cold}): {len(k_sites)}/{L}")
	if len(k_sites) and h3n2 is not None:
		in_cons = (h3n2.rank(method="first").values[k_sites] <= args.top_frac * L).sum()
		print(f"  of which in the {args.top_frac:.0%} most conserved H3N2 sites: {in_cons} "
			f"(chance: {len(k_sites) * args.top_frac:.1f})")
		print(f"  median H3N2 entropy: K-sites {np.median(h3n2.values[k_sites]):.3f}, "
			f"other sites {np.median(np.delete(h3n2.values, k_sites)):.3f}")
	print(f"\nOutputs in {args.out_dir}: summary_by_T.tsv, site_entropy_by_T.csv, metrics_boxplots.png, "
		f"metrics_vs_T.png, site_entropy_profile.png" + (", site_entropy_vs_h3n2.png, "
		"h3n2_entropy_by_sampled_class.png" if h3n2 is not None else ""))


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = p.add_subparsers(dest="cmd", required=True)
	d = sub.add_parser("decode-ref")
	d.add_argument("--out", required=True)
	d.add_argument("--ref", default="zero_polymer")
	d.add_argument("--n-random", type=int, default=5)
	d.add_argument("--num-steps", type=int, default=8)
	d.add_argument("--seed", type=int, default=0)
	q = sub.add_parser("plot")
	q.add_argument("--runs", nargs="+", required=True, help="LABEL=DIR, a label may repeat to pool chains")
	q.add_argument("--ref", default="zero_polymer")
	q.add_argument("--baseline", default=None, help="CSV from decode-ref")
	q.add_argument("--h3n2", default=None, help="per-site entropy of the H3N2 alignment (566 rows)")
	q.add_argument("--cold", nargs="*", default=[], help="labels below the transition, for the K-sites")
	q.add_argument("--burn-in-frac", type=float, default=0.5)
	q.add_argument("--k-site-threshold", type=float, default=0.05)
	q.add_argument("--top-frac", type=float, default=0.1, help="fraction of most conserved sites for the overlap")
	q.add_argument("--out-dir", default="nsites_study/structure")
	args = p.parse_args()
	if args.cmd == "decode-ref":
		sys.path.insert(0, os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code")))
		sys.path.insert(0, os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs")))
		cmd_decode_ref(args)
	else:
		cmd_plot(args)


if __name__ == "__main__":
	main()
