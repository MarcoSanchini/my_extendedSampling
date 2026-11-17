"""
Single-mutant scan of the CONTACT-MAP energy on PTEN: every one of the
324 x 19 = 6156 point mutants is decoded by ESM3 and its contact map compared
with the reference's, the way Zambon et al 2024 define the energy:

    E = sum_ij |D_ij(mut) - D_ij(ref)| / sum_ij (D_ij(mut) + D_ij(ref)),  |i-j| > 2

with D the CA contact map at 8 A (the decoder writes backbone + CB only, so the
paper's 4 A heavy-atom map is not available). Both maps come from ESM3 at the
SAME number of generation steps, so E(reference) = 0 exactly (decoding is at
temperature 0 and was measured deterministic, bench_decode.py 2026-11-16).

The question it answers. bench_decode.py showed a 2-step decode costs 0.13 s,
about one U_am move, so a contact-map energy INSIDE the Monte Carlo loop is
affordable. What it did not show is whether such an energy can see a single
mutation at all: a structure predictor that returns the same contact map for
most point mutants gives dE = 0 for most proposals, and a sampler on a flat
landscape is a random walk. This scan measures that directly, and lines each
mutant up with the U_am scan (scan_single_mutants.py, same 6156 mutants) so
the two energies can be compared substitution by substitution.

Columns: site, ref_aa, mut_aa, E, n_diff (contacts gained + lost), rmsd (CA
RMSD to the reference's prediction, A), plddt (mean, 0-100), t (s).

Cost: one decode per mutant, 0.13 s at 2 steps -> ~14 min for all 6156 on
astaroth; 1.6 s at 32 steps, so use --sites for a subset there. Resumable like
scan_single_mutants.py: appended and flushed per site, complete sites skipped.

SOFT CONTACTS (added 2026-11-17). The measured problem with E is not that it is flat but
that it is dominated by the 8 A threshold: a point mutant moves the structure ~0.45 A, so
only pairs sitting within a fraction of an A of the cutoff can change the map, and which
pairs those are is a property of the decoding trajectory. Changing the number of decoding
steps then moves E further than the mutation does (median |E_32 - E_2| is 1.6x the spread
of E_2 across mutants), and no two step counts agree per mutant (Spearman 0.0 to 0.3).
--soft-widths replaces the step by a sigmoid of the CA distance, which is the SAME energy
with the discontinuity removed, and computes it alongside the hard one from the same decode
at no extra GPU cost. Width -> 0 reproduces the hard E exactly, and that limit is checked
on random coordinates at startup. The diagnostic question: run two step counts and read the
"agreement between the two" block. If the soft columns agree far better than the hard one,
the threshold was the problem and a soft E may be usable in the loop; if they do not, the
decode itself is the problem and no reweighting of distances will fix it.

Usage (from softmax/monte_carlo, GPU idle):
    nohup python -u scan_mutants_E.py --num-steps 2 --out pten_natural/pten_mutants_E2.csv \\
        > scan_mutants_E2.log 2>&1 &
    # the same 20 sites at 32 steps, to check that 2 steps see the same mutants
    nohup python -u scan_mutants_E.py --num-steps 32 --sites $(seq 0 17 323) \\
        --out pten_natural/pten_mutants_E32.csv > scan_mutants_E32.log 2>&1 &

    # the soft-contact diagnostic: the same 20 sites at 2, 8 and 32 steps, hard and soft E
    # in one file each, then each compared with the 2-step run (~25 min in total)
    for s in 2 8 32; do
        python -u scan_mutants_E.py --num-steps $s --sites $(seq 0 17 323) \\
            --out pten_natural/pten_mutants_soft_E$s.csv \\
            --compare pten_natural/pten_mutants_soft_E2.csv > scan_mutants_soft_E$s.log 2>&1
    done

The summary at the end (also on its own, no GPU: --summary-only) gives the
share of mutants with E = 0, the distribution of E, and Spearman(E, dU_am)
pooled, within site and per site; with --compare, the agreement with another
E scan (e.g. 2 vs 32 steps) on the sites both contain.
"""
import argparse
import os
import sys
import time

import numpy as np

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code"))
CUSTOMS_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs"))
for p in (MONTE_CARLO_DIR, CODE_DIR, CUSTOMS_DIR, os.path.join(CODE_DIR, "tests")):
	if p not in sys.path:
		sys.path.insert(0, p)

from bench_decode import contact_map, kabsch_rmsd

AA = "ACDEFGHIKLMNPQRSTVWY"
SOFT_WIDTHS = (0.5, 1.0, 2.0)


def soft_columns(widths):
	"""Column name for each softening width, e.g. 1.0 -> 'E_w1.0'."""
	return [f"E_w{w:g}" for w in widths]


def columns(widths):
	return "site,ref_aa,mut_aa,E,n_diff," + "".join(f"{c}," for c in soft_columns(widths)) + "rmsd,plddt,t\n"


def soft_contact(ca, cutoff, width):
	"""Contact map with the hard cutoff replaced by a sigmoid of the CA distance:

	    c_ij = 1 / (1 + exp((d_ij - cutoff) / width)),   |i - j| > 2

	so a pair at the cutoff counts 0.5, one `width` closer counts 0.73, one `width`
	further 0.27. As width -> 0 this becomes exactly `contact_map`, which is the point:
	it is the same energy with the discontinuity smoothed, and nothing else changed.
	Rows with no CA, and |i - j| <= 2, are zero as in the hard map."""
	n = len(ca)
	ok = np.isfinite(ca).all(1)
	d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
	with np.errstate(over="ignore", invalid="ignore"):
		c = 1.0 / (1.0 + np.exp((d - cutoff) / width))
	keep = (np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 2) & ok[:, None] & ok[None, :]
	return np.triu(np.where(keep, np.nan_to_num(c), 0.0), 3)


def soft_energy(a, b):
	"""The paper's E with |D_ij(mut) - D_ij(ref)| read on the soft maps. Identical
	in form to `bench_decode.energy`, which is the XOR of two boolean maps."""
	s = a.sum() + b.sum()
	return float(np.abs(a - b).sum() / s) if s else float("nan")


def done_sites(path, header):
	"""Sites with all 19 rows in an existing CSV, and those rows."""
	if not os.path.exists(path):
		return set(), []
	keep, counts = [], {}
	with open(path) as f:
		first = f.readline()
		if first != header:
			raise SystemExit(f"{path} has another header:\n  found    {first.strip()}\n  expected {header.strip()}\n"
				"Move it aside, pass another --out, or match --soft-widths to the file.")
		for line in f:
			s = line.split(",", 1)[0]
			if s.isdigit():
				counts[int(s)] = counts.get(int(s), 0) + 1
				keep.append(line)
	full = {s for s, n in counts.items() if n == 19}
	return full, [l for l in keep if int(l.split(",", 1)[0]) in full]


def summary(path, am_scan, compare):
	import pandas as pd
	m = pd.read_csv(path)
	print(f"\n# SUMMARY of {path}: {len(m)} mutants over {m.site.nunique()} sites")
	print(f"# time per mutant: median {m.t.median():.3f} s")
	zero = (m.n_diff == 0).mean()
	print(f"# E = 0 (identical contact map to the reference): {zero:.1%} of mutants; "
		f"n_diff <= 2: {(m.n_diff <= 2).mean():.1%}")
	q = m.E.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
	print("# E quantiles 10/25/50/75/90/max: " + " ".join(f"{v:.4f}" for v in q.values))
	print("# n_diff quantiles 10/25/50/75/90/max: " + " ".join(f"{v:.0f}" for v in m.n_diff.quantile([.1, .25, .5, .75, .9, 1.]).values))
	print(f"# RMSD to the reference's prediction: median {m.rmsd.median():.2f} A, 90% {m.rmsd.quantile(.9):.2f}, "
		f"max {m.rmsd.max():.2f}; pLDDT median {m.plddt.median():.1f}, min {m.plddt.min():.1f}")
	per_site_zero = m.groupby("site").n_diff.apply(lambda x: (x == 0).mean())
	print(f"# sites where EVERY mutant has E = 0: {(per_site_zero == 1).sum()} of {m.site.nunique()}; "
		f"where none does: {(per_site_zero == 0).sum()}")
	# "E" is the paper's hard-cutoff energy; "E_w*" are the same energy with the 8 A step
	# replaced by a sigmoid of that width. They cost nothing extra (same decode), so every
	# number below is printed for each of them and the columns can be read side by side.
	energies = ["E"] + [c for c in m.columns if c.startswith("E_w")]
	for c in energies[1:]:
		q = m[c].quantile([0.1, 0.5, 0.9])
		print(f"# {c} quantiles 10/50/90: " + " ".join(f"{v:.4f}" for v in q.values))
	if am_scan and os.path.exists(am_scan):
		u = pd.read_csv(am_scan)
		j = m.merge(u, on=["site", "ref_aa", "mut_aa"], how="inner")
		print(f"# joined with {am_scan}: {len(j)} mutants")
		print("#   agreement with U_am          " + "".join(f"{c:>10}" for c in energies))
		pooled = [j[c].corr(j.dU, method="spearman") for c in energies]
		print("#     pooled                     " + "".join(f"{v:>+10.3f}" for v in pooled))
		wit = [np.nanmedian([g[c].corr(g.dU, method="spearman") for _, g in j.groupby("site") if g[c].nunique() > 2])
			for c in energies]
		print("#     within site (median)       " + "".join(f"{v:>+10.3f}" for v in wit))
		g = j.groupby("site").agg(lU=("dU", lambda x: np.log10(x).mean()), **{c: (c, "mean") for c in energies})
		per = [g[c].corr(g.lU, method="spearman") for c in energies]
		print("#     per site (means)           " + "".join(f"{v:>+10.3f}" for v in per))
		for lab, sub in (("E = 0", j[j.n_diff == 0]), ("E > 0", j[j.n_diff > 0])):
			if len(sub):
				print(f"#   dU_am where {lab}: median {sub.dU.median():.1f} (n={len(sub)})")
	if compare and os.path.exists(compare):
		c = pd.read_csv(compare)
		j = m.merge(c, on=["site", "ref_aa", "mut_aa"], suffixes=("", "_c"))
		shared = [e for e in energies if f"{e}_c" in j.columns]
		print(f"# against {compare} on {len(j)} shared mutants. This is the diagnostic the softening is for:")
		print("#   agreement between the two     " + "".join(f"{e:>10}" for e in shared))
		print("#     pooled                     " + "".join(f"{j[e].corr(j[f'{e}_c'], method='spearman'):>+10.3f}" for e in shared))
		print("#     within site (median)       " + "".join(
			f"{np.nanmedian([g[e].corr(g[f'{e}_c'], method='spearman') for _, g in j.groupby('site') if g[e].nunique() > 2]):>+10.3f}"
			for e in shared))
		gg = j.groupby("site").agg(**{f"a{i}": (e, "mean") for i, e in enumerate(shared)},
			**{f"b{i}": (f"{e}_c", "mean") for i, e in enumerate(shared)})
		print("#     per site (means)           " + "".join(
			f"{gg[f'a{i}'].corr(gg[f'b{i}'], method='spearman'):>+10.3f}" for i in range(len(shared))))
		print("#     median |difference|        " + "".join(
			f"{np.median(np.abs(j[e] - j[f'{e}_c'])):>10.4f}" for e in shared))
		print("#     as a share of its own IQR  " + "".join(
			f"{np.median(np.abs(j[e] - j[f'{e}_c'])) / (j[e].quantile(.75) - j[e].quantile(.25)):>10.2f}" for e in shared))
		print(f"#   E = 0 in both {((j.n_diff == 0) & (j.n_diff_c == 0)).mean():.1%}, "
			f"in this only {((j.n_diff == 0) & (j.n_diff_c > 0)).mean():.1%}, "
			f"in the other only {((j.n_diff > 0) & (j.n_diff_c == 0)).mean():.1%}")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--ref", default="pten_1d5r")
	ap.add_argument("--num-steps", type=int, default=2)
	ap.add_argument("--sites", type=int, nargs="+", help="0-based sites (default all)")
	ap.add_argument("--ca-cutoff", type=float, default=8.0)
	ap.add_argument("--soft-widths", type=float, nargs="*", default=list(SOFT_WIDTHS),
		help="sigmoid widths in A for the soft-contact version of E, computed alongside the hard one "
			"at no extra cost (default 0.5 1.0 2.0; pass none for the hard E only). Width -> 0 is the "
			"hard cutoff, so these interpolate away the 8 A discontinuity")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--out", required=True)
	ap.add_argument("--am-scan", default=os.path.join(MONTE_CARLO_DIR, "pten_natural", "pten_single_mutants.csv"),
		help="the U_am scan to compare with")
	ap.add_argument("--compare", default=None, help="another E scan (e.g. at 32 steps) to compare with")
	ap.add_argument("--summary-only", action="store_true")
	args = ap.parse_args()
	widths = sorted(w for w in (args.soft_widths or []) if w > 0)
	header = columns(widths)
	if args.summary_only:
		summary(args.out, args.am_scan, args.compare)
		return
	# The soft energy must reduce to the hard one as the width goes to zero. Check it on
	# random coordinates before spending any GPU time, so a mistake here cannot be mistaken
	# for a result later.
	rng = np.random.default_rng(0)
	for _ in range(3):
		p, q = rng.normal(0, 6, (40, 3)), rng.normal(0, 6, (40, 3))
		hard = (contact_map(p, args.ca_cutoff) ^ contact_map(q, args.ca_cutoff)).sum() / (
			contact_map(p, args.ca_cutoff).sum() + contact_map(q, args.ca_cutoff).sum())
		tiny = soft_energy(soft_contact(p, args.ca_cutoff, 1e-6), soft_contact(q, args.ca_cutoff, 1e-6))
		if abs(hard - tiny) > 1e-6:
			raise SystemExit(f"soft_contact does not reduce to contact_map: {hard:.6f} vs {tiny:.6f}")
	print(f"# soft widths {widths or 'none'} A; the width -> 0 limit reproduces the hard E exactly (checked)",
		flush=True)

	from references import get_reference
	_, ref = get_reference(args.ref)
	L = len(ref)

	import torch
	from compute_structural_metrics import CA_IDX
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from custom_esm.sdk.api import ESMProteinTensor

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	model, decoder = sampler.model, sampler.model.get_structure_decoder()
	cfg = init_structure_config(num_steps=args.num_steps)
	sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

	def predict(seq):
		torch.manual_seed(args.seed)
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		ep.expand()
		sync(); t0 = time.perf_counter()
		with torch.no_grad():
			out = model.generate(input=ESMProteinTensor(sequence=ep.tokens), config=cfg)
			tok = out.structure if out.structure.dim() > 1 else out.structure.unsqueeze(0)
			dec = decoder.decode(tok)
		sync()
		ca = dec["bb_pred"][0, 1:-1, CA_IDX].float().cpu().numpy()
		pl = dec["plddt"][0, 1:-1].float().cpu().numpy() if "plddt" in dec else np.full(L, np.nan)
		pl = pl * 100 if np.nanmax(pl) <= 1 else pl
		return ca, float(np.nanmean(pl)), time.perf_counter() - t0

	ref_ca, ref_pl, _ = predict(ref)
	ref_ca2, _, _ = predict(ref)
	ref_map = contact_map(ref_ca, args.ca_cutoff)
	n_ref = int(ref_map.sum())
	ref_soft = [soft_contact(ref_ca, args.ca_cutoff, w) for w in widths]
	again = int((contact_map(ref_ca2, args.ca_cutoff) ^ ref_map).sum())
	print(f"# {args.ref}, L={L}, {args.num_steps} steps: reference pLDDT {ref_pl:.1f}, {n_ref} contacts; "
		f"same sequence twice differs in {again} contacts (must be 0)", flush=True)
	if widths:
		floor = [soft_energy(soft_contact(ref_ca2, args.ca_cutoff, w), s) for w, s in zip(widths, ref_soft)]
		print("# soft noise floor, same sequence twice: " + ", ".join(
			f"E_w{w:g} {v:.2e}" for w, v in zip(widths, floor)) + " (must be 0)", flush=True)
	if again:
		print("# WARNING: the decode is not deterministic here, so E has a noise floor; see the line above.", flush=True)

	sites = args.sites if args.sites is not None else list(range(L))
	have, kept = done_sites(args.out, header)
	todo = [s for s in sites if s not in have]
	print(f"# {len(sites)} sites requested, {len(have & set(sites))} already in {args.out}, {len(todo)} to do "
		f"({19 * len(todo)} decodes)", flush=True)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		f.write(header)
		f.writelines(kept)

	t0 = time.perf_counter()
	with open(args.out, "a") as out:
		for k, s in enumerate(todo, 1):
			cur = ref[s]
			Es = []
			for a in AA:
				if a == cur:
					continue
				ca, pl, dt = predict(ref[:s] + a + ref[s + 1:])
				cmap = contact_map(ca, args.ca_cutoff)
				n_diff = int((cmap ^ ref_map).sum())
				E = n_diff / (int(cmap.sum()) + n_ref)
				soft = "".join(f"{soft_energy(soft_contact(ca, args.ca_cutoff, w), s0):.6f},"
					for w, s0 in zip(widths, ref_soft))
				out.write(f"{s},{cur},{a},{E:.6f},{n_diff},{soft}{kabsch_rmsd(ca, ref_ca):.3f},{pl:.2f},{dt:.4f}\n")
				Es.append(E)
			out.flush()
			if k % 10 == 0 or k == len(todo):
				el = time.perf_counter() - t0
				print(f"# {k}/{len(todo)} sites, {el / 60:.1f} min, {el / k / 19:.3f} s per mutant, "
					f"{(len(todo) - k) * el / k / 60:.1f} min left; site {s} {cur}: E min {min(Es):.4f} "
					f"median {np.median(Es):.4f} max {max(Es):.4f}, zeros {sum(e == 0 for e in Es)}/19", flush=True)

	summary(args.out, args.am_scan, args.compare)


if __name__ == "__main__":
	main()
