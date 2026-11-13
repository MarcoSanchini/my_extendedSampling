"""
K-sites of PTEN: what biology and two mutational scans say about every site of
the pten_1d5r reference, and how the sampler's per-site entropy compares.

"K-sites" as in Zambon et al 2024 and identify_k_sites.py: the sites the
sampler keeps completely conserved in the cold regime. For protein G the paper
compares them with the sites conserved in the natural homologs. For PTEN there
are three independent references for "this site cannot change":

  natural    per-site entropy of the natural homologs of the 1D5R chain
             (jackhmmer on UniRef90, from OpenProteinSet, CC BY 4.0), at four
             depths: vertebrate PTEN (>= 90% identity), orthologs (>= 70%),
             the PTEN-like family (PTEN, TPTE/VSP, tensins, auxilin/GAK), and
             every hit that covers the phosphatase domain;
  abundance  VAMP-seq, Matreyek et al 2018 Nat Genet 50:874 (PMID 29785012):
             steady-state abundance of ~4100 missense variants, 1 = wild type,
             0 = nonsense-like. Plus the 2021 fill-in (Matreyek et al, Genome
             Med 13:165), which takes it to ~5100 variants;
  activity   Mighell et al 2018 AJHG 102:943 (PMC5986715): lipid phosphatase
             fitness in yeast of ~7260 missense variants, 0 = wild type, about
             -4 = nonsense-like; below -1.11 counts as loss of function.

Subcommands
  fetch    downloads the raw files into --raw-dir (github raw + the OpenFold
           S3 bucket; both reachable from the cloud container, unlike NCBI,
           UniProt and EBI). Raw files are NOT committed: the VAMP-seq scores
           are (c) University of Washington, academic non-commercial licence
           (see the README of github.com/FowlerLab/VAMPseq).
  build    one row per reference site -> pten_site_table.csv, and the profile
           figure. No sampler data needed.
  compare  adds the sampled per-site entropy of one or more chains
           (LABEL=RUN_DIR, rebuilt from start.txt + the accepted moves in
           data.dat, so no checkpoints and no torch are needed) and tests it
           against the three references.
  family   feasibility pass for the wider protein tyrosine phosphatase
           superfamily: which PTEN sites the HHsearch profile alignments
           reach (PDB70 templates, classical PTPs = Pfam PF00102 among them),
           and the PF00102 conservation carried over from PTP1B.

Numbering. The reference is the 1D5R construct, 324 aa: UniProt P60484
residues 7-285 and 310-353, the 286-309 loop deleted. Two reference residues
are not native: site 0 (Met in place of Glu7) and site 279 (a Val at the
deletion junction). They have no experimental score. `site` is 0-based as in
data.dat, `uniprot_pos` is the numbering both papers use.

Usage:
    python pten_k_sites.py fetch
    python pten_k_sites.py build
    python pten_k_sites.py compare --runs T1=pten1d5r_study/equil/T1/chain0 \\
        T2=pten1d5r_study/equil/T2/chain0 T4=pten1d5r_study/equil/T4/chain0 --cold T1 T2
    python pten_k_sites.py family
"""
import argparse
import difflib
import gzip
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(MONTE_CARLO_DIR, "../code/tests"))
from references import get_reference
from ha_site_entropy import AA, GAP, encode, neighbour_counts
from structure_zero_polymer import spearman, set_overlap, label_T

REF_NAME = "pten_1d5r"
# UniProt P60484, canonical isoform, 403 aa. Both scans number variants on it.
PTEN_UNIPROT = (
	"MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNIDDVVRFLDSK"
	"HKNHYKIYNLCAERHYDTAKFNCRVAQYPFEDHNPPQLELIKPFCEDLDQWLSEDDNHVA"
	"AIHCKAGKGRTGVMICAYLLHRGKFLKAQEALDFYGEVRTRDKKGVTIPSQRRYVYYYSY"
	"LLKNHLDYRPVALLFHKMMFETIPMFSGGTCNPQFVVCQLKVKIYSSNSGPTRREDKFMY"
	"FEFPQPLPVCGDIKVEFFHKQNKMLKKDKMFHFWVNTFFIPGPEETSEKVENGSLCDQEI"
	"DSICSIERADNDKEYLVLTLTKNDLDKANKDKANRYFSPNFKVKLYFTKTVEEPSNPEAS"
	"SSTSVTPDVSDNEPDHYRYSDTTDSDPENEPFDEDQHTQITKV"
)
PHOSPHATASE = (14, 185)     # UniProt numbering, Lee et al 1999 (the 1D5R paper)
C2 = (186, 351)
# Landmarks drawn on the profile (UniProt numbering).
LANDMARKS = {"P loop\n123-130": (123, 130), "WPD loop\n88-98": (88, 98), "TI loop\n160-171": (160, 171),
	"CBR3\n260-269": (260, 269)}
ACTIVITY_LOF = -1.11        # Matreyek 2021 (PTEN_composite_analysis.Rmd, pten_func_syn)
MIN_VARIANTS = 5            # scored missense variants a site needs for a positional mean

GH = "https://raw.githubusercontent.com/"
S3 = "https://openfold.s3.amazonaws.com/pdb/"
RAW_FILES = {
	"PTEN_variant_data.tsv": GH + "FowlerLab/VAMPseq/master/PTEN_variant_data.tsv",
	"PTEN_positional_data.tsv": GH + "FowlerLab/VAMPseq/master/PTEN_positional_data.tsv",
	"VAMPseq_README.md": GH + "FowlerLab/VAMPseq/master/README.md",
	"Composite_abundance_data.tsv": GH + "MatreyekLab/PTEN_composite/master/output_datatables/Composite_abundance_data.tsv",
	"Mighell_PTEN_phosphatase_data.csv": GH + "MatreyekLab/PTEN_composite/master/input_datatables/Mighell_PTEN_phosphatase_data.csv",
	"1d5r_A_uniref90_hits.a3m": S3 + "1d5r_A/a3m/uniref90_hits.a3m",
	"1d5r_A_pdb70_hits.hhr": S3 + "1d5r_A/hhr/pdb70_hits.hhr",
	"2cm2_A_uniref90_hits.a3m": S3 + "2cm2_A/a3m/uniref90_hits.a3m",
}
C_NAT, C_ABU, C_ACT, C_SAMP = "#2a7f62", "#3b6fb6", "#c2571a", "#444444"


# ------------------------------------------------------------------ input --- #

def cmd_fetch(args):
	os.makedirs(args.raw_dir, exist_ok=True)
	for name, url in RAW_FILES.items():
		path = os.path.join(args.raw_dir, name)
		if os.path.isfile(path) and os.path.getsize(path) > 0 and not args.force:
			print(f"have   {name}")
			continue
		req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
		with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
			f.write(r.read())
		print(f"fetched {name}  {os.path.getsize(path)/1e6:.2f} MB")
	print(f"\nRaw files in {args.raw_dir} (not committed). The VAMP-seq abundance scores are the copyright "
		"of the University of Washington, academic non-commercial use: see VAMPseq_README.md.")


def reference_map():
	"""One row per reference site: 0-based site, UniProt position, native flag, domain."""
	ref = get_reference(REF_NAME)[1]
	sm = difflib.SequenceMatcher(None, ref, PTEN_UNIPROT, autojunk=False)
	upos = np.full(len(ref), -1)
	for b in sm.get_matching_blocks():
		if b.size >= 10:            # the two real segments; the single-letter junction matches are chance
			upos[b.a:b.a + b.size] = np.arange(b.b, b.b + b.size) + 1
	# Non-native residues keep the position they occupy in the construct numbering
	# when it is unambiguous (the N-terminal Met sits at 7), else none.
	native = upos > 0
	if not native[0] and native[1]:
		upos[0] = upos[1] - 1
	df = pd.DataFrame({"site": np.arange(len(ref)), "ref_aa": list(ref), "uniprot_pos": upos, "native": native})
	df["domain"] = np.where(df.uniprot_pos < PHOSPHATASE[0], "N-term",
		np.where(df.uniprot_pos <= PHOSPHATASE[1], "phosphatase", "C2"))
	df.loc[df.uniprot_pos < 0, "domain"] = "junction"
	return ref, df


def read_a3m(path, L):
	"""a3m -> names and sequences on the query's columns (insertions, lowercase, dropped)."""
	names, seqs = [], []
	opener = gzip.open if path.endswith(".gz") else open
	with opener(path, "rt") as f:
		for line in f:
			if line.startswith(">"):
				names.append(line[1:].strip()); seqs.append([])
			elif names:
				seqs[-1].append(line.strip())
	seqs = [re.sub("[a-z.]", "", "".join(s)) for s in seqs]
	bad = [n for n, s in zip(names, seqs) if len(s) != L]
	if bad:
		raise ValueError(f"{path}: {len(bad)} sequences do not have {L} match columns, e.g. {bad[0]!r}")
	return names, seqs


def weighted_entropy(X, w, max_gap):
	"""Per column: Shannon entropy in nats over the residues, gap fraction, consensus and its frequency."""
	L = X.shape[1]
	counts = np.zeros((L, GAP + 1))
	for k in range(GAP + 1):
		counts[:, k] = ((X == k) * w[:, None]).sum(axis=0)
	gap = counts[:, GAP] / counts.sum(axis=1)
	res = counts[:, :GAP]
	with np.errstate(divide="ignore", invalid="ignore"):
		p = res / res.sum(axis=1, keepdims=True)
		H = -np.nansum(np.where(p > 0, p * np.log(p), 0.), axis=1)
	H[(gap > max_gap) | (res.sum(axis=1) == 0)] = np.nan
	best = res.argmax(axis=1)
	return H, gap, np.array(list(AA))[best], p[np.arange(L), best]


def natural_tiers(a3m, ref, rmap, args):
	"""Four nested natural alignments of the reference and their per-site entropies."""
	L = len(ref)
	names, seqs = read_a3m(a3m, L)
	if seqs[0] != ref:
		raise ValueError(f"{a3m}: the query is not the {REF_NAME} reference")
	X = encode(seqs, L)
	aligned = X != GAP
	ident = ((X == X[0]) & aligned).sum(1) / np.maximum(aligned.sum(1), 1)
	cov = aligned.mean(1)
	pd_cols = (rmap.domain == "phosphatase").values
	cov_pd = aligned[:, pd_cols].mean(1)
	tiers = {
		# the same protein in vertebrates, where PTEN is nearly invariant
		"vertebrate": (cov >= 0.8) & (ident >= 0.9),
		# the same protein in other species: both domains present and clearly PTEN
		"orthologs": (cov >= 0.8) & (ident >= args.ortholog_identity),
		# PTEN-like family: both domains present (PTEN, TPTE/VSP, tensins, auxilin/GAK)
		"family": cov >= 0.8,
		# anything whose phosphatase domain aligns; C2 columns are mostly gaps here
		"pdomain": cov_pd >= 0.7,
	}
	# The ortholog tiers are within ~70-100% identity of each other, so the 0.8 weighting would
	# collapse them to one or two clusters (N_eff 1-6): they are weighted at 0.95 instead.
	theta = {"vertebrate": args.ortholog_weight_identity, "orthologs": args.ortholog_weight_identity}
	out, summary = {}, []
	for tier, keep in tiers.items():
		Xt = X[keep]
		w = 1. / neighbour_counts(Xt, np.ones(len(Xt)), theta.get(tier, args.weight_identity))
		H, gap, cons, cons_f = weighted_entropy(Xt, w, args.max_gap)
		if tier == "pdomain":
			H[~pd_cols] = np.nan
		out[tier] = (H, gap, cons, cons_f)
		summary.append({"tier": tier, "sequences": int(keep.sum()), "weight_identity": theta.get(tier, args.weight_identity), "n_eff": w.sum(),
			"median_identity_to_PTEN": float(np.median(ident[keep])), "sites_with_entropy": int(np.isfinite(H).sum()),
			"mean_entropy_nats": float(np.nanmean(H)), "sites_entropy_lt_0.05": int((H < 0.05).sum())})
	return out, pd.DataFrame(summary)


def positional(df, pos_col, score_col, lof_below):
	"""Missense variants -> per UniProt position: n scored, mean score, fraction below the loss threshold."""
	g = df.dropna(subset=[score_col]).groupby(pos_col)[score_col]
	out = pd.DataFrame({"n": g.size(), "mean": g.mean(), "frac_lof": g.apply(lambda s: (s < lof_below).mean())})
	out.loc[out.n < MIN_VARIANTS, ["mean", "frac_lof"]] = np.nan
	return out


def load_scans(raw_dir):
	v = pd.read_csv(os.path.join(raw_dir, "PTEN_variant_data.tsv"), sep="\t", low_memory=False)
	v = v[v["class"] == "missense"]
	# authors' classes: "low" = score and upper CI below the 5th percentile of synonymous
	v_low = v.loc[v.abundance_class.notna(), ["position", "abundance_class"]].copy()
	v_low["low"] = (v_low.abundance_class == "low").astype(float)
	vamp = positional(v, "position", "score", lof_below=np.inf).drop(columns="frac_lof")
	vamp["frac_low"] = v_low.groupby("position").low.mean()
	vamp.loc[vamp.n < MIN_VARIANTS, "frac_low"] = np.nan
	p = pd.read_csv(os.path.join(raw_dir, "PTEN_positional_data.tsv"), sep="\t").set_index("position")
	vamp["class_2018"] = p.abundance_class          # the authors' own positional call: intolerant / enhanced
	vamp["rsa"] = p.rsa                             # relative solvent accessibility in 1D5R [%], from the same table
	c = pd.read_csv(os.path.join(raw_dir, "Composite_abundance_data.tsv"), sep="\t", low_memory=False)
	c = c[c["class"] == "missense"]
	comp = positional(c, "position", "score_total", lof_below=np.inf).drop(columns="frac_lof")
	m = pd.read_csv(os.path.join(raw_dir, "Mighell_PTEN_phosphatase_data.csv"), encoding="utf-8-sig")
	m = m[(m.Type == "missense") & (m.High_conf == True)].copy()      # noqa: E712  (NaN must fail)
	m["position"] = m["Variant (one letter)"].str.extract(r"(\d+)").astype(int)
	act = positional(m, "position", "Cum_score", lof_below=ACTIVITY_LOF)
	return vamp, comp, act


# ------------------------------------------------------------------ build --- #

def build_table(args):
	ref, t = reference_map()
	tiers, tier_summary = natural_tiers(os.path.join(args.raw_dir, "1d5r_A_uniref90_hits.a3m"), ref, t, args)
	for tier, (H, gap, cons, cons_f) in tiers.items():
		t[f"nat_{tier}_entropy"] = H
		if tier == "family":
			t["nat_family_gap"], t["nat_family_consensus"], t["nat_family_consensus_freq"] = gap, cons, cons_f
	vamp, comp, act = load_scans(args.raw_dir)
	pos = t.uniprot_pos.where(t.native)          # the two non-native residues get no experimental score
	t["abundance_n"] = pos.map(vamp.n)
	t["abundance_mean"] = pos.map(vamp["mean"])
	t["abundance_frac_low"] = pos.map(vamp.frac_low)
	t["abundance_class_2018"] = pos.map(vamp.class_2018)
	t["rsa"] = pos.map(vamp.rsa)
	t["abundance2021_n"] = pos.map(comp.n)
	t["abundance2021_mean"] = pos.map(comp["mean"])
	t["activity_n"] = pos.map(act.n)
	t["activity_mean"] = pos.map(act["mean"])
	t["activity_frac_lof"] = pos.map(act.frac_lof)
	# The three K-like sets. Each is the reference's own statement "this site cannot change".
	t["K_natural"] = t.nat_family_entropy < args.natural_k
	t["K_abundance"] = t.abundance_class_2018 == "intolerant"
	t["K_activity"] = t.activity_frac_lof >= args.activity_k
	return ref, t, tier_summary


def shade_domains(ax, t):
	for dom, col in (("phosphatase", "#f3ead9"), ("C2", "#e3ecf5")):
		s = t.site[t.domain == dom]
		ax.axvspan(s.min() - .5, s.max() + .5, color=col, zorder=0, lw=0)


def plot_profile(t, tier_summary, path, sampled=None):
	rows = [("nat_orthologs_entropy", "orthologs\nentropy [nats]", C_NAT),
		("nat_family_entropy", "PTEN-like family\nentropy [nats]", C_NAT),
		("abundance2021_mean", "abundance\n(VAMP-seq, mean)", C_ABU),
		("activity_mean", "activity\n(Mighell, mean)", C_ACT)]
	n_extra = 0 if sampled is None else sampled.shape[1]
	fig, axes = plt.subplots(len(rows) + n_extra + 1, 1, figsize=(13, 1.55 * (len(rows) + n_extra) + 1.3), sharex=True,
		gridspec_kw={"height_ratios": [1] * (len(rows) + n_extra) + [.55]})
	for ax, (col, lab, c) in zip(axes, rows):
		shade_domains(ax, t)
		ax.bar(t.site, t[col].fillna(0), width=1., color=c, lw=0)
		miss = t.site[t[col].isna()]
		ax.plot(miss, np.zeros(len(miss)), "|", color="#b0b0b0", ms=4)      # no data at this site
		ax.set_ylabel(lab, fontsize=8)
		if col == "abundance2021_mean":
			ax.axhline(1, color="k", lw=.5, ls=":")
		if col == "activity_mean":
			ax.axhline(ACTIVITY_LOF, color="k", lw=.5, ls=":")
	if sampled is not None:
		for ax, label in zip(axes[len(rows):], sampled.columns):
			shade_domains(ax, t)
			ax.bar(t.site, sampled[label], width=1., color=C_SAMP, lw=0)
			ax.set_ylabel(f"sampled {label}\nentropy [nats]", fontsize=8)
	ax = axes[-1]
	shade_domains(ax, t)
	for y, (col, c, lab) in enumerate((("K_natural", C_NAT, "natural"), ("K_abundance", C_ABU, "abundance"), ("K_activity", C_ACT, "activity"))):
		s = t.site[t[col]]
		ax.plot(s, np.full(len(s), y), "|", color=c, ms=9, mew=1.2)
	ax.set_yticks(range(3), ["natural", "abundance", "activity"], fontsize=8)
	ax.set_ylim(-.6, 2.6)
	ax.set_ylabel("K-like\nsets", fontsize=8)
	top = axes[0]
	for name, (a, b) in LANDMARKS.items():
		s = t.site[(t.uniprot_pos >= a) & (t.uniprot_pos <= b)]
		for ax in axes:
			ax.axvspan(s.min() - .5, s.max() + .5, color="#d9534f", alpha=.10, lw=0, zorder=0)
		top.text(s.mean(), 1.04, name, transform=top.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
	ticks = [p for p in range(10, 354, 20) if (t.uniprot_pos == p).any()]
	ax.set_xticks([int(t.site[t.uniprot_pos == p].iloc[0]) for p in ticks], ticks, fontsize=8)
	ax.set_xlim(-1, len(t))
	ax.set_xlabel("UniProt position (286-309 are not in the 1D5R construct; shading: phosphatase domain, C2 domain)")
	fig.align_ylabels(axes)
	fig.subplots_adjust(hspace=.12, top=.93, bottom=.07, left=.09, right=.99)
	fig.savefig(path, dpi=150)
	plt.close(fig)


def flatness(x):
	"""How flat a per-site profile is: coefficient of variation and the share of sites near its floor."""
	x = np.asarray(x, float); x = x[np.isfinite(x)]
	return {"sites": len(x), "mean": x.mean(), "sd": x.std(), "cv": x.std() / abs(x.mean()) if x.mean() else np.nan,
		"p10": np.percentile(x, 10), "median": np.median(x), "p90": np.percentile(x, 90)}


def cmd_build(args):
	ref, t, tier_summary = build_table(args)
	os.makedirs(args.out_dir, exist_ok=True)
	t.to_csv(os.path.join(args.out_dir, "pten_site_table.csv"), index=False, float_format="%.4f")
	tier_summary.to_csv(os.path.join(args.out_dir, "natural_tiers.tsv"), sep="\t", index=False, float_format="%.3f")
	plot_profile(t, tier_summary, os.path.join(args.out_dir, "pten_site_profile.png"))
	print(f"reference {REF_NAME}: {len(ref)} sites, {int(t.native.sum())} native, "
		f"UniProt {t.uniprot_pos[t.native].min()}-{t.uniprot_pos.max()}, "
		f"{(t.domain == 'phosphatase').sum()} phosphatase, {(t.domain == 'C2').sum()} C2\n")
	print(f"natural alignment tiers (weights 1/neighbours at the identity shown, entropy NaN above {args.max_gap:.0%} gaps):")
	print(tier_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
	print("\nflatness of each reference profile over the sites it covers:")
	cols = ["nat_vertebrate_entropy", "nat_orthologs_entropy", "nat_family_entropy", "nat_pdomain_entropy", "abundance_mean", "abundance2021_mean", "activity_mean"]
	print(pd.DataFrame({c: flatness(t[c]) for c in cols}).T.to_string(float_format=lambda x: f"{x:.3f}"))
	print("\nK-like sets (sites of the reference):")
	for col, rule in (("K_natural", f"family entropy < {args.natural_k} nats"),
			("K_abundance", "authors' positional class 'intolerant' (Matreyek 2018)"),
			("K_activity", f">= {args.activity_k:.0%} of scored missense variants below {ACTIVITY_LOF}")):
		k = t[t[col]]
		print(f"  {col:12s} {len(k):3d} sites  ({(k.domain == 'phosphatase').sum()} phosphatase, {(k.domain == 'C2').sum()} C2)   {rule}")
	L = len(t)
	print("\npairwise overlap of the sets (observed, expected by chance, P(overlap >= observed)):")
	names = ["K_natural", "K_abundance", "K_activity"]
	for i, a in enumerate(names):
		for b in names[i + 1:]:
			k, exp, p = set_overlap(t.site[t[a]], t.site[t[b]], L)
			print(f"  {a} & {b}: {k}  expected {exp:.1f}  P={p:.2g}")
	all3 = t[t.K_natural & t.K_abundance & t.K_activity]
	print(f"  in all three: {len(all3)} sites: " + " ".join(f"{a}{p}" for a, p in zip(all3.ref_aa, all3.uniprot_pos)))
	print("\nSpearman between the reference profiles (sites where both exist):")
	prof = t[["nat_orthologs_entropy", "nat_family_entropy", "abundance2021_mean", "activity_mean"]]
	print(prof.corr(method="spearman").to_string(float_format=lambda x: f"{x:+.2f}"))
	print(f"\nOutputs in {args.out_dir}: pten_site_table.csv, natural_tiers.tsv, pten_site_profile.png")


# ---------------------------------------------------------------- compare --- #

def chain_site_stats(run_dir, ref, burn_in_frac):
	"""Per-site statistics of one chain, rebuilt from start.txt and the accepted moves of data.dat.

	entropy     Shannon entropy [nats] of the residue at the site over the kept moves (every move
	            counts once, accepted or not, as a time average must)
	native_frac share of the kept moves the site spends on the reference residue
	proposals, accepted   over the WHOLE chain
	mean_dU     mean U(proposed) - U(current) of the single-site proposals at the site, whole chain
	"""
	with open(os.path.join(run_dir, "start.txt")) as f:
		start = [l.strip() for l in f if l.strip() and not l.startswith("#")][0]
	path = os.path.join(run_dir, "data.dat")
	opener = open
	if not os.path.isfile(path) and os.path.isfile(path + ".gz"):
		path, opener = path + ".gz", gzip.open
	rows, seen = [], set()
	with opener(path, "rt") as f:
		for line in f:
			s = line.rstrip("\n").split("\t")
			if len(s) < 14 or not s[0].isdigit() or s[0] in seen:       # header, comments, resumed duplicates
				continue
			seen.add(s[0])
			rows.append((s[6], s[8], float(s[9]), s[13] == "1"))
	L, N = len(ref), len(rows)
	first = int(burn_in_frac * N)
	seq = np.frombuffer(start.encode(), dtype=np.uint8).copy()
	ref_b = np.frombuffer(ref.encode(), dtype=np.uint8)
	counts = np.zeros((L, 256))
	since = np.full(L, first)              # move at which each site took its current residue (clipped to burn-in)
	prop, acc, dU = np.zeros(L), np.zeros(L), np.zeros(L)
	for k, (sites, aas, du, ok) in enumerate(rows):
		sites = [int(x) for x in sites.split(",")]
		if len(sites) == 1:
			prop[sites[0]] += 1; dU[sites[0]] += du; acc[sites[0]] += ok
		if ok:
			for s_, a in zip(sites, aas.split(",")):
				if k >= first:
					counts[s_, seq[s_]] += k - since[s_]
					since[s_] = k
				seq[s_] = ord(a)
	counts[np.arange(L), seq] += N - since
	p = counts / counts.sum(axis=1, keepdims=True)
	with np.errstate(divide="ignore", invalid="ignore"):
		H = -np.where(p > 0, p * np.log(p), 0.).sum(axis=1)
	return pd.DataFrame({"entropy": H, "native_frac": p[np.arange(L), ref_b], "proposals": prop, "accepted": acc,
		"mean_dU": np.where(prop > 0, dU / np.maximum(prop, 1), np.nan)}), {
		"moves": N, "start_is_reference": start == ref, "final_Hd": int((seq != ref_b).sum()),
		"accepted_kept": int(sum(r[3] for r in rows[first:]))}


def partial_spearman(x, y, z):
	"""Spearman of x and y after the rank of z has been regressed out of both ranks."""
	x, y, z = (pd.Series(np.asarray(v, float)) for v in (x, y, z))
	ok = x.notna() & y.notna() & z.notna()
	if ok.sum() < 5:
		return np.nan
	rx, ry, rz = x[ok].rank(), y[ok].rank(), z[ok].rank()
	ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
	ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
	return float(np.corrcoef(ex, ey)[0, 1])


def shift_p(x, member):
	"""Two-sided p for mean(x[member]) - mean(x[~member]) against every circular shift of the set
	along the sequence. Neighbouring sites resemble each other (a buried strand, an exposed loop),
	so shuffling sites independently would overstate the significance; shifting keeps the set's
	run structure. The smallest value it can give is 1/L."""
	x, member = np.asarray(x, float), np.asarray(member, bool)
	if member.sum() < 2 or (~member).sum() < 2:
		return np.nan
	diff = lambda m: x[m].mean() - x[~m].mean()
	obs = diff(member)
	null = np.array([diff(np.roll(member, k)) for k in range(1, len(x))])
	return (1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / len(x)


def cmd_compare(args):
	ref, t, tier_summary = build_table(args)
	os.makedirs(args.out_dir, exist_ok=True)
	runs = {}
	for item in args.runs:
		label, _, path = item.partition("=")
		if not path:
			raise ValueError(f"--runs entries are LABEL=DIR, got {item!r}")
		runs[label] = path
	runs = dict(sorted(runs.items(), key=lambda kv: label_T(kv[0])))
	stats, ent = {}, pd.DataFrame(index=t.site)
	print("chains (entropy over the last "
		f"{1 - args.burn_in_frac:.0%} of the moves):")
	for label, path in runs.items():
		st, info = chain_site_stats(path, ref, args.burn_in_frac)
		stats[label] = st
		ent[label] = st.entropy.values
		for c in ("entropy", "native_frac", "proposals", "accepted", "mean_dU"):
			t[f"{label}_{c}"] = st[c].values
		print(f"  {label:6s} {info['moves']:7d} moves, start {'reference' if info['start_is_reference'] else 'other'}, "
			f"final Hd {info['final_Hd']:3d}, {info['accepted_kept']:5d} accepted in the kept part, "
			f"mean entropy {st.entropy.mean():.3f}, sites at entropy 0: {(st.entropy == 0).sum()}/{len(ref)}, "
			f"sites never accepted: {(st.accepted == 0).sum()}")

	refs = {"orthologs entropy": t.nat_orthologs_entropy, "family entropy": t.nat_family_entropy,
		"abundance mean": t.abundance2021_mean, "activity mean": t.activity_mean}
	print("\nSpearman, sampled entropy vs each reference (all are 'higher = more tolerant'; n = sites with data).")
	print("A second line gives the same for -mean_dU of the proposals, the sampler's own mutation cost.")
	rows = []
	for label, st in stats.items():
		rows.append({"chain": label, "quantity": "entropy", **{k: spearman(st.entropy, v) for k, v in refs.items()}})
		rows.append({"chain": label, "quantity": "-mean_dU", **{k: spearman(-st.mean_dU, v) for k, v in refs.items()}})
	sp = pd.DataFrame(rows)
	print(sp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
	sp.to_csv(os.path.join(args.out_dir, "sampled_vs_references_spearman.tsv"), sep="\t", index=False, float_format="%.4f")
	n_eff = {k: int(np.isfinite(v).sum()) for k, v in refs.items()}
	print("  n: " + ", ".join(f"{k} {v}" for k, v in n_eff.items())
		+ "; |rho| above about " + f"{1.96 / np.sqrt(min(n_eff.values()) - 1):.2f} is beyond chance at the 5% level")

	print("\nHow much of that is burial? Spearman of each quantity with the relative solvent accessibility (RSA) of the")
	print(f"crystal, and the sampled-vs-reference Spearman with RSA regressed out (the {int(t.rsa.notna().sum())} sites resolved in 1D5R):")
	print("  RSA vs " + ", ".join(f"{k} {spearman(t.rsa, v):+.2f}" for k, v in refs.items()))
	rows = []
	for label, st in stats.items():
		rows.append({"chain": label, "entropy vs RSA": spearman(st.entropy, t.rsa),
			**{f"{k} | RSA": partial_spearman(st.entropy, v, t.rsa) for k, v in refs.items()}})
	pr = pd.DataFrame(rows)
	print(pr.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
	pr.to_csv(os.path.join(args.out_dir, "sampled_vs_references_partial_rsa.tsv"), sep="\t", index=False, float_format="%.4f")

	print("\nIs the sampled entropy flat across each K-like set? mean inside / mean outside,")
	print("p against circular shifts of the set along the sequence (smallest possible 1/L = " + f"{1 / len(ref):.3f}):")
	rows = []
	for label, st in stats.items():
		for col in ("K_natural", "K_abundance", "K_activity"):
			inside, outside = st.entropy[t[col].values], st.entropy[~t[col].values]
			rows.append({"chain": label, "set": col, "n_in": len(inside), "mean_in": inside.mean(), "mean_out": outside.mean(),
				"ratio": inside.mean() / outside.mean() if outside.mean() else np.nan,
				"frac_zero_in": (inside == 0).mean(), "frac_zero_out": (outside == 0).mean(),
				"shift_p": shift_p(st.entropy.values, t[col].values)})
	fl = pd.DataFrame(rows)
	print(fl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
	fl.to_csv(os.path.join(args.out_dir, "sampled_entropy_in_K_sets.tsv"), sep="\t", index=False, float_format="%.4f")

	cold = args.cold or list(runs)[:1]
	k_sampled = t.site[(ent[cold] < args.k_site_threshold).all(axis=1)].values
	t["K_sampled"] = t.site.isin(k_sampled)
	print(f"\nsampled K-sites (entropy < {args.k_site_threshold} at every one of {cold}): {len(k_sampled)}/{len(ref)}")
	for col in ("K_natural", "K_abundance", "K_activity"):
		k, exp, p = set_overlap(k_sampled, t.site[t[col]], len(ref))
		print(f"  overlap with {col}: {k}/{int(t[col].sum())}  expected {exp:.1f}  P={p:.2g}")

	t.to_csv(os.path.join(args.out_dir, "pten_site_table_sampled.csv"), index=False, float_format="%.4f")
	show = ent[[c for c in ent.columns if c in (args.show or ent.columns)]]
	plot_profile(t, tier_summary, os.path.join(args.out_dir, "pten_site_profile_sampled.png"), sampled=show)

	# scatter: sampled entropy (one chain per column) against the three references
	fig, axes = plt.subplots(3, len(show.columns), figsize=(3.1 * len(show.columns), 8.2), squeeze=False, sharey="row")
	for j, label in enumerate(show.columns):
		for i, (name, c) in enumerate((("family entropy", C_NAT), ("abundance mean", C_ABU), ("activity mean", C_ACT))):
			ax = axes[i, j]
			ax.scatter(show[label], refs[name], s=7, color=c, alpha=.55, lw=0)
			ax.set_title(f"{label}: rho = {spearman(show[label], refs[name]):+.2f}", fontsize=9)
			if j == 0:
				ax.set_ylabel(name)
			if i == 2:
				ax.set_xlabel("sampled entropy [nats]")
	fig.tight_layout()
	fig.savefig(os.path.join(args.out_dir, "sampled_vs_references.png"), dpi=140)
	plt.close(fig)
	print(f"\nOutputs in {args.out_dir}: pten_site_table_sampled.csv, sampled_vs_references_spearman.tsv, "
		"sampled_vs_references_partial_rsa.tsv, sampled_entropy_in_K_sets.tsv, pten_site_profile_sampled.png, sampled_vs_references.png")


# ----------------------------------------------------------------- family --- #

CLASSICAL = re.compile(r"non-receptor type|receptor-type tyrosine|PTP1B|YopH|SHP-?[12]\b|leukocyte common antigen|"
	r"tyrosine-protein phosphatase (10D|69D|99A|non-receptor)|PTPR|PTPN|LAR\b|STEP\b", re.I)


def parse_hhr(path):
	"""HHsearch .hhr -> list of hits: id, description, probability, {query column (0-based): template residue}."""
	with open(path) as f:
		text = f.read()
	hits = []
	for block in re.split(r"\nNo \d+\s*\n", text)[1:]:
		lines = block.split("\n")
		desc = lines[0][1:]
		prob = float(re.search(r"Probab=([\d.]+)", lines[1]).group(1))
		tid = desc.split()[0]
		q = [l for l in lines if l.startswith("Q query")]
		tl = [l for l in lines if l.startswith(f"T {tid[:14]}") or (l.startswith("T ") and not l.startswith(("T Consensus", "T ss_")))]
		tl = [l for l in tl if not l.startswith(("T Consensus", "T ss_"))]
		cols, tpos = {}, {}
		for ql, tline in zip(q, tl):
			qs = ql.split(); ts = tline.split()
			qi, qseq = int(qs[2]) - 1, qs[3]
			ti, tseq = int(ts[2]), ts[3]
			for a, b in zip(qseq, tseq):
				if a != "-" and b != "-":
					cols[qi] = b; tpos[qi] = ti
				qi += a != "-"; ti += b != "-"
		hits.append({"id": tid, "desc": desc, "prob": prob, "cols": cols, "tpos": tpos})
	return hits


def cmd_family(args):
	ref, t, tier_summary = build_table(args)
	os.makedirs(args.out_dir, exist_ok=True)
	L = len(ref)
	hits = [h for h in parse_hhr(os.path.join(args.raw_dir, "1d5r_A_pdb70_hits.hhr")) if h["prob"] >= args.min_prob]
	# PTEN itself and its close relatives (tensin, auxilin, VSP) cover both domains; drop them from the
	# superfamily count so that it measures the DISTANT phosphatases only.
	pd_sites = t.site[t.domain == "phosphatase"].values
	c2_sites = t.site[t.domain == "C2"].values
	for h in hits:
		h["classical"] = bool(CLASSICAL.search(h["desc"]))
		h["pten_like"] = sum(s in h["cols"] for s in c2_sites) > 0.5 * len(c2_sites)
	far = [h for h in hits if not h["pten_like"]]
	classical = [h for h in far if h["classical"]]
	other = [h for h in far if not h["classical"]]
	print(f"HHsearch of the {REF_NAME} profile against PDB70: {len(hits)} templates at probability >= {args.min_prob}")
	print(f"  PTEN-like (C2 domain aligned too): {len(hits) - len(far)}")
	print(f"  classical PTPs, the Pfam PF00102 fold:   {len(classical)}")
	print(f"  other phosphatases (dual-specificity, PRL, CDC14, myotubularin, RNA 5'-phosphatases, ...): {len(other)}")

	def coverage(group):
		cov = np.zeros(L)
		for h in group:
			cov[list(h["cols"])] += 1
		return cov / max(len(group), 1)

	def cons_identity(group):
		# share of the templates covering a PTEN site that carry PTEN's own residue there
		same, n = np.zeros(L), np.zeros(L)
		for h in group:
			for s, a in h["cols"].items():
				n[s] += 1; same[s] += a == ref[s]
		return np.where(n >= 5, same / np.maximum(n, 1), np.nan)

	t["cov_classical"], t["cov_other"] = coverage(classical), coverage(other)
	t["same_as_PTEN_classical"], t["same_as_PTEN_other"] = cons_identity(classical), cons_identity(other)
	for name, group in (("classical PTPs", classical), ("other phosphatases", other)):
		n_cols = np.array([len(h["cols"]) for h in group])
		starts = np.array([min(h["cols"]) for h in group]); ends = np.array([max(h["cols"]) for h in group])
		up = t.uniprot_pos.values
		print(f"\n{name}: aligned PTEN columns per template, median {np.median(n_cols):.0f} (range {n_cols.min()}-{n_cols.max()}); "
			f"typical span UniProt {up[int(np.median(starts))]}-{up[int(np.median(ends))]}")
		cov = coverage(group)
		reached = cov[pd_sites] >= 0.5
		print(f"  phosphatase-domain sites reached by at least half of them: {reached.sum()}/{len(pd_sites)} "
			f"({reached.mean():.0%}); C2 sites: {(cov[c2_sites] >= 0.5).sum()}/{len(c2_sites)}")

	# PF00102 conservation carried onto PTEN through PTP1B (2CM2 chain A)
	ptp1b = next((h for h in hits if h["id"].upper().startswith(args.bridge.upper())), None)
	bridge_path = os.path.join(args.raw_dir, f"{args.bridge.lower()[:4]}_{args.bridge[5:]}_uniref90_hits.a3m")
	if ptp1b is not None and os.path.isfile(bridge_path):
		with open(bridge_path) as f:
			f.readline(); q = f.readline().strip()
		names, seqs = read_a3m(bridge_path, len(q))
		X = encode(seqs, len(q))
		keep = (X != GAP).mean(1) >= 0.7
		w = 1. / neighbour_counts(X[keep], np.ones(int(keep.sum())), args.weight_identity)
		H, gap, cons, cons_f = weighted_entropy(X[keep], w, args.max_gap)
		# template numbering in the hhr is the index along the template's own sequence
		t["pf00102_entropy"] = np.nan
		t["ptp1b_residue"] = ""
		n_ok = 0
		for s, ti in ptp1b["tpos"].items():
			if ti - 1 < len(q) and q[ti - 1] == ptp1b["cols"][s]:
				t.loc[s, "pf00102_entropy"] = H[ti - 1]; t.loc[s, "ptp1b_residue"] = f"{q[ti - 1]}{ti}"; n_ok += 1
		print(f"\nPF00102 through PTP1B ({ptp1b['id']}, HHsearch probability {ptp1b['prob']:.1f}): its UniRef90 alignment has "
			f"{int(keep.sum())} sequences at coverage >= 0.7, N_eff {w.sum():.0f}; {n_ok} PTEN sites receive a PF00102 entropy "
			f"(UniProt {t.uniprot_pos[t.pf00102_entropy.notna()].min()}-{t.uniprot_pos[t.pf00102_entropy.notna()].max()})")
		for name, col in (("PTEN-like family entropy", "nat_family_entropy"), ("phosphatase-domain tier entropy", "nat_pdomain_entropy"),
				("abundance mean", "abundance2021_mean"), ("activity mean", "activity_mean")):
			ok = t.pf00102_entropy.notna() & t[col].notna()
			print(f"  Spearman, PF00102 entropy vs {name}: {spearman(t.pf00102_entropy, t[col]):+.2f}  (n={int(ok.sum())})")
		shared = t[(t.pf00102_entropy < 0.5) & (t.nat_pdomain_entropy < 0.5)]
		print("  sites conserved on both sides (entropy < 0.5 nats in PF00102 and in PTEN's own phosphatase tier): "
			+ " ".join(f"{a}{p}" for a, p in zip(shared.ref_aa, shared.uniprot_pos)))
	else:
		print(f"\n(no {args.bridge} alignment in {args.raw_dir}: PF00102 entropy not carried over)")

	t.to_csv(os.path.join(args.out_dir, "pten_site_table_family.csv"), index=False, float_format="%.4f")
	pdt = t[t.domain.isin(["N-term", "phosphatase"])]
	fig, axes = plt.subplots(3, 1, figsize=(12, 6.3), sharex=True)
	ax = axes[0]
	ax.fill_between(pdt.site, pdt.cov_other, step="mid", color="#8c6bb1", alpha=.75, lw=0, label=f"other phosphatases ({len(other)})")
	ax.step(pdt.site, pdt.cov_classical, where="mid", color="k", lw=1.2, label=f"classical PTPs, PF00102 ({len(classical)})")
	ax.set_ylabel("share of templates\naligned at the site", fontsize=8); ax.legend(fontsize=8, frameon=False, loc="upper left")
	ax = axes[1]
	ax.bar(pdt.site, pdt.nat_pdomain_entropy.fillna(0), width=1., color=C_NAT, lw=0)
	ax.set_ylabel("PTEN side: phosphatase\ntier entropy [nats]", fontsize=8)
	ax = axes[2]
	if "pf00102_entropy" in t:
		ax.bar(pdt.site, pdt.pf00102_entropy.fillna(0), width=1., color="k", lw=0)
	ax.set_ylabel("PF00102 side: entropy\nvia PTP1B [nats]", fontsize=8)
	for name, (a, b) in LANDMARKS.items():
		s = t.site[(t.uniprot_pos >= a) & (t.uniprot_pos <= b)]
		if s.max() <= pdt.site.max():
			for ax in axes:
				ax.axvspan(s.min() - .5, s.max() + .5, color="#d9534f", alpha=.10, lw=0, zorder=0)
			axes[0].text(s.mean(), 1.03, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
	ticks = [p for p in range(10, 190, 10)]
	axes[2].set_xticks([int(t.site[t.uniprot_pos == p].iloc[0]) for p in ticks], ticks, fontsize=8)
	axes[2].set_xlim(pdt.site.min() - .5, pdt.site.max() + .5)
	axes[2].set_xlabel("UniProt position, PTEN phosphatase domain")
	fig.align_ylabels(axes)
	fig.subplots_adjust(hspace=.12, top=.92, bottom=.09, left=.09, right=.99)
	fig.savefig(os.path.join(args.out_dir, "pten_ptp_superfamily.png"), dpi=150)
	plt.close(fig)
	print(f"\nOutputs in {args.out_dir}: pten_site_table_family.csv, pten_ptp_superfamily.png")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = ap.add_subparsers(dest="cmd", required=True)
	common = argparse.ArgumentParser(add_help=False)
	common.add_argument("--raw-dir", default=os.path.join(MONTE_CARLO_DIR, "pten_natural", "raw"))
	common.add_argument("--out-dir", default=os.path.join(MONTE_CARLO_DIR, "pten_natural"))
	common.add_argument("--weight-identity", type=float, default=0.8,
		help="redundancy weights: a sequence counts 1/(number of sequences at identity >= this), as in ha_site_entropy.py")
	common.add_argument("--max-gap", type=float, default=0.5,
		help="weighted gap fraction above which a site gets entropy NaN. 0.5 and not the 0.3 of ha_site_entropy.py: natural PTEN "
			"has the 286-309 loop the construct lacks, so many local alignments stop there and the 310-353 columns sit at "
			"31-48%% 'gaps' that are truncation, not deletion")
	common.add_argument("--ortholog-identity", type=float, default=0.7, help="identity to PTEN above which a hit counts as an ortholog")
	common.add_argument("--ortholog-weight-identity", type=float, default=0.95, help="redundancy threshold inside the ortholog tiers")
	common.add_argument("--natural-k", type=float, default=0.5, help="family entropy [nats] below which a site is in the natural K-like set")
	common.add_argument("--activity-k", type=float, default=0.5,
		help=f"share of scored missense variants below {ACTIVITY_LOF} above which a site is in the activity K-like set")
	p = sub.add_parser("fetch", parents=[common]); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_fetch)
	p = sub.add_parser("build", parents=[common]); p.set_defaults(func=cmd_build)
	p = sub.add_parser("compare", parents=[common])
	p.add_argument("--runs", nargs="+", required=True, help="LABEL=RUN_DIR, a run dir holding start.txt and data.dat(.gz)")
	p.add_argument("--burn-in-frac", type=float, default=0.5)
	p.add_argument("--cold", nargs="+", help="labels whose entropy defines the sampled K-sites (default: the coldest)")
	p.add_argument("--show", nargs="+", help="labels to draw (default: all)")
	p.add_argument("--k-site-threshold", type=float, default=0.05)
	p.set_defaults(func=cmd_compare)
	p = sub.add_parser("family", parents=[common])
	p.add_argument("--min-prob", type=float, default=90., help="HHsearch probability a template needs")
	p.add_argument("--bridge", default="2CM2_A", help="PF00102 member whose alignment is carried onto PTEN (PTP1B)")
	p.set_defaults(func=cmd_family)
	args = ap.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
