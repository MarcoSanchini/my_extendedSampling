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

  pfam     the natural reference rebuilt from Pfam: asks InterPro which Pfam
           families lie on human PTEN, downloads their alignments, projects
           them on the reference through PTEN's own row and compares the
           per-site entropy with the tiers above and with the two scans.
           Needs www.ebi.ac.uk, which the cloud container cannot reach: run
           it on a machine with ordinary internet access.

Thresholds. --vertebrate-identity (0.9), --ortholog-identity (0.7) and
--min-coverage (0.8) decide which hits enter the vertebrate, orthologs and
family tiers. Give 0 or 'none' to remove one: with no identity threshold a
tier becomes the family tier, with no coverage threshold every hit counts.

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
    python pten_k_sites.py compare --runs-dir pten1d5r_study/equil --cold T1 T2    # every chain
    python pten_k_sites.py family
    python pten_k_sites.py pfam                                # local machine only
    python pten_k_sites.py build --vertebrate-identity 0.95    # or: none
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
	# A threshold of 0 (or "none" on the command line) switches that filter off.
	full = cov >= args.min_coverage
	tiers = {
		# the same protein in vertebrates, where PTEN is nearly invariant
		"vertebrate": full & (ident >= args.vertebrate_identity),
		# the same protein in other species: both domains present and clearly PTEN
		"orthologs": full & (ident >= args.ortholog_identity),
		# PTEN-like family: both domains present (PTEN, TPTE/VSP, tensins, auxilin/GAK)
		"family": full,
		# anything whose phosphatase domain aligns; C2 columns are mostly gaps here
		"pdomain": cov_pd >= args.pdomain_coverage,
	}
	min_id = {"vertebrate": args.vertebrate_identity, "orthologs": args.ortholog_identity, "family": 0., "pdomain": 0.}
	min_cov = {"vertebrate": args.min_coverage, "orthologs": args.min_coverage, "family": args.min_coverage,
		"pdomain": args.pdomain_coverage}
	# The ortholog tiers are within ~70-100% identity of each other, so the 0.8 weighting would
	# collapse them to one or two clusters (N_eff 1-6): they are weighted at 0.95 instead.
	# A tier whose identity threshold has been removed is as broad as the family, so it is weighted like it.
	theta = {k: args.ortholog_weight_identity for k in ("vertebrate", "orthologs") if min_id[k] >= 0.5}
	out, summary = {}, []
	for tier, keep in tiers.items():
		Xt = X[keep]
		w = 1. / neighbour_counts(Xt, np.ones(len(Xt)), theta.get(tier, args.weight_identity))
		H, gap, cons, cons_f = weighted_entropy(Xt, w, args.max_gap)
		if tier == "pdomain":
			H[~pd_cols] = np.nan
		out[tier] = (H, gap, cons, cons_f)
		summary.append({"tier": tier, "min_identity": min_id[tier], "min_coverage": min_cov[tier],
			"sequences": int(keep.sum()), "weight_identity": theta.get(tier, args.weight_identity), "n_eff": w.sum(),
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
	fig.savefig(path, dpi=150 if n_extra <= 8 else 100)
	plt.close(fig)


def flatness(x):
	"""How flat a per-site profile is: coefficient of variation and the share of sites near its floor."""
	x = np.asarray(x, float); x = x[np.isfinite(x)]
	return {"sites": len(x), "mean": x.mean(), "sd": x.std(), "cv": x.std() / abs(x.mean()) if x.mean() else np.nan,
		"p10": np.percentile(x, 10), "median": np.median(x), "p90": np.percentile(x, 90)}


def flatness_table(t, tier_summary):
	"""flatness() of every reference profile, with how much data stands behind each."""
	ts = tier_summary.set_index("tier")
	rows = {}
	for tier in ts.index:
		rows[f"nat_{tier}_entropy"] = {"n": int(ts.sequences[tier]), "n_eff": ts.n_eff[tier], **flatness(t[f"nat_{tier}_entropy"])}
	for col, ncol in (("abundance_mean", "abundance_n"), ("abundance2021_mean", "abundance2021_n"), ("activity_mean", "activity_n")):
		rows[col] = {"n": int(t[ncol].sum()), "n_eff": np.nan, **flatness(t[col])}
	return pd.DataFrame(rows).T


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
	fl = flatness_table(t, tier_summary)
	print(fl.to_string(float_format=lambda x: f"{x:.3f}"))
	fl.to_csv(os.path.join(args.out_dir, "flatness.tsv"), sep="\t", float_format="%.4f")
	print("  n = sequences in the alignment (natural tiers) or scored missense variants on the construct's sites (scans);")
	print("  n_eff = effective number of sequences after the redundancy weighting")
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
	print(f"\nOutputs in {args.out_dir}: pten_site_table.csv, natural_tiers.tsv, flatness.tsv, pten_site_profile.png")


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


def discover_runs(args):
	"""LABEL -> run dir, from --runs and/or every T*/chain* under --runs-dir.

	Discovered labels are the temperature directory for chain0 (T4) and T4.c1, T4.c2 for the
	other chains, so that labels given by hand (T1, T2, ...) keep meaning chain0."""
	runs = {}
	for item in args.runs or []:
		label, _, path = item.partition("=")
		if not path:
			raise ValueError(f"--runs entries are LABEL=DIR, got {item!r}")
		runs[label] = path
	if args.runs_dir:
		for tdir in sorted(os.listdir(args.runs_dir)):
			for cdir in sorted(os.listdir(os.path.join(args.runs_dir, tdir))) if os.path.isdir(os.path.join(args.runs_dir, tdir)) else []:
				path = os.path.join(args.runs_dir, tdir, cdir)
				m = re.fullmatch(r"chain(\d+)", cdir)
				if not m or not os.path.isfile(os.path.join(path, "start.txt")):
					continue
				if not (os.path.isfile(os.path.join(path, "data.dat")) or os.path.isfile(os.path.join(path, "data.dat.gz"))):
					continue
				runs.setdefault(tdir if m.group(1) == "0" else f"{tdir}.c{m.group(1)}", path)
	if not runs:
		raise SystemExit("compare needs --runs LABEL=DIR ... or --runs-dir DIR (a directory of T*/chain*)")
	return dict(sorted(runs.items(), key=lambda kv: (label_T(kv[0]), kv[0])))


def grid_scatter(show, y, name, color, path, ncols=6):
	"""Sampled entropy of every chain against one reference, one small panel per chain."""
	n = len(show.columns)
	ncols = min(ncols, n)
	nrows = -(-n // ncols)
	fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.5 * nrows), squeeze=False, sharex=True, sharey=True)
	for ax in axes.ravel()[n:]:
		ax.axis("off")
	for ax, label in zip(axes.ravel(), show.columns):
		ax.scatter(show[label], y, s=6, color=color, alpha=.5, lw=0)
		ax.set_title(f"{label}: rho = {spearman(show[label], y):+.2f}", fontsize=8)
	for ax in axes[:, 0]:
		ax.set_ylabel(name, fontsize=8)
	for ax in axes[-1]:
		ax.set_xlabel("sampled entropy [nats]", fontsize=8)
	fig.tight_layout()
	fig.savefig(path, dpi=130)
	plt.close(fig)


def plot_by_T(sp, pr, ref_started, path):
	"""Spearman of the sampled entropy with each reference against temperature, reference-started
	chains only: raw (solid) and with solvent accessibility regressed out (dashed)."""
	e = sp[(sp.quantity == "entropy") & sp.chain.isin(ref_started)].copy()
	e["T"] = e.chain.map(label_T)
	q = pr[pr.chain.isin(ref_started)].copy()
	q["T"] = q.chain.map(label_T)
	fig, ax = plt.subplots(figsize=(6.4, 4.2))
	for name, c in (("family entropy", C_NAT), ("abundance mean", C_ABU), ("activity mean", C_ACT), ("orthologs entropy", "#9a9a9a")):
		# points: every chain; lines: the mean over the chains at that temperature
		ax.plot(e["T"], e[name], "o", color=c, ms=4)
		ax.plot(q["T"], q[f"{name} | RSA"], "o", color=c, ms=3, alpha=.8)
		ax.plot(e.groupby("T")[name].mean(), "-", color=c, lw=1.4, label=name)
		ax.plot(q.groupby("T")[f"{name} | RSA"].mean(), "--", color=c, lw=1., alpha=.8)
	ax.plot([], [], "k--", lw=1., label="dashed: burial (RSA) regressed out")
	ax.axhline(0, color="k", lw=.5)
	ax.set_xscale("log")
	ax.set_xlabel("sampling temperature T")
	ax.set_ylabel("Spearman with the sampled per-site entropy")
	ax.legend(fontsize=8, frameon=False)
	fig.tight_layout()
	fig.savefig(path, dpi=140)
	plt.close(fig)


def cmd_compare(args):
	ref, t, tier_summary = build_table(args)
	os.makedirs(args.out_dir, exist_ok=True)
	runs = discover_runs(args)
	stats, ent, ref_started = {}, pd.DataFrame(index=t.site), []
	print("chains (entropy over the last "
		f"{1 - args.burn_in_frac:.0%} of the moves):")
	for label, path in runs.items():
		st, info = chain_site_stats(path, ref, args.burn_in_frac)
		if args.reference_only and not info["start_is_reference"]:
			print(f"  {label:6s} skipped: not started from the reference")
			continue
		if info["start_is_reference"]:
			ref_started.append(label)
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

	cold = [c for c in (args.cold or (ref_started or list(stats))[:1]) if c in ent.columns]
	if not cold:
		raise SystemExit(f"none of --cold {args.cold} is among the chains {list(ent.columns)}")
	k_sampled = t.site[(ent[cold] < args.k_site_threshold).all(axis=1)].values
	t["K_sampled"] = t.site.isin(k_sampled)
	print(f"\nsampled K-sites (entropy < {args.k_site_threshold} at every one of {cold}): {len(k_sampled)}/{len(ref)}")
	for col in ("K_natural", "K_abundance", "K_activity"):
		k, exp, p = set_overlap(k_sampled, t.site[t[col]], len(ref))
		print(f"  overlap with {col}: {k}/{int(t[col].sum())}  expected {exp:.1f}  P={p:.2g}")

	t.to_csv(os.path.join(args.out_dir, "pten_site_table_sampled.csv"), index=False, float_format="%.4f")
	show = ent[[c for c in ent.columns if c in (args.show or ent.columns)]]
	plot_profile(t, tier_summary, os.path.join(args.out_dir, "pten_site_profile_sampled.png"), sampled=show)

	for name, c, tag in (("family entropy", C_NAT, "family"), ("abundance mean", C_ABU, "abundance"), ("activity mean", C_ACT, "activity")):
		grid_scatter(show, refs[name], name, c, os.path.join(args.out_dir, f"sampled_vs_{tag}.png"))
	if len(ref_started) > 1:
		plot_by_T(sp, pr, ref_started, os.path.join(args.out_dir, "sampled_vs_references_by_T.png"))
	print(f"\nOutputs in {args.out_dir}: pten_site_table_sampled.csv, sampled_vs_references_spearman.tsv, "
		"sampled_vs_references_partial_rsa.tsv, sampled_entropy_in_K_sets.tsv, pten_site_profile_sampled.png, "
		"sampled_vs_family.png, sampled_vs_abundance.png, sampled_vs_activity.png, sampled_vs_references_by_T.png")


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


# ------------------------------------------------------------------- pfam --- #

INTERPRO = "https://www.ebi.ac.uk/interpro/api/"
PTEN_ACC = "P60484"


def http_get(url, retries=5):
	"""GET with retries. InterPro answers 408 while it prepares a large file: wait and ask again."""
	import time
	import urllib.error
	req = urllib.request.Request(url, headers={"User-Agent": "python-urllib", "Accept": "*/*"})
	for attempt in range(retries):
		try:
			with urllib.request.urlopen(req, timeout=300) as r:
				data = r.read()
			return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
		except urllib.error.HTTPError as e:
			if e.code in (408, 429, 500, 502, 503, 504) and attempt < retries - 1:
				time.sleep(20 * (attempt + 1))
				continue
			raise SystemExit(f"{url}\n  answered HTTP {e.code}. Open it in a browser to check; a file saved by hand "
				"can be given as --sto ACCESSION=PATH.")
		except OSError as e:
			if attempt < retries - 1:
				time.sleep(10 * (attempt + 1))
				continue
			raise SystemExit(f"{url}\n  could not be reached ({e}). This step needs www.ebi.ac.uk, so run it on a machine "
				"with ordinary internet access, not in the cloud container.")


def pfam_entries_of_pten(raw_dir):
	"""The Pfam families InterPro places on human PTEN: [(accession, name, [(start, end), ...]), ...]."""
	import json
	path = os.path.join(raw_dir, f"interpro_pfam_{PTEN_ACC}.json")
	if not os.path.isfile(path):
		with open(path, "wb") as f:
			f.write(http_get(f"{INTERPRO}entry/pfam/protein/uniprot/{PTEN_ACC}"))
	with open(path) as f:
		doc = json.load(f)
	out = []
	for r in doc.get("results", []):
		md = r.get("metadata", {})
		frags = [(fr["start"], fr["end"]) for pr_ in r.get("proteins", []) for loc in pr_.get("entry_protein_locations") or []
			for fr in loc.get("fragments", [])]
		name = md.get("name")
		out.append((md.get("accession"), name.get("name", "") if isinstance(name, dict) else str(name), sorted(frags)))
	return sorted(out, key=lambda e: e[2][0] if e[2] else 0)


def read_stockholm(text):
	"""Stockholm -> {name: aligned sequence}; blocks of an interleaved file are joined."""
	seqs = {}
	for line in text.splitlines():
		if not line or line[0] == "#" or line.startswith("//"):
			continue
		parts = line.split()
		if len(parts) == 2:
			seqs[parts[0]] = seqs.get(parts[0], "") + parts[1]
	return seqs


def ungap(s):
	return re.sub("[^A-Za-z]", "", s).upper()


def find_anchor(seqs):
	"""The row that is human PTEN: by name, else by its residues being a stretch of PTEN's sequence."""
	for name, s in seqs.items():
		if name.upper().startswith(("PTEN_HUMAN", PTEN_ACC)) and ungap(s) in PTEN_UNIPROT:
			return name
	for name, s in seqs.items():
		u = ungap(s)
		if len(u) >= 30 and u in PTEN_UNIPROT:
			return name
	return None


def hmmalign_with_pten(acc, seqs, raw_dir):
	"""PTEN is not a row of this alignment: realign its sequences together with PTEN on the family's
	profile HMM (hmmalign, from HMMER). Returns the new alignment and the name of the PTEN row."""
	import shutil
	import subprocess
	import tempfile
	if shutil.which("hmmalign") is None:
		raise SystemExit(f"{acc}: human PTEN is not in this alignment, and placing it there needs HMMER's hmmalign.\n"
			"  Install it (conda install -c bioconda hmmer, or apt install hmmer), or try --alignment uniprot.")
	hmm = os.path.join(raw_dir, f"{acc}.hmm")
	if not os.path.isfile(hmm):
		with open(hmm, "wb") as f:
			f.write(http_get(f"{INTERPRO}entry/pfam/{acc}/?annotation=hmm"))
	with tempfile.TemporaryDirectory() as tmp:
		fa = os.path.join(tmp, "all.fa")
		with open(fa, "w") as f:
			f.write(f">PTEN_HUMAN\n{PTEN_UNIPROT}\n")
			for k, (name, s) in enumerate(seqs.items()):
				f.write(f">s{k}\n{ungap(s)}\n")
		res = subprocess.run(["hmmalign", "--trim", "--outformat", "Pfam", hmm, fa], capture_output=True, text=True)
		if res.returncode != 0:
			raise SystemExit(f"hmmalign failed for {acc}:\n{res.stderr[-800:]}")
	return read_stockholm(res.stdout), "PTEN_HUMAN"


def pfam_onto_reference(acc, seqs, anchor, t, args, rng):
	"""Project a Pfam alignment on the reference sites through its PTEN row; weighted per-site entropy."""
	L = len(t)
	a = seqs[anchor]
	start = PTEN_UNIPROT.find(ungap(a))
	if start < 0:
		raise SystemExit(f"{acc}: the anchor row {anchor} is not a stretch of PTEN's sequence")
	site_of = dict(zip(t.uniprot_pos[t.native], t.site[t.native]))
	cols, sites, pos = [], [], start
	for c, ch in enumerate(a):
		if ch.isalpha():
			pos += 1                                  # UniProt position of this anchor residue
			if pos in site_of:
				cols.append(c); sites.append(site_of[pos])
	cols = np.array(cols)
	rows = [s for name, s in seqs.items() if len(s) == len(a)]
	sub = ["".join(s[c] if s[c].isalpha() else "-" for c in cols).upper() for s in rows]
	Xs = encode(sub, len(cols))
	cov = (Xs != GAP).mean(1)
	Xs = Xs[cov >= args.pfam_coverage]
	n_all = len(Xs)
	if n_all > args.max_seqs:                         # the one-hot products are quadratic in the number of rows
		Xs = Xs[rng.choice(n_all, args.max_seqs, replace=False)]
	a_row = encode(["".join(a[c] for c in cols).upper()], len(cols))[0]
	ident = ((Xs == a_row) & (Xs != GAP)).sum(1) / np.maximum((Xs != GAP).sum(1), 1)
	w = 1. / neighbour_counts(Xs, np.ones(len(Xs)), args.weight_identity)
	H, gap, cons, cons_f = weighted_entropy(Xs, w, args.max_gap)
	out = np.full(L, np.nan)
	out[sites] = H
	info = {"accession": acc, "rows_in_alignment": len(rows), f"rows_at_coverage_{args.pfam_coverage:g}": n_all,
		"rows_used": len(Xs), "n_eff": w.sum(), "median_identity_to_PTEN": float(np.median(ident)),
		"reference_sites": len(sites), "sites_with_entropy": int(np.isfinite(out).sum()),
		"uniprot_from": int(t.uniprot_pos[sites].min()), "uniprot_to": int(t.uniprot_pos[sites].max()),
		"mean_entropy_nats": float(np.nanmean(out)), f"sites_entropy_lt_{args.natural_k:g}": int((out < args.natural_k).sum())}
	return out, info


def cmd_pfam(args):
	ref, t, tier_summary = build_table(args)
	os.makedirs(args.raw_dir, exist_ok=True)
	os.makedirs(args.out_dir, exist_ok=True)
	rng = np.random.default_rng(0)
	manual = dict(item.split("=", 1) for item in args.sto or [])
	if args.accessions or manual:
		entries = [(acc, "", []) for acc in dict.fromkeys((args.accessions or []) + list(manual))]
	else:
		entries = pfam_entries_of_pten(args.raw_dir)
		print(f"Pfam families InterPro places on human PTEN ({PTEN_ACC}):")
		for acc, name, frags in entries:
			print(f"  {acc}  {name:28s} residues " + ", ".join(f"{a}-{b}" for a, b in frags))
		if not entries:
			raise SystemExit("InterPro returned no Pfam entry for PTEN; give them with --accessions.")
	summary, used = [], []
	for acc, name, frags in entries:
		path = manual.get(acc) or os.path.join(args.raw_dir, f"{acc}_{args.alignment}.sto")
		if not os.path.isfile(path):
			print(f"\ndownloading the {args.alignment} alignment of {acc} ...")
			with open(path, "wb") as f:
				f.write(http_get(f"{INTERPRO}entry/pfam/{acc}/?annotation=alignment:{args.alignment}&download"))
		opener = gzip.open if path.endswith(".gz") else open
		with opener(path, "rt", errors="replace") as f:
			seqs = read_stockholm(f.read())
		if not seqs:
			raise SystemExit(f"{path}: no sequences read; is it a Stockholm alignment?")
		anchor = find_anchor(seqs)
		how = "PTEN is a row of the alignment"
		if anchor is None:
			print(f"  {acc}: human PTEN is not among its {len(seqs)} rows; placing it with hmmalign")
			seqs, anchor = hmmalign_with_pten(acc, seqs, args.raw_dir)
			how = "PTEN placed on the family HMM with hmmalign (PTEN is not in this alignment: read the result with care)"
		H, info = pfam_onto_reference(acc, seqs, anchor, t, args, rng)
		t[f"pfam_{acc}_entropy"] = H
		info["anchor"] = how
		summary.append(info); used.append(acc)
		print(f"\n{acc} {name}: {info['rows_in_alignment']} rows, {info['rows_used']} used, N_eff {info['n_eff']:.0f}, "
			f"median identity to PTEN {info['median_identity_to_PTEN']:.2f}; {info['sites_with_entropy']} reference sites "
			f"(UniProt {info['uniprot_from']}-{info['uniprot_to']}) get an entropy, mean {info['mean_entropy_nats']:.2f} nats. {how}.")
		for label, col in (("PTEN-like family entropy", "nat_family_entropy"), ("phosphatase tier entropy", "nat_pdomain_entropy"),
				("orthologs entropy", "nat_orthologs_entropy"), ("abundance mean", "abundance2021_mean"), ("activity mean", "activity_mean")):
			ok = np.isfinite(H) & t[col].notna().values
			if ok.sum() >= 10:
				print(f"  Spearman with {label:26s} {spearman(H, t[col]):+.2f}  (n={int(ok.sum())})")
		k = t.site[H < args.natural_k].values
		for col in ("K_natural", "K_abundance", "K_activity"):
			inside = t.site[t[col] & np.isfinite(H)].values
			n_cmp = int(np.isfinite(H).sum())
			kk, exp, pp = set_overlap(k, inside, n_cmp)
			print(f"  sites below {args.natural_k} nats: {len(k)}; in {col}: {kk}/{len(inside)}  expected {exp:.1f}  P={pp:.2g}")
		print("  conserved sites: " + " ".join(f"{a}{p_}" for a, p_ in zip(t.ref_aa[H < args.natural_k], t.uniprot_pos[H < args.natural_k])))
	pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "pfam_summary.tsv"), sep="\t", index=False, float_format="%.3f")
	t.to_csv(os.path.join(args.out_dir, "pten_site_table_pfam.csv"), index=False, float_format="%.4f")

	rows = [("nat_family_entropy", "PTEN-like family\n(jackhmmer)", C_NAT), ("nat_pdomain_entropy", "phosphatase tier\n(jackhmmer)", C_NAT)]
	rows += [(f"pfam_{acc}_entropy", f"Pfam {acc}", "k") for acc in used]
	rows += [("abundance2021_mean", "abundance\n(VAMP-seq)", C_ABU), ("activity_mean", "activity\n(Mighell)", C_ACT)]
	fig, axes = plt.subplots(len(rows), 1, figsize=(13, 1.5 * len(rows) + .8), sharex=True)
	for ax, (col, lab, c) in zip(axes, rows):
		shade_domains(ax, t)
		ax.bar(t.site, t[col].fillna(0), width=1., color=c, lw=0)
		ax.set_ylabel(lab + ("\n[nats]" if "entropy" in col else ""), fontsize=8)
	ticks = [p_ for p_ in range(10, 354, 20) if (t.uniprot_pos == p_).any()]
	axes[-1].set_xticks([int(t.site[t.uniprot_pos == p_].iloc[0]) for p_ in ticks], ticks, fontsize=8)
	axes[-1].set_xlim(-1, len(t))
	axes[-1].set_xlabel("UniProt position (shading: phosphatase domain, C2 domain)")
	fig.align_ylabels(axes)
	fig.subplots_adjust(hspace=.12, top=.98, bottom=.07, left=.09, right=.99)
	fig.savefig(os.path.join(args.out_dir, "pten_pfam_profile.png"), dpi=150)
	plt.close(fig)
	print(f"\nOutputs in {args.out_dir}: pten_site_table_pfam.csv, pfam_summary.tsv, pten_pfam_profile.png")


def threshold(text):
	"""A fraction in [0, 1]; 'none', 'off' or 'no' mean no threshold (0)."""
	if str(text).lower() in ("none", "off", "no"):
		return 0.
	v = float(text)
	if v > 1:
		v /= 100.           # 90 is read as 90%
	if not 0. <= v <= 1.:
		raise argparse.ArgumentTypeError(f"{text!r} is not a fraction between 0 and 1")
	return v


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
	common.add_argument("--vertebrate-identity", type=threshold, default=0.9,
		help="identity to PTEN a hit needs for the 'vertebrate' tier (default 0.9). 0 or 'none' removes the threshold")
	common.add_argument("--ortholog-identity", type=threshold, default=0.7,
		help="identity to PTEN a hit needs for the 'orthologs' tier (default 0.7). 0 or 'none' removes the threshold")
	common.add_argument("--min-coverage", type=threshold, default=0.8,
		help="share of the 324 reference columns a hit must cover for the vertebrate, orthologs and family tiers "
			"(default 0.8, which demands both domains). 0 or 'none' keeps every hit")
	common.add_argument("--pdomain-coverage", type=threshold, default=0.7,
		help="share of the phosphatase-domain columns a hit must cover for the 'pdomain' tier (default 0.7)")
	common.add_argument("--ortholog-weight-identity", type=float, default=0.95, help="redundancy threshold inside the ortholog tiers")
	common.add_argument("--natural-k", type=float, default=0.5, help="family entropy [nats] below which a site is in the natural K-like set")
	common.add_argument("--activity-k", type=float, default=0.5,
		help=f"share of scored missense variants below {ACTIVITY_LOF} above which a site is in the activity K-like set")
	p = sub.add_parser("fetch", parents=[common]); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_fetch)
	p = sub.add_parser("build", parents=[common]); p.set_defaults(func=cmd_build)
	p = sub.add_parser("compare", parents=[common])
	p.add_argument("--runs", nargs="+", help="LABEL=RUN_DIR, a run dir holding start.txt and data.dat(.gz)")
	p.add_argument("--runs-dir", help="take every T*/chain* under this directory (e.g. pten1d5r_study/equil): chain0 is "
		"labelled T4, the others T4.c1, ...")
	p.add_argument("--reference-only", action="store_true", help="leave out the chains that did not start from the reference")
	p.add_argument("--burn-in-frac", type=float, default=0.5)
	p.add_argument("--cold", nargs="+", help="labels whose entropy defines the sampled K-sites (default: the coldest)")
	p.add_argument("--show", nargs="+", help="labels to draw (default: all)")
	p.add_argument("--k-site-threshold", type=float, default=0.05)
	p.set_defaults(func=cmd_compare)
	p = sub.add_parser("family", parents=[common])
	p.add_argument("--min-prob", type=float, default=90., help="HHsearch probability a template needs")
	p.add_argument("--bridge", default="2CM2_A", help="PF00102 member whose alignment is carried onto PTEN (PTP1B)")
	p.set_defaults(func=cmd_family)
	p = sub.add_parser("pfam", parents=[common])
	p.add_argument("--accessions", nargs="+", help="Pfam accessions to use (default: the ones InterPro places on human PTEN)")
	p.add_argument("--alignment", default="full", choices=["full", "seed", "uniprot", "rp15", "rp35", "rp55", "rp75"],
		help="which Pfam alignment: seed (curated, tens to hundreds of rows), full (default, every match in Pfam's "
			"reference set), uniprot (every match in UniProt, can be very large), rpNN (representative proteomes)")
	p.add_argument("--sto", nargs="+", metavar="ACC=PATH", help="use a Stockholm file already on disk for this accession "
		"(e.g. saved from the Alignment tab of the entry's InterPro page) instead of downloading")
	p.add_argument("--pfam-coverage", type=threshold, default=0.7, help="share of PTEN's columns a row must cover (default 0.7)")
	p.add_argument("--max-seqs", type=int, default=20000, help="rows above this are subsampled at random (seed 0)")
	p.set_defaults(func=cmd_pfam)
	args = ap.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
