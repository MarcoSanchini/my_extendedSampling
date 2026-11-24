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

  loop     what the references say about the 286-309 loop that the construct
           lacks: tiers of the full-length alignment (PDB 7JVX chain A) and
           the two scans on the loop sites.
  msa      robustness of the natural reference to the sequence database: the
           same tiers from the BFD and MGnify alignments of the 1D5R chain
           (OpenProteinSet), plus any FASTA given with --fasta (e.g. a UniRef100
           download), compared with UniRef90 site by site, on the K-like set and
           on the disease-variant enrichment of pten_variants.py.
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
    python pten_k_sites.py compare --burn-in-frac 0.8 --cold T1 T2 \
        --runs-dir pten1d5r_study/equil_ref pten1d5r_study/equil      # the whole ladder
    python pten_k_sites.py family
    python pten_k_sites.py mutants                             # needs scan_single_mutants.py's CSV (GPU)
    python pten_k_sites.py msa                                 # BFD and MGnify vs UniRef90
    python pten_k_sites.py msa --fasta uniref100=../../databases/<file>.fasta
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
	"7jvx_A_uniref90_hits.a3m": S3 + "7jvx_A/a3m/uniref90_hits.a3m",      # full-length PTEN, for the loop subcommand
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
	# raw/ is gitignored on purpose (the VAMP-seq scores are not ours to
	# redistribute), so it is missing on every fresh clone and `fetch` is a
	# required first step rather than an optional one. Say that instead of
	# letting a bare FileNotFoundError come out of the reader.
	if not os.path.exists(path):
		raise SystemExit(f"{path} is missing.\nThe raw data is gitignored and has to be downloaded on each machine:\n"
			f"    python pten_k_sites.py fetch\n(about 20 MB from raw.githubusercontent.com and the OpenFold S3 bucket)")
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


def natural_tiers(a3m, ref, rmap, args, seqs=None):
	"""Four nested natural alignments of the reference and their per-site entropies.
	`seqs` (query first, one string of len(ref) match columns each) replaces reading `a3m`."""
	L = len(ref)
	if seqs is None:
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


def positional(df, pos_col, score_col, lof_below, min_variants=MIN_VARIANTS):
	"""Missense variants -> per UniProt position: n scored, mean score, fraction below the loss threshold."""
	g = df.dropna(subset=[score_col]).groupby(pos_col)[score_col]
	out = pd.DataFrame({"n": g.size(), "mean": g.mean(), "frac_lof": g.apply(lambda s: (s < lof_below).mean())})
	out.loc[out.n < min_variants, ["mean", "frac_lof"]] = np.nan
	return out


def load_scans(raw_dir, activity_lof=ACTIVITY_LOF, min_variants=MIN_VARIANTS):
	v = pd.read_csv(os.path.join(raw_dir, "PTEN_variant_data.tsv"), sep="\t", low_memory=False)
	v = v[v["class"] == "missense"]
	# authors' classes: "low" = score and upper CI below the 5th percentile of synonymous
	v_low = v.loc[v.abundance_class.notna(), ["position", "abundance_class"]].copy()
	v_low["low"] = (v_low.abundance_class == "low").astype(float)
	# "below" = low or possibly low, i.e. the score is under the 5th percentile of the synonymous variants
	v_low["below"] = v_low.abundance_class.isin(["low", "possibly_low"]).astype(float)
	vamp = positional(v, "position", "score", lof_below=np.inf, min_variants=min_variants).drop(columns="frac_lof")
	vamp["frac_low"] = v_low.groupby("position").low.mean()
	vamp["frac_below"] = v_low.groupby("position").below.mean()
	vamp.loc[vamp.n < min_variants, ["frac_low", "frac_below"]] = np.nan
	p = pd.read_csv(os.path.join(raw_dir, "PTEN_positional_data.tsv"), sep="\t").set_index("position")
	vamp["class_2018"] = p.abundance_class          # the authors' own positional call: intolerant / enhanced
	vamp["rsa"] = p.rsa                             # relative solvent accessibility in 1D5R [%], from the same table
	c = pd.read_csv(os.path.join(raw_dir, "Composite_abundance_data.tsv"), sep="\t", low_memory=False)
	c = c[c["class"] == "missense"]
	comp = positional(c, "position", "score_total", lof_below=np.inf, min_variants=min_variants).drop(columns="frac_lof")
	m = pd.read_csv(os.path.join(raw_dir, "Mighell_PTEN_phosphatase_data.csv"), encoding="utf-8-sig")
	m = m[(m.Type == "missense") & (m.High_conf == True)].copy()      # noqa: E712  (NaN must fail)
	m["position"] = m["Variant (one letter)"].str.extract(r"(\d+)").astype(int)
	act = positional(m, "position", "Cum_score", lof_below=activity_lof, min_variants=min_variants)
	return vamp, comp, act


# ------------------------------------------------------------------ build --- #

def build_table(args):
	ref, t = reference_map()
	tiers, tier_summary = natural_tiers(os.path.join(args.raw_dir, "1d5r_A_uniref90_hits.a3m"), ref, t, args)
	for tier, (H, gap, cons, cons_f) in tiers.items():
		t[f"nat_{tier}_entropy"] = H
		if tier == "family":
			t["nat_family_gap"], t["nat_family_consensus"], t["nat_family_consensus_freq"] = gap, cons, cons_f
	vamp, comp, act = load_scans(args.raw_dir, args.activity_lof, args.min_variants)
	pos = t.uniprot_pos.where(t.native)          # the two non-native residues get no experimental score
	t["abundance_n"] = pos.map(vamp.n)
	t["abundance_mean"] = pos.map(vamp["mean"])
	t["abundance_frac_low"] = pos.map(vamp.frac_low)
	t["abundance_frac_below"] = pos.map(vamp.frac_below)
	t["abundance_class_2018"] = pos.map(vamp.class_2018)
	t["rsa"] = pos.map(vamp.rsa)
	t["abundance2021_n"] = pos.map(comp.n)
	t["abundance2021_mean"] = pos.map(comp["mean"])
	t["activity_n"] = pos.map(act.n)
	t["activity_mean"] = pos.map(act["mean"])
	t["activity_frac_lof"] = pos.map(act.frac_lof)
	# The three K-like sets. Each is the reference's own statement "this site cannot change".
	t["K_natural"] = t[f"nat_{args.natural_tier}_entropy"] < args.natural_k
	if args.abundance_k is None:
		t["K_abundance"] = t.abundance_class_2018 == "intolerant"
	else:
		t["K_abundance"] = t.abundance_frac_below >= args.abundance_k
	t["K_activity"] = t.activity_frac_lof >= args.activity_k
	return ref, t, tier_summary


def shade_domains(ax, t):
	for dom, col in (("phosphatase", "#f3ead9"), ("C2", "#e3ecf5")):
		s = t.site[t.domain == dom]
		ax.axvspan(s.min() - .5, s.max() + .5, color=col, zorder=0, lw=0)


def plot_profile(t, tier_summary, path, sampled=None, extra_ylabel="sampled {label}\nentropy [nats]"):
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
			ax.axhline(ACTIVITY_LOF, color="k", lw=.5, ls=":")     # the default cut; --activity-lof moves the set, not this line
	if sampled is not None:
		for ax, label in zip(axes[len(rows):], sampled.columns):
			shade_domains(ax, t)
			ax.bar(t.site, sampled[label], width=1., color=C_SAMP, lw=0)
			ax.set_ylabel(extra_ylabel.format(label=label), fontsize=8)
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
	for col, rule in (("K_natural", f"{args.natural_tier} entropy < {args.natural_k} nats"),
			("K_abundance", "authors' positional class 'intolerant' (Matreyek 2018)" if args.abundance_k is None else
				f">= {args.abundance_k:.0%} of scored missense variants low or possibly low (below the synonymous 5th percentile)"),
			("K_activity", f">= {args.activity_k:.0%} of scored missense variants below {args.activity_lof}")):
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

def chain_site_stats(run_dir, ref, burn_in_frac, match_accepted=None):
	"""Per-site statistics of one chain, rebuilt from start.txt and the accepted moves of data.dat.

	`match_accepted` overrides `burn_in_frac`: the window start is moved back from the end of
	the chain until the kept part holds about that many ACCEPTED moves. Use it for any
	comparison ACROSS temperatures. A fixed burn-in fraction does not compare like with like,
	because the per-site entropy is estimated from the accepted moves in the window and is
	attenuated when there are few of them, so a chain with a longer window or a higher
	acceptance rate scores higher for that reason alone. Measured on T=8's 300000-move chain,
	family-entropy Spearman with RSA partialled out: 0.311 over a 30000-move window, 0.382
	over 60000, 0.457 over 150000 -- same chain, same equilibrium, three different answers.
	That attenuation is what produced the spurious "maximum at T=6" of 2026-11-19: at a fixed
	0.8 the hot temperatures had 18000 to 36000-move windows while T=2 and T=4 had 130000 and
	80000. At a matched 16000 accepted moves, T=6, T=8 and T=16 are indistinguishable.

	entropy     Shannon entropy [nats] of the residue at the site over the kept moves (every move
	            counts once, accepted or not, as a time average must)
	native_frac share of the kept moves the site spends on the reference residue
	proposals, accepted   over the WHOLE chain
	mean_dU     mean U(proposed) - U(current) of the single-site proposals at the site, whole chain
	proposals_kept, accepted_kept, mean_dU_kept   the same three over the KEPT window only, i.e.
	            behind the same k >= first guard as the entropy, so they can be compared with it.
	            Both groups are returned rather than one replacing the other, because they answer
	            different questions and were once confused for each other: "this site never
	            accepted anything in 300000 moves" is about the landscape, "this site accepted
	            nothing after t_eq" is what the entropy of 0 in the same window means.
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
	if match_accepted is None:
		first = int(burn_in_frac * N)
	else:
		# Walk back from the end until the window holds match_accepted accepted moves. A
		# chain without that many in total keeps everything, and the caller is told so by
		# the returned accepted_kept being short of the target.
		seen_acc, first = 0, 0
		for k in range(N - 1, -1, -1):
			seen_acc += rows[k][3]
			if seen_acc >= match_accepted:
				first = k
				break
	seq = np.frombuffer(start.encode(), dtype=np.uint8).copy()
	ref_b = np.frombuffer(ref.encode(), dtype=np.uint8)
	counts = np.zeros((L, 256))
	since = np.full(L, first)              # move at which each site took its current residue (clipped to burn-in)
	prop, acc, dU = np.zeros(L), np.zeros(L), np.zeros(L)
	prop_k, acc_k, dU_k = np.zeros(L), np.zeros(L), np.zeros(L)
	for k, (sites, aas, du, ok) in enumerate(rows):
		sites = [int(x) for x in sites.split(",")]
		if len(sites) == 1:
			prop[sites[0]] += 1; dU[sites[0]] += du; acc[sites[0]] += ok
			if k >= first:
				prop_k[sites[0]] += 1; dU_k[sites[0]] += du; acc_k[sites[0]] += ok
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
		"mean_dU": np.where(prop > 0, dU / np.maximum(prop, 1), np.nan),
		"proposals_kept": prop_k, "accepted_kept": acc_k,
		"mean_dU_kept": np.where(prop_k > 0, dU_k / np.maximum(prop_k, 1), np.nan)}), {
		"moves": N, "start_is_reference": start == ref, "final_Hd": int((seq != ref_b).sum()),
		"accepted_kept": int(sum(r[3] for r in rows[first:])), "window_start": first}


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
	# Several --runs-dir are allowed because the ladder is split across two of them
	# (equil_ref holds T1..T2, T4, T8, T16; equil holds T3, T5, T6), and taking only one
	# silently drops temperatures -- which is exactly what hid the T=6 maximum until
	# 2026-11-19. setdefault means the FIRST directory listed wins a duplicate label, so
	# list the one with the longer chains first.
	for runs_dir in args.runs_dir or []:
		for tdir in sorted(os.listdir(runs_dir)):
			for cdir in sorted(os.listdir(os.path.join(runs_dir, tdir))) if os.path.isdir(os.path.join(runs_dir, tdir)) else []:
				path = os.path.join(runs_dir, tdir, cdir)
				m = re.fullmatch(r"chain(\d+)", cdir)
				if not m or not os.path.isfile(os.path.join(path, "start.txt")):
					continue
				if not (os.path.isfile(os.path.join(path, "data.dat")) or os.path.isfile(os.path.join(path, "data.dat.gz"))):
					continue
				runs.setdefault(tdir if m.group(1) == "0" else f"{tdir}.c{m.group(1)}", path)
	if not runs:
		raise SystemExit("compare needs --runs LABEL=DIR ... or --runs-dir DIR [DIR ...] (directories of T*/chain*)")
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
	chain_cols = {}                 # collected and concatenated once below: inserting ~100
	                                # columns one at a time fragments the frame and warns
	if args.match_accepted:
		print(f"chains (entropy over the window holding the last {args.match_accepted} ACCEPTED "
			"moves, so the temperatures are comparable; check each window start clears t_eq):")
	else:
		print("chains (entropy over the last "
			f"{1 - args.burn_in_frac:.0%} of the moves):")
	for label, path in runs.items():
		st, info = chain_site_stats(path, ref, args.burn_in_frac, args.match_accepted)
		if args.reference_only and not info["start_is_reference"]:
			print(f"  {label:6s} skipped: not started from the reference")
			continue
		if info["start_is_reference"]:
			ref_started.append(label)
		stats[label] = st
		ent[label] = st.entropy.values
		for c in ("entropy", "native_frac", "proposals", "accepted", "mean_dU",
				"proposals_kept", "accepted_kept", "mean_dU_kept"):
			chain_cols[f"{label}_{c}"] = st[c].values
		print(f"  {label:6s} {info['moves']:7d} moves, start {'reference' if info['start_is_reference'] else 'other'}, "
			f"final Hd {info['final_Hd']:3d}, window from {info['window_start']:6d}, "
			f"{info['accepted_kept']:5d} accepted in the kept part, "
			f"mean entropy {st.entropy.mean():.3f}, sites at entropy 0: {(st.entropy == 0).sum()}/{len(ref)}, "
			f"sites never accepted: {(st.accepted == 0).sum()} anywhere / "
			f"{(st.accepted_kept == 0).sum()} in the kept window")

	if chain_cols:
		t = pd.concat([t, pd.DataFrame(chain_cols, index=t.index)], axis=1)

	refs = {"orthologs entropy": t.nat_orthologs_entropy, "family entropy": t.nat_family_entropy,
		"abundance mean": t.abundance2021_mean, "activity mean": t.activity_mean}
	print("\nSpearman, sampled entropy vs each reference (all are 'higher = more tolerant'; n = sites with data).")
	print("Further lines give the same for -mean_dU of the proposals, the sampler's own mutation cost, over")
	print("the whole chain and over the kept window. The whole-chain one includes the approach to")
	print("equilibrium, exactly the flaw --burn-in-frac exists to remove from the entropy, so compare the two.")
	rows = []
	for label, st in stats.items():
		rows.append({"chain": label, "quantity": "entropy", **{k: spearman(st.entropy, v) for k, v in refs.items()}})
		rows.append({"chain": label, "quantity": "-mean_dU", **{k: spearman(-st.mean_dU, v) for k, v in refs.items()}})
		rows.append({"chain": label, "quantity": "-mean_dU_kept",
			**{k: spearman(-st.mean_dU_kept, v) for k, v in refs.items()}})
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


# ---------------------------------------------------------------- mutants --- #
# The in-silico deep mutational scan: scan_single_mutants.py writes dU for all
# 324 x 19 point mutants of the reference (dU = U(mutant), since U(reference) = 0
# is the global minimum of a squared log-distance), and this reads that CSV.
# It is the only comparison in this script that works VARIANT BY VARIANT rather
# than site by site, because both scans measure single substitutions too.

def load_variants(raw_dir, activity_lof=ACTIVITY_LOF):
	"""One row per measured missense variant: UniProt position, wild-type and mutant residue,
	the 2018 and 2021 abundance scores with the authors' classes, and Mighell's activity score."""
	v = pd.read_csv(os.path.join(raw_dir, "PTEN_variant_data.tsv"), sep="\t", low_memory=False)
	v = v[v["class"] == "missense"][["position", "start", "end", "score", "abundance_class"]].rename(
		columns={"start": "wt_aa", "end": "mut_aa", "score": "abundance_2018", "abundance_class": "class_2018"})
	c = pd.read_csv(os.path.join(raw_dir, "Composite_abundance_data.tsv"), sep="\t", low_memory=False)
	c = c[c["class"] == "missense"][["position", "start", "end", "score_total", "abundance_class"]].rename(
		columns={"start": "wt_aa", "end": "mut_aa", "score_total": "abundance_2021", "abundance_class": "class_2021"})
	m = pd.read_csv(os.path.join(raw_dir, "Mighell_PTEN_phosphatase_data.csv"), encoding="utf-8-sig")
	m = m[(m.Type == "missense") & (m.High_conf == True)].copy()           # noqa: E712  (NaN must fail)
	code = m["Variant (one letter)"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
	m["wt_aa"], m["position"], m["mut_aa"] = code[0], code[1].astype(float), code[2]
	m = m.dropna(subset=["position"])[["position", "wt_aa", "mut_aa", "Cum_score"]].rename(columns={"Cum_score": "activity"})
	m["position"] = m.position.astype(int)
	out = v.merge(c, on=["position", "wt_aa", "mut_aa"], how="outer").merge(m, on=["position", "wt_aa", "mut_aa"], how="outer")
	# Both flags are left NaN where there is no measurement, so that an unmeasured
	# variant is not silently counted as a negative in the AUCs below.
	out["activity_is_lof"] = np.where(out.activity.notna(), out.activity < activity_lof, np.nan)
	# "below the synonymous 5th percentile": the authors' own call, the same one
	# the --abundance-k rule aggregates per site.
	out["abundance_is_low"] = np.where(out.class_2018.notna(),
		out.class_2018.isin(["low", "possibly_low"]), np.nan)
	# The same call on the 2021 composite, so that each experiment's AUC is taken
	# against its OWN class column. The 2021 file writes "unknown" where the 2018
	# one leaves the class empty.
	out["abundance2021_is_low"] = np.where(out.class_2021.notna() & (out.class_2021 != "unknown"),
		out.class_2021.isin(["low", "possibly_low"]), np.nan)
	return out


def auc(score, positive):
	"""Area under the ROC curve of `score` for the binary label `positive`, as the
	Mann-Whitney statistic: the probability that a random positive scores above a
	random negative. 0.5 is chance, so |AUC - 0.5| is the useful quantity."""
	score, positive = np.asarray(score, float), np.asarray(positive, bool)
	ok = np.isfinite(score)
	score, positive = score[ok], positive[ok]
	if positive.sum() < 2 or (~positive).sum() < 2:
		return np.nan
	r = pd.Series(score).rank().values
	n1, n0 = positive.sum(), (~positive).sum()
	return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_site_spearman(df, x, y, by="site"):
	"""Spearman of x and y after each has been ranked WITHIN its site and centred.
	NOTE ON SIGN: pass the same orientation as everywhere else, i.e. -dU and not dU,
	so that a POSITIVE value means agreement. Getting this wrong once made the
	within-site result look like the exact opposite of the pooled one.
	Across all variants, a correlation can come entirely from the sites: a buried
	site is expensive and its variants are all destabilising. This asks the sharper
	question -- at a given site, does the model rank the 19 substitutions the way
	the experiment does -- which no per-site statistic can answer."""
	d = df[[by, x, y]].dropna()
	g = d.groupby(by)
	d = d[g[x].transform("size") >= 5]
	if len(d) < 20:
		return np.nan, 0
	g = d.groupby(by)
	cx = g[x].rank() - g[x].transform(lambda s: s.rank().mean())
	cy = g[y].rank() - g[y].transform(lambda s: s.rank().mean())
	return float(np.corrcoef(cx, cy)[0, 1]), int(d[by].nunique())


def plot_mutants(t, mv, path_profile, path_sites, path_variants):
	"""Three figures: the per-site cost along the sequence next to the two experiments,
	the per-site scatters against every reference, and the variant-level scatters."""
	plot_profile(t, None, path_profile, sampled=t[["dU_log10_mean"]].rename(
		columns={"dU_log10_mean": "mutation cost\nmean log10 dU"}), extra_ylabel="{label}")

	refs = [("PTEN-like family entropy [nats]", t.nat_family_entropy, C_NAT),
		("abundance (VAMP-seq 2021, mean)", t.abundance2021_mean, C_ABU),
		("activity (Mighell, mean)", t.activity_mean, C_ACT),
		("RSA in 1D5R [%]", t.rsa, "#7a5195")]
	fig, axes = plt.subplots(1, len(refs), figsize=(3.1 * len(refs), 3.1))
	for ax, (name, y, col) in zip(axes, refs):
		ax.scatter(t.dU_mean, y, s=7, color=col, alpha=.6, lw=0)
		ax.set_xscale("log")
		ax.set_xlabel("mean dU of the 19 mutants")
		ax.set_ylabel(name, fontsize=8)
		ax.set_title(f"rho(-dU, y) = {spearman(-t.dU_mean, y):+.2f}", fontsize=9)
	fig.tight_layout()
	fig.savefig(path_sites, dpi=130)
	plt.close(fig)

	panels = [("abundance_2021", "abundance (VAMP-seq 2021)", C_ABU, 1.),
		("abundance_2018", "abundance (VAMP-seq 2018)", C_ABU, 1.),
		("activity", "activity (Mighell)", C_ACT, ACTIVITY_LOF)]
	fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.3))
	for ax, (col, name, c, line) in zip(axes, panels):
		d = mv.dropna(subset=["dU", col]).assign(minus_dU=lambda f: -f.dU)
		ax.scatter(d.dU, d[col], s=5, color=c, alpha=.35, lw=0)
		ax.set_xscale("log")
		ax.axhline(line, color="k", lw=.5, ls=":")
		ax.set_xlabel("dU of that exact variant")
		ax.set_ylabel(name, fontsize=8)
		w, ns = within_site_spearman(d, "minus_dU", col)
		ax.set_title(f"n = {len(d)}  rho = {spearman(-d.dU, d[col]):+.2f}\nwithin site {w:+.2f} ({ns} sites)", fontsize=9)
	fig.tight_layout()
	fig.savefig(path_variants, dpi=130)
	plt.close(fig)


def cmd_mutants(args):
	ref, t, tier_summary = build_table(args)
	os.makedirs(args.out_dir, exist_ok=True)
	if not os.path.exists(args.mutants):
		raise SystemExit(f"{args.mutants} does not exist. It is written by scan_single_mutants.py, which needs the GPU:\n"
			f"    nohup python -u scan_single_mutants.py --out {args.mutants} > scan_single_mutants.log 2>&1 &")
	m = pd.read_csv(args.mutants)
	# Any per-mutant cost can go through the same comparison: U_am's dU (the default),
	# or the contact-map E written by scan_mutants_E.py (--energy-column E). Inside
	# this function the chosen column is called dU throughout, and every output file
	# carries --tag so that two energies never overwrite each other.
	need = {"site", "ref_aa", "mut_aa", args.energy_column}
	if not need <= set(m.columns):
		raise SystemExit(f"{args.mutants} must have the columns {sorted(need)}, found {list(m.columns)}")
	if args.energy_column != "dU":
		m = m.drop(columns=[c for c in ("dU",) if c in m.columns]).rename(columns={args.energy_column: "dU"})
		print(f"energy: column {args.energy_column!r} of {args.mutants}, called dU below")
	if args.same_mutants_as:
		other = pd.read_csv(args.same_mutants_as)[["site", "ref_aa", "mut_aa"]]
		n0 = len(m)
		m = m.merge(other, on=["site", "ref_aa", "mut_aa"])
		print(f"restricted to the {len(m)} of {n0} mutants also in {args.same_mutants_as} "
			"(like-for-like comparison of two scans)")
	tag = f"_{args.tag}" if args.tag else ""

	def out(name):
		stem, ext = os.path.splitext(name)
		return os.path.join(args.out_dir, f"{stem}{tag}{ext}")

	per_site = m.groupby("site").dU.size()
	print(f"{len(m)} mutants over {m.site.nunique()}/{len(ref)} sites "
		f"({per_site.min()}-{per_site.max()} per site; the scan writes 19)")
	# U_am has U(reference) = 0 as its global minimum, so dU > 0 always and the mean of
	# log10 is the right per-site location statistic. A cross-entropy (dU_ce) has neither
	# property: its reference value is not 0 and a mutant can score below the reference.
	# So the aggregate falls back to the plain mean whenever anything is non-positive,
	# and says so, rather than clipping negatives into a log.
	positive = bool((m.dU > 0).all())
	print(f"dU: min {m.dU.min():.4g}, median {m.dU.median():.4g}, max {m.dU.max():.4g}. "
		+ (f"U(reference) = 0 is the global minimum, so every dU must be > 0: {(m.dU <= 0).sum()} are not."
			if positive else
			f"{int((m.dU <= 0).sum())} of {len(m)} are <= 0, which is expected for a cross-entropy "
			"difference: the per-site statistic below is the PLAIN mean, not the mean of log10."))
	loc = "mean log10 dU" if positive else "mean dU"
	if m.site.nunique() < len(ref):
		print(f"  (incomplete scan: {len(ref) - m.site.nunique()} sites missing. Everything below uses what is there.)")

	# Per-site summaries. dU spans orders of magnitude, so the mean of log10 dU is
	# the better-behaved location statistic; the plain mean is kept because it is
	# what a Boltzmann weight over the site's mutants would be dominated by.
	m["log10_dU"] = np.log10(m.dU.clip(lower=1e-12)) if positive else m.dU
	g = m.groupby("site")
	agg = pd.DataFrame({"dU_mean": g.dU.mean(), "dU_median": g.dU.median(), "dU_min": g.dU.min(),
		"dU_max": g.dU.max(), "dU_log10_mean": g.log10_dU.mean(), "dU_n": g.dU.size()})
	t = t.join(agg, on="site")
	# The reference residue the scan mutated must be the one in the site table.
	bad = t[(t.dU_n.notna()) & (t.ref_aa != g.ref_aa.first().reindex(t.site).values)]
	if len(bad):
		raise SystemExit(f"{args.mutants} disagrees with the reference at sites {list(bad.site)}: "
			"it was scanned on a different sequence")

	refs = {"orthologs entropy": t.nat_orthologs_entropy, "family entropy": t.nat_family_entropy,
		"abundance mean": t.abundance2021_mean, "activity mean": t.activity_mean}
	print("\n--- per site -------------------------------------------------------------")
	print("Spearman against each reference. Every reference is 'higher = more tolerant' and dU is a COST,")
	print("so the correlation is taken with -dU: a positive number means the sampler's energy agrees.")
	rows = []
	quantities = [("-mean dU", -t.dU_mean), ("-median dU", -t.dU_median), ("-min dU", -t.dU_min)]
	if positive:          # otherwise the location statistic IS the plain mean, already the first row
		quantities.append((f"-{loc}", -t.dU_log10_mean))
	for name, col in quantities:
		rows.append({"quantity": name, **{k: spearman(col, v) for k, v in refs.items()},
			"RSA": spearman(col, t.rsa)})
	sp = pd.DataFrame(rows)
	print(sp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
	n_eff = {k: int((np.isfinite(v) & t.dU_mean.notna()).sum()) for k, v in refs.items()}
	print("  n: " + ", ".join(f"{k} {v}" for k, v in n_eff.items())
		+ "; |rho| above about " + f"{1.96 / np.sqrt(min(n_eff.values()) - 1):.2f} is beyond chance at the 5% level")
	print("\nWith burial regressed out (partial Spearman given RSA), the same question minus the fact that")
	print("buried sites are both expensive to mutate and conserved:")
	pr = pd.DataFrame([{"quantity": f"-{loc}",
		**{f"{k} | RSA": partial_spearman(-t.dU_log10_mean, v, t.rsa) for k, v in refs.items()}}])
	print(pr.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
	sp.to_csv(out("mutants_vs_references_spearman.tsv"), sep="\t", index=False, float_format="%.4f")
	pr.to_csv(out("mutants_vs_references_partial_rsa.tsv"), sep="\t", index=False, float_format="%.4f")

	print(f"\nIs the cost raised inside each K-like set? {loc} inside / outside, with p against")
	print(f"circular shifts of the set along the sequence (smallest possible 1/L = {1 / len(ref):.3f}):")
	rows = []
	for col in ("K_natural", "K_abundance", "K_activity"):
		x = t.dU_log10_mean.values
		inside, outside = x[t[col].values], x[~t[col].values]
		rows.append({"set": col, "n_in": int(t[col].sum()), "mean_in": np.nanmean(inside), "mean_out": np.nanmean(outside),
			"ratio_of_dU": 10**(np.nanmean(inside) - np.nanmean(outside)) if positive else np.nan,
			"shift_p": shift_p(x, t[col].values)})
	ks = pd.DataFrame(rows)
	print(ks.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
	ks.to_csv(out("mutants_in_K_sets.tsv"), sep="\t", index=False, float_format="%.4f")

	# A K-like set of our own, on the same footing as the other three: the sites
	# the energy says cannot change. --mutant-k sets where the cut goes.
	cut = t.dU_log10_mean.quantile(args.mutant_k)
	t["K_mutant"] = t.dU_log10_mean >= cut
	print(f"\nmutational K-sites ({loc} in the top {1 - args.mutant_k:.0%}, i.e. >= {cut:.3f}): "
		f"{int(t.K_mutant.sum())}/{int(t.dU_log10_mean.notna().sum())} scanned sites")
	for col in ("K_natural", "K_abundance", "K_activity"):
		k, exp, p = set_overlap(t.site[t.K_mutant], t.site[t[col]], len(ref))
		print(f"  overlap with {col}: {k}/{int(t[col].sum())}  expected {exp:.1f}  P={p:.2g}")
	print("  (size-matched: the same comparison with the mutational set cut to each other set's size,")
	print("   which removes the dependence of the overlap on --mutant-k)")
	order = t.dropna(subset=["dU_log10_mean"]).sort_values("dU_log10_mean", ascending=False)
	for col in ("K_natural", "K_abundance", "K_activity"):
		n = int(t[col].sum())
		k, exp, p = set_overlap(order.site.head(n), t.site[t[col]], len(ref))
		print(f"  top {n:3d} vs {col}: {k}/{n}  expected {exp:.1f}  P={p:.2g}")

	if args.versus:
		# How much do two energies for the same mutants agree with EACH OTHER? Without this,
		# a weak agreement with the experiments cannot be told apart from the two energies
		# measuring different things: U_am (attention) and E (contact map) are both read off
		# ESM3, so this is the ceiling on how similarly they can behave downstream.
		o = pd.read_csv(args.versus)
		if args.versus_column not in o.columns:
			raise SystemExit(f"{args.versus} has no column {args.versus_column!r}, found {list(o.columns)}")
		o = o[["site", "ref_aa", "mut_aa", args.versus_column]].rename(columns={args.versus_column: "other"})
		j = m.merge(o, on=["site", "ref_aa", "mut_aa"])
		print(f"\n--- against the other energy ---------------------------------------------")
		print(f"{args.energy_column} of {os.path.basename(args.mutants)} vs {args.versus_column} of "
			f"{os.path.basename(args.versus)}, on the {len(j)} mutants both scans contain.")
		print("Both are costs, so a POSITIVE correlation means they rank mutants the same way.")
		j["minus_dU"], j["minus_other"] = -j.dU, -j.other
		w, ns = within_site_spearman(j, "minus_dU", "minus_other")
		per = j.groupby("site").agg(a=("dU", lambda x: np.log10(x.clip(lower=1e-12)).mean()),
			b=("other", lambda x: np.log10(x.clip(lower=1e-12)).mean()))
		print(f"  pooled over all mutants: {spearman(j.dU, j.other):+.3f}")
		print(f"  within site (centred ranks, {ns} sites): {w:+.3f}")
		print(f"  per site, on the mean of log10: {spearman(per.a, per.b):+.3f}")
		pd.DataFrame([{"a": f"{args.energy_column}:{os.path.basename(args.mutants)}",
			"b": f"{args.versus_column}:{os.path.basename(args.versus)}", "mutants": len(j),
			"pooled": spearman(j.dU, j.other), "within_site": w, "sites": ns,
			"per_site": spearman(per.a, per.b)}]).to_csv(out("energy_vs_energy.tsv"), sep="\t",
			index=False, float_format="%.4f")

	print("\n--- per variant ----------------------------------------------------------")
	print("Each in-silico mutant matched to the same substitution in the two experiments. This is the")
	print("comparison the papers themselves make, and the one a per-site profile cannot: it asks whether")
	print("the energy ranks the 19 substitutions at a site, not just which sites are expensive.")
	var = load_variants(args.raw_dir, args.activity_lof)
	mv = m.merge(t[["site", "uniprot_pos", "native"]], on="site")
	mv = mv[mv.native].rename(columns={"uniprot_pos": "position", "ref_aa": "wt_aa"})
	mv.position = mv.position.astype(int)
	mv = mv.merge(var, on=["position", "wt_aa", "mut_aa"], how="left")
	rows = []
	mv["minus_dU"] = -mv.dU               # one orientation for every number below: + is agreement
	# Each experiment carries its own damaging call, so the AUC of each row is taken
	# against that experiment's own label and lands in the same table as its rho.
	for col, name, flag, call in (
			("abundance_2021", "abundance 2021 (VAMP-seq composite)", "abundance2021_is_low", "low or possibly low (2021 classes)"),
			("abundance_2018", "abundance 2018 (VAMP-seq)", "abundance_is_low", "low or possibly low (2018 classes)"),
			("activity", "activity (Mighell high-confidence)", "activity_is_lof", f"score < {args.activity_lof}")):
		d = mv.dropna(subset=["dU", col])
		w, ns = within_site_spearman(d, "minus_dU", col)
		# The label is the DAMAGING outcome and the score is dU itself, so above 0.5 means
		# the energy calls the damaging variants the expensive ones: the agreement direction.
		lab = mv.dropna(subset=["dU"])
		lab = lab[lab[flag].notna()]
		rows.append({"experiment": name, "variants matched": len(d), "rho(-dU, score)": spearman(-d.dU, d[col]),
			"rho within site": w, "sites": ns, "damaging call": call,
			"AUC(dU, damaging)": auc(lab.dU, lab[flag]), "positives": int(lab[flag].sum()), "labelled": len(lab)})
	vs = pd.DataFrame(rows)
	print(vs.drop(columns=["damaging call"]).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
	print("\nThe last three columns are the energy used as a CLASSIFIER of the authors' own variant calls,")
	print("one call per experiment. AUC is the area under the ROC curve, equivalently the probability that")
	print("a randomly chosen damaging variant has a higher dU than a randomly chosen tolerated one. The label")
	print("is the DAMAGING outcome and the score is dU itself, so above 0.5 is the agreement direction and")
	print("0.5 is chance. Variants with no measurement are left out rather than counted as tolerated:")
	for r in rows:
		print(f"  {r['experiment']}: damaging = {r['damaging call']}, "
			f"{r['positives']} of {r['labelled']} labelled variants")

	mv.to_csv(out("pten_mutants_variant_level.csv"), index=False, float_format="%.5f")
	vs.to_csv(out("mutants_vs_experiments_variant_level.tsv"), sep="\t", index=False, float_format="%.4f")
	t.to_csv(out("pten_site_table_mutants.csv"), index=False, float_format="%.4f")
	plot_mutants(t, mv, out("pten_mutant_profile.png"),
		out("mutants_vs_references.png"),
		out("mutants_vs_experiments.png"))
	print(f"\nOutputs in {args.out_dir}, each name ending in {tag or '(no tag)'}: pten_site_table_mutants, "
		"pten_mutants_variant_level, mutants_vs_references_spearman, mutants_vs_references_partial_rsa, "
		"mutants_in_K_sets, mutants_vs_experiments_variant_level, pten_mutant_profile.png, "
		"mutants_vs_references.png, mutants_vs_experiments.png")


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


# ------------------------------------------------------------------- loop --- #

LOOP = (286, 309)       # UniProt numbering: the C2-domain loop deleted in the 1D5R construct


def cmd_loop(args):
	"""What the references say about the 286-309 loop, which the sampled construct does not have.

	Natural side: the UniRef90 alignment of a full-length PTEN chain (PDB 7JVX chain A, same
	pipeline as the construct's), cut into the same tiers over UniProt 7-353. Scans: both were
	done on the full-length protein, so the loop sites have scores."""
	ref, t, tier_summary = build_table(args)
	path = os.path.join(args.raw_dir, "7jvx_A_uniref90_hits.a3m")
	if not os.path.isfile(path):
		raise SystemExit(f"{path} is missing: run `python pten_k_sites.py fetch` first")
	with open(path) as f:
		f.readline(); q = f.readline().strip()
	names, seqs = read_a3m(path, len(q))
	up = np.full(len(q), -1)            # UniProt position of each query column (the chain carries a linker in the tail)
	for b in difflib.SequenceMatcher(None, q, PTEN_UNIPROT, autojunk=False).get_matching_blocks():
		if b.size >= 8:
			up[b.a:b.a + b.size] = np.arange(b.b, b.b + b.size) + 1
	X = encode(seqs, len(q))
	al = X != GAP
	first = int(t.uniprot_pos[t.native].min()) - 1
	core = (up >= first) & (up <= t.uniprot_pos.max())
	upc = up[core]
	in_loop, after = (upc >= LOOP[0]) & (upc <= LOOP[1]), upc > LOOP[1]
	ident = ((X == X[0]) & al)[:, core].sum(1) / np.maximum(al[:, core].sum(1), 1)
	cov = al[:, core].mean(1)
	hits_c = int(tier_summary.sequences.max())
	print(f"full-length query (7JVX chain A): {len(seqs) - 1} hits; tiers over UniProt {first}-{int(t.uniprot_pos.max())}, loop included")
	print(f"  hits covering at least half of the loop {LOOP[0]}-{LOOP[1]}: {int((al[:, core][:, in_loop].mean(1) >= .5).sum())}")
	print(f"  hits covering at least half of {LOOP[1] + 1}-{int(t.uniprot_pos.max())}: {int((al[:, core][:, after].mean(1) >= .5).sum())}")
	rows = []
	H_family = None
	for tier, keep, theta in (("vertebrate", (cov >= args.min_coverage) & (ident >= args.vertebrate_identity), args.ortholog_weight_identity),
			("orthologs", (cov >= args.min_coverage) & (ident >= args.ortholog_identity), args.ortholog_weight_identity),
			("family", cov >= args.min_coverage, args.weight_identity)):
		Xt = X[keep][:, core]
		w = 1. / neighbour_counts(Xt, np.ones(len(Xt)), theta)
		H, gap, _, _ = weighted_entropy(Xt, w, args.max_gap)
		ts = tier_summary.set_index("tier")
		rows.append({"tier": tier, "sequences": int(keep.sum()), "n_eff": w.sum(),
			"construct_sequences": int(ts.sequences[tier]), "construct_n_eff": ts.n_eff[tier],
			"loop_gap": gap[in_loop].mean(), "loop_entropy": np.nanmean(H[in_loop]),
			"rest_entropy": np.nanmean(H[~in_loop]), "after_loop_gap": gap[after].mean()})
		if tier == "family":
			H_family = pd.Series(H, index=upc)
	out = pd.DataFrame(rows)
	print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
	m = t[t.native & t.nat_family_entropy.notna()]
	print(f"  family entropy, full-length query vs construct query: Spearman "
		f"{spearman(H_family.reindex(m.uniprot_pos.values).values, m.nat_family_entropy.values):+.3f} over {len(m)} shared sites")
	vamp, comp, act = load_scans(args.raw_dir, args.activity_lof, args.min_variants)
	print(f"\nscans on the loop sites {LOOP[0]}-{LOOP[1]} against the construct's sites:")
	on = t.uniprot_pos[t.native].values
	for name, d in (("abundance 2018", vamp), ("abundance 2018+2021", comp), ("activity", act)):
		lp = d[(d.index >= LOOP[0]) & (d.index <= LOOP[1])]
		print(f"  {name:20s} loop: {int(lp['mean'].notna().sum())}/24 sites with a mean, {int(lp.n.sum())} variants, mean {lp['mean'].mean():+.2f}"
			f"   construct: mean {d['mean'].reindex(on).mean():+.2f}")
	lp = (vamp.index >= LOOP[0]) & (vamp.index <= LOOP[1])
	print(f"  loop sites called 'intolerant' by the authors: {int((vamp.class_2018[lp] == 'intolerant').sum())}; "
		f"loop sites meeting the activity rule: {int((act.frac_lof[(act.index >= LOOP[0]) & (act.index <= LOOP[1])] >= args.activity_k).sum())}")
	os.makedirs(args.out_dir, exist_ok=True)
	out.to_csv(os.path.join(args.out_dir, "loop_tiers.tsv"), sep="\t", index=False, float_format="%.3f")
	print(f"\nOutput in {args.out_dir}: loop_tiers.tsv")


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


# -------------------------------------------------------------------- msa --- #
# Does the natural reference depend on which sequence database the homologs come from?
# OpenProteinSet ships three alignments of the same 1D5R chain, searched the AlphaFold 2 way:
#   uniref90   jackhmmer on UniRef90 (UniProt clustered at 90% identity) -- the one used everywhere above
#   bfd        HHblits on BFD + Uniclust30 (UniProt plus metagenome/metatranscriptome assemblies, clustered)
#   mgnify     jackhmmer on MGnify (proteins predicted from environmental metagenomes, EBI)
# and --fasta adds any unaligned set of sequences (e.g. a UniRef100 download), aligned here onto the
# 324 reference columns with a profile HMM built from the UniRef90 alignment (pyhmmer).

MSA_SOURCES = {"uniref90": "1d5r_A_uniref90_hits.a3m", "bfd": "1d5r_A_bfd_uniclust_hits.a3m",
	"mgnify": "1d5r_A_mgnify_hits.a3m"}
MSA_URLS = {f: S3 + "1d5r_A/a3m/" + f[len("1d5r_A_"):] for f in MSA_SOURCES.values()}


def read_fasta(path):
	names, seqs = [], []
	opener = gzip.open if path.endswith(".gz") else open
	with opener(path, "rt") as f:
		for line in f:
			line = line.strip()
			if line.startswith(">"):
				names.append(line[1:]); seqs.append([])
			elif names and line:
				seqs[-1].append(line)
	return names, ["".join(s).upper() for s in seqs]


def align_onto_reference(names, raw, ref, seed_seqs, max_evalue):
	"""Unaligned sequences -> rows on the reference's match columns, through a profile HMM of `seed_seqs`
	(the reference's own alignment, every column a match state). Every domain hit of a sequence fills its
	columns (PTEN's phosphatase and C2 often come out as two domains); a column two domains both cover keeps
	the better-scoring one. Returns the kept names, their rows, and a per-sequence log table."""
	import pyhmmer
	alpha = pyhmmer.easel.Alphabet.amino()
	L = len(ref)
	seed = [s.replace(".", "-") for s in seed_seqs]
	msa = pyhmmer.easel.TextMSA(name="ref", sequences=[pyhmmer.easel.TextSequence(name=f"s{i}", sequence=s)
		for i, s in enumerate(seed)])
	msa.reference = "x" * L                     # hand architecture: exactly the L reference columns are match states
	builder = pyhmmer.plan7.Builder(alpha, architecture="hand")
	hmm, _, _ = builder.build_msa(msa.digitize(alpha), pyhmmer.plan7.Background(alpha))
	assert hmm.M == L, (hmm.M, L)
	targets = [pyhmmer.easel.TextSequence(name=f"t{i}", sequence=re.sub("[^A-Z]", "X", s)).digitize(alpha)
		for i, s in enumerate(raw)]
	rows = {}
	logrows = []
	for hits in pyhmmer.hmmsearch([hmm], targets, E=max_evalue, domE=max_evalue, Z=len(targets), domZ=len(targets)):
		for hit in hits:
			i = int(hit.name.decode()[1:]) if isinstance(hit.name, bytes) else int(hit.name[1:])
			if hit.evalue > max_evalue:
				continue
			row = ["-"] * L
			score = [-np.inf] * L
			for dom in hit.domains:
				if dom.i_evalue > max_evalue:
					continue
				ali = dom.alignment
				pos = ali.hmm_from - 1
				for hc, tc in zip(ali.hmm_sequence, ali.target_sequence):
					if hc == ".":               # insertion in the target, no reference column
						continue
					if pos < L and dom.score > score[pos]:
						row[pos] = tc.upper() if tc.isalpha() else "-"
						score[pos] = dom.score
					pos += 1
			rows[i] = "".join(row)
			logrows.append({"name": names[i], "length": len(raw[i]), "evalue": hit.evalue, "score": hit.score,
				"domains": sum(d.i_evalue <= max_evalue for d in hit.domains),
				"coverage": sum(c != "-" for c in rows[i]) / L})
	keep = sorted(rows)
	return [names[i] for i in keep], [rows[i] for i in keep], pd.DataFrame(logrows)


def enrichment_in(t, kmask, m, snv):
	"""Observed / expected SNV-reachable distinct variants of each category inside a site set
	(the pten_variants.py `profiles` statistic, for the categories that carry its results)."""
	nat = t[t.native]
	s = snv.merge(nat[["uniprot_pos", "site"]], left_on="position", right_on="uniprot_pos")
	reach = s[s.snv == True].groupby("site").size().reindex(t.site, fill_value=0).values      # noqa: E712
	frac = reach[kmask].sum() / reach.sum()
	cls = m.clinvar_class
	tolerated = m.in_gnomad & (m.gnomad_ac >= 2) & (cls != "P/LP")
	damaging = (cls == "P/LP") | (m.cbio_patients >= 2)
	cats = {"cbio_recurrent": m.cbio_patients >= 2, "clinvar_PLP": cls == "P/LP", "gnomad_all": m.in_gnomad,
		"consensus_damaging": damaging & ~tolerated}
	out = {}
	for name, mask in cats.items():
		d = m[mask.values & (m.snv == True).values]                                         # noqa: E712
		inside = d.site.astype(int).isin(t.site[kmask]).sum()
		out[name] = inside / (len(d) * frac) if len(d) else np.nan
	return out


def cmd_msa(args):
	from scipy.stats import spearmanr
	ref, t, _ = build_table(args)
	out = os.path.join(args.out_dir, "msa_sources")
	os.makedirs(out, exist_ok=True)
	sources = {}
	for name, fname in MSA_SOURCES.items():
		path = os.path.join(args.raw_dir, fname)
		if not os.path.exists(path):
			print(f"downloading {fname}")
			urllib.request.urlretrieve(MSA_URLS[fname], path)
		sources[name] = read_a3m(path, len(ref))[1]
	for spec in args.fasta or []:
		label, path = spec.split("=", 1)
		names, raw = read_fasta(path)
		kept, rows, lg = align_onto_reference(names, raw, ref, sources["uniref90"], args.max_evalue)
		lg.to_csv(os.path.join(out, f"{label}_hits.tsv"), sep="\t", index=False, float_format="%.4g")
		# identical sequences would only inflate the count (the weighting removes them anyway): report them
		n_unique = len(set(raw))
		print(f"{label}: {len(raw)} sequences in {os.path.basename(path)} ({n_unique} distinct), {len(kept)} hit the "
			f"PTEN profile at E <= {args.max_evalue:g}; {int((lg.coverage >= args.min_coverage).sum())} cover >= "
			f"{args.min_coverage:.0%} of the reference")
		with open(os.path.join(out, f"{label}_aligned.a3m"), "w") as f:
			f.write(f">query\n{ref}\n" + "".join(f">{n}\n{s}\n" for n, s in zip(kept, rows)))
		sources[label] = [ref] + rows

	summ, sites = [], {"site": t.site, "uniprot_pos": t.uniprot_pos, "ref_aa": t.ref_aa}
	tiers_by = {}
	for name, seqs in sources.items():
		tiers, ts = natural_tiers(name, ref, t, args, seqs=seqs)
		tiers_by[name] = tiers
		ts.insert(0, "source", name)
		summ.append(ts)
		for tier, (H, *_r) in tiers.items():
			sites[f"{name}_{tier}"] = H
	summ = pd.concat(summ)
	sites = pd.DataFrame(sites)
	m = snv = None
	vdir = os.path.join(MONTE_CARLO_DIR, "pten_variants")
	if os.path.exists(os.path.join(vdir, "pten_variants_merged.tsv")):
		m = pd.read_csv(os.path.join(vdir, "pten_variants_merged.tsv"), sep="\t", low_memory=False)
		m = m[m.in_construct].copy()
		snv = pd.read_csv(os.path.join(vdir, "snv_reachability.tsv"), sep="\t")

	cmp = []
	base = "uniref90"
	k_base = {tier: (sites[f"{base}_{tier}"] < args.natural_k).values for tier in tiers_by[base]}
	for name in sources:
		for tier in tiers_by[name]:
			H = sites[f"{name}_{tier}"]
			row = {"source": name, "tier": tier,
				"sequences": int(summ[(summ.source == name) & (summ.tier == tier)].sequences.iloc[0]),
				"n_eff": float(summ[(summ.source == name) & (summ.tier == tier)].n_eff.iloc[0]),
				"sites_with_entropy": int(np.isfinite(H).sum())}
			if row["n_eff"] < args.min_neff:
				cmp.append(row)          # too few sequences for an entropy: counts only
				continue
			ok = np.isfinite(H) & np.isfinite(sites[f"{base}_{tier}"])
			row["rho_vs_uniref90"] = spearmanr(H[ok], sites[f"{base}_{tier}"][ok])[0] if ok.sum() > 10 else np.nan
			K = (H < args.natural_k).values
			row["K_sites"] = int(K.sum())
			row["K_shared_with_uniref90"] = int((K & k_base[tier]).sum())
			row["K_jaccard"] = (K & k_base[tier]).sum() / max((K | k_base[tier]).sum(), 1)
			row["rho_vs_abundance"] = spearman(H, t.abundance2021_mean)
			row["rho_vs_activity"] = spearman(H, t.activity_mean)
			row["rho_vs_activity_given_RSA"] = partial_spearman(H, t.activity_mean, t.rsa)
			# The absolute cut is not portable between databases (a metagenomic alignment is more diverse at
			# every site), so also take the same NUMBER of sites as UniRef90's set, lowest entropy first.
			n_k = int(k_base[tier].sum())
			Hr = np.where(np.isfinite(H), H, np.inf)
			Kr = np.zeros(len(H), bool)
			Kr[np.argsort(Hr, kind="stable")[:n_k]] = True
			row["Krank_shared_with_uniref90"] = int((Kr & k_base[tier]).sum())
			if m is not None:
				for cat, r in enrichment_in(t, K, m, snv).items():
					row[f"enrich_{cat}"] = r if K.sum() else np.nan
				for cat, r in enrichment_in(t, Kr, m, snv).items():
					row[f"enrich_rank_{cat}"] = r
			cmp.append(row)
	cmp = pd.DataFrame(cmp)
	pd.set_option("display.width", 250)
	print("\nalignment tiers per source (weights 1/neighbours, entropy NaN above "
		f"{args.max_gap:.0%} gaps):")
	print(summ[["source", "tier", "sequences", "n_eff", "median_identity_to_PTEN", "sites_with_entropy",
		"mean_entropy_nats", "sites_entropy_lt_0.05"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
	print(f"\neach source against UniRef90, per tier (K = entropy < {args.natural_k} nats; enrich = observed/expected "
		"SNV-reachable variants of the category inside K):")
	print(cmp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
	summ.to_csv(os.path.join(out, "msa_tiers.tsv"), sep="\t", index=False, float_format="%.4f")
	cmp.to_csv(os.path.join(out, "msa_comparison.tsv"), sep="\t", index=False, float_format="%.4f")
	sites.to_csv(os.path.join(out, "msa_site_entropy.tsv"), sep="\t", index=False, float_format="%.4f")

	fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
	colors = dict(zip(sources, ["#2a7f62", "#3b6fb6", "#c2571a", "#7a3fa0", "#444444"]))
	for ax, tier in zip(axes, ("orthologs", "family")):
		for name in sources:
			if summ[(summ.source == name) & (summ.tier == tier)].n_eff.iloc[0] < args.min_neff:
				continue                 # e.g. BFD has no orthologs: a flat zero line would read as "conserved"
			ax.plot(t.site, sites[f"{name}_{tier}"], lw=1.1, color=colors[name], label=name, alpha=0.85)
		ax.axhline(args.natural_k, color="k", lw=0.6, ls=":")
		ax.set_ylabel(f"{tier} entropy [nats]")
		ax.legend(ncol=len(sources), fontsize=8, loc="upper right", frameon=False)
	axes[-1].set_xlabel("reference site (0-based)")
	fig.suptitle("PTEN natural entropy from different sequence databases (same 1D5R chain, same tier rules)", fontsize=10)
	fig.tight_layout()
	fig.savefig(os.path.join(out, "msa_profiles.png"), dpi=150)
	print(f"\nOutputs in {out}: msa_tiers.tsv, msa_comparison.tsv, msa_site_entropy.tsv, msa_profiles.png"
		+ ("" if not args.fasta else ", <label>_hits.tsv, <label>_aligned.a3m"))


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
	common.add_argument("--natural-k", type=float, default=0.5,
		help="entropy [nats] below which a site is in the natural K-like set (default 0.5)")
	common.add_argument("--natural-tier", default="family", choices=["vertebrate", "orthologs", "family", "pdomain"],
		help="the natural tier whose entropy defines the natural K-like set (default family)")
	common.add_argument("--abundance-k", type=threshold, default=None,
		help="abundance K-like set by a rule of our own: the share of a site's scored missense variants that are low or "
			"possibly low (score under the 5th percentile of the synonymous variants) must reach this value. "
			"Default: not set, and the set is the authors' own list of 'intolerant' positions")
	common.add_argument("--activity-k", type=threshold, default=0.5,
		help="share of a site's scored missense variants below --activity-lof needed for the activity K-like set (default 0.5)")
	common.add_argument("--activity-lof", type=float, default=ACTIVITY_LOF,
		help=f"activity score below which a variant counts as loss of function (default {ACTIVITY_LOF}, Matreyek 2021; "
			"Mighell 2018 uses about -1 for 'not wild-type-like' and about -2.15 for 'truncation-like')")
	common.add_argument("--min-variants", type=int, default=MIN_VARIANTS,
		help=f"scored missense variants a site needs to get a mean or a fraction (default {MIN_VARIANTS})")
	p = sub.add_parser("fetch", parents=[common]); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_fetch)
	p = sub.add_parser("build", parents=[common]); p.set_defaults(func=cmd_build)
	p = sub.add_parser("compare", parents=[common])
	p.add_argument("--runs", nargs="+", help="LABEL=RUN_DIR, a run dir holding start.txt and data.dat(.gz)")
	p.add_argument("--runs-dir", nargs="+", help="take every T*/chain* under these directories (e.g. pten1d5r_study/equil_ref pten1d5r_study/equil): chain0 is "
		"labelled T4, the others T4.c1, ...")
	p.add_argument("--reference-only", action="store_true", help="leave out the chains that did not start from the reference")
	p.add_argument("--burn-in-frac", type=float, default=0.5)
	p.add_argument("--match-accepted", type=int, default=None,
		help="instead of a fixed burn-in fraction, keep the window holding the last N ACCEPTED moves of "
			"each chain. Required for comparing temperatures: see chain_site_stats' docstring for why a "
			"fixed fraction does not compare like with like")
	p.add_argument("--cold", nargs="+", help="labels whose entropy defines the sampled K-sites (default: the coldest)")
	p.add_argument("--show", nargs="+", help="labels to draw (default: all)")
	p.add_argument("--k-site-threshold", type=float, default=0.05)
	p.set_defaults(func=cmd_compare)
	p = sub.add_parser("family", parents=[common])
	p.add_argument("--min-prob", type=float, default=90., help="HHsearch probability a template needs")
	p.add_argument("--bridge", default="2CM2_A", help="PF00102 member whose alignment is carried onto PTEN (PTP1B)")
	p.set_defaults(func=cmd_family)
	p = sub.add_parser("mutants", parents=[common])
	p.add_argument("--mutants", default=os.path.join(MONTE_CARLO_DIR, "pten_natural", "pten_single_mutants.csv"),
		help="the CSV written by scan_single_mutants.py (site, ref_aa, mut_aa, dU), which needs the GPU")
	p.add_argument("--mutant-k", type=threshold, default=0.75,
		help="quantile of the per-site mean log10 dU above which a site joins the mutational K-like set "
			"(default 0.75, the most expensive quarter). 'none' puts every site in it")
	p.add_argument("--energy-column", default="dU",
		help="which column of --mutants is the cost: dU for scan_single_mutants.py (U_am, the default), "
			"E for scan_mutants_E.py (the contact-map energy of Zambon et al)")
	p.add_argument("--tag", default="",
		help="appended to every output file name, e.g. E2, so that two energies do not overwrite each other")
	p.add_argument("--versus", default=None,
		help="another scan's CSV: also report how much the two energies agree with each other, "
			"pooled, within site and per site (e.g. the contact-map E against U_am)")
	p.add_argument("--versus-column", default="dU",
		help="which column of --versus is its cost (default dU; use E for a scan_mutants_E.py file)")
	p.add_argument("--same-mutants-as", default=None,
		help="another scan's CSV: keep only the mutants it also contains, for a like-for-like comparison "
			"(e.g. U_am or E2 on the 380 mutants of the 32-step E scan)")
	p.set_defaults(func=cmd_mutants)
	p = sub.add_parser("loop", parents=[common]); p.set_defaults(func=cmd_loop)
	p = sub.add_parser("msa", parents=[common])
	p.add_argument("--fasta", nargs="+", help="LABEL=PATH: unaligned sequences (FASTA, .gz ok) to align onto the reference "
		"with a profile HMM of the UniRef90 alignment and compare like the three OpenProteinSet sources "
		"(e.g. uniref100=../../databases/uniref100_pten.fasta). Needs pyhmmer (pip install pyhmmer)")
	p.add_argument("--min-neff", type=float, default=10., help="effective sequences a tier needs for its entropy to be "
		"compared (default 10; the BFD alignment has no orthologs at all)")
	p.add_argument("--max-evalue", type=float, default=1e-5, help="E-value a FASTA sequence and each of its domains need "
		"against the PTEN profile to be kept (default 1e-5)")
	p.set_defaults(func=cmd_msa)
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
