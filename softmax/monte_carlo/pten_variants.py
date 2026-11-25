"""
PTEN missense variants from three databases -- cBioPortal (tumours), ClinVar
(germline submissions) and gnomAD (population) -- cleaned, deduplicated,
intersected, profiled per K-set and ranked by the sampler's energies.

Input: the three files Marco pushed to databases/ on 2026-11-18. Nothing is
downloaded: this script never contacts cBioPortal, ClinVar, gnomAD or any other
host. Everything else it reads is already in the repo (pten_natural/*.csv and
the chains under pten1d5r_study/), plus, optionally, pten_natural/raw/
PTEN_variant_data.tsv on astaroth for its `snv` column and 15 predictor calls.

Subcommands (all CPU except `score`):

  clean     parse and filter each file to canonical PTEN missense variants,
            deduplicate, write one clean table per database.
  merge     one row per protein change across the three, the overlaps at the
            protein and the DNA level, the databases' own cross-references
            checked against each other, and the compound tumour sequences
            that `score` needs.
  profiles  per category: per-site variant density and substitution entropy
            along the construct, against the sampled entropy (rebuilt from the
            chains at --burn-in-frac 0.8) and the K-sets.
  rank      every variant looked up in the existing 324 x 19 scans (U_am, the
            2-step contact energy E2, U_structure_ce), its percentile, and the
            AUC of each energy for each category contrast with bootstrap CIs.
            Reads compound_scores.csv too when `score` has run.
  score     GPU, ESM3: U_am, U_ce and E2 of the compound tumour sequences, and
            a re-decode of single mutants to check E2 against the 2-step scan.
  all       clean, merge, profiles, rank in order.

THE THREE TRAPS in the raw files, each handled in `clean` and each reported in
its log:
  1. ClinVar's header has 24 fields and every data row 25, so a plain
     read_csv takes the first column as the index and shifts every label by
     one ("Molecular consequence" then holds the classification). Read with
     index_col=False; the script checks the field counts first.
  2. PTEN-L. The upstream-start isoform adds 173 residues, so ClinVar's
     `Protein change` lists both numberings ("M174L, M1L") and half of
     gnomAD's missense rows sit on ENST00000693560, i.e. in the 5' UTR of
     canonical PTEN. gnomAD's "C65S, AF 0.994" is one of those. Only the
     NM_000314(PTEN) HGVS in ClinVar's Name and only ENST00000371953 rows in
     gnomAD are used.
  3. KLLN overlaps the PTEN promoter; its rows go with the same filter.

Numbering: canonical PTEN (UniProt P60484, NM_000314, ENST00000371953, 403 aa).
`site` is the 0-based construct position of pten_1d5r (UniProt 7-285 and
310-353), empty for the 176 or so variants outside it.

Usage (from softmax/monte_carlo):
    python pten_variants.py all
    nohup python -u pten_variants.py score > pten_variants_score.log 2>&1 &     # GPU
    python pten_variants.py rank                                               # again, with the compounds
"""
import argparse
import os
import re
import sys
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../.."))
sys.path.insert(0, MONTE_CARLO_DIR)
from pten_k_sites import (PTEN_UNIPROT, chain_site_stats, partial_spearman, shift_p, auc,
	within_site_spearman)
from structure_zero_polymer import spearman

NAT_DIR = os.path.join(MONTE_CARLO_DIR, "pten_natural")
DB_DIR = os.path.join(REPO_DIR, "databases")
FILES = {"cbioportal": "cbioportal_pten.tsv", "clinvar": "clinvar_download_20261118_064848.tsv",
	"gnomad": "gnomAD_download_pten.csv"}
CANONICAL_TX = "ENST00000371953"
CANONICAL_NM = "NM_000314"
GRCH37_MINUS_GRCH38 = 1759757       # constant over the PTEN locus; re-checked on every run
CDS_LEN = 3 * (len(PTEN_UNIPROT) + 1)
AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = dict(Ala="A", Arg="R", Asn="N", Asp="D", Cys="C", Gln="Q", Glu="E", Gly="G", His="H", Ile="I",
	Leu="L", Lys="K", Met="M", Phe="F", Pro="P", Ser="S", Thr="T", Trp="W", Tyr="Y", Val="V")
BASES = "TCAG"
CODE = dict(zip((a + b + c for a, b, c in product(BASES, repeat=3)),
	"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"))
DOMAINS = [("N-term", 1, 13), ("phosphatase", 14, 185), ("C2", 186, 285), ("loop 286-309", 286, 309),
	("C2", 310, 351), ("C-tail", 352, 403)]
CLASS_ORDER = {"P/LP": 3, "conflicting": 2, "VUS": 1, "B/LB": 0, "other": -1}
DB_COLORS = {"cbioportal": "#c2571a", "clinvar": "#3b6fb6", "gnomad": "#2a7f62"}


def domain_of(pos):
	for name, a, b in DOMAINS:
		if a <= pos <= b:
			return name
	return "none"


def log(fh, *msg):
	text = " ".join(str(m) for m in msg)
	print(text, flush=True)
	if fh:
		fh.write(text + "\n")


def check_wt(df, where):
	"""Every wild-type letter must match canonical PTEN, or the numbering is wrong."""
	bad = df[[PTEN_UNIPROT[p - 1] != w if 1 <= p <= len(PTEN_UNIPROT) else True
		for p, w in zip(df.position, df.wt_aa)]]
	if len(bad):
		raise SystemExit(f"{where}: {len(bad)} wild-type letters disagree with canonical PTEN, e.g. "
			+ ", ".join(bad.variant.head(5)))


def norm_clinvar_class(text):
	t = str(text).lower().replace("_", " ")
	if t in ("", "nan"):
		return np.nan
	if "conflicting" in t:
		return "conflicting"
	if "pathogenic" in t and "benign" not in t and "uncertain" not in t:
		return "P/LP"
	if "benign" in t and "pathogenic" not in t and "uncertain" not in t:
		return "B/LB"
	if "uncertain" in t:
		return "VUS"
	return "other"


def review_stars(text):
	t = str(text).lower()
	if "practice guideline" in t:
		return 4
	if "expert panel" in t:
		return 3
	if "multiple submitters, no conflicts" in t:
		return 2
	if "single submitter" in t or "conflicting" in t:
		return 1
	return 0


def condition_groups(text):
	t = str(text).lower()
	out = []
	if any(k in t for k in ("hamartoma", "cowden", "bannayan", "lhermitte", "proteus", "pten-related")):
		out.append("PHTS")
	if any(k in t for k in ("autism", "macrocephaly")):
		out.append("autism/macrocephaly")
	if any(k in t for k in ("cancer", "neoplasm", "glioma", "carcinoma", "tumor")):
		out.append("cancer")
	return "|".join(out)


# ------------------------------------------------------------------- clean --- #

def clean_cbioportal(path, args, fh):
	c = pd.read_csv(path, sep="\t", low_memory=False)
	log(fh, f"\n== cBioPortal: {len(c)} rows, {c['Study of Origin'].nunique()} studies, "
		f"{c['Sample ID'].nunique()} samples")
	log(fh, "   mutation types: " + ", ".join(f"{k} {v}" for k, v in c["Mutation Type"].value_counts().items()))
	m = c[c["Mutation Type"] == "Missense_Mutation"].copy()
	code = m["Protein Change"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
	bad = code[0].isna()
	if bad.any():
		log(fh, f"   {bad.sum()} missense rows with an unparsable protein change dropped: "
			+ ", ".join(m["Protein Change"][bad].head(5)))
	m, code = m[~bad], code[~bad]
	m["wt_aa"], m["position"], m["mut_aa"] = code[0], code[1].astype(int), code[2]
	m["variant"] = m.wt_aa + m.position.astype(str) + m.mut_aa
	check_wt(m, "cBioPortal")
	tx = m.HGVSc.str.split(":").str[0].str.split(".").str[0]
	if (tx != CANONICAL_TX).any():
		log(fh, f"   WARNING: {int((tx != CANONICAL_TX).sum())} rows on another transcript")
	m["hgvsc"] = m.HGVSc.str.split(":").str[1]
	m["patient"] = m["Sample ID"].str[:12]
	m["grch37"] = m["Start Pos"].astype(int)
	m["grch38"] = m.grch37 - GRCH37_MINUS_GRCH38
	m["dna_key"] = "10:" + m.grch38.astype(str) + ":" + m.Ref + ":" + m.Var
	m["n_mut_sample"] = pd.to_numeric(m["# Mut in Sample"], errors="coerce")
	m["hypermutated"] = m.n_mut_sample > args.hypermutated
	m["second_hit"] = m["Copy #"].isin(["ShallowDel", "DeepDel"])
	m["oncokb_oncogenic"] = m.Annotation.fillna("").str.contains(r"OncoKB: (?:Likely )?Oncogenic")
	m["hotspot"] = m.Annotation.fillna("").str.contains("CancerHotspot: yes")
	m["alphamissense"] = pd.to_numeric(m["Functional Impact"].str.extract(r"AlphaMissense:[^;]*score: ([\d.]+)")[0],
		errors="coerce")
	m["clinvar_label_cbio"] = m["ClinVar"].map(norm_clinvar_class)
	n0 = len(m)
	assert not m.duplicated(["Sample ID", "variant"]).any(), "same sample, same change twice"
	dup = m.duplicated(["patient", "variant"], keep="first")
	log(fh, f"   missense rows {n0} in {m['Sample ID'].nunique()} samples / {m.patient.nunique()} patients; "
		f"exact duplicates (sample, change) 0; same patient + same change in two samples {int(dup.sum())} "
		f"-> dropped (one count per patient)")
	m = m[~dup]
	multi = m.groupby("Sample ID").size()
	log(fh, f"   samples with 2+ PTEN missense: {int((multi > 1).sum())} (kept: they become compound sequences)")
	nt = m.groupby("variant").dna_key.nunique()
	log(fh, f"   protein changes reached by 2+ DNA changes: {int((nt > 1).sum())} ({', '.join(nt[nt > 1].index)})")
	log(fh, f"   hypermutated samples (> {args.hypermutated} mutations): "
		f"{m.loc[m.hypermutated, 'Sample ID'].nunique()}, carrying {int(m.hypermutated.sum())} of the missense rows")
	keep = ["variant", "position", "wt_aa", "mut_aa", "hgvsc", "dna_key", "grch37", "grch38", "patient",
		"Sample ID", "Cancer Type", "Cancer Type Detailed", "Study of Origin", "n_mut_sample", "hypermutated",
		"Copy #", "second_hit", "Allele Freq (T)", "oncokb_oncogenic", "hotspot", "alphamissense",
		"clinvar_label_cbio", "Sample Type"]
	return m[keep].rename(columns={"Sample ID": "sample", "Cancer Type": "cancer_type",
		"Cancer Type Detailed": "cancer_type_detailed", "Study of Origin": "study", "Copy #": "copy_number",
		"Allele Freq (T)": "tumour_af", "Sample Type": "sample_type"})


def clean_clinvar(path, args, fh):
	with open(path) as f:
		n_head = len(f.readline().rstrip("\n").split("\t"))
		n_data = {len(l.rstrip("\n").split("\t")) for l in f}
	log(fh, f"\n== ClinVar: header {n_head} fields, data rows {sorted(n_data)} fields"
		+ ("  -> TRAP 1: reading with index_col=False so the labels do not shift" if n_data != {n_head} else ""))
	v = pd.read_csv(path, sep="\t", low_memory=False, index_col=False)
	if not v["Variant type"].astype(str).str.contains("single nucleotide variant").any():
		raise SystemExit("ClinVar columns look shifted: 'Variant type' holds no 'single nucleotide variant'")
	log(fh, f"   {len(v)} variations")
	mis = v[v["Molecular consequence"].fillna("").str.contains("missense")].copy()
	name = mis.Name.str.extract(r"^(NM_\d+)\.\d+\((\w+)\):(c\.\S+)(?: \(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\))?")
	on_pten = (name[0] == CANONICAL_NM) & (name[1] == "PTEN")
	log(fh, f"   with a missense consequence: {len(mis)}; named on {CANONICAL_NM}(PTEN): {int(on_pten.sum())}; "
		f"named on another gene (KLLN, TRAP 3): {int((name[1] != 'PTEN').sum())}")
	is_mis = on_pten & name[5].isin(list(THREE)) & name[3].isin(list(THREE))
	log(fh, f"   PTEN-named but no canonical missense p. in the name (PTEN-L-only or UTR, TRAP 2): "
		f"{int((on_pten & ~is_mis).sum())}")
	mis, name = mis[is_mis].copy(), name[is_mis]
	mis["wt_aa"] = name[3].map(THREE)
	mis["position"] = name[4].astype(int)
	mis["mut_aa"] = name[5].map(THREE)
	mis["hgvsc"] = name[2]
	mis["variant"] = mis.wt_aa + mis.position.astype(str) + mis.mut_aa
	check_wt(mis, "ClinVar")
	mis["clinvar_class"] = mis["Germline classification"].map(norm_clinvar_class)
	mis["stars"] = mis["Germline review status"].map(review_stars)
	mis["is_snv"] = mis["Variant type"] == "single nucleotide variant"
	spdi = mis["Canonical SPDI"].str.extract(r"^NC_000010\.\d+:(\d+):([ACGT]+):([ACGT]+)$")
	mis["dna_key"] = np.where(spdi[0].notna(),
		"10:" + (pd.to_numeric(spdi[0], errors="coerce") + 1).astype("Int64").astype(str) + ":" + spdi[1] + ":" + spdi[2],
		None)
	mis["grch37"] = pd.to_numeric(mis.GRCh37Location, errors="coerce")
	mis["grch38"] = pd.to_numeric(mis.GRCh38Location, errors="coerce")
	mis["condition_groups"] = mis["Condition(s)"].map(condition_groups)
	log(fh, f"   canonical missense rows {len(mis)}; duplicate VariationID {int(mis.VariationID.duplicated().sum())}, "
		f"duplicate SPDI {int(mis['Canonical SPDI'].duplicated().sum())}; non-SNV (MNV/indel) rows "
		f"{int((~mis.is_snv).sum())}")
	log(fh, "   classes: " + ", ".join(f"{k} {v}" for k, v in mis.clinvar_class.value_counts().items()))
	n_multi = mis.groupby("variant").size()
	log(fh, f"   protein changes carried by 2+ variations: {int((n_multi > 1).sum())} -> collapsed: class = most "
		"severe call among the rows at the highest review tier")
	rows = []
	for var, g in mis.groupby("variant", sort=False):
		top = g[g.stars == g.stars.max()]
		cls = max(top.clinvar_class.fillna("other"), key=lambda c: CLASS_ORDER.get(c, -1))
		rows.append({"variant": var, "position": g.position.iloc[0], "wt_aa": g.wt_aa.iloc[0],
			"mut_aa": g.mut_aa.iloc[0], "clinvar_class": cls, "clinvar_stars": int(g.stars.max()),
			"clinvar_classes_all": "|".join(sorted(set(g.clinvar_class.fillna("other")))),
			"clinvar_n_rows": len(g), "clinvar_any_snv": bool(g.is_snv.any()),
			"clinvar_variation_ids": "|".join(g.VariationID.astype(str)),
			"clinvar_conditions": "|".join(sorted({x for s in g.condition_groups for x in s.split("|") if x})),
			"clinvar_somatic": "|".join(sorted(set(g["Somatic clinical impact"].dropna()))),
			"clinvar_oncogenicity": "|".join(sorted(set(g["Oncogenicity classification"].dropna()))),
			"dna_keys": "|".join(g.dna_key.dropna()), "hgvsc": "|".join(g.hgvsc)})
	out = pd.DataFrame(rows)
	log(fh, f"   -> {len(out)} distinct protein changes")
	# The liftover offset used for cBioPortal, checked on every SNV that carries both builds.
	snv = v[v["Variant type"] == "single nucleotide variant"]
	d = (pd.to_numeric(snv.GRCh37Location, errors="coerce") - pd.to_numeric(snv.GRCh38Location, errors="coerce")).dropna()
	ok = (d == GRCH37_MINUS_GRCH38).mean()
	log(fh, f"   GRCh37 - GRCh38 on {len(d)} SNVs: {GRCH37_MINUS_GRCH38} for {ok:.1%} of them")
	if ok < 0.999:
		raise SystemExit("the GRCh37 -> GRCh38 offset is not constant over the locus; cBioPortal needs a real liftover")
	return out, mis, v


def clean_gnomad(path, args, fh):
	g = pd.read_csv(path, low_memory=False)
	log(fh, f"\n== gnomAD: {len(g)} alleles; transcripts: "
		+ ", ".join(f"{k} {v}" for k, v in g.Transcript.value_counts().items()))
	m = g[g["VEP Annotation"] == "missense_variant"].copy()
	tx = m.Transcript.str.split(".").str[0]
	log(fh, f"   missense {len(m)}; on {CANONICAL_TX}: {int((tx == CANONICAL_TX).sum())}; on ENST00000693560 "
		f"(PTEN-L extension = canonical 5' UTR, TRAP 2): {int((tx == 'ENST00000693560').sum())}; on others "
		f"(KLLN etc, TRAP 3): {int(((tx != CANONICAL_TX) & (tx != 'ENST00000693560')).sum())}")
	if (tx != CANONICAL_TX).any():
		top = m[tx != CANONICAL_TX].sort_values("Allele Frequency", ascending=False).head(3)
		log(fh, "   e.g. dropped: " + ", ".join(f"{p} AF {a:.3g}" for p, a in zip(top["Protein Consequence"],
			top["Allele Frequency"])))
	m = m[tx == CANONICAL_TX].copy()
	notpass = m["Filters - joint"] != "PASS"
	log(fh, f"   joint filter not PASS: {int(notpass.sum())} -> dropped")
	m = m[~notpass]
	code = m["Protein Consequence"].str.extract(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
	m["wt_aa"], m["position"], m["mut_aa"] = code[0].map(THREE), code[1].astype(int), code[2].map(THREE)
	m["variant"] = m.wt_aa + m.position.astype(str) + m.mut_aa
	check_wt(m, "gnomAD")
	m["hgvsc"] = m["Transcript Consequence"]
	m["dna_key"] = "10:" + m.Position.astype(str) + ":" + m.Reference + ":" + m.Alternate
	m["flag"] = m.Flags.fillna("")
	m["clinvar_label_gnomad"] = m["ClinVar Germline Classification"].map(norm_clinvar_class)
	log(fh, f"   duplicate variant IDs {int(m['gnomAD ID'].duplicated().sum())}, duplicate protein changes "
		f"{int(m.variant.duplicated().sum())}; flagged segdup/lcr {int((m.flag != '').sum())} (kept, flagged)")
	ac = m["Allele Count"]
	log(fh, f"   allele count: singletons {int((ac == 1).sum())}, 2+ {int((ac >= 2).sum())}, 10+ "
		f"{int((ac >= 10).sum())}; homozygotes present at {int((m['Homozygote Count'] > 0).sum())}")
	agg = m.groupby("variant").agg(position=("position", "first"), wt_aa=("wt_aa", "first"),
		mut_aa=("mut_aa", "first"), gnomad_ac=("Allele Count", "sum"), gnomad_an=("Allele Number", "max"),
		gnomad_af=("Allele Frequency", "sum"), gnomad_hom=("Homozygote Count", "sum"),
		gnomad_flag=("flag", lambda s: "|".join(sorted({x for x in s if x}))),
		gnomad_source=("Source", "first"), gnomad_faf=("GroupMax FAF frequency", "max"),
		cadd=("cadd", "max"), revel=("revel_max", "max"), dna_keys=("dna_key", "|".join),
		hgvsc=("hgvsc", "|".join), clinvar_label_gnomad=("clinvar_label_gnomad", "first")).reset_index()
	log(fh, f"   -> {len(agg)} distinct protein changes")
	return agg, m


def cds_codons(cb, cv_rows, cv_all, gn_all, raw_dir, fh):
	"""The canonical CDS, as far as the three files reveal it.

	Every SNV written as c.N[ACGT]>[ACGT] on the canonical transcript names the reference base at
	CDS position N (missense, synonymous, nonsense alike). Codons the files leave incomplete are
	narrowed by the protein: only codons that encode the wild-type residue and agree with the known
	bases are kept. A substitution is SNV-reachable when it is one base away from EVERY remaining
	candidate codon, unreachable when from none, and unknown otherwise."""
	known = {}

	def add(c):
		mm = re.match(r"^c\.(\d+)([ACGT])>([ACGT])$", str(c))
		if mm and 1 <= int(mm[1]) <= CDS_LEN:
			known.setdefault(int(mm[1]), set()).add(mm[2])

	for c in cb.hgvsc:
		add(c)
	for n in cv_all.Name.dropna():
		mm = re.match(rf"^{CANONICAL_NM}\.\d+\(PTEN\):(c\.\S+)", n)
		if mm:
			add(mm[1])
	for t, c in zip(gn_all.Transcript.astype(str), gn_all["Transcript Consequence"]):
		if t.startswith(CANONICAL_TX):
			add(c)
	conflict = [p for p, s in known.items() if len(s) > 1]
	if conflict:
		raise SystemExit(f"conflicting reference bases at CDS positions {conflict[:10]}")
	cands = []
	for i, wt in enumerate(PTEN_UNIPROT):
		pat = [next(iter(known[3 * i + k + 1])) if 3 * i + k + 1 in known else None for k in range(3)]
		cc = [cod for cod, aa in CODE.items() if aa == wt and all(p is None or p == b for p, b in zip(pat, cod))]
		if not cc:
			raise SystemExit(f"no codon for {wt}{i + 1} agrees with the bases {pat}")
		cands.append(cc)
	n_full = sum(all(3 * i + k + 1 in known for k in range(3)) for i in range(len(PTEN_UNIPROT)))
	n_one = sum(len(c) == 1 for c in cands)
	log(fh, f"\n== CDS from the files: {len(known)} of {CDS_LEN} bases named; codons fully named {n_full}, "
		f"resolved to one codon with the protein's help {n_one} of {len(PTEN_UNIPROT)}")

	def one_base(cod):
		return {CODE[cod[:k] + b + cod[k + 1:]] for k in range(3) for b in "ACGT" if b != cod[k]}

	rows = []
	for i, cc in enumerate(cands):
		sets = [one_base(c) for c in cc]
		for a in AA:
			if a == PTEN_UNIPROT[i]:
				continue
			hits = [a in s for s in sets]
			rows.append({"position": i + 1, "wt_aa": PTEN_UNIPROT[i], "mut_aa": a,
				"snv_files": True if all(hits) else (False if not any(hits) else np.nan),
				"codon": cc[0] if len(cc) == 1 else "|".join(cc)})
	snv = pd.DataFrame(rows)
	snv["snv"] = snv.snv_files
	src = os.path.join(raw_dir, "PTEN_variant_data.tsv")
	if os.path.exists(src):
		pv = pd.read_csv(src, sep="\t", low_memory=False)
		pv = pv[pv["class"] == "missense"][["position", "start", "end", "snv"]].rename(
			columns={"start": "wt_aa", "end": "mut_aa", "snv": "snv_vampseq"})
		pv["snv_vampseq"] = pv.snv_vampseq.map(lambda x: str(x).strip().lower() in ("true", "1", "yes"))
		snv = snv.merge(pv, on=["position", "wt_aa", "mut_aa"], how="left")
		both = snv.snv_files.notna() & snv.snv_vampseq.notna()
		agree = (snv.snv_files[both].astype(bool) == snv.snv_vampseq[both]).mean()
		log(fh, f"   PTEN_variant_data.tsv found: its `snv` column agrees with the files' codons on "
			f"{agree:.1%} of {int(both.sum())} substitutions; used wherever the files leave it unknown")
		snv["snv"] = snv.snv_files.where(snv.snv_files.notna(), snv.snv_vampseq)
	else:
		log(fh, f"   ({src} not found: SNV reachability from the files alone)")
	r = snv.snv
	log(fh, f"   substitutions: SNV-reachable {int((r == True).sum())}, not {int((r == False).sum())}, "
		f"unknown {int(r.isna().sum())} of {len(snv)}")
	return snv


def cmd_clean(args):
	os.makedirs(args.out_dir, exist_ok=True)
	with open(os.path.join(args.out_dir, "clean.log"), "w") as fh:
		cb = clean_cbioportal(os.path.join(args.db_dir, FILES["cbioportal"]), args, fh)
		cv, cv_rows, cv_all = clean_clinvar(os.path.join(args.db_dir, FILES["clinvar"]), args, fh)
		gn, gn_rows = clean_gnomad(os.path.join(args.db_dir, FILES["gnomad"]), args, fh)
		gn_all = pd.read_csv(os.path.join(args.db_dir, FILES["gnomad"]), low_memory=False)
		snv = cds_codons(cb, cv_rows, cv_all, gn_all, args.raw_dir, fh)
		# A variant observed as a single-base change must be reachable by one; a failure here
		# would mean the codon reconstruction is wrong.
		obs = pd.concat([cb[["position", "wt_aa", "mut_aa"]],
			cv_rows[cv_rows.is_snv][["position", "wt_aa", "mut_aa"]], gn_rows[["position", "wt_aa", "mut_aa"]]])
		chk = obs.merge(snv, on=["position", "wt_aa", "mut_aa"])
		bad = chk[chk.snv_files == False]          # noqa: E712
		log(fh, f"   check: of {len(chk)} variants observed as SNVs, {len(bad)} come out unreachable (must be 0)")
		if len(bad):
			raise SystemExit("codon reconstruction is inconsistent with the observed SNVs")
	cb.to_csv(os.path.join(args.out_dir, "cbioportal_missense_clean.tsv"), sep="\t", index=False)
	cv.to_csv(os.path.join(args.out_dir, "clinvar_missense_clean.tsv"), sep="\t", index=False)
	gn.to_csv(os.path.join(args.out_dir, "gnomad_missense_clean.tsv"), sep="\t", index=False)
	snv.to_csv(os.path.join(args.out_dir, "snv_reachability.tsv"), sep="\t", index=False)
	print(f"\nOutputs in {args.out_dir}: clean.log, cbioportal/clinvar/gnomad_missense_clean.tsv, snv_reachability.tsv")


# ------------------------------------------------------------------- merge --- #

def site_table():
	t = pd.read_csv(os.path.join(NAT_DIR, "pten_site_table.csv"))
	return t


def load_clean(args):
	f = lambda n: pd.read_csv(os.path.join(args.out_dir, n), sep="\t", low_memory=False)
	try:
		return (f("cbioportal_missense_clean.tsv"), f("clinvar_missense_clean.tsv"), f("gnomad_missense_clean.tsv"),
			f("snv_reachability.tsv"))
	except FileNotFoundError:
		raise SystemExit("run `python pten_variants.py clean` first")


def cmd_merge(args):
	cb, cv, gn, snv = load_clean(args)
	fh = open(os.path.join(args.out_dir, "merge.log"), "w")
	t = site_table()
	nat = t[t.native]
	pos2site = dict(zip(nat.uniprot_pos, nat.site))

	agg = cb.groupby("variant").agg(position=("position", "first"), wt_aa=("wt_aa", "first"),
		mut_aa=("mut_aa", "first"), cbio_patients=("patient", "nunique"),
		cbio_patients_nonhyper=("patient", lambda s: s[~cb.loc[s.index, "hypermutated"]].nunique()),
		cbio_patients_second_hit=("patient", lambda s: s[cb.loc[s.index, "second_hit"]].nunique()),
		cbio_cancer_types=("cancer_type", lambda s: "|".join(sorted(set(s)))),
		cbio_endometrial=("cancer_type", lambda s: int((s == "Endometrial Cancer").sum())),
		cbio_glioblastoma=("cancer_type", lambda s: int((s == "Glioblastoma").sum())),
		cbio_oncokb_oncogenic=("oncokb_oncogenic", "any"), cbio_hotspot=("hotspot", "any"),
		alphamissense=("alphamissense", "max"), clinvar_label_cbio=("clinvar_label_cbio", "first"),
		cbio_dna_keys=("dna_key", lambda s: "|".join(sorted(set(s))))).reset_index()

	m = (agg.merge(cv.rename(columns={"dna_keys": "clinvar_dna_keys", "hgvsc": "clinvar_hgvsc"}),
			on=["variant", "position", "wt_aa", "mut_aa"], how="outer")
		.merge(gn.rename(columns={"dna_keys": "gnomad_dna_keys", "hgvsc": "gnomad_hgvsc"}),
			on=["variant", "position", "wt_aa", "mut_aa"], how="outer"))
	m["in_cbioportal"] = m.cbio_patients.notna()
	m["in_clinvar"] = m.clinvar_class.notna()
	m["in_gnomad"] = m.gnomad_ac.notna()
	m["cbio_patients"] = m.cbio_patients.fillna(0).astype(int)
	m["cbio_patients_nonhyper"] = m.cbio_patients_nonhyper.fillna(0).astype(int)
	m["gnomad_ac"] = m.gnomad_ac.fillna(0).astype(int)
	m["domain"] = m.position.map(domain_of)
	m["site"] = m.position.map(pos2site).astype("Int64")
	m["in_construct"] = m.site.notna()
	m = m.merge(snv[["position", "wt_aa", "mut_aa", "snv"]], on=["position", "wt_aa", "mut_aa"], how="left")
	m["sets"] = (m.in_cbioportal.map({True: "C", False: ""}) + m.in_clinvar.map({True: "V", False: ""})
		+ m.in_gnomad.map({True: "G", False: ""}))
	m = m.sort_values(["position", "mut_aa"]).reset_index(drop=True)

	log(fh, f"== merged: {len(m)} distinct canonical missense changes; cBioPortal {int(m.in_cbioportal.sum())}, "
		f"ClinVar {int(m.in_clinvar.sum())}, gnomAD {int(m.in_gnomad.sum())}")
	names = {"C": "cBioPortal only", "V": "ClinVar only", "G": "gnomAD only", "CV": "cBioPortal + ClinVar",
		"CG": "cBioPortal + gnomAD", "VG": "ClinVar + gnomAD", "CVG": "all three"}
	ov = m.sets.value_counts().reindex(list(names)).fillna(0).astype(int)
	log(fh, "   overlaps at the protein level (exclusive regions):")
	for k, v in ov.items():
		log(fh, f"     {names[k]:22s} {v:5d}   in construct {int(((m.sets == k) & m.in_construct).sum()):5d}")
	log(fh, f"   inside the 324-aa construct: {int(m.in_construct.sum())}; outside: {int((~m.in_construct).sum())} "
		"(" + ", ".join(f"{k} {v}" for k, v in m[~m.in_construct].domain.value_counts().items()) + ")")

	# The same overlaps at the DNA level, so a protein change reached through two codons stays visible.
	dna = {}
	for db, col in (("C", "cbio_dna_keys"), ("V", "clinvar_dna_keys"), ("G", "gnomad_dna_keys")):
		dna[db] = {k for s in m[col].dropna() for k in str(s).split("|") if k and k != "None"}
	log(fh, f"   DNA level (GRCh38 chr:pos:ref:alt; cBioPortal lifted by -{GRCH37_MINUS_GRCH38}): C {len(dna['C'])}, "
		f"V {len(dna['V'])}, G {len(dna['G'])}; C&V {len(dna['C'] & dna['V'])}, C&G {len(dna['C'] & dna['G'])}, "
		f"V&G {len(dna['V'] & dna['G'])}, all {len(dna['C'] & dna['V'] & dna['G'])}")
	same_aa_other_codon = m[m.in_cbioportal & m.in_clinvar].apply(
		lambda r: not (set(str(r.cbio_dna_keys).split("|")) & set(str(r.clinvar_dna_keys).split("|"))), axis=1).sum()
	log(fh, f"   protein changes shared by cBioPortal and ClinVar through a DIFFERENT DNA change: {int(same_aa_other_codon)}")

	# The databases' own cross-references against the actual ClinVar file.
	for col, who in (("clinvar_label_cbio", "cBioPortal's ClinVar column"),
			("clinvar_label_gnomad", "gnomAD's ClinVar column")):
		d = m[m[col].notna()]
		both = d[d.clinvar_class.notna()]
		dis = both[both[col] != both.clinvar_class]
		log(fh, f"   {who}: {len(d)} variants labelled, {len(both)} in the ClinVar file, {len(dis)} disagree"
			+ (": " + ", ".join(f"{v} {a}->{b}" for v, a, b in zip(dis.variant, dis[col], dis.clinvar_class))[:600]
				if len(dis) else ""))
		miss = d[d.clinvar_class.isna()]
		if len(miss):
			log(fh, f"     labelled there but absent from the ClinVar file: {len(miss)} ({', '.join(miss.variant.head(10))})")
	log(fh, f"   tumour changes also in gnomAD: {int((m.in_cbioportal & m.in_gnomad).sum())}: "
		+ ", ".join(f"{v} ({p} pt, AC {a})" for v, p, a in zip(*[m[m.in_cbioportal & m.in_gnomad][c]
			for c in ("variant", "cbio_patients", "gnomad_ac")])))
	pg = m[m.in_gnomad & (m.clinvar_class == "P/LP")]
	log(fh, f"   ClinVar P/LP also in gnomAD: {len(pg)} ({', '.join(f'{v} AC {a}' for v, a in zip(pg.variant, pg.gnomad_ac))})")
	r = m.snv
	log(fh, f"   SNV-reachable: {int((r == True).sum())}, not {int((r == False).sum())}, unknown {int(r.isna().sum())}; "
		f"not reachable but observed: " + ", ".join(f"{v} ({s})" for v, s in zip(m[r == False].variant, m[r == False].sets)))

	# Compound tumour sequences for `score`: every sample with 2+ PTEN missense changes, construct only.
	ref = "".join(t.ref_aa)
	rows = []
	for (sample, patient), g in cb.groupby(["sample", "patient"]):
		if len(g) < 2:
			continue
		inside = g[g.position.isin(pos2site)]
		seq = list(ref)
		for p, a in zip(inside.position, inside.mut_aa):
			seq[pos2site[p]] = a
		rows.append({"sample": sample, "patient": patient, "cancer_type": g.cancer_type.iloc[0],
			"hypermutated": bool(g.hypermutated.iloc[0]), "n_mut_sample": g.n_mut_sample.iloc[0],
			"n_missense": len(g), "n_in_construct": len(inside),
			"mutations": "|".join(g.variant), "mutations_in_construct": "|".join(inside.variant),
			"sites": "|".join(str(pos2site[p]) for p in inside.position), "sequence": "".join(seq)})
	comp = pd.DataFrame(rows)
	log(fh, f"   compound tumour sequences: {len(comp)} samples with 2+ PTEN missense, {int((comp.n_in_construct >= 2).sum())} "
		f"with 2+ inside the construct ({int(comp.hypermutated.sum())} hypermutated)")
	fh.close()
	m.to_csv(os.path.join(args.out_dir, "pten_variants_merged.tsv"), sep="\t", index=False)
	comp.to_csv(os.path.join(args.out_dir, "compound_sequences.tsv"), sep="\t", index=False)
	plot_overview(m, ov, names, os.path.join(args.out_dir, "variants_overview.png"))
	print(f"\nOutputs in {args.out_dir}: merge.log, pten_variants_merged.tsv, compound_sequences.tsv, variants_overview.png")


def _snv_pair_mask():
	"""20x20 boolean, True where wt -> mut can be made by one base change from SOME codon of wt
	(standard code); the 150 reachable pairs of the 380."""
	b = "TCAG"
	aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
	code = {x + y + z: aas[16 * i + 4 * j + k] for i, x in enumerate(b) for j, y in enumerate(b) for k, z in enumerate(b)}
	ok = np.zeros((20, 20), bool)
	for c, a in code.items():
		if a == "*":
			continue
		for p in range(3):
			for n in b:
				d = code[c[:p] + n + c[p + 1:]]
				if d not in ("*", a):
					ok[AA.index(a), AA.index(d)] = True
	return ok


def plot_overview(m, ov, names, path):
	fig = plt.figure(figsize=(12, 7.5))
	gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1])
	ax = fig.add_subplot(gs[0, :])
	pos = np.arange(1, len(PTEN_UNIPROT) + 1)
	for db in ("clinvar", "cbioportal", "gnomad"):
		y = m[m[f"in_{db}"]].groupby("position").size().reindex(pos, fill_value=0)
		ax.bar(pos, y.values, color=DB_COLORS[db], alpha=.55, width=1, label=f"{db} ({int(m[f'in_{db}'].sum())})")
	for name, a, b in DOMAINS:
		ax.axvspan(a - .5, b + .5, color={"N-term": "#eee", "phosphatase": "#f6efe4", "C2": "#e8eef7",
			"loop 286-309": "#ddd", "C-tail": "#eee"}[name], zorder=-1)
	ax.set_xlim(0, len(PTEN_UNIPROT) + 1)
	ax.set_xlabel("UniProt position (canonical PTEN, 403 aa); shading: phosphatase, C2, the loop absent from the construct")
	ax.set_ylabel("distinct missense\nchanges per site")
	ax.legend(fontsize=8, ncol=3)
	ax2 = fig.add_subplot(gs[1:, 0])
	order = list(names)
	ax2.barh(range(len(order)), ov.values, color="#666")
	ax2.set_yticks(range(len(order)), [names[k] for k in order], fontsize=8)
	for i, v in enumerate(ov.values):
		ax2.text(v, i, f" {v}", va="center", fontsize=8)
	ax2.invert_yaxis()
	ax2.set_title("overlaps (protein level)", fontsize=9)
	for j, db in enumerate(("cbioportal", "gnomad")):
		ax3 = fig.add_subplot(gs[1:, 1 + j])
		d = m[m[f"in_{db}"]]
		mat = pd.crosstab(d.wt_aa, d.mut_aa).reindex(index=list(AA), columns=list(AA), fill_value=0)
		reach = _snv_pair_mask()
		ax3.imshow(np.where(reach, np.nan, 1.0), cmap="Blues", vmin=0, vmax=8)  # pale blue = no one-base change
		im = ax3.imshow(np.ma.masked_where(~reach, mat.values), cmap="Greys", vmin=0,
			vmax=max(1, int(mat.values.max())))
		cb_ = fig.colorbar(im, ax=ax3, fraction=.046, pad=.03)
		cb_.set_label("distinct changes in PTEN", fontsize=7)
		cb_.ax.tick_params(labelsize=6)
		ax3.set_xticks(range(20), list(AA), fontsize=6)
		ax3.set_yticks(range(20), list(AA), fontsize=6)
		ax3.set_xlabel("mutant", fontsize=8)
		ax3.set_ylabel("wild type", fontsize=8)
		ax3.set_title(f"{db}: substitution matrix", fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=130)
	plt.close(fig)


# -------------------------------------------------------------- categories --- #

def categories(m):
	"""name -> (group, boolean mask over the merged table). The per-database categories come first,
	then the merged ones; `all` and the plots keep this order."""
	cls = m.clinvar_class
	tolerated = m.in_gnomad & (m.gnomad_ac >= 2) & (cls != "P/LP")
	damaging = (cls == "P/LP") | (m.cbio_patients >= 2)
	cond = m.clinvar_conditions.fillna("")
	return {
		"cbio_all": ("cBioPortal", m.in_cbioportal),
		"cbio_recurrent": ("cBioPortal", m.cbio_patients >= 2),
		"cbio_singleton": ("cBioPortal", m.cbio_patients == 1),
		"cbio_nonhyper": ("cBioPortal", m.cbio_patients_nonhyper >= 1),
		"cbio_endometrial": ("cBioPortal", m.cbio_endometrial.fillna(0) > 0),
		"cbio_glioblastoma": ("cBioPortal", m.cbio_glioblastoma.fillna(0) > 0),
		"cbio_second_hit": ("cBioPortal", m.cbio_patients_second_hit.fillna(0) > 0),
		"clinvar_all": ("ClinVar", m.in_clinvar),
		"clinvar_PLP": ("ClinVar", cls == "P/LP"),
		"clinvar_VUS": ("ClinVar", cls == "VUS"),
		"clinvar_conflicting": ("ClinVar", cls == "conflicting"),
		"clinvar_expert": ("ClinVar", m.clinvar_stars.fillna(0) >= 3),
		"clinvar_PHTS": ("ClinVar", cond.str.contains("PHTS")),
		"clinvar_autism": ("ClinVar", cond.str.contains("autism")),
		"gnomad_all": ("gnomAD", m.in_gnomad),
		"gnomad_singleton": ("gnomAD", m.in_gnomad & (m.gnomad_ac == 1)),
		"gnomad_ac2": ("gnomAD", m.in_gnomad & (m.gnomad_ac >= 2)),
		"merged_all": ("merged", m.in_cbioportal | m.in_clinvar | m.in_gnomad),
		"tumour_only": ("merged", m.in_cbioportal & ~m.in_gnomad),
		"population_only": ("merged", m.in_gnomad & ~m.in_cbioportal),
		"tumour_and_population": ("merged", m.in_cbioportal & m.in_gnomad),
		"consensus_damaging": ("merged", damaging & ~tolerated),
		"consensus_tolerated": ("merged", tolerated & ~damaging),
	}


# ---------------------------------------------------------------- profiles --- #

def sampled_entropy(args, t, fh):
	"""Mean per-site entropy over the chains of each temperature, kept window = last 1 - burn_in."""
	ref = "".join(t.ref_aa)
	out = pd.DataFrame(index=t.site)
	for T in args.temperatures:
		cols = []
		for c in sorted(os.listdir(os.path.join(args.runs_dir, T))):
			path = os.path.join(args.runs_dir, T, c)
			if not os.path.isfile(os.path.join(path, "start.txt")):
				continue
			st, info = chain_site_stats(path, ref, args.burn_in_frac)
			cols.append(st.entropy.values)
			log(fh, f"   {T}/{c}: {info['moves']} moves, start {'reference' if info['start_is_reference'] else 'other'}, "
				f"kept from move {int(args.burn_in_frac * info['moves'])}, mean entropy {st.entropy.mean():.3f}, "
				f"sites at 0: {int((st.entropy == 0).sum())}")
		out[f"sampled_{T}"] = np.mean(cols, axis=0)
	return out


def entropy_of(counts):
	p = np.asarray(counts, float)
	p = p[p > 0] / p.sum() if p.sum() > 0 else p[:0]
	return float(-(p * np.log(p)).sum()) if len(p) else np.nan


def cmd_profiles(args):
	m = pd.read_csv(os.path.join(args.out_dir, "pten_variants_merged.tsv"), sep="\t", low_memory=False)
	snv = pd.read_csv(os.path.join(args.out_dir, "snv_reachability.tsv"), sep="\t")
	t = site_table()
	fh = open(os.path.join(args.out_dir, "profiles.log"), "w")
	log(fh, f"== sampled entropy, {args.runs_dir}, burn-in fraction {args.burn_in_frac}")
	se = sampled_entropy(args, t, fh)
	t = t.join(se, on="site")
	# Sampler K-sites: the least variable --sampled-k-frac of the native sites at each temperature
	# (at T >= 4 almost no site sits at exactly 0, so a threshold at 0 would leave a handful).
	ksets = {"K_natural": t.K_natural, "K_abundance": t.K_abundance, "K_activity": t.K_activity}
	for T in args.temperatures:
		col = f"sampled_{T}"
		H = t.loc[t.native, col]
		if args.sampled_k_sigma is None:
			cut = H.quantile(args.sampled_k_frac)
			t[f"K_sampled_{T}"] = t.native & (t[col] <= cut)
			rule = f"lowest {args.sampled_k_frac:.0%} of the native sites"
		else:
			# A site is a sampler K-site when its entropy lies more than k standard deviations below
			# the median of the native sites. 'mad' uses 1.4826 x the median absolute deviation, the
			# standard deviation of a Gaussian with that MAD, robust to the skew of a quantity bounded
			# at 0; 'sd' uses the plain standard deviation. Strict '<': when more than half of the
			# sites are frozen at 0 (T <= 3) the median is about 0 and no site can lie below it.
			med = H.median()
			sig = 1.4826 * (H - med).abs().median() if args.sampled_k_scale == "mad" else H.std()
			cut = med - args.sampled_k_sigma * sig
			t[f"K_sampled_{T}"] = t.native & (t[col] < cut)
			rule = (f"median {med:.3f} - {args.sampled_k_sigma:g} x sigma {sig:.3f} "
				f"({args.sampled_k_scale}); frozen sites (entropy 0): {int((H == 0).sum())}")
		ksets[f"K_sampled_{T}"] = t[f"K_sampled_{T}"]
		log(fh, f"   K_sampled_{T}: {int(t[f'K_sampled_{T}'].sum())} sites at entropy below {cut:.3f} ({rule})")

	# Per site: how many SNV-reachable substitutions exist (the denominator of every density).
	nat = t[t.native].copy()
	s = snv.merge(nat[["uniprot_pos", "site"]], left_on="position", right_on="uniprot_pos")
	reach = s[s.snv == True].groupby("site").size().reindex(t.site, fill_value=0)       # noqa: E712
	t["n_reachable"] = reach.values
	log(fh, f"   SNV-reachable substitutions inside the construct: {int(reach.sum())} over {int(t.native.sum())} native sites")

	mc = m[m.in_construct].copy()
	mc["site"] = mc.site.astype(int)
	cats = categories(mc)
	weights = {"cBioPortal": "cbio_patients", "gnomAD": "gnomad_ac"}
	prof, enr, cor = {"site": t.site, "uniprot_pos": t.uniprot_pos, "ref_aa": t.ref_aa}, [], []
	refs = {"family entropy": t.nat_family_entropy, "abundance mean": t.abundance2021_mean,
		"activity mean": t.activity_mean, **{f"sampled {T}": t[f"sampled_{T}"] for T in args.temperatures}}
	for name, (group, mask) in cats.items():
		d = mc[mask.values]
		if len(d) < args.min_category:
			log(fh, f"   {name}: {len(d)} variants in the construct, below --min-category {args.min_category}: counts only")
			enr.append({"category": name, "group": group, "n_variants": len(d)})
			continue
		w = d[weights[group]] if group in weights else pd.Series(1, index=d.index)
		n_obs = d.groupby("site").size().reindex(t.site, fill_value=0)
		n_obs_r = d[d.snv == True].groupby("site").size().reindex(t.site, fill_value=0)      # noqa: E712
		dens = np.where(t.n_reachable > 0, n_obs_r / t.n_reachable.replace(0, np.nan), np.nan)
		H = [entropy_of(g.groupby("mut_aa")[0].sum()) if len(g) else 0.0
			for _, g in pd.concat([d.site, w.rename(0), d.mut_aa], axis=1).groupby("site")]
		Hs = pd.Series(H, index=sorted(d.site.unique())).reindex(t.site, fill_value=0.0)
		prof[f"{name}_n"] = n_obs.values
		prof[f"{name}_density"] = dens
		prof[f"{name}_subst_entropy"] = Hs.values
		# Enrichment: distinct SNV-reachable variants inside each K-set against a uniform draw over every
		# SNV-reachable substitution of the construct (binomial), plus the circular-shift p of the
		# per-site density, which keeps the sequence's run structure.
		from scipy.stats import binom
		N = int(n_obs_r.sum())
		for kname, kmask in ksets.items():
			kmask = np.asarray(kmask, bool)
			frac = t.n_reachable[kmask].sum() / t.n_reachable.sum()
			k = int(n_obs_r[kmask].sum())
			exp = N * frac
			p_hi, p_lo = binom.sf(k - 1, N, frac), binom.cdf(k, N, frac)
			ok = np.isfinite(dens)
			enr.append({"category": name, "group": group, "n_variants": len(d), "n_snv_in_construct": N,
				"kset": kname, "kset_sites": int(kmask.sum()), "observed": k, "expected": exp,
				"ratio": k / exp if exp else np.nan, "p_two_sided": min(1.0, 2 * min(p_hi, p_lo)),
				"density_shift_p": shift_p(np.where(ok, dens, 0.0), kmask)})
		row = {"category": name, "group": group, "n_variants": len(d)}
		for rname, rv in refs.items():
			row[f"rho density vs {rname}"] = spearman(dens, rv)
			row[f"rho density vs {rname} | RSA"] = partial_spearman(dens, rv, t.rsa)
		cor.append(row)
	prof = pd.DataFrame(prof)
	for T in args.temperatures:
		prof[f"sampled_{T}"] = t[f"sampled_{T}"].values
	for k in ksets:
		prof[k] = np.asarray(ksets[k], bool)
	enr, cor = pd.DataFrame(enr), pd.DataFrame(cor)
	pd.set_option("display.width", 250)
	log(fh, "\n== variants in each K-set: observed / expected under a uniform draw over the SNV-reachable substitutions")
	piv = enr.dropna(subset=["kset"]).pivot_table(index="category", columns="kset", values="ratio", sort=False)
	log(fh, piv.to_string(float_format=lambda x: f"{x:.2f}"))
	log(fh, "\n   two-sided binomial p:")
	log(fh, enr.dropna(subset=["kset"]).pivot_table(index="category", columns="kset", values="p_two_sided",
		sort=False).to_string(float_format=lambda x: f"{x:.1e}"))
	log(fh, "\n== Spearman of the per-site density (SNV-reachable, 'higher = more variants land here') with each "
		"reference ('higher = more tolerant'); a NEGATIVE value means variants land on conserved/intolerant sites")
	log(fh, cor.set_index("category").to_string(float_format=lambda x: f"{x:+.3f}"))
	fh.close()
	prof.to_csv(os.path.join(args.out_dir, "kset_profiles.tsv"), sep="\t", index=False, float_format="%.4f")
	enr.to_csv(os.path.join(args.out_dir, "kset_enrichment.tsv"), sep="\t", index=False, float_format="%.4g")
	cor.to_csv(os.path.join(args.out_dir, "kset_density_correlations.tsv"), sep="\t", index=False, float_format="%.4f")
	plot_profiles(prof, t, cats, args, os.path.join(args.out_dir, "kset_profiles"))
	plot_enrichment(piv, os.path.join(args.out_dir, "kset_enrichment.png"))
	print(f"\nOutputs in {args.out_dir}: profiles.log, kset_profiles.tsv, kset_enrichment.tsv, "
		"kset_density_correlations.tsv, kset_enrichment.png, kset_profiles/*.png")


def plot_profiles(prof, t, cats, args, out):
	os.makedirs(out, exist_ok=True)
	x = prof.site.values
	ticks = [p for p in range(10, 354, 20) if (t.uniprot_pos == p).any()]
	tick_x = [int(t.site[t.uniprot_pos == p].iloc[0]) for p in ticks]
	kcols = [c for c in prof.columns if c.startswith("K_")]
	for name, (group, _) in cats.items():
		if f"{name}_density" not in prof:
			continue
		fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True,
			gridspec_kw={"height_ratios": [1.2, 1, 1, .9]})
		for T, c in zip(args.temperatures, ("#888", "#222", "#555", "#aaa")):
			axes[0].plot(x, prof[f"sampled_{T}"], lw=.8, color=c, label=f"sampled entropy {T}")
		axes[0].plot(x, t.nat_family_entropy, lw=.8, color="#2a7f62", label="family entropy")
		axes[0].set_ylabel("entropy [nats]", fontsize=8)
		axes[0].legend(fontsize=7, ncol=4)
		axes[1].bar(x, prof[f"{name}_density"], width=1, color="#c2571a")
		axes[1].set_ylabel("variant density\n(of SNV-reachable)", fontsize=8)
		axes[2].bar(x, prof[f"{name}_subst_entropy"], width=1, color="#3b6fb6")
		axes[2].set_ylabel("substitution\nentropy [nats]", fontsize=8)
		for k, c in enumerate(kcols):
			s = x[prof[c].values]
			axes[3].scatter(s, np.full(len(s), k), marker="|", s=40, color="k")
		axes[3].set_yticks(range(len(kcols)), kcols, fontsize=7)
		axes[3].set_xticks(tick_x, ticks, fontsize=7)
		axes[3].set_xlabel("UniProt position (construct sites)")
		axes[0].set_title(f"{name} ({group}): {int(prof[f'{name}_n'].sum())} variants in the construct", fontsize=10)
		fig.tight_layout()
		fig.savefig(os.path.join(out, f"{name}.png"), dpi=120)
		plt.close(fig)


def plot_enrichment(piv, path):
	fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 3, .32 * piv.shape[0] + 1.2))
	v = np.log2(piv.values.astype(float))
	im = ax.imshow(v, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
	ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", fontsize=8)
	ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
	for i in range(piv.shape[0]):
		for j in range(piv.shape[1]):
			if np.isfinite(piv.values[i, j]):
				ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
	fig.colorbar(im, ax=ax, label="log2 observed / expected")
	ax.set_title("variants inside each K-set vs a uniform draw over SNV-reachable substitutions", fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=130)
	plt.close(fig)


# -------------------------------------------------------------------- rank --- #

def load_scans():
	am = pd.read_csv(os.path.join(NAT_DIR, "pten_single_mutants.csv")).rename(columns={"dU": "U_am"})
	e2 = pd.read_csv(os.path.join(NAT_DIR, "pten_mutants_E2.csv"))[["site", "mut_aa", "E"]].rename(columns={"E": "E2"})
	ce = pd.read_csv(os.path.join(NAT_DIR, "pten_mutants_ce.csv"))[["site", "mut_aa", "dU_ce"]].rename(
		columns={"dU_ce": "U_ce"})
	s = am.merge(e2, on=["site", "mut_aa"]).merge(ce, on=["site", "mut_aa"])
	if len(s) != 6156:
		raise SystemExit(f"the three scans share {len(s)} mutants, expected 6156")
	return s


def boot_auc(score, label, n, rng):
	score, label = np.asarray(score, float), np.asarray(label, bool)
	ok = np.isfinite(score)
	score, label = score[ok], label[ok]
	a = auc(score, label)
	if not np.isfinite(a):
		return a, np.nan, np.nan, int(label.sum()), int((~label).sum())
	pos, neg = np.where(label)[0], np.where(~label)[0]
	bs = []
	for _ in range(n):
		i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
		bs.append(auc(score[i], label[i]))
	lo, hi = np.nanpercentile(bs, [2.5, 97.5])
	return a, lo, hi, len(pos), len(neg)


def within_site_auc(site, score, pos, neg):
	"""P(positive scores above negative) over pairs drawn from the SAME site (ties count half),
	pooled over sites. Returns the AUC, the number of pairs and of sites that contributed."""
	conc, n, sites = 0.0, 0, 0
	for s in np.unique(site[pos]):
		at = site == s
		a = score[at & pos]
		b = score[at & neg]
		a, b = a[np.isfinite(a)], b[np.isfinite(b)]
		if not len(a) or not len(b):
			continue
		diff = a[:, None] - b[None, :]
		conc += (diff > 0).sum() + 0.5 * (diff == 0).sum()
		n += diff.size
		sites += 1
	return (conc / n if n else np.nan), n, sites


def cmd_rank(args):
	m = pd.read_csv(os.path.join(args.out_dir, "pten_variants_merged.tsv"), sep="\t", low_memory=False)
	snv = pd.read_csv(os.path.join(args.out_dir, "snv_reachability.tsv"), sep="\t")
	fh = open(os.path.join(args.out_dir, "rank.log"), "w")
	t = site_table()
	scans = load_scans()
	energies = ["U_am", "E2", "U_ce"]
	nat = t[t.native][["site", "uniprot_pos"]]
	scans = scans.merge(nat, on="site").rename(columns={"uniprot_pos": "position", "ref_aa": "wt_aa"})
	scans = scans.merge(snv[["position", "wt_aa", "mut_aa", "snv"]], on=["position", "wt_aa", "mut_aa"], how="left")
	for e in energies:
		scans[f"{e}_pct"] = scans[e].rank(pct=True) * 100
		scans[f"{e}_pct_snv"] = scans[e].where(scans.snv == True).rank(pct=True) * 100       # noqa: E712
		scans[f"{e}_rank_in_site"] = scans.groupby("site")[e].rank().astype(int)
	exp = pd.read_csv(os.path.join(NAT_DIR, "pten_mutants_variant_level_Uam.csv"))[
		["site", "mut_aa", "abundance_2021", "abundance2021_is_low", "activity", "activity_is_lof"]]
	scans = scans.merge(exp, on=["site", "mut_aa"], how="left")
	m = m.merge(scans.drop(columns=["site", "snv"]), on=["position", "wt_aa", "mut_aa"], how="left")
	log(fh, f"== {len(m)} variants, {int(m.U_am.notna().sum())} found in the scans (the rest lie outside the construct)")

	# Optional predictors from PTEN_variant_data.tsv (astaroth), on every variant alike.
	pv_path = os.path.join(args.raw_dir, "PTEN_variant_data.tsv")
	pred_cols = []
	if os.path.exists(pv_path):
		pv = pd.read_csv(pv_path, sep="\t", low_memory=False)
		pv = pv[pv["class"] == "missense"].rename(columns={"start": "wt_aa", "end": "mut_aa"})
		num = [c for c in pv.columns if pv[c].dtype.kind == "f" and any(k in c.lower() for k in
			("sift", "polyphen", "cadd", "provean", "fathmm", "snap", "mutation", "lrt", "meta", "m_cap", "mcap",
			"ptenpred", "fraction_deleterious"))]
		pv = pv[["position", "wt_aa", "mut_aa"] + num].drop_duplicates(["position", "wt_aa", "mut_aa"])
		m = m.merge(pv, on=["position", "wt_aa", "mut_aa"], how="left")
		pred_cols = num
		log(fh, f"   predictors from PTEN_variant_data.tsv: {', '.join(num)}")
	else:
		log(fh, f"   ({pv_path} not found: the 15 predictors are left out)")

	cats = categories(m)
	c = {k: v[1].values for k, v in cats.items()}
	contrasts = [
		("tumour (any) vs population-only", c["cbio_all"], c["population_only"]),
		("tumour recurrent vs gnomAD AC>=2", c["cbio_recurrent"], c["gnomad_ac2"] & ~c["cbio_all"]),
		("tumour non-hypermutated vs population-only", c["cbio_nonhyper"], c["population_only"]),
		("tumour recurrent vs tumour singleton", c["cbio_recurrent"], c["cbio_singleton"]),
		("ClinVar P/LP vs gnomAD (not P/LP)", c["clinvar_PLP"], c["gnomad_all"] & ~c["clinvar_PLP"]),
		("ClinVar P/LP vs ClinVar VUS", c["clinvar_PLP"], c["clinvar_VUS"]),
		("ClinVar PHTS vs autism/macrocephaly", c["clinvar_PHTS"] & ~c["clinvar_autism"],
			c["clinvar_autism"] & ~c["clinvar_PHTS"]),
		("consensus damaging vs consensus tolerated", c["consensus_damaging"], c["consensus_tolerated"]),
		("observed anywhere vs never observed", None, None),
	]
	# Scores oriented so that HIGHER = predicted damaging; AUC = P(positive scores above negative).
	scores = {e: m[e] for e in energies}
	# The same energies averaged over the site's 19 mutants: what a SITE-level readout knows. The
	# contact-map verdict says E2 is only usable this way; the gap between the two rows is what the
	# energy adds by telling substitutions apart.
	for e in energies:
		scores[f"{e} site mean"] = m.site.astype(float).map(scans.groupby("site")[e].mean())
	scores["-abundance_2021"] = -m.abundance_2021
	scores["-activity"] = -m.activity
	scores["AlphaMissense (cBioPortal only)"] = m.alphamissense
	for p in pred_cols:
		scores[p] = m[p]
	rng = np.random.default_rng(0)
	rows = []
	inside = m.U_am.notna().values
	for restrict in ("all", "snv"):
		base = inside & ((m.snv == True).values if restrict == "snv" else True)          # noqa: E712
		for name, pos, neg in contrasts:
			if pos is None:
				# every SNV-reachable mutant of the scan: observed in any database vs in none
				obs = set(zip(m.position, m.mut_aa))
				d = scans[(scans.snv == True) if restrict == "snv" else scans.snv.notna() | True]     # noqa: E712
				lab = np.array([(p_, a_) in obs for p_, a_ in zip(d.position, d.mut_aa)])
				for sname in energies + ["-abundance_2021", "-activity"]:
					sc = -d[sname[1:]] if sname.startswith("-") else d[sname]
					a, lo, hi, n1, n0 = boot_auc(sc, lab, args.bootstrap, rng)
					rows.append({"restrict": restrict, "contrast": name, "score": sname, "auc": a, "lo": lo, "hi": hi,
						"n_pos": n1, "n_neg": n0})
				continue
			sel = base & (pos | neg) & ~(pos & neg)
			lab = pos[sel]
			for sname, sc in scores.items():
				sv = np.asarray(sc, float)[sel]
				# a score reported only where BOTH classes have 10+ values (AlphaMissense exists for the
				# cBioPortal rows only, so most of its negatives would be missing)
				if (np.isfinite(sv) & lab).sum() < 10 or (np.isfinite(sv) & ~lab).sum() < 10:
					continue
				a, lo, hi, n1, n0 = boot_auc(sv, lab, args.bootstrap, rng)
				rows.append({"restrict": restrict, "contrast": name, "score": sname, "auc": a, "lo": lo, "hi": hi,
					"n_pos": n1, "n_neg": n0})
	res = pd.DataFrame(rows)
	pd.set_option("display.width", 250)
	for restrict in ("snv", "all"):
		log(fh, f"\n== AUC (higher score = damaging), {'SNV-reachable variants only (the fair comparison)' if restrict == 'snv' else 'all variants (genetic code NOT controlled)'}")
		d = res[res.restrict == restrict].copy()
		d["cell"] = d.apply(lambda r: f"{r.auc:.3f} [{r.lo:.2f},{r.hi:.2f}]" if np.isfinite(r.auc) else "", axis=1)
		log(fh, d.pivot_table(index="contrast", columns="score", values="cell", aggfunc="first", sort=False).to_string())
		log(fh, "   n (pos/neg): " + "; ".join(f"{k}: {a}/{b}" for k, a, b in
			d[d.score == "U_am"][["contrast", "n_pos", "n_neg"]].itertuples(index=False)))

	# Where the database variants sit in the ranking of all 6156 mutants.
	log(fh, "\n== median percentile among all 6156 single mutants (100 = most expensive), SNV-reachable variants")
	rows = []
	for name, (group, mask) in cats.items():
		d = m[mask.values & inside & (m.snv == True).values]          # noqa: E712
		rows.append({"category": name, "n": len(d), **{f"{e} pct": d[f"{e}_pct"].median() for e in energies},
			**{f"{e} pct among SNV": d[f"{e}_pct_snv"].median() for e in energies}})
	pct = pd.DataFrame(rows)
	log(fh, pct.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
	log(fh, "\n   top 15 by U_am among tumour variants:")
	top = m[m.in_cbioportal & inside].sort_values("U_am", ascending=False).head(15)
	log(fh, top[["variant", "cbio_patients", "clinvar_class", "gnomad_ac", "U_am", "U_am_pct", "E2", "E2_pct",
		"U_ce", "U_ce_pct", "abundance_2021", "activity"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
	log(fh, "\n   the R130 hotspot and other recurrent tumour variants, energy percentiles:")
	rec = m[(m.cbio_patients >= 5) & inside].sort_values("cbio_patients", ascending=False)
	log(fh, rec[["variant", "cbio_patients", "U_am_pct", "E2_pct", "U_ce_pct", "U_am_rank_in_site",
		"abundance_2021", "activity"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

	# Within site: pairs of variants at the SAME site, one in each class. Burial and every other site
	# property cancel, so this asks whether the energy tells SUBSTITUTIONS apart (U_am's strength in the
	# VAMP-seq comparison). 0.5 = no substitution resolution.
	log(fh, "\n== within-site AUC: only pairs at the same site, one variant in each class (SNV-reachable)")
	obs = set(zip(m.position, m.mut_aa))
	sv = scans[scans.snv == True].copy()                 # noqa: E712
	sv["observed"] = [(p_, a_) in obs for p_, a_ in zip(sv.position, sv.mut_aa)]
	sv = sv.merge(m[["position", "mut_aa", "in_cbioportal", "in_gnomad", "clinvar_class"]], on=["position", "mut_aa"],
		how="left")
	pairs = [("tumour vs population-only", sv.in_cbioportal == True, (sv.in_gnomad == True) & (sv.in_cbioportal != True)),
		("ClinVar P/LP vs VUS", sv.clinvar_class == "P/LP", sv.clinvar_class == "VUS"),
		("tumour vs never observed", sv.in_cbioportal == True, ~sv.observed),
		("ClinVar P/LP vs never observed", sv.clinvar_class == "P/LP", ~sv.observed),
		("population-only vs never observed", (sv.in_gnomad == True) & (sv.in_cbioportal != True), ~sv.observed)]
	for name, pos, neg in pairs:
		cells = []
		for e in energies + ["abundance_2021", "activity"]:
			sc = -sv[e] if e in ("abundance_2021", "activity") else sv[e]
			a, npairs, nsites = within_site_auc(sv.site.values, sc.values, pos.fillna(False).values,
				neg.fillna(False).values)
			cells.append(f"{e if e in energies else '-' + e} {a:.3f}")
		log(fh, f"   {name:36s} " + "  ".join(cells) + f"   ({npairs} pairs at {nsites} sites, for -activity)")

	compound(m, scans, args, fh)
	fh.close()
	m.to_csv(os.path.join(args.out_dir, "pten_variants_ranked.tsv"), sep="\t", index=False, float_format="%.6g")
	res.to_csv(os.path.join(args.out_dir, "rank_auc.tsv"), sep="\t", index=False, float_format="%.4f")
	pct.to_csv(os.path.join(args.out_dir, "rank_percentiles.tsv"), sep="\t", index=False, float_format="%.2f")
	plot_rank(m, cats, energies, os.path.join(args.out_dir, "rank_by_category.png"))
	plot_auc(res, os.path.join(args.out_dir, "rank_auc.png"))
	print(f"\nOutputs in {args.out_dir}: rank.log, pten_variants_ranked.tsv, rank_auc.tsv, rank_percentiles.tsv, "
		"rank_by_category.png" + (", compound_additivity.png" if os.path.exists(os.path.join(args.out_dir, "compound_scores.csv")) else ""))


def compound(m, scans, args, fh):
	path = os.path.join(args.out_dir, "compound_scores.csv")
	if not os.path.exists(path):
		log(fh, f"\n== compound tumour sequences: {path} not found yet (run `score` on astaroth, then `rank` again)")
		return
	cs = pd.read_csv(path)
	single = scans.set_index(["site", "mut_aa"])
	rows = []
	for r in cs.itertuples(index=False):
		sites = [int(x) for x in str(r.sites).split("|") if x not in ("", "nan")]
		muts = [v[-1] for v in str(r.mutations_in_construct).split("|") if v not in ("", "nan")]
		row = {"sample": r.sample, "n": len(sites), "hypermutated": r.hypermutated}
		for e in ("U_am", "E2", "U_ce"):
			row[f"{e}_sum_singles"] = sum(single.loc[(s, a), e] for s, a in zip(sites, muts))
			row[f"{e}_max_single"] = max((single.loc[(s, a), e] for s, a in zip(sites, muts)), default=np.nan)
			row[e] = getattr(r, e)
		rows.append(row)
	d = pd.DataFrame(rows)
	d = d[d.n >= 2]
	log(fh, f"\n== compound tumour sequences: {len(d)} with 2+ mutations inside the construct")
	for e in ("U_am", "E2", "U_ce"):
		ratio = d[e] / d[f"{e}_sum_singles"]
		log(fh, f"   {e}: compound / sum of singles median {ratio.median():.3f} (IQR {ratio.quantile(.25):.3f}-"
			f"{ratio.quantile(.75):.3f}); Spearman(compound, sum) {spearman(d[e], d[f'{e}_sum_singles']):+.3f}; "
			f"compound above its worst single in {int((d[e] > d[f'{e}_max_single']).sum())}/{len(d)}")
	d.to_csv(os.path.join(args.out_dir, "compound_additivity.tsv"), sep="\t", index=False, float_format="%.6g")
	fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
	for ax, e in zip(axes, ("U_am", "E2", "U_ce")):
		ax.scatter(d[f"{e}_sum_singles"], d[e], s=12, c=np.where(d.hypermutated, "#c2571a", "#3b6fb6"))
		lim = [min(d[f"{e}_sum_singles"].min(), d[e].min()), max(d[f"{e}_sum_singles"].max(), d[e].max())]
		ax.plot(lim, lim, "k:", lw=.8)
		ax.set_xlabel(f"sum of single-mutant {e}")
		ax.set_ylabel(f"{e} of the compound sequence")
	axes[0].set_title("orange = hypermutated sample", fontsize=8)
	fig.tight_layout()
	fig.savefig(os.path.join(args.out_dir, "compound_additivity.png"), dpi=130)
	plt.close(fig)


def plot_auc(res, path):
	"""Forest plot: AUC with its 95% bootstrap interval, one panel per contrast, SNV-reachable variants."""
	order = ["U_am", "U_am site mean", "U_ce", "U_ce site mean", "E2", "E2 site mean", "-abundance_2021", "-activity"]
	colors = {"U_am": "#3b6fb6", "U_ce": "#2a7f62", "E2": "#c2571a", "-abundance_2021": "#7a5195", "-activity": "#444444"}
	d = res[(res.restrict == "snv") & res.score.isin(order)]
	contrasts = [c for c in d.contrast.unique() if not c.startswith("observed")]
	fig, axes = plt.subplots(2, (len(contrasts) + 1) // 2, figsize=(3.2 * ((len(contrasts) + 1) // 2), 6.4), sharey=True)
	for ax, c in zip(axes.flat, contrasts):
		e = d[d.contrast == c].set_index("score").reindex(order)
		y = np.arange(len(order))[::-1]
		for yi, (name, r) in zip(y, e.iterrows()):
			if not np.isfinite(r.auc):
				continue
			col = colors[name.replace(" site mean", "")]
			hollow = name.endswith("site mean")
			ax.errorbar(r.auc, yi, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o", ms=5, color=col,
				mfc="white" if hollow else col, capsize=2, lw=1)
		ax.axvline(.5, color="k", lw=.6, ls=":")
		ax.set_xlim(.3, 1.)
		n = e.dropna(subset=["auc"]).iloc[0]
		ax.set_title(f"{c}\n({int(n.n_pos)} vs {int(n.n_neg)})", fontsize=8)
		ax.set_yticks(np.arange(len(order))[::-1], order, fontsize=8)
		ax.set_xlabel("AUC", fontsize=8)
	for ax in list(axes.flat)[len(contrasts):]:
		ax.axis("off")
	fig.suptitle("AUC with 95% bootstrap interval, SNV-reachable variants; hollow = the energy's mean over the site's 19 mutants",
		fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)


def plot_rank(m, cats, energies, path):
	show = ["cbio_recurrent", "cbio_singleton", "clinvar_PLP", "clinvar_VUS", "gnomad_singleton", "gnomad_ac2",
		"consensus_damaging", "consensus_tolerated"]
	fig, axes = plt.subplots(1, len(energies), figsize=(4 * len(energies), 3.6), sharey=True)
	inside = m.U_am.notna().values & (m.snv == True).values           # noqa: E712
	for ax, e in zip(axes, energies):
		data = [m.loc[cats[k][1].values & inside, f"{e}_pct"].dropna().values for k in show]
		ax.boxplot(data, orientation="horizontal", showfliers=False)
		for i, v in enumerate(data, 1):
			ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-.18, .18, len(v)), s=3, alpha=.4,
				color="k")
		ax.set_yticks(range(1, len(show) + 1), [f"{k} ({len(v)})" for k, v in zip(show, data)], fontsize=8)
		ax.axvline(50, color="k", lw=.5, ls=":")
		ax.set_xlabel(f"{e} percentile among the 6156 single mutants")
	fig.suptitle("SNV-reachable variants inside the construct; 100 = the most expensive substitution", fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=130)
	plt.close(fig)


# ------------------------------------------------------------------- score --- #

def cmd_score(args):
	"""U_am, U_structure_ce and the 2-step contact energy E2 of every compound tumour sequence, with the
	same machinery as scan_single_mutants.py (_exact_energy, batch 1, lambda_structure_ce = 1 so one
	call gives both) and scan_mutants_E.py (generate at --num-steps, decode, CA map at 8 A, seed 0).
	Then --repeat single mutants are decoded again and compared with pten_mutants_E2.csv, the E2
	repeatability check of the plan (the decode was measured deterministic within a session on
	2026-11-16; this checks it across sessions)."""
	import time
	sys.path.insert(0, os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code")))
	sys.path.insert(0, os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../../customs")))
	sys.path.append(os.path.abspath(os.path.join(MONTE_CARLO_DIR, "../code/tests")))
	import torch
	from classes.rate_sampler import ExtendedProteinRateSampler
	from classes.ExtendedProtein import ExtendedProtein
	from utils.predictions import init_structure_config
	from custom_esm.sdk.api import ESMProteinTensor
	from compute_structural_metrics import CA_IDX
	from bench_decode import contact_map
	from references import get_reference
	from scan_single_mutants import PARS

	comp = pd.read_csv(os.path.join(args.out_dir, "compound_sequences.tsv"), sep="\t")
	comp = comp[comp.n_in_construct >= 2].reset_index(drop=True)
	_, ref = get_reference("pten_1d5r")
	t = site_table()
	if "".join(t.ref_aa) != ref:
		raise SystemExit("pten_site_table.csv does not match the pten_1d5r reference")
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	print(f"# {len(comp)} compound sequences, device {device}", flush=True)
	sampler = ExtendedProteinRateSampler(config_settings={})
	sampler.model.to(device)
	from generator.custom_generator import CustomGenerator
	sampler.generator = CustomGenerator(seed=0, device=device)
	sampler._canonical_idx = sampler._canonical_idx.to(device)
	sampler.ref_eprot = ExtendedProtein(sequence=ref, requires_grad=False, device=device)
	sampler.ref_eprot.expand()
	sampler.ref_eprot.am = sampler.model.predict_attention(sequence_probs=sampler.ref_eprot.get_probs())
	pars = dict(PARS, lambda_structure_ce=1.0)
	with torch.no_grad():
		ref_logits = sampler.model.predict_structure_logits(sequence_probs=sampler.ref_eprot.get_probs())
		sampler.ref_structure_tokens = ref_logits.argmax(dim=-1)[1:-1]

	def energies(seq):
		e = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		e.expand()
		_, u_am, e = sampler._exact_energy(e, pars)
		return float(u_am), float(e.U_structure_ce.item())

	u_am_ref, u_ce_ref = energies(ref)
	print(f"# reference: U_am {u_am_ref:.3e} (must be 0), U_ce {u_ce_ref:.6f} (the scan's was 2.397402)", flush=True)

	model, decoder = sampler.model, sampler.model.get_structure_decoder()
	cfg = init_structure_config(num_steps=args.num_steps)

	def ca_of(seq):
		torch.manual_seed(0)
		ep = ExtendedProtein(sequence=seq, requires_grad=False, device=device)
		ep.expand()
		with torch.no_grad():
			out = model.generate(input=ESMProteinTensor(sequence=ep.tokens), config=cfg)
			tok = out.structure if out.structure.dim() > 1 else out.structure.unsqueeze(0)
			dec = decoder.decode(tok)
		return dec["bb_pred"][0, 1:-1, CA_IDX].float().cpu().numpy()

	ref_map = contact_map(ca_of(ref), 8.0)
	n_ref = int(ref_map.sum())

	def E(seq):
		cm = contact_map(ca_of(seq), 8.0)
		return int((cm ^ ref_map).sum()) / (int(cm.sum()) + n_ref)

	t0 = time.perf_counter()
	rows = []
	for k, r in comp.iterrows():
		u_am, u_ce = energies(r.sequence)
		rows.append({**r.drop("sequence").to_dict(), "U_am": u_am, "U_ce": u_ce - u_ce_ref, "E2": E(r.sequence)})
		print(f"{r['sample']} {r.mutations_in_construct}: U_am {u_am:.3f} U_ce {u_ce - u_ce_ref:+.4f} "
			f"E{args.num_steps} {rows[-1]['E2']:.4f}", flush=True)
	pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "compound_scores.csv"), index=False, float_format="%.6f")
	print(f"# {len(rows)} compounds in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

	# E2 repeatability against the scan, on a fixed random subset of single mutants.
	e2 = pd.read_csv(os.path.join(NAT_DIR, "pten_mutants_E2.csv"))
	sub = e2.sample(args.repeat, random_state=0)
	d = []
	for r in sub.itertuples(index=False):
		seq = ref[:r.site] + r.mut_aa + ref[r.site + 1:]
		d.append({"site": r.site, "mut_aa": r.mut_aa, "E2_scan": r.E, "E2_now": E(seq)})
	d = pd.DataFrame(d)
	diff = (d.E2_now - d.E2_scan).abs()
	print(f"# E{args.num_steps} repeat on {len(d)} single mutants: identical {int((diff < 1e-6).sum())}/{len(d)}, "
		f"max |difference| {diff.max():.2e}, Spearman {spearman(d.E2_now, d.E2_scan):+.4f}", flush=True)
	d.to_csv(os.path.join(args.out_dir, "e2_repeat.csv"), index=False, float_format="%.6f")
	print(f"# done. next: python pten_variants.py rank", flush=True)


# -------------------------------------------------------------------- main --- #

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = ap.add_subparsers(dest="cmd", required=True)
	common = argparse.ArgumentParser(add_help=False)
	common.add_argument("--db-dir", default=DB_DIR, help="where the three pushed files are")
	common.add_argument("--out-dir", default=os.path.join(MONTE_CARLO_DIR, "pten_variants"))
	common.add_argument("--raw-dir", default=os.path.join(NAT_DIR, "raw"),
		help="pten_k_sites.py's raw dir; PTEN_variant_data.tsv there is used if present (snv column, predictors)")
	common.add_argument("--hypermutated", type=int, default=500,
		help="a tumour sample with more mutations than this counts as hypermutated (default 500)")
	common.add_argument("--runs-dir", default=os.path.join(MONTE_CARLO_DIR, "pten1d5r_study", "equil_ref"))
	common.add_argument("--temperatures", nargs="+", default=["T4", "T8"])
	common.add_argument("--burn-in-frac", type=float, default=0.8,
		help="share of each chain dropped before the entropy (0.8 clears t_eq at every T; see pten_k_sites)")
	common.add_argument("--sampled-k-frac", type=float, default=0.1,
		help="with --sampled-k-sigma none: sampler K-sites = this lowest share of the native sites by sampled "
			"entropy (default 0.1)")
	common.add_argument("--sampled-k-sigma", type=lambda s: None if s.lower() == "none" else float(s), default=1.5,
		help="sampler K-sites = native sites whose entropy is more than this many standard deviations below the "
			"median (default 1.5; 2 already empties T=4 and T=5). Empty when most sites are frozen (T <= 3). "
			"'none' switches to --sampled-k-frac, the rule of the first version of the report")
	common.add_argument("--sampled-k-scale", choices=["mad", "sd"], default="mad",
		help="the standard deviation used by --sampled-k-sigma: 1.4826 x MAD (default, robust) or the plain SD")
	common.add_argument("--min-category", type=int, default=20,
		help="categories with fewer variants inside the construct get counts only (default 20)")
	common.add_argument("--bootstrap", type=int, default=1000)
	for name, fn in (("clean", cmd_clean), ("merge", cmd_merge), ("profiles", cmd_profiles), ("rank", cmd_rank)):
		sub.add_parser(name, parents=[common]).set_defaults(func=fn)
	p = sub.add_parser("score", parents=[common])
	p.add_argument("--num-steps", type=int, default=2)
	p.add_argument("--repeat", type=int, default=50, help="single mutants re-decoded to check E2 (default 50)")
	p.add_argument("--device", default="cuda:0")
	p.set_defaults(func=cmd_score)
	sub.add_parser("all", parents=[common]).set_defaults(
		func=lambda a: [f(a) for f in (cmd_clean, cmd_merge, cmd_profiles, cmd_rank)])
	args = ap.parse_args()
	os.makedirs(args.out_dir, exist_ok=True)
	args.func(args)


if __name__ == "__main__":
	main()
