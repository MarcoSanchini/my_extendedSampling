"""
Natural per-site entropy of influenza A hemagglutinin beyond human H3N2,
mapped onto the 566 sites of the zero_polymer reference, so that
structure_zero_polymer.py (and structure_zero_polymer.sh with H3N2=<file>)
can compare it with the sampled entropy exactly as it does with
FASTA/H3N2/site_entropy_566.csv.

  fetch  downloads HA protein sequences per subtype from NCBI Protein
         (E-utilities), one FASTA per subtype. Needs internet access to
         eutils.ncbi.nlm.nih.gov. Any other source works as well (BV-BRC,
         GISAID, the NCBI Influenza Virus Resource): build only needs FASTA.
  build  from SUBTYPE=FASTA pairs:
           1. filters by length and ambiguous letters, removes exact
              duplicates (keeping their multiplicity) and, above
              --max-unique per subtype, subsamples the unique sequences;
           2. aligns them together with the reference with MAFFT and keeps
              only the columns where the reference has a residue (insertions
              relative to zero_polymer are dropped);
           3. reweights for redundancy, DCA style: every sequence counts
              1 / (number of sequences at identity >= --weight-identity with
              it, itself included); the sum of the weights is the effective
              number of sequences N_eff, printed overall and per subtype;
           4. per site: weighted gap fraction and, over the residues only,
              Shannon entropy in bits. A site with gap fraction above
              --max-gap gets entropy NaN (it is not a 21st letter).
         Output columns: position (1-based), entropy_bits, n_distinct,
         most_common_aa, most_common_frequency (as site_entropy_566.csv),
         then ref_aa, gap_fraction, n_eff. A <out>.summary.tsv lists the
         per-subtype counts and a <out>.png the profile, with the signal
         peptide / HA1 / HA2 boundaries of zero_polymer.

Subtype groups (phylogenetic groups of influenza A HA):
  group 2: H3 H4 H7 H10 H14 H15        (H3 is zero_polymer's own subtype)
  group 1: H1 H2 H5 H6 H8 H9 H11 H12 H13 H16 (H17 H18: bat, optional)
Influenza B HA (about 25% identity) is left out on purpose.

Usage:
    # on a machine with internet access
    python ha_site_entropy.py fetch --subtypes group2 --out-dir ha_natural/fasta
    python ha_site_entropy.py fetch --subtypes group1 --out-dir ha_natural/fasta
    # two entropy files: group 2 only, and all subtypes
    python ha_site_entropy.py build --fasta ha_natural/fasta/H{3,4,7,10,14,15}.fasta \\
        --out ha_natural/ha_group2_entropy_566.csv
    python ha_site_entropy.py build --fasta ha_natural/fasta/H*.fasta \\
        --out ha_natural/ha_allA_entropy_566.csv
    # then, as for H3N2
    H3N2=ha_natural/ha_group2_entropy_566.csv SKIP_DECODE=1 bash structure_zero_polymer.sh

A --fasta item is SUBTYPE=PATH, or a PATH whose file name starts with the
subtype (H5.fasta, H5_avian.fa). Several files may share a subtype.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

MONTE_CARLO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(MONTE_CARLO_DIR, "../code/tests"))
from references import get_reference

GROUPS = {
	"group2": ["H3", "H4", "H7", "H10", "H14", "H15"],
	"group1": ["H1", "H2", "H5", "H6", "H8", "H9", "H11", "H12", "H13", "H16"],
	"bat": ["H17", "H18"],
}
GROUPS["all"] = GROUPS["group2"] + GROUPS["group1"]
AA = "ACDEFGHIKLMNPQRSTVWY"
GAP = len(AA)                      # every other letter (gap, X, B, Z, J, U, O) is index 20
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# ------------------------------------------------------------------- fasta --- #
def read_fasta(path):
	records, name, chunks = [], None, []
	with open(path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			if line.startswith(">"):
				if name is not None:
					records.append((name, "".join(chunks)))
				name, chunks = line[1:], []
			else:
				chunks.append(line.upper())
	if name is not None:
		records.append((name, "".join(chunks)))
	return records


def write_fasta(path, records):
	with open(path, "w") as f:
		for name, seq in records:
			f.write(f">{name}\n")
			for i in range(0, len(seq), 80):
				f.write(seq[i:i + 80] + "\n")


# ------------------------------------------------------------------- fetch --- #
def ncbi_term(subtype, min_len, max_len, host):
	# The NCBI taxonomy has one node per HxNy subtype ("H5N1 subtype"); N1-N9
	# cover every combination that occurs (unknown N stays out).
	org = " OR ".join(f'"{subtype}N{n} subtype"[Organism]' for n in range(1, 10))
	term = f"({org}) AND hemagglutinin[Protein Name] AND {min_len}:{max_len}[SLEN]"
	if host:
		term += f" AND {host}[All Fields]"
	return term


def eutils(endpoint, params, retries=4):
	data = urllib.parse.urlencode(params).encode()
	for attempt in range(retries):
		try:
			with urllib.request.urlopen(EUTILS + endpoint, data=data, timeout=120) as r:
				return r.read().decode()
		except Exception as e:   # network hiccups and NCBI rate limits
			if attempt == retries - 1:
				raise
			wait = 2 ** (attempt + 1)
			print(f"    {endpoint}: {e}; retrying in {wait}s")
			time.sleep(wait)


def cmd_fetch(args):
	subtypes = []
	for s in args.subtypes:
		subtypes += GROUPS.get(s, [s])
	os.makedirs(args.out_dir, exist_ok=True)
	extra = {"tool": "ha_site_entropy"}
	if args.api_key:
		extra["api_key"] = args.api_key
	for st in subtypes:
		out = os.path.join(args.out_dir, f"{st}.fasta")
		if os.path.isfile(out) and not args.overwrite:
			print(f"{st}: {out} exists, skipped (--overwrite to refetch)")
			continue
		term = ncbi_term(st, args.min_len, args.max_len, args.host)
		res = eutils("esearch.fcgi", {"db": "protein", "term": term, "usehistory": "y", "retmax": 0, **extra})
		count = int(re.search(r"<Count>(\d+)</Count>", res).group(1))
		webenv = re.search(r"<WebEnv>(\S+)</WebEnv>", res).group(1)
		key = re.search(r"<QueryKey>(\d+)</QueryKey>", res).group(1)
		n = min(count, args.max_records)
		print(f"{st}: {count} records" + (f", fetching the first {n}" if n < count else ""))
		with open(out + ".part", "w") as f:
			for start in range(0, n, args.batch):
				f.write(eutils("efetch.fcgi", {"db": "protein", "query_key": key, "WebEnv": webenv,
					"rettype": "fasta", "retmode": "text", "retstart": start,
					"retmax": min(args.batch, n - start), **extra}))
				time.sleep(0.12 if args.api_key else 0.4)   # NCBI: 10 / 3 requests per second
		os.replace(out + ".part", out)
		print(f"    -> {out} ({sum(1 for _ in read_fasta(out))} sequences)")


# ------------------------------------------------------------------- build --- #
def parse_fasta_args(items):
	out = OrderedDict()
	for item in items:
		if "=" in item:
			st, path = item.split("=", 1)
		else:
			path = item
			m = re.match(r"(H\d+)", os.path.basename(path), re.I)
			if not m:
				raise SystemExit(f"{item}: give SUBTYPE=PATH, the subtype is not in the file name")
			st = m.group(1).upper()
		out.setdefault(st, []).append(path)
	return out


def load_subtype(paths, args, rng):
	raw = []
	for p in paths:
		raw += [s.replace("*", "").replace("-", "") for _, s in read_fasta(p)]
	n_raw = len(raw)
	ok = [s for s in raw if args.min_len <= len(s) <= args.max_len
		and sum(c not in AA for c in s) <= args.max_ambiguous * len(s)]
	counts = Counter(ok)
	uniq = list(counts)
	n_uniq = len(uniq)
	if n_uniq > args.max_unique:
		keep = rng.choice(n_uniq, args.max_unique, replace=False)
		uniq = [uniq[i] for i in sorted(keep)]
	return uniq, [counts[s] for s in uniq], {"raw": n_raw, "passed_filters": len(ok), "unique": n_uniq, "kept_unique": len(uniq)}


def run_mafft(ref_seq, seqs, args):
	# Reference first; returns the aligned sequences restricted to the
	# reference's residue columns (len(ref_seq) letters each, '-' for gaps).
	mafft = shutil.which(args.mafft)
	if mafft is None:
		raise SystemExit(f"{args.mafft} not found (conda install -c bioconda mafft, or apt install mafft)")
	with tempfile.TemporaryDirectory() as tmp:
		ref_path, in_path = os.path.join(tmp, "ref.fa"), os.path.join(tmp, "in.fa")
		write_fasta(ref_path, [("ref", ref_seq)])
		write_fasta(in_path, [(f"s{i}", s) for i, s in enumerate(seqs)])
		if args.align == "add":
			# Each sequence against the reference only, insertions deleted by MAFFT.
			cmd = [mafft, "--thread", str(args.threads), "--quiet", "--anysymbol",
				"--add", in_path, "--keeplength", ref_path]
		else:
			# Full multiple alignment (better for distant subtypes), projected below.
			write_fasta(in_path, [("ref", ref_seq)] + [(f"s{i}", s) for i, s in enumerate(seqs)])
			cmd = [mafft, "--auto", "--thread", str(args.threads), "--quiet", "--anysymbol", in_path]
		print("  " + " ".join(cmd[:6]) + " ...")
		t0 = time.time()
		res = subprocess.run(cmd, capture_output=True, text=True)
		if res.returncode != 0:
			raise SystemExit(f"mafft failed:\n{res.stderr[-2000:]}")
		aln_path = os.path.join(tmp, "aln.fa")
		with open(aln_path, "w") as f:
			f.write(res.stdout)
		aln = dict(read_fasta(aln_path))
		print(f"  aligned {len(seqs)} sequences in {time.time() - t0:.0f}s")
	ref_aln = aln.pop("ref")
	cols = np.array([i for i, c in enumerate(ref_aln) if c != "-"])
	if len(cols) != len(ref_seq) or "".join(ref_aln[i] for i in cols) != ref_seq:
		raise SystemExit("the reference changed in the alignment")
	return ["".join(aln[f"s{i}"][j] for j in cols) for i in range(len(seqs))]


def encode(seqs, L):
	lut = np.full(256, GAP, dtype=np.int8)
	for k, a in enumerate(AA):
		lut[ord(a)] = k
	X = np.empty((len(seqs), L), dtype=np.int8)
	for i, s in enumerate(seqs):
		X[i] = lut[np.frombuffer(s.encode(), dtype=np.uint8)]
	return X


def neighbour_counts(X, mult, theta, chunk=1024):
	# For each row u: sum of mult[v] over rows v with identity(u, v) >= theta,
	# identity = fraction of the L sites with the same symbol (gaps included,
	# as in DCA). One-hot matrix products, chunked over rows.
	n, L = X.shape
	onehot = np.zeros((n, L * (GAP + 1)), dtype=np.float32)
	onehot[np.repeat(np.arange(n), L), (np.arange(L) * (GAP + 1))[None, :].repeat(n, 0).ravel() + X.ravel()] = 1.
	need = theta * L - 1e-6
	mult = np.asarray(mult, dtype=np.float64)
	out = np.empty(n)
	for a in range(0, n, chunk):
		same = onehot[a:a + chunk] @ onehot.T
		out[a:a + chunk] = (same >= need) @ mult
	return out


def site_table(X, w, ref_seq, max_gap):
	n, L = X.shape
	counts = np.zeros((L, GAP + 1))
	for k in range(GAP + 1):
		counts[:, k] = ((X == k) * w[:, None]).sum(axis=0)
	total = counts.sum(axis=1)
	gap = counts[:, GAP] / total
	res = counts[:, :GAP]
	with np.errstate(divide="ignore", invalid="ignore"):
		p = res / res.sum(axis=1, keepdims=True)
		H = -np.nansum(np.where(p > 0, p * np.log2(p), 0.), axis=1)
	H[(gap > max_gap) | (res.sum(axis=1) == 0)] = np.nan
	best = res.argmax(axis=1)
	return pd.DataFrame({
		"position": np.arange(1, L + 1),
		"entropy_bits": H,
		"n_distinct": (res > 0).sum(axis=1),
		"most_common_aa": [AA[b] for b in best],
		"most_common_frequency": p[np.arange(L), best],
		"ref_aa": list(ref_seq),
		"gap_fraction": gap,
		"n_eff": total,
	})


def ha_regions(ref_seq):
	# zero_polymer: signal peptide 1-16, HA1 17 up to the cleavage site, HA2
	# from the fusion peptide (GLFGAIAGF / GIFGAIAGF) to the end, which
	# includes the transmembrane and cytoplasmic tail. 0-based, half-open.
	m = re.search(r"G[IL]FGAIAGF", ref_seq)
	ha2 = m.start() if m else None
	return {"signal": (0, 16), "HA1": (16, ha2 or len(ref_seq)), "HA2": (ha2, len(ref_seq))} if ha2 else {"signal": (0, 16)}


def plot_profile(table, ref_seq, title, path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	reg = ha_regions(ref_seq)
	fig, axes = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
	x = np.arange(len(ref_seq))
	axes[0].bar(x, table["entropy_bits"].values, width=1., color="0.3")
	axes[0].set_ylabel("entropy [bits]")
	axes[1].bar(x, table["gap_fraction"].values, width=1., color="tab:orange")
	axes[1].set_ylabel("gap fraction"); axes[1].set_xlabel("site (0-based)")
	for ax in axes:
		for name, (a, b) in reg.items():
			ax.axvspan(a, b, color={"signal": "tab:gray", "HA1": "tab:blue", "HA2": "tab:green"}[name], alpha=.07)
	for name, (a, b) in reg.items():
		axes[0].text((a + b) / 2, axes[0].get_ylim()[1] * .92, name, ha="center", fontsize=8)
	axes[0].set_title(title, fontsize=9)
	fig.tight_layout()
	fig.savefig(path, dpi=130)
	plt.close(fig)


def cmd_build(args):
	_, ref_seq = get_reference(args.ref)
	L = len(ref_seq)
	rng = np.random.default_rng(args.seed)
	files = parse_fasta_args(args.fasta)

	seqs, mult, labels, summary = [], [], [], []
	for st, paths in files.items():
		u, m, info = load_subtype(paths, args, rng)
		print(f"{st}: {info['raw']} sequences, {info['passed_filters']} pass the filters, "
			f"{info['unique']} unique" + (f", {info['kept_unique']} kept" if info["kept_unique"] < info["unique"] else ""))
		seqs += u; mult += m; labels += [st] * len(u)
		summary.append({"subtype": st, **info, "copies_kept": int(sum(m))})
	if not seqs:
		raise SystemExit("no sequences left")

	print(f"aligning {len(seqs)} unique sequences to {args.ref} ({L} sites), mode {args.align}")
	X = encode(run_mafft(ref_seq, seqs, args), L)
	mult = np.asarray(mult, dtype=float)
	labels = np.array(labels)

	# Identity to the reference, a quick check of the mapping.
	id_ref = (X == encode([ref_seq], L)[0]).mean(axis=1)

	if args.weight_identity > 0:
		t0 = time.time()
		nb = neighbour_counts(X, mult, args.weight_identity)
		w = mult / nb     # all copies of a unique sequence share its neighbourhood
		print(f"  weights at identity >= {args.weight_identity} in {time.time() - t0:.0f}s")
	else:
		w = mult.copy()
	for row in summary:
		sel = labels == row["subtype"]
		row["n_eff"] = round(float(w[sel].sum()), 1)
		row["median_identity_to_ref"] = round(float(np.median(id_ref[sel])), 3) if sel.any() else np.nan
	summary = pd.DataFrame(summary)

	table = site_table(X, w, ref_seq, args.max_gap)
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	table.to_csv(args.out, index=False)
	stem = os.path.splitext(args.out)[0]
	summary.to_csv(stem + ".summary.tsv", sep="\t", index=False)

	n_eff = w.sum()
	print()
	with pd.option_context("display.width", 200):
		print(summary.to_string(index=False))
	print(f"\ntotal: {int(mult.sum())} sequences, {len(seqs)} unique, N_eff = {n_eff:.1f} "
		f"(weights 1/neighbours at identity >= {args.weight_identity})")
	if args.weight_identity >= 0.8 and len(files) > 1:
		print("  N_eff near the number of subtypes means each subtype collapses to about one\n"
			"  cluster at this identity: the entropy is then mostly between-subtype variation.\n"
			"  Build a second file at --weight-identity 0.95 (or 0) to see the difference.")
	nan = table["entropy_bits"].isna()
	print(f"sites with gap fraction > {args.max_gap}: {nan.sum()}/{L} (entropy NaN)"
		+ (f": {', '.join(str(i) for i in np.where(nan)[0])} (0-based)" if 0 < nan.sum() <= 40 else ""))
	for name, (a, b) in ha_regions(ref_seq).items():
		h = table["entropy_bits"].values[a:b]
		print(f"  {name:6s} sites {a}-{b - 1}: median entropy {np.nanmedian(h):.3f} bits, "
			f"{np.nanmean(h < args.conserved):.0%} below {args.conserved}")
	title = (f"{os.path.basename(args.out)}: {', '.join(files)}; {len(seqs)} unique, N_eff {n_eff:.0f}, "
		f"weights at {args.weight_identity}, NaN above {args.max_gap:.0%} gaps")
	plot_profile(table, ref_seq, title, stem + ".png")
	print(f"\nwrote {args.out}, {stem}.summary.tsv, {stem}.png")


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = p.add_subparsers(dest="cmd", required=True)
	f = sub.add_parser("fetch", help="download HA sequences per subtype from NCBI Protein")
	f.add_argument("--subtypes", nargs="+", default=["all"], help="H3 H5 ... or group1, group2, all, bat")
	f.add_argument("--out-dir", default="ha_natural/fasta")
	f.add_argument("--min-len", type=int, default=540)
	f.add_argument("--max-len", type=int, default=600)
	f.add_argument("--host", default=None, help="optional extra term, e.g. avian or swine")
	f.add_argument("--max-records", type=int, default=50000, help="per subtype")
	f.add_argument("--batch", type=int, default=500)
	f.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
	f.add_argument("--overwrite", action="store_true")
	b = sub.add_parser("build", help="align, reweight and write the per-site entropy (566 rows)")
	b.add_argument("--fasta", nargs="+", required=True, help="SUBTYPE=PATH, or PATH starting with the subtype")
	b.add_argument("--out", required=True)
	b.add_argument("--ref", default="zero_polymer")
	b.add_argument("--min-len", type=int, default=540)
	b.add_argument("--max-len", type=int, default=600)
	b.add_argument("--max-ambiguous", type=float, default=0.01, help="max fraction of non-standard letters")
	b.add_argument("--max-unique", type=int, default=3000, help="unique sequences kept per subtype (random)")
	b.add_argument("--weight-identity", type=float, default=0.8, help="0 for no reweighting")
	b.add_argument("--max-gap", type=float, default=0.3)
	b.add_argument("--align", choices=["msa", "add"], default="msa",
		help="msa: full MAFFT alignment projected on the reference; add: --add --keeplength")
	b.add_argument("--mafft", default="mafft")
	b.add_argument("--threads", type=int, default=4)
	b.add_argument("--conserved", type=float, default=0.1, help="entropy (bits) for the per-region summary only")
	b.add_argument("--seed", type=int, default=0)
	args = p.parse_args()
	{"fetch": cmd_fetch, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
	main()
