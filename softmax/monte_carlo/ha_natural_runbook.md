# Natural HA entropy beyond human H3N2

The sampled per-site entropy of the zero_polymer chains is compared with a
natural per-site entropy by `structure_zero_polymer.py plot --h3n2 <file>`. So
far that file was `FASTA/H3N2/site_entropy_566.csv`: 19 380 human H3N2
hemagglutinins, all already 566 residues long, so no alignment was needed and
no redundancy correction was applied.

`ha_site_entropy.py` builds the same kind of file from any set of influenza A
HA subtypes. The flag keeps its name, so nothing in
`structure_zero_polymer.sh` changes:

    H3N2=ha_natural/ha_group2_entropy_566.csv SKIP_DECODE=1 bash structure_zero_polymer.sh

## Why more than H3N2

Human H3N2 is one antigenically drifting lineage, so its entropy mixes real
structural constraint with the particular history of one virus in one host.
Across the subtypes of a whole HA phylogenetic group, a site that stays
conserved is conserved because the fold needs it. Two files are worth having:

- **group 2 only** (H3, H4, H7, H10, H14, H15) — the group zero_polymer
  belongs to, so the alignment to its 566 positions is reliable everywhere,
  H4 and H14 being the closest to H3;
- **all influenza A subtypes** (group 2 plus H1, H2, H5, H6, H8, H9, H11, H12,
  H13, H16) — the widest view, at the price of a looser alignment in the HA1
  head.

Influenza B HA is left out: about 25% identity to influenza A is too far for a
position-by-position mapping.

## Step 1, fetch (needs internet; blocked in the cloud container)

NCBI, BV-BRC, UniProt and EBI are all refused by this session's egress proxy,
so the fetch runs on the GPU machine or any machine with plain internet:

    cd softmax/monte_carlo
    nohup python -u ha_site_entropy.py fetch --subtypes all --out-dir ha_natural/fasta \
        > ha_fetch.log 2>&1 &

It queries NCBI Protein per subtype (`"H3N1 subtype"[Organism] OR ... AND
hemagglutinin[Protein Name] AND 540:600[SLEN]`) and writes one FASTA per
subtype. An `NCBI_API_KEY` in the environment raises the rate limit from 3 to
10 requests per second. `--host avian` (or `swine`, `equine`, `canine`) adds a
host term, for the non-human H3 set.

Any other source works the same way, since `build` only reads FASTA: BV-BRC
(the former FluDB/IRD) has a bulk download per subtype, and GISAID needs an
account. Pfam PF00509 is a shortcut, but check that it covers HA2 before
using it.

## Step 2, build

    # group 2 only
    python ha_site_entropy.py build \
        --fasta ha_natural/fasta/H3.fasta ha_natural/fasta/H4.fasta ha_natural/fasta/H7.fasta \
                ha_natural/fasta/H10.fasta ha_natural/fasta/H14.fasta ha_natural/fasta/H15.fasta \
        --out ha_natural/ha_group2_entropy_566.csv

    # all influenza A subtypes
    python ha_site_entropy.py build --fasta ha_natural/fasta/H*.fasta \
        --out ha_natural/ha_allA_entropy_566.csv

What it does, in order:

1. **Filter.** Length 540 to 600, at most 1% non-standard letters. Exact
   duplicates are collapsed, keeping their multiplicity. Above
   `--max-unique` (3000) unique sequences per subtype it subsamples at
   random, which keeps MAFFT and the weighting tractable.
2. **Align.** One MAFFT run over all the kept sequences plus the zero_polymer
   reference, then only the columns where the reference has a residue are
   kept, so insertions relative to zero_polymer are dropped and the output has
   exactly 566 rows. `--align add` uses `--add --keeplength` instead, which
   aligns each sequence to the reference alone: faster, and fine within one
   subtype, but worse across groups.
3. **Reweight.** Each sequence counts 1 / (number of sequences at identity
   at or above `--weight-identity`, 0.8 by default, itself included), the
   usual DCA correction. The effective number of sequences N_eff is printed
   overall and per subtype. `--weight-identity 0` turns it off.
4. **Entropy.** Per site, the weighted gap fraction, and the Shannon entropy
   in bits over the residues only. A site with more than `--max-gap` (30%)
   gaps gets entropy NaN, not a 21st letter.

Output: the CSV with the columns `position, entropy_bits, n_distinct,
most_common_aa, most_common_frequency` that `site_entropy_566.csv` already
had, plus `ref_aa, gap_fraction, n_eff`; a `.summary.tsv` with the per-subtype
counts, N_eff and median identity to the reference; and a `.png` of the
profile with the signal peptide, HA1 and HA2 spans marked.

### One caveat on the weighting

At 0.8 identity each subtype collapses to roughly one cluster, so N_eff comes
out near the number of subtypes and the entropy is then almost entirely
between-subtype variation. That is a defensible definition of "what the fold
requires", but it is a choice, so build a second file at
`--weight-identity 0.95` (within-subtype drift kept, each subtype weighted by
how much it was sequenced) or at `0` (raw counts, i.e. sampling-biased towards
H1, H3 and H5) and check that the K-site comparison does not depend on it.

## Checks already done in the cloud container

- Rebuilding the existing H3N2 file with this script (`--align add
  --weight-identity 0`, no subsampling) reproduces it: largest per-site
  difference 0.053 bits, rank correlation 0.989 over the 566 sites. The small
  differences come from MAFFT re-aligning the 8 sequences whose length is not
  566 and from the 1%-ambiguity filter.
- A two-subtype run (H3 plus the 410 H1 hemagglutinins already in
  `FASTA/sequences_H1N1_homo.fasta`) maps cleanly: median identity to the
  reference 0.966 for H3 and 0.435 for H1, 12 of 566 sites NaN for gaps (two
  short insertion/deletion regions in the signal peptide and the HA1 head),
  and median entropy already splits the way it should, 1.00 bits in HA1
  against 0.084 bits in HA2.

So the only missing piece is the sequence download.

## What the result should look like

If the sampler is finding real constraint, HA2 (the fusion peptide at its
start, the long helices, the stalk) should be conserved in the natural
alignment and the HA1 head antigenic sites variable. The T12 K-sites
concentrating in HA2 and the stalk is evidence on its own, before any
correlation number.
