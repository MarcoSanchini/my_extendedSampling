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

### The one real judgement call: the weighting threshold

Every sequence gets the weight `w = 1 / m`, where `m` counts the sequences at
identity `--weight-identity` or above to it, itself included, and the effective
number of sequences is `N_eff = sum(w)`. This is the standard DCA correction
(Morcos et al. 2011, PNAS 108:E1293: 80% threshold, weight `1/m`, `N_eff` the
sum of the weights). A sequence with no close neighbours counts fully; each
member of a cluster of 100 near-identical sequences counts 1/100, so the
cluster counts about 1 in total.

Now apply that at the default 0.8 to a set spanning subtypes. Within a subtype,
sequences sit far above 80% identity to each other (human H3 is at 0.966 median
identity to the reference), so they are all mutual neighbours. Between
subtypes they are far below it (H1 against our H3 reference is about 0.44). So
the neighbour sets essentially coincide with the subtypes: `m` is about the size
of the sequence's own subtype, each subtype's weights sum to about 1, and

    N_eff ~ the number of subtypes

which is why the two-subtype test printed `N_eff = 2.0`. That looks broken and
is not: it is exactly what the formula is for.

The consequence is the part that matters. With these weights the entropy at a
site is almost entirely **between**-subtype variation: it asks "do the
different subtypes agree at this position?" and is nearly blind to variation
inside a subtype, including all of human H3N2's antigenic drift. That is a
defensible, arguably the right, definition of "what the fold requires", since
the within-subtype signal is the part most contaminated by host-specific immune
pressure and by how much each virus happened to be sequenced. But it is a
choice, not a fact, so show that it does not decide the answer:

- `--weight-identity 0.8` (default): one effective sequence per subtype, pure
  between-subtype conservation.
- `--weight-identity 0.95`: within-subtype drift kept, since only nearly
  identical sequences count as neighbours, and each subtype contributes in
  proportion to how much of it was sequenced. A middle course, and the closest
  to what the original H3N2-only file measured.
- `--weight-identity 0`: raw counts, so the entropy follows sequencing effort,
  in practice H1, H3 and H5. The sampling-biased extreme, useful as a control.

Build the file at 0.8 and at 0.95, run the K-site comparison on both, and
report that the conclusion is the same. If it is not the same, that is a finding
about which kind of conservation the sampler reproduces, and more interesting
than either number on its own.

`softmax/docs/ha_natural_methods.tex` works through this, and the rest of the
method, with references.

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
