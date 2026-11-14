# PTEN K-sites: three references and the sampler

The long version, with the two papers summarised, every figure explained, the thresholds and
the Pfam step, is `pten_k_sites_explained.md` (PDF in the project files).

Written 2026-11-12. Script `monte_carlo/pten_k_sites.py`, outputs and logs in
`monte_carlo/pten_natural/`. Everything here ran in the cloud container; nothing needs the GPU.

**A K-site is a site the sampler keeps conserved in the cold regime** (Zambon et al 2024, not
lysine). The question is whether the sites PTEN cannot change in reality are the ones the sampler
keeps. Three independent references say which sites those are.

## The three references

| reference | what it measures | data | sites covered (of 324) |
| --- | --- | --- | --- |
| natural | variation among natural homologs of the 1D5R chain | UniRef90 jackhmmer alignment from OpenProteinSet, 7989 hits | 280 to 305 |
| abundance | does the variant still exist in the cell, about "does it fold" | VAMP-seq, Matreyek 2018 (PMID 29785012), 4112 missense; 5083 with the 2021 fill-in | 256 / 289 |
| activity | does the variant still dephosphorylate PIP3 | Mighell 2018 (PMC5986715), 6564 high-confidence missense (7244 with stops and deletions) | 311 |

The reference is UniProt 7-285 + 310-353 (the 286-309 loop is deleted in the crystal construct).
Site 0 (Met for Glu7) and site 279 (a junction Val) are not native and carry no score.

## Is each reference flat? Only the orthologs.

| profile | mean | median | reading |
| --- | --- | --- | --- |
| vertebrate PTEN (>= 90% identity, 92 seq) | 0.17 nats | 0.01 | flat: 176 of 280 sites below 0.05 |
| orthologs (>= 70%, 123 seq) | 0.31 | 0.19 | nearly flat; maximum possible is ln 20 = 3.0 |
| PTEN-like family (2716 seq, N_eff 506) | 1.66 | 1.76 | not flat: 0.56 (10th percentile) to 2.57 (90th) |
| abundance, mean score per site | 0.67 | 0.71 | not flat: 0.34 to 0.94 |
| activity, mean score per site | -1.01 | -0.48 | not flat: -2.8 to 0.0 |

So the expectation holds for the ortholog alignment and fails for the other two. The family
tier (PTEN, TPTE/VSP, tensins, auxilin/GAK, all with both domains) is the usable natural reference.

## The three K-like sets

- **natural** (family entropy < 0.5 nats): 28 sites, 23 of them in the phosphatase domain. They
  include the catalytic H123, C124, G127, G129, R130, G132, plus D24, P38, P96, the TI loop
  (P169, S170, Q171, R173, Y174, Y177) and D252 in the C2 domain.
- **abundance** (the authors' positional class "intolerant"): 108 sites, 61 phosphatase, 47 C2.
- **activity** (at least half of the scored missense variants below -1.11): 97 sites, 77 phosphatase.

Overlaps: natural with activity 24 where 8.4 are expected (P = 2e-10); abundance with activity 56
against 32 (P = 2e-9); **natural with abundance 12 against 9.3 (P = 0.18), no better than chance.**
Eleven sites are in all three: L23 D24 L25 Y27 V53 H61 C124 G129 S170 Y177 D252.

The non-overlap is the important biology: R15, H123, R130 are among the most conserved residues
in the family and have abundance 0.92-1.03, i.e. the protein folds perfectly well without them.
Natural conservation in this family is mostly about catalysis; abundance is about the fold. U_am
knows nothing about catalysis, so **abundance is the reference the sampler should be judged against**,
as `pten_target.md` argued before the data were in hand.

## The sampler against them: not flat

Per-site entropy of the reference-started chains of the 2026-11-12 ladder (30000 moves, last half),
rebuilt from `start.txt` and the accepted moves in `data.dat`.

Spearman of sampled entropy with each reference (0.12 is the 5% chance level):

| T | orthologs | family | abundance | activity |
| --- | --- | --- | --- | --- |
| 1 | +0.10 | +0.12 | +0.23 | +0.18 |
| 2 | +0.20 | +0.24 | +0.30 | +0.29 |
| 4 | +0.22 | +0.39 | +0.36 | +0.44 |
| 8 | +0.24 | +0.42 | +0.36 | +0.47 |
| 16 | +0.20 | +0.40 | +0.32 | +0.44 |
| 64 | +0.22 | +0.40 | +0.32 | +0.42 |

Mean sampled entropy inside each K-like set over the mean outside it: 0.20-0.25 at T=1, 0.17-0.42
at T=2, 0.34-0.49 at T=4, 0.54-0.69 at T=8, 0.75-0.83 at T=64. Every ratio from T=2 up is
significant against circular shifts of the set along the sequence (p <= 0.02, mostly at the 0.003
floor). The sampler's strict K-sites (entropy < 0.05 at T=1 and T=2) hold 23 of the 28 natural,
76 of the 108 abundance and 66 of the 97 activity sites, each above chance (P = 0.003, 0.0003, 0.004).

**Three cautions, in decreasing order of weight.**

1. *About half of the signal is burial.* Relative solvent accessibility in the crystal correlates
   +0.65 with abundance, +0.64 with activity, +0.53 with family entropy and +0.46 with the sampled
   entropy at T=4-8. With RSA regressed out, the sampler keeps +0.18 with abundance at T=4 and
   +0.24 with activity, down from +0.36 and +0.44. So the model has learnt "buried sites cost more",
   plus something beyond that, and the something is the interesting part.
2. *The chains are not equilibrated* (`t_eq` at no temperature). At T=1 there are 146 accepted
   moves in the kept half and 284 sites at entropy exactly 0, so "183 sampled K-sites" is mostly
   "not visited yet". The T=4 and T=8 rows are the trustworthy ones; the strict K-site list is not
   yet a result.
3. *A single chain's time entropy underestimates the ensemble entropy.* Longer chains raise every
   value, so the ratios, not the absolute numbers, are what carries over.

A by-product worth keeping: `-mean_dU` of the proposals at a site (the sampler's own cost of
mutating it, available from `data.dat` with no equilibration at all) correlates with abundance
+0.30 at T=1 and +0.46 at T=64. That points at the cleaner experiment below.

## Update 2026-11-13: the 220000-move chains

Main now holds T=1, 2, 4, 8 pushed to 220000 moves on the reference side (and T=3, 5, 6 at
30000). `compare` was re-run on them; `pten_natural/compare.log`, the tables and the two
sampled figures are from this run, the numbers in the section above are the 30000-move ones.

| T | moves | sites at entropy 0 | family | abundance | activity | abundance, RSA out | activity, RSA out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 220000 | 220 | +0.15 | +0.19 | +0.19 | +0.11 | +0.07 |
| 2 | 220000 | 100 | +0.37 | +0.31 | +0.38 | +0.09 | +0.19 |
| 4 | 220000 | 25 | +0.52 | +0.41 | +0.53 | +0.15 | +0.32 |
| 8 | 220000 | 9 | +0.61 | +0.46 | +0.61 | +0.19 | +0.41 |

Every correlation at T >= 2 went up with seven times the moves (T=8: abundance +0.36 to +0.46,
activity +0.47 to +0.61, family +0.42 to +0.61), so the 30000-move values were held down by
sampling noise, not inflated by it. The strict K-site list (entropy < 0.05 at T=1 and T=2) fell
from 183 to 104 sites and still overlaps all three sets above chance (16/28 natural, 43/108
abundance, 46/97 activity). Two things did not improve: T=1 is still trapped (198 sites never
accepted a move), and the part of the abundance correlation that survives the burial control
stays at +0.15 to +0.19, while for activity and family entropy it grows to +0.41 and +0.45.
Read plainly: beyond burial, the sampler tracks what the family conserves more than what keeps
the protein abundant. That is worth putting to the supervisors, and it is one more reason for
the single-mutant scan of U_am.

## The PTP superfamily (Pfam PF00102): feasible, but PF00102 is the wrong family

PTEN is not a member of PF00102. PF00102 is the classical PTP domain (PTP1B, SHP2, CD45, about
240 residues); PTEN's catalytic domain belongs to the same superfamily (the CX5R phosphatases)
through the dual-specificity branch. Measured with the HHsearch alignments of the PTEN profile
against PDB70 (195 templates at probability >= 90):

- 71 classical PTPs align to PTEN over a median of 91 columns, UniProt 89-179: the WPD loop, the
  P loop and the helices after it. **They reach 53% of the phosphatase domain** and nothing
  before residue ~85.
- 116 other phosphatases (dual-specificity, PRL, CDC14, KAP, RNA 5'-phosphatases) align over a
  median of 138 columns, UniProt 24-180, 78% of the domain.
- No distant phosphatase aligns to the C2 domain, as expected.

First pass on the conservation itself, carrying the classical-PTP entropy (PTP1B's UniRef90
alignment, 9890 sequences, N_eff 174) onto the 90 PTEN sites it reaches: Spearman +0.48 with
PTEN's own family entropy and +0.49 with activity, but **-0.06 with abundance**. The only sites
conserved on both sides are P96, H123 and C124. So the superfamily alignment tells us where the
chemistry is, which the PTEN-like family already tells us over the whole chain, and it says
nothing about the fold of PTEN in particular.

Verdict: do not build on PF00102. If a deeper natural reference is wanted, the 6002-sequence
phosphatase tier already in hand (N_eff 1125, columns 14-185, `nat_pdomain_entropy`) is the
superfamily alignment anchored on PTEN, and it needed no new alignment. A proper Pfam version
would use PTEN's own entries (DSPc-like for the phosphatase domain, PF10409 for C2) and needs
InterPro, which the cloud container cannot reach.

## What to do next

1. **An in-silico mutational scan of U_am**: U(mutant) for all 324 x 19 single mutants of the
   reference, about 6200 forward passes, minutes on the GPU. It is the direct counterpart of
   VAMP-seq, needs no equilibration, and separates "the energy is right" from "the chain has not
   moved yet". Not written yet.
2. Done 2026-11-13, see the update above.
3. Keep abundance as the primary reference and report every correlation both raw and with RSA
   regressed out.

## Reproduce

    cd softmax/monte_carlo
    python pten_k_sites.py fetch       # github raw + the OpenFold S3 bucket, 20 MB
    python pten_k_sites.py build
    R=pten1d5r_study/equil
    python pten_k_sites.py compare --runs T1=$R/T1/chain0 T2=$R/T2/chain0 T3=$R/T3/chain0 \
        T4=$R/T4/chain0 T5=$R/T5/chain0 T6=$R/T6/chain0 T8=$R/T8/chain0 T16=$R/T16/chain0 \
        T32=$R/T32/chain0 T64=$R/T64/chain0 \
        --cold T1 T2 --show T1 T2 T4 T8
    python pten_k_sites.py family

Figures (png is git-ignored under monte_carlo, so copies are kept in `docs/figures/`):
`pten_k_pten_site_profile_sampled.png`, `pten_k_sampled_vs_references.png`, `pten_k_pten_ptp_superfamily.png`.
