# PTEN missense variants in cBioPortal, ClinVar and gnomAD

Reading of `monte_carlo/pten_variants.py` (2026-11-18). All numbers come from the logs in
`monte_carlo/pten_variants/` and can be regenerated with `python pten_variants.py all`
(CPU, about 40 s). The GPU step (`score`, compound tumour sequences) has not run yet;
section 6 says what it will add.

Data: the three files in `databases/` (commit cbce85c). No database or data host was
contacted. Numbering: canonical PTEN (P60484, NM_000314, ENST00000371953, 403 aa). The
sampler's construct is pten_1d5r, UniProt 7-285 + 310-353 (324 sites).

## 1. What the files hold after cleaning

| Database | Rows in file | Canonical missense rows | Distinct protein changes | Inside the construct |
| --- | --- | --- | --- | --- |
| cBioPortal (TCGA PanCancer, 25 studies) | 1,203 | 516 (417 patients) | 263 | 254 |
| ClinVar | 4,277 | 1,223 | 1,164 | 1,001 |
| gnomAD | 1,868 | 235 | 235 | 176 |
| Union | | | 1,269 | 1,093 |

The 176 outside the construct are 93 in the C-tail, 60 in the 286-309 loop, 23 in the
N-terminus. They are counted but have no energy (decision 1 of the plan).

### The three traps

1. **ClinVar's header is one column short** (24 fields, every row 25). A plain
   `read_csv` takes column 1 as the index and every label moves one place to the right.
   `clean` checks the field counts and reads with `index_col=False`.
2. **PTEN-L.** The upstream-start isoform adds 173 residues. ClinVar lists both numberings
   ("M174L, M1L"); gnomAD puts 232 of its 469 missense rows on ENST00000693560, i.e. on
   the canonical 5' UTR. gnomAD's "C65S, AF 0.994" and "G4R, AF 0.143" are those rows.
   Only `NM_000314(PTEN)` names in ClinVar and only `ENST00000371953` rows in gnomAD are
   used.
3. **KLLN** overlaps the PTEN promoter: 5 ClinVar rows and 2 gnomAD rows (S153 on
   ENST00000472832, intronic in PTEN) go with the same filter.

### Duplicates

- **cBioPortal:** no (sample, change) duplicate. One patient carries the same change in two
  samples: counted once. 77 samples carry 2-5 PTEN missense changes; they are kept as
  compound sequences. 6 protein changes are reached by 2 different DNA changes (C124S,
  D92E, G129R, G36R, S10R, W111R). 119 samples carry > 500 mutations (hypermutated) and
  179 of the 516 missense rows.
- **ClinVar:** no duplicate VariationID or SPDI. 55 protein changes are carried by 2+
  variations (different codons, and 22 MNV/indel rows). Collapsed to one row per change:
  class = most severe call among the rows with the highest review status. Classes after
  collapse: 798 VUS rows, 274 P/LP, 141 conflicting, 10 B/LB.
- **gnomAD:** no duplicates on the canonical transcript; 0 rows fail the joint filter;
  42 segdup/lcr rows kept and flagged. 146 singletons, 89 with AC >= 2.

### SNV reachability from the files' own codons

The files name the reference base at 1,065 of the 1,212 CDS positions (every
`c.N X>Y`, synonymous and nonsense included). With the protein constraining the rest,
362 of 403 codons resolve to one codon. Of the 7,657 substitutions, 2,348 are one base
away, 5,202 are not, 107 stay unknown. Check: all 1,951 variants observed as SNVs come out
reachable. The 12 observed but unreachable changes are all ClinVar MNVs. On astaroth,
`PTEN_variant_data.tsv`'s `snv` column fills the unknowns and is cross-checked.

## 2. Intersections

| Set (exclusive) | Variants | In construct |
| --- | --- | --- |
| cBioPortal only | 64 | 60 |
| ClinVar only | 796 | 687 |
| gnomAD only | 38 | 30 |
| cBioPortal + ClinVar | 174 | 170 |
| cBioPortal + gnomAD | 3 | 2 |
| ClinVar + gnomAD | 172 | 122 |
| all three | 22 | 22 |

At the DNA level (GRCh38; cBioPortal lifted by the constant 1,759,757, verified on all
3,000 ClinVar SNVs): C&V 196, C&G 23, V&G 189, all three 20. Four protein changes shared by
cBioPortal and ClinVar arrive through a different DNA change.

- gnomAD sits mostly inside ClinVar (194 of 235).
- 25 tumour changes are also in gnomAD, among them hotspots R173C (13 patients, AC 5),
  R142W (11, AC 5), R130L (7, AC 1) and P246L (6, AC 3). 24 ClinVar P/LP changes are in
  gnomAD, all at AC <= 5. So "seen in gnomAD" is not "benign" for PTEN, and the
  population categories exclude P/LP where it matters.
- The databases' own cross-references disagree with the ClinVar file in 19 of 184
  variants (cBioPortal's ClinVar column, an older snapshot, mostly VUS <-> conflicting)
  and 1 of 189 (gnomAD's: F241L conflicting -> P/LP). The file wins; the list is in
  `merge.log`.

## 3. Categories

Per database: cBioPortal all / recurrent (2+ patients) / singleton / non-hypermutated /
endometrial / glioblastoma / with loss of the other allele; ClinVar all / P/LP / VUS /
conflicting / expert panel / PHTS / autism-macrocephaly; gnomAD all / singleton / AC >= 2.
Merged: tumour-only, population-only, both; consensus damaging = ClinVar P/LP or tumour
recurrent, and not tolerated; consensus tolerated = gnomAD AC >= 2 and not P/LP, and not
damaging. Every category with 20+ variants in the construct is profiled.

## 4. K-site entropy profiles

For each category and site: the density (observed SNV-reachable changes / SNV-reachable
changes at the site), the substitution entropy of the observed mutant residues (weighted
by patients in cBioPortal, allele count in gnomAD), and the K-sets under it. Sampled
entropy: equil_ref chains 0 and 1, `--burn-in-frac 0.8`, mean of the two. Sampler
K-sites: the 10% least variable native sites at T4 (33 sites, entropy <= 0.168) and T8
(33, <= 0.677). Figures: `kset_profiles/<category>.png`.

**Variants in each K-set, observed / expected** (expected = uniform draw over the 1,875
SNV-reachable substitutions of the construct; full table with p in `profiles.log`):

| Category | K_natural | K_abundance | K_activity | K_sampled T4 | K_sampled T8 |
| --- | --- | --- | --- | --- | --- |
| tumour, all | 2.95 | 1.45 | 2.23 | 2.07 | 1.90 |
| tumour, recurrent | 4.51 | 1.34 | 2.80 | 3.26 | 3.41 |
| tumour, non-hypermutated | 3.25 | 1.52 | 2.50 | 2.00 | 2.07 |
| ClinVar P/LP | 3.79 | 1.37 | 2.68 | 2.20 | 2.60 |
| ClinVar VUS | 0.50 | 1.01 | 0.74 | 0.74 | 0.71 |
| ClinVar autism/macrocephaly | 1.25 | 1.41 | 1.59 | 1.11 | 1.33 |
| gnomAD, all | 0.95 | 0.87 | 0.83 | 0.85 | 0.87 |
| consensus damaging | 3.74 | 1.39 | 2.62 | 2.22 | 2.61 |
| consensus tolerated | 0.26 | 0.53 | 0.22 | 0.70 | 0.90 |

What it says:

1. **Disease variants land on conserved, catalytic sites, much more than on
   fold-critical ones.** Tumour and P/LP variants are 3-4.5x enriched in K_natural and
   2.2-2.8x in K_activity, but only 1.3-1.5x in K_abundance. Per site, the density of
   consensus-damaging variants correlates -0.64 with activity and -0.62 with family
   entropy, but only -0.27 with abundance, and that drops to -0.01 once burial (RSA) is
   regressed out (activity -0.55, family -0.56). The pattern "conservation = catalysis,
   abundance = fold" from the K-site work reappears in patients.
2. **The sampler's K-sites are disease sites too.** Consensus-damaging variants are 2.2x
   (T4) and 2.6x (T8) enriched in them, P < 1e-7; recurrent tumour variants 3.3-3.4x. The
   density correlates -0.40 with the sampled entropy, -0.24/-0.21 after RSA. Weaker than
   the family alignment, stronger than abundance after RSA.
3. **Population variants are spread uniformly.** gnomAD is at 0.83-0.95 everywhere (no P
   below 0.17), and consensus tolerated is depleted (0.26 in K_natural, 0.22 in
   K_activity).
4. **VUS sit where P/LP are not** (0.50 in K_natural, positive correlation with
   tolerance). That is how ClinVar classifies, not biology: a variant at a known critical
   site gets called pathogenic, and what is left as VUS is elsewhere.
5. **Autism/macrocephaly variants are not enriched** (38 variants, every P > 0.03), in
   line with the published picture of hypomorphic, partially active PTEN alleles in
   autism. Small sample; a lead, not a result.

## 5. Ranking by U_am, the 2-step contact energy E2 and U_structure_ce

Every variant inside the construct is read from the existing scans
(`pten_single_mutants.csv`, `pten_mutants_E2.csv`, `pten_mutants_ce.csv` float32).
Output `pten_variants_ranked.tsv`: energies, percentiles among all 6,156 and among the SNV-
reachable mutants, rank within the site, experiments and labels.

**AUC, higher score = damaging, SNV-reachable variants, 95% bootstrap CI** (`rank.log`):

| Contrast (n pos / neg) | U_am | E2 | U_ce | U_am site mean | U_ce site mean | -abundance | -activity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tumour vs population-only (249/150) | 0.670 [0.61,0.73] | 0.519 | 0.710 [0.66,0.76] | 0.560 | 0.698 | 0.700 | 0.792 |
| tumour recurrent vs gnomAD AC>=2 (71/45) | 0.709 [0.60,0.81] | 0.503 | 0.773 [0.68,0.85] | 0.564 | 0.759 | 0.678 | 0.855 |
| ClinVar P/LP vs gnomAD (244/150) | 0.705 [0.65,0.76] | 0.541 | 0.751 [0.70,0.80] | 0.552 | 0.718 | 0.767 | 0.897 |
| ClinVar P/LP vs VUS (244/612) | 0.682 [0.64,0.72] | 0.543 | 0.704 [0.67,0.74] | 0.555 | 0.658 | 0.674 | 0.861 |
| consensus damaging vs tolerated (270/45) | 0.707 [0.61,0.79] | 0.504 | 0.768 [0.68,0.84] | 0.556 | 0.741 | 0.804 | 0.901 |
| tumour recurrent vs singleton (71/178) | 0.525 | 0.537 | 0.579 | 0.495 | 0.549 | **0.386** | 0.613 |

**Within-site AUC** (pairs at the same site only, so burial cancels):

| Contrast | U_am | E2 | U_ce | -abundance | -activity |
| --- | --- | --- | --- | --- | --- |
| ClinVar P/LP vs VUS (225 pairs, 74 sites) | **0.708** | 0.543 | 0.682 | 0.709 | 0.684 |
| ClinVar P/LP vs never observed | 0.659 | 0.545 | 0.655 | 0.691 | 0.738 |
| tumour vs never observed | 0.579 | 0.493 | 0.594 | 0.560 | 0.615 |

What it says:

1. **U_am and U_ce both separate disease from population variants, at 0.67-0.71 and
   0.70-0.77.** Comparable to the VAMP-seq abundance score itself (0.67-0.80), below
   Mighell's activity (0.79-0.90). Restricting to SNV-reachable variants changes almost
   nothing here, because the observed variants are nearly all SNVs on both sides.
2. **The two energies know different things, as on VAMP-seq.** U_am's per-site mean
   falls to 0.55-0.58: its power is in ranking substitutions. U_ce's site mean stays at
   0.66-0.76: it reads the site. Within a site, U_am separates P/LP from VUS at 0.708,
   level with abundance (0.709) and above activity (0.684).
3. **E2 is at chance: 0.50-0.54 per variant, 0.43-0.50 as a site mean.** The 2-step
   contact energy sees nothing of disease here, consistent with the verdict of 2026-11-17
   (schedule-dependent per mutant). Its site mean did not rescue it either.
4. **Tumour hotspots are stable but dead, and the energies miss them.** Recurrent tumour
   variants are MORE abundant than singletons (AUC 0.386 for -abundance): R130Q has
   abundance 1.42 and activity -2.04, R130G 1.12 and -2.14. U_am puts R130Q at the 59th
   percentile, U_ce at the 44th. Both energies read the fold. The commonest PTEN tumour
   mutations destroy catalysis without touching the fold, so no fold energy can rank them.
   This is the patient-side form of "U_am finds the pocket's packing, not its chemistry".
5. **Most expensive tumour variants by U_am:** P281S (99.1 pct, VUS, also in gnomAD AC 2),
   F215S, M198K (P/LP), R335P, V217G, F195S (P/LP), L325R, H272P (P/LP). Mostly C2-domain
   core substitutions; P281S is the one that disagrees with everything else
   (abundance 0.91, activity 0.04).

### Caveats

- **Circularity on the ClinVar side.** ClinVar's PTEN expert panel can use functional
  assays and computational predictors as evidence, so the abundance, activity and
  AlphaMissense AUCs against ClinVar labels may be inflated. U_am and U_ce never saw
  ClinVar, so their AUCs carry no such inflation.
- **gnomAD is not benign.** It means "seen in gnomAD", mostly once. 24 P/LP
  changes are in it. The contrasts exclude P/LP from the population side.
- **cBioPortal here is TCGA PanCancer only.** Half of the missense rows are endometrial.
  Hypermutated samples are reported separately and change little (non-hypermutated
  tumour vs population: U_am 0.684, U_ce 0.724).
- **PHTS vs autism** has 12 negatives: no conclusion.

## 6. Still to run on astaroth (GPU)

`python pten_variants.py score` computes U_am, U_ce and E2 for the 75 tumour samples
with 2+ PTEN missense mutations inside the construct (42 of them hypermutated), and
re-decodes 50 single mutants to check E2 against the scan. Then `rank` again adds the
additivity of the compounds (compound vs the sum of its singles) to `rank.log` and
`compound_additivity.png`. About a minute of GPU plus the model load.

On astaroth, `clean` and `rank` also pick up `pten_natural/raw/PTEN_variant_data.tsv`:
its `snv` column for the 107 unknown substitutions, and its 15 predictors next to U_am.

## 7. Other databases checked (2026-11-19, not in `databases/`)

- **MaveDB urn:mavedb:00000013-a-1** (Matreyek 2018 VAMP-seq): score = 2018 abundance (rho 0.95 with `abundance_2021`); calibration columns are score thresholds turned into PS3/BS3, not labels; ClinVar annotations (yearly 2015-2026) are 95% identical to our ClinVar file and add no benign missense. Only new: on 2018 labels activity AUC is 0.70 vs 0.85 on 2026 labels, the energies barely move (circularity of the activity AUC).
- **ClinGen Evidence Repository, PTEN VCEP** (complete exports, 235 records): 122 missense (83 P/LP, 33 VUS, 6 B/LB), all already in the merged table, all but M134I same class, still only 6 B/LB; 69/83 P/LP use PS3. P/LP vs VUS in construct (80 vs 27): U_am 0.67, U_ce 0.65, E2 0.58, abundance 0.68, activity 0.87 (circular). Details in `pten_variants.tex`, section "Other databases checked".
