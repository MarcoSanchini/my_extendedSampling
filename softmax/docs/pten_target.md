# PTEN as the sampling target

Written 2026-11-10, after the supervisor chose PTEN over the alternatives that
passed the decode gate. Nothing here has been run; it is the case for and
against the choice, and what has to change in the plan because of it.

## What PTEN is

**P**hosphatase and **TEN**sin homolog, a human tumour suppressor, 403 residues
in the full protein. Three parts:

- residues ~1-13, a short motif that binds the membrane lipid PIP2;
- residues ~14-185, the **phosphatase domain**: the fold of a protein tyrosine
  phosphatase, with the usual CX5R catalytic loop, but a wider and more basic
  active site, because its real substrate is a lipid head group rather than a
  phosphotyrosine. The catalytic cysteine is C124;
- residues ~186-350, a **C2 domain**: a beta-sandwich that binds the membrane
  and clamps the phosphatase domain against it, so the two domains work as one
  unit rather than as beads on a string;
- residues ~351-403, a disordered regulatory tail carrying phosphorylation
  sites and a PDZ-binding motif.

The reference `pten_1d5r` is the 1D5R construct, 324 residues from SEQRES, which
covers the first three parts and not the tail. A 31-residue loop in the C2
domain (282-312) has no coordinates in the crystal and is restored from SEQRES.

## What it does

PI3 kinase phosphorylates the membrane lipid PIP2 to make PIP3. PIP3 recruits
AKT to the membrane, and AKT switches on growth, proliferation and survival.
PTEN is the brake: it takes the 3-phosphate back off, turning PIP3 into PIP2
and shutting the signal down. It is the main negative regulator of the
PI3K/AKT pathway, and that single reaction is most of its biology.

Lose PTEN and the brake is gone: AKT stays on, and the cell behaves as though
it is permanently being told to grow.

## Medical relevance

Considerable, and easy to state to a non-specialist.

- PTEN is among the most frequently lost or mutated tumour suppressors in human
  cancer, prominently in glioblastoma, endometrial, prostate and breast cancer.
- Germline mutations cause **PTEN hamartoma tumour syndrome**, which includes
  Cowden syndrome and Bannayan-Riley-Ruvalcaba syndrome: benign overgrowths plus
  a high lifetime risk of breast, thyroid and endometrial cancer.
- Germline PTEN variants also appear in a subset of autism spectrum disorder
  with macrocephaly.
- Because so many clinical variants are missense and of unknown significance,
  predicting whether a given substitution destabilises the protein is an actual
  clinical problem, not a hypothetical one.

That last point is the one worth making in the thesis: the question this
project asks of a sequence - does it still fold - is the question the clinic
asks about a PTEN variant.

## Sequence variability, and what the K-site comparison becomes

This is where the choice changes the plan, and it should be discussed before
the comparison is built.

**Natural orthologs are too conserved to be informative on their own.** PTEN is
almost invariant across vertebrates; human and mouse differ at a handful of
positions. An alignment of orthologs therefore has near-zero entropy in most
columns, which cannot discriminate between sites: everything looks conserved.
Widening to the whole protein tyrosine phosphatase superfamily (Pfam PF00102)
restores depth, but then only the phosphatase domain aligns, the C2 domain
(PF10409) has to be handled separately, and pairwise identity falls to the
20-30% range where the alignment is a model rather than a transcription. That is
the same tradeoff written up in `target_selection.md`, with the balance tipped
further than usual because this protein is *so* conserved.

**But PTEN has something better than an alignment: experimental mutational
scans.** Two large datasets exist, both from 2018:

- Matreyek et al., *Nature Genetics* 50:874 (VAMP-seq), which measured the
  steady-state cellular **abundance** of roughly four thousand PTEN missense
  variants;
- Mighell et al., *American Journal of Human Genetics* 102:943, which measured
  the **function** of roughly seven and a half thousand.

VAMP-seq is the interesting one for us. Abundance in the cell is largely set by
whether the variant folds and escapes degradation, so a VAMP-seq score is close
to an experimental measurement of folding stability, per position, for most of
the single mutations. That is a far better match to what the sampler's energy
knows than a natural alignment is: U_am is a statement about foldability and
nothing else, with no immunity, no catalysis and no phylogeny in it, and the
long-standing objection to the HA comparison was precisely that natural entropy
mixes all of those in.

So the proposal is to **compare the sampled per-site tolerance against the
VAMP-seq abundance landscape**, with the functional scan and the natural
alignment as secondary references rather than the primary one. The comparison
gets stronger, not weaker, by the switch: we would be testing the energy
against a measurement of the thing it claims to model.

Both references need checking against the papers before any of this is built;
they are cited here from reading, not from files in hand.

## Compared with topoisomerase I

Both passed the gate. The differences that matter:

| | PTEN (1D5R) | Topoisomerase I (1A36) |
| --- | --- | --- |
| Reference length | 324 aa | 591 aa |
| pLDDT | 93.8 | 91.3 |
| Core RMSD | 0.82 A over 86% | 1.04 A over 93% |
| TM-score | 0.939 | 0.977 |
| Contact-map E | 0.111 | 0.061 |
| Crystal | protein alone | protein clamped around a 22 bp DNA duplex |
| Construct | wild type | catalytically dead active-site mutant |
| Per-step cost | baseline | roughly 3x the attention cost at 1.8x the length |

**Topoisomerase I is the better prediction and the better fit to the length
criterion.** It is 591 residues against Marco's stated floor of about 350, and
the one-pass gradient move was measured to pay off most in high dimension, at
2.38x over `site_soft` at 566 residues. On the structural numbers it beats PTEN
on everything except mean pLDDT.

**PTEN is the cleaner experiment.** Its crystal is the protein by itself: no
DNA, no engineered mutation, one compact two-domain chain. Topoisomerase I's
crystal is an enzyme closed around a DNA duplex, so every structural comparison
of a sampled ensemble against it would carry an unanswerable question - is the
difference a folding difference, or the difference between the DNA-bound and
DNA-free conformations? The model happens to have closed the clamp correctly
without the DNA, which is impressive, but it does not remove the ambiguity from
later comparisons.

**And PTEN is three times cheaper per step.** Attention is quadratic in length,
so 591 against 324 residues is roughly a factor of three in the dominant cost of
every single move. In a study whose whole point is sampling speed, that is not
a detail.

## The one real concern, and how to settle it

324 residues is below the ~350 floor set when the small candidates were
rejected, and the reason for that floor was that the gradient-informed move
should pay off most in high dimension. 324 is only 7% short, so this is a
quantitative question rather than a reason to object, and it has a cheap answer:
**benchmark the one-pass move at 324 residues before committing**, the way it
was benchmarked at 566. If the speed-up holds, the objection is closed with a
number. If it collapses, that is an argument to put to the supervisor, with
topoisomerase I as the ready alternative.

The other option, using full-length PTEN at 403 residues instead of the crystal
construct, buys 79 residues but adds a disordered tail that no structure
comparison can score. Not worth it.

## Recommendation

Go with PTEN. It is the supervisor's call and it is a defensible one: cleaner
crystal, cheaper sampling, genuine clinical weight, and a mutational-scan
dataset that fits this project's energy better than any natural alignment we
have had so far. Run the speed benchmark at 324 residues first so the length
question is answered with a measurement rather than an argument, and plan the
K-site comparison against VAMP-seq.
