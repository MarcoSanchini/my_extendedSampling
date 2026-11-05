# Choosing the next sampling target

Status: open decision, drawn up 2026-11-05. Nothing here has been run.

## Why a new target

`zero_polymer` is an influenza hemagglutinin HA0 precursor, 566 aa. The decode
diagnosis of 2026-11-04 (DEVLOG, PRs #24-#29) found that ESM3-open does not
fold it: pLDDT 0.51-0.62 as a monomer and 0.42-0.48 as a glycine-linked trimer,
against 0.91-0.93 for single-chain controls of comparable length (MBP 370 aa,
amylase 496 aa, luciferase 550 aa). The HA0 reference's confidence sits barely
above random 566-mers. Since the effective energy is defined on the predicted
structure, a target whose structure the predictor does not get right makes every
energy difference suspect, so the study needs a target the predictor handles.

## Criteria

1. **Long chain, >= ~350 aa.** The thesis is about making the sampler faster, and
   the gradient-informed (one-pass) move is expected to pay off most in high
   dimension. A 76- or 129-residue domain would sample quickly but would not
   test the thing being built. This rules out ubiquitin, lysozyme and the
   SH3/WW domains.
2. **Deep natural sequence variation.** The K-site / per-site-entropy comparison
   is the main way the sampled ensemble is judged against reality, and it needs a
   natural alignment deep enough to estimate a 20-letter distribution per column.
   See the section on what "deep" has to mean.
3. **Folds as a single chain under ESM3-open.** Require pLDDT ~0.9 on one test
   decode of the reference before spending sampling time, exactly as the monomer
   controls did. This is the criterion HA failed.
4. An experimental structure to compare against, which all candidates below have.

## Candidates

| Target | Length | Decode | Natural variation |
| --- | --- | --- | --- |
| Neuraminidase, catalytic head | ~390 of ~470 aa | untested | deepest; same route as the HA work |
| Cytochrome P450, soluble bacterial | 414 aa (P450cam), ~455 aa (BM3 heme domain) | untested | very large family (PF00067) |
| Firefly luciferase | 550 aa | pLDDT 0.93 measured | large but divergent family (PF00501) |
| Maltose binding protein | 370 aa | pLDDT 0.93 measured | moderate |
| Protein kinase catalytic domain | ~290 aa | untested | one of the largest families (PF00069) |

**Neuraminidase** is the supervisor's suggestion and the current first choice. Its
catalytic head is a self-contained six-bladed beta-propeller: cutting the
cytoplasmic tail, the transmembrane helix and the stalk leaves a domain that is
believed to fold on its own, which is the structural difference from HA, whose
stalk only exists inside the trimer. Native NA is a tetramer, so the quaternary
risk is not zero and criterion 3 has to be checked before anything else.
Structures: 2BAT (N2), 3TI6 / 3B7E (N1). Sequence data lives in influenza
segment 6, so `ha_site_entropy.py` applies with only the segment changed; the
FASTA sets already in the repo are segment 4 (HA) and do not cover it.

**Cytochrome P450** is the best non-viral option: monomeric, soluble, in HA's
length class, and from a family diverse enough that what is conserved across it
is structural rather than historical.

**Luciferase** and **MBP** are the low-risk options, since both already have a
confident decode on record; MBP needs no new data at all.

**Protein kinase** is listed for family depth but is short for criterion 1.

## What "deep natural variation" has to mean

The per-site entropy comparison asks whether the sampler conserves the sites
nature conserves. That is only a statement about foldability if the natural
entropy itself measures constraint on the fold, so the two families differ in
more than how many sequences they contain.

**Viral families (influenza HA, NA).** Easy and dangerous. Easy because all
sequences are the same gene at nearly the same length, so the alignment is
almost gap-free and columns map one-to-one onto the reference's residues, with
no alignment ambiguity to argue about. Dangerous for two reasons:

- *Shared ancestry.* Every human H3N2 HA sequence descends from one ancestor
  circulating since 1968. They are repeated samples of a single evolutionary
  trajectory, not independent draws from the space of foldable sequences, so a
  site can be invariant simply because no substitution has been tried there yet.
- *Selection that is not foldability.* Variation in the HA head is driven by
  antibody escape and receptor binding. The sampler has no immune system, so
  comparing against that part of the profile compares against a pressure the
  energy does not contain. This is the H3N2 objection already hit on 2026-11-03:
  the measured agreement was concentrated in the head and the receptor-binding
  site, while the stalk, whose constraint is quaternary, looked variable.

The planned partial fix is to widen to a whole HA group, or all subtypes, so that
conservation across deeply diverged lineages means the fold needs it. But the
redundancy weighting, `1/(neighbours at >= 80% identity)`, then collapses each
subtype to roughly one cluster, so the effective count N_eff falls to about the
number of subtypes, a handful. That is the trap: 19,380 raw sequences can carry
an N_eff of order tens, which cannot support a per-site 20-letter distribution
over 566 columns.

**Large protein families (P450, kinases, PF-scale alignments).** The opposite
balance. Members come from bacteria, fungi, plants and animals separated by
hundreds of millions to billions of years, under selective pressures that do not
correlate with each other, so the variation is close to independent and what
survives across the family is overwhelmingly structural and catalytic. N_eff
after weighting stays in the thousands, which is what per-site frequencies
actually need. The cost is that at 20-60% pairwise identity there are genuine
insertions, deletions, variable loop lengths and sometimes extra domains, so the
alignment is a model rather than a transcription: it needs profile methods and a
gap-fraction cutoff (30% in `ha_site_entropy.py`), sites inside long insertions
get no natural entropy at all and drop out of the comparison, and loops that are
structurally real but compositionally free inflate the entropy.

**So the choice is about what the natural entropy is a measure of.** The
effective energy knows only whether a sequence folds to r_0: no function, no
immunity, no phylogeny. The natural reference that matches it is one that is
itself as close as possible to pure foldability constraint, which argues for a
deep, diverse family over a convenience sample of one circulating lineage.

Practical consequences either way:

- Report N_eff per column next to the entropy, not raw sequence counts.
- For NA, build the profile across subtypes and host species rather than one
  human lineage, and cross-check the weighting threshold at 0.95 as well as 0.80.
- Vary the gap cutoff to see which sites the comparison actually covers.
- Check whether the natural entropy correlates with simple structural proxies
  (relative solvent accessibility, contact number). The part of the profile the
  energy can be expected to reproduce is the part explained by burial and
  packing; if the comparison is dominated by that, it is a fair test.

## Blocked in the cloud

UniProt, NCBI, BV-BRC and EBI are all refused by the cloud container's egress
proxy, so every sequence fetch for a new target has to run on the GPU machine:
`tests/add_reference.py` for single chains, `ha_site_entropy.py fetch` for the
influenza sets.
