# Reading a structure comparison, from scratch

Written 2026-11-06 for the neuraminidase result, but nothing here is specific to
it. It explains every number `compare_structures.py` prints, and the vocabulary
around it, assuming you had not opened a `.pdb` file before this week. The
companion documents are `../monte_carlo/structure_comparison_guide.md` (how to
run it) and `structure_comparison.tex` (the same as a typeset walk-through).

---

## 1. Reminder: what is in a .pdb file

A PDB file is a text file, one line per atom. The lines that matter here start
with `ATOM`, and the columns are at fixed character positions, which is why
these files are parsed by position and not by splitting on spaces:

```
ATOM      1  N   VAL A  83       0.065  65.327  77.027  1.00 32.79           N
             ^    ^   ^   ^      ^^^^^^^^^^^^^^^^^^^^^^ ^^^^ ^^^^^
             |    |   |   |      x, y, z in angstrom    occ  B-factor
             |    |   |   residue number (83)
             |    |   chain id (A)
             |    residue name (valine)
             atom name (backbone nitrogen)
```

Vocabulary you need:

* **Angstrom (Å)**: 10⁻¹⁰ m. A C–C bond is about 1.5 Å, consecutive Cα atoms are
  3.8 Å apart, a small domain is 30–40 Å across.
* **Cα ("C-alpha")**: the central carbon of each amino acid, the one the side
  chain hangs off. One per residue, so the list of Cα coordinates is a compact
  summary of the backbone's path. Every number in the comparison uses Cα only.
* **Residue numbering is the author's choice**, not an index. 2HU0 starts at 83
  because the construct is a fragment of a longer protein, and it can skip
  numbers (disordered residues) or repeat one with a letter suffix (insertion
  codes, e.g. 412A). Never assume `number = position in the file`.
* **B-factor**: the last numeric column. In a crystal structure it measures how
  smeared out an atom is. In our *predicted* files it does not: `decode_to_pdb.py`
  writes the per-residue **pLDDT** there instead, because every viewer can colour
  by that column. The file's provenance tells you which meaning applies.

Format reference: <https://www.wwpdb.org/documentation/file-format>. The
experimental file here is RCSB entry 2HU0, <https://www.rcsb.org/structure/2HU0>.

## 2. Superposition and RMSD

Two structures of the same protein come in arbitrary positions and orientations:
the coordinates are relative to whatever box the crystallographer or the model
happened to use. Comparing them means first moving one on top of the other, and
only then measuring what is left over.

**Superposition** finds the rotation and translation that minimise the summed
squared distance between paired atoms. There is a closed-form solution, Kabsch's
algorithm, which is what `kabsch()` in `compare_to_pdb.py` implements. Note what
it does *not* do: it never bends or stretches anything. The structures are rigid;
only their placement changes.

**Pairing** comes first, and is done by sequence alignment, not by residue
number — which is how the 2HU0 numbering (83–468) is matched to the prediction's
(1–385) despite the gaps and insertion codes.

**RMSD** (root-mean-square deviation) is then

&nbsp;&nbsp;&nbsp;&nbsp;RMSD = sqrt( (1/N) · Σᵢ dᵢ² )

where dᵢ is the distance, in Å, between residue i's Cα in the prediction and the
Cα of its paired residue in the experimental structure, after superposition.
**That distance is what "deviation" means everywhere in the output**: deviation
from the experimental structure, per residue, in Å, measured after the best
rigid fit. One number per residue; the CSV lists all of them.

Rules of thumb: below 1 Å is the same structure; 1–2 Å is the same structure
solved twice or a close homologue; 2–4 Å is recognisably the same fold with real
differences; above ~8 Å, whatever similarity remains is not visible in the
number.

**Why RMSD alone misleads.** It is a mean of *squares*, so one badly placed tail
dominates it. Ten residues at 40 Å contribute more to the sum than 300 residues
at 2 Å. A single number therefore cannot distinguish "uniformly 15 Å wrong" from
"mostly right, two pieces flung away" — and those are completely different
results.

## 3. Core RMSD: what "a 116-residue core superposes at 1.17 Å" means

The fix is to ask a second question: *is there a subset of the structure that
does match, and how big is it?*

`core_fit()` answers it by iterated outlier rejection. Superpose on everything,
drop every residue pair deviating by more than the cutoff (default 2.0 Å),
superpose again on what is left, repeat (default 5 cycles). What survives is the
part of the prediction that is genuinely right, fitted without the rest dragging
it around.

So "core RMSD 1.17 Å over 116 residues (30%)" reads: **116 of the 385 residues,
30% of the chain, agree with the crystal to 1.17 Å — essentially perfectly — once
the other 70% is excluded from the fit.** It is two numbers and both matter. A
core of 1.2 Å over 30% is very different from 1.2 Å over 90% (that would be a
solved structure) and from 1.2 Å over 5% (which any two structures can manage by
accident). Reporting the pair is the honest form; the cutoff and cycle count
belong with it, since a looser cutoff always buys a bigger core.

## 4. TM-score

TM-score (template modelling score) is the standard answer to "is this the same
fold?", designed to fix two problems with RMSD: outlier domination, and length
dependence (random structures of 50 residues score much better by RMSD than
random structures of 500).

&nbsp;&nbsp;&nbsp;&nbsp;TM = (1/L) · Σᵢ 1 / (1 + (dᵢ/d₀)²),&nbsp;&nbsp; d₀ ≈ 1.24·∛(L−15) − 1.8

Each residue contributes between 0 and 1: close to 1 when it lands well within
d₀ of where it should be, close to 0 when it is far. A far-flung tail just stops
contributing instead of swamping the sum, which is exactly what RMSD gets wrong.
d₀ grows with chain length L, which is what makes scores comparable across
proteins.

The conventional thresholds: **above 0.5 means the same fold, below 0.17 is what
unrelated structures score.** Our 0.563 is above the fold threshold — and the
script labels it a *lower bound* because finding the optimal TM superposition is
a search, and `tm_score()` does a limited fragment-seeded version of that search,
so the true value can only be higher, never lower.

References: Zhang & Skolnick, *Proteins* **57**:702 (2004) for the score;
Xu & Zhang, *Bioinformatics* **26**:889 (2010) for the 0.5 threshold.

## 5. pLDDT, and what "the sub-50 residues" means

**lDDT** (local distance difference test) scores a model by how well it
reproduces the *local* interatomic distances of the reference, without any
superposition at all. It is local by construction, so one wrong domain does not
penalise a correct one.

**pLDDT** is a *prediction* of that score, produced by the structure predictor
itself, per residue, 0–100. It is the model's own confidence: no experimental
structure is involved, which is why we can read it for sequences that have never
been crystallised — the whole reason it matters for this project. AlphaFold's
published bands are the ones the viewers colour by: >90 high, 70–90 confident,
50–70 low, <50 very low and often disordered.

So "the sub-50 residues average 24.6 Å of deviation" means: take the 67 residues
the model marked with pLDDT below 50 (its own "I don't know"), look up their
deviations from §2, and the mean is 24.6 Å. Beside them, the 36 residues it
marked above 90 average 2.13 Å. **The model's confidence and its actual error
agree**, which is the point of quoting them together: Spearman correlation −0.80
between pLDDT and deviation (negative = low confidence where the error is large).

That is itself a result. For HA0 the confidence was uniformly mediocre and told
us nothing; here the model is specific about which parts it is unsure of, so the
pLDDT map can be trusted as a map of *where* the prediction fails.

References: Mariani et al., *Bioinformatics* **29**:2722 (2013) for lDDT;
Jumper et al., *Nature* **596**:583 (2021), §"Confidence measures", for pLDDT.

## 6. The per-block table

"Per 20-residue block" is nothing more than: chop the prediction's residues into
consecutive groups of 20 (1–20, 21–40, …) and compute the RMSD of §2 *within each
group*, using the deviations from the one global superposition. It is a way of
reading the per-residue CSV without plotting it — it localises the error along
the chain. The same information is in `*_per_residue.png` as a curve.

For the NA run:

| block (predicted numbering) | RMSD |
|---|---|
| 1–20 | 29.6 Å |
| 21–100 | 5.2–11.6 Å |
| 101–240 | 2.8–4.7 Å |
| 241–300 | 7.7–14.8 Å |
| 301–340 | 15.5, 32.6 Å |
| 341–385 | 21.5–41.1 Å |

Read it as: the middle of the chain is close to right, both ends are far away.
And the ends are not merely floppy tails, because cutting 30 residues off each
end still leaves 12.1 Å over the remaining 325.

## 7. The β-propeller hypothesis

This is the part worth drawing on paper.

A **β-propeller** is a fold built from repeats called **blades**. Each blade is a
four-stranded antiparallel β-sheet, and the blades are arranged around a central
axis like the vanes of a turbine — hence the name, and hence the pinwheel you see
looking down that axis. Influenza neuraminidase is a **six-bladed** propeller
(Varghese, Laver & Colman, *Nature* **303**:35, 1983), with the active site in the
funnel on the axis.

The structurally important detail: the chain runs through the blades in order,
blade 1 first, blade 6 last — so **the first and last strands of the chain end up
side by side in the same blade**, and that blade is only complete because the
C-terminal strand closes it against the N-terminal ones. This closure is called
the **molecular Velcro**, and it is what makes the ring a ring instead of a spiral
(Fülöp & Jones, *Curr. Opin. Struct. Biol.* **9**:715, 1999).

Now the hypothesis. If a predictor builds the middle blades correctly but fails
to make that last contact, the ring does not close: the chain's two ends, which
in the real structure sit against each other, are instead free, and both drift to
wherever the rest of the model pushes them. The signature of that failure is
exactly what we measured — a correct middle and *both* ends thrown far out
together, not one end, and not a uniform spread of error.

That is a reading of the numbers, not an observation. Confirming it takes thirty
seconds in a viewer: look down the propeller axis, count the blades, and see
whether the first and last strands lie in the same blade or point off into space.
If the ring is open, the hypothesis holds; if the ring is closed and the error is
somewhere else entirely, it does not.

## 8. What `compare_structures.py` writes

With `--out-prefix structure_compare/na_head_vs_2HU0`:

| file | what it is |
|---|---|
| `..._superposed.pdb` | the prediction, already rotated onto the experimental structure. Use this one in any viewer that does not fit for you |
| `..._per_residue.csv` | one row per paired residue: predicted chain and number, amino acid, pLDDT, matching experimental chain and number, Cα deviation. This is the file that carries the numbering correspondence, so use it instead of doing arithmetic on residue numbers |
| `..._per_residue.png` | deviation and pLDDT plotted along the sequence; the visual form of §6 |
| `..._view.pml` | PyMOL script: loads both, strips what should be stripped, colours by pLDDT |
| `..._view.cxc` | the same for ChimeraX. Run either from the folder it is in, since the paths inside are relative |

The terminal output is not saved, which is why the runs redirect it into
`na_head_decode.log` and `na_head_cmp.log`.

## 9. The short version, for the NA result

| number | value | meaning |
|---|---|---|
| mean pLDDT | 67.7 | the model's own confidence, below the ~90 of a protein it folds properly |
| Rg | 19.7 Å vs 18.9 Å | the prediction is the right size and compact |
| RMSD, all | 14.93 Å | as a whole, it does not match |
| RMSD, core | 1.17 Å over 116 res | but 30% of it matches essentially perfectly |
| TM-score | 0.563 (lower bound) | above the 0.5 same-fold line |
| Spearman(pLDDT, dev) | −0.80 | the model knows which parts are wrong |

Which is why the verdict is "fails the gate, but informatively": not a structure
we can define an effective energy on, yet not the flat failure HA0 was.
