# Comparing an ESM3 structure with an experimental one

How to put the structure ESM3 predicts for a sequence next to a real
(experimental) structure from the Protein Data Bank, look at the two together,
and put numbers on how similar they are. The worked example is our HA reference
(`zero_polymer`, influenza hemagglutinin HA0) against PDB **2HMG** (an H3
hemagglutinin crystal structure).

Two new scripts in this folder do the work:

| script | what it does | needs |
|---|---|---|
| `decode_to_pdb.py` | decodes one sequence with ESM3 and saves the structure as a `.pdb` file, pLDDT in the B-factor column | GPU (astaroth) |
| `compare_structures.py` | superimposes predicted onto experimental, reports RMSD, TM-score, pLDDT, writes files for the viewers | numpy only, runs anywhere |

`decode_controls.py` and `decode_trimer.py` decode too, but they only keep summary
numbers (Rg, mean pLDDT) and never save coordinates, which is why
`decode_to_pdb.py` exists.

---

## 1. What a .pdb file is

A plain text file, one line per atom. Open one with `less 2HMG.pdb` and you see
lines like:

```
ATOM      2  CA  GLN A   1     -34.605  19.045  13.576  1.00 62.40           C
       │      │   │   │   │       x        y        z     occ   B-factor
       │      │   │   │   residue number
       │      │   │   chain (A, B, ...)
       │      │   residue (3-letter code)
       │      atom name (CA = the alpha carbon, one per residue)
       atom serial number
```

* **Chains.** A crystal can hold several protein chains. 2HMG holds the
  cleaved HA: HA1 and HA2 are separate chains, and the file has the trimer
  (three HA1 + three HA2 chains). ESM3 instead predicts one chain, HA0, the
  uncleaved precursor with the signal peptide.
* **B-factor column.** In an experimental file it measures how much each atom
  wobbles in the crystal. In a predicted file (AlphaFold, ESMFold, and the files
  `decode_to_pdb.py` writes) the same column holds **pLDDT**, the model's
  confidence per residue, 0 to 100. Viewers can colour by this column.
* **Coordinates** are in angstrom (1 Å = 0.1 nm; a C–C bond is about 1.5 Å, CA
  atoms of neighbouring residues are 3.8 Å apart).

## 2. The numbers, in plain words

What your supervisor called "wmd" is almost certainly **RMSD**, root-mean-square
deviation. It is the standard number for "how different are two structures".

* **RMSD.** Pair up the CA atom of each residue in the prediction with the CA of
  the same residue in the experiment. Rotate and shift the prediction to lie on
  top of the experiment as well as possible (the Kabsch algorithm finds the best
  rotation exactly). Then RMSD = sqrt(mean of the squared distances between
  paired atoms). It is a typical distance, in Å:
  * about 1 Å or less: the same structure, up to noise
  * 2 to 3 Å: the same fold, some loops or a domain off
  * 4 to 8 Å: large parts placed differently
  * above about 10 Å for a 500-residue protein: not the same shape

  Its weakness: one misplaced region inflates it for the whole protein. The
  script therefore also reports:
* **Core RMSD.** Fit, drop residues more than 2 Å off, refit, five times (this
  is what PyMOL's `align` reports by default). It says how well the part that
  matches matches. Always read it with the number of residues it kept: "1.1 Å
  over 40% of residues" is not a good result.
* **Fraction within 1, 2, 4, 8 Å.** After the global fit, what share of residues
  sits within that distance of its partner. "70% within 4 Å" is often easier to
  interpret than an RMSD.
* **TM-score.** Between 0 and 1, and unlike RMSD it does not grow with protein
  length. Above 0.5 means the same fold; below about 0.17 is what two unrelated
  structures give.
* **Per-chain RMSD.** HA1 fitted on its own and HA2 fitted on its own. If each
  is fine (low) but the overall RMSD is high, the pieces are right but packed
  wrongly against each other.
* **pLDDT.** ESM3's own confidence per residue: above 90 very confident, 70 to
  90 confident, 50 to 70 low, below 50 roughly a guess. The script also gives
  the Spearman correlation between pLDDT and each residue's deviation: negative
  means the model is least confident exactly where it is wrong, which is what
  you want from a confidence score.

What to expect for HA: our earlier decodes give HA0 a mean pLDDT around 50 to 60
(DEVLOG 2026-11-02/04), so expect a large RMSD and a low TM-score. That is the
point of the comparison: it turns "the decoder does not fold HA" into a number
against the real structure. To see what a *good* result looks like, run the
positive control in section 3c too.

## 3. Run it on astaroth

Paste these from the repository's `softmax/monte_carlo` folder (adjust the first
`cd` to where your clone lives).

### 3a. Get the experimental structure

```bash
cd ~/my_extendedSampling/softmax/monte_carlo
mkdir -p structure_compare
wget -O structure_compare/2HMG.pdb https://files.rcsb.org/download/2HMG.pdb
grep -c '^ATOM' structure_compare/2HMG.pdb     # a few thousand lines: download worked
```

### 3b. Predict HA0 with ESM3 and save it as .pdb (GPU)

```bash
nohup python -u decode_to_pdb.py --ref zero_polymer --num-steps 32 \
    --out structure_compare/ha0_pred.pdb > structure_compare/decode_ha0.log 2>&1 &
tail -f structure_compare/decode_ha0.log          # Ctrl-C stops watching, not the job
```

It prints the mean pLDDT and Rg and ends with `Saved -> ...`. `--num-steps` is the same knob as in
`decode_controls.py` (8, 32, 128 were used there).

### 3b'. The G-linked trimer (GPU)

The trimer is not in `references.py`; `decode_trimer.py` builds it on the fly
from `zero_polymer` (residues 17-529, i.e. the python slice `16:529`, three
copies joined by 25 glycines, 1589 residues). `decode_to_pdb.py` does the same
and writes the copies as chains A, B, C (linkers decoded but not written):

```bash
nohup python -u decode_to_pdb.py --ref zero_polymer --residues 17:529 --copies 3 --linker 25 \
    --num-steps 32 --out structure_compare/ha_trimer_pred.pdb \
    > structure_compare/decode_trimer.log 2>&1 &
python compare_structures.py --pred structure_compare/ha_trimer_pred.pdb \
    --exp structure_compare/2HMG.pdb --out-prefix structure_compare/trimer_vs_2HMG
```

`compare_structures.py` puts 2HMG's three HA1/HA2 pairs on the three predicted
chains and tries every order of the copies, keeping the lowest RMSD. The full
walk-through, including calling `write_pdb()` from your own code, is in
`softmax/docs/structure_comparison.pdf`.

### 3c. Optional positive control: a protein ESM3 does fold

Maltose-binding protein (UniProt P0AEX9, mature chain, 370 aa) decoded at pLDDT
0.93 in `decode_controls.py`, and PDB 1ANF is its crystal structure. Same two
steps:

```bash
wget -O structure_compare/1ANF.pdb https://files.rcsb.org/download/1ANF.pdb
nohup python -u decode_to_pdb.py --uniprot P0AEX9 --num-steps 32 \
    --out structure_compare/mbp_pred.pdb > structure_compare/decode_mbp.log 2>&1 &
```

### 3d. Compare (CPU, a few seconds)

```bash
python compare_structures.py --pred structure_compare/ha0_pred.pdb \
    --exp structure_compare/2HMG.pdb --out-prefix structure_compare/ha0_vs_2HMG \
    | tee structure_compare/ha0_vs_2HMG.txt

python compare_structures.py --pred structure_compare/mbp_pred.pdb \
    --exp structure_compare/1ANF.pdb --out-prefix structure_compare/mbp_vs_1ANF \
    | tee structure_compare/mbp_vs_1ANF.txt
```

Read the output top to bottom:

1. **Which experimental chains were used.** For 2HMG it should take one HA1
   chain (about 320 residues) and one HA2 chain (about 175), and skip the other
   copies as "symmetry copy". Check the identity: it tells you how close the
   2HMG strain is to our sequence. If the automatic choice looks wrong, force
   it with `--exp-chains A,B`.
2. **Unpaired residues.** The signal peptide, the transmembrane anchor and the
   cytoplasmic tail are not in the crystal, so they cannot be compared and are
   left out of every number.
3. **Global comparison**, **per-chain**, **pLDDT** and **largest deviations**, as
   explained in section 2.

Files written next to the prefix:

* `ha0_vs_2HMG_superposed.pdb`: the prediction already moved on top of 2HMG
* `ha0_vs_2HMG_per_residue.csv`: per residue pLDDT, matching 2HMG residue, deviation
* `ha0_vs_2HMG_per_residue.png`: deviation (blue) and pLDDT (red) along the sequence; dashed line = boundary between experimental chains (HA1 | HA2)
* `ha0_vs_2HMG_view.pml` / `.cxc`: ready-made PyMOL / ChimeraX sessions

## 4. Look at them

A viewer draws 3D graphics, so it needs a graphical session on the machine that
runs it. astaroth has a screen, so there are three routes and the one to pick
depends on where you are sitting:

**At astaroth's keyboard.** Install the viewer there and open the files in
place. Nothing to copy.

**Over ssh, no display.** Either render the picture on astaroth and copy only
the image back, which needs no graphics session at all:

```bash
chimerax --offscreen --exit --cmd "open na_head_vs_2HU0_view.cxc ; \
    save na_head.png width 2000 supersample 3"
```

(`--offscreen` needs a build with OSMesa support, which the Linux ChimeraX
builds normally have; if yours does not it says so immediately rather than
failing halfway.)

Or copy the folder to your laptop and look at it there (run this from a
terminal on the laptop; replace the login and path with yours):

```bash
scp -r marco.sanchini@astaroth:~/my_project/extendedSampling/softmax/monte_carlo/structure_compare .
cd structure_compare
```

Copy the whole folder, not single files: the `.cxc` and `.pml` scripts refer to
the other files by relative path.

`ssh -X` forwarding also works, but it pushes every frame over the network, so
rotating a structure is slow enough to be unpleasant. Use it only to check that
a file opens.

Pick one viewer. All are free.

### Option A: ChimeraX (recommended to start with)

Install from <https://www.cgl.ucsf.edu/chimerax/download.html> (free for
academic use). Then either:

**The quick way.** `File > Open...` and choose `ha0_vs_2HMG_view.cxc`
(or `chimerax ha0_vs_2HMG_view.cxc` from the folder). You get 2HMG in grey and
the prediction coloured by pLDDT (AlphaFold colours: dark blue > 90, light blue
70-90, yellow 50-70, orange < 50), already superimposed.

**By hand**, to learn the commands. Type them in the command line at the bottom
of the window:

```
open 2HMG.pdb                       # becomes model #1
open ha0_pred.pdb                   # becomes model #2
delete solvent                      # remove the crystal waters
delete #1 & ~/A,B                   # keep one protomer: HA1 = chain A, HA2 = chain B (*)
matchmaker #2 to #1/A               # superimpose the prediction on HA1; prints an RMSD
color #1 lightgray
color bfactor #2 palette alphafold  # colour the prediction by pLDDT
view                                # fit everything in the window
```

(*) Use the chains that `compare_structures.py` reported as "used"; A and B are
what it is expected to pick for 2HMG, but trust its output over this guide.

`matchmaker` can also be reached with `Tools > Structure Analysis > Matchmaker`.
It reports two RMSDs in the log: one over "pruned" atom pairs (its outlier
rejection, like the core RMSD) and one "across all pairs" (like the script's
global RMSD). Because it fits on HA1 alone, its numbers differ somewhat from
the script's, which fits HA1 + HA2 together.

Useful extras: hover over a residue to see its name and number; `label #2
residues` shows numbers; `cartoon hide #1` toggles one structure; `save
picture.png width 2000 supersample 3` saves an image for the thesis.

### Option B: PyMOL

Open-source build: `conda install -c conda-forge pymol-open-source`
(Schrödinger's PyMOL needs a free educational licence). Then either open
`ha0_vs_2HMG_view.pml` with `File > Run Script...` (or `pymol ha0_vs_2HMG_view.pml`
from the folder), or type at the `PyMOL>` prompt:

```
load 2HMG.pdb, exp
load ha0_pred.pdb, pred
remove solvent
remove exp and not chain A+B
align pred and name CA, exp and name CA, cycles=0    # all-residue RMSD, no outlier rejection
align pred and name CA, exp and name CA              # PyMOL default: drops outliers, like core RMSD
hide everything
show cartoon
color grey70, exp
spectrum b, red_white_blue, pred, minimum=30, maximum=90   # red = low pLDDT, blue = high
orient
```

The RMSD is printed in the log line `Executive: RMSD = ... (N to N atoms)`.
From the GUI the same fit is the `A` (Action) button next to `pred` >
`align` > `to molecule` > `exp`.

### Option C: in the browser, nothing to install

Go to <https://molstar.org/viewer/>, drag both `2HMG.pdb` and
`ha0_vs_2HMG_superposed.pdb` into the window (use the superposed file, since
Mol* will not fit them for you). To colour by pLDDT, click the prediction in
the right-hand panel and set its colour theme to "Uncertainty/Disorder", which
reads the B-factor column.

## 5. Things that trip people up

* **Numbering differs.** The prediction is numbered 1-566 along HA0 including the
  signal peptide; 2HMG numbers HA1 and HA2 separately in the H3 convention. The
  script pairs residues by sequence alignment, not by number, and the CSV lists
  both numbers side by side.
* **One chain against two.** ESM3 gives HA0 as one chain; the crystal is cleaved
  into HA1 + HA2. That is why the script maps several experimental chains onto
  the one predicted chain, and why ChimeraX's `matchmaker` (one chain to one
  chain) gives a different number.
* **Monomer against trimer.** 2HMG is a trimer and HA's stalk helices are held in
  place by the other two copies. A single-chain prediction has no partners, so
  part of any disagreement is quaternary, not a decoder error. The per-chain
  RMSD helps separate the two.
* **Step count.** HA0's decode changes with `--num-steps` (Rg 45 to 70 Å between
  8 and 128 steps, DEVLOG 2026-11-04), so state the step count with every RMSD,
  or run two (`--num-steps 8` and `32`, different `--out` names) and report both.
* **A different strain.** If the identity line says well below 100%, the
  experimental HA differs from ours in sequence. Small differences barely move
  the backbone, but say so when you report the number.

## 6. The neuraminidase candidate (added 2026-11-06)

HA0 failed the decode gate, so the sampling target is being chosen again
(`../docs/target_selection.md`). The first candidate is the influenza
neuraminidase catalytic head, and the supervisor supplied an experimental
structure for it: 2HU0 chain A, stripped to one chain, kept at
`../code/bridge_test/pdb/na_head_2HU0_monomer.pdb`. Its sequence is
`references.py` key `na_head_2hu0` (385 aa).

The comparison is easier than HA's in every way that mattered there: one
predicted chain against one experimental chain, no cleavage, no trimer, and no
waters or ligands left in the file. So there is nothing to strip and
`matchmaker #2 to #1` needs no chain selector.

```bash
python decode_to_pdb.py --ref na_head_2hu0 --num-steps 32 \
    --out structure_compare/na_head_pred.pdb

python compare_structures.py \
    --pred structure_compare/na_head_pred.pdb \
    --exp ../code/bridge_test/pdb/na_head_2HU0_monomer.pdb \
    --out-prefix structure_compare/na_head_vs_2HU0
```

Reading the result:

* **The numbers.** pLDDT around 0.9 with a low RMSD means the target passes the
  gate and sampling can start on it; pLDDT near HA0's 0.5 means drop it and move
  to the next candidate (cytochrome P450). HA never had an RMSD check at all,
  because no structure of our exact construct was in hand, so this candidate is
  being judged on more evidence than HA was.
* **The picture.** A six-bladed beta-propeller is a pinwheel: rotate until you
  look down the central axis and six four-stranded sheets should fan out around
  it. A prediction that is compact but has no pinwheel has failed even if its
  RMSD is not catastrophic, so the visual check carries information the numbers
  do not.
* **Step count.** Run `--num-steps 8` and `128` as well, to different `--out`
  names. A confident decode barely moves between step counts; HA0 wandered from
  Rg 45 to 70 A, which was the first sign something was wrong.

Specific traps for this pair, in addition to section 5:

* **Numbering.** The crystal is numbered 83-468, with five gaps totalling six
  unobserved loop residues and insertion codes 169A and 412A-D; the prediction
  is numbered 1-385. The offset is therefore not a constant 82 and changes at
  every gap. Take the pairing from `na_head_vs_2HU0_per_residue.csv`, which
  lists both numbers side by side, and never compute it by hand.
* **The reference is the observed sequence, not the construct's.** Those six
  unobserved residues are missing from `na_head_2hu0`, which closes up the gaps.
  That is acceptable for the gate, since all six are in loops, but before any
  sampling run or K-site comparison the reference has to be replaced by the true
  sequence, from 2HU0's SEQRES (`../code/tests/pdb_to_reference.py --use-seqres`
  on the unstripped file) or from the strain's UniProt entry
  (`../code/tests/add_reference.py`).
* **A tetramer, but not HA's problem.** Native NA is a homotetramer with a Ca2+
  site on the fourfold axis, so a monomer prediction is missing partners here
  too. The difference from HA is that the head's fold is entirely intra-chain, a
  self-contained beta-propeller, whereas HA2's long helix exists only as part of
  the trimeric coiled coil. Expect the interface surface to be the loosest part
  of the prediction, and flag interface residues before using the per-site
  entropy comparison, because they are conserved for quaternary reasons a
  single-chain energy cannot see.
