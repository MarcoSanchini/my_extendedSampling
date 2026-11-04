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

astaroth has no screen, so copy the folder to your laptop first (from a terminal
on the laptop; replace `user` and the path with yours):

```bash
scp -r user@astaroth:~/my_extendedSampling/softmax/monte_carlo/structure_compare .
cd structure_compare
```

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
