# pten_natural

Outputs of `../pten_k_sites.py` (see its docstring, and `../../docs/pten_k_sites.md` for the reading).
`raw/` is filled by `python pten_k_sites.py fetch` and is not committed.

Sources, all per-site columns here are derived from them:

- Abundance: Matreyek et al 2018, Nat Genet 50:874 (PMID 29785012), github.com/FowlerLab/VAMPseq, and
  Matreyek et al 2021, Genome Med 13:165, github.com/MatreyekLab/PTEN_composite.
  **TPMT/PTEN variant abundance scores are the copyright of the University of Washington**, made
  available for non-profit, non-commercial use only; the columns `abundance_*` and `rsa` are
  per-position derivatives of them and carry the same terms.
- Activity: Mighell et al 2018, AJHG 102:943 (PMC5986715), table as redistributed in PTEN_composite.
- Natural alignments and HHsearch hits: OpenProteinSet (Ahdritz et al 2023), s3://openfold/pdb/1d5r_A
  and 2cm2_A, CC BY 4.0.
