# pten_variants

Outputs of `../pten_variants.py` (reading: `../../docs/pten_variants.md`). Inputs are the
three files in `databases/` pushed on 2026-11-18 (cBioPortal TCGA PanCancer, ClinVar,
gnomAD); nothing here was downloaded.

- `clean.log`, `*_missense_clean.tsv`, `snv_reachability.tsv`: `clean`
- `merge.log`, `pten_variants_merged.tsv` (one row per protein change), `compound_sequences.tsv`,
  `variants_overview.png`: `merge`
- `profiles.log`, `kset_profiles.tsv`, `kset_enrichment.tsv`, `kset_density_correlations.tsv`,
  `kset_enrichment.png`, `kset_profiles/<category>.png`: `profiles`
- `rank.log`, `pten_variants_ranked.tsv`, `rank_auc.tsv`, `rank_percentiles.tsv`,
  `rank_by_category.png`: `rank`
- `compound_scores.csv`, `e2_repeat.csv` (GPU, `score`) and `compound_additivity.*` (`rank` after it):
  not yet produced.
