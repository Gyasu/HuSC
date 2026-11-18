# Analysis scripts

These scripts build the derived tables that the figure notebooks in `figures/` read. All tables are on Dryad
(https://doi.org/10.5061/dryad.1vhhmgr88), so the notebooks can be run without these scripts; they document how each
table was made. Run them from the repository root after `pip install -e .` (they use `husc.data` to read the gnomAD variants).

Inputs:
- HuSC outputs for the human proteome (`HuSC_8/`, one `<uniprot_id>_husc.tsv` per protein), and for the radius comparison
  runs at 5, 11 and 14 Å (`HuSC_5/`, `HuSC_11/`, `HuSC_14/`, same pipeline with `radius` changed in `config.json`).
- The per-residue tables of other conservation and constraint scores for all residues and for the ClinVar residues
  (from COSMIS, Li et al. 2022), the UniProt human proteome FASTA, the gnomAD variants file (see the main README), the AlphaFold
  DB v6 human models, and the ClinVar `variant_summary.txt.gz` release.
- Gene class lists from the COSMIS repository (`figure-code-data/fig_5c/data/`).

| script | writes | used for |
|---|---|---|
| `collect_scores.py` | `all_husc_values_2026-10.txt`, `all_husc_syn_values_2026-10.txt` | Fig 2A |
| `gene_class_scores.py` | `gene_set_scores_HuSC_8_2026-10.tsv` | Fig 2C |
| `add_husc_scores.py` | `all_conservation_scores_with_HuSC_2026-10.csv`, `context_clinvar_with_HuSC_2026-10.csv` | Figs 3, 4, S3, S4 |
| `hsc_tables.py` | `uniprot_gene_names_2026-10.tsv`, `hsc_threshold_inputs_2026-10.tsv.gz`, `hsc_low_coverage_residues_2026-10.tsv` | Figs 4, S4 |
| `zinc_finger_positions.py` | `znf_finger_positions_2026-10.tsv` | Fig 4D |
| `clinvar_variant_af.py` | `clinvar_variant_af_2026-10.tsv` | input to `clinvar_inheritance.py` |
| `clinvar_inheritance.py` | `clinvar_inheritance_2026-10.tsv` | Fig S3B |
| `radius_scan/collect_radius.py` | `radius_<r>.npz` | input to the other radius scripts |
| `radius_scan/score_histograms.py` | `S2_score_histograms_2026-10.csv` | Fig S2A |
| `radius_scan/radius_summary.py` | `S2_radius_summary_2026-10.csv`, `S2_clinvar_by_radius_2026-10.csv` | Figs S2B-C, S3C-D |
| `radius_scan/background_auroc.py` | `S2_background_auroc_by_radius_2026-10.csv` | Fig S2D |

Order: `collect_scores`, `gene_class_scores`, `add_husc_scores`, `hsc_tables`, `zinc_finger_positions`, `clinvar_variant_af`,
`clinvar_inheritance`, then the `radius_scan` scripts in the order listed. Each script's docstring gives its arguments.

The ESM2 tables (ProteinGym scores, wild-type NLL, Spearman components, ESM2-650M ClinVar scores) are computed in the
notebooks themselves, and the fine-tuning splits and adapters in `esm_finetuning/`.
