# HuSC — Human Spatial Constraint

**HuSC** (Human Spatial Constraint) quantifies intraspecies constraint on missense variation by combining population-scale human genetic variation (gnomAD) with 3D protein structures (AlphaFold). For every residue it compares the missense variation observed in the residue's 3D neighborhood with what is expected under neutral mutation. The comparison accounts for the sequence context of mutation rates, CpG methylation, and the allele frequency spectrum.

We use HuSC to fine-tune protein language models (PLMs), improving their ability to predict variant effects, particularly by reducing bias toward wild-type sequences in regions that tolerate variation.

> **Associated publication:** Bajracharya G. and Capra J.A. "Fine-tuning protein language models on human spatial constraint improves variant effect prediction by reducing wild-type sequence bias." *bioRxiv* (2026).

**Just want the scores?** Precomputed HuSC scores for the human proteome are available on Zenodo (see [Data](#data)); you don't need to run the pipeline.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Data](#data)
- [Running on your own proteins or the whole proteome](#running-on-your-own-proteins-or-the-whole-proteome)
- [Output](#output)
- [Method](#method)
- [Rebuilding the derived inputs](#rebuilding-the-derived-inputs)
- [Repository layout](#repository-layout)

## Installation

Requires Python ≥ 3.9. The large gnomAD file is streamed rather than loaded whole, so the full proteome runs on a laptop with 8 GB of RAM.

```bash
git clone https://github.com/gyasu/HuSC.git
cd HuSC
pip install -e .
```

## Quick start

This scores three proteins (TP53, BRCA1, PTEN) in about a minute, plus download time.

**1. Download the data files** from Zenodo (see [Data](#data)) into `datafiles/`:

```
datafiles/
├── gnomad_variant_counts_hg38.json.gz
├── gnomad_prob_AFdist_exp.tsv.gz
├── Homo_sapiens.GRCh38.cds.all.fa.gz
└── UP000005640_9606.fasta.gz
```

The other inputs (UniProt → Ensembl mapping, CpG files) ship with the repository in `resources/`.

**2. Check `config.json`.** The defaults point to `datafiles/`, `resources/`, `structures/` and `outputs/HuSC_8/`, so no edits are needed if you use these folders. Gzipped and uncompressed versions of the data files both work; just make the file names in `config.json` match.

**3. Download the AlphaFold structures** for the proteins in the input file:

```bash
python -m husc.download_structures -c config.json -i inputs/example.txt
```

**4. Compute HuSC:**

```bash
python -m husc.main -c config.json -i inputs/example.txt -v
```

This writes one table per protein to `outputs/HuSC_8/` (e.g. `P04637_husc.tsv`); the `HuSC` column is the score. A log is written to `husc.log`, including a summary of any proteins that were skipped and why.

## Data

GitHub cannot store the large input files, so they are archived on Zenodo:

> **Zenodo:** https://doi.org/10.5281/zenodo.XXXXXXX *(to be added)*

| file | size | contents | source |
|---|---|---|---|
| `gnomad_variant_counts_hg38.json.gz` | | gnomAD v2.1.1 coding SNVs per Ensembl transcript: `[protein change, allele count, allele number]` | [COSMIS](https://github.com/CapraLab/cosmis) |
| `gnomad_prob_AFdist_exp.tsv.gz` | | per-transcript mutability, observed and expected variant counts, allele frequency lists | built by `husc.build_transcript_table` |
| `Homo_sapiens.GRCh38.cds.all.fa.gz` | 22 MB | Ensembl GRCh38 coding sequences | Ensembl |
| `UP000005640_9606.fasta.gz` | 8 MB | UniProt human reference proteome | UniProt |
| `HuSC_8_scores.tar.gz` | | precomputed HuSC scores for the human proteome (8 Å) | this pipeline |

**Variant data provenance.** The variants file comes from [COSMIS](https://github.com/CapraLab/cosmis) (Li et al. 2022). It holds per-transcript variant counts from gnomAD v2.1.1 (141,456 individuals), lifted over to GRCh38. Sites without a PASS filter flag were removed with vcftools, and only single-nucleotide variants were kept.

Shipped in this repository (`resources/`):

| file | contents |
|---|---|
| `uniprot_to_enst_mapping.json` | UniProt accession → Ensembl transcript IDs (UniProt ID Mapping service) |
| `enst_ids.txt` | transcript list used to build the transcript table |
| `cpg_syn_maf.txt` | pooled synonymous MAFs at CpG codons (built by `husc.prepare_cpg`) |
| `cpg_scale.json` | CpG mutation-rate factors by local CpG o/e, with a held-out fit report (built by `husc.prepare_cpg`) |

AlphaFold structures are downloaded separately (step 3 above, or the bulk archive below).

## Running on your own proteins or the whole proteome

**Input file.** One protein per line: a UniProt accession, optionally followed by an AlphaFold model name. If the model name is left out, the AlphaFold DB v6 model `AF-<accession>-F1-model_v6.pdb` is used. Lines starting with `#` are ignored.

```
P04637
P38398 AF-P38398-F1-model_v6.pdb
```

`inputs/human_proteome.txt` lists the 20,504 human proteins used in the paper.

**Structures for the whole proteome.** Rather than downloading ~20,000 files one by one, get the AlphaFold DB human proteome archive (5.2 GB) and keep the PDB files:

```bash
mkdir -p structures
curl -O https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar
tar -xf UP000005640_9606_HUMAN_v6.tar -C structures --wildcards '*.pdb.gz'
```

**Run.**

```bash
python -m husc.main -c config.json -i inputs/human_proteome.txt -l husc.log -v
```

- Expect about 1–2 s per protein on a laptop, plus about a minute to load the data.
- Proteins that already have an output file are skipped, so an interrupted run resumes with the same command. Use `-w/--overwrite` to recompute them.
- To parallelize, split the input file into parts and run one process per part with the same `config.json`. Each protein gets its own random seed (from `seed` and its UniProt ID), so results don't depend on how proteins are split or ordered.
- Proteins are skipped, with the reason logged, if they have no matching Ensembl transcript, no gnomAD data, or a missing or low-quality structure. AlphaFold DB splits proteins longer than 2,700 residues into fragments, and these don't match a transcript.

### Configuration (`config.json`)

| key | meaning |
|---|---|
| `ensembl_cds` | Ensembl CDS FASTA |
| `uniprot_pep` | UniProt proteome FASTA (used by `husc.prepare_cpg`) |
| `gnomad_variants` | gnomAD variants per transcript (JSON, `.gz` allowed) |
| `uniprot_to_enst` | UniProt → Ensembl transcript mapping (JSON) |
| `enst_mp_counts` | per-transcript table (TSV, `.gz` allowed) |
| `cpg_maf`, `cpg_scale` | CpG inputs |
| `output_dir` | where `<uniprot_id>_husc.tsv` files are written |
| `pdb_dir` | directory of AlphaFold models, named `<model name>.gz` |
| `radius` | contact radius in Å (default 8) |
| `seed` | base random seed (default 0) |

## Output

One tab-separated file per protein, one row per residue. *cs* = contact set, the residue plus all residues within the contact radius.

| column | meaning |
|---|---|
| `uniprot_id`, `enst_id` | protein and the Ensembl transcript used |
| `uniprot_pos`, `uniprot_aa` | residue position and amino acid |
| **`HuSC`** | **the score: negative = constrained** (less missense variation than expected), positive = more than expected; range about [−3.89, +3.89] |
| `mis_p_value` | mid-p permutation p-value of the missense test (probability of this little variation or less); `HuSC` = Φ⁻¹(`mis_p_value`) |
| `syn_p_value` | the same test on synonymous variants, a calibration check (roughly uniform under a well-calibrated null) |
| `seq_separations` | sequence offsets of the contacting residues |
| `num_contacts` | contact-set size, including the residue itself |
| `cs_mis_obs`, `cs_syn_obs` | observed missense / synonymous minor allele frequency (MAF) sum in the contact set |
| `cs_mis_exp`, `cs_syn_exp` | null mean of the missense / synonymous MAF sum |
| `mis_var_sites`, `syn_var_sites` | contact-set codons with ≥ 1 observed missense / synonymous variant |
| `total_mis_sites`, `total_syn_sites` | nonsynonymous / synonymous sites in the contact set |
| `cs_mis_poss`, `cs_syn_poss` | possible missense / synonymous SNVs in the contact set |
| `cs_mis_prob`, `cs_syn_prob` | summed missense / synonymous mutation rate (before CpG scaling) |
| `cs_gc_content` | GC content of the contact-set codons (±1 nt) |
| `enst_mis_obs`, `enst_syn_obs` | observed missense / synonymous variant counts in the transcript |
| `enst_mis_exp`, `enst_syn_exp` | expected counts after synonymous calibration (used by the null) |
| `plddt` | AlphaFold pLDDT of the residue |
| `uniprot_length` | protein length |

## Method

For each residue of an AlphaFold model, HuSC defines the **contact set**: the residue plus every residue whose Cβ atom (Cα for glycine) lies within 8 Å. The test statistic is the summed MAF of the gnomAD missense variants in the contact set. It is compared with a permutation null of neutral variation (10,000 permutations per transcript):

1. **Expected counts.** The transcript's expected synonymous and missense counts come from a linear fit of synonymous count on mutability across transcripts. Both are scaled by the transcript's synonymous observed/expected ratio, shrunk toward 1: `(syn_obs + 10) / (syn_exp + 10)`.
2. **Saturating placement.** Each possible SNV is observed at most once per permutation, with probability `p = 1 − exp(−β · rate)`. Here `rate` is its trinucleotide-context mutation rate and `β` is solved per transcript so that the probabilities sum to the expected count. Unlike linear allocation, this keeps highly mutable (CpG) sites from absorbing more than one variant.
3. **CpG methylation.** CpG transition rates are scaled by an empirical factor of the local CpG observed/expected ratio (±100 bp of CDS), so CpG-island-like, mostly unmethylated sites mutate more slowly. Factors are fit on synonymous variation only (`resources/cpg_scale.json`).
4. **Allele frequencies.** Each placed CpG SNV draws a MAF from a pooled list of synonymous MAFs at CpG codons, because recurrent mutation raises CpG allele frequencies. Other SNVs draw from the transcript's own non-CpG synonymous MAFs.

The score is the probit of the mid-p permutation p-value:

```
p    = (#{null < obs} + 0.5 · #{null = obs} + 0.5) / (N + 1)
HuSC = Φ⁻¹(p)
```

Because p is the probability of observing this little variation or less, negative HuSC means constraint. With N = 10,000 permutations, HuSC is bounded at about ±3.89.

## Rebuilding the derived inputs

You only need these to change the gene set or the gnomAD release; the published files are on Zenodo and in `resources/`.

```bash
# per-transcript mutability, observed/expected counts and allele frequency lists (~2 min)
python -m husc.build_transcript_table -c config.json -t resources/enst_ids.txt

# pooled CpG MAF list and CpG rate factors, from synonymous variation of the given genes (~10 min)
python -m husc.prepare_cpg -c config.json -i inputs/human_proteome.txt
```

## Repository layout

```
husc/
├── main.py                    # HuSC pipeline
├── null_model.py              # permutation null (calibration, saturating placement, CpG handling)
├── data.py                    # input loading and transcript selection
├── download_structures.py     # AlphaFold DB downloads
├── build_transcript_table.py  # builds the per-transcript table
├── prepare_cpg.py             # builds the CpG inputs
└── utils/                     # sequence, structure and mutation-rate helpers
resources/                     # small inputs shipped with the code
inputs/                        # example and proteome input lists
helper_scripts/, figures/, esm_finetuning/   # analyses and figures for the paper
```
