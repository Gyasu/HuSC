# HuSC — Human Spatial Constraint

**HuSC** (Human Spatial Constraint) is a computational framework for quantifying intraspecies constraint on missense variants by integrating population-scale human genetic variation with 3D protein structures. HuSC scores model the expected frequency of missense variation under neutral evolution and compare it to observed variation, accounting for both variation in mutational processes and 3D structural context.

We use HuSC to fine-tune protein language models (PLMs), improving their ability to predict variant effects, particularly by reducing bias toward wild-type sequences in regions that tolerate variation.

> **Associated publication:** Bajracharya G. and Capra J.A. "Fine-tuning protein language models on human spatial constraint improves variant effect prediction by reducing wild-type sequence bias." *bioRxiv* (2026).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gyasu/HuSC.git
cd HSC
pip install -e .
```

---

## Protein complexes

`husc/complex.py` computes HuSC scores for the subunits of a complex, following the
logic of COSMIS' `cosmis-complex`: contacts are searched across all mapped chains, so a
residue's neighborhood includes residues of the other subunits; observed variants,
possible sites and mutation rates are summed over that neighborhood using each
residue's own protein's data; and the permutation null is built per chain and summed
over the same cross-chain neighborhood.

```bash
python husc/complex.py -c config.json -i inputs/test_complex_input.txt -r 8
```

Each input line is `<complex_id> <structure_file> <chain>:<uniprot>[:<offset>] ...`
(PDB or mmCIF, optionally `.gz`; resolved against `pdb_dir`). `offset` is added to the
structure residue number to obtain the UniProt position; if omitted it is inferred
(0 for AlphaFold models). Chains whose sequence agrees with UniProt at <90% of residues
are skipped. Output is `<complex_id>_husc_complex.tsv` with the monomer columns plus
`complex_id`, `chain_id`, `num_intra_contacts`, `num_inter_contacts` and
`inter_chain_contacts`. Options: `--chains A,B` (score only these chains),
`--shared-permutations` (identical subunits share null draws), `--seed`.

---

## Allele-frequency distributions in the permutation null

The permutation null draws a minor allele frequency (MAF) for each simulated variant from the
transcript's empirical MAF distribution (`gnomad_prob_AFdist_exp.tsv`, columns `syn_dist` / `mis_dist`,
in percent). Earlier versions of `read_enst_mp_count_dist` silently discarded these lists (an `isdigit()`
check rejected decimals), so every variant was assigned a constant MAF of 0.0007 percent. This is now
fixed (confirmed with the method authors: the fixed behaviour is intended). Both the missense and the
synonymous permutations draw MAFs from the synonymous list (`syn_dist`): synonymous variation is the neutral
spectrum, consistent with the expected missense count being predicted from the synonymous regression.
`mis_dist` is read but not used. `husc/complex.py` follows the same behaviour (it shares `load_datasets`).

**The published HuSC scores (Dryad dataset) were generated with the constant MAF.** To regenerate
scores that match them, set `"use_af_dist": false` in the config JSON. The default (`true`) uses the
empirical distributions, as described in the manuscript's Methods.
