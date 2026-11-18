"""Inputs for the human-specific constrained (HSC) residues (Fig 4, Fig. S4).

HSC residue: HuSC < -2.326 and ConSurf > 3, excluding residues with low local gnomAD coverage. Local coverage (rel_AN) is the
median allele number of the transcript's gnomAD variants within +-15 codons, divided by the transcript's median allele number;
residues with rel_AN < 0.5 are excluded because the null model is calibrated per transcript, so poorly covered regions look
constrained.

usage: python hsc_tables.py <all_conservation_scores_with_HuSC_2026-10.csv> <HuSC output dir> <UniProt proteome fasta.gz>
                            <gnomad_variant_counts_hg38.json[.gz]> <out dir>
writes uniprot_gene_names_2026-10.tsv (UniProt accession -> gene symbol, fills gene names missing from the table),
       hsc_threshold_inputs_2026-10.tsv.gz (every residue: HuSC, synonymous score, ConSurf, gene, rel_AN),
       hsc_low_coverage_residues_2026-10.tsv (HSC-threshold residues removed by the coverage filter)"""
import gzip, re, sys
import numpy as np
import pandas as pd
from scipy.special import ndtri
from husc import data

table, husc_dir, uniprot_fasta, gnomad, out_dir = sys.argv[1:]
N_PERM, HUSC_MAX, CONSURF_MIN, WINDOW, MIN_REL_AN = 10000, -2.326, 3, 15, 0.5

gene = {}
with gzip.open(uniprot_fasta, "rt") as h:
    for line in h:
        if line.startswith(">"):
            m = re.search(r" GN=(\S+)", line)
            if m:
                gene[line.split("|")[1]] = m.group(1)
gene = pd.Series(gene, name="gene_name").rename_axis("uniprot_id")
gene.to_csv(f"{out_dir}/uniprot_gene_names_2026-10.tsv", sep="\t")

cs = pd.read_csv(table, usecols=["uniprot_id", "uniprot_pos", "HuSC", "consurf", "gene_name"])
cs["gene_name"] = cs.gene_name.fillna(cs.uniprot_id.map(gene))
cs = cs.replace([np.inf, -np.inf], np.nan).dropna(subset=["HuSC", "consurf"])
cs = cs.sort_values(["uniprot_id", "uniprot_pos"]).reset_index(drop=True)

syn, enst = [], {}
for u in cs.uniprot_id.unique():
    f = pd.read_csv(f"{husc_dir}/{u}_husc.tsv", sep="\t", usecols=["enst_id", "uniprot_pos", "syn_p_value"])
    enst[u] = f.enst_id.iloc[0]
    syn.append(f.assign(uniprot_id=u)[["uniprot_id", "uniprot_pos", "syn_p_value"]])
cs = cs.merge(pd.concat(syn), on=["uniprot_id", "uniprot_pos"], how="left")
cs["syn_z"] = ndtri(np.clip(cs.syn_p_value, 0.5 / (N_PERM + 1), (N_PERM + 0.5) / (N_PERM + 1)))

cov = {}
for e, v in data.iter_variants(gnomad, set(enst.values())):
    if len(v.pos):
        o = np.argsort(v.pos)
        cov[e] = (v.pos[o], v.an[o], np.median(v.an))
rel = np.full(len(cs), np.nan)
for u, g in cs.groupby("uniprot_id"):
    c = cov.get(enst[u])
    if c is None:
        continue
    pos, an, median_an = c
    p = g.uniprot_pos.values
    lo, hi = np.searchsorted(pos, p - WINDOW), np.searchsorted(pos, p + WINDOW, side="right")
    rel[g.index] = [np.median(an[a:b]) / median_an if b > a else np.nan for a, b in zip(lo, hi)]
cs["rel_AN"] = rel

at_cut = cs[(cs.HuSC < HUSC_MAX) & (cs.consurf > CONSURF_MIN)]
at_cut[at_cut.rel_AN < MIN_REL_AN][["uniprot_id", "uniprot_pos", "HuSC", "consurf", "rel_AN"]].to_csv(
    f"{out_dir}/hsc_low_coverage_residues_2026-10.tsv", sep="\t", index=False)
keep = at_cut[~(at_cut.rel_AN < MIN_REL_AN) & at_cut.gene_name.notna()]
print(f"HSC: {len(keep):,} residues in {keep.gene_name.nunique():,} genes ({int((at_cut.rel_AN < MIN_REL_AN).sum())} removed for coverage)")
cs = cs.dropna(subset=["gene_name"])
cs[["uniprot_id", "uniprot_pos", "gene_name", "HuSC", "syn_z", "consurf", "rel_AN"]].to_csv(
    f"{out_dir}/hsc_threshold_inputs_2026-10.tsv.gz", sep="\t", index=False, compression="gzip")
