"""ClinVar residues with variant allele frequency and the inheritance mode of their gene (Fig. S3B).

inheritance: 'dominant/HI' (gene in the dominant or haploinsufficient list only), 'recessive' (recessive list only), 'both',
or 'unlisted'; gene lists as in gene_class_scores.py.

usage: python clinvar_inheritance.py <context_clinvar_with_HuSC_2026-10.csv> <clinvar_variant_af_2026-10.tsv>
                                     <gene_set_scores_HuSC_8_2026-10.tsv> <out.tsv>"""
import sys
import numpy as np
import pandas as pd

clinvar_table, variant_af, gene_classes, out = sys.argv[1:]
cv = pd.read_csv(clinvar_table).dropna(subset=["HuSC"])
af = pd.read_csv(variant_af, sep="\t")
cv = cv.merge(af[["uniprot_id", "uniprot_pos", "variant_af_max"]], on=["uniprot_id", "uniprot_pos"], how="left")
gc = (pd.read_csv(gene_classes, sep="\t", usecols=["uniprot_id", "class"]).drop_duplicates()
      .groupby("uniprot_id")["class"].apply(lambda s: ";".join(sorted(s))))
cv = cv.drop_duplicates(["uniprot_id", "uniprot_pos"])
g = cv.uniprot_id.map(gc).fillna("")
rec, dom = g.str.contains("Recessive"), g.str.contains("Dominant|Haplo")
cv["inheritance"] = np.select([rec & ~dom, dom & ~rec, rec & dom], ["recessive", "dominant/HI", "both"], "unlisted")
cols = ["uniprot_id", "uniprot_pos", "label", "variant_af_max", "inheritance", "HuSC", "consurf", "phylop", "cosmis", "mtr3d"]
cv[cols].to_csv(out, sep="\t", index=False)
print(f"{len(cv):,} residues;", cv.inheritance.value_counts().to_dict(), f"; AF missing for {cv.variant_af_max.isna().sum():,}")
