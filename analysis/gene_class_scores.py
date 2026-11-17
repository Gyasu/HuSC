"""HuSC of every residue of the proteins in each gene class (Fig 2C).

Gene class lists (UniProt IDs) are those used for COSMIS (Li et al. 2022; github.com/CapraLab/cosmis, figure 5c data):
olfactory receptors, non-essential, recessive, dominant, essential (cell lines) and ClinGen haploinsufficient genes.

usage: python gene_class_scores.py <HuSC output dir> <gene list dir> <out.tsv>"""
import csv, os, sys

husc_dir, list_dir, out = sys.argv[1:]
LISTS = {"Olfactory": "olfactory_receptor_uniprot", "Non essential": "non_essential_genes_uniprot",
         "Recessive": "all_recessive_uniprot", "Dominant": "all_dominant_uniprot",
         "Essential": "essential_genes_uniprot", "Haploinsufficient": "clingen_level3_haploinsufficient_uniprot"}
with open(out, "w") as fo:
    fo.write("uniprot_id\tHuSC\tclass\n")
    for cls, name in LISTS.items():
        ids = [r["uniprot_id"] for r in csv.DictReader(open(f"{list_dir}/{name}.tsv"), delimiter="\t")]
        n_genes = n_res = 0
        for u in ids:
            path = f"{husc_dir}/{u}_husc.tsv"
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                i = fh.readline().rstrip("\n").split("\t").index("HuSC")
                k = 0
                for line in fh:
                    fo.write(f"{u}\t{line.rstrip(chr(10)).split(chr(9))[i]}\t{cls}\n")
                    k += 1
            n_genes += k > 0
            n_res += k
        print(f"{cls:18s} {n_genes:5,} proteins {n_res:9,} residues")
