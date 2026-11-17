"""gnomAD allele frequency of the ClinVar variants behind each ClinVar residue (frequency-matched analyses, Fig. S3B).

For every residue of the ClinVar table with a HuSC score, the ClinVar missense SNVs at that UniProt residue (matched by gene
symbol and protein position in variant_summary.txt.gz, GRCh38; wild-type residue must match) whose clinical significance
agrees with the residue's label are collected, and variant_af_max is the largest gnomAD v2.1.1 AF of those variants on the
residue's transcript (0 if absent from gnomAD; NaN if no agreeing ClinVar variant).

usage: python clinvar_variant_af.py <context_clinvar_with_HuSC_2026-10.csv> <ClinVar variant_summary.txt.gz>
                                    <UniProt proteome fasta.gz> <gnomad_variant_counts_hg38.json[.gz]> <out.tsv>"""
import csv, gzip, re, sys
from collections import defaultdict
from husc import data

clinvar_table, variant_summary, uniprot_fasta, gnomad, out = sys.argv[1:]
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
       "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
PCH = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")
is_path = lambda s: "athogenic" in s and "onflicting" not in s and "enign" not in s
is_ben = lambda s: "enign" in s and "onflicting" not in s and "athogenic" not in s

gene_of = {}
with gzip.open(uniprot_fasta, "rt") as h:
    for line in h:
        if line.startswith(">"):
            m = re.search(r" GN=(\S+)", line)
            if m:
                gene_of[line.split("|")[1]] = m.group(1)

rows = [r for r in csv.DictReader(open(clinvar_table)) if r["HuSC"]]
wanted = {(gene_of[r["uniprot_id"]], int(r["uniprot_pos"])) for r in rows if r["uniprot_id"] in gene_of}

by_res = defaultdict(list)          # (gene, position) -> [(ref, alt, clinical significance)]
with gzip.open(variant_summary, "rt") as h:
    ix = {c: i for i, c in enumerate(h.readline().lstrip("#").rstrip("\n").split("\t"))}
    for line in h:
        f = line.rstrip("\n").split("\t")
        if f[ix["Assembly"]] != "GRCh38" or f[ix["Type"]] != "single nucleotide variant":
            continue
        m = PCH.search(f[ix["Name"]])
        if not m or m.group(1) not in AA3 or m.group(3) not in AA3 or m.group(1) == m.group(3):
            continue
        key = (f[ix["GeneSymbol"]], int(m.group(2)))
        if key in wanted:
            by_res[key].append((AA3[m.group(1)], AA3[m.group(3)], f[ix["ClinicalSignificance"]]))

variants = data.load_variants(gnomad, {r["enst_id"] for r in rows})
with open(out, "w") as fo:
    fo.write("uniprot_id\tuniprot_pos\tlabel\tvariants\tvariant_af_max\n")
    for r in rows:
        u, pos, label = r["uniprot_id"], int(r["uniprot_pos"]), int(r["label"])
        agree = is_path if label == 1 else is_ben
        hits = [(ref, alt) for ref, alt, sig in by_res.get((gene_of.get(u), pos), []) if ref == r["uniprot_aa"] and agree(sig)]
        v = variants.get(r["enst_id"])
        af = {}
        if v is not None:
            for p, ref, alt, ac, an in zip(v.pos, v.ref, v.alt, v.ac, v.an):
                if an:
                    af[(int(p), chr(ref), chr(alt))] = ac / an
        afs = [af.get((pos, ref, alt), 0.0) for ref, alt in hits]
        fo.write(f"{u}\t{pos}\t{label}\t{';'.join(f'{a}{pos}{b}' for a, b in hits)}\t{f'{max(afs):.3g}' if afs else 'nan'}\n")
print(f"{len(rows):,} ClinVar residues")
