"""Put HuSC into the residue-level comparison tables (Fig 3, Fig 4, Figs S3-S4).

The two input tables hold the other per-residue scores (COSMIS, MTR3D, MTR, RVIS, pLI, missense Z, ConSurf, phyloP, GERP++,
phastCons) for all residues and for the ClinVar pathogenic/benign residues. This script fills their HuSC column from the HuSC
outputs, matched on uniprot_id + uniprot_pos, and leaves every other column unchanged (residues without a HuSC score get an
empty value). Other HuSC columns (HuSC_<radius>) are dropped from the ClinVar table.

usage: python add_husc_scores.py <HuSC output dir> <all-residue table.csv> <ClinVar table.csv> <out dir>
writes all_conservation_scores_with_HuSC_2026-10.csv and context_clinvar_with_HuSC_2026-10.csv"""
import csv, os, sys

husc_dir, all_in, clinvar_in, out_dir = sys.argv[1:]
cache = {}


def husc_of(u):
    if u not in cache:
        m, path = {}, f"{husc_dir}/{u}_husc.tsv"
        if os.path.exists(path):
            for r in csv.DictReader(open(path), delimiter="\t"):
                m[r["uniprot_pos"]] = r["HuSC"]
        cache[u] = m
    return cache[u]


def fill(src, dst, drop=()):
    n = found = 0
    with open(src) as f, open(dst, "w", newline="") as fo:
        rd = csv.DictReader(f)
        cols = [c for c in rd.fieldnames if c not in drop]
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rd:
            pos = r["uniprot_pos"][:-2] if r["uniprot_pos"].endswith(".0") else r["uniprot_pos"]
            r["HuSC"] = husc_of(r["uniprot_id"]).get(pos, "")
            n += 1; found += r["HuSC"] != ""
            w.writerow(r)
    print(f"{os.path.basename(dst)}: {n:,} rows, HuSC for {found:,}")


fill(all_in, f"{out_dir}/all_conservation_scores_with_HuSC_2026-10.csv")
with open(clinvar_in) as f:
    radius_cols = [c for c in next(csv.reader(f)) if c.startswith("HuSC_")]
fill(clinvar_in, f"{out_dir}/context_clinvar_with_HuSC_2026-10.csv", drop=radius_cols)
