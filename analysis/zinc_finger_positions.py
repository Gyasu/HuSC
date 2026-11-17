"""ConSurf > 3 residues inside C2H2 zinc fingers and their position in the finger (Fig 4D).

Fingers are matches of C.{2,4}C.{12}H.{3,5}H in the AlphaFold DB v6 model sequence (non-overlapping). offset = position
relative to the first zinc-binding histidine (H1); the recognition helix is offsets -7 to -1. hsc = human-specific constrained
(HuSC < -2.326 and ConSurf > 3, low-coverage residues excluded).

usage: python zinc_finger_positions.py <all_conservation_scores_with_HuSC_2026-10.csv> <hsc_low_coverage_residues_2026-10.tsv>
                                       <uniprot_gene_names_2026-10.tsv> <AlphaFold structure dir> <out.tsv>"""
import gzip, os, re, sys
import numpy as np
import pandas as pd
from Bio.SeqUtils import seq1

table, low_cov, gene_names, structures, out = sys.argv[1:]
cs = pd.read_csv(table, usecols=["uniprot_id", "uniprot_pos", "HuSC", "consurf"]).replace([np.inf, -np.inf], np.nan).dropna()
low = pd.read_csv(low_cov, sep="\t")
low = set(zip(low.uniprot_id, low.uniprot_pos))
cs["elig"] = cs.consurf > 3
cs["hsc"] = cs.elig & (cs.HuSC < -2.326) & ~pd.Series([k in low for k in zip(cs.uniprot_id, cs.uniprot_pos)], index=cs.index)
gene = pd.read_csv(gene_names, sep="\t").set_index("uniprot_id").gene_name

finger = re.compile(r"C.{2,4}C.{12}H.{3,5}H")
rows = []
for u in cs.loc[cs.elig, "uniprot_id"].unique():
    path = f"{structures}/AF-{u}-F1-model_v6.pdb.gz"
    if not os.path.exists(path):
        continue
    seq = {}
    with gzip.open(path, "rt") as h:
        for line in h:
            if line.startswith("ATOM") and line[12:16] == " CA ":
                seq[int(line[22:26])] = seq1(line[17:20]) or "X"
    s = "".join(seq.get(i, "X") for i in range(1, max(seq) + 1))
    for m in finger.finditer(s):
        h1 = m.start() + re.match(r"C.{2,4}C.{12}", m.group()).end()      # 0-based index of H1
        rows += [(u, i + 1, i - h1) for i in range(m.start(), m.end())]
fz = pd.DataFrame(rows, columns=["uniprot_id", "uniprot_pos", "offset"])
x = cs.merge(fz, on=["uniprot_id", "uniprot_pos"])
x = x[x.elig].astype({"offset": int}).assign(gene=lambda d: d.uniprot_id.map(gene))
x[["uniprot_id", "gene", "uniprot_pos", "offset", "HuSC", "consurf", "hsc"]].to_csv(out, sep="\t", index=False)
print(f"{len(x):,} ConSurf > 3 finger residues in {x.uniprot_id.nunique()} proteins; HSC {int(x.hsc.sum())}")
