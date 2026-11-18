"""Proteome-wide score lists for Fig 2A: the missense HuSC of every scored residue and its synonymous counterpart
(probit of the synonymous mid-p, same null and neighborhood).

usage: python collect_scores.py <HuSC output dir> <out dir>
writes all_husc_values_2026-10.txt and all_husc_syn_values_2026-10.txt (one value per line, same residue order)"""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.special import ndtri

N_PERM = 10000
husc_dir, out_dir = sys.argv[1], sys.argv[2]
files = sorted(f for f in glob.glob(f"{husc_dir}/*_husc.tsv") if not os.path.basename(f).startswith("._"))
d = pd.concat([pd.read_csv(f, sep="\t", usecols=["HuSC", "syn_p_value"], dtype=str) for f in files], ignore_index=True)
mis = pd.to_numeric(d.HuSC, errors="coerce")
# syn_p_value is written with 4 significant digits, so the largest mid-p, (N + 0.5) / (N + 1), rounds to 1.000;
# clip to the permutation bounds so those residues get the same bound as HuSC
syn = ndtri(np.clip(d.syn_p_value.astype(float).to_numpy(), 0.5 / (N_PERM + 1), (N_PERM + 0.5) / (N_PERM + 1)))
with open(f"{out_dir}/all_husc_values_2026-10.txt", "w") as h:
    h.writelines(v + "\n" for v in d.HuSC[mis.notna()])
np.savetxt(f"{out_dir}/all_husc_syn_values_2026-10.txt", syn, fmt="%.3f")
print(f"{len(files):,} proteins, {mis.notna().sum():,} missense and {len(syn):,} synonymous scores")
