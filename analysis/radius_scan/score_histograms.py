"""Histogram counts (bin width 0.1) of missense HuSC and the synonymous score at each radius (Fig. S2A).

usage: python score_histograms.py <dir with radius_<r>.npz>      writes S2_score_histograms_2026-10.csv there"""
import sys
import numpy as np
import pandas as pd

S = sys.argv[1]
edges = np.round(np.arange(-4.0, 4.0001, 0.1), 2)
rows = []
for r in [5, 8, 11, 14]:
    z = np.load(f"{S}/radius_{r}.npz")
    for kind, v in [("Missense", z["husc"]), ("Synonymous", z["syn"])]:
        c, _ = np.histogram(v, edges)
        assert c.sum() == len(v), (r, kind)
        rows += [dict(radius=r, kind=kind, bin_left=edges[i], bin_right=edges[i + 1], count=int(c[i])) for i in range(len(c))]
pd.DataFrame(rows).to_csv(f"{S}/S2_score_histograms_2026-10.csv", index=False)
