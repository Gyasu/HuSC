"""Per-radius summary (Fig. S2B-C) and the ClinVar residues scored at every radius (Fig. S2D, Fig. S3C-D).

usage: python radius_summary.py <dir with radius_<r>.npz> <clinvar_inheritance_2026-10.tsv>
writes S2_radius_summary_2026-10.csv and S2_clinvar_by_radius_2026-10.csv there"""
import sys
import numpy as np
import pandas as pd

S, CLIN = sys.argv[1], sys.argv[2]
RADII, CUT = [5, 8, 11, 14], -2.326
Z = {r: np.load(f"{S}/radius_{r}.npz") for r in RADII}
rows = []
for r in RADII:
    z = Z[r]
    h, s, nc, obs, sp = z["husc"], z["syn"], z["nc"], z["mis_obs"], z["syn_p"]
    grid, ss = np.linspace(0, 1, 1001), np.sort(sp)
    ks = np.max(np.abs(np.searchsorted(ss, grid, side="right") / len(ss) - grid))
    rows.append(dict(radius=r, residues=len(h), proteins=len(np.unique(z["uid"])), contacts_mean=nc.mean(),
                     contacts_median=np.median(nc), zero_obs_pct=100 * np.mean(obs == 0), mis_median=np.median(h), mis_sd=h.std(),
                     syn_median=np.median(s), syn_sd=s.std(), mis_below_pct=100 * np.mean(h < CUT),
                     syn_below_pct=100 * np.mean(s < CUT), emp_fdr=np.mean(s < CUT) / np.mean(h < CUT),
                     syn_p05_pct=100 * np.mean(sp < 0.05), syn_ks=ks, mis_at_floor_pct=100 * np.mean(h <= -3.89)))
pd.DataFrame(rows).to_csv(f"{S}/S2_radius_summary_2026-10.csv", index=False)

c = pd.read_csv(CLIN, sep="\t")[["uniprot_id", "uniprot_pos", "label", "variant_af_max", "inheritance", "HuSC"]]
for r in RADII:
    z = Z[r]
    c = c.merge(pd.DataFrame({"uniprot_id": z["uid"], "uniprot_pos": z["pos"], f"HuSC_{r}": z["husc"]}),
                on=["uniprot_id", "uniprot_pos"], how="left")
assert np.allclose(c.HuSC, c.HuSC_8, atol=1e-3, equal_nan=True)
c.dropna(subset=[f"HuSC_{r}" for r in RADII]).to_csv(f"{S}/S2_clinvar_by_radius_2026-10.csv", index=False)
