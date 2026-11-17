"""ClinVar pathogenic residues vs background residues of the same proteins, by radius (Fig. S2D).

Background = up to 100 random non-ClinVar residues per protein (default_rng(42)) from proteins with >= 1 pathogenic residue;
labels do not use allele frequency. AUROC of -HuSC over all residues, and structure-matched (within strata of 8 Å contact-count
quartile x pLDDT bin <50/50-70/70-90/>90, weighted by pathogenic count, strata with >= 10 pathogenic). Differences vs 8 Å with
1,000 paired protein bootstrap draws.

usage: python background_auroc.py <dir with radius_<r>.npz and S2_clinvar_by_radius_2026-10.csv>
writes S2_background_auroc_by_radius_2026-10.csv there"""
import sys, numpy as np, pandas as pd
from scipy.stats import rankdata
S = sys.argv[1]; RADII = [5, 8, 11, 14]
def auroc(y, s):
    rk = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
clin = pd.read_csv(f'{S}/S2_clinvar_by_radius_2026-10.csv')
path = clin[clin.label == 1][['uniprot_id', 'uniprot_pos']]
prots = set(path.uniprot_id); clin_keys = set(zip(clin.uniprot_id, clin.uniprot_pos))
Z = {r: np.load(f'{S}/radius_{r}.npz') for r in RADII}
z8 = Z[8]; sel = np.isin(z8['uid'], list(prots))
base = pd.DataFrame({'uniprot_id': z8['uid'][sel], 'uniprot_pos': z8['pos'][sel], 'nc8': z8['nc'][sel], 'plddt': z8['plddt'][sel]})
for r in RADII:
    z = Z[r]; m = np.isin(z['uid'], list(prots))
    base = base.merge(pd.DataFrame({'uniprot_id': z['uid'][m], 'uniprot_pos': z['pos'][m], f'H{r}': z['husc'][m]}), on=['uniprot_id', 'uniprot_pos'], how='inner')
pk = set(zip(path.uniprot_id, path.uniprot_pos))
key = list(zip(base.uniprot_id, base.uniprot_pos))
base['label'] = [1 if k in pk else (-1 if k in clin_keys else 0) for k in key]
rng = np.random.default_rng(42)
bg = pd.concat([g.sample(min(100, len(g)), random_state=int(rng.integers(1 << 31))) for _, g in base[base.label == 0].groupby('uniprot_id')])
d = pd.concat([base[base.label == 1], bg]).reset_index(drop=True)
d['cq'] = pd.qcut(d.nc8, 4, labels=False, duplicates='drop'); d['pb'] = np.digitize(d.plddt, [50, 70, 90])
y = d.label.to_numpy(); H = {r: -d[f'H{r}'].to_numpy() for r in RADII}
strata = [g.index.to_numpy() for _, g in d.groupby(['cq', 'pb'])]
print(f'{int(y.sum())} pathogenic residues / {int((y == 0).sum())} background residues from {d.uniprot_id.nunique()} proteins; {len(strata)} strata')
def strat(ii_mask_idx, r):
    num = den = 0.0
    for s in ii_mask_idx:
        ys = y[s]
        if ys.sum() >= 10 and ys.sum() < len(ys):
            w = ys.sum(); num += w * auroc(ys, H[r][s]); den += w
    return num / den
groups = [g.index.to_numpy() for _, g in d.groupby('uniprot_id')]
point = {r: (auroc(y, H[r]), strat(strata, r)) for r in RADII}
cq, pb = d.cq.to_numpy(), d.pb.to_numpy()
B = []
for b in range(1000):
    ii = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
    yy = y[ii]; sk = cq[ii] * 10 + pb[ii]
    row = []
    for r in RADII:
        hh = H[r][ii]; row.append(auroc(yy, hh))
        num = den = 0.0
        for u in np.unique(sk):
            s = sk == u; ys = yy[s]
            if ys.sum() >= 10 and ys.sum() < len(ys):
                w = ys.sum(); num += w * auroc(ys, hh[s]); den += w
        row.append(num / den)
    B.append(row)
B = np.array(B).reshape(-1, len(RADII), 2); i8 = RADII.index(8); out = []
for k, (name) in enumerate(['Pathogenic vs background', 'Pathogenic vs background (structure-matched)']):
    for j, r in enumerate(RADII):
        diff = B[:, j, k] - B[:, i8, k]; a = point[r][k]
        out.append(dict(set=name, n_path=int(y.sum()), n_ben=int((y == 0).sum()), radius=r, auroc=a, lo=np.percentile(B[:, j, k], 2.5),
                        hi=np.percentile(B[:, j, k], 97.5), diff_vs8=a - point[8][k], diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5),
                        p_vs8=1.0 if r == 8 else max(1 / len(B), 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))))
o = pd.DataFrame(out); print(o.round(4).to_string(index=False)); o.to_csv(f'{S}/S2_background_auroc_by_radius_2026-10.csv', index=False)
