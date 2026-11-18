"""Collect per-residue scores of the HuSC runs at 5, 8, 11 and 14 Å (Fig. S2, Fig. S3C-D).

Each run uses the same pipeline and inputs with config.json "radius" set to 5, 8, 11 or 14, written to <runs dir>/HuSC_<radius>.

usage: python collect_radius.py <runs dir> <out dir>      writes radius_<r>.npz"""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.special import ndtri

runs, out = sys.argv[1], sys.argv[2]
N_PERM = 10000
cols = ["uniprot_id", "uniprot_pos", "num_contacts", "mis_var_sites", "syn_var_sites", "cs_mis_obs", "HuSC", "mis_p_value",
        "syn_p_value", "plddt"]
for r in [8, 5, 11, 14]:
    files = sorted(f for f in glob.glob(f"{runs}/HuSC_{r}/*_husc.tsv") if not os.path.basename(f).startswith("._"))
    d = pd.concat([pd.read_csv(f, sep="\t", usecols=cols) for f in files], ignore_index=True)
    syn = np.clip(d.syn_p_value.to_numpy(), 0.5 / (N_PERM + 1), (N_PERM + 0.5) / (N_PERM + 1))
    np.savez_compressed(f"{out}/radius_{r}.npz", uid=d.uniprot_id.to_numpy().astype("U12"), pos=d.uniprot_pos.to_numpy(np.int32),
                        husc=d.HuSC.to_numpy(np.float32), syn=ndtri(syn).astype(np.float32),
                        mis_p=d.mis_p_value.to_numpy(np.float32), syn_p=d.syn_p_value.to_numpy(np.float32),
                        nc=d.num_contacts.to_numpy(np.int16), mis_obs=d.cs_mis_obs.to_numpy(np.float32),
                        mis_sites=d.mis_var_sites.to_numpy(np.int32), plddt=d.plddt.to_numpy(np.float32))
    print(f"{r} Å: {len(files):,} proteins, {len(d):,} residues", flush=True)
