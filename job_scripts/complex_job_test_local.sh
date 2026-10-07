#!/bin/bash
#
# Local (non-SLURM) test of husc/complex.py on the KCNQ1-KCNE3-CaM complex (PDB 6V00, KCNQ1.pdb).
#
# Run on a laptop from anywhere, with the conda env that has the HuSC dependencies ACTIVE:
#     conda activate husc
#     bash job_scripts/complex_job_test_local.sh
#
# Uses a small gnomAD subset (only the 5 transcripts of KCNQ1, KCNE3 and CALM1), so it does not
# load the 2.2 GB gnomAD JSON and fits in laptop memory. Everything else comes from datafiles/.
#
# Override defaults with environment variables, e.g.:
#     TESTDIR=/path/to/husc_complex_test EXTRA_ARGS="--shared-permutations --seed 1" bash job_scripts/complex_job_test_local.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# test folder: ../husc_complex_test (laptop layout) or <repo>/husc_complex_test (cluster layout)
if [ -z "${TESTDIR:-}" ]; then
    if   [ -d "$REPO/../husc_complex_test" ]; then TESTDIR="$REPO/../husc_complex_test"
    elif [ -d "$REPO/husc_complex_test" ];    then TESTDIR="$REPO/husc_complex_test"
    else echo "ERROR: husc_complex_test folder not found; set TESTDIR." >&2; exit 1; fi
fi
TESTDIR="$(cd "$TESTDIR" && pwd)"

INPUT="${INPUT:-$TESTDIR/KCNQ1_complex_input.txt}"
OUTDIR="${OUTDIR:-$TESTDIR/output}"
RADIUS="${RADIUS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:---seed 1}"      # e.g. "--shared-permutations --chains A,B --seed 1"
CONFIG="$TESTDIR/config_local.json"
LOG="$OUTDIR/husc_complex_test_local.log"

# ---- sanity checks ----------------------------------------------------------
for f in "$TESTDIR/KCNQ1.pdb" "$TESTDIR/KCNQ1_chain_to_uniprot.txt" "$INPUT" \
         "$TESTDIR/local_data/gnomad_variant_counts_subset.json" \
         "$TESTDIR/local_data/gnomad_prob_AFdist_exp_subset.tsv" \
         "$REPO/datafiles/Homo_sapiens.GRCh38.cds.all.fa.gz" \
         "$REPO/datafiles/UP000005640_9606.fasta.gz" \
         "$REPO/datafiles/uniprot_to_enst_mapping.json"; do
    [ -s "$f" ] || { echo "ERROR: missing or empty: $f" >&2; exit 1; }
done
python -c "import Bio, numpy, tqdm" 2>/dev/null || {
    echo "ERROR: Biopython/numpy/tqdm not importable. Activate your conda env first (conda activate <env>)." >&2; exit 1; }

mkdir -p "$OUTDIR"

# ---- local config (absolute paths, so it works from any directory) ---------
cat > "$CONFIG" <<JSON
{
    "ensembl_cds": "$REPO/datafiles/Homo_sapiens.GRCh38.cds.all.fa.gz",
    "uniprot_pep": "$REPO/datafiles/UP000005640_9606.fasta.gz",
    "gnomad_variants": "$TESTDIR/local_data/gnomad_variant_counts_subset.json",
    "uniprot_to_enst": "$REPO/datafiles/uniprot_to_enst_mapping.json",
    "enst_mp_counts": "$TESTDIR/local_data/gnomad_prob_AFdist_exp_subset.tsv",
    "output_dir": "$OUTDIR",
    "pdb_dir": "$TESTDIR",
    "use_af_dist": true
}
JSON

echo "Repo:    $REPO"
echo "Test:    $TESTDIR"
echo "Output:  $OUTDIR"
date

cd "$REPO"
PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 \
    python husc/complex.py -c "$CONFIG" -i "$INPUT" -r "$RADIUS" -l "$LOG" -w $EXTRA_ARGS

echo "Done. Scores: $OUTDIR/<complex_id>_husc_complex.tsv   Log: $LOG"
date
