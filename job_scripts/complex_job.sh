#!/bin/bash
#
# HuSC on protein complexes (CoreHPC / SLURM). Submit from the repo root:
#   sbatch job_scripts/complex_job.sh
#
#SBATCH --job-name=husc_complex
#SBATCH --output=/mnt/scratch/group/CX500147_DS2/jtamsil/capra_fa26/husc_output_test/complex_job_%j.out
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G          # loads the 2.2 GB gnomAD JSON + Ensembl CDS + UniProt FASTA; one 10,000 x L permutation matrix per chain
#SBATCH --time=02:00:00

date
hostname

module load CBI miniforge3
conda activate husc

cd "${SLURM_SUBMIT_DIR:?submit this script with sbatch from the HuSC repo root}"

INPUT=inputs/test_complex_input.txt
CONFIG=config.json
RADIUS=8
LOG=/mnt/scratch/group/CX500147_DS2/jtamsil/capra_fa26/husc_output_test/husc_complex_${SLURM_JOB_ID}.log

# add --shared-permutations for homo-oligomers, --chains A,B to score only some chains
srun -u python husc/complex.py -c "$CONFIG" -i "$INPUT" -r "$RADIUS" -l "$LOG"

echo "Done. Output: <output_dir>/<complex_id>_husc_complex.tsv   Log: $LOG"
date
