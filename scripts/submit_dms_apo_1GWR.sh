#!/bin/bash
#SBATCH --account=jgray21
#SBATCH --partition=parallel
#SBATCH --job-name=ER_DMS_H524_apo_1GWR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch4/jgray21/zhuggan1/repos/ER_Mutations/logs/%x_%A.out
#SBATCH --error=/scratch4/jgray21/zhuggan1/repos/ER_Mutations/logs/%x_%A.err
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate pyrosetta

REPO=/scratch4/jgray21/zhuggan1/repos/ER_Mutations
mkdir -p "${REPO}/logs" "${REPO}/outputs/dms_H524_apo_1GWR"

# Step 1: Generate apo monomer + coactivator PDB from 1GWR (chain A protein, chain C
# TIF2 peptide, no estradiol)
if [[ ! -f "${REPO}/inputs/1GWR_clean_apo.pdb" ]]; then
    echo "[$(date)] Generating 1GWR apo monomer PDB ..."
    python3 "${REPO}/scripts/clean_1gwr.py" \
        --out_pdb "${REPO}/inputs/1GWR_clean_apo.pdb" \
        --apo
else
    echo "[$(date)] Using existing ${REPO}/inputs/1GWR_clean_apo.pdb"
fi

# Step 2: DMS
echo "[$(date)] Starting 1GWR apo monomer + peptide DMS at H524 ..."
python3 "${REPO}/scripts/dms_H524_apo_1GWR.py" \
    --wt_pdb  "${REPO}/inputs/1GWR_clean_apo.pdb" \
    --xml     "${REPO}/scripts/dms_H524_apo_1GWR.xml" \
    --out_dir "${REPO}/outputs/dms_H524_apo_1GWR" \
    --prefix  1GWR \
    --nstruct 5 \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "[$(date)] Done."
