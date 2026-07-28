#!/bin/bash
#SBATCH --account=jgray21
#SBATCH --partition=parallel
#SBATCH --job-name=ER_DMS_H524_dimer_1GWR
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
mkdir -p "${REPO}/logs" "${REPO}/outputs/dms/dms_1GWR/dms_H524_dimer_holo_1GWR"

# Step 1: Generate holo dimer + coactivator PDB from 1GWR (chains A+B protein, chains
# C+D estradiol, chains E+F TIF2 peptide)
if [[ ! -f "${REPO}/inputs/1GWR/1GWR_clean_dimer.pdb" ]]; then
    echo "[$(date)] Generating 1GWR holo dimer PDB ..."
    python3 "${REPO}/scripts/clean/clean_1gwr.py" \
        --out_pdb "${REPO}/inputs/1GWR/1GWR_clean_dimer.pdb" \
        --dimer-holo
else
    echo "[$(date)] Using existing ${REPO}/inputs/1GWR/1GWR_clean_dimer.pdb"
fi

# Step 2: EST params check
if [[ ! -f "${REPO}/inputs/est_ligand/EST.params" ]]; then
    echo "ERROR: ${REPO}/inputs/est_ligand/EST.params not found."
    echo "Run:  bash ${REPO}/scripts/clean/gen_EST_params.sh"
    exit 1
fi

# Step 3: DMS
echo "[$(date)] Starting 1GWR holo homodimer + peptide DMS at H524 ..."
python3 "${REPO}/scripts/dms/dms_H524_dimer_1GWR.py" \
    --wt_pdb     "${REPO}/inputs/1GWR/1GWR_clean_dimer.pdb" \
    --est_params "${REPO}/inputs/est_ligand/EST.params" \
    --xml        "${REPO}/scripts/dms/dms_H524_dimer_1GWR.xml" \
    --out_dir    "${REPO}/outputs/dms/dms_1GWR/dms_H524_dimer_holo_1GWR" \
    --prefix     1GWR \
    --nstruct    5 \
    --workers    "${SLURM_CPUS_PER_TASK}"

echo "[$(date)] Done."
