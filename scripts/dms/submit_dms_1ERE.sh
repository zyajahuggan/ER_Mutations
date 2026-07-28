#!/bin/bash
#SBATCH --account=jgray21
#SBATCH --partition=parallel
#SBATCH --job-name=ER_DMS_H524_1ERE
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
mkdir -p "${REPO}/logs" "${REPO}/outputs/dms/dms_1ERE/dms_H524_holo_1ERE"

# Step 1: Generate holo monomer PDB from 1ERE (chain A protein, chain B estradiol)
if [[ ! -f "${REPO}/inputs/1ERE/1ERE_clean.pdb" ]]; then
    echo "[$(date)] Generating 1ERE holo monomer PDB ..."
    python3 "${REPO}/scripts/clean/clean_1ERE.py" \
        --out_pdb "${REPO}/inputs/1ERE/1ERE_clean.pdb"
else
    echo "[$(date)] Using existing ${REPO}/inputs/1ERE/1ERE_clean.pdb"
fi

# Step 2: EST params check
if [[ ! -f "${REPO}/inputs/est_ligand/EST.params" ]]; then
    echo "ERROR: ${REPO}/inputs/est_ligand/EST.params not found."
    echo "Run:  bash ${REPO}/scripts/clean/gen_EST_params.sh"
    exit 1
fi

# Step 3: DMS
echo "[$(date)] Starting 1ERE holo monomer DMS at H524 ..."
python3 "${REPO}/scripts/dms/dms_H524.py" \
    --wt_pdb     "${REPO}/inputs/1ERE/1ERE_clean.pdb" \
    --est_params "${REPO}/inputs/est_ligand/EST.params" \
    --xml        "${REPO}/scripts/dms/dms_H524.xml" \
    --out_dir    "${REPO}/outputs/dms/dms_1ERE/dms_H524_holo_1ERE" \
    --prefix     1ERE \
    --nstruct    5 \
    --workers    "${SLURM_CPUS_PER_TASK}"

echo "[$(date)] Done."
