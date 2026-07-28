#!/bin/bash
#SBATCH --account=jgray21
#SBATCH --partition=parallel
#SBATCH --job-name=ER_DMS_H524_dimer_apo
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
mkdir -p "${REPO}/logs" "${REPO}/outputs/dms_H524_dimer_apo"

# Step 1: Generate apo dimer PDB (chains A+B protein, no estradiol)
if [[ ! -f "${REPO}/inputs/1A52_clean_dimer_apo.pdb" ]]; then
    echo "[$(date)] Generating apo dimer PDB ..."
    python3 "${REPO}/scripts/clean_1A52.py" \
        --out_pdb "${REPO}/inputs/1A52_clean_dimer_apo.pdb" \
        --dimer
else
    echo "[$(date)] Using existing ${REPO}/inputs/1A52_clean_dimer_apo.pdb"
fi

# Step 2: DMS
echo "[$(date)] Starting apo homodimer DMS at H524 ..."
python3 "${REPO}/scripts/dms_H524_dimer_apo.py" \
    --wt_pdb  "${REPO}/inputs/1A52_clean_dimer_apo.pdb" \
    --xml     "${REPO}/scripts/dms_H524_dimer_apo.xml" \
    --out_dir "${REPO}/outputs/dms_H524_dimer_apo" \
    --nstruct 5 \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "[$(date)] Done."
