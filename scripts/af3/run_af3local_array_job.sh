#!/bin/bash

#SBATCH --job-name=ER_wt_monomer_af3
#SBATCH --output=/scratch4/jgray21/zhuggan1/repos/ER_Mutations/logs/ER_wt_monomer_af3_%A_%a.out
#SBATCH --error=/scratch4/jgray21/zhuggan1/repos/ER_Mutations/logs/ER_wt_monomer_af3_%A_%a.err
#SBATCH --partition=a100
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --account=jgray21_gpu
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=zhuggan1@jh.edu
#SBATCH --mail-type=ALL

## Load AlphaFold3 module
module load alphafold/3

INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p "${OUTPUT_DIR}"

JSON_FILES=($(basename -a ${INPUT_DIR}/*.json))

INPUT_FILE=${JSON_FILES[$SLURM_ARRAY_TASK_ID]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running AF3 on: $INPUT_FILE"

run_af3.sh \
--input_dir=$INPUT_DIR \
--input_file=$INPUT_FILE \
--output_dir=$OUTPUT_DIR \
--models_dir=/scratch4/datasets/alphafold3_models \
--db_dir=/scratch4/datasets/alphafold3 \
--cpu_partition=parallel \
--cpus=10 \
