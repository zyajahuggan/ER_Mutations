#!/usr/bin/env bash
module load alphafold/3
module load jq
singularity exec --pwd /app/alphafold --nv   --bind "inputs/af3_jsonwt_251_2000_batched":/root/af_input   --bind "/home/zhuggan1/scr4_jgray21/zhuggan1/repos/ER_Mutations/af_part1_cEb3lo":/root/af_output   --bind "/scratch4/datasets/alphafold3_models":/root/models   --bind "/scratch4/datasets/alphafold3":/root/public_databases   /data/apps/extern/singularity/alphafold/alphafold3_exec python run_alphafold.py \
  --norun_inference \
  --json_path=/root/af_input/1A52_WT_holo_s751_1000.json \
  --model_dir=/root/models \
  --db_dir=/root/public_databases \
  --output_dir=/root/af_output \
  
