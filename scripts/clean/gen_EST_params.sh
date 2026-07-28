#!/usr/bin/env bash
# gen_EST_params.sh
#
# Generates the Rosetta params file for estradiol (EST) from a mol2/sdf file.
# Run this ONCE before launching the DMS. The output EST.params goes in inputs/.
#
# Prerequisites:
#   1. An SDF or mol2 file for 17β-estradiol (EST).
#      Download from PDB Ligand Expo: https://www.rcsb.org/ligand/EST
#      (choose the "Ideal Coordinates SDF" or mol2 file)
#
#   2. The Rosetta molfile_to_params.py script:
#      $ROSETTA/main/source/scripts/python/public/molfile_to_params.py
#
# Usage:
#   cd inputs/est_ligand/
#   bash ../../scripts/clean/gen_EST_params.sh

set -euo pipefail

MOLFILE_TO_PARAMS="/home/zhuggan1/scr4_jgray21/zhuggan1/rosetta/source/scripts/python/public/molfile_to_params.py"

INPUTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../inputs/est_ligand" && pwd)"
EST_MOL="${INPUTS_DIR}/EST.sdf"   # download from PDB Ligand Expo → Ideal SDF

if [[ ! -f "${EST_MOL}" ]]; then
    echo "ERROR: ${EST_MOL} not found."
    echo "Download estradiol ideal SDF from:"
    echo "  https://files.rcsb.org/ligands/download/EST_ideal.sdf"
    echo "Then rename it to inputs/est_ligand/EST.sdf and re-run this script."
    exit 1
fi

python3 "${MOLFILE_TO_PARAMS}" \
    -n EST \
    -p "${INPUTS_DIR}/EST" \
    --conformers-in-one-file \
    "${EST_MOL}"

echo "Generated: ${INPUTS_DIR}/EST.params"
echo "           ${INPUTS_DIR}/EST_conformers.pdb  (if any)"
echo
echo "Now run the DMS:"
echo "  cd scripts/dms/ && sbatch submit_dms_dimer.sh (or the relevant submit_dms_*.sh)"
