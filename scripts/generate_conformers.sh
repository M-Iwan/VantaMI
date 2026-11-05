#!/usr/bin/env bash
#
# Run conformer generation with SLURM with nohup.
#
# Example:
#   nohup bash run_conformers.sh > conformers.log 2>&1 &
#

set -euo pipefail
conda activate ionmelon

PYTHON_PATH="/path/to/python3.12"
PYTHON_SCRIPT="/path/to/generate_conformers.py"
INPUT_DATA="/path/to/input_dataset.parquet"
OUTPUT_MANAGER="/path/to/output/deep_molecule_manager.pkl"

SMILES_COLUMN="SMILES"
N_CONFORMERS=32
ENERGY_METHOD="MMFF94"
BATCH_SIZE=256
N_CPUS=96

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "CPUs provided to Python/joblib: ${N_CPUS}"

export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON_PATH}" -u "${PYTHON_SCRIPT}" \
  --data-path "${INPUT_DATA}" \
  --save-path "${OUTPUT_MANAGER}" \
  --smiles-col "${SMILES_COLUMN}" \
  --n-conformers "${N_CONFORMERS}" \
  --energy-method "${ENERGY_METHOD}" \
  --n-jobs "${N_CPUS}" \
  --batch-size "${BATCH_SIZE}"

echo "Finished: $(date)"
