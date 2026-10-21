#!/usr/bin/env bash
#
# Run Uni-pKa inference for the included SAMPL6 benchmark dataset.
#
# Requirements:
#   - Docker image: uni-pka:local
#   - Host GPU 1 is available and assigned to this job
#   - Checkpoints exist under:
#       ~/.vantami/data/Uni-pKa/models/fold_{0..4}/checkpoint_best.pt
#
# Output:
#   ~/.vantami/data/Uni-pKa/results/mean_results.csv
#

set -euo pipefail

REPO_DIR="${HOME}/.vantami/data/Uni-pKa"
INPUT_DIR="${REPO_DIR}/dataset"
MODEL_DIR="${REPO_DIR}/models"
RESULTS_DIR="${REPO_DIR}/results"
IMAGE="uni-pka:local"
GPU_ID="2"

echo "=== Host-side validation ==="

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not available in PATH." >&2
    exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: Docker image '${IMAGE}' is not available." >&2
    exit 1
fi

if [[ ! -f "${INPUT_DIR}/sampl6.tsv" ]]; then
    echo "ERROR: Missing input dataset: ${INPUT_DIR}/sampl6.tsv" >&2
    exit 1
fi

for fold in 0 1 2 3 4; do
    checkpoint="${MODEL_DIR}/fold_${fold}/checkpoint_best.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi
done

mkdir -p "${RESULTS_DIR}"

echo "Docker image: ${IMAGE}"
echo "Physical host GPU selected: ${GPU_ID}"
echo "Input: ${INPUT_DIR}/sampl6.tsv"
echo "Models: ${MODEL_DIR}"
echo "Results: ${RESULTS_DIR}"
echo

# This removes output from an interrupted or earlier run. It never modifies
# inputs or model checkpoints.
rm -rf "${RESULTS_DIR:?}/"*

docker run --rm -i \
    --gpus "\"device=${GPU_ID}\"" \
    -v "${INPUT_DIR}:/work/input:ro" \
    -v "${MODEL_DIR}:/work/models:ro" \
    -v "${RESULTS_DIR}:/work/results" \
    "${IMAGE}" \
    bash -s <<'CONTAINER_SCRIPT'

set -euo pipefail

REPO_DIR="/opt/uni-pka"
RUNTIME_DATA="/work/runtime_data"
RUNTIME_PATCH="/work/runtime_patch"
RESULTS_DIR="/work/results"

echo "=== Container started ==="
echo "Repository path: ${REPO_DIR}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "ERROR: ${REPO_DIR} does not exist inside the image." >&2
    exit 1
fi

cd "${REPO_DIR}"

echo
echo "=== GPU visible inside container ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "=== Creating NumPy checkpoint compatibility shim ==="

mkdir -p "${RUNTIME_PATCH}"

cat > "${RUNTIME_PATCH}/sitecustomize.py" <<'PYTHON_SHIM'
"""
Compatibility aliases for checkpoints serialized with NumPy 2.x.

The Uni-pKa runtime image uses an older NumPy release exposing numpy.core,
while the downloaded checkpoint references numpy._core during torch.load().
"""
import importlib
import sys

try:
    import numpy
    import numpy.core

    sys.modules.setdefault("numpy._core", numpy.core)

    for module_name in (
        "multiarray",
        "_multiarray_umath",
        "umath",
        "numeric",
        "fromnumeric",
        "shape_base",
        "overrides",
        "records",
        "function_base",
        "getlimits",
        "arrayprint",
        "defchararray",
        "einsumfunc",
        "memmap",
        "numerictypes",
    ):
        try:
            module = importlib.import_module("numpy.core." + module_name)
            sys.modules.setdefault("numpy._core." + module_name, module)
        except Exception:
            pass
except Exception as error:
    print("Warning: NumPy compatibility shim could not be fully initialized:", error)
PYTHON_SHIM

export PYTHONPATH="${RUNTIME_PATCH}:${REPO_DIR}:${PYTHONPATH:-}"

echo
echo "=== Verifying NumPy compatibility shim ==="

python - <<'PYTHON_TEST'
import numpy
import numpy._core
import numpy._core.multiarray

print("NumPy version:", numpy.__version__)
print("NumPy compatibility import numpy._core: OK")
PYTHON_TEST

echo
echo "=== Preparing inference data ==="

rm -rf "${RUNTIME_DATA}"
mkdir -p "${RUNTIME_DATA}"

cp /work/input/sampl6.tsv "${RUNTIME_DATA}/sampl6.tsv"

echo
echo "=== Generating conformers and LMDB data ==="

python scripts/preprocess_pka.py \
    --raw-csv-file "${RUNTIME_DATA}/sampl6.tsv" \
    --processed-lmdb-dir "${RUNTIME_DATA}" \
    --task-name sampl6 \
    --nthreads 8

cp unimol/examples/dict.txt "${RUNTIME_DATA}/dict.txt"
cp unimol/examples/dict_charge.txt "${RUNTIME_DATA}/dict_charge.txt"

echo
echo "=== Validating mounted checkpoints ==="

for fold in 0 1 2 3 4; do
    checkpoint="/work/models/fold_${fold}/checkpoint_best.pt"
    test -f "${checkpoint}"
    echo "fold_${fold}: OK"
done

echo
echo "=== Running Uni-pKa inference: 5 folds, sequentially ==="

for fold in 0 1 2 3 4; do
    echo
    echo "===== Fold ${fold} started ====="

    python ./unimol/infer.py \
        --user-dir ./unimol \
        "${RUNTIME_DATA}" \
        --task-name sampl6 \
        --valid-subset sampl6 \
        --results-path "${RESULTS_DIR}/fold_${fold}" \
        --num-workers 8 \
        --ddp-backend=c10d \
        --batch-size 32 \
        --task mol_pka \
        --loss finetune_mse \
        --arch unimol_pka \
        --classification-head-name chembl \
        --num-classes 1 \
        --dict-name dict.txt \
        --charge-dict-name dict_charge.txt \
        --conf-size 11 \
        --only-polar -1 \
        --path "/work/models/fold_${fold}/checkpoint_best.pt" \
        --fp16 \
        --fp16-init-scale 4 \
        --fp16-scale-window 256 \
        --log-interval 50 \
        --log-format simple \
        --required-batch-size-multiple 1

    echo "===== Fold ${fold} finished ====="
done

echo
echo "=== Aggregating five-fold predictions ==="

python scripts/infer_mean_ensemble.py \
    --task pka \
    --nfolds 5 \
    --results-path "${RESULTS_DIR}"

echo
echo "=== Container inference completed successfully ==="
echo "Files written:"
ls -lh "${RESULTS_DIR}"

CONTAINER_SCRIPT

echo
echo "=== Host-side completion check ==="

FINAL_RESULT="${RESULTS_DIR}/mean_results.csv"

if [[ ! -f "${FINAL_RESULT}" ]]; then
    echo "ERROR: Container exited but final result is missing: ${FINAL_RESULT}" >&2
    exit 1
fi

echo "Final result created:"
echo "${FINAL_RESULT}"
