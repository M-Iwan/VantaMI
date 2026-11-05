#!/usr/bin/env bash
#
# Run five-fold Uni-pKa GPU inference only.
#
# Usage:
#   ./run_unipka.sh PREPARED_DIR FOLD_RESULTS_DIR \
#       [GPU_IDS] [SHM_SIZE] [NUM_WORKERS] [BATCH_SIZE] [DATA_BUFFER_SIZE]
#
# Examples:
#   ./run_unipka.sh prepared_unipka unipka_fold_results 0 64g 8 16 8
#   ./run_unipka.sh prepared_unipka unipka_fold_results "0,1,2,3,4" 64g 8 16 8
#
# Outputs in FOLD_RESULTS_DIR:
#   fold_0/*.pkl ... fold_4/*.pkl
#

set -Eeuo pipefail

REPO_DIR="${HOME}/.vantami/data/Uni-pKa"
MODEL_DIR="${REPO_DIR}/models"
IMAGE="uni-pka:local"
TASK_NAME="sampl6"

PREPARED_DIR="${1:-}"
FOLD_RESULTS_DIR="${2:-}"
GPU_IDS_RAW="${3:-0}"
SHM_SIZE="${4:-64g}"
NUM_WORKERS="${5:-8}"
BATCH_SIZE="${6:-16}"
DATA_BUFFER_SIZE="${7:-8}"

usage() {
    cat >&2 <<EOF
Usage:
  $0 PREPARED_DIR FOLD_RESULTS_DIR [GPU_IDS] [SHM_SIZE] [NUM_WORKERS] [BATCH_SIZE] [DATA_BUFFER_SIZE]

PREPARED_DIR must be created by prepare_unipka.sh
FOLD_RESULTS_DIR receives raw five-fold Uni-pKa inference outputs.

GPU_IDS may be one GPU ID or a comma-/whitespace-separated list of up to five
GPU IDs, for example: 0, 0,1, or "0 1 2 3 4".

Defaults:
  GPU_IDS:          0
  SHM_SIZE:         64g
  NUM_WORKERS:      8
  BATCH_SIZE:       16
  DATA_BUFFER_SIZE: 8

Examples:
  $0 prepared_unipka unipka_fold_results 0 64g 8 16 8
  $0 prepared_unipka unipka_fold_results "0,1,2,3,4" 64g 8 16 8
EOF
    exit 2
}

host_error() {
    local status=$?
    local line=$1
    printf 'ERROR: host script failed at line %s (exit status %s).\n' \
        "${line}" "${status}" >&2
    exit "${status}"
}

trap 'host_error $LINENO' ERR

[[ -n "${PREPARED_DIR}" && -n "${FOLD_RESULTS_DIR}" ]] || usage

[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || {
    echo "Arguments: NUM_WORKERS must be a non-negative integer." >&2
    exit 2
}

[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Arguments: BATCH_SIZE must be a positive integer." >&2
    exit 2
}

[[ "${DATA_BUFFER_SIZE}" =~ ^[0-9]+$ ]] || {
    echo "Arguments: DATA_BUFFER_SIZE must be a non-negative integer." >&2
    exit 2
}

[[ -n "${SHM_SIZE}" ]] || {
    echo "Arguments: SHM_SIZE must not be empty." >&2
    exit 2
}

[[ -d "${PREPARED_DIR}" ]] || {
    echo "Input: prepared directory not found: ${PREPARED_DIR}" >&2
    exit 1
}

required_prepared_files=(
    unique_input.tsv
    "${TASK_NAME}.tsv"
    transition_metadata.tsv
    microstate_metadata.tsv
    graph_diagnostics.tsv
    "${TASK_NAME}.lmdb"
    dict.txt
    dict_charge.txt
)

for file in "${required_prepared_files[@]}"; do
    [[ -f "${PREPARED_DIR}/${file}" ]] || {
        echo "Input: missing ${PREPARED_DIR}/${file}" >&2
        exit 1
    }
done

command -v docker >/dev/null 2>&1 || {
    echo "Docker: command not found." >&2
    exit 1
}

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
    echo "Docker: image '${IMAGE}' not found." >&2
    exit 1
}

for fold in 0 1 2 3 4; do
    checkpoint="${MODEL_DIR}/fold_${fold}/checkpoint_best.pt"

    [[ -f "${checkpoint}" ]] || {
        echo "Models: missing checkpoint for fold ${fold}: ${checkpoint}" >&2
        exit 1
    }
done

GPU_IDS_NORMALIZED="${GPU_IDS_RAW//,/ }"
read -r -a GPU_IDS <<< "${GPU_IDS_NORMALIZED}"

[[ "${#GPU_IDS[@]}" -ge 1 ]] || {
    echo "Arguments: at least one GPU ID is required." >&2
    exit 2
}

[[ "${#GPU_IDS[@]}" -le 5 ]] || {
    echo "Arguments: at most five GPU IDs may be supplied." >&2
    exit 2
}

declare -A SEEN_GPU_IDS=()

for gpu_id in "${GPU_IDS[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
        echo "Arguments: GPU IDs must be non-negative integers." >&2
        exit 2
    }

    [[ -z "${SEEN_GPU_IDS[${gpu_id}]:-}" ]] || {
        echo "Arguments: duplicate GPU ID supplied: ${gpu_id}" >&2
        exit 2
    }

    SEEN_GPU_IDS["${gpu_id}"]=1
done

PREPARED_DIR="$(cd "${PREPARED_DIR}" && pwd)"
mkdir -p "${FOLD_RESULTS_DIR}"
FOLD_RESULTS_DIR="$(cd "${FOLD_RESULTS_DIR}" && pwd)"

OWNERSHIP_REPAIRED=0
repair_results_ownership_once() {
    if [[ "${OWNERSHIP_REPAIRED}" -eq 1 ]]; then
        return 0
    fi

    OWNERSHIP_REPAIRED=1

    if [[ ! -d "${FOLD_RESULTS_DIR:-}" ]]; then
        return 0
    fi

    echo
    echo "=== Repairing results ownership once after fold containers stopped ==="

    docker run --rm \
        -v "${FOLD_RESULTS_DIR}:/work/results" \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        "${IMAGE}" \
        bash -c 'set -euo pipefail; chown -R "${HOST_UID}:${HOST_GID}" /work/results; chmod -R u+rwX /work/results' \
        || {
            echo "WARNING: ownership repair failed for ${FOLD_RESULTS_DIR}." >&2
            return 1
        }
}

on_exit() {
    local status=$?
    trap - EXIT

    repair_results_ownership_once || {
        if [[ "${status}" -eq 0 ]]; then
            status=1
        fi
    }

    exit "${status}"
}

trap on_exit EXIT

# Bash wait -n -p is available in Bash >= 5.1. Fail early with a clear message
# instead of hanging later on older login nodes.
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "Host: this script requires Bash >= 5.1 for 'wait -n -p'." >&2
    exit 1
fi

echo "=== Uni-pKa GPU inference only ==="
echo "Prepared data:     ${PREPARED_DIR}"
echo "Fold results:      ${FOLD_RESULTS_DIR}"
echo "GPUs:              ${GPU_IDS[*]}"
echo "Shared memory:     ${SHM_SIZE}"
echo "Data workers:      ${NUM_WORKERS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Data buffer size:  ${DATA_BUFFER_SIZE}"

run_fold() {
    local fold="$1"
    local gpu_id="$2"

    echo "Launching fold ${fold} on GPU ${gpu_id}."

    docker run --rm -i \
        --shm-size="${SHM_SIZE}" \
        --gpus "device=${gpu_id}" \
        -v "${PREPARED_DIR}:/work/runtime_data:ro" \
        -v "${MODEL_DIR}:/work/models:ro" \
        -v "${FOLD_RESULTS_DIR}:/work/results" \
        -e TASK_NAME="${TASK_NAME}" \
        -e FOLD="${fold}" \
        -e NUM_WORKERS="${NUM_WORKERS}" \
        -e BATCH_SIZE="${BATCH_SIZE}" \
        -e DATA_BUFFER_SIZE="${DATA_BUFFER_SIZE}" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        -e NUMEXPR_NUM_THREADS=1 \
        -e VECLIB_MAXIMUM_THREADS=1 \
        -e PYTHONUNBUFFERED=1 \
        "${IMAGE}" \
        bash -s <<'CONTAINER_FOLD_SCRIPT'
set -Eeuo pipefail

REPO_DIR="/opt/uni-pka"
RUNTIME_DATA="/work/runtime_data"
RUNTIME_PATCH="/work/runtime_patch"
RESULTS_DIR="/work/results"

[[ -d "${REPO_DIR}" ]] || {
    echo "Container: Uni-pKa repository not found: ${REPO_DIR}" >&2
    exit 1
}

required_files=(
    unique_input.tsv
    "${TASK_NAME}.tsv"
    transition_metadata.tsv
    microstate_metadata.tsv
    graph_diagnostics.tsv
    "${TASK_NAME}.lmdb"
    dict.txt
    dict_charge.txt
)

for file in "${required_files[@]}"; do
    [[ -f "${RUNTIME_DATA}/${file}" ]] || {
        echo "Input: missing ${file} in container." >&2
        exit 1
    }
done

checkpoint="/work/models/fold_${FOLD}/checkpoint_best.pt"

[[ -f "${checkpoint}" ]] || {
    echo "Models: missing fold ${FOLD} checkpoint." >&2
    exit 1
}

cd "${REPO_DIR}"
mkdir -p "${RUNTIME_PATCH}"

cat > "${RUNTIME_PATCH}/sitecustomize.py" <<'PYTHON_SHIM'
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
    print("Warning: NumPy compatibility shim failed:", error, flush=True)
PYTHON_SHIM

export PYTHONPATH="${RUNTIME_PATCH}:${REPO_DIR}:${PYTHONPATH:-}"

python - <<'PYTHON_TEST'
import numpy
import numpy._core
import numpy._core.multiarray

print("NumPy version:", numpy.__version__, flush=True)
print("NumPy compatibility import: OK", flush=True)
PYTHON_TEST

echo "=== Fold ${FOLD} inference ==="
nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader

python ./unimol/infer.py \
    --user-dir ./unimol \
    "${RUNTIME_DATA}" \
    --task-name "${TASK_NAME}" \
    --valid-subset "${TASK_NAME}" \
    --results-path "${RESULTS_DIR}/fold_${FOLD}" \
    --num-workers "${NUM_WORKERS}" \
    --data-buffer-size "${DATA_BUFFER_SIZE}" \
    --ddp-backend c10d \
    --batch-size "${BATCH_SIZE}" \
    --task mol_pka \
    --loss finetune_mse \
    --arch unimol_pka \
    --classification-head-name chembl \
    --num-classes 1 \
    --dict-name dict.txt \
    --charge-dict-name dict_charge.txt \
    --conf-size 11 \
    --only-polar -1 \
    --path "${checkpoint}" \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --log-interval 50 \
    --log-format simple \
    --required-batch-size-multiple 1
CONTAINER_FOLD_SCRIPT
}

echo
echo "=== Running five-fold inference asynchronously ==="

available_gpus=("${GPU_IDS[@]}")
next_fold=0

declare -a RUNNING_PIDS=()
declare -A PID_TO_GPU=()
declare -A PID_TO_FOLD=()

launch_available_folds() {
    while [[ "${next_fold}" -lt 5 && "${#available_gpus[@]}" -gt 0 ]]; do
        local gpu_id="${available_gpus[0]}"
        available_gpus=("${available_gpus[@]:1}")

        local fold="${next_fold}"
        next_fold=$((next_fold + 1))

        run_fold "${fold}" "${gpu_id}" &
        local pid="$!"

        RUNNING_PIDS+=("${pid}")
        PID_TO_GPU["${pid}"]="${gpu_id}"
        PID_TO_FOLD["${pid}"]="${fold}"

        echo "Fold ${fold} is running with PID ${pid} on GPU ${gpu_id}."
    done
}

remove_running_pid() {
    local completed_pid="$1"
    local retained=()

    for pid in "${RUNNING_PIDS[@]}"; do
        if [[ "${pid}" != "${completed_pid}" ]]; then
            retained+=("${pid}")
        fi
    done

    RUNNING_PIDS=("${retained[@]}")
}

terminate_remaining_folds() {
    local skip_pid="${1:-}"

    for pid in "${RUNNING_PIDS[@]}"; do
        if [[ -z "${skip_pid}" || "${pid}" != "${skip_pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    wait || true
}

launch_available_folds

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    completed_pid=""

    if ! wait -n -p completed_pid "${RUNNING_PIDS[@]}"; then
        failed_fold="${PID_TO_FOLD[${completed_pid}]:-unknown}"
        failed_gpu="${PID_TO_GPU[${completed_pid}]:-unknown}"

        echo "Inference: fold ${failed_fold} failed on GPU ${failed_gpu}." >&2
        echo "Inference: terminating remaining fold containers before ownership repair." >&2

        terminate_remaining_folds "${completed_pid}"
        exit 1
    fi

    completed_fold="${PID_TO_FOLD[${completed_pid}]}"
    completed_gpu="${PID_TO_GPU[${completed_pid}]}"

    echo "Fold ${completed_fold} completed; GPU ${completed_gpu} returned to pool."

    remove_running_pid "${completed_pid}"

    unset "PID_TO_FOLD[${completed_pid}]"
    unset "PID_TO_GPU[${completed_pid}]"

    available_gpus+=("${completed_gpu}")

    launch_available_folds
done

echo
echo "=== Validating fold result directories ==="
for fold in 0 1 2 3 4; do
    fold_dir="${FOLD_RESULTS_DIR}/fold_${fold}"
    [[ -d "${fold_dir}" ]] || {
        echo "Output: missing fold directory: ${fold_dir}" >&2
        exit 1
    }

    shopt -s nullglob
    pkl_files=("${fold_dir}"/*.pkl)
    shopt -u nullglob

    [[ "${#pkl_files[@]}" -eq 1 ]] || {
        echo "Output: fold ${fold} should contain exactly one .pkl file; found ${#pkl_files[@]}." >&2
        exit 1
    }

    echo "  fold_${fold}: ${pkl_files[0]}"
done

echo
echo "=== GPU inference completed ==="
echo "Raw fold outputs are available in: ${FOLD_RESULTS_DIR}"
echo "Run aggregate_unipka_results.sh next to create transition_predictions.tsv, microstate_populations.tsv, and final_states.tsv."
