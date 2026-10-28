#!/usr/bin/env bash
#
# Run five-fold Uni-pKa inference on one or more selected GPUs.
#
# Usage:
#   ./run_unipka_gpu.sh PREPARED_DIR OUTPUT_DIR \
#       [GPU_IDS] [pH] [SHM_SIZE] [NUM_WORKERS] [BATCH_SIZE] [DATA_BUFFER_SIZE]
#
# Examples:
#   ./run_unipka_gpu.sh prepared_unipka unipka_results 0 7.4 128g 16 32 32
#   ./run_unipka_gpu.sh prepared_unipka unipka_results "0,1,2,3,4" 7.4 128g 16 32 32
#
# PREPARED_DIR must have been created by prepare_microstates.sh.
#
# Outputs in OUTPUT_DIR:
#   transition_predictions.tsv
#       Fold-mean and fold-standard-deviation Uni-pKa predictions per transition.
#
#   microstate_populations.tsv
#       Relative probabilities for individual microstates, derived from the
#       transition graph at the requested pH.
#
#   final_states.tsv
#       Exactly one preferred microstate per input SMILES.
#

set -euo pipefail

REPO_DIR="${HOME}/.vantami/data/Uni-pKa"
MODEL_DIR="${REPO_DIR}/models"
IMAGE="uni-pka:local"
TASK_NAME="sampl6"

PREPARED_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
GPU_IDS_RAW="${3:-0}"
PH="${4:-7.4}"
SHM_SIZE="${5:-128g}"
NUM_WORKERS="${6:-16}"
BATCH_SIZE="${7:-32}"
DATA_BUFFER_SIZE="${8:-32}"

usage() {
    cat >&2 <<EOF
Usage:
  $0 PREPARED_DIR OUTPUT_DIR [GPU_IDS] [pH] [SHM_SIZE] [NUM_WORKERS] [BATCH_SIZE] [DATA_BUFFER_SIZE]

PREPARED_DIR must be created by prepare_microstate_unipka.sh.

GPU_IDS may be one GPU ID or a comma-/whitespace-separated list of up to five
GPU IDs, for example: 0, 0,1, or "0 1 2 3 4".

Defaults:
  GPU_IDS:          0
  pH:               7.4
  SHM_SIZE:         64g
  NUM_WORKERS:      2
  BATCH_SIZE:       1
  DATA_BUFFER_SIZE: 2

Examples:
  $0 prepared_unipka unipka_results 0 7.4 64g 2 1 2
  $0 prepared_unipka unipka_results "0,1,2,3,4" 7.4 64g 2 1 2
EOF
    exit 2
}

[[ -n "${PREPARED_DIR}" && -n "${OUTPUT_DIR}" ]] || usage

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

# Convert comma-separated and/or whitespace-separated GPU IDs into an array.
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

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

TEMP_RESULTS_DIR="$(mktemp -d "${OUTPUT_DIR}/.unipka-results.XXXXXX")"

cleanup() {
    rm -rf "${TEMP_RESULTS_DIR}"
}

trap cleanup EXIT

echo "=== Uni-pKa GPU inference ==="
echo "Prepared data:     ${PREPARED_DIR}"
echo "Output:            ${OUTPUT_DIR}"
echo "GPUs:              ${GPU_IDS[*]}"
echo "pH:                ${PH}"
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
        -v "${TEMP_RESULTS_DIR}:/work/results" \
        -e TASK_NAME="${TASK_NAME}" \
        -e FOLD="${fold}" \
        -e NUM_WORKERS="${NUM_WORKERS}" \
        -e BATCH_SIZE="${BATCH_SIZE}" \
        -e DATA_BUFFER_SIZE="${DATA_BUFFER_SIZE}" \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e OPENBLAS_NUM_THREADS=1 \
        -e NUMEXPR_NUM_THREADS=1 \
        -e VECLIB_MAXIMUM_THREADS=1 \
        -e PYTHONUNBUFFERED=1 \
        "${IMAGE}" \
        bash -s <<'CONTAINER_FOLD_SCRIPT'
set -euo pipefail

REPO_DIR="/opt/uni-pka"
RUNTIME_DATA="/work/runtime_data"
RUNTIME_PATCH="/work/runtime_patch"
RESULTS_DIR="/work/results"

return_results_ownership() {
    if [[ -d "${RESULTS_DIR:-}" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" "${RESULTS_DIR}" 2>/dev/null || true
        chmod -R u+rwX "${RESULTS_DIR}" 2>/dev/null || true
    fi
}

trap return_results_ownership EXIT

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
            sys.modules.setdefault(
                "numpy._core." + module_name,
                module,
            )
        except Exception:
            pass

except Exception as error:
    print("Warning: NumPy compatibility shim failed:", error)
PYTHON_SHIM

export PYTHONPATH="${RUNTIME_PATCH}:${REPO_DIR}:${PYTHONPATH:-}"

python - <<'PYTHON_TEST'
import numpy
import numpy._core
import numpy._core.multiarray

print("NumPy version:", numpy.__version__)
print("NumPy compatibility import: OK")
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

# Folds are independent. At most one Docker fold job is assigned to a selected
# GPU at once. As a job exits, its GPU ID is immediately returned to the
# available queue and is assigned to the next waiting fold.
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

launch_available_folds

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
    completed_pid=""

    if ! wait -n -p completed_pid "${RUNNING_PIDS[@]}"; then
        failed_fold="${PID_TO_FOLD[${completed_pid}]:-unknown}"
        failed_gpu="${PID_TO_GPU[${completed_pid}]:-unknown}"

        echo \
            "Inference: fold ${failed_fold} failed on GPU ${failed_gpu}." \
            >&2

        for pid in "${RUNNING_PIDS[@]}"; do
            if [[ "${pid}" != "${completed_pid}" ]]; then
                kill "${pid}" 2>/dev/null || true
            fi
        done

        wait || true
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
echo "=== Aggregating five-fold predictions at pH ${PH} ==="

docker run --rm -i \
    -v "${PREPARED_DIR}:/work/runtime_data:ro" \
    -v "${TEMP_RESULTS_DIR}:/work/results" \
    -e PH="${PH}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    -e PYTHONUNBUFFERED=1 \
    "${IMAGE}" \
    bash -s <<'CONTAINER_AGGREGATION_SCRIPT'
set -euo pipefail

RUNTIME_DATA="/work/runtime_data"
RESULTS_DIR="/work/results"
RUNTIME_PATCH="/work/runtime_patch"

return_results_ownership() {
    if [[ -d "${RESULTS_DIR:-}" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" "${RESULTS_DIR}" 2>/dev/null || true
        chmod -R u+rwX "${RESULTS_DIR}" 2>/dev/null || true
    fi
}

trap return_results_ownership EXIT

mkdir -p "${RUNTIME_PATCH}"

cat > "${RUNTIME_PATCH}/sitecustomize.py" <<'PYTHON_SHIM'
import importlib
import sys

try:
    import numpy
    import numpy.core

    sys.modules.setdefault("numpy._core", numpy.core)

    for module_name in ("multiarray", "_multiarray_umath", "umath", "numeric", "fromnumeric", "shape_base",
    "overrides", "records", "function_base", "getlimits", "arrayprint", "defchararray", "einsumfunc", "memmap",
    "numerictypes",
    ):
        try:
            module = importlib.import_module("numpy.core." + module_name)
            sys.modules.setdefault(
                "numpy._core." + module_name,
                module,
            )
        except Exception:
            pass
except Exception as error:
    print("Warning: NumPy compatibility shim failed:", error)
PYTHON_SHIM

export PYTHONPATH="${RUNTIME_PATCH}:${PYTHONPATH:-}"

python - \
    "${RESULTS_DIR}" \
    "${RUNTIME_DATA}/unique_input.tsv" \
    "${RUNTIME_DATA}/transition_metadata.tsv" \
    "${RUNTIME_DATA}/microstate_metadata.tsv" \
    "${RUNTIME_DATA}/graph_diagnostics.tsv" \
    "${PH}" <<'PYTHON_AGGREGATE'
"""Aggregate five-fold pKa predictions into microstate populations.

For a validated transition acid(q) >> base(q - 1):

    log10(P_base) - log10(P_acid) = pH - pKa

The log-populations are fitted by least squares over each connected component,
which makes the result robust to small inconsistencies in cycles of predicted
pKa values. For a connected molecular graph, RelativePopulation is normalized
over all microstates of that parent molecule.

For disconnected graphs, relative probabilities between graph components are
not determined by the available transitions. The output therefore records only
component-normalized probabilities, while final_states.tsv chooses the most
probable state in the best-supported component deterministically.
"""
import math
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


(
    results_dir,
    unique_input_path,
    transition_metadata_path,
    microstate_metadata_path,
    graph_diagnostics_path,
) = map(Path, sys.argv[1:6])

try:
    ph = float(sys.argv[6])
except ValueError as error:
    raise SystemExit("Aggregation: pH must be numeric.") from error

if not math.isfinite(ph):
    raise SystemExit("Aggregation: pH must be finite.")

unique_input = pd.read_csv(
    unique_input_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

transition_meta = pd.read_csv(
    transition_metadata_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

microstate_meta = pd.read_csv(
    microstate_metadata_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

graph_diagnostics = pd.read_csv(
    graph_diagnostics_path,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

if transition_meta.empty:
    raise SystemExit("Aggregation: transition metadata is empty.")

if microstate_meta.empty:
    raise SystemExit("Aggregation: microstate metadata is empty.")

n_transitions = len(transition_meta)
fold_predictions = []

for fold in range(5):
    fold_dir = results_dir / f"fold_{fold}"
    pkl_files = sorted(fold_dir.glob("*.pkl"))

    if len(pkl_files) != 1:
        raise SystemExit(
            f"Aggregation: fold {fold} output is invalid."
        )

    with pkl_files[0].open("rb") as handle:
        batches = pickle.load(handle)

    predictions = [
        float(batch["predict"][index].cpu().item())
        for batch in batches
        for index in range(int(batch["bsz"]))
    ]

    if len(predictions) != n_transitions:
        raise SystemExit(
            f"Aggregation: fold {fold} prediction count is invalid "
            f"({len(predictions)} != {n_transitions})."
        )

    fold_predictions.append(predictions)

prediction_matrix = np.asarray(fold_predictions, dtype=float)

transition_predictions = transition_meta.copy()
transition_predictions["pKa"] = prediction_matrix.mean(axis=0)
transition_predictions["pKaStdAcrossFolds"] = prediction_matrix.std(axis=0)

transition_predictions.to_csv(
    results_dir / "transition_predictions.tsv",
    sep="\t",
    index=False,
)


def connected_components(node_ids, transitions):
    """Return connected components of the undirected microstate graph."""
    adjacency = {node_id: set() for node_id in node_ids}

    for _, row in transitions.iterrows():
        acid_id = row["AcidMicrostateID"]
        base_id = row["BaseMicrostateID"]

        adjacency.setdefault(acid_id, set()).add(base_id)
        adjacency.setdefault(base_id, set()).add(acid_id)

    seen = set()
    components = []

    for start in adjacency:
        if start in seen:
            continue

        component = []
        queue = deque([start])
        seen.add(start)

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)

        components.append(sorted(component))

    return components


def fit_component_log_populations(component_ids, component_transitions):
    """Fit log10 relative populations for one connected component."""
    ordered_ids = list(component_ids)
    index_by_id = {
        microstate_id: index
        for index, microstate_id in enumerate(ordered_ids)
    }

    if len(ordered_ids) == 1:
        return {ordered_ids[0]: 0.0}, 0.0

    rows = []
    values = []

    for _, transition in component_transitions.iterrows():
        acid_id = transition["AcidMicrostateID"]
        base_id = transition["BaseMicrostateID"]

        equation = np.zeros(len(ordered_ids), dtype=float)
        equation[index_by_id[base_id]] = 1.0
        equation[index_by_id[acid_id]] = -1.0

        rows.append(equation)
        values.append(ph - float(transition["pKa"]))

    # Set an arbitrary reference potential for the component.
    reference = np.zeros(len(ordered_ids), dtype=float)
    reference[0] = 1.0

    rows.append(reference)
    values.append(0.0)

    matrix = np.vstack(rows)
    target = np.asarray(values, dtype=float)

    solution, _, _, _ = np.linalg.lstsq(
        matrix,
        target,
        rcond=None,
    )

    fitted = matrix[:-1] @ solution
    residual = float(
        np.sqrt(np.mean((fitted - target[:-1]) ** 2))
    ) if len(target) > 1 else 0.0

    return {
        microstate_id: float(solution[index_by_id[microstate_id]])
        for microstate_id in ordered_ids
    }, residual


population_rows = []
final_rows = []

diagnostics_by_mol = graph_diagnostics.set_index("MolID").to_dict("index")

for _, input_row in unique_input.iterrows():
    mol_id = input_row["MolID"]
    input_smiles = input_row["SMILES"]

    molecule_microstates = microstate_meta[
        microstate_meta["MolID"] == mol_id
    ].copy()

    molecule_transitions = transition_predictions[
        transition_predictions["MolID"] == mol_id
    ].copy()

    if molecule_microstates.empty:
        final_rows.append({
            "MolID": mol_id,
            "InputSMILES": input_smiles,
            "pH": ph,
            "PreferredMicrostateID": "",
            "PreferredMicrostateSMILES": "",
            "PreferredCharge": np.nan,
            "PreferredRelativePopulation": np.nan,
            "GraphStatus": "NO_MICROSTATES",
            "PreferredSelectionStatus": "NO_MICROSTATES",
        })
        continue

    molecule_microstates["Charge"] = molecule_microstates["Charge"].astype(int)

    node_ids = molecule_microstates["MicrostateID"].tolist()

    components = connected_components(
        node_ids,
        molecule_transitions,
    )

    graph_status = diagnostics_by_mol.get(
        mol_id,
        {},
    ).get(
        "GraphStatus",
        "DISCONNECTED",
    )

    component_summaries = []

    for component_index, component_ids in enumerate(components, start=1):
        component_set = set(component_ids)

        component_transitions = molecule_transitions[
            molecule_transitions["AcidMicrostateID"].isin(component_set)
            & molecule_transitions["BaseMicrostateID"].isin(component_set)
        ].copy()

        log_populations, fit_residual = fit_component_log_populations(
            component_ids,
            component_transitions,
        )

        max_log_population = max(log_populations.values())

        weights = {
            microstate_id: 10.0 ** (
                log_population - max_log_population
            )
            for microstate_id, log_population in log_populations.items()
        }

        total_weight = sum(weights.values())

        component_probability = {
            microstate_id: weights[microstate_id] / total_weight
            for microstate_id in component_ids
        }

        component_summaries.append({
            "ComponentIndex": component_index,
            "MicrostateIDs": component_ids,
            "TransitionCount": len(component_transitions),
            "FitResidualLog10": fit_residual,
            "Probabilities": component_probability,
        })

    is_connected = len(component_summaries) == 1

    # For a connected graph, one normalized probability is defined over all
    # microstates. For disconnected graphs, only within-component probabilities
    # are scientifically identifiable from the supplied transition graph.
    global_probabilities = {}

    if is_connected:
        global_probabilities = component_summaries[0]["Probabilities"]

    for component in component_summaries:
        component_ids = component["MicrostateIDs"]
        probability_map = component["Probabilities"]

        component_microstates = molecule_microstates[
            molecule_microstates["MicrostateID"].isin(component_ids)
        ].copy()

        component_microstates["ComponentIndex"] = component[
            "ComponentIndex"
        ]

        component_microstates["ComponentTransitionCount"] = component[
            "TransitionCount"
        ]

        component_microstates["ComponentFitResidualLog10"] = component[
            "FitResidualLog10"
        ]

        component_microstates[
            "RelativePopulationWithinComponent"
        ] = component_microstates["MicrostateID"].map(probability_map)

        component_microstates["RelativePopulation"] = (
            component_microstates["MicrostateID"].map(
                global_probabilities
            )
            if is_connected
            else np.nan
        )

        component_microstates["GraphStatus"] = (
            "CONNECTED"
            if is_connected
            else "DISCONNECTED"
        )

        component_microstates["pH"] = ph

        population_rows.append(component_microstates)

    # If the graph is disconnected, cross-component populations cannot be
    # determined. Select from the best-supported component: most transitions,
    # then most microstates, then lowest component index.
    preferred_component = sorted(
        component_summaries,
        key=lambda component: (
            -component["TransitionCount"],
            -len(component["MicrostateIDs"]),
            component["ComponentIndex"],
        ),
    )[0]

    preferred_probability_map = preferred_component["Probabilities"]

    preferred_candidates = molecule_microstates[
        molecule_microstates["MicrostateID"].isin(
            preferred_component["MicrostateIDs"]
        )
    ].copy()

    preferred_candidates["SelectionProbability"] = (
        preferred_candidates["MicrostateID"].map(
            preferred_probability_map
        )
    )

    preferred = preferred_candidates.sort_values(
        by=[
            "SelectionProbability",
            "Charge",
            "MicrostateID",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).iloc[0]

    if is_connected:
        selection_status = "CONNECTED_GRAPH"
        preferred_relative_population = float(
            preferred["SelectionProbability"]
        )
    else:
        selection_status = (
            "DISCONNECTED_GRAPH_"
            "PREFERRED_WITHIN_BEST_SUPPORTED_COMPONENT"
        )
        preferred_relative_population = float(
            preferred["SelectionProbability"]
        )

    final_rows.append({
        "MolID": mol_id,
        "InputSMILES": input_smiles,
        "pH": ph,
        "PreferredMicrostateID": preferred["MicrostateID"],
        "PreferredMicrostateSMILES": preferred["MicrostateSMILES"],
        "PreferredCharge": int(preferred["Charge"]),
        "PreferredRelativePopulation": preferred_relative_population,
        "PreferredComponentIndex": int(
            preferred_component["ComponentIndex"]
        ),
        "PreferredComponentTransitionCount": int(
            preferred_component["TransitionCount"]
        ),
        "GraphStatus": (
            "CONNECTED"
            if is_connected
            else "DISCONNECTED"
        ),
        "PreferredSelectionStatus": selection_status,
    })

microstate_populations = pd.concat(
    population_rows,
    ignore_index=True,
) if population_rows else pd.DataFrame()

final_states = pd.DataFrame(final_rows)

microstate_populations.to_csv(
    results_dir / "microstate_populations.tsv",
    sep="\t",
    index=False,
)

final_states.to_csv(
    results_dir / "final_states.tsv",
    sep="\t",
    index=False,
)

print(
    "Wrote transition predictions: "
    f"{results_dir / 'transition_predictions.tsv'}"
)

print(
    "Wrote microstate populations: "
    f"{results_dir / 'microstate_populations.tsv'}"
)

print(
    "Wrote preferred final states: "
    f"{results_dir / 'final_states.tsv'}"
)
PYTHON_AGGREGATE
CONTAINER_AGGREGATION_SCRIPT

required_output_files=(
    transition_predictions.tsv
    microstate_populations.tsv
    final_states.tsv
)

for file in "${required_output_files[@]}"; do
    [[ -f "${TEMP_RESULTS_DIR}/${file}" ]] || {
        echo "Output: missing ${file}." >&2
        exit 1
    }
done

echo
echo "=== Copying results ==="

for file in "${required_output_files[@]}"; do
    cp "${TEMP_RESULTS_DIR}/${file}" "${OUTPUT_DIR}/${file}"
done

echo
echo "=== GPU inference completed ==="
echo "Wrote outputs:"

for file in "${required_output_files[@]}"; do
    echo "  ${OUTPUT_DIR}/${file}"
done
