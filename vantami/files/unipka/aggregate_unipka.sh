#!/usr/bin/env bash
#
# Aggregate raw five-fold Uni-pKa outputs into transition predictions,
# microstate populations, and preferred states at a requested pH.
#
# Usage:
#   ./aggregate_unipka.sh PREPARED_DIR FOLD_RESULTS_DIR OUTPUT_DIR [pH]
#
# Example:
#   ./aggregate_unipka.sh prepared_unipka unipka_fold_results unipka_results 7.4
#
# FOLD_RESULTS_DIR must contain exactly one *.pkl file in each fold_0 through
# fold_4 directory, as created by run_unipka_gpu_inference_only.sh.
#
# On failure, diagnostics and partial outputs are retained at:
#   OUTPUT_DIR.failed-<timestamp>-<pid>/
#

set -Eeuo pipefail

IMAGE="uni-pka:local"
TASK_NAME="sampl6"

PREPARED_DIR="${1:-}"
FOLD_RESULTS_DIR="${2:-}"
OUTPUT_DIR="${3:-}"
PH="${4:-7.4}"

usage() {
    cat >&2 <<EOF
Usage:
  $0 PREPARED_DIR FOLD_RESULTS_DIR OUTPUT_DIR [pH]

PREPARED_DIR must be created by prepare_unipka.sh
FOLD_RESULTS_DIR must contain raw outputs from all five Uni-pKa folds.
pH must be a finite numeric value; default: 7.4.
EOF
    exit 2
}

host_error() {
    local status=$?
    local line=$1
    printf 'ERROR: host aggregation script failed at line %s (exit status %s).\n' \
        "${line}" "${status}" >&2
    exit "${status}"
}

trap 'host_error $LINENO' ERR

[[ -n "${PREPARED_DIR}" && -n "${FOLD_RESULTS_DIR}" && -n "${OUTPUT_DIR}" ]] || usage

[[ -d "${PREPARED_DIR}" ]] || {
    echo "Input: prepared directory not found: ${PREPARED_DIR}" >&2
    exit 1
}

[[ -d "${FOLD_RESULTS_DIR}" ]] || {
    echo "Input: fold results directory not found: ${FOLD_RESULTS_DIR}" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || {
    echo "Docker: command not found." >&2
    exit 1
}

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
    echo "Docker: image '${IMAGE}' not found." >&2
    exit 1
}

required_prepared_files=(
    unique_input.tsv
    "${TASK_NAME}.tsv"
    transition_metadata.tsv
    microstate_metadata.tsv
    graph_diagnostics.tsv
)

for file in "${required_prepared_files[@]}"; do
    [[ -f "${PREPARED_DIR}/${file}" ]] || {
        echo "Input: missing ${PREPARED_DIR}/${file}" >&2
        exit 1
    }
done

for fold in 0 1 2 3 4; do
    fold_dir="${FOLD_RESULTS_DIR}/fold_${fold}"
    [[ -d "${fold_dir}" ]] || {
        echo "Input: missing fold output directory: ${fold_dir}" >&2
        exit 1
    }

    shopt -s nullglob
    pkl_files=("${fold_dir}"/*.pkl)
    shopt -u nullglob

    [[ "${#pkl_files[@]}" -eq 1 ]] || {
        echo "Input: fold ${fold} must contain exactly one .pkl file; found ${#pkl_files[@]}." >&2
        exit 1
    }
done

PREPARED_DIR="$(cd "${PREPARED_DIR}" && pwd)"
FOLD_RESULTS_DIR="$(cd "${FOLD_RESULTS_DIR}" && pwd)"

OUTPUT_PARENT="$(cd "$(dirname "${OUTPUT_DIR}")" && pwd)"
OUTPUT_NAME="$(basename "${OUTPUT_DIR}")"
OUTPUT_DIR="${OUTPUT_PARENT}/${OUTPUT_NAME}"

RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"
STAGING_DIR="${OUTPUT_PARENT}/.${OUTPUT_NAME}.staging-${RUN_TOKEN}"
FAILED_DIR="${OUTPUT_PARENT}/${OUTPUT_NAME}.failed-${RUN_TOKEN}"
BACKUP_DIR="${OUTPUT_PARENT}/.${OUTPUT_NAME}.backup-${RUN_TOKEN}"

[[ ! -e "${STAGING_DIR}" && ! -e "${FAILED_DIR}" && ! -e "${BACKUP_DIR}" ]] || {
    echo "Output: generated temporary run paths already exist; retry the command." >&2
    exit 1
}

mkdir -p "${STAGING_DIR}"

preserve_failed_staging() {
    local status=$?
    trap - EXIT

    if [[ "${status}" -ne 0 && -d "${STAGING_DIR}" ]]; then
        if mv "${STAGING_DIR}" "${FAILED_DIR}"; then
            printf 'FAILED: retained aggregation diagnostics in %s\n' "${FAILED_DIR}" >&2
        else
            printf 'FAILED: unable to preserve aggregation diagnostics at %s\n' \
                "${FAILED_DIR}" >&2
        fi
    fi

    exit "${status}"
}

trap preserve_failed_staging EXIT

publish_output() {
    local previous_output=0

    if [[ -e "${OUTPUT_DIR}" ]]; then
        mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
        previous_output=1
    fi

    if mv "${STAGING_DIR}" "${OUTPUT_DIR}"; then
        if [[ "${previous_output}" -eq 1 ]]; then
            rm -rf "${BACKUP_DIR}"
        fi
        return 0
    fi

    echo "Output: unable to publish completed aggregation directory." >&2

    if [[ "${previous_output}" -eq 1 && ! -e "${OUTPUT_DIR}" ]]; then
        if ! mv "${BACKUP_DIR}" "${OUTPUT_DIR}"; then
            echo "Output: unable to restore previous output directory." >&2
        fi
    fi

    return 1
}

echo "=== Uni-pKa aggregation ==="
echo "Prepared data:     ${PREPARED_DIR}"
echo "Fold results:      ${FOLD_RESULTS_DIR}"
echo "Output:            ${OUTPUT_DIR}"
echo "Staging:           ${STAGING_DIR}"
echo "pH:                ${PH}"

docker run --rm -i \
    -v "${PREPARED_DIR}:/work/runtime_data:ro" \
    -v "${FOLD_RESULTS_DIR}:/work/fold_results:ro" \
    -v "${STAGING_DIR}:/work/output" \
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
set -Eeuo pipefail

RUNTIME_DATA="/work/runtime_data"
FOLD_RESULTS_DIR="/work/fold_results"
OUTPUT_DIR="/work/output"
RUNTIME_PATCH="/work/runtime_patch"

restore_ownership() {
    local failed=0

    if [[ -d "${OUTPUT_DIR:-}" ]]; then
        if ! chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_DIR}"; then
            echo "ERROR: unable to restore ownership for ${OUTPUT_DIR}." >&2
            failed=1
        fi

        if ! chmod -R u+rwX "${OUTPUT_DIR}"; then
            echo "ERROR: unable to restore permissions for ${OUTPUT_DIR}." >&2
            failed=1
        fi
    fi

    return "${failed}"
}

container_exit() {
    local status=$?
    trap - EXIT

    if ! restore_ownership && [[ "${status}" -eq 0 ]]; then
        status=1
    fi

    exit "${status}"
}

trap container_exit EXIT

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

export PYTHONPATH="${RUNTIME_PATCH}:${PYTHONPATH:-}"

python - \
    "${FOLD_RESULTS_DIR}" \
    "${RUNTIME_DATA}/unique_input.tsv" \
    "${RUNTIME_DATA}/transition_metadata.tsv" \
    "${RUNTIME_DATA}/microstate_metadata.tsv" \
    "${RUNTIME_DATA}/graph_diagnostics.tsv" \
    "${OUTPUT_DIR}" \
    "${PH}" <<'PYTHON_AGGREGATE'
"""Validate and aggregate five-fold Uni-pKa inference outputs.

Each validated acid(q) >> base(q - 1) transition supplies:

    log10(P_base) - log10(P_acid) = pH - pKa

The script validates each serialized fold result before extracting predictions,
uses a preallocated float32 matrix to lower peak memory, writes a manifest for
reproducibility, and logs progress so that I/O, deserialization, and graph
fitting delays are distinguishable. It also writes graph-size diagnostics to
make unusually dense microstate graphs visible.
"""
import csv
import math
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

(
    fold_results_dir,
    unique_input_path,
    transition_metadata_path,
    microstate_metadata_path,
    graph_diagnostics_path,
    output_dir,
) = map(Path, sys.argv[1:7])

try:
    ph = float(sys.argv[7])
except ValueError as error:
    raise SystemExit("Aggregation: pH must be numeric.") from error

if not math.isfinite(ph):
    raise SystemExit("Aggregation: pH must be finite.")

output_dir.mkdir(parents=True, exist_ok=True)
log_path = output_dir / "aggregation.log"
manifest_path = output_dir / "fold_result_manifest.tsv"
graph_fit_path = output_dir / "graph_fit_diagnostics.tsv"


def log(message):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_tsv(path, label):
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as error:
        raise SystemExit(f"Aggregation: unable to read {label} ({path}): {error}.") from error
    log(f"Loaded {label}: {len(frame)} rows from {path.name}.")
    return frame


def tensor_or_array_to_vector(value, fold, batch_index):
    """Convert supported prediction containers to a finite 1D float32 array."""
    source_type = type(value).__name__

    # PyTorch is intentionally optional: infer.py outputs can be tensors,
    # NumPy arrays, Python lists, or scalar values depending on UniMol version.
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception as error:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: could not detach prediction "
                f"object of type {source_type}: {error}."
            ) from error

    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception as error:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: could not move prediction "
                f"object of type {source_type} to CPU: {error}."
            ) from error

    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception as error:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: could not convert prediction "
                f"object of type {source_type} to NumPy: {error}."
            ) from error

    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception as error:
        raise RuntimeError(
            f"Fold {fold}, batch {batch_index}: unsupported prediction object "
            f"of type {source_type}: {error}."
        ) from error

    array = np.ravel(array)

    if not np.isfinite(array).all():
        invalid_count = int((~np.isfinite(array)).sum())
        raise RuntimeError(
            f"Fold {fold}, batch {batch_index}: prediction vector contains "
            f"{invalid_count} non-finite values."
        )

    return array


def load_fold_predictions(pickle_path, fold, expected_count):
    """Load one fold incrementally into a preallocated vector with checks."""
    size_bytes = pickle_path.stat().st_size
    log(
        f"Fold {fold}: loading {pickle_path.name} "
        f"({size_bytes / (1024 ** 2):.1f} MiB)."
    )

    start = time.monotonic()
    try:
        with pickle_path.open("rb") as handle:
            batches = pickle.load(handle)
    except Exception as error:
        raise RuntimeError(
            f"Fold {fold}: unable to deserialize {pickle_path}: {error}. "
            "The raw fold directory has been preserved for inspection."
        ) from error

    log(
        f"Fold {fold}: deserialized object type {type(batches).__name__} "
        f"in {time.monotonic() - start:.1f} s."
    )

    if not isinstance(batches, (list, tuple)):
        raise RuntimeError(
            f"Fold {fold}: expected a list/tuple of batch records, got "
            f"{type(batches).__name__}."
        )

    vector = np.empty(expected_count, dtype=np.float32)
    offset = 0

    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: expected a dict, got "
                f"{type(batch).__name__}."
            )

        if "predict" not in batch or "bsz" not in batch:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: missing required keys "
                f"'predict' and/or 'bsz'; available keys: {sorted(batch.keys())}."
            )

        try:
            batch_size = int(batch["bsz"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: invalid bsz={batch['bsz']!r}."
            ) from error

        if batch_size < 0:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: bsz must not be negative."
            )

        values = tensor_or_array_to_vector(batch["predict"], fold, batch_index)

        if len(values) != batch_size:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: prediction length {len(values)} "
                f"does not match bsz {batch_size}."
            )

        if offset + batch_size > expected_count:
            raise RuntimeError(
                f"Fold {fold}, batch {batch_index}: cumulative prediction count "
                f"would exceed expected transition count {expected_count}."
            )

        vector[offset:offset + batch_size] = values
        offset += batch_size

    # Release the often-large deserialized batch structure before moving on.
    del batches

    if offset != expected_count:
        raise RuntimeError(
            f"Fold {fold}: extracted {offset} predictions; expected {expected_count}."
        )

    log(f"Fold {fold}: validated and extracted {offset} predictions.")
    return vector, size_bytes


def connected_components(node_ids, transitions):
    adjacency = {node_id: set() for node_id in node_ids}

    for row in transitions.itertuples(index=False):
        adjacency[row.AcidMicrostateID].add(row.BaseMicrostateID)
        adjacency[row.BaseMicrostateID].add(row.AcidMicrostateID)

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
    """Fit log10 populations with a dense least-squares system.

    A warning is recorded for large systems. This preserves the scientific
    behavior of the original method while identifying graphs that should be
    migrated to a sparse incidence-matrix implementation if needed.
    """
    ordered_ids = list(component_ids)
    n_nodes = len(ordered_ids)
    n_edges = len(component_transitions)
    index_by_id = {microstate_id: index for index, microstate_id in enumerate(ordered_ids)}

    if n_nodes == 1:
        return {ordered_ids[0]: 0.0}, 0.0, n_nodes, n_edges, False

    dense_elements = (n_edges + 1) * n_nodes
    large_dense_fit = dense_elements > 5_000_000

    matrix = np.zeros((n_edges + 1, n_nodes), dtype=np.float64)
    target = np.empty(n_edges + 1, dtype=np.float64)

    for row_index, row in enumerate(component_transitions.itertuples(index=False)):
        matrix[row_index, index_by_id[row.BaseMicrostateID]] = 1.0
        matrix[row_index, index_by_id[row.AcidMicrostateID]] = -1.0
        target[row_index] = ph - float(row.pKa)

    # Set one reference potential, fixing the otherwise arbitrary offset.
    matrix[-1, 0] = 1.0
    target[-1] = 0.0

    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    fitted = matrix[:-1] @ solution
    residual = float(np.sqrt(np.mean((fitted - target[:-1]) ** 2))) if n_edges else 0.0

    return (
        {microstate_id: float(solution[index_by_id[microstate_id]]) for microstate_id in ordered_ids},
        residual,
        n_nodes,
        n_edges,
        large_dense_fit,
    )


log("Aggregation started.")
unique_input = read_tsv(unique_input_path, "unique input")
transition_meta = read_tsv(transition_metadata_path, "transition metadata")
microstate_meta = read_tsv(microstate_metadata_path, "microstate metadata")
graph_diagnostics = read_tsv(graph_diagnostics_path, "graph diagnostics")

if transition_meta.empty:
    raise SystemExit("Aggregation: transition metadata is empty.")

if microstate_meta.empty:
    raise SystemExit("Aggregation: microstate metadata is empty.")

required_transition_columns = {
    "MolID", "AcidMicrostateID", "BaseMicrostateID", "DatasetIndex"
}
missing_transition_columns = sorted(required_transition_columns - set(transition_meta.columns))
if missing_transition_columns:
    raise SystemExit(
        f"Aggregation: transition metadata is missing required columns "
        f"{missing_transition_columns}."
    )

required_microstate_columns = {"MolID", "MicrostateID", "MicrostateSMILES", "Charge"}
missing_microstate_columns = sorted(required_microstate_columns - set(microstate_meta.columns))
if missing_microstate_columns:
    raise SystemExit(
        f"Aggregation: microstate metadata is missing required columns "
        f"{missing_microstate_columns}."
    )

if len(transition_meta) != len(transition_meta["DatasetIndex"].unique()):
    raise SystemExit("Aggregation: DatasetIndex values are not unique.")

try:
    dataset_indices = transition_meta["DatasetIndex"].astype(int).to_numpy()
except ValueError as error:
    raise SystemExit("Aggregation: DatasetIndex contains non-integer values.") from error

expected_indices = np.arange(len(transition_meta), dtype=int)
if not np.array_equal(np.sort(dataset_indices), expected_indices):
    raise SystemExit(
        "Aggregation: DatasetIndex must be a complete zero-based sequence "
        "matching transition TSV order."
    )

n_transitions = len(transition_meta)
log(f"Validated metadata: {n_transitions} transitions, {len(microstate_meta)} microstates, {len(unique_input)} molecules.")

prediction_matrix = np.empty((5, n_transitions), dtype=np.float32)
manifest_rows = []

for fold in range(5):
    fold_dir = fold_results_dir / f"fold_{fold}"
    pkl_files = sorted(fold_dir.glob("*.pkl"))

    if len(pkl_files) != 1:
        raise SystemExit(
            f"Aggregation: fold {fold} must contain exactly one .pkl file; "
            f"found {len(pkl_files)}."
        )

    vector, size_bytes = load_fold_predictions(pkl_files[0], fold, n_transitions)
    prediction_matrix[fold, :] = vector
    del vector

    manifest_rows.append({
        "Fold": fold,
        "PredictionFile": str(pkl_files[0]),
        "FileSizeBytes": size_bytes,
        "PredictionCount": n_transitions,
        "MinPrediction": float(prediction_matrix[fold, :].min()),
        "MaxPrediction": float(prediction_matrix[fold, :].max()),
    })

pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
log(f"Validated all five folds and wrote {manifest_path.name}.")

transition_predictions = transition_meta.copy()
transition_predictions["pKa"] = prediction_matrix.mean(axis=0, dtype=np.float64)
transition_predictions["pKaStdAcrossFolds"] = prediction_matrix.std(axis=0, dtype=np.float64)

if not np.isfinite(transition_predictions["pKa"].to_numpy(dtype=float)).all():
    raise SystemExit("Aggregation: mean pKa values contain non-finite values.")

transition_predictions.to_csv(output_dir / "transition_predictions.tsv", sep="\t", index=False)
log("Wrote transition_predictions.tsv.")

# No longer needed after pKa mean and standard deviation have been calculated.
del prediction_matrix

microstate_meta = microstate_meta.copy()
try:
    microstate_meta["Charge"] = microstate_meta["Charge"].astype(int)
except ValueError as error:
    raise SystemExit("Aggregation: microstate metadata contains non-integer charges.") from error

if graph_diagnostics["MolID"].duplicated().any():
    raise SystemExit("Aggregation: graph diagnostics contains duplicate MolID values.")

diagnostics_by_mol = graph_diagnostics.set_index("MolID").to_dict("index")

population_rows = []
final_rows = []
graph_fit_rows = []

for molecule_number, input_row in enumerate(unique_input.itertuples(index=False), start=1):
    mol_id = input_row.MolID
    input_smiles = input_row.SMILES

    if molecule_number == 1 or molecule_number % 100 == 0 or molecule_number == len(unique_input):
        log(f"Fitting populations for molecule {molecule_number}/{len(unique_input)}: {mol_id}.")

    molecule_microstates = microstate_meta[microstate_meta["MolID"] == mol_id].copy()
    molecule_transitions = transition_predictions[transition_predictions["MolID"] == mol_id].copy()

    if molecule_microstates.empty:
        final_rows.append({
            "MolID": mol_id,
            "InputSMILES": input_smiles,
            "pH": ph,
            "PreferredMicrostateID": "",
            "PreferredMicrostateSMILES": "",
            "PreferredCharge": np.nan,
            "PreferredRelativePopulation": np.nan,
            "PreferredComponentIndex": np.nan,
            "PreferredComponentTransitionCount": 0,
            "GraphStatus": "NO_MICROSTATES",
            "PreferredSelectionStatus": "NO_MICROSTATES",
        })
        continue

    node_ids = molecule_microstates["MicrostateID"].tolist()
    components = connected_components(node_ids, molecule_transitions)
    component_summaries = []

    for component_index, component_ids in enumerate(components, start=1):
        component_set = set(component_ids)
        component_transitions = molecule_transitions[
            molecule_transitions["AcidMicrostateID"].isin(component_set)
            & molecule_transitions["BaseMicrostateID"].isin(component_set)
        ].copy()

        start = time.monotonic()
        (
            log_populations,
            fit_residual,
            n_nodes,
            n_edges,
            large_dense_fit,
        ) = fit_component_log_populations(component_ids, component_transitions)
        fit_seconds = time.monotonic() - start

        if large_dense_fit:
            log(
                f"WARNING: {mol_id} component {component_index} uses a large dense "
                f"least-squares system ({n_edges + 1} x {n_nodes}); fit took {fit_seconds:.1f} s."
            )

        max_log_population = max(log_populations.values())
        weights = {
            microstate_id: 10.0 ** (log_population - max_log_population)
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
            "TransitionCount": n_edges,
            "FitResidualLog10": fit_residual,
            "Probabilities": component_probability,
        })

        graph_fit_rows.append({
            "MolID": mol_id,
            "ComponentIndex": component_index,
            "MicrostateCount": n_nodes,
            "TransitionCount": n_edges,
            "DenseMatrixRows": n_edges + 1,
            "DenseMatrixColumns": n_nodes,
            "LargeDenseFitWarning": large_dense_fit,
            "FitResidualLog10": fit_residual,
            "FitSeconds": fit_seconds,
        })

    is_connected = len(component_summaries) == 1
    original_graph_status = diagnostics_by_mol.get(mol_id, {}).get("GraphStatus", "UNKNOWN")
    graph_status = "CONNECTED" if is_connected else original_graph_status
    if not is_connected and graph_status == "CONNECTED":
        graph_status = "DISCONNECTED"

    global_probabilities = component_summaries[0]["Probabilities"] if is_connected else {}

    for component in component_summaries:
        component_ids = component["MicrostateIDs"]
        component_microstates = molecule_microstates[
            molecule_microstates["MicrostateID"].isin(component_ids)
        ].copy()

        component_microstates["ComponentIndex"] = component["ComponentIndex"]
        component_microstates["ComponentTransitionCount"] = component["TransitionCount"]
        component_microstates["ComponentFitResidualLog10"] = component["FitResidualLog10"]
        component_microstates["RelativePopulationWithinComponent"] = component_microstates["MicrostateID"].map(component["Probabilities"])
        component_microstates["RelativePopulation"] = component_microstates["MicrostateID"].map(global_probabilities) if is_connected else np.nan
        component_microstates["GraphStatus"] = graph_status
        component_microstates["pH"] = ph
        population_rows.append(component_microstates)

    preferred_component = sorted(
        component_summaries,
        key=lambda component: (
            -component["TransitionCount"],
            -len(component["MicrostateIDs"]),
            component["ComponentIndex"],
        ),
    )[0]

    preferred_candidates = molecule_microstates[
        molecule_microstates["MicrostateID"].isin(preferred_component["MicrostateIDs"])
    ].copy()
    preferred_candidates["SelectionProbability"] = preferred_candidates["MicrostateID"].map(preferred_component["Probabilities"])

    preferred = preferred_candidates.sort_values(
        by=["SelectionProbability", "Charge", "MicrostateID"],
        ascending=[False, False, True],
    ).iloc[0]

    final_rows.append({
        "MolID": mol_id,
        "InputSMILES": input_smiles,
        "pH": ph,
        "PreferredMicrostateID": preferred["MicrostateID"],
        "PreferredMicrostateSMILES": preferred["MicrostateSMILES"],
        "PreferredCharge": int(preferred["Charge"]),
        "PreferredRelativePopulation": float(preferred["SelectionProbability"]),
        "PreferredComponentIndex": int(preferred_component["ComponentIndex"]),
        "PreferredComponentTransitionCount": int(preferred_component["TransitionCount"]),
        "GraphStatus": graph_status,
        "PreferredSelectionStatus": (
            "CONNECTED_GRAPH"
            if is_connected
            else "DISCONNECTED_GRAPH_PREFERRED_WITHIN_BEST_SUPPORTED_COMPONENT"
        ),
    })

microstate_populations = pd.concat(population_rows, ignore_index=True) if population_rows else pd.DataFrame()
final_states = pd.DataFrame(final_rows)
graph_fit_diagnostics = pd.DataFrame(graph_fit_rows)

microstate_populations.to_csv(output_dir / "microstate_populations.tsv", sep="\t", index=False)
final_states.to_csv(output_dir / "final_states.tsv", sep="\t", index=False)
graph_fit_diagnostics.to_csv(graph_fit_path, sep="\t", index=False)

log("Wrote microstate_populations.tsv.")
log("Wrote final_states.tsv.")
log(f"Wrote {graph_fit_path.name}.")
log("Aggregation completed successfully.")
PYTHON_AGGREGATE
CONTAINER_AGGREGATION_SCRIPT

required_output_files=(
    transition_predictions.tsv
    microstate_populations.tsv
    final_states.tsv
    fold_result_manifest.tsv
    graph_fit_diagnostics.tsv
    aggregation.log
)

for file in "${required_output_files[@]}"; do
    [[ -f "${STAGING_DIR}/${file}" ]] || {
        echo "Output: missing ${file}." >&2
        exit 1
    }
done

publish_output

echo
echo "=== Uni-pKa aggregation completed ==="
echo "Wrote outputs:"
for file in "${required_output_files[@]}"; do
    echo "  ${OUTPUT_DIR}/${file}"
done

