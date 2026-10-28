#!/usr/bin/env bash
#
# Run five-fold Uni-pKa inference on one selected GPU.
#
# Usage:
#   ./run_unipka_gpu.sh PREPARED_DIR OUTPUT_DIR \
#       [GPU_ID] [pH] [SHM_SIZE] [NUM_WORKERS] [BATCH_SIZE] [DATA_BUFFER_SIZE]
#
# Example:
#   ./run_unipka_gpu.sh prepared_unipka unipka_results 0 7.4 64g 2 1 2
#
# PREPARED_DIR must have been created by prepare_unipka_cpu.sh.
#
# Outputs in OUTPUT_DIR:
#   - transition_predictions.tsv
#   - state_populations.tsv
#   - final_states.tsv

set -euo pipefail

REPO_DIR="${HOME}/.vantami/data/Uni-pKa"
MODEL_DIR="${REPO_DIR}/models"
IMAGE="uni-pka:local"
TASK_NAME="sampl6"

PREPARED_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
GPU_ID="${3:-0}"
PH="${4:-7.4}"
SHM_SIZE="${5:-64g}"
NUM_WORKERS="${6:-2}"
BATCH_SIZE="${7:-1}"
DATA_BUFFER_SIZE="${8:-2}"

usage() {
    cat >&2 <<EOF
Usage:
  $0 PREPARED_DIR OUTPUT_DIR [GPU_ID] [pH] [SHM_SIZE] [NUM_WORKERS] [BATCH_SIZE] [DATA_BUFFER_SIZE]

PREPARED_DIR must be created by prepare_unipka_cpu.sh.

Defaults:
  GPU_ID:           0
  pH:               7.4
  SHM_SIZE:         64g
  NUM_WORKERS:      2
  BATCH_SIZE:       1
  DATA_BUFFER_SIZE: 2

Recommended first run for a NVIDIA A100-SXM4-40GB:
  $0 prepared_unipka unipka_results 0 7.4 64g 2 1 2

EOF
    exit 2
}

[[ -n "${PREPARED_DIR}" && -n "${OUTPUT_DIR}" ]] || usage

[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || {
    echo "Arguments: GPU_ID must be a non-negative integer." >&2
    exit 2
}

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

for file in \
    unique_input.tsv \
    "${TASK_NAME}.tsv" \
    transition_metadata.tsv \
    macrostate_metadata.tsv \
    "${TASK_NAME}.lmdb" \
    dict.txt \
    dict_charge.txt
do
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

PREPARED_DIR="$(cd "${PREPARED_DIR}" && pwd)"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

TEMP_RESULTS_DIR="$(mktemp -d "${OUTPUT_DIR}/.unipka-results.XXXXXX")"
trap 'rm -rf "${TEMP_RESULTS_DIR}"' EXIT

echo "=== Uni-pKa GPU inference ==="
echo "Prepared data:     ${PREPARED_DIR}"
echo "Output:            ${OUTPUT_DIR}"
echo "GPU:               ${GPU_ID}"
echo "pH:                ${PH}"
echo "Shared memory:     ${SHM_SIZE}"
echo "Data workers:      ${NUM_WORKERS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Data buffer size:  ${DATA_BUFFER_SIZE}"

docker run --rm -i \
    --shm-size="${SHM_SIZE}" \
    --gpus "device=${GPU_ID}" \
    -v "${PREPARED_DIR}:/work/runtime_data:ro" \
    -v "${MODEL_DIR}:/work/models:ro" \
    -v "${TEMP_RESULTS_DIR}:/work/results" \
    -e PH="${PH}" \
    -e TASK_NAME="${TASK_NAME}" \
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
    bash -s <<'CONTAINER_SCRIPT'

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

for file in \
    unique_input.tsv \
    "${TASK_NAME}.tsv" \
    transition_metadata.tsv \
    macrostate_metadata.tsv \
    "${TASK_NAME}.lmdb" \
    dict.txt \
    dict_charge.txt
do
    [[ -f "${RUNTIME_DATA}/${file}" ]] || {
        echo "Input: missing ${file} in container." >&2
        exit 1
    }
done

cd "${REPO_DIR}"

echo
echo "=== GPU ==="
nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader

echo
echo "=== Runtime configuration ==="
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DATA_BUFFER_SIZE=${DATA_BUFFER_SIZE}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"

echo
echo "=== NumPy compatibility shim ==="

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

echo
echo "=== Validating checkpoints ==="

for fold in 0 1 2 3 4; do
    checkpoint="/work/models/fold_${fold}/checkpoint_best.pt"

    [[ -f "${checkpoint}" ]] || {
        echo "Models: missing fold ${fold} checkpoint." >&2
        exit 1
    }

    echo "fold_${fold}: OK"
done

echo
echo "=== Running five-fold inference ==="

for fold in 0 1 2 3 4; do
    echo
    echo "===== Fold ${fold} ====="

    python ./unimol/infer.py \
        --user-dir ./unimol \
        "${RUNTIME_DATA}" \
        --task-name "${TASK_NAME}" \
        --valid-subset "${TASK_NAME}" \
        --results-path "${RESULTS_DIR}/fold_${fold}" \
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
        --path "/work/models/fold_${fold}/checkpoint_best.pt" \
        --fp16 \
        --fp16-init-scale 4 \
        --fp16-scale-window 256 \
        --log-interval 50 \
        --log-format simple \
        --required-batch-size-multiple 1
done

echo
echo "=== Aggregating predictions at pH ${PH} ==="

python - \
    "${RESULTS_DIR}" \
    "${RUNTIME_DATA}/unique_input.tsv" \
    "${RUNTIME_DATA}/transition_metadata.tsv" \
    "${RUNTIME_DATA}/macrostate_metadata.tsv" \
    "${PH}" <<'PYTHON_AGGREGATE'
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

results_dir = Path(sys.argv[1])
unique_input_path = Path(sys.argv[2])
transition_metadata_path = Path(sys.argv[3])
macrostate_metadata_path = Path(sys.argv[4])

try:
    ph = float(sys.argv[5])
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

transition_meta = pd.read_csv(transition_metadata_path, sep="\t")
macro_meta = pd.read_csv(macrostate_metadata_path, sep="\t")

n_transitions = len(transition_meta)
fold_predictions = []

for fold in range(5):
    fold_dir = results_dir / f"fold_{fold}"
    pkl_files = sorted(fold_dir.glob("*.pkl"))

    if len(pkl_files) != 1:
        raise SystemExit(f"Aggregation: fold {fold} output is invalid.")

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


def compute_populations(mol_transitions):
    """
    For acid(q+1) >> base(q):
    log10(P_base) - log10(P_acid) = pH - pKa.
    """
    charges = sorted(
        set(mol_transitions["AcidCharge"].astype(int))
        | set(mol_transitions["BaseCharge"].astype(int))
    )

    log_population = {max(charges): 0.0}
    changed = True

    while changed:
        changed = False

        for _, row in mol_transitions.iterrows():
            acid_charge = int(row["AcidCharge"])
            base_charge = int(row["BaseCharge"])
            delta = ph - float(row["pKa"])

            if acid_charge in log_population and base_charge not in log_population:
                log_population[base_charge] = log_population[acid_charge] + delta
                changed = True

            if base_charge in log_population and acid_charge not in log_population:
                log_population[acid_charge] = log_population[base_charge] - delta
                changed = True

    max_log_population = max(log_population.values())

    weights = {
        charge: 10 ** (value - max_log_population)
        for charge, value in log_population.items()
    }

    total_weight = sum(weights.values())

    return pd.DataFrame({
        "Charge": list(weights.keys()),
        "RelativePopulation": [
            weights[charge] / total_weight
            for charge in weights
        ],
    })


population_rows = []
summary_rows = []

for mol_id, mol_transitions in transition_predictions.groupby("MolID", sort=False):
    populations = compute_populations(mol_transitions.copy())
    populations["MolID"] = mol_id

    molecule_macrostates = macro_meta[
        macro_meta["MolID"] == mol_id
    ].copy()

    populations = populations.merge(
        molecule_macrostates,
        on=["MolID", "Charge"],
        how="left",
    )

    population_rows.append(populations)

    dominant_state = populations.sort_values(
        "RelativePopulation",
        ascending=False,
    ).iloc[0]

    relevant_transition = (
        mol_transitions.assign(
            DistanceToPH=(mol_transitions["pKa"] - ph).abs()
        )
        .sort_values("DistanceToPH")
        .iloc[0]
    )

    summary_rows.append({
        "MolID": mol_id,
        "OriginalSMILES": dominant_state.get("OriginalSMILES", ""),
        "pH": ph,
        "DominantCharge": int(dominant_state["Charge"]),
        "DominantRelativePopulation": float(
            dominant_state["RelativePopulation"]
        ),
        "DominantMacrostateSMILES": dominant_state.get(
            "MacrostateSMILES", ""
        ),
        "RelevantTransitionID": relevant_transition["TransitionID"],
        "RelevantAcidCharge": int(relevant_transition["AcidCharge"]),
        "RelevantBaseCharge": int(relevant_transition["BaseCharge"]),
        "RelevantTransitionSMILES": relevant_transition["SMILES"],
        "RelevantPka": float(relevant_transition["pKa"]),
        "RelevantPkaStdAcrossFolds": float(
            relevant_transition["pKaStdAcrossFolds"]
        ),
    })

state_populations = pd.concat(population_rows, ignore_index=True)
molecule_summary = pd.DataFrame(summary_rows)

state_populations.to_csv(
    results_dir / "state_populations.tsv",
    sep="\t",
    index=False,
)

final_states = (
    unique_input[["MolID", "SMILES"]]
    .rename(columns={"SMILES": "InputSMILES"})
    .merge(molecule_summary, on="MolID", how="left")
)

final_states.to_csv(
    results_dir / "final_states.tsv",
    sep="\t",
    index=False,
)

print(f"Wrote transition predictions: {results_dir / 'transition_predictions.tsv'}")
print(f"Wrote state populations:      {results_dir / 'state_populations.tsv'}")
print(f"Wrote final states:           {results_dir / 'final_states.tsv'}")
PYTHON_AGGREGATE

for file in transition_predictions.tsv state_populations.tsv final_states.tsv; do
    [[ -f "${RESULTS_DIR}/${file}" ]] || {
        echo "Output: missing ${file}." >&2
        exit 1
    }
done

echo
echo "=== GPU inference completed ==="
ls -lh "${RESULTS_DIR}"

CONTAINER_SCRIPT

echo
echo "=== Copying results ==="

for file in transition_predictions.tsv state_populations.tsv final_states.tsv; do
    cp "${TEMP_RESULTS_DIR}/${file}" "${OUTPUT_DIR}/${file}"
done

echo "Wrote outputs:"
for file in transition_predictions.tsv state_populations.tsv final_states.tsv; do
    echo "  ${OUTPUT_DIR}/${file}"
done
