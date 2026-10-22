#!/usr/bin/env bash
#
# Run Uni-pKa inference on a user-supplied TSV file.
#
# Usage:
#   ./run_unipka.sh ./input.tsv ./output_dir [gpu_id] [pH]
#
# Input requirements:
#   - Tab-separated .tsv file
#   - Header row containing a column named exactly: SMILES
#
# Output:
#   - OUTPUT_DIR/transition_predictions.tsv
#   - OUTPUT_DIR/state_populations.tsv
#   - OUTPUT_DIR/final_states.tsv

set -euo pipefail

REPO_DIR="${HOME}/.vantami/data/Uni-pKa"
MODEL_DIR="${REPO_DIR}/models"
IMAGE="uni-pka:local"

INPUT_TSV="${1:-}"
OUTPUT_DIR="${2:-}"
GPU_ID="${3:-0}"
PH="${4:-7.4}"


usage() {
    echo "Usage: $0 INPUT_TSV OUTPUT_DIR [GPU_ID] [pH]" >&2
    echo "  INPUT_TSV:  Tab-separated .tsv file containing a SMILES column." >&2
    echo "  OUTPUT_DIR: Destination directory for Uni-pKa result TSV files." >&2
    echo "  GPU_ID:     Optional non-negative GPU index; defaults to 0." >&2
    echo "  pH:         Optional pH for dominant-state estimate; defaults to 7.4." >&2
    exit 2
}

[[ -n "${INPUT_TSV}" && -n "${OUTPUT_DIR}" ]] || usage

[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: GPU ID must be a non-negative integer; received: ${GPU_ID}" >&2
    exit 2
}

[[ "${INPUT_TSV}" == *.tsv ]] || {
    echo "ERROR: INPUT_TSV must have a .tsv extension: ${INPUT_TSV}" >&2
    exit 2
}

[[ -f "${INPUT_TSV}" ]] || {
    echo "ERROR: Input TSV does not exist: ${INPUT_TSV}" >&2
    exit 1
}

INPUT_DIR="$(cd "$(dirname "${INPUT_TSV}")" && pwd)"
INPUT_FILE="$(basename "${INPUT_TSV}")"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

echo "=== Host-side validation ==="

command -v docker >/dev/null 2>&1 || {
    echo "ERROR: Docker is not available in PATH." >&2
    exit 1
}

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
    echo "ERROR: Docker image '${IMAGE}' is not available." >&2
    exit 1
}

for fold in 0 1 2 3 4; do
    checkpoint="${MODEL_DIR}/fold_${fold}/checkpoint_best.pt"
    [[ -f "${checkpoint}" ]] || {
        echo "ERROR: Missing checkpoint: ${checkpoint}" >&2
        exit 1
    }
done

TEMP_RESULTS_DIR="$(mktemp -d "${OUTPUT_DIR}/.unipka-results.XXXXXX")"

echo "Docker image: ${IMAGE}"
echo "Physical host GPU selected: ${GPU_ID}"
echo "Input TSV: ${INPUT_TSV}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Models: ${MODEL_DIR}"
echo "pH: ${PH}"
echo

docker run --rm -i \
    --gpus "\"device=${GPU_ID}\"" \
    -v "${INPUT_DIR}:/work/input:ro" \
    -v "${MODEL_DIR}:/work/models:ro" \
    -v "${TEMP_RESULTS_DIR}:/work/results" \
    -e INPUT_FILE="${INPUT_FILE}" \
    -e PH="${PH}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${IMAGE}" \
    bash -s <<'CONTAINER_SCRIPT'

set -euo pipefail

REPO_DIR="/opt/uni-pka"
RUNTIME_DATA="/work/runtime_data"
RUNTIME_PATCH="/work/runtime_patch"
RESULTS_DIR="/work/results"
TASK_NAME="sampl6"
INPUT_TSV="/work/input/${INPUT_FILE}"

echo "=== Container started ==="

[[ -d "${REPO_DIR}" ]] || {
    echo "ERROR: ${REPO_DIR} does not exist inside the image." >&2
    exit 1
}

[[ -f "${INPUT_TSV}" ]] || {
    echo "ERROR: Input TSV is unavailable inside the container: ${INPUT_TSV}" >&2
    exit 1
}

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
while downloaded checkpoints may reference numpy._core during torch.load().
"""
import importlib
import sys

try:
    import numpy
    import numpy.core

    sys.modules.setdefault("numpy._core", numpy.core)

    for module_name in ("multiarray", "_multiarray_umath", "umath", "numeric", "fromnumeric", "shape_base",
                        "overrides", "records", "function_base", "getlimits", "arrayprint", "defchararray",
                        "einsumfunc", "memmap", "numerictypes"):
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
echo "=== Preparing unique molecule table ==="

rm -rf "${RUNTIME_DATA}"
mkdir -p "${RUNTIME_DATA}"

python - "${INPUT_TSV}" "${RUNTIME_DATA}/unique_input.tsv" "${RUNTIME_DATA}/input_row_map.tsv" <<'PYTHON_PREPARE'
import sys
import pandas as pd
from pathlib import Path

input_path = Path(sys.argv[1])
unique_output = Path(sys.argv[2])
row_map_output = Path(sys.argv[3])

try:
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
except Exception as error:
    raise SystemExit(f"ERROR: Could not read input TSV: {error}")

if "SMILES" not in df.columns:
    raise SystemExit(f"INPUT ERROR: Input file must contain < SMILES > column. Found: {df.columns} instead.")

if df.empty:
    raise SystemExit("ERROR ERROR: Input file is empty.")

unique_smiles = list(dict.fromkeys(df["SMILES"].tolist()))
mol_ids = {smi: f"mol_{i:06d}" for i, smi in enumerate(unique_smiles)}

unique_df = pd.DataFrame({
        "MolID": [mol_ids[smi] for smi in unique_smiles],
        "SMILES": unique_smiles,
        "TARGET": -1.0,
    })

row_map = pd.DataFrame({
        "InputRow": list(range(len(df))),
        "MolID": [mol_ids[smi] for smi in df["SMILES"]],
        "InputSMILES": df["SMILES"],
    })

unique_df.to_csv(unique_output, sep="\t", index=False)
row_map.to_csv(row_map_output, sep="\t", index=False)

print(f"Input rows: {len(df)}")
print(f"Unique SMILES: {len(unique_df)}")
PYTHON_PREPARE

echo
echo "=== Enumerating protonation ensembles ==="

python enumerator/main.py ensemble \
    --input "${RUNTIME_DATA}/unique_input.tsv" \
    --output "${RUNTIME_DATA}/ensemble.tsv" \
    --upper 2 \
    --lower -2 \
    --maxiter 32 \
    --template /opt/uni-pka/enumerator/simple_smarts_pattern.tsv

echo
echo "=== Converting ensemble charge states into Uni-pKa acid>>base transitions ==="

python - \
    "${RUNTIME_DATA}/unique_input.tsv" \
    "${RUNTIME_DATA}/ensemble.tsv" \
    "${RUNTIME_DATA}/${TASK_NAME}.tsv" \
    "${RUNTIME_DATA}/transition_metadata.tsv" \
    "${RUNTIME_DATA}/macrostate_metadata.tsv" <<'PYTHON_TRANSITIONS'
import sys
from pathlib import Path
import pandas as pd

unique_path = Path(sys.argv[1])
ensemble_path = Path(sys.argv[2])
pka_input_path = Path(sys.argv[3])
transition_metadata_path = Path(sys.argv[4])
macrostate_metadata_path = Path(sys.argv[5])

unique_df = pd.read_csv(unique_path, sep="\t", dtype=str, keep_default_na=False)
ensemble = pd.read_csv(ensemble_path, sep="\t", dtype=str, keep_default_na=False)

# enumerator/main.py writes an index column; drop it if present.
unnamed_cols = [c for c in ensemble.columns if c.startswith("Unnamed") or c == ""]
if unnamed_cols:
    ensemble = ensemble.drop(columns=unnamed_cols)

if len(ensemble) != len(unique_df):
    raise SystemExit(f"PROCESSING ERROR: Ensemble count {len(ensemble)} does not match input  {len(unique_df)}.")

charge_min = -2
charge_max = 2

def clean_state(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text

transition_rows = []
pka_rows = []
macrostate_rows = []

for idx, row in ensemble.iterrows():
    mol_id = unique_df.loc[idx, "MolID"]
    original_smiles = unique_df.loc[idx, "SMILES"]

    states = {}
    for charge in range(charge_min, charge_max + 1):
        state = clean_state(row.get(str(charge)))
        states[charge] = state

        if state is not None:
            macrostate_rows.append({
                    "MolID": mol_id,
                    "OriginalSMILES": original_smiles,
                    "Charge": charge,
                    "MacrostateSMILES": state,
                })

    for base_charge in range(charge_min, charge_max):
        acid_charge = base_charge + 1

        acid_states = states.get(acid_charge)
        base_states = states.get(base_charge)

        if acid_states is None or base_states is None:
            continue

        transition_id = f"{mol_id}_q{acid_charge}_to_q{base_charge}"
        transition_smiles = f"{acid_states}>>{base_states}"

        transition_rows.append({
                "TransitionID": transition_id,
                "MolID": mol_id,
                "OriginalSMILES": original_smiles,
                "AcidCharge": acid_charge,
                "BaseCharge": base_charge,
                "SMILES": transition_smiles,
            })

        # preprocess_pka.py keeps only SMILES and TARGET.
        pka_rows.append({
                "SMILES": transition_smiles,
                "TARGET": -1.0,
            })

if not pka_rows:
    raise SystemExit("ERROR: No adjacent acid/base transitions generated. Cannot run Uni-pKa pKa inference.")

pka_input = pd.DataFrame(pka_rows)
transition_metadata = pd.DataFrame(transition_rows)
macrostate_metadata = pd.DataFrame(macrostate_rows)

pka_input.to_csv(pka_input_path, sep="\t", index=False)
transition_metadata.to_csv(transition_metadata_path, sep="\t", index=False)
macrostate_metadata.to_csv(macrostate_metadata_path, sep="\t", index=False)

print(f"Generated pKa transition rows: {len(pka_input)}")
print(f"Generated macrostate rows: {len(macrostate_metadata)}")
PYTHON_TRANSITIONS

echo
echo "=== Generating conformers and LMDB data ==="

python scripts/preprocess_pka.py \
    --raw-csv-file "${RUNTIME_DATA}/${TASK_NAME}.tsv" \
    --processed-lmdb-dir "${RUNTIME_DATA}" \
    --task-name "${TASK_NAME}" \
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
echo "=== Running Uni-pKa inference: five folds ==="

for fold in 0 1 2 3 4; do
    echo
    echo "===== Fold ${fold} started ====="

    python ./unimol/infer.py \
        --user-dir ./unimol \
        "${RUNTIME_DATA}" \
        --task-name "${TASK_NAME}" \
        --valid-subset "${TASK_NAME}" \
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
echo "=== Aggregating fold predictions and estimating dominant state at pH ${PH} ==="

python - \
    "${RESULTS_DIR}" \
    "${RUNTIME_DATA}/transition_metadata.tsv" \
    "${RUNTIME_DATA}/macrostate_metadata.tsv" \
    "${RUNTIME_DATA}/input_row_map.tsv" \
    "${PH}" <<'PYTHON_PH'
import sys
import pickle
from pathlib import Path
import math
import numpy as np
import pandas as pd

results_dir = Path(sys.argv[1])
transition_metadata_path = Path(sys.argv[2])
macrostate_metadata_path = Path(sys.argv[3])
input_row_map_path = Path(sys.argv[4])
ph = float(sys.argv[5])

transition_meta = pd.read_csv(transition_metadata_path, sep="\t")
macro_meta = pd.read_csv(macrostate_metadata_path, sep="\t")
input_row_map = pd.read_csv(input_row_map_path, sep="\t")

n_transitions = len(transition_meta)

fold_predictions = []

for fold in range(5):
    fold_dir = results_dir / f"fold_{fold}"
    pkl_files = sorted(fold_dir.glob("*.pkl"))

    if len(pkl_files) != 1:
        raise SystemExit(f"ERROR: Expected one .pkl file in {fold_dir}, found {len(pkl_files)}.")

    with pkl_files[0].open("rb") as handle:
        batches = pickle.load(handle)

    preds = []
    smiles_seen = []

    for batch in batches:
        bsz = int(batch["bsz"])
        for i in range(bsz):
            preds.append(float(batch["predict"][i].cpu().item()))
            smiles_seen.append(batch["smi_name"][i])

    if len(preds) != n_transitions:
        raise SystemExit(f"ERROR: Fold {fold} produced {len(preds)} predictions, but {n_transitions} rows were expected.")

    fold_predictions.append(preds)

prediction_matrix = np.array(fold_predictions, dtype=float)
mean_pka = prediction_matrix.mean(axis=0)
std_pka = prediction_matrix.std(axis=0)

transition_predictions = transition_meta.copy()
transition_predictions["pKa"] = mean_pka
transition_predictions["pKaStdAcrossFolds"] = std_pka

transition_predictions.to_csv(results_dir / "transition_predictions.tsv", sep="\t", index=False)

def compute_populations_for_molecule(mol_transitions: pd.DataFrame) -> pd.DataFrame:
    """
    For transition acid(q+1) >> base(q):
      log10(P_base) - log10(P_acid) = pH - pKa
    """
    charges = sorted(
        set(mol_transitions["AcidCharge"].astype(int)) | set(mol_transitions["BaseCharge"].astype(int))
    )

    # Use the highest charge as reference when available
    reference_charge = max(charges)
    logp = {reference_charge: 0.0}

    changed = True
    while changed:
        changed = False

        for _, row in mol_transitions.iterrows():
            acid_q = int(row["AcidCharge"])
            base_q = int(row["BaseCharge"])
            pka = float(row["pKa"])

            delta = ph - pka

            if acid_q in logp and base_q not in logp:
                logp[base_q] = logp[acid_q] + delta
                changed = True

            if base_q in logp and acid_q not in logp:
                logp[acid_q] = logp[base_q] - delta
                changed = True

    # If disconnected components exist, ignore unreachable states for the
    # distribution and report only the connected protonation ladder.
    max_logp = max(logp.values())
    weights = {charge: 10 ** (value - max_logp) for charge, value in logp.items()}
    total = sum(weights.values())

    return pd.DataFrame({
            "Charge": list(weights.keys()),
            "RelativePopulation": [weights[c] / total for c in weights],
        })

population_rows = []
summary_rows = []

for mol_id, mol_transitions in transition_predictions.groupby("MolID", sort=False):
    mol_transitions = mol_transitions.copy()

    pop = compute_populations_for_molecule(mol_transitions)
    pop["MolID"] = mol_id

    mol_macro = macro_meta[macro_meta["MolID"] == mol_id].copy()
    pop = pop.merge(mol_macro, on=["MolID", "Charge"], how="left")

    population_rows.append(pop)

    dominant = pop.sort_values("RelativePopulation", ascending=False).iloc[0]

    relevant = (
        mol_transitions.assign(DistanceToPH=(mol_transitions["pKa"] - ph).abs())
        .sort_values("DistanceToPH")
        .iloc[0]
    )

    summary_rows.append(
        {
            "MolID": mol_id,
            "OriginalSMILES": dominant.get("OriginalSMILES", ""),
            "pH": ph,
            "DominantCharge": int(dominant["Charge"]),
            "DominantRelativePopulation": float(dominant["RelativePopulation"]),
            "DominantMacrostateSMILES": dominant.get("MacrostateSMILES", ""),
            "RelevantTransitionID": relevant["TransitionID"],
            "RelevantAcidCharge": int(relevant["AcidCharge"]),
            "RelevantBaseCharge": int(relevant["BaseCharge"]),
            "RelevantTransitionSMILES": relevant["SMILES"],
            "RelevantPka": float(relevant["pKa"]),
            "RelevantPkaStdAcrossFolds": float(relevant["pKaStdAcrossFolds"]),
        }
    )

state_populations = pd.concat(population_rows, ignore_index=True)
molecule_summary = pd.DataFrame(summary_rows)

state_populations.to_csv(results_dir / "state_populations.tsv", sep="\t", index=False)

molecule_summary.to_csv(results_dir / "molecule_summary_unique.tsv", sep="\t", index=False)

# Expand back to the original input rows, so duplicated SMILES are preserved.
final_by_input_row = input_row_map.merge(molecule_summary, on="MolID", how="left")

final_by_input_row.to_csv(results_dir / "final_states.tsv", sep="\t", index=False)

print(f"Wrote transition predictions: {results_dir / 'transition_predictions.tsv'}")
print(f"Wrote state populations:      {results_dir / 'state_populations.tsv'}")
print(f"Wrote final states:           {results_dir / 'final_states.tsv'}")
PYTHON_PH

[[ -f "${RESULTS_DIR}/final_states.tsv" ]] || {
    echo "ERROR: Final state output final_states.tsv was not created." >&2
    exit 1
}

echo
echo "=== Container inference completed successfully ==="
echo "Files written:"
ls -lh "${RESULTS_DIR}"

echo
echo "=== Returning temporary result ownership to host user ==="
chown -R "${HOST_UID}:${HOST_GID}" "${RESULTS_DIR}"

CONTAINER_SCRIPT

RESULT_FILES=(
    "transition_predictions.tsv"
    "state_populations.tsv"
    "final_states.tsv"
)

for result_file in "${RESULT_FILES[@]}"; do
    source_file="${TEMP_RESULTS_DIR}/${result_file}"
    [[ -f "${source_file}" ]] || {
        echo "ERROR: Expected result is missing: ${source_file}" >&2
        exit 1
    }
done

echo
echo "=== Copying Uni-pKa result TSV files ==="

for result_file in "${RESULT_FILES[@]}"; do
    cp "${TEMP_RESULTS_DIR}/${result_file}" "${OUTPUT_DIR}/${result_file}"
done

echo "Wrote TSV outputs:"
for result_file in "${RESULT_FILES[@]}"; do
    echo "${OUTPUT_DIR}/${result_file}"
done