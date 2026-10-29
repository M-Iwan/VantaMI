#!/usr/bin/env bash
#
# Prepare microstate-aware inputs for Uni-pKa inference.
#
# Usage:
#   ./prepare_microstates.sh INPUT_TSV PREPARED_DIR [N_JOBS] [BATCH_SIZE]
#
# On failure, diagnostics are retained at:
#   PREPARED_DIR.failed-<timestamp>-<pid>/
#

set -Eeuo pipefail

IMAGE="uni-pka:local"
INPUT_TSV="${1:-}"
PREPARED_DIR="${2:-}"
N_JOBS="${3:-1}"
BATCH_SIZE="${4:-512}"
TASK_NAME="sampl6"

usage() {
    cat >&2 <<EOF
Usage: $0 INPUT_TSV PREPARED_DIR [N_JOBS] [BATCH_SIZE]

INPUT_TSV must be a tab-separated file containing one unique, non-empty,
RDKit-parseable SMILES column.
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

[[ -n "${INPUT_TSV}" && -n "${PREPARED_DIR}" ]] || usage
[[ "${INPUT_TSV}" == *.tsv ]] || {
    echo "Input: expected a .tsv file." >&2
    exit 2
}
[[ -f "${INPUT_TSV}" ]] || {
    echo "Input: file not found: ${INPUT_TSV}" >&2
    exit 1
}
[[ "${N_JOBS}" =~ ^[0-9]+$ && "${N_JOBS}" -ge 1 ]] || {
    echo "Arguments: N_JOBS must be a positive integer." >&2
    exit 2
}
[[ "${BATCH_SIZE}" =~ ^[0-9]+$ && "${BATCH_SIZE}" -ge 1 ]] || {
    echo "Arguments: BATCH_SIZE must be a positive integer." >&2
    exit 2
}

command -v docker >/dev/null 2>&1 || {
    echo "Docker: command not found." >&2
    exit 1
}

docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
    echo "Docker: image '${IMAGE}' not found." >&2
    exit 1
}

INPUT_DIR="$(cd "$(dirname "${INPUT_TSV}")" && pwd)"
INPUT_FILE="$(basename "${INPUT_TSV}")"

PREPARED_PARENT="$(cd "$(dirname "${PREPARED_DIR}")" && pwd)"
PREPARED_NAME="$(basename "${PREPARED_DIR}")"
PREPARED_DIR="${PREPARED_PARENT}/${PREPARED_NAME}"

RUN_TOKEN="$(date +%Y%m%dT%H%M%S)-$$"
STAGING_DIR="${PREPARED_PARENT}/.${PREPARED_NAME}.staging-${RUN_TOKEN}"
FAILED_DIR="${PREPARED_PARENT}/${PREPARED_NAME}.failed-${RUN_TOKEN}"
BACKUP_DIR="${PREPARED_PARENT}/.${PREPARED_NAME}.backup-${RUN_TOKEN}"

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
            printf 'FAILED: retained diagnostics in %s\n' "${FAILED_DIR}" >&2
        else
            printf 'FAILED: unable to preserve staging diagnostics at %s\n' \
                "${FAILED_DIR}" >&2
        fi
    fi

    exit "${status}"
}

trap preserve_failed_staging EXIT

publish_output() {
    local previous_output=0

    if [[ -e "${PREPARED_DIR}" ]]; then
        mv "${PREPARED_DIR}" "${BACKUP_DIR}"
        previous_output=1
    fi

    if mv "${STAGING_DIR}" "${PREPARED_DIR}"; then
        if [[ "${previous_output}" -eq 1 ]]; then
            rm -rf "${BACKUP_DIR}"
        fi
        return 0
    fi

    echo "Output: unable to publish completed staging directory." >&2

    if [[ "${previous_output}" -eq 1 && ! -e "${PREPARED_DIR}" ]]; then
        if ! mv "${BACKUP_DIR}" "${PREPARED_DIR}"; then
            echo "Output: unable to restore previous output directory." >&2
        fi
    fi

    return 1
}

echo "=== Uni-pKa microstate-aware CPU preparation ==="
echo "Input:      ${INPUT_TSV}"
echo "Output:     ${PREPARED_DIR}"
echo "Staging:    ${STAGING_DIR}"
echo "CPU jobs:   ${N_JOBS}"
echo "Batch size: ${BATCH_SIZE}"

docker run --rm -i \
    -v "${INPUT_DIR}:/work/input:ro" \
    -v "${STAGING_DIR}:/work/prepared" \
    --cpus="${N_JOBS}" \
    -e INPUT_FILE="${INPUT_FILE}" \
    -e N_JOBS="${N_JOBS}" \
    -e BATCH_SIZE="${BATCH_SIZE}" \
    -e TASK_NAME="${TASK_NAME}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${IMAGE}" bash -s <<'CONTAINER_SCRIPT'
set -Eeuo pipefail

REPO_DIR="/opt/uni-pka"
INPUT_TSV="/work/input/${INPUT_FILE}"
PREPARED_DIR="/work/prepared"
ENSEMBLE_BATCH_DIR="${PREPARED_DIR}/.ensemble_batches"
LMDB_BATCH_DIR="${PREPARED_DIR}/.lmdb_batches"

container_error() {
    local status=$?
    local line=$1

    printf 'ERROR: container script failed at line %s (exit status %s).\n' \
        "${line}" "${status}" >&2
    printf 'ERROR: inspect retained batch directories and logs under %s.\n' \
        "${PREPARED_DIR}" >&2
    exit "${status}"
}

trap 'container_error $LINENO' ERR

restore_ownership() {
    local failed=0

    if [[ -d "${PREPARED_DIR:-}" ]]; then
        if ! chown -R "${HOST_UID}:${HOST_GID}" "${PREPARED_DIR}"; then
            printf 'ERROR: chown failed for %s; requested owner %s:%s.\n' \
                "${PREPARED_DIR}" "${HOST_UID}" "${HOST_GID}" >&2
            failed=1
        fi

        if ! chmod -R u+rwX "${PREPARED_DIR}"; then
            printf 'ERROR: chmod failed for %s.\n' "${PREPARED_DIR}" >&2
            failed=1
        fi
    fi

    return "${failed}"
}

container_exit() {
    local status=$?
    trap - EXIT

    if ! restore_ownership; then
        if [[ "${status}" -eq 0 ]]; then
            status=1
        fi
    fi

    exit "${status}"
}

trap container_exit EXIT

[[ -d "${REPO_DIR}" ]] || {
    echo "Container: repository not found: ${REPO_DIR}" >&2
    exit 1
}

[[ -f "${INPUT_TSV}" ]] || {
    echo "Input: unavailable inside container: ${INPUT_TSV}" >&2
    exit 1
}

cd "${REPO_DIR}"
mkdir -p "${ENSEMBLE_BATCH_DIR}" "${LMDB_BATCH_DIR}"

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1

export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1
export RAYON_NUM_THREADS=1


echo
echo "=== Preparing and validating input ==="

python - "${INPUT_TSV}" "${PREPARED_DIR}/unique_input.tsv" <<'PYTHON_PREPARE'
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

source, target = map(Path, sys.argv[1:3])

try:
    df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
except Exception as error:
    raise SystemExit(f"Input: unable to read TSV ({error}).")

if "SMILES" not in df.columns:
    raise SystemExit("Input: missing required SMILES column.")

if df.empty:
    raise SystemExit("Input: TSV contains no data rows.")

df["SMILES"] = df["SMILES"].astype(str).str.strip()

empty_rows = df.index[df["SMILES"].eq("")].tolist()
if empty_rows:
    rows = ", ".join(str(index + 2) for index in empty_rows[:20])
    suffix = " ..." if len(empty_rows) > 20 else ""
    raise SystemExit(f"Input: empty SMILES at TSV rows {rows}{suffix}.")

if df["SMILES"].duplicated().any():
    duplicates = df.loc[df["SMILES"].duplicated(keep=False), "SMILES"].tolist()
    examples = ", ".join(repr(value) for value in duplicates[:5])
    raise SystemExit(f"Input: duplicate SMILES found, including {examples}.")

invalid = []
for index, smiles in df["SMILES"].items():
    if Chem.MolFromSmiles(smiles) is None:
        invalid.append((index + 2, smiles))

if invalid:
    examples = "; ".join(
        f"row {row}: {smiles!r}" for row, smiles in invalid[:10]
    )
    suffix = " ..." if len(invalid) > 10 else ""
    raise SystemExit(
        f"Input: {len(invalid)} SMILES values could not be parsed by RDKit "
        f"({examples}{suffix})."
    )

pd.DataFrame(
    {
        "MolID": [f"mol_{index:06d}" for index in range(len(df))],
        "SMILES": df["SMILES"],
        "TARGET": -1.0,
    }
).to_csv(target, sep="\t", index=False)

print(f"SMILES: {len(df)}")
PYTHON_PREPARE

echo
echo "=== Enumerating protonation ensembles ==="

python - \
    "${PREPARED_DIR}/unique_input.tsv" \
    "${PREPARED_DIR}/ensemble.tsv" \
    "${ENSEMBLE_BATCH_DIR}" \
    "${N_JOBS}" \
    "${BATCH_SIZE}" \
    "${REPO_DIR}" <<'PYTHON_ENUMERATE'
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

source, output, root = map(Path, sys.argv[1:4])
requested_jobs, batch_size = map(int, sys.argv[4:6])
repo = Path(sys.argv[6])

required_columns = {"SMILES", "-2", "-1", "0", "1", "2"}

df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)

if df.empty:
    raise SystemExit("Enumeration: no SMILES are available.")

if df["MolID"].duplicated().any():
    raise SystemExit("Enumeration: input contains duplicate MolID values.")

n_jobs = min(effective_n_jobs(requested_jobs), len(df))
n_batches = max(n_jobs, math.ceil(len(df) / batch_size))

batches = [
    indices for indices in np.array_split(np.arange(len(df)), n_batches) if len(indices)
]


def validate_and_attach_molid(expected, observed, batch_index, log_path):
    """
    Validate enumerator output and add the stable input MolID column.
    """
    missing = sorted(required_columns - set(observed.columns))

    if missing:
        raise RuntimeError(
            f"Enumeration: batch {batch_index} output is missing required "
            f"columns {missing}; available columns are "
            f"{list(observed.columns)}; inspect {log_path}."
        )

    if len(observed) != len(expected):
        raise RuntimeError(
            f"Enumeration: batch {batch_index} produced {len(observed)} rows; "
            f"expected {len(expected)}; inspect {log_path}."
        )

    expected_smiles = expected["SMILES"].tolist()
    observed_smiles = observed["SMILES"].astype(str).str.strip().tolist()

    mismatches = [
        {
            "batch_row": row_index + 1,
            "MolID": expected.iloc[row_index]["MolID"],
            "input_smiles": expected_smiles[row_index],
            "output_smiles": observed_smiles[row_index],
        }
        for row_index in range(len(expected))
        if expected_smiles[row_index] != observed_smiles[row_index]
    ]

    if mismatches:
        raise RuntimeError(
            f"Enumeration: batch {batch_index} changed or reordered SMILES; "
            f"examples: {mismatches[:5]}; inspect {log_path}."
        )

    restored = observed.copy()
    restored.insert(0, "MolID", expected["MolID"].to_numpy())

    if restored["MolID"].duplicated().any():
        raise RuntimeError(
            f"Enumeration: batch {batch_index} has duplicate restored MolID "
            f"values; inspect {log_path}."
        )

    return restored


def run_batch(batch_index, rows):
    work = root / f"batch_{batch_index:06d}"
    work.mkdir(parents=True, exist_ok=False)

    input_path = work / "input.tsv"
    output_path = work / "ensemble.tsv"
    normalized_output_path = work / "ensemble.with_molid.tsv"
    log_path = work / "enumerator.log"

    rows.to_csv(input_path, sep="\t", index=False)

    command = [
        sys.executable,
        str(repo / "enumerator/main.py"),
        "ensemble",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--upper",
        "2",
        "--lower",
        "-2",
        "--maxiter",
        "32",
        "--template",
        str(repo / "enumerator/simple_smarts_pattern.tsv"),
    ]

    completed = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(completed.stdout or "", encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Enumeration: batch {batch_index} failed with exit status "
            f"{completed.returncode}; inspect {log_path}."
        )

    if not output_path.is_file():
        raise RuntimeError(
            f"Enumeration: batch {batch_index} produced no ensemble.tsv; "
            f"inspect {log_path}."
        )

    try:
        observed = pd.read_csv(output_path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as error:
        raise RuntimeError(
            f"Enumeration: batch {batch_index} output is unreadable ({error}); "
            f"inspect {log_path}."
        ) from error

    restored = validate_and_attach_molid(
        rows,
        observed,
        batch_index,
        log_path,
    )

    restored.to_csv(normalized_output_path, sep="\t", index=False)

    return batch_index, normalized_output_path


inputs = [
    (batch_index, df.iloc[indices].reset_index(drop=True))
    for batch_index, indices in enumerate(batches)
]

print(f"SMILES: {len(df)}; workers: {n_jobs}; batches: {len(inputs)}")

try:
    paths = dict(
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_batch)(batch_index, rows)
            for batch_index, rows in inputs
        )
    )

    frames = [
        pd.read_csv(paths[batch_index], sep="\t", dtype=str, keep_default_na=False)
        for batch_index, _ in inputs
    ]

    combined = pd.concat(frames, ignore_index=True)

    if len(combined) != len(df):
        raise SystemExit(
            f"Enumeration: combined row count is {len(combined)}; "
            f"expected {len(df)}."
        )

    if combined["MolID"].duplicated().any():
        raise SystemExit(
            "Enumeration: combined output contains duplicate restored MolID values."
        )

    expected = df[["MolID", "SMILES"]].copy()

    merged = expected.merge(combined, on="MolID", how="left",
        suffixes=("_input", "_output"), validate="one_to_one", indicator=True
    )

    identity_errors = merged[
        (merged["_merge"] != "both")
        | (merged["SMILES_input"] != merged["SMILES_output"])
    ]

    if not identity_errors.empty:
        raise SystemExit(
            "Enumeration: final output does not preserve input identity; "
            f"examples: {identity_errors.head(5).to_dict(orient='records')}."
        )

    combined = combined.set_index("MolID").loc[df["MolID"]].reset_index()
    combined.to_csv(output, sep="\t", index=False)

    print(f"Wrote ensemble: {output}")

except Exception:
    print(f"Enumeration failed. Retained batch inputs and logs under: {root}", file=sys.stderr)
    raise

else:
    shutil.rmtree(root)
PYTHON_ENUMERATE

echo
echo "=== Building microstate graph and Uni-pKa transitions ==="

python - \
    "${PREPARED_DIR}/unique_input.tsv" \
    "${PREPARED_DIR}/ensemble.tsv" \
    "${PREPARED_DIR}/${TASK_NAME}.tsv" \
    "${PREPARED_DIR}/transition_metadata.tsv" \
    "${PREPARED_DIR}/microstate_metadata.tsv" \
    "${PREPARED_DIR}/graph_diagnostics.tsv" \
    "${PREPARED_DIR}/rejected_transition_candidates.tsv" \
    "${N_JOBS}" <<'PYTHON_TRANSITIONS'
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from rdkit import Chem
from rdkit.Chem import rdFMCS

MIN_CHARGE = -2
MAX_CHARGE = 2

MICROSTATE_COLUMNS = [
    "MolID",
    "OriginalSMILES",
    "MicrostateID",
    "Charge",
    "MicrostateSMILES",
]

TRANSITION_COLUMNS = [
    "TransitionID",
    "DatasetIndex",
    "MolID",
    "OriginalSMILES",
    "AcidMicrostateID",
    "BaseMicrostateID",
    "AcidCharge",
    "BaseCharge",
    "AcidMicrostateSMILES",
    "BaseMicrostateSMILES",
    "SMILES",
]

REJECTED_COLUMNS = [
    "MolID",
    "AcidMicrostateID",
    "BaseMicrostateID",
    "AcidCharge",
    "BaseCharge",
    "AcidMicrostateSMILES",
    "BaseMicrostateSMILES",
    "RejectionReason",
]

DIAGNOSTIC_COLUMNS = [
    "MolID",
    "OriginalSMILES",
    "MicrostateCount",
    "TransitionCount",
    "ConnectedComponentCount",
    "GraphStatus",
]

(unique_path, ensemble_path, pka_path, transition_path,
 microstate_path, diagnostics_path, rejected_path) = map(
    Path,
    sys.argv[1:8],
)

requested_jobs = int(sys.argv[8])

unique = pd.read_csv(unique_path, sep="\t", dtype=str, keep_default_na=False)
ensemble = pd.read_csv(ensemble_path, sep="\t", dtype=str, keep_default_na=False)

if unique.empty:
    raise SystemExit("Transitions: no molecules are available.")

required_columns = {"MolID", "SMILES", "-2", "-1", "0", "1", "2"}
missing = sorted(required_columns - set(ensemble.columns))
if missing:
    raise SystemExit(f"Transitions: ensemble is missing required columns {missing}.")

if unique["MolID"].duplicated().any():
    raise SystemExit("Transitions: unique input contains duplicate MolID values.")

if ensemble["MolID"].duplicated().any():
    raise SystemExit("Transitions: ensemble contains duplicate MolID values.")

merged = unique[["MolID", "SMILES"]].merge(
    ensemble,
    on="MolID",
    how="left",
    suffixes=("_input", "_ensemble"),
    validate="one_to_one",
    indicator=True,
)

identity_errors = merged[
    (merged["_merge"] != "both") | (merged["SMILES_input"] != merged["SMILES_ensemble"])
]

if not identity_errors.empty:
    examples = identity_errors.head(5).to_dict(orient="records")
    raise SystemExit(
        "Transitions: ensemble does not preserve MolID/SMILES identity; "
        f"examples: {examples}."
    )

ensemble = merged.drop(columns=["SMILES_input", "SMILES_ensemble", "_merge"])


def split_microstates(value):
    if value is None:
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    seen = set()
    result = []

    for candidate in text.split(","):
        candidate = candidate.strip()

        if not candidate:
            continue

        molecule = Chem.MolFromSmiles(candidate)

        if molecule is None:
            raise ValueError(f"Enumerator emitted unparsable microstate SMILES {candidate!r}")

        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result


def hydrogen_count(molecule):
    molecule_with_hydrogens = Chem.AddHs(molecule)
    return sum(
        atom.GetAtomicNum() == 1 for atom in molecule_with_hydrogens.GetAtoms()
    )

def formal_charge(molecule):
    return sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())

def validate_one_proton_edge(acid_smiles, base_smiles, acid_charge, base_charge):
    acid = Chem.MolFromSmiles(acid_smiles)
    base = Chem.MolFromSmiles(base_smiles)

    if acid is None or base is None:
        return False, "RDKit parse failure"

    observed_acid_charge = formal_charge(acid)
    observed_base_charge = formal_charge(base)

    if observed_acid_charge != acid_charge:
        return False, (f"acid formal charge {observed_acid_charge} != declared {acid_charge}")

    if observed_base_charge != base_charge:
        return False, (f"base formal charge {observed_base_charge} != declared {base_charge}")

    if acid_charge != base_charge + 1:
        return False, "declared charges are not adjacent"

    if hydrogen_count(acid) != hydrogen_count(base) + 1:
        return False, "structures do not differ by exactly one hydrogen"

    if acid.GetNumHeavyAtoms() != base.GetNumHeavyAtoms():
        return False, "heavy-atom counts differ"

    try:
        mcs = rdFMCS.FindMCS(
            [acid, base],
            timeout=10,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=False,
        )
    except Exception as error:
        return False, f"MCS failure: {error}"

    if mcs.canceled:
        return False, "MCS timeout"

    if (mcs.numAtoms != acid.GetNumHeavyAtoms() or mcs.numBonds != acid.GetNumBonds()):
        return False, "heavy-atom graphs or bond orders differ"

    return True, ""


def build_molecule_transition_graph(index, mol_id, original_smiles, values):
    microstates = []
    transitions = []
    pka_rows = []
    rejected = []

    nodes_by_charge = defaultdict(list)

    for charge in range(MIN_CHARGE, MAX_CHARGE + 1):
        try:
            candidates = split_microstates(values[str(charge)])
        except Exception as error:
            raise RuntimeError(f"Transitions: MolID {mol_id}, charge {charge}: {error}") from error

        for ordinal, smiles in enumerate(candidates, start=1):
            node = {
                "MolID": mol_id,
                "OriginalSMILES": original_smiles,
                "MicrostateID": f"{mol_id}_q{charge}_m{ordinal:03d}",
                "Charge": charge,
                "MicrostateSMILES": smiles,
            }
            nodes_by_charge[charge].append(node)
            microstates.append(node)

    edges = []

    for base_charge in range(MIN_CHARGE, MAX_CHARGE):
        acid_charge = base_charge + 1

        for acid in nodes_by_charge[acid_charge]:
            for base in nodes_by_charge[base_charge]:
                valid, reason = validate_one_proton_edge(
                    acid["MicrostateSMILES"],
                    base["MicrostateSMILES"],
                    acid_charge,
                    base_charge,
                )

                if not valid:
                    rejected.append(
                        {
                            "MolID": mol_id,
                            "AcidMicrostateID": acid["MicrostateID"],
                            "BaseMicrostateID": base["MicrostateID"],
                            "AcidCharge": acid_charge,
                            "BaseCharge": base_charge,
                            "AcidMicrostateSMILES": acid["MicrostateSMILES"],
                            "BaseMicrostateSMILES": base["MicrostateSMILES"],
                            "RejectionReason": reason,
                        }
                    )
                    continue

                transition_smiles = (
                    f"{acid['MicrostateSMILES']}>>{base['MicrostateSMILES']}"
                )

                edge = {
                    "TransitionID": (
                        f"{acid['MicrostateID']}_to_{base['MicrostateID']}"
                    ),
                    "MolID": mol_id,
                    "OriginalSMILES": original_smiles,
                    "AcidMicrostateID": acid["MicrostateID"],
                    "BaseMicrostateID": base["MicrostateID"],
                    "AcidCharge": acid_charge,
                    "BaseCharge": base_charge,
                    "AcidMicrostateSMILES": acid["MicrostateSMILES"],
                    "BaseMicrostateSMILES": base["MicrostateSMILES"],
                    "SMILES": transition_smiles,
                }

                edges.append(edge)
                transitions.append(edge)
                pka_rows.append({"SMILES": transition_smiles, "TARGET": -1.0})

    nodes = [node for charge_nodes in nodes_by_charge.values() for node in charge_nodes]

    adjacency = {node["MicrostateID"]: set() for node in nodes}

    for edge in edges:
        acid_id = edge["AcidMicrostateID"]
        base_id = edge["BaseMicrostateID"]
        adjacency[acid_id].add(base_id)
        adjacency[base_id].add(acid_id)

    components = 0
    seen = set()

    for node_id in adjacency:
        if node_id in seen:
            continue

        components += 1
        queue = deque([node_id])
        seen.add(node_id)

        while queue:
            current = queue.popleft()

            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

    if not nodes:
        status = "NO_MICROSTATES"
    elif not edges:
        status = "NO_TRANSITIONS"
    elif components == 1:
        status = "CONNECTED"
    else:
        status = "DISCONNECTED"

    diagnostics = {
        "MolID": mol_id,
        "OriginalSMILES": original_smiles,
        "MicrostateCount": len(nodes),
        "TransitionCount": len(edges),
        "ConnectedComponentCount": components,
        "GraphStatus": status,
    }

    return index, microstates, transitions, pka_rows, rejected, diagnostics


n_jobs = min(effective_n_jobs(requested_jobs), len(unique))

items = [
    (
        index,
        row["MolID"],
        row["SMILES"],
        ensemble.iloc[index].to_dict(),
    )
    for index, row in unique.iterrows()
]

print(f"Molecules: {len(items)}; transition workers: {n_jobs}")

results = Parallel(
    n_jobs=n_jobs,
    backend="loky",
    verbose=10,
    batch_size=1,
)(
    delayed(build_molecule_transition_graph)(
        index,
        mol_id,
        original_smiles,
        values,
    )
    for index, mol_id, original_smiles, values in items
)

results.sort(key=lambda result: result[0])

microstate_rows = []
transition_rows = []
pka_rows = []
rejected_rows = []
diagnostic_rows = []

for _, microstates, transitions, pka, rejected, diagnostics in results:
    microstate_rows.extend(microstates)
    transition_rows.extend(transitions)
    pka_rows.extend(pka)
    rejected_rows.extend(rejected)
    diagnostic_rows.append(diagnostics)

if not pka_rows:
    raise SystemExit(
        "Transitions: no validated one-proton transitions were generated."
    )

for dataset_index, transition in enumerate(transition_rows):
    transition["DatasetIndex"] = dataset_index

pd.DataFrame(
    pka_rows,
    columns=["SMILES", "TARGET"],
).to_csv(pka_path, sep="\t", index=False)

pd.DataFrame(
    transition_rows,
    columns=TRANSITION_COLUMNS,
).to_csv(transition_path, sep="\t", index=False)

pd.DataFrame(
    microstate_rows,
    columns=MICROSTATE_COLUMNS,
).to_csv(microstate_path, sep="\t", index=False)

pd.DataFrame(
    diagnostic_rows,
    columns=DIAGNOSTIC_COLUMNS,
).to_csv(diagnostics_path, sep="\t", index=False)

pd.DataFrame(
    rejected_rows,
    columns=REJECTED_COLUMNS,
).to_csv(rejected_path, sep="\t", index=False)

print(f"Microstates: {len(microstate_rows)}")
print(f"Validated transitions: {len(pka_rows)}")
print(f"Rejected candidate pairs: {len(rejected_rows)}")
print(
    "Graphs without usable connected transitions: "
    f"{sum(row['GraphStatus'] != 'CONNECTED' for row in diagnostic_rows)}"
)
PYTHON_TRANSITIONS

echo
echo "=== Generating conformers and UniMol LMDB data ==="

python - \
    "${PREPARED_DIR}/${TASK_NAME}.tsv" \
    "${PREPARED_DIR}/${TASK_NAME}.lmdb" \
    "${LMDB_BATCH_DIR}" \
    "${N_JOBS}" \
    "${BATCH_SIZE}" \
    "${REPO_DIR}" \
    "${TASK_NAME}" <<'PYTHON_LMDB'
import math
import shutil
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

COMMIT_INTERVAL = 256

source, final_lmdb, root = map(Path, sys.argv[1:4])
requested_jobs, batch_size = map(int, sys.argv[4:6])
repo = Path(sys.argv[6])
task_name = sys.argv[7]

df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)

if df.empty or not {"SMILES", "TARGET"}.issubset(df.columns):
    raise SystemExit("Conformers: transition TSV is empty or invalid.")

workers = 1

parallel_jobs = min(effective_n_jobs(requested_jobs), len(df))

n_batches = max(parallel_jobs, math.ceil(len(df) / batch_size))

batches = [
    indices for indices in np.array_split(np.arange(len(df)), n_batches) if len(indices)
]


def numeric_lmdb_records(transaction):
    records = list(transaction.cursor())

    try:
        return sorted(records, key=lambda item: int(item[0]))
    except ValueError as error:
        keys = [key.decode("utf-8", errors="replace") for key, _ in records[:10]]
        raise RuntimeError(
            "Conformers: source LMDB contains non-numeric keys; "
            f"examples: {keys}."
        ) from error


def run_batch(batch_index, rows):
    work = root / f"batch_{batch_index:06d}"
    work.mkdir(parents=True, exist_ok=False)

    input_path = work / f"{task_name}.tsv"
    output_path = work / f"{task_name}.lmdb"
    log_path = work / "preprocess.log"

    rows.to_csv(input_path, sep="\t", index=False)

    command = [
        sys.executable,
        str(repo / "scripts/preprocess_pka.py"),
        "--raw-csv-file",
        str(input_path),
        "--processed-lmdb-dir",
        str(work),
        "--task-name",
        task_name,
        "--nthreads",
        str(workers),
    ]

    completed = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(completed.stdout or "", encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Conformers: batch {batch_index} failed with exit status "
            f"{completed.returncode}; inspect {log_path}."
        )

    if not output_path.is_file():
        raise RuntimeError(
            f"Conformers: batch {batch_index} produced no LMDB; "
            f"inspect {log_path}."
        )

    environment = lmdb.open(
        str(output_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )

    try:
        with environment.begin() as transaction:
            count = sum(1 for _ in transaction.cursor())
    finally:
        environment.close()

    if count != len(rows):
        raise RuntimeError(
            f"Conformers: batch {batch_index} LMDB contains {count} records; "
            f"expected {len(rows)}; inspect {log_path}."
        )

    return batch_index, output_path


inputs = [
    (batch_index, df.iloc[indices].reset_index(drop=True))
    for batch_index, indices in enumerate(batches)
]

print(
    f"Transitions: {len(df)}; "
    f"preprocess jobs: {parallel_jobs}; "
    f"workers/job: {workers}; "
    f"batches: {len(inputs)}"
)

shutil.rmtree(root, ignore_errors=True)
root.mkdir(parents=True)

candidate_lmdb = final_lmdb.with_name(f"{final_lmdb.name}.partial")

try:
    paths = dict(
        Parallel(n_jobs=parallel_jobs, backend="loky", verbose=10)(
            delayed(run_batch)(batch_index, rows)
            for batch_index, rows in inputs
        )
    )

    if candidate_lmdb.exists():
        candidate_lmdb.unlink()

    out_environment = lmdb.open(
        str(candidate_lmdb),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    output_index = 0
    output_transaction = out_environment.begin(write=True)

    try:
        for batch_index, _ in inputs:
            in_environment = lmdb.open(
                str(paths[batch_index]),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )

            try:
                with in_environment.begin() as input_transaction:
                    for _, value in numeric_lmdb_records(input_transaction):
                        key = f"{output_index:012d}".encode("ascii")

                        if not output_transaction.put(key, value):
                            raise RuntimeError(
                                f"Conformers: unable to write merged LMDB "
                                f"record {output_index}."
                            )

                        output_index += 1

                        if output_index % COMMIT_INTERVAL == 0:
                            output_transaction.commit()
                            output_transaction = out_environment.begin(write=True)

            finally:
                in_environment.close()

        output_transaction.commit()
        output_transaction = None

    except Exception:
        if output_transaction is not None:
            output_transaction.abort()
        raise

    finally:
        out_environment.close()

    if output_index != len(df):
        raise SystemExit(
            f"Conformers: merged LMDB contains {output_index} records; "
            f"expected {len(df)}."
        )

    if final_lmdb.exists():
        final_lmdb.unlink()

    candidate_lmdb.replace(final_lmdb)

    print(f"Wrote LMDB: {final_lmdb}; records: {output_index}")

except Exception:
    print(
        f"Conformer generation failed. Retained batch inputs and logs under: {root}",
        file=sys.stderr,
    )
    raise

else:
    shutil.rmtree(root)
PYTHON_LMDB

cp unimol/examples/dict.txt "${PREPARED_DIR}/dict.txt"
cp unimol/examples/dict_charge.txt "${PREPARED_DIR}/dict_charge.txt"

required_files=(
    unique_input.tsv
    ensemble.tsv
    microstate_metadata.tsv
    transition_metadata.tsv
    graph_diagnostics.tsv
    rejected_transition_candidates.tsv
    "${TASK_NAME}.tsv"
    "${TASK_NAME}.lmdb"
    dict.txt
    dict_charge.txt
)

for file in "${required_files[@]}"; do
    [[ -f "${PREPARED_DIR}/${file}" ]] || {
        echo "Preparation: missing required output file: ${file}" >&2
        exit 1
    }
done

echo
echo "=== CPU preparation completed in staging directory ==="
for file in "${required_files[@]}"; do
    echo "  ${PREPARED_DIR}/${file}"
done
CONTAINER_SCRIPT

publish_output

echo
echo "=== CPU preparation published ==="
echo "Prepared files:"
for file in \
    unique_input.tsv \
    ensemble.tsv \
    microstate_metadata.tsv \
    transition_metadata.tsv \
    graph_diagnostics.tsv \
    rejected_transition_candidates.tsv \
    "${TASK_NAME}.tsv" \
    "${TASK_NAME}.lmdb" \
    dict.txt \
    dict_charge.txt; do
    echo "  ${PREPARED_DIR}/${file}"
done
