#!/usr/bin/env bash
#
# Prepare microstate-aware inputs for Uni-pKa inference.
#
# Usage:
#   ./prepare_microstate_unipka.sh INPUT_TSV PREPARED_DIR [N_JOBS] [BATCH_SIZE]
#
# Required input:
#   - TSV with one unique, non-empty SMILES column
#
# Primary output files in PREPARED_DIR:
#   unique_input.tsv             input SMILES with stable MolID values
#   ensemble.tsv                 unmodified enumerator output
#   microstate_metadata.tsv      one row per individual enumerated microstate
#   transition_metadata.tsv      one row per validated microstate transition
#   graph_diagnostics.tsv        graph connectivity diagnostics per molecule
#   sampl6.tsv                   Uni-pKa transition inputs
#   sampl6.lmdb                  UniMol conformer dataset for sampl6.tsv
#   dict.txt, dict_charge.txt
#

set -euo pipefail

IMAGE="uni-pka:local"
INPUT_TSV="${1:-}"
PREPARED_DIR="${2:-}"
N_JOBS="${3:-1}"
BATCH_SIZE="${4:-512}"
TASK_NAME="sampl6"

usage() {
    cat >&2 <<EOF
Usage: $0 INPUT_TSV PREPARED_DIR [N_JOBS] [BATCH_SIZE]

INPUT_TSV must be a tab-separated file containing a unique SMILES column.
EOF
    exit 2
}

[[ -n "${INPUT_TSV}" && -n "${PREPARED_DIR}" ]] || usage
[[ "${INPUT_TSV}" == *.tsv ]] || { echo "Input: expected .tsv file." >&2; exit 2; }
[[ -f "${INPUT_TSV}" ]] || { echo "Input: file not found." >&2; exit 1; }
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
mkdir -p "${PREPARED_DIR}"
PREPARED_DIR="$(cd "${PREPARED_DIR}" && pwd)"

echo "=== Uni-pKa microstate-aware CPU preparation ==="
echo "Input:      ${INPUT_TSV}"
echo "Output:     ${PREPARED_DIR}"
echo "CPU jobs:   ${N_JOBS}"
echo "Batch size: ${BATCH_SIZE}"

docker run --rm -i \
    -v "${INPUT_DIR}:/work/input:ro" \
    -v "${PREPARED_DIR}:/work/prepared" \
    -e INPUT_FILE="${INPUT_FILE}" \
    -e N_JOBS="${N_JOBS}" \
    -e BATCH_SIZE="${BATCH_SIZE}" \
    -e TASK_NAME="${TASK_NAME}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${IMAGE}" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

REPO_DIR="/opt/uni-pka"
INPUT_TSV="/work/input/${INPUT_FILE}"
PREPARED_DIR="/work/prepared"
ENSEMBLE_BATCH_DIR="${PREPARED_DIR}/.ensemble_batches"
LMDB_BATCH_DIR="${PREPARED_DIR}/.lmdb_batches"

return_prepared_ownership() {
    if [[ -d "${PREPARED_DIR:-}" ]]; then
        chown -R "${HOST_UID}:${HOST_GID}" "${PREPARED_DIR}" 2>/dev/null || true
        chmod -R u+rwX "${PREPARED_DIR}" 2>/dev/null || true
    fi
}

trap return_prepared_ownership EXIT

[[ -d "${REPO_DIR}" ]] || {
    echo "Container: Uni-pKa repository not found." >&2
    exit 1
}
[[ -f "${INPUT_TSV}" ]] || {
    echo "Input: unavailable in container." >&2
    exit 1
}

cd "${REPO_DIR}"

rm -rf "${ENSEMBLE_BATCH_DIR}" "${LMDB_BATCH_DIR}"
mkdir -p "${ENSEMBLE_BATCH_DIR}"

echo
echo "=== Preparing and validating input ==="

python - "${INPUT_TSV}" "${PREPARED_DIR}/unique_input.tsv" <<'PYTHON_PREPARE'

import sys
from pathlib import Path
import pandas as pd

source, target = map(Path, sys.argv[1:3])

try:
    df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
except Exception as error:
    raise SystemExit(f"Input: could not read TSV ({error}).")

if "SMILES" not in df.columns:
    raise SystemExit("Input: missing SMILES column.")

if df.empty:
    raise SystemExit("Input: file is empty.")

df["SMILES"] = df["SMILES"].astype(str).str.strip()

if df["SMILES"].duplicated().any():
    raise SystemExit("Input: duplicate SMILES found.")

pd.DataFrame({
    "MolID": [f"mol_{i:06d}" for i in range(len(df))],
    "SMILES": df["SMILES"],
    "TARGET": -1.0,
}).to_csv(target, sep="\t", index=False)

print(f"SMILES: {len(df)}")
PYTHON_PREPARE

echo
echo "=== Enumerating protonation ensembles ==="

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)

if df.empty:
    raise SystemExit("Enumeration: no SMILES to enumerate.")

n_jobs = min(effective_n_jobs(requested_jobs), len(df))
n_batches = max(n_jobs, math.ceil(len(df) / batch_size))
indices_batches = [
    indices for indices in np.array_split(np.arange(len(df)), n_batches) if len(indices)
]

def run_batch(index, rows):
    work = root / f"batch_{index:06d}"
    work.mkdir(parents=True, exist_ok=False)

    inp = work / "input.tsv"
    out = work / "ensemble.tsv"

    rows.to_csv(inp, sep="\t", index=False)

    command = [
        sys.executable,
        str(repo / "enumerator/main.py"),
        "ensemble",
        "--input",
        str(inp),
        "--output",
        str(out),
        "--upper",
        "2",
        "--lower",
        "-2",
        "--maxiter",
        "32",
        "--template",
        str(repo / "enumerator/simple_smarts_pattern.tsv"),
    ]

    subprocess.run(command, check=True, cwd=repo)

    if not out.is_file():
        raise RuntimeError(f"Enumeration: batch {index} produced no output.")

    return index, out


inputs = [
    (index, df.iloc[indices].reset_index(drop=True))
    for index, indices in enumerate(indices_batches)
]

print(f"SMILES: {len(df)}; workers: {n_jobs}; batches: {len(inputs)}")

try:
    result_paths = dict(
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_batch)(index, rows) for index, rows in inputs
        )
    )

    frames = []

    for index, rows in inputs:
        ensemble = pd.read_csv(result_paths[index], sep="\t", dtype=str, keep_default_na=False)

        if len(ensemble) != len(rows):
            raise SystemExit(f"Enumeration: invalid row count for batch {index}.")

        frames.append(ensemble)

    combined = pd.concat(frames, ignore_index=True)

    if len(combined) != len(df):
        raise SystemExit("Enumeration: invalid combined row count.")

    combined.to_csv(output, sep="\t", index=False)

    print(f"Wrote ensemble: {output}")

finally:
    shutil.rmtree(root, ignore_errors=True)
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

(unique_path, ensemble_path, pka_input_path, transition_path,
microstate_path, diagnostics_path, rejected_path) = map(Path, sys.argv[1:8])

requested_jobs = int(sys.argv[8])

unique = pd.read_csv(unique_path, sep="\t", dtype=str, keep_default_na=False)

ensemble = pd.read_csv(ensemble_path, sep="\t", dtype=str, keep_default_na=False)

if len(unique) != len(ensemble):
    raise SystemExit("Transitions: ensemble and input row counts differ.")

if unique.empty:
    raise SystemExit("Transitions: no molecules available.")

def split_microstates(value):
    """Split comma-delimited enumerator candidates deterministically."""
    if value is None:
        return []

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    seen = set()
    result = []

    for candidate in text.split(","):
        candidate = candidate.strip()

        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)

    return result


def hydrogen_count(mol):
    """Return total hydrogen count, including implicit and explicit H atoms."""
    with_h = Chem.AddHs(mol)
    return sum(atom.GetAtomicNum() == 1 for atom in with_h.GetAtoms())

def formal_charge(mol):
    """Return total formal charge of a molecule."""
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

def validate_one_proton_edge(acid_smiles, base_smiles, acid_charge, base_charge):
    acid = Chem.MolFromSmiles(acid_smiles)
    base = Chem.MolFromSmiles(base_smiles)

    if acid is None or base is None:
        return False, "RDKit parse failure"

    observed_acid_charge = formal_charge(acid)
    observed_base_charge = formal_charge(base)

    if observed_acid_charge != acid_charge:
        return (False, f"acid formal charge {observed_acid_charge} " f"!= declared {acid_charge}")

    if observed_base_charge != base_charge:
        return (False, f"base formal charge {observed_base_charge} " f"!= declared {base_charge}")

    if acid_charge != base_charge + 1:
        return False, "declared charges are not adjacent"

    if hydrogen_count(acid) != hydrogen_count(base) + 1:
        return False, "structures do not differ by exactly one hydrogen"

    if acid.GetNumHeavyAtoms() != base.GetNumHeavyAtoms():
        return False, "heavy-atom counts differ"

    try:
        mcs = rdFMCS.FindMCS([acid, base], timeout=10,
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


def build_molecule_transition_graph(index, mol_id, original_smiles, ensemble_values):
    """Build graph outputs for one independent parent molecule."""
    microstate_rows = []
    transition_rows = []
    pka_rows = []
    rejected_rows = []

    nodes_by_charge = defaultdict(list)

    for charge in range(MIN_CHARGE, MAX_CHARGE + 1):
        candidates = split_microstates(ensemble_values.get(str(charge)))

        for ordinal, smiles in enumerate(candidates, start=1):
            node_id = f"{mol_id}_q{charge}_m{ordinal:03d}"

            node = {
                "MolID": mol_id,
                "OriginalSMILES": original_smiles,
                "MicrostateID": node_id,
                "Charge": charge,
                "MicrostateSMILES": smiles,
            }

            nodes_by_charge[charge].append(node)
            microstate_rows.append(node)

    molecule_edges = []

    for base_charge in range(MIN_CHARGE, MAX_CHARGE):
        acid_charge = base_charge + 1

        for acid in nodes_by_charge[acid_charge]:
            for base in nodes_by_charge[base_charge]:
                valid, reason = validate_one_proton_edge(
                    acid["MicrostateSMILES"], base["MicrostateSMILES"],
                    acid_charge, base_charge,
                )

                if not valid:
                    rejected_rows.append({
                        "MolID": mol_id,
                        "AcidMicrostateID": acid["MicrostateID"],
                        "BaseMicrostateID": base["MicrostateID"],
                        "AcidCharge": acid_charge,
                        "BaseCharge": base_charge,
                        "AcidMicrostateSMILES": acid["MicrostateSMILES"],
                        "BaseMicrostateSMILES": base["MicrostateSMILES"],
                        "RejectionReason": reason,
                    })
                    continue

                edge_id = (f"{acid['MicrostateID']}_to_{base['MicrostateID']}")

                transition_smiles = (f"{acid['MicrostateSMILES']}>>{base['MicrostateSMILES']}")

                edge = {
                    "TransitionID": edge_id,
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

                molecule_edges.append(edge)
                transition_rows.append(edge)

                pka_rows.append({"SMILES": transition_smiles, "TARGET": -1.0})

    molecule_nodes = [
        node for charge_nodes in nodes_by_charge.values()
        for node in charge_nodes
    ]

    adjacency = {
        node["MicrostateID"]: set() for node in molecule_nodes
    }

    for edge in molecule_edges:
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

            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)

    graph_status = "CONNECTED" if molecule_nodes and components == 1 else "DISCONNECTED"

    diagnostics_row = {
        "MolID": mol_id,
        "OriginalSMILES": original_smiles,
        "MicrostateCount": len(molecule_nodes),
        "TransitionCount": len(molecule_edges),
        "ConnectedComponentCount": components,
        "GraphStatus": graph_status,
    }

    return (
        index,
        microstate_rows,
        transition_rows,
        pka_rows,
        rejected_rows,
        diagnostics_row,
    )


n_jobs = min(effective_n_jobs(requested_jobs), len(unique))

work_items = [
    (
        index,
        unique.at[index, "MolID"],
        unique.at[index, "SMILES"],
        ensemble.iloc[index].to_dict(),
    )
    for index in range(len(unique))
]

print(f"Molecules: {len(work_items)}; transition workers: {n_jobs}")

transition_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, batch_size=1)(
    delayed(build_molecule_transition_graph)(
        index, mol_id, original_smiles, ensemble_values,
        )
    for index, mol_id, original_smiles, ensemble_values in work_items
)

transition_results.sort(key=lambda result: result[0])

microstate_rows = []
transition_rows = []
pka_rows = []
rejected_rows = []
diagnostics_rows = []

for (
    _,
    molecule_microstate_rows,
    molecule_transition_rows,
    molecule_pka_rows,
    molecule_rejected_rows,
    molecule_diagnostics_row,
) in transition_results:
    microstate_rows.extend(molecule_microstate_rows)
    transition_rows.extend(molecule_transition_rows)
    pka_rows.extend(molecule_pka_rows)
    rejected_rows.extend(molecule_rejected_rows)
    diagnostics_rows.append(molecule_diagnostics_row)

if not pka_rows:
    raise SystemExit("Transitions: no validated one-proton transitions generated.")

pd.DataFrame(pka_rows).to_csv(pka_input_path, sep="\t", index=False)
pd.DataFrame(transition_rows).to_csv(transition_path, sep="\t", index=False)
pd.DataFrame(microstate_rows).to_csv(microstate_path, sep="\t", index=False)
pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, sep="\t", index=False)

pd.DataFrame(rejected_rows).to_csv(rejected_path, sep="\t", index=False)

print(f"Microstates: {len(microstate_rows)}")
print(f"Validated transitions: {len(pka_rows)}")
print(f"Rejected candidate pairs: {len(rejected_rows)}")
print(f"Disconnected molecular graphs: {sum(row['GraphStatus'] != 'CONNECTED' for row in diagnostics_rows)}")
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

MAX_THREADS_PER_PREPROCESS = 4

source, final_lmdb, root = map(Path, sys.argv[1:4])
requested_jobs, batch_size = map(int, sys.argv[4:6])
repo = Path(sys.argv[6])
task_name = sys.argv[7]

df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)

if df.empty or not {"SMILES", "TARGET"}.issubset(df.columns):
    raise SystemExit("Conformers: transition TSV is empty or invalid.")

workers = min(MAX_THREADS_PER_PREPROCESS, effective_n_jobs(requested_jobs))

parallel_jobs = min(max(1, effective_n_jobs(requested_jobs) // workers), len(df))

n_batches = max(parallel_jobs, math.ceil(len(df) / batch_size))

index_batches = [
    indices
    for indices in np.array_split(np.arange(len(df)), n_batches)
    if len(indices)
]


def run_batch(index, rows):
    work = root / f"batch_{index:06d}"
    work.mkdir(parents=True, exist_ok=False)

    inp = work / f"{task_name}.tsv"
    output = work / f"{task_name}.lmdb"

    rows.to_csv(inp, sep="\t", index=False)

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/preprocess_pka.py"),
            "--raw-csv-file",
            str(inp),
            "--processed-lmdb-dir",
            str(work),
            "--task-name",
            task_name,
            "--nthreads",
            str(workers),
        ],
        check=True,
        cwd=repo,
    )

    if not output.is_file():
        raise RuntimeError(f"Conformers: batch {index} produced no LMDB.")

    return index, output


inputs = [
    (index, df.iloc[indices].reset_index(drop=True))
    for index, indices in enumerate(index_batches)
]

print(
    f"Transitions: {len(df)}; "
    f"preprocess jobs: {parallel_jobs}; "
    f"workers/job: {workers}; "
    f"batches: {len(inputs)}"
)

shutil.rmtree(root, ignore_errors=True)
root.mkdir(parents=True)

try:
    paths = dict(
        Parallel(
            n_jobs=parallel_jobs,
            backend="loky",
            verbose=10,
        )(
            delayed(run_batch)(index, rows)
            for index, rows in inputs
        )
    )

    if final_lmdb.exists():
        final_lmdb.unlink()

    out_env = lmdb.open(
        str(final_lmdb),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    output_index = 0

    with out_env.begin(write=True) as out_txn:
        for index, _ in inputs:
            in_env = lmdb.open(
                str(paths[index]),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )

            with in_env.begin() as in_txn:
                for _, value in in_txn.cursor():
                    out_txn.put(
                        str(output_index).encode("ascii"),
                        value,
                    )
                    output_index += 1

            in_env.close()

    out_env.close()

    if output_index != len(df):
        raise SystemExit(f"Conformers: LMDB contains {output_index} records; expected {len(df)}.")

    print(f"Wrote LMDB: {final_lmdb}; records: {output_index}")

finally:
    shutil.rmtree(root, ignore_errors=True)
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
        echo "Preparation: missing ${file}." >&2
        exit 1
    }
done

echo
echo "=== CPU preparation completed ==="
echo "Prepared files:"

for file in "${required_files[@]}"; do
    echo "  ${PREPARED_DIR}/${file}"
done
CONTAINER_SCRIPT
