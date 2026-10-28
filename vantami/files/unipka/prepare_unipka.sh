#!/usr/bin/env bash
#
# Prepare all CPU-based Uni-pKa inputs:
#   1. Validate a TSV containing unique SMILES
#   2. Enumerate protonation ensembles in parallel
#   3. Build acid/base transition records
#   4. Generate RDKit conformers and the UniMol LMDB dataset
#
# Input requirements:
#   - Tab-separated .tsv file
#   - Header containing a column named exactly: SMILES
#   - SMILES values must already be unique
#
# Usage:
#   ./prepare_unipka_cpu.sh INPUT_TSV PREPARED_DIR [N_JOBS] [BATCH_SIZE]
#
# Example:
#   ./prepare_unipka_cpu.sh input_unique.tsv prepared_unipka 16 512
#
# Outputs in PREPARED_DIR:
#   - unique_input.tsv
#   - ensemble.tsv
#   - sampl6.tsv
#   - transition_metadata.tsv
#   - macrostate_metadata.tsv
#   - sampl6.lmdb
#   - dict.txt
#   - dict_charge.txt
#
# These files are consumed by run_unipka_gpu.sh.

set -euo pipefail

IMAGE="uni-pka:local"

INPUT_TSV="${1:-}"
PREPARED_DIR="${2:-}"
N_JOBS="${3:-$(nproc)}"
BATCH_SIZE="${4:-512}"
TASK_NAME="sampl6"

usage() {
    cat >&2 <<EOF
Usage: $0 INPUT_TSV PREPARED_DIR [N_JOBS] [BATCH_SIZE]

INPUT_TSV must be a TSV with a unique SMILES column.
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

echo "=== Uni-pKa CPU preparation ==="
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
    "${IMAGE}" \
    bash -s <<'CONTAINER_SCRIPT'

set -euo pipefail

REPO_DIR="/opt/uni-pka"
INPUT_TSV="/work/input/${INPUT_FILE}"
PREPARED_DIR="/work/prepared"
BATCH_DIR="${PREPARED_DIR}/.ensemble_batches"

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

rm -rf "${BATCH_DIR}"
mkdir -p "${BATCH_DIR}"

echo
echo "=== Preparing input ==="

python - \
    "${INPUT_TSV}" \
    "${PREPARED_DIR}/unique_input.tsv" <<'PYTHON_PREPARE'
import sys
from pathlib import Path

import pandas as pd

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

try:
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
except Exception as error:
    raise SystemExit(f"Input: could not read TSV ({error}).")

if "SMILES" not in df.columns:
    raise SystemExit("Input: missing SMILES column.")

if df.empty:
    raise SystemExit("Input: file is empty.")

if (df["SMILES"].str.strip() == "").any():
    raise SystemExit("Input: empty SMILES found.")

if df["SMILES"].duplicated().any():
    raise SystemExit("Input: duplicate SMILES found.")

unique_input = pd.DataFrame({
    "MolID": [f"mol_{i:06d}" for i in range(len(df))],
    "SMILES": df["SMILES"],
    "TARGET": -1.0,
})

unique_input.to_csv(output_path, sep="\t", index=False)
print(f"SMILES: {len(unique_input)}")
PYTHON_PREPARE

echo
echo "=== Enumerating protonation ensembles ==="

# Parallel enumerator processes should not each create their own large
# OpenMP/BLAS thread pools.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - \
    "${PREPARED_DIR}/unique_input.tsv" \
    "${PREPARED_DIR}/ensemble.tsv" \
    "${BATCH_DIR}" \
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


def _prepare_batches(df, smiles_col, n_jobs, batch_size):
    """Prepare ordered batches of unique SMILES for downstream processing."""
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}.")

    smiles = df[smiles_col].drop_duplicates(keep="first").to_list()
    if not smiles:
        return [], 0, []

    n_jobs = min(effective_n_jobs(n_jobs), len(smiles))
    n_batches = max(n_jobs, math.ceil(len(smiles) / batch_size))
    return smiles, n_jobs, np.array_split(smiles, n_batches)


def enumerate_batch(batch_index, batch_df, batch_dir, repo_dir):
    """Write one batch, run the enumerator, and return its output path."""
    work_dir = batch_dir / f"batch_{batch_index:06d}"
    work_dir.mkdir(parents=True, exist_ok=False)

    input_path = work_dir / "input.tsv"
    output_path = work_dir / "ensemble.tsv"
    batch_df.to_csv(input_path, sep="\t", index=False)

    command = [
        sys.executable,
        str(repo_dir / "enumerator/main.py"),
        "ensemble",
        "--input", str(input_path),
        "--output", str(output_path),
        "--upper", "2",
        "--lower", "-2",
        "--maxiter", "32",
        "--template", str(repo_dir / "enumerator/simple_smarts_pattern.tsv"),
    ]

    try:
        subprocess.run(command, check=True, cwd=repo_dir)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Enumeration: batch {batch_index} failed.") from error

    if not output_path.is_file():
        raise RuntimeError(f"Enumeration: batch {batch_index} produced no output.")

    return batch_index, output_path


input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
batch_dir = Path(sys.argv[3])
requested_jobs = int(sys.argv[4])
batch_size = int(sys.argv[5])
repo_dir = Path(sys.argv[6])

df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)

if df.empty:
    raise SystemExit("Preparation: no SMILES to enumerate.")

smiles, n_jobs, smiles_batches = _prepare_batches(
    df=df,
    smiles_col="SMILES",
    n_jobs=requested_jobs,
    batch_size=batch_size,
)

if not smiles:
    raise SystemExit("Preparation: no SMILES to enumerate.")

rows_by_smiles = df.set_index("SMILES", drop=False)
batch_inputs = [
    (index, rows_by_smiles.loc[list(batch_smiles)].reset_index(drop=True))
    for index, batch_smiles in enumerate(smiles_batches)
]

print(f"SMILES:  {len(smiles)}")
print(f"Workers: {n_jobs}")
print(f"Batches: {len(batch_inputs)}")

try:
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(enumerate_batch)(batch_index, batch_df, batch_dir, repo_dir)
        for batch_index, batch_df in batch_inputs
    )

    result_paths = dict(results)
    ensemble_frames = []

    for batch_index, (_, batch_input) in enumerate(batch_inputs):
        batch_output = result_paths[batch_index]

        try:
            ensemble = pd.read_csv(
                batch_output,
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )
        except Exception as error:
            raise SystemExit(
                f"Enumeration: could not read batch {batch_index} ({error})."
            )

        unnamed_cols = [
            column for column in ensemble.columns
            if column.startswith("Unnamed") or column == ""
        ]

        if unnamed_cols:
            ensemble = ensemble.drop(columns=unnamed_cols)

        if len(ensemble) != len(batch_input):
            raise SystemExit(
                f"Enumeration: invalid row count for batch {batch_index}."
            )

        ensemble_frames.append(ensemble)

    combined_ensemble = pd.concat(ensemble_frames, ignore_index=True)

    if len(combined_ensemble) != len(df):
        raise SystemExit("Enumeration: invalid combined row count.")

    combined_ensemble.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote ensemble: {output_path}")
finally:
    shutil.rmtree(batch_dir, ignore_errors=True)
PYTHON_ENUMERATE

echo
echo "=== Creating acid/base transitions ==="

python - \
    "${PREPARED_DIR}/unique_input.tsv" \
    "${PREPARED_DIR}/ensemble.tsv" \
    "${PREPARED_DIR}/${TASK_NAME}.tsv" \
    "${PREPARED_DIR}/transition_metadata.tsv" \
    "${PREPARED_DIR}/macrostate_metadata.tsv" <<'PYTHON_TRANSITIONS'
import sys
from pathlib import Path

import pandas as pd

unique_path = Path(sys.argv[1])
ensemble_path = Path(sys.argv[2])
pka_input_path = Path(sys.argv[3])
transition_metadata_path = Path(sys.argv[4])
macrostate_metadata_path = Path(sys.argv[5])

unique_df = pd.read_csv(unique_path, sep="\t", dtype=str, keep_default_na=False)
ensemble_df = pd.read_csv(ensemble_path, sep="\t", dtype=str, keep_default_na=False)

unnamed_cols = [
    column for column in ensemble_df.columns
    if column.startswith("Unnamed") or column == ""
]

if unnamed_cols:
    ensemble_df = ensemble_df.drop(columns=unnamed_cols)

if len(ensemble_df) != len(unique_df):
    raise SystemExit("Transitions: ensemble and input row counts differ.")


def clean_state(value):
    if value is None:
        return None

    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "null"} else text


transition_rows = []
pka_rows = []
macrostate_rows = []

for index, ensemble_row in ensemble_df.iterrows():
    mol_id = unique_df.loc[index, "MolID"]
    original_smiles = unique_df.loc[index, "SMILES"]
    states = {}

    for charge in range(-2, 3):
        state = clean_state(ensemble_row.get(str(charge)))
        states[charge] = state

        if state is not None:
            macrostate_rows.append({
                "MolID": mol_id,
                "OriginalSMILES": original_smiles,
                "Charge": charge,
                "MacrostateSMILES": state,
            })

    for base_charge in range(-2, 2):
        acid_charge = base_charge + 1
        acid_state = states[acid_charge]
        base_state = states[base_charge]

        if acid_state is None or base_state is None:
            continue

        transition_id = f"{mol_id}_q{acid_charge}_to_q{base_charge}"
        transition_smiles = f"{acid_state}>>{base_state}"

        transition_rows.append({
            "TransitionID": transition_id,
            "MolID": mol_id,
            "OriginalSMILES": original_smiles,
            "AcidCharge": acid_charge,
            "BaseCharge": base_charge,
            "SMILES": transition_smiles,
        })

        pka_rows.append({
            "SMILES": transition_smiles,
            "TARGET": -1.0,
        })

if not pka_rows:
    raise SystemExit("Transitions: no acid/base transitions generated.")

pd.DataFrame(pka_rows).to_csv(pka_input_path, sep="\t", index=False)
pd.DataFrame(transition_rows).to_csv(transition_metadata_path, sep="\t", index=False)
pd.DataFrame(macrostate_rows).to_csv(macrostate_metadata_path, sep="\t", index=False)

print(f"Transitions: {len(pka_rows)}")
print(f"Macrostates: {len(macrostate_rows)}")
PYTHON_TRANSITIONS

echo
echo "=== Generating conformers and LMDB data ==="

# Run several preprocessing jobs concurrently. Each preprocess_pka.py invocation
# is limited to at most four RDKit worker processes.
python - \
    "${PREPARED_DIR}/${TASK_NAME}.tsv" \
    "${PREPARED_DIR}/${TASK_NAME}.lmdb" \
    "${PREPARED_DIR}/.lmdb_batches" \
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


def _prepare_batches(df, smiles_col, n_jobs, batch_size):
    """Prepare ordered batches of transition SMILES."""
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}.")

    smiles = df[smiles_col].to_list()
    if not smiles:
        return [], 0, []

    n_jobs = min(effective_n_jobs(n_jobs), len(smiles))
    n_batches = max(n_jobs, math.ceil(len(smiles) / batch_size))
    return smiles, n_jobs, np.array_split(np.arange(len(smiles)), n_batches)


def preprocess_batch(batch_index, batch_df, batch_root, repo_dir, task_name, nthreads):
    """Run preprocess_pka.py for one transition batch."""
    batch_dir = batch_root / f"batch_{batch_index:06d}"
    batch_dir.mkdir(parents=True, exist_ok=False)

    input_path = batch_dir / f"{task_name}.tsv"
    lmdb_path = batch_dir / f"{task_name}.lmdb"
    batch_df.to_csv(input_path, sep="\t", index=False)

    command = [
        sys.executable,
        str(repo_dir / "scripts" / "preprocess_pka.py"),
        "--raw-csv-file", str(input_path),
        "--processed-lmdb-dir", str(batch_dir),
        "--task-name", task_name,
        "--nthreads", str(nthreads),
    ]

    try:
        subprocess.run(command, check=True, cwd=repo_dir)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Conformers: batch {batch_index} failed.") from error

    if not lmdb_path.is_file():
        raise RuntimeError(f"Conformers: batch {batch_index} produced no LMDB.")

    return batch_index, lmdb_path


input_path = Path(sys.argv[1])
final_lmdb_path = Path(sys.argv[2])
batch_root = Path(sys.argv[3])
requested_jobs = int(sys.argv[4])
batch_size = int(sys.argv[5])
repo_dir = Path(sys.argv[6])
task_name = sys.argv[7]

try:
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
except Exception as error:
    raise SystemExit(f"Conformers: could not read transition TSV ({error}).")

if df.empty:
    raise SystemExit("Conformers: no transitions found.")

if "SMILES" not in df.columns or "TARGET" not in df.columns:
    raise SystemExit("Conformers: transition TSV is invalid.")

# A single preprocess_pka.py instance uses multiprocessing.Pool. Limit each
# instance to four workers, then launch enough instances in parallel to use the
# requested CPU budget without nested-process oversubscription.
workers_per_job = min(MAX_THREADS_PER_PREPROCESS, effective_n_jobs(requested_jobs))
parallel_jobs = max(1, effective_n_jobs(requested_jobs) // workers_per_job)
parallel_jobs = min(parallel_jobs, len(df))

_, _, index_batches = _prepare_batches(
    df=df,
    smiles_col="SMILES",
    n_jobs=parallel_jobs,
    batch_size=batch_size,
)

batch_inputs = [
    (batch_index, df.iloc[indices].reset_index(drop=True))
    for batch_index, indices in enumerate(index_batches)
    if len(indices) > 0
]

if not batch_inputs:
    raise SystemExit("Conformers: no batches created.")

print(f"Transitions:       {len(df)}")
print(f"Batch size:        {batch_size}")
print(f"Preprocess jobs:   {parallel_jobs}")
print(f"Workers per job:   {workers_per_job}")
print(f"Maximum CPU use:   {parallel_jobs * workers_per_job}")
print(f"LMDB batches:      {len(batch_inputs)}")

shutil.rmtree(batch_root, ignore_errors=True)
batch_root.mkdir(parents=True, exist_ok=True)

try:
    results = Parallel(n_jobs=parallel_jobs, backend="loky", verbose=10)(
        delayed(preprocess_batch)(
            batch_index=batch_index,
            batch_df=batch_df,
            batch_root=batch_root,
            repo_dir=repo_dir,
            task_name=task_name,
            nthreads=workers_per_job,
        )
        for batch_index, batch_df in batch_inputs
    )

    batch_lmdb_paths = dict(results)

    if final_lmdb_path.exists():
        final_lmdb_path.unlink()

    output_env = lmdb.open(
        str(final_lmdb_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    output_index = 0

    with output_env.begin(write=True) as output_txn:
        for batch_index, batch_df in batch_inputs:
            batch_lmdb_path = batch_lmdb_paths[batch_index]

            input_env = lmdb.open(
                str(batch_lmdb_path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )

            with input_env.begin() as input_txn:
                cursor = input_txn.cursor()

                for _, value in cursor:
                    output_txn.put(str(output_index).encode("ascii"), value)
                    output_index += 1

            input_env.close()

    output_env.close()

    if output_index != len(df):
        raise SystemExit(
            f"Conformers: LMDB contains {output_index} records; "
            f"expected {len(df)}."
        )

    print(f"Wrote LMDB: {final_lmdb_path}")
    print(f"LMDB records: {output_index}")

finally:
    shutil.rmtree(batch_root, ignore_errors=True)
PYTHON_LMDB

cp unimol/examples/dict.txt "${PREPARED_DIR}/dict.txt"
cp unimol/examples/dict_charge.txt "${PREPARED_DIR}/dict_charge.txt"

for file in \
    unique_input.tsv \
    ensemble.tsv \
    "${TASK_NAME}.tsv" \
    transition_metadata.tsv \
    macrostate_metadata.tsv \
    "${TASK_NAME}.lmdb" \
    dict.txt \
    dict_charge.txt
do
    [[ -f "${PREPARED_DIR}/${file}" ]] || {
        echo "Preparation: missing ${file}." >&2
        exit 1
    }
done

echo
echo "=== CPU preparation completed ==="
echo "Prepared files:"
for file in \
    unique_input.tsv \
    ensemble.tsv \
    "${TASK_NAME}.tsv" \
    transition_metadata.tsv \
    macrostate_metadata.tsv \
    "${TASK_NAME}.lmdb" \
    dict.txt \
    dict_charge.txt
do
    echo "  ${PREPARED_DIR}/${file}"
done

CONTAINER_SCRIPT
