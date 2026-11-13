from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data


_SCHEMA_VERSION = 1
_META_KEY = b"__meta__"
_RECORD_PREFIX = b"record:"


def _record_key(record_id: int) -> bytes:
    """
    Return a fixed-width, lexicographically sortable LMDB key.
    """
    return _RECORD_PREFIX + f"{record_id:020d}".encode("ascii")


def _serialize(value: Any) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _as_tensor(value: Any, *, column: str, record_id: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Convert a scalar/list/NumPy value into a detached CPU tensor.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 1:
        raise ValueError(
            f"Record {record_id}: column {column!r} must contain one-dimensional values; "
            f"received shape {tuple(tensor.shape)}."
        )
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _load_graph(graph_path: str | Path, record_id: int) -> Data:
    path = Path(graph_path)
    if not path.is_file():
        raise FileNotFoundError(f"Graph file does not exist: {path}")

    graph = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(graph, Data):
        raise TypeError(f"Graph file must contain torch_geometric.data.Data, got {type(graph)!r}.")

    for name in ("x", "edge_index", "edge_attr"):
        if not hasattr(graph, name) or getattr(graph, name) is None:
            raise ValueError(f"Graph is missing required attribute {name!r}.")

    if graph.x.ndim != 2:
        raise ValueError(f"graph.x must have shape [num_nodes, num_features], got {tuple(graph.x.shape)}.")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(f"graph.edge_index must have shape [2, num_edges], got {tuple(graph.edge_index.shape)}.")
    if graph.edge_attr.ndim < 1 or graph.edge_attr.shape[0] != graph.edge_index.shape[1]:
        raise ValueError("graph.edge_attr must be aligned with graph.edge_index.")

    for key, value in graph:
        if isinstance(value, torch.Tensor):
            graph[key] = value.detach().cpu()

    return graph


def _normalise_optional_columns(columns: str | Sequence[str] | None, *, parameter: str) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    if not isinstance(columns, Sequence) or not all(isinstance(column, str) for column in columns):
        raise TypeError(f"{parameter} must be a column name or a sequence of column names.")
    return list(columns)


def write_graph_lmdb(df: pl.DataFrame, lmdb_path: str | Path, *, target_col: str, graph_col: str,
                     record_id: str = "record_id", molecule_id: str = "molecule_id", smiles_col: str = "SMILES",
                     weights_col: str | None = None, group_col: str | None = None,
                     optional_columns: str | Sequence[str] | None = None, map_size: int | None = None, overwrite: bool = False,
                     commit_interval: int = 16_384, reject_report_path: str | Path | None = None) -> dict[str, Any]:
    """
    Write graph-learning samples from a Polars DataFrame to a standalone LMDB.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a polars.DataFrame.")
    if not df.height:
        raise ValueError("df must contain at least one row.")
    if commit_interval < 1:
        raise ValueError("commit_interval must be at least 1.")

    optional_columns = _normalise_optional_columns(optional_columns, parameter="optional_columns")
    required_columns = [record_id, molecule_id, smiles_col, target_col, graph_col]
    provided_columns = required_columns + ([weights_col] if weights_col else []) + ([group_col] if group_col else []) + optional_columns
    missing = sorted(set(provided_columns) - set(df.columns))
    if missing:
        raise KeyError(f"DataFrame does not contain required column(s): {missing}.")
    if df.get_column(record_id).n_unique() != df.height:
        raise ValueError(f"Column {record_id!r} must be unique.")

    overlap = set(optional_columns) & set(required_columns + ([weights_col] if weights_col else []) + ([group_col] if group_col else []))
    if overlap:
        raise ValueError(f"optional_columns duplicates dedicated fields: {sorted(overlap)}.")

    destination = Path(lmdb_path)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"LMDB destination already exists: {destination}")
        if destination.is_dir():
            import shutil
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    resolved_map_size = map_size or max(1 << 30, df.height * 256_000)
    env = lmdb.open(
        str(destination),
        map_size=resolved_map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )

    rejects: list[dict[str, Any]] = []
    written = 0
    target_size: int | None = None
    graph_fields: set[str] = set()
    rows = df.select(provided_columns).iter_rows(named=True)

    try:
        txn = env.begin(write=True)
        for row_index, row in enumerate(rows, start=1):
            row_record_id = row[record_id]
            try:
                if row_record_id is None:
                    raise ValueError("record_id is null.")
                row_record_id = int(row_record_id)
                graph = _load_graph(row[graph_col], row_record_id)
                target = _as_tensor(row[target_col], column=target_col, record_id=row_record_id)
                weights = (
                    _as_tensor(row[weights_col], column=weights_col, record_id=row_record_id, dtype=torch.float32)
                    if weights_col is not None
                    else torch.ones_like(target, dtype=torch.float32)
                )
                if target.numel() != weights.numel():
                    raise ValueError(
                        f"Target length ({target.numel()}) and weight length ({weights.numel()}) differ."
                    )
                if target_size is None:
                    target_size = target.numel()
                elif target.numel() != target_size:
                    raise ValueError(f"Target length differs from expected size {target_size}.")

                payload: dict[str, Any] = {
                    "record_id": row_record_id,
                    "molecule_id": row[molecule_id],
                    "smiles": row[smiles_col],
                    "graph": graph,
                    "y_true": target,
                    "y_wgts": weights,
                }
                if group_col is not None:
                    payload["group"] = row[group_col]
                for column in optional_columns:
                    payload[column] = row[column]

                if not txn.put(_record_key(row_record_id), _serialize(payload), overwrite=False):
                    raise KeyError(f"Duplicate LMDB key for record_id={row_record_id}.")
                graph_fields.update(graph.keys())
                written += 1

                if row_index % commit_interval == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            except Exception as exc:
                rejects.append({
                    "record_id": row_record_id,
                    "molecule_id": row.get(molecule_id),
                    "graph_path": row.get(graph_col),
                    "reason": f"{type(exc).__name__}: {exc}",
                })

        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "num_records": written,
            "num_rejected": len(rejects),
            "record_id_column": record_id,
            "target_column": target_col,
            "num_targets": target_size,
            "graph_path_column": graph_col,
            "has_explicit_weights": weights_col is not None,
            "has_groups": group_col is not None,
            "optional_columns": optional_columns,
            "graph_required_attributes": ["x", "edge_index", "edge_attr"],
            "graph_attribute_policy": "preserve_all",
            "graph_attributes_observed": sorted(graph_fields),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        txn.put(_META_KEY, _serialize(metadata), overwrite=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()

    if reject_report_path is not None:
        report_path = Path(reject_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            rejects,
            schema={"record_id": pl.Int64, "molecule_id": pl.String, "graph_path": pl.String, "reason": pl.String},
        ).write_csv(report_path)

    return {
        "lmdb_path": str(destination),
        "num_records": written,
        "num_rejected": len(rejects),
        "num_targets": target_size,
        "reject_report_path": str(reject_report_path) if reject_report_path is not None else None,
    }