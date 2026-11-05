"""
Zarr serialization for prepared Polars multimodal dataframes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

_SUPPORTED_MODALITIES = frozenset({
    "string", "graph", "descriptor", "image", "target", "sample_weight", "group",
})
_FEATURE_MODALITIES = frozenset({
    "string", "graph", "descriptor", "image"
})
_CANONICAL_KEYS = {
    "target": "y_true",
    "sample_weight": "y_wgts",
    "group": "group",
}
_SCHEMA_VERSION = 1


def _normalize_config(df: pl.DataFrame, config: Mapping[str, str]):
    """Validate source config and derive batch-key/source-column mappings.

    Returns
    -------
    source_config:
        Original source-column-to-modality mapping.
    batch_config:
        Model-facing batch-key-to-modality mapping.
    source_columns:
        Model-facing batch-key-to-original-DataFrame-column mapping.
    """
    if not config:
        raise ValueError("config must not be empty")

    source_config = dict(config)
    missing = set(source_config).difference(set(df.columns))
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {sorted(missing)}")

    unsupported = set(source_config.values()) - _SUPPORTED_MODALITIES
    if unsupported:
        raise ValueError(f"Unsupported modalities: {sorted(unsupported)}")

    counts = defaultdict(int)
    for modality in source_config.values():
        counts[modality] += 1
    for modality in ("target", "sample_weight", "group"):
        if counts[modality] > 1:
            raise ValueError(f"Modality < {modality} > can only be assigned to one column")
    if counts["target"] != 1:
        raise ValueError("Exactly one column must have modality 'target'")
    if not any(modality in _FEATURE_MODALITIES for modality in source_config.values()):
        raise ValueError("At least one feature modality is required")

    batch_config = {}
    source_columns = {}
    for source_name, modality in source_config.items():
        batch_key = _CANONICAL_KEYS.get(modality, source_name)
        batch_config[batch_key] = modality
        source_columns[batch_key] = source_name
    return source_config, batch_config, source_columns


def _as_numpy(value: Any, *, name: str) -> npt.NDArray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} cannot be stored as an object-dtype Zarr array")
    return array


def _create_array(group: Any, name: str, *, shape: Tuple[int, ...], dtype: Any,
                  chunks: Tuple[int, ...]) -> Any:
    if hasattr(group, "create_array"):
        return group.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
    return group.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)


def _append(array: Any, values: npt.NDArray, *, axis: int = 0) -> None:
    values = np.asarray(values)
    if values.ndim != len(array.shape):
        raise ValueError("Cannot append an array with a different rank")
    if any(values.shape[dim] != array.shape[dim] for dim in range(values.ndim) if dim != axis):
        raise ValueError("Cannot append an array with incompatible non-appended dimensions")

    old_size = array.shape[axis]
    new_shape = list(array.shape)
    new_shape[axis] += values.shape[axis]
    array.resize(tuple(new_shape))

    write_slice = [slice(None)] * values.ndim
    write_slice[axis] = slice(old_size, new_shape[axis])
    array[tuple(write_slice)] = values


def _iter_batches(length: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def _validate_storage_key(key: str) -> str:
    if not key or "/" in key:
        raise ValueError(f"Batch key {key!r} cannot be used as a Zarr storage key")
    return key


def _write_dense(root: Any, df: pl.DataFrame, source_column: str, batch_key: str, modality: str, batch_size: int):
    storage_key = _validate_storage_key(batch_key)
    values = df[source_column]
    first = _as_numpy(values[0], name=batch_key)

    if modality == "group" and first.ndim == 0 and first.dtype.kind in "OUS":
        categories = sorted({str(value) for value in values.to_list()})
        lookup = {value: i for i, value in enumerate(categories)}
        node = root.require_group(f"dense/{storage_key}")
        node.attrs["encoding"] = "categorical"
        max_len = max(1, max(map(len, categories)))
        _create_array(
            node, "categories", shape=(len(categories),), dtype=f"<U{max_len}",
            chunks=(max(1, min(batch_size, len(categories))),),
        )[:] = categories
        codes = _create_array(
            node, "codes", shape=(df.height,), dtype=np.int64,
            chunks=(min(batch_size, df.height),),
        )
        for start, end in _iter_batches(df.height, batch_size):
            codes[start:end] = np.asarray(
                [lookup[str(value)] for value in values[start:end].to_list()], dtype=np.int64,
            )
        return

    node = root.require_group(f"dense/{storage_key}")
    expected_shape = first.shape
    array = _create_array(
        node, "values", shape=(df.height,) + expected_shape, dtype=first.dtype,
        chunks=(min(batch_size, df.height),) + expected_shape,
    )
    for start, end in _iter_batches(df.height, batch_size):
        batch = [_as_numpy(value, name=batch_key) for value in values[start:end].to_list()]
        if any(value.shape != expected_shape for value in batch):
            raise ValueError(f"Dense modality {batch_key!r} has non-uniform sample shapes")
        array[start:end] = np.stack(batch)


def _write_strings(root: Any, df: pl.DataFrame, source_column: str, batch_key: str,
                   batch_size: int) -> None:
    """
    Write strings from < StringVectorizer > class.
    """
    storage_key = _validate_storage_key(batch_key)
    column = df[source_column]
    first = column[0]
    if not isinstance(first, (tuple, list)) or len(first) != 2:
        raise TypeError(f"String modality {batch_key!r} must contain (token_ids, length) tuples")
    first_tokens = _as_numpy(first[0], name=batch_key)
    if first_tokens.ndim != 1:
        raise ValueError(f"String modality {batch_key!r} token ids must be one-dimensional")
    first_length = int(first[1])
    if not 0 <= first_length <= first_tokens.size:
        raise ValueError(f"String modality {batch_key!r} has an invalid true length")

    node = root.require_group(f"strings/{storage_key}")
    token_array = _create_array(
        node, "values", shape=(df.height, first_tokens.size), dtype=first_tokens.dtype,
        chunks=(min(batch_size, df.height), first_tokens.size),
    )
    length_array = _create_array(
        node, "lengths", shape=(df.height,), dtype=np.int64,
        chunks=(min(batch_size, df.height),),
    )
    for start, end in _iter_batches(df.height, batch_size):
        token_rows: List[npt.NDArray] = []
        true_lengths: List[int] = []
        for value in column[start:end].to_list():
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"String modality {batch_key!r} contains an invalid token tuple")
            tokens = _as_numpy(value[0], name=batch_key)
            true_length = int(value[1])
            if tokens.ndim != 1 or tokens.shape != first_tokens.shape or tokens.dtype != first_tokens.dtype:
                raise ValueError(
                    f"String modality {batch_key!r} must have one fixed padded token shape and dtype"
                )
            if not 0 <= true_length <= tokens.size:
                raise ValueError(f"String modality {batch_key!r} has an invalid true length")
            token_rows.append(tokens)
            true_lengths.append(true_length)
        token_array[start:end] = np.stack(token_rows, axis=0)
        length_array[start:end] = np.asarray(true_lengths, dtype=np.int64)


def _graph_fields(graph: Any) -> Dict[str, str]:
    """
    Classify serializable PyG attributes by their storage alignment.
    """
    keys = list(graph.keys()) if callable(graph.keys) else list(graph.keys)
    n_nodes = int(graph.x.size(0))
    n_edges = int(graph.edge_index.size(1))

    if "edge_attr" not in keys:
        raise ValueError("Graph modality requires an 'edge_attr' tensor")
    edge_attr = graph["edge_attr"]
    if not isinstance(edge_attr, torch.Tensor):
        raise TypeError("Graph attribute 'edge_attr' must be a torch.Tensor")
    if edge_attr.ndim < 1 or edge_attr.size(0) != n_edges:
        raise ValueError(
            "Graph attribute 'edge_attr' must have leading dimension equal to edge_index.size(1)"
        )

    out: Dict[str, str] = {"edge_attr": "edge"}
    for field in keys:
        if field in {"x", "edge_index", "edge_attr"}:
            continue
        value = graph[field]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Graph attribute {field!r} must be a torch.Tensor")
        if value.ndim > 0 and value.size(0) == n_nodes:
            out[field] = "node"
        elif value.ndim > 0 and value.size(0) == n_edges:
            out[field] = "edge"
        else:
            out[field] = "graph"
    return out


def _write_graphs(root: Any, df: pl.DataFrame, source_column: str, batch_key: str, batch_size: int) -> None:
    """
    Write graphs from < GraphVectorizer > class.
    """
    storage_key = _validate_storage_key(batch_key)
    first = df[source_column][0]
    if not hasattr(first, "x") or not hasattr(first, "edge_index") or not hasattr(first, "edge_attr"):
        raise TypeError(
            f"Graph modality {batch_key!r} must contain PyG Data-like objects with x, edge_index, and edge_attr"
        )
    x0 = _as_numpy(first.x, name=f"{batch_key}.x")
    ei0 = _as_numpy(first.edge_index, name=f"{batch_key}.edge_index")
    if x0.ndim != 2 or ei0.ndim != 2 or ei0.shape[0] != 2:
        raise ValueError(f"Graph modality {batch_key!r} requires x=[nodes, features], edge_index=[2, edges]")
    fields = _graph_fields(first)
    node = root.require_group(f"graphs/{storage_key}")
    node.attrs["field_metadata"] = json.dumps(fields)
    x_array = _create_array(node, "x", shape=(0,) + x0.shape[1:], dtype=x0.dtype,
                            chunks=(max(batch_size, 1),) + x0.shape[1:])
    ei_array = _create_array(node, "edge_index", shape=(2, 0), dtype=ei0.dtype,
                             chunks=(2, max(batch_size, 1)))
    n_offsets = _create_array(node, "node_offsets", shape=(1,), dtype=np.int64,
                              chunks=(max(batch_size, 1),))
    e_offsets = _create_array(node, "edge_offsets", shape=(1,), dtype=np.int64,
                              chunks=(max(batch_size, 1),))
    n_offsets[0] = e_offsets[0] = 0

    field_arrays: Dict[str, Any] = {}
    for field, scope in fields.items():
        value = _as_numpy(first[field], name=f"{batch_key}.{field}")
        trailing = value.shape[1:] if scope in {"node", "edge"} else value.shape
        field_arrays[field] = _create_array(
            node, f"{scope}/{field}", shape=(0,) + trailing, dtype=value.dtype,
            chunks=(max(batch_size, 1),) + trailing,
        )

    total_nodes = total_edges = 0

    for start, end in _iter_batches(df.height, batch_size):
        graphs = df[source_column][start:end].to_list()
        node_counts, edge_counts = [], []
        per_field = {field: [] for field in fields}
        xs, edges = [], []

        for graph in graphs:
            current_fields = _graph_fields(graph)
            if current_fields != fields:
                raise ValueError(f"Graph modality {batch_key!r} has inconsistent tensor attributes")
            x = _as_numpy(graph.x, name=f"{batch_key}.x")
            edge_index = _as_numpy(graph.edge_index, name=f"{batch_key}.edge_index")
            if x.shape[1:] != x0.shape[1:] or edge_index.shape[0] != 2:
                raise ValueError(f"Graph modality {batch_key!r} has inconsistent x or edge_index shape")
            edge_attr = _as_numpy(graph.edge_attr, name=f"{batch_key}.edge_attr")
            if edge_attr.ndim < 1 or edge_attr.shape[0] != edge_index.shape[1]:
                raise ValueError(f"Graph modality {batch_key!r} requires edge_attr.shape[0] to equal edge_index.shape[1]")
            xs.append(x)
            edges.append(edge_index)
            node_counts.append(x.shape[0])
            edge_counts.append(edge_index.shape[1])
            for field, scope in fields.items():
                value = _as_numpy(graph[field], name=f"{batch_key}.{field}")
                expected = field_arrays[field].shape[1:]
                actual = value.shape[1:] if scope in {"node", "edge"} else value.shape
                if actual != expected:
                    raise ValueError(f"Graph field {field!r} has inconsistent shape")
                per_field[field].append(value if scope in {"node", "edge"} else value[None, ...])
        _append(x_array, np.concatenate(xs, axis=0))
        _append(ei_array, np.concatenate(edges, axis=1), axis=1)
        _append(n_offsets, total_nodes + np.cumsum(node_counts, dtype=np.int64))
        _append(e_offsets, total_edges + np.cumsum(edge_counts, dtype=np.int64))
        total_nodes += sum(node_counts)
        total_edges += sum(edge_counts)
        for field, values in per_field.items():
            _append(field_arrays[field], np.concatenate(values, axis=0))


def polars_2_zarr(df: pl.DataFrame, config: Mapping[str, str], path: Union[str, Path], batch_size: int = 1024,
                  overwrite: bool = False) -> None:
    """
    Persist a prepared Polars dataframe in this module's Zarr schema.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if df.height == 0:
        raise ValueError("Cannot serialize an empty DataFrame")
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Function < polars_2_zarr > requires the optional < zarr > dependency") from exc

    source_config, batch_config, source_columns = _normalize_config(df, config)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {path}")
    root = zarr.open_group(str(path), mode="w" if overwrite else "w-")

    storage_keys = {batch_key: _validate_storage_key(batch_key) for batch_key in batch_config}
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "length": df.height,
        "source_config": source_config,
        "batch_config": batch_config,
        "source_columns": source_columns,
        "storage_keys": storage_keys,
    }

    for batch_key, modality in batch_config.items():
        source_column = source_columns[batch_key]
        if modality == "string":
            _write_strings(root, df, source_column, batch_key, batch_size)
        elif modality == "graph":
            _write_graphs(root, df, source_column, batch_key, batch_size)
        else:
            _write_dense(root, df, source_column, batch_key, modality, batch_size)
    root.attrs["mm_dataset_manifest"] = json.dumps(manifest)
