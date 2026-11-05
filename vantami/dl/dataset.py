"""
Multimodal PyTorch datasets backed by Polars or lazily read Zarr stores.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class MMDataset(Dataset, ABC):
    """
    Abstract Base Class for multimodal datasets.
    """

    SUPPORTED_MODALITIES = frozenset({
        "string", "graph", "descriptor", "image", "target", "sample_weight", "group"
    })
    FEATURE_MODALITIES = frozenset({
        "string", "graph", "descriptor", "image"
    })
    _CANONICAL_KEYS = {
        "target": "y_true",
        "sample_weight": "y_wgts",
        "group": "group",
    }

    def __init__(self, config: Dict[str, str]) -> None:
        if not config:
            raise ValueError("< config > must not be empty")

        self.source_config = dict(config)
        self._validate_source_config(self.source_config)

        self.config = {}
        self.source_columns = {}
        self.feature_columns = []
        self.target_column = None
        self.weight_column = None
        self.group_column = None

        for source_name, modality in self.source_config.items():
            batch_key = self._CANONICAL_KEYS.get(modality, source_name)
            self.config[batch_key] = modality
            self.source_columns[batch_key] = source_name
            if modality in self.FEATURE_MODALITIES:
                self.feature_columns.append(source_name)
            elif modality == "target":
                self.target_column = source_name
            elif modality == "sample_weight":
                self.weight_column = source_name
            elif modality == "group":
                self.group_column = source_name

    @classmethod
    def _validate_source_config(cls, config: Dict[str, str]):
        unsupported = set(config.values()).difference(cls.SUPPORTED_MODALITIES)
        if unsupported:
            raise ValueError(f"Unsupported modalities: {sorted(unsupported)}")

        counts = defaultdict(int)
        for modality in config.values():
            counts[modality] += 1
        for modality in ("target", "sample_weight", "group"):
            if counts[modality] > 1:
                raise ValueError(f"Modality < {modality} > can only be assigned to one column")
        if counts["target"] != 1:
            raise ValueError("Exactly one column must have modality < target >")
        if not any(modality in cls.FEATURE_MODALITIES for modality in config.values()):
            raise ValueError("At least one feature modality is required")

    @abstractmethod
    def __len__(self) -> int:
        """
        Number of samples in the dataset.
        """

    @abstractmethod
    def _get_value(self, batch_key: str, idx: int):
        """
        Return one materialized value for < batch_key > and < idx >.
        """

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index < {index} > is out of range for dataset of length {len(self)}")
        return {batch_key: self._get_value(batch_key, index) for batch_key in self.config}

    def get_feature_columns(self):
        return self.feature_columns.copy()

    def get_target_column(self):
        return self.target_column

    def get_group_column(self):
        return self.group_column

    def get_modality_info(self):
        out = defaultdict(list)
        for batch_key, modality in self.config.items():
            out[modality].append(batch_key)
        return dict(out)

    def has_sample_weights(self) -> bool:
        return self.weight_column is not None

    def has_groups(self) -> bool:
        return self.group_column is not None


class PolarsMMDataset(MMDataset):
    """
    In-memory MMDataset over an already prepared Polars DataFrame.
    """

    def __init__(self, df: pl.DataFrame, config: Dict[str, str]) -> None:
        missing = set(config).difference(set(df.columns))
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {sorted(missing)}")
        super().__init__(config)
        self.df = df

    def __len__(self) -> int:
        return self.df.height

    def _get_value(self, batch_key: str, idx: int):
        """
        Materialize Polars nested numeric values as CPU tensors.
        """
        value = self.df[self.source_columns[batch_key]][idx]
        modality = self.config[batch_key]
        if modality in {"descriptor", "image", "target", "sample_weight"}:
            if isinstance(value, pl.Series):
                value = value.to_numpy()
            if isinstance(value, np.ndarray):
                return torch.from_numpy(np.array(value, copy=True))
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value)
        return value


class ZarrMMDataset(MMDataset):
    """
    Lazy, sample-indexed dataset stored by < polars_2_zarr >.
    """
    SCHEMA_VERSION = 1

    def __init__(self, path: Union[str, Path], config: Dict[str, str]) -> None:
        self.path = str(path)
        self._group = None
        self._owner_pid: Optional[int] = None
        root = self._open_group()
        try:
            manifest = json.loads(root.attrs["mm_dataset_manifest"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Not an MMDataset Zarr store: missing valid manifest") from exc
        if (version := manifest.get("schema_version")) != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version < {version} >; expected {self.SCHEMA_VERSION}")
        self._manifest = manifest
        self._length = int(manifest["length"])
        super().__init__(config)
        if self.config != manifest["batch_config"]:
            raise ValueError("Zarr manifest batch configuration is inconsistent")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_group"] = None
        state["_owner_pid"] = None
        return state

    def _open_group(self):
        if self._group is None or self._owner_pid != os.getpid():
            try:
                import zarr
            except ImportError as exc:
                raise ImportError("< ZarrMMDataset > requires the < zarr > library") from exc
            self._group = zarr.open_group(self.path, mode="r")
            self._owner_pid = os.getpid()
        return self._group

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _tensor(value: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(value, copy=True))

    def _get_value(self, batch_key: str, idx: int):
        root = self._open_group()
        modality = self.config[batch_key]
        storage_key = self._manifest["storage_keys"][batch_key]

        if modality == "string":
            node = root[f"strings/{storage_key}"]
            tokens = self._tensor(node["values"][idx])
            true_length = int(node["lengths"][idx])
            return tokens, true_length

        if modality == "graph":
            return self._read_graph(root[f"graphs/{storage_key}"], idx)

        node = root[f"dense/{storage_key}"]
        if node.attrs.get("encoding") == "categorical":
            category = node["categories"][int(node["codes"][idx])]
            return str(category)
        return self._tensor(node["values"][idx])

    def _read_graph(self, node: Any, idx: int) -> Any:
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError("Reading graph modalities requires torch-geometric") from exc

        n0, n1 = (int(x) for x in node["node_offsets"][idx:idx + 2])
        e0, e1 = (int(x) for x in node["edge_offsets"][idx:idx + 2])
        kwargs: Dict[str, Any] = {
            "x": self._tensor(node["x"][n0:n1]),
            "edge_index": self._tensor(node["edge_index"][:, e0:e1]),
        }
        metadata = json.loads(node.attrs["field_metadata"])
        for field, scope in metadata.items():
            if scope == "node":
                kwargs[field] = self._tensor(node[f"node/{field}"][n0:n1])
            elif scope == "edge":
                kwargs[field] = self._tensor(node[f"edge/{field}"][e0:e1])
            else:
                kwargs[field] = self._tensor(node[f"graph/{field}"][idx])
        return Data(**kwargs)


class MMBatch:
    """
    Batched sample dictionary with modality-aware convenience methods.
    """

    def __init__(self, data: Dict[str, Any], config: Mapping[str, str]):
        self.data = data
        self.config = dict(config)

    def __getitem__(self, key: str):
        return self.data[key]

    def __contains__(self, key: str):
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def get_modality(self, key: str) -> Optional[str]:
        return self.config.get(key)

    def get_by_modality(self, modality: str) -> Dict[str, Any]:
        return {key: self.data[key] for key, mod in self.config.items()
                if mod == modality and key in self.data}

    def to(self, device: Union[str, torch.device], non_blocking: bool = False):
        new_data = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.to(device, non_blocking=non_blocking)
            elif isinstance(value, tuple):
                new_data[key] = tuple(
                    item.to(device, non_blocking=non_blocking) if isinstance(item, torch.Tensor) else item
                    for item in value
                )
            elif hasattr(value, "to"):
                new_data[key] = value.to(device)
            else:
                new_data[key] = value
        return MMBatch(new_data, self.config)

    def __repr__(self) -> str:
        return f"MMBatch({list(self.data.keys())})"
