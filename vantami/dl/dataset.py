"""
Multimodal PyTorch datasets backed by Polars or lazily read Zarr stores.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

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


class LMDBMMDataset(MMDataset):
    """
    Lazy, sample-indexed multimodal dataset backed by LMDB.

    Expected LMDB layout
    --------------------
    Reserved metadata key:
        b"__len__" -> pickled int

    Sample key:
        index.to_bytes(8, byteorder="big", signed=False)

    Sample value:
        pickled dictionary whose keys correspond to the *source column names*
        passed in ``config``.

    Example
    -------
    config = {
        "graph": "graph",
        "mordred": "target",
        "molecule_id": "string",
    }

    record = {
        "graph": {
            "x": npt.NDArray,
            "edge_index": npt.NDArray,
            "edge_attr": npt.NDArray,
            "pos": npt.NDArray,
            "atomic_numbers": npt.NDArray,
        },
        "mordred": npt.NDArray,
        "molecule_id": "mol_12345",
    }
    """

    LENGTH_KEY = b"__len__"
    KEY_WIDTH = 8

    def __init__(self, path: str | Path, config: Dict[str, str], *, subdir: bool = False,
                 key_fn: Optional[Callable[[int], bytes]] = None, readahead: bool = False,
                 max_readers: int = 256) -> None:
        self.path = str(path)
        self.subdir = subdir
        self.readahead = readahead
        self.max_readers = max_readers
        self.key_fn = key_fn or self.index_to_key

        self._env = None
        self._owner_pid: Optional[int] = None

        env = self._open_env()
        try:
            with env.begin(write=False) as txn:
                raw_length = txn.get(self.LENGTH_KEY)

            if raw_length is None:
                raise ValueError(
                    f"LMDB store < {self.path} > does not contain the required length key < {self.LENGTH_KEY!r} >."
                )

            self._length = int(pickle.loads(bytes(raw_length)))
        finally:
            self.close()

        super().__init__(config)

    @classmethod
    def index_to_key(cls, index: int) -> bytes:
        """
        Encode a non-negative dataset index as a fixed-width LMDB key.
        """
        if index < 0:
            raise ValueError(f"Expected a non-negative index, got {index}.")

        max_index = (1 << (8 * cls.KEY_WIDTH)) - 1
        if index > max_index:
            raise ValueError(f"Index < {index} > exceeds the maximum {cls.KEY_WIDTH}-byte unsigned integer.")

        return index.to_bytes(cls.KEY_WIDTH, byteorder="big", signed=False)

    def __getstate__(self):
        """
        Prevent an open LMDB environment from being passed to DataLoader workers.
        """
        state = self.__dict__.copy()
        state["_env"] = None
        state["_owner_pid"] = None
        return state

    def _open_env(self):
        """
        Open a read-only LMDB environment lazily for the current process.
        """
        current_pid = os.getpid()

        if self._env is None or self._owner_pid != current_pid:
            try:
                import lmdb
            except ImportError as exc:
                raise ImportError("< LMDBMMDataset > requires the < lmdb > library.") from exc

            self.close()

            self._env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=self.readahead,
                meminit=False,
                max_readers=self.max_readers,
                subdir=self.subdir,
            )
            self._owner_pid = current_pid

        return self._env

    def close(self) -> None:
        """
        Close the read-only LMDB environment for this process.
        """
        if self._env is not None:
            self._env.close()

        self._env = None
        self._owner_pid = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        """
        Materialize a dense value as a detached CPU tensor.
        """
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()

        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.array(value, copy=True))

        return torch.as_tensor(value)

    @classmethod
    def _to_graph(cls, value: Any):
        """
        Materialize an LMDB graph record as PyG Data.

        Accepted graph representations:
        - an already serialized PyG Data object;
        - a mapping of Data field names to arrays/tensors/scalars.
        """
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError("Reading graph modalities requires < torch-geometric >.") from exc

        if isinstance(value, Data):
            return value.clone()

        if not isinstance(value, Mapping):
            raise TypeError("Graph records must be either PyG Data objects or mappings, got < {type(value)} >.")

        if "edge_index" not in value:
            raise ValueError("Graph record is missing required field < edge_index >.")

        kwargs: Dict[str, Any] = {}

        for field, field_value in value.items():
            if isinstance(field_value, (np.ndarray, torch.Tensor, list, tuple)):
                tensor = cls._to_tensor(field_value)

                if field == "edge_index":
                    tensor = tensor.to(torch.long)

                kwargs[field] = tensor
            else:
                kwargs[field] = field_value

        return Data(**kwargs)

    def _read_record(self, idx: int) -> Mapping[str, Any]:
        env = self._open_env()
        key = self.key_fn(idx)

        with env.begin(write=False, buffers=True) as txn:
            raw = txn.get(key)

        if raw is None:
            raise KeyError(f"No LMDB record exists for dataset index < {idx} > under key < {key!r} >.")

        record = pickle.loads(bytes(raw))

        if not isinstance(record, Mapping):
            raise TypeError(f"LMDB record at index < {idx} > must deserialize to a mapping, got < {type(record)} >.")

        return record

    def _get_value(self, batch_key: str, idx: int):
        record = self._read_record(idx)

        source_column = self.source_columns[batch_key]
        modality = self.config[batch_key]

        if source_column not in record:
            raise KeyError(
                f"LMDB record < {idx} > does not contain < {source_column} > required for batch key < {batch_key} >."
            )

        value = record[source_column]

        if modality == "graph":
            return self._to_graph(value)

        if modality in {"descriptor", "image", "target", "sample_weight"}:
            return self._to_tensor(value)

        return value

    def __getitem__(self, index: int):
        """
        Read the LMDB record once, then materialize all configured modalities.
        """
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(f"Index < {index} > out of range for dataset of length {len(self)}.")

        record = self._read_record(index)
        output: Dict[str, Any] = {}

        for batch_key, modality in self.config.items():
            source_column = self.source_columns[batch_key]

            if source_column not in record:
                raise KeyError(
                    f"LMDB record < {index} > does not contain < {source_column} > required for batch key < {batch_key} >."
                )

            value = record[source_column]

            if modality == "graph":
                output[batch_key] = self._to_graph(value)
            elif modality in {"descriptor", "image", "target", "sample_weight"}:
                output[batch_key] = self._to_tensor(value)
            else:
                output[batch_key] = value

        return output


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
