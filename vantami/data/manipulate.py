from typing import Union, List, Tuple, Optional, Iterable
import pickle
import os

import numpy as np
import pandas as pd
import polars as pl
from rdkit import DataStructs


def bin_data(data: Iterable[float], n_bins: int = 5):
    """
    Assign entries in data into bins.

    Parameters
    ----------
    data: Iterable[float]
        E.g. list, 1D np.array, pd/pl.Series
    n_bins: int
        Into how many bins data should be assigned

    Returns
    -------
    bins: List[int]
    """
    quantiles = list(np.quantile(data, q=np.linspace(0, 1, n_bins+1)[1:-1]))

    def to_bin(value: float, qnts: List[float]):
        for idx, quantile in enumerate(qnts):
            if value < quantile:
                return idx + 1
        return len(quantiles) + 1

    bins = [to_bin(value, qnts=quantiles) for value in data]
    return bins


def is_valid_fingerprint(fingerprint: np.ndarray) -> bool:
    """
    Check if numpy array contains only [0,1] values

    Parameters
    ----------
    fingerprint: np.ndarray
        The array to check

    Returns
    -------
    bool
    """

    return np.all(np.isin(fingerprint, [0, 1]))


def embeddings_to_rdkit(embeddings: Iterable[np.ndarray]) -> list:
    """
    Convert numpy arrays to RDKit's native DataStructs instance

    Parameters
    ----------
    embeddings: Iterable[np.ndarray]
        Embeddings to convert (e.g. List[np.ndarray])

    Returns
    -------
    fingerprints: List
    """

    fingerprints = []
    for emb in embeddings:
        fingerprint = ndarray_to_binary_string(emb)
        fingerprints.append(DataStructs.CreateFromBitString(fingerprint))
    return fingerprints


def ndarray_to_binary_string(array: np.ndarray) -> str:
    """
    Convert a binary fingerprint represented as numpy array string.

    Parameters
    ----------
    array: np.ndarray
        The array to convert

    Returns
    -------
    str
    """

    if not len(array) or not is_valid_fingerprint(array):
        raise ValueError(
            "Invalid fingerprint array. Expected binary Morgan fingerprint with 0s and 1s."
        )
    return "".join(array.astype(str).tolist())


def convert_embeddings(array: Union[npt.NDArray, Iterable[npt.NDArray]], metric: str = "jaccard"):
    """
    Convert binary NumPy fingerprints into RDKit ExplicitBitVect objects when
    an RDKit fingerprint-similarity metric is selected.

    Continuous-distance inputs are returned unchanged.

    Parameters
    ----------
    array: Union[npt.NDArray, Iterable[npt.NDArray]]
        A numpy array or an iterable of numpy arrays.
    metric: str
        Distance metric that will be used.

    Returns
    -------
    Union[npt.NDArray, ExplicitBitVect, List[npt.NDArray], List[ExplicitBitVect]]
    """

    rdkit_metrics = {
        "jaccard",
        "braunblanquet",
        "dice",
        "kulczynski",
        "mcconnaughey",
        "rogotgoldberg",
        "russel",
        "sokal",
    }

    other_metrics = {
        "euclidean",
        "minkowski",
        "cityblock",
        "seuclidean",
        "sqeuclidean",
        "correlation",
        "hamming",
        "jensenshannon",
        "chebyshev",
        "canberra",
        "braycurtis",
        "mahalanobis",
        "yule",
        "matching",
        "russellrao",
    }

    if metric not in rdkit_metrics | other_metrics:
        raise ValueError(f"Unsupported metric: {metric!r}.")

    if metric in other_metrics:
        return array

    def to_explicit_bitvect(bits: npt.NDArray) -> DataStructs.ExplicitBitVect:
        """
        Convert numpy's array to rdkit's ExplicitBitVect
        """

        bits = np.asarray(bits)

        if bits.ndim != 1:
            raise ValueError(
                f"Expected a 1D fingerprint, got {bits.ndim}D instead."
            )

        if not np.isin(bits, (0, 1)).all():
            raise ValueError(
                "RDKit bit-vector conversion requires binary values: 0 or 1."
            )

        bit_vector = DataStructs.ExplicitBitVect(bits.size)
        bit_vector.SetBitsFromList(np.flatnonzero(bits).tolist())

        return bit_vector

    if isinstance(array, np.ndarray):
        return to_explicit_bitvect(array)

    if isinstance(array, list):
        if not all(isinstance(item, np.ndarray) for item in array):
            raise TypeError(
                "Expected every item in array to be a NumPy array."
            )

        return [to_explicit_bitvect(bits) for bits in array]

    raise TypeError(
        f"Expected array to be a NumPy array or a list of NumPy arrays, got {type(array)} instead."
    )
