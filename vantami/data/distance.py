from typing import Optional, Tuple, List, Dict, Sequence, Union
from joblib import Parallel, delayed
from itertools import chain

import numpy as np
import numpy.typing as npt
import polars as pl

from scipy.spatial.distance import cdist
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


"""
Supporting functions
"""


Embedding = Union[npt.NDArray, ExplicitBitVect]
EmbeddingSequence = Sequence[Embedding]


rdkit_similarity_functions = {
    "jaccard": DataStructs.BulkTanimotoSimilarity,
    "tanimoto": DataStructs.BulkTanimotoSimilarity,
    "braunblanquet": DataStructs.BulkBraunBlanquetSimilarity,
    "dice": DataStructs.BulkDiceSimilarity,
    "kulczynski": DataStructs.BulkKulczynskiSimilarity,
    "mcconnaughey": DataStructs.BulkMcConnaugheySimilarity,
    "rogotgoldberg": DataStructs.BulkRogotGoldbergSimilarity,
    "russel": DataStructs.BulkRusselSimilarity,
    "sokal": DataStructs.BulkSokalSimilarity,
}


def _convert_embeddings(array: Union[npt.NDArray, Iterable[npt.NDArray]], metric: str = "jaccard"):
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
        "jaccard", "braunblanquet", "dice", "kulczynski",
        "mcconnaughey", "rogotgoldberg", "russel", "sokal",
    }

    other_metrics = {
        "euclidean", "minkowski", "cityblock", "seuclidean",
        "sqeuclidean", "correlation", "hamming", "jensenshannon",
        "chebyshev", "canberra", "braycurtis", "mahalanobis",
        "yule", "matching", "russellrao",
    }

    if metric not in rdkit_metrics | other_metrics:
        raise ValueError(f"Unsupported metric: {metric}.")

    if metric in other_metrics:
        return array

    def to_explicit_bitvect(bits: npt.NDArray) -> DataStructs.ExplicitBitVect:
        """
        Convert numpy's array to rdkit's ExplicitBitVect
        """

        bits = np.asarray(bits)

        if bits.ndim != 1:
            raise ValueError(f"Expected a 1D fingerprint, got {bits.ndim}D instead.")

        if not np.isin(bits, (0, 1)).all():
            raise ValueError("RDKit bit-vector conversion requires binary values: 0 or 1.")

        bit_vector = DataStructs.ExplicitBitVect(bits.size)
        bit_vector.SetBitsFromList(np.flatnonzero(bits).tolist())

        return bit_vector

    if isinstance(array, np.ndarray):
        return to_explicit_bitvect(array)

    if isinstance(array, list):
        if not all(isinstance(item, np.ndarray) for item in array):
            raise TypeError("Expected every item in array to be a NumPy array.")

        return [to_explicit_bitvect(bits) for bits in array]

    raise TypeError(
        f"Expected array to be a NumPy array or a list of NumPy arrays, got {type(array)} instead."
    )


def _rdkit_triu_b(array: Sequence[ExplicitBitVect], metric: str, threshold: Optional[float],
                  start_idx: int, end_idx: int, ):
    """
    Calculate strict upper-triangular RDKit distances for a row block.
    """
    similarity_fn = rdkit_similarity_functions[metric]
    rows = []

    for i in range(start_idx, end_idx):
        similarities = similarity_fn(array[i], array[i + 1:])

        distances = 1.0 - np.asarray(similarities, dtype=np.float64)

        if threshold is not None:
            distances = (distances <= threshold).astype(np.uint8)

        rows.append((i, distances))

    return start_idx, rows


def _rdkit_rect_b(query_array: Sequence[ExplicitBitVect], ref_array: Sequence[ExplicitBitVect], metric: str,
                  threshold: Optional[float], batch_idx: int):
    """
    Calculate one rectangular RDKit distance-matrix block.
    """
    similarity_fn = rdkit_similarity_functions[metric]
    rows = []

    for array in query_array:
        distances = 1.0 - np.asarray(similarity_fn(array, ref_array), dtype=np.float64)

        if threshold is not None:
            distances = (distances <= threshold).astype(np.uint8)

        rows.append(distances)

    return batch_idx, np.vstack(rows)


def _scipy_triu_b(array: npt.NDArray, metric: str, threshold: Optional[float],
                  start_idx: int, end_idx: int):
    """
    Calculate one upper-triangular SciPy block.
    """
    distances = cdist(XA=array[start_idx:end_idx], XB=array[start_idx + 1:], metric=metric)

    if threshold is not None:
        distances = (distances <= threshold).astype(np.uint8)

    for local_row in range(end_idx - start_idx):
        distances[local_row, :local_row] = 0

    return start_idx, distances


def _scipy_rect_b(query_array: npt.NDArray, ref_array: npt.NDArray, metric: str,
                  threshold: Optional[float], batch_idx: int):
    """
    Calculate a single rectangular SciPy distance-matrix block.
    """
    distances = cdist(XA=query_array, XB=ref_array, metric=metric)

    if threshold is not None:
        distances = (distances <= threshold).astype(np.uint8)

    return batch_idx, distances


def _as_rdkit_array(values: EmbeddingSequence, metric: str):
    """
    Return ExplicitBitVect objects, converting NumPy binary vectors if needed.
    """
    if len(values) == 0:
        return []

    all_rdkit = all(isinstance(value, ExplicitBitVect) for value in values)
    any_rdkit = any(isinstance(value, ExplicitBitVect) for value in values)

    if any_rdkit and not all_rdkit:
        raise TypeError(
            "Cannot mix NumPy arrays and ExplicitBitVect objects in the same sequence."
        )

    if all_rdkit:
        return values

    if not all(isinstance(value, np.ndarray) for value in values):
        raise TypeError(
            "RDKit metrics require a sequence containing either only NumPy "
            "arrays or only ExplicitBitVect objects."
        )

    return _convert_embeddings(values, metric=metric)


def _as_numpy_matrix(values: EmbeddingSequence, name: str):
    """
    Stack a sequence of equally sized one-dimensional NumPy arrays.
    """
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float64)

    if not all(isinstance(value, np.ndarray) for value in values):
        raise TypeError(f"SciPy metrics require every item in {name} to be a NumPy array.")

    if not all(value.ndim == 1 for value in values):
        raise ValueError(f"Expected every array in {name} to be one-dimensional.")

    return np.vstack(values)


def _k_smallest_rows(array: npt.NDArray, k: int) -> npt.NDArray:
    """
    Find the k smallest values across each row. NaN values are treated as positive infinity.
    """
    if array.shape[1] <= k:
        return np.nan_to_num(array, copy=True, nan=np.inf)

    values = np.nan_to_num(array, copy=True, nan=np.inf)

    col_idx = values.argpartition(k - 1, axis=1)[:, :k]
    row_idx = np.arange(values.shape[0])[:, None]

    return values[row_idx, col_idx]


def _k_largest_rows(array: npt.NDArray, k: int) -> npt.NDArray:
    """
    Find the k largest values across each row. NaN values are treated as negative infinity.
    """
    if array.shape[1] <= k:
        return np.nan_to_num(array, copy=True, nan=-np.inf)

    values = np.nan_to_num(array, copy=True, nan=-np.inf)

    col_idx = values.argpartition(-k, axis=1)[:, -k:]
    row_idx = np.arange(values.shape[0])[:, None]

    return values[row_idx, col_idx]


def _reduce_k_neighbors_block(distances: npt.NDArray, max_nearest_k: int, max_furthest_k: int):
    """
    Reduce one query-by-reference distance block.
    """
    distance_sums = np.nansum(distances, axis=1)
    distance_counts = np.sum(~np.isnan(distances), axis=1, dtype=np.int64)

    nearest_distances = _k_smallest_rows(distances, k=max_nearest_k)
    furthest_distances = _k_largest_rows(distances, k=max_furthest_k)

    return (
        distance_sums,
        distance_counts,
        nearest_distances,
        furthest_distances,
    )


def _rdkit_k_neighbors_block(query_array: Sequence[ExplicitBitVect], ref_array: Sequence[ExplicitBitVect],
                             metric: str, self_comparison: bool, ref_start_idx: int,
                             max_nearest_k: int, max_furthest_k: int, batch_idx: int):
    """
    Calculate exact distance statistics for one RDKit reference block.
    """
    similarity_fn = rdkit_similarity_functions[metric]

    n_query = len(query_array)
    n_ref = len(ref_array)

    distances = np.empty((n_query, n_ref), dtype=np.float64)

    for query_idx, fingerprint in enumerate(query_array):
        distances[query_idx] = 1.0 - np.asarray(similarity_fn(fingerprint, ref_array), dtype=np.float64)

    if self_comparison:
        reference_indices = np.arange(ref_start_idx, ref_start_idx + n_ref)
        valid_indices = reference_indices < n_query

        distances[reference_indices[valid_indices], np.flatnonzero(valid_indices)] = np.nan

    (distance_sums, distance_counts, nearest_distances, furthest_distances,) = (
        _reduce_k_neighbors_block(distances=distances,
                                  max_nearest_k=max_nearest_k,
                                  max_furthest_k=max_furthest_k,
    ))

    return (
        batch_idx,
        distance_sums,
        distance_counts,
        nearest_distances,
        furthest_distances,
    )


def _scipy_k_neighbors_block(query_array: npt.NDArray, ref_array: npt.NDArray, metric: str, self_comparison: bool,
                             ref_start_idx: int, max_nearest_k: int, max_furthest_k: int, batch_idx: int):
    """
    Calculate exact distance statistics for one SciPy reference block.
    """
    distances = cdist(XA=query_array, XB=ref_array, metric=metric)

    if self_comparison:
        reference_indices = np.arange(ref_start_idx, ref_start_idx + ref_array.shape[0])
        valid_indices = reference_indices < query_array.shape[0]

        distances[reference_indices[valid_indices], np.flatnonzero(valid_indices)] = np.nan

    (distance_sums, distance_counts, nearest_distances, furthest_distances) = (
        _reduce_k_neighbors_block(
            distances=distances,
            max_nearest_k=max_nearest_k,
            max_furthest_k=max_furthest_k,
    ))

    return (
        batch_idx,
        distance_sums,
        distance_counts,
        nearest_distances,
        furthest_distances,
    )


"""
Main functions
"""


def distance_matrix(array_1: EmbeddingSequence, array_2: Optional[EmbeddingSequence] = None, metric: str = "jaccard",
                    n_jobs: int = 1, batch_size: int = 1024, return_triu: bool = True, threshold: Optional[float] = None):
    """
    Compute an exact pairwise distance or binary neighbourhood matrix.

    RDKit is used for operations on binary vectors.
    Other metrics are passed to scipy.spatial.distance.cdist.

    Parameters
    ----------
    array_1: Sequence[numpy.ndarray | ExplicitBitVect]
        First sequence of arrays
    array_2: Sequence[numpy.ndarray | ExplicitBitVect] | None, optional
        Second sequence of arrays
        If None, array_1 is compared with itself (which is also faster). Default is None.
    metric: str, optional
        Distance metric to use. RDKit metrics:
            "jaccard", "braunblanquet", "dice", "kulczynski", "mcconnaughey",
            "rogotgoldberg","russel","sokal"
    n_jobs: int, optional
        Number of parallel worker processes. Default is 1.
    batch_size: int, optional
        Number of rows processed in each work block. Default is 1024.
    return_triu: bool, optional
        If True, return only the strict upper triangle. The diagonal and lower triangle are zero.
        Relevant only when array_2 is None.
    threshold : float | None, optional
        If numeric, return a uint8 binary neighborhood matrix in which a pair is 1 when
        its distance is less than or equal to threshold, otherwise 0. If None, return float64 distances.

    Returns
    -------
    numpy.ndarray
        Distance matrix or binary neighborhood matrix.

        For self-comparison with return_triu=True, only positions where
        i < j are populated.

        For rectangular comparisons, return_triu does not apply and a full
        matrix is returned.
    """

    if not isinstance(array_1, Sequence):
        raise TypeError(
            f"Expected array_1 to be a sequence of NumPy arrays or ExplicitBitVect objects, "
            f"got {type(array_1)} instead."
        )

    if array_2 is not None and not isinstance(array_2, Sequence):
        raise TypeError(
            "Expected array_2 to be a sequence of NumPy arrays, ExplicitBitVect objects, or None; "
            f"got {type(array_2)} instead."
        )

    if batch_size < 1:
        raise ValueError(f"Expected batch_size to be at least 1, got {batch_size} instead.")

    if (self_comparison := array_2 is None):
        array_2 = array_1

    output_dtype = np.uint8 if threshold is not None else np.float64

    similarity_fn = rdkit_similarity_functions.get(metric)

    # RDKit path
    if similarity_fn is not None:
        fingerprints_1 = _as_rdkit_array(
            array_1,
            metric=metric,
        )

        fingerprints_2 = (
            fingerprints_1 if self_comparison
            else _as_rdkit_array(array_2, metric=metric)
        )

        if (n_rows := len(fingerprints_1)) == 0:
            return np.empty((0, n_cols), dtype=output_dtype)

        if (n_cols := len(fingerprints_2)) == 0:
            return np.empty((n_rows, 0), dtype=output_dtype)

        # Self-comparison path; upper triu by default
        if self_comparison:
            result = np.zeros((n_rows, n_rows), dtype=output_dtype)

            if n_rows < 2:
                return result

            batches = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_rdkit_triu_b)(
                    array=fingerprints_1,
                    metric=metric,
                    threshold=threshold,
                    start_idx=start_idx,
                    end_idx=min(start_idx + batch_size, n_rows - 1),
                )
                for start_idx in range(0, n_rows - 1, batch_size)
            )

            for _, rows in sorted(batches, key=lambda batch: batch[0]):
                for row_idx, distances in rows:
                    result[row_idx, row_idx + 1:] = distances

            if not return_triu:
                result += result.T

            return result

        # rectangular matrix if array_2 is not None
        batches = [
            (batch_idx, fingerprints_1[start_idx:start_idx + batch_size])
            for batch_idx, start_idx in enumerate(range(0, n_rows, batch_size))
        ]

        chunks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_rdkit_rect_b)(
                query_array=batch,
                ref_array=fingerprints_2,
                metric=metric,
                threshold=threshold,
                batch_idx=batch_idx,
            )
            for batch_idx, batch in batches
        )

        return np.vstack(
            [
                block
                for _, block in sorted(
                    chunks,
                    key=lambda chunk: chunk[0],
                )
            ]
        )

    # Scipy path
    array_1 = _as_numpy_matrix(array_1, "array_1")

    array_2 = (array_1 if self_comparison else _as_numpy_matrix(array_2, "array_2"))

    if (n_rows := array_1.shape[0]) == 0:
        return np.empty((0, n_cols), dtype=output_dtype)

    if (n_cols := array_2.shape[0]) == 0:
        return np.empty((n_rows, 0), dtype=output_dtype)

    if array_1.shape[1] != array_2.shape[1]:
        raise ValueError("Embeddings in array_1 and array_2 must have the same length.")

    # "triu"-like calculations
    if self_comparison:
        result = np.zeros((n_rows, n_rows), dtype=output_dtype)

        if n_rows < 2:
            return result

        blocks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_scipy_triu_b)(
                array=array_1,
                metric=metric,
                threshold=threshold,
                start_idx=start_idx,
                end_idx=min(start_idx + batch_size, n_rows - 1),
            )
            for start_idx in range(0, n_rows - 1, batch_size)
        )

        for start_idx, block in sorted(blocks, key=lambda block: block[0]):
            end_idx = start_idx + block.shape[0]
            result[start_idx:end_idx, start_idx + 1:] = block

        if not return_triu:
            result += result.T

        return result

    # if array_2 is not None
    batches = [
        (batch_idx, array_1[start_idx:start_idx + batch_size])
        for batch_idx, start_idx in enumerate(range(0, n_rows, batch_size))
    ]

    chunks = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_scipy_rect_b)(
            query_array=batch,
            ref_array=array_2,
            metric=metric,
            threshold=threshold,
            batch_idx=batch_idx,
        )
        for batch_idx, batch in batches
    )

    return np.vstack(
        [block for _, block in sorted(chunks, key=lambda chunk: chunk[0])]
    )


def k_neighbors_distance(query_array: EmbeddingSequence, ref_array: Optional[EmbeddingSequence] = None,
                         metric: str = "jaccard", n_jobs: int = 1, batch_size: int = 1024,
                         nearest_k: Optional[Union[int, List[int]]] = None,
                         furthest_k: Optional[Union[int, List[int]]] = None):
    """
    Calculate distance statistics between points in query and reference arrays. For each entry in query array,
    calculates the minimum, mean, and maximum distance to all entries in reference array. If it is not passed,
    a self-comparison is made (where distances to self are masked and not included).

    Additionally, computes the average distance to the k nearest and k furthest neighbors,
    where k values are specified in nearest_k and furthest_k parameters.

    Parameters
    ----------
    query_array: npt.NDArray
        Query array, typically the test set.
    ref_array: npt.NDArray, optional
        Reference array, typically the training set. If not passed, self-comparison.
    metric: str, optional
        Distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int, optional
        Number of jobs for parallel processing. Default is 1.
    batch_size: int, optional
        Number of rows to process in a batch. Default is 1024.
    nearest_k: Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k nearest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].
    furthest_k: Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k furthest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].

    Returns
    -------
    pl.DataFrame
        A DataFrame containing distance statistics for each entry in query array:
        - 'Min': Minimum distance (equivalent to '1 Nearest')
        - 'Mean': Average distance to all entries in reference array
        - 'Max': Maximum distance (equivalent to '1 Furthest')
        - '{k} Nearest': Average distance to k nearest neighbors for each k in nearest_k
        - '{k} Furthest': Average distance to k furthest neighbors for each k in furthest_k
    """
    if not isinstance(query_array, Sequence):
        raise TypeError("Expected query_array to be a sequence of NumPy arrays or "
                        f"ExplicitBitVect objects, got {type(query_array)} instead."
        )

    if ref_array is not None and not isinstance(ref_array, Sequence):
        raise TypeError("Expected ref_array to be a sequence of NumPy arrays, ExplicitBitVect objects, or None "
                        f"got {type(ref_array)} instead."
        )

    if batch_size < 1:
        raise ValueError(f"Expected batch_size to be at least 1, got {batch_size} instead.")

    if not (isinstance(nearest_k, (int, list)) or nearest_k is None):
        raise TypeError(f"Expected nearest_k to be int, list, or None got {type(nearest_k)} instead.")

    if not (isinstance(furthest_k, (int, list)) or furthest_k is None):
        raise TypeError(f"Expected furthest_k to be int, list, or None; got {type(furthest_k)} instead.")

    if isinstance(nearest_k, int):
        nearest_k = [nearest_k]

    if isinstance(furthest_k, int):
        furthest_k = [furthest_k]

    nearest_k = ([1] if nearest_k is None else sorted(set(nearest_k + [1])))
    furthest_k = ([1] if furthest_k is None else sorted(set(furthest_k + [1])))

    if (self_comparison := ref_array is None):
        ref_array = query_array if self_comparison else ref_array

    if (available_neighbours := len(ref_array) - int(self_comparison)) < 1:
        raise ValueError("Self-comparison requires at least two entries."
                         "A reference comparison requires at least one reference entry."
        )

    max_nearest_k = max(nearest_k)
    max_furthest_k = max(furthest_k)

    if max_nearest_k > available_neighbours:
        raise ValueError(
            f"Requested nearest_k={max_nearest_k}, but only {available_neighbours} neighbours are available."
        )

    if max_furthest_k > available_neighbours:
        raise ValueError(
            f"Requested furthest_k={max_furthest_k}, but only {available_neighbours} neighbours are available."
        )

    reference_starts = range(0, len(ref_array), batch_size)

    # RDKit route
    if metric in rdkit_similarity_functions:
        query_array = _as_rdkit_array(query_array, metric=metric)

        ref_array = query_array if self_comparison else _as_rdkit_array(ref_array, metric=metric)

        blocks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_rdkit_k_neighbors_block)(
                query_array=query_array,
                ref_array=ref_array[ref_start_idx:ref_start_idx + batch_size],
                metric=metric,
                self_comparison=self_comparison,
                ref_start_idx=ref_start_idx,
                max_nearest_k=max_nearest_k,
                max_furthest_k=max_furthest_k,
                batch_idx=batch_idx,
            )
            for batch_idx, ref_start_idx in enumerate(reference_starts)
        )

    # SciPy route
    else:
        query_array = _as_numpy_matrix(query_array, "query_array")

        ref_array = query_array if self_comparison else _as_numpy_matrix(ref_array, "ref_array")

        if query_array.shape[1] != ref_array.shape[1]:
            raise ValueError("Query and reference arrays must have the same length.")

        blocks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_scipy_k_neighbors_block)(
                query_array=query_array,
                ref_array=ref_array[ref_start_idx:ref_start_idx + batch_size],
                metric=metric,
                self_comparison=self_comparison,
                ref_start_idx=ref_start_idx,
                max_nearest_k=max_nearest_k,
                max_furthest_k=max_furthest_k,
                batch_idx=batch_idx,
            )
            for batch_idx, ref_start_idx in enumerate(reference_starts)
        )

    blocks = sorted(blocks, key=lambda block: block[0])

    distance_sums = np.sum([block[1] for block in blocks], axis=0)
    distance_counts = np.sum([block[2] for block in blocks], axis=0)

    mean_distances = distance_sums / distance_counts

    nearest_candidates = np.hstack([block[3] for block in blocks])
    furthest_candidates = np.hstack([block[4] for block in blocks])

    nearest_distances = np.sort(_k_smallest_rows(nearest_candidates, k=max_nearest_k), axis=1)
    furthest_distances = np.sort(_k_largest_rows(furthest_candidates, k=max_furthest_k), axis=1)

    result = pl.DataFrame({"Mean": mean_distances})

    result = result.with_columns([
            pl.Series(f"{k} Nearest", nearest_distances[:, :k].mean(axis=1))
            for k in nearest_k
    ])

    result = result.with_columns([
            pl.Series(f"{k} Furthest", furthest_distances[:, -k:].mean(axis=1))
            for k in furthest_k
    ])

    return result.rename({
            "1 Nearest": "Min",
            "1 Furthest": "Max"
    })


def group_k_neighbors_distance(df: pl.DataFrame, features_col: str, group_col: str, metric: str = "jaccard",
                               n_jobs: int = 1, batch_size: int = 1024, nearest_k: Optional[List[int]] = None,
                               furthest_k: Optional[List[int]] = None):
    """
    Calculate intra- and intergroup distance distribution. The group_col should contain either
    integers or lists of integers.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        The name of the column with features to use for distance calculation
    group_col: str
        The name of the column with group assignments for comparison.
    metric : str, optional
        Distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is 1.
    batch_size : int, optional
        Number of rows to process in a single batch. Default is 1024.
    nearest_k : Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k nearest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].
    furthest_k : Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k furthest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].

    Returns
    -------
    pl.DataFrame
    """

    required_columns = [features_col, group_col]

    if (missing_columns := [column for column in required_columns if column not in df.columns]):
        raise KeyError(f"Missing required columns: {missing_columns}.")

    group_type = df[group_col].dtype

    # Assume each entry belongs to a single group
    if isinstance(group_type, (pl.Int32, pl.Int64, pl.String)):
        unique_groups = df[group_col].drop_nulls().unique().to_list()
        print(f"Group column treated a Single-assignment")
        group_type = 'Single'

    # Assume each entry *might* belong to multiple groups
    elif isinstance(group_type, pl.List):
        unique_groups = set(chain.from_iterable(df[group_col].drop_nulls().unique().to_list()))
        print(f"Group column treated a Multiple-assignment")
        group_type = 'Multiple'

    else:
        raise TypeError(f"Expected the dtype of group_col to be either Int32, Int64, or String for single group"
                        f"assignment, and pl.List for multi-group assignments. Got {df[group_col].dtype} instead.")

    if not unique_groups:
        return None

    results = []

    for group in unique_groups:
        if group_type == "Single":
            group_mask = pl.col(group_col) == group
        else:
            group_mask = pl.col(group_col).list.contains(group)

        query_array = [a.reshape(-1) for a in df.filter(group_mask)[features_col].to_numpy()]
        ref_array = [a.reshape(-1) for a in df.filter(~group_mask)[features_col].to_numpy()]

        if len(query_array) < 2:
            raise ValueError(
                f"Group {group} contains only {len(query_array)} member(s). "
                "At least two members are required for intra-group statistics."
            )

        if len(ref_array) == 0:
            raise ValueError(
                f"Group {group} has no entries outside the group. "
                "Inter-group statistics cannot be calculated."
            )

        intra_knd = k_neighbors_distance(
            query_array=query_array,
            ref_array=None,
            metric=metric,
            n_jobs=n_jobs,
            batch_size=batch_size,
            nearest_k=nearest_k,
            furthest_k=furthest_k,
        )

        inter_knd = k_neighbors_distance(
            query_array=query_array,
            ref_array=ref_array,
            metric=metric,
            n_jobs=n_jobs,
            batch_size=batch_size,
            nearest_k=nearest_k,
            furthest_k=furthest_k,
        )

        intra_df = pl.DataFrame({
                "Scope": ["Intra"] * len(intra_knd.columns),
                "Group": [group] * len(intra_knd.columns),
                "Aggregation": intra_knd.columns,
                "Values": [intra_knd.get_column(column).to_list() for column in intra_knd.columns]
        })

        inter_df = pl.DataFrame({
                "Scope": ["Inter"] * len(inter_knd.columns),
                "Group": [group] * len(inter_knd.columns),
                "Aggregation": inter_knd.columns,
                "Values": [inter_knd.get_column(column).to_list() for column in inter_knd.columns]
        })

        results.append(pl.concat([intra_df, inter_df], how="vertical_relaxed"))

    return pl.concat(results, how="vertical_relaxed")


def list_similarity(string: str, target_strings: list, method: str = 'fuzzy', threshold: float = 0.75,
                    num_matches: int = 3):
    """
    Find the 3 most similar strings in passed target_strings using either fuzzy matching or Levenshtein distance.
    Parameters
    ----------
    string : str
        The input string to compare against the internal database.
    target_strings : list
        The list of strings to search for similarities
    method : str, optional
        The similarity method to use, either 'fuzzy' or 'levenshtein'. Default is 'fuzzy'.
    threshold : float, optional
        The minimum similarity threshold for considering a match. Default is 0.75
    num_matches: int, optional
        Number of top matches to return

    Returns
    -------
    list[tuple]
        A list of up to 3 tuples, each containing a matching string and its similarity score,
        sorted by similarity in descending order.
    """
    try:
        import Levenshtein
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("Function < list_similarity > requires < Levenshtein > and < rapidfuzz > libraries."
                          "Please install them using < pip install Levenshtein rapidfuzz >")

    similarities = []

    for item in target_strings:
        if method == 'levenshtein':
            sim = np.round(1 - Levenshtein.distance(string, item) / max(len(string), len(item)), 3)
        elif method == 'fuzzy':
            sim = np.round(fuzz.ratio(string, item) / 100.0, 3)
        else:
            raise ValueError(f"Available options for method are: levenshtein, fuzzy")
        if sim >= threshold:
            similarities.append((item, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:num_matches]


def dict_similarity(string: str, target_mapping: dict, method: str = 'fuzzy', threshold: float = 0.75,
                    num_matches: int = 3):
    """
    Find the most similar strings in keys of passed mapping using either
    fuzzy matching or Levenshtein distance.

    Parameters
    ----------
    string : str
        The input string to compare against the internal database.
    target_mapping : dict
        The mapping from trade_name to ingredients (dict)
    method : str, optional
        The similarity method to use, either 'fuzzy' or 'levenshtein'. Default is 'fuzzy'.
    threshold : float, optional
        The minimum similarity threshold for considering a match. Default is 0.75
    num_matches: int, optional
        Number of top matches to return

    Returns
    -------
    list[tuple]
        A list of up to num_matches tuples, each containing a matching string and its similarity score,
        sorted by similarity in descending order.
    """
    try:
        import Levenshtein
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("Function < dict_similarity > requires < Levenshtein > and < rapidfuzz > libraries."
                          "Please install them using < pip install Levenshtein rapidfuzz >")

    similarities = []

    for key, value in target_mapping.items():  # trade_name : mixture
        if method == 'levenshtein':
            sim = np.round(1 - Levenshtein.distance(string, key) / max(len(string), len(key)), 3)
        elif method == 'fuzzy':
            sim = np.round(fuzz.ratio(string, key) / 100.0, 3)
        else:
            raise ValueError(f"Available options for method are: levenshtein, fuzzy")

        if sim >= threshold:
            similarities.append((key, value, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)

    return similarities[:num_matches]


def string_distance(string_1, string_2, method: str = 'levenshtein'):
    """
    Calculate distance between two strings.

    Parameters
    ----------
    string_1: str
        First string to compare
    string_2: str
        Second string to compare
    method: str
        Distance calculation method. Default is 'levenshtein'. Alternative is fuzzy

    Returns
    -------
    distance: float
        Calculated string distance
    """
    try:
        import Levenshtein
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("Function < dict_similarity > requires < Levenshtein > and < rapidfuzz > libraries."
                          "Please install them using < pip install Levenshtein rapidfuzz >")

    if not all([isinstance(string_1, str), isinstance(string_2, str)]):
        raise TypeError(f"Expected both passed values to be strings, got {type(string_1)} and {type(string_2)} instead")

    if method == "levenshtein":
        distance = np.round(Levenshtein.distance(string_1, string_2) / max(len(string_1), len(string_2)), 3)
    elif method == "fuzzy":
        distance = 1 - np.round(fuzz.ratio(string_1, string_2) / 100.0, 3)
    else:
        raise ValueError(f"Available options for method are: levenshtein, fuzzy")

    return distance
