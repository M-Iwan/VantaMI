from typing import Optional, Tuple, List, Dict, Sequence
from joblib import Parallel, delayed
from itertools import chain

import numpy as np
import numpy.typing as npt
import polars as pl

from scipy.spatial.distance import cdist
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from vantami.data.manipulate import convert_embeddings


"""
Supporting functions for distance_matrix 
"""


Embedding = Union[npt.NDArray, ExplicitBitVect]
EmbeddingSequence = Sequence[Embedding]


RDKIT_SIMILARITY_FUNCTIONS = {
    "jaccard": DataStructs.BulkTanimotoSimilarity,
    "braunblanquet": DataStructs.BulkBraunBlanquetSimilarity,
    "dice": DataStructs.BulkDiceSimilarity,
    "kulczynski": DataStructs.BulkKulczynskiSimilarity,
    "mcconnaughey": DataStructs.BulkMcConnaugheySimilarity,
    "rogotgoldberg": DataStructs.BulkRogotGoldbergSimilarity,
    "russel": DataStructs.BulkRusselSimilarity,
    "sokal": DataStructs.BulkSokalSimilarity,
}


def _rdkit_triu_b(array: Sequence[ExplicitBitVect], metric: str, threshold: Optional[float],
                  start_idx: int, end_idx: int, ):
    """
    Calculate strict upper-triangular RDKit distances for a row block.
    """
    similarity_fn = RDKIT_SIMILARITY_FUNCTIONS[metric]
    rows = []

    for i in range(start_idx, end_idx):
        similarities = similarity_fn(
            array[i],
            array[i + 1:],
        )

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
    similarity_fn = RDKIT_SIMILARITY_FUNCTIONS[metric]
    rows = []

    for array in query_array:
        similarities = similarity_fn(
            array,
            ref_array,
        )

        distances = 1.0 - np.asarray(similarities, dtype=np.float64)

        if threshold is not None:
            distances = (distances <= threshold).astype(np.uint8)

        rows.append(distances)

    return batch_idx, np.vstack(rows)


def _scipy_triu_b(matrix: npt.NDArray, metric: str, threshold: Optional[float],
                  start_idx: int, end_idx: int):
    """
    Calculate one upper-triangular SciPy block.
    """
    distances = cdist(
        XA=matrix[start_idx:end_idx],
        XB=matrix[start_idx + 1:],
        metric=metric,
    )

    if threshold is not None:
        distances = (distances <= threshold).astype(np.uint8)

    for local_row in range(end_idx - start_idx):
        distances[local_row, :local_row] = 0

    return start_idx, distances


def _scipy_rect_b(query_matrix: npt.NDArray, ref_matrix: npt.NDArray, metric: str,
                  threshold: Optional[float], batch_idx: int):
    """
    Calculate one rectangular SciPy distance-matrix block.
    """
    distances = cdist(
        XA=query_matrix,
        XB=reference_matrix,
        metric=metric,
    )

    if threshold is not None:
        distances = (distances <= threshold).astype(np.uint8)

    return batch_idx, distances


def _as_rdkit_array(values: EmbeddingSequence, metric: str):
    """
    Return ExplicitBitVect objects, converting NumPy binary vectors if needed.
    """
    if len(values) == 0:
        return []

    all_rdkit = all(
        isinstance(value, ExplicitBitVect)
        for value in values
    )

    any_rdkit = any(
        isinstance(value, ExplicitBitVect)
        for value in values
    )

    if any_rdkit and not all_rdkit:
        raise TypeError(
            "Cannot mix NumPy arrays and ExplicitBitVect objects in the same sequence."
        )

    if all_rdkit:
        return list(values)

    if not all(isinstance(value, np.ndarray) for value in values):
        raise TypeError(
            "RDKit metrics require a sequence containing either only NumPy "
            "arrays or only ExplicitBitVect objects."
        )

    converted = convert_embeddings(
        list(values),
        metric=metric,
    )
    return converted


def _as_numpy_matrix(values: EmbeddingSequence, name: str):
    """
    Stack a sequence of equally sized one-dimensional NumPy arrays.
    """
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float64)

    if not all(isinstance(value, np.ndarray) for value in values):
        raise TypeError(
            f"SciPy metrics require every item in {name} to be a NumPy array."
        )

    if not all(value.ndim == 1 for value in values):
        raise ValueError(
            f"Expected every array in {name} to be one-dimensional."
        )

    return np.vstack(values)


"""
Main functions
"""


def distane_matrix(array_1: EmbeddingSequence, array_2: Optional[EmbeddingSequence] = None, metric: str = "jaccard",
                   n_jobs: int = 1, batch_size: int = 1024, return_triu: bool = True, threshold: Optional[float] = None):
    """
    Compute an exact pairwise distance or binary neighbourhood matrix.

    RDKit is used for metrics in RDKIT_SIMILARITY_FUNCTIONS.
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
        If numeric, return a uint8 binary neighbourhood matrix in which a pair is 1 when
        its distance is less than or equal to threshold, otherwise 0. If None, return float64 distances.

    Returns
    -------
    numpy.ndarray
        Distance matrix or binary neighbourhood matrix.

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

    self_comparison = array_2 is None

    values_1 = list(array_1)
    values_2 = values_1 if self_comparison else list(array_2)

    output_dtype = np.uint8 if threshold is not None else np.float64

    similarity_fn = RDKIT_SIMILARITY_FUNCTIONS.get(metric)

    # RDKit path

    if similarity_fn is not None:
        fingerprints_1 = _as_rdkit_array(
            values_1,
            metric=metric,
        )

        fingerprints_2 = (
            fingerprints_1 if self_comparison
            else _as_rdkit_array(values_2, metric=metric)
        )

        n_rows = len(fingerprints_1)
        n_cols = len(fingerprints_2)

        if n_rows == 0:
            return np.empty((0, n_cols), dtype=output_dtype)

        if n_cols == 0:
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
            (
                batch_idx,
                fingerprints_1[start_idx:start_idx + batch_size],
            )
            for batch_idx, start_idx in enumerate(
                range(0, n_rows, batch_size)
            )
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
    matrix_1 = _as_numpy_matrix(values_1, "array_1")

    matrix_2 = (
        matrix_1 if self_comparison
        else _as_numpy_matrix(values_2, "array_2")
    )

    n_rows = matrix_1.shape[0]
    n_cols = matrix_2.shape[0]

    if n_rows == 0:
        return np.empty((0, n_cols), dtype=output_dtype)

    if n_cols == 0:
        return np.empty((n_rows, 0), dtype=output_dtype)

    if matrix_1.shape[1] != matrix_2.shape[1]:
        raise ValueError(
            "Embeddings in array_1 and array_2 must have the same length."
        )

    # "triu"-like calculations
    if self_comparison:
        result = np.zeros((n_rows, n_rows), dtype=output_dtype)

        if n_rows < 2:
            return result

        blocks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_scipy_triu_b)(
                matrix=matrix_1,
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
        (
            batch_idx,
            matrix_1[start_idx:start_idx + batch_size],
        )
        for batch_idx, start_idx in enumerate(
            range(0, n_rows, batch_size)
        )
    ]

    chunks = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_scipy_rect_b)(
            query_matrix=batch,
            ref_matrix=matrix_2,
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


def k_smallest_rows(array: npt.NDArray, k: int):
    """
    Find the k smallest values across each row.
    """
    # Not enough columns, just return the array
    if array.shape[1] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=np.inf)

    col_idx = array.argpartition(k, axis=1)[:, :k]
    row_idx = np.arange(array.shape[0])[:, None]
    return array[row_idx, col_idx]


def k_smallest_columns(array: npt.NDArray, k: int):
    """
    Find the k smallest values across each column.
    """
    # Not enough rows, just return the array
    if array.shape[0] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=np.inf)

    row_idx = array.argpartition(k, axis=0)[:k, :]
    col_idx = np.arange(array.shape[1])[None, :]
    return array[row_idx, col_idx]


def k_largest_rows(array: np.ndarray, k: int):
    """
    Find the k largest values across each row.
    """
    if array.shape[1] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=-np.inf)

    col_idx = array.argpartition(-k, axis=1)[:, -k:]
    row_idx = np.arange(array.shape[0])[:, None]
    return array[row_idx, col_idx]


def k_largest_columns(array: np.ndarray, k: int):
    """
    Find the k largest values across each column.
    """
    if array.shape[0] <= k:
        return array

    array = np.nan_to_num(array, copy=True, nan=-np.inf)

    row_idx = array.argpartition(-k, axis=0)[-k:, :]
    col_idx = np.arange(array.shape[1])[None, :]
    return array[row_idx, col_idx]


def k_neighbors_distance(query_array: npt.NDArray, ref_array: npt.NDArray = None, metric: str = 'jaccard', n_jobs: int = 1,
                         nearest_k: Optional[List[int]] = None, furthest_k: Optional[List[int]] = None) -> pl.DataFrame:
    """
    Calculate distance statistics between points in query and reference arrays. For each entry in query array,
    calculates the minimum, mean, and maximum distance to all entries in reference array. If it is not passed,
    a self-comparison is made (where distances to self are masked and not included).

    Additionally, computes the average distance to the k nearest and k furthest neighbors,
    where k values are specified in nearest_k and furthest_k parameters.

    Parameters
    ----------
    query_array : npt.NDArray
        Query array, typically the test set.
    ref_array : npt.NDArray, optional
        Reference array, typically the training set. If not passed, self-comparison.
    metric : str, optional
        Distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is 1.
    nearest_k : Optional[List[int]] or int, optional
        List of k values for which to calculate average distance to the k nearest neighbors.
        Can be a single integer or a list of integers. Default is None, which is equivalent to [1].
    furthest_k : Optional[List[int]] or int, optional
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
    self_comparison = False

    if not isinstance(query_array, np.ndarray):
        raise TypeError(f'Expected query array to be np.ndarray, got {type(query_array)} instead.')

    # Self-comparison
    if ref_array is not None:
        if not isinstance(ref_array, np.ndarray):
            raise TypeError(f'Expected reference array to be np.ndarray, got {type(ref_array)} instead.')
    else:
        ref_array = query_array # maybe deepcopy just to be safe?
        self_comparison = True
        max_n_jobs = max(query_array.shape[0] // 2, 1)
        if n_jobs > max_n_jobs:
            print(f"Query array too small for < {n_jobs} > jobs. {max_n_jobs} jobs will be used.")
            n_jobs = max_n_jobs

    if metric not in ['braycurtis', 'canberra', 'chebychev', 'cityblock', 'correlation', 'cosine', 'dice',
                      'euclidean', 'hamming', 'minkowski', 'pnorm', 'jaccard', 'jensenshannon', 'kulczynski1',
                      'mahalanobis', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                      'sqeuclidean', 'sqeuclid', 'yule']:
        raise ValueError(f'Metric {metric} is not supported.')

    if not isinstance(n_jobs, int):
        raise TypeError(f'Expected n_jobs to be int, got {type(n_jobs)} instead.')

    if n_jobs < 1:
        raise ValueError(f'Expected n_jobs to be at least 1, got {n_jobs} instead.')

    if not (isinstance(nearest_k, (int, list)) or nearest_k is None):
        raise TypeError(f'Expected nearest_k to be int or list, got {type(nearest_k)} instead.')
    if not (isinstance(furthest_k, (int, list)) or furthest_k is None):
        raise TypeError(f'Expected furthest_k to be int or list, got {type(furthest_k)} instead.')

    # QoL
    if isinstance(nearest_k, int):
        nearest_k = [nearest_k]
    if isinstance(furthest_k, int):
        furthest_k = [furthest_k]

    # Only find the distance to the most and least similar entry; make sure the nearest/furthest entry is always there
    if nearest_k is None:
        nearest_k = [1]
    else:
        nearest_k = sorted(set(nearest_k + [1]))

    if furthest_k is None:
        furthest_k = [1]
    else:
        furthest_k = sorted(set(furthest_k + [1]))

    # We only need to keep track of these entries for final calculations
    max_n_k = np.max(nearest_k)
    max_f_k = np.max(furthest_k)

    splits = np.array_split(ref_array, n_jobs)

    # Only relevant when one array is passed
    k_indices = {0: 0}
    for k, size in enumerate([split.shape[0] for split in splits][:-1], 1):
        k_indices[k] = k_indices[k-1] + size

    def neighbor_chunk(query_array: npt.NDArray, sub_ref_array: npt.NDArray , metric: str, idx: int, k_indices: Dict,
                       self_comparison: bool, max_n_k: int = 1, max_f_k: int = 1):

        chunk_distance = cdist(XA=sub_ref_array, XB=query_array, metric=metric)

        N = chunk_distance.shape[0]
        M = chunk_distance.shape[1]
        k = k_indices.get(idx)

        if self_comparison:
            chunk_distance[np.eye(N=N, M=M, k=k, dtype=bool)] = np.nan

        chunk_means = np.nanmean(chunk_distance, axis=0)
        weights = np.full_like(chunk_means, N)

        if self_comparison:
            weights[k:k+N] = N - 1

        # Find the k nearest/farthest neighbors for given entry
        nearest_distance = k_smallest_columns(chunk_distance, k=max_n_k)
        furthest_distance = k_largest_columns(chunk_distance, k=max_f_k)

        return idx, chunk_means, weights, nearest_distance, furthest_distance

    chunks = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(neighbor_chunk)(
            query_array=query_array,
            sub_ref_array=sub_ref_array,
            metric=metric,
            idx=idx,
            k_indices=k_indices,
            self_comparison=self_comparison,
            max_n_k=max_n_k,
            max_f_k=max_f_k
        )
        for idx, sub_ref_array in enumerate(splits)
    )

    # A single chunk, no need to concatenate anything
    if n_jobs == 1:
        mean_distances = chunks[0][1]
        nearest_array = chunks[0][3]
        furthest_array = chunks[0][4]

    else:
        # Sort by index to ensure the same assignment
        sorted_chunks = sorted(chunks, key=lambda x: x[0])

        # Take weighted average of chunk-means to get the global mean
        means_array = np.vstack([chunk[1] for chunk in sorted_chunks])
        weights = np.vstack([chunk[2] for chunk in sorted_chunks])
        mean_distances = np.average(means_array, axis=0, weights=weights)

        # Concatenate the results from each chunk
        nearest_array = np.vstack([chunk[3] for chunk in sorted_chunks])
        furthest_array = np.vstack([chunk[4] for chunk in sorted_chunks])

    # Aggregate all the results, find the averaged k-nearest and k-furthest values
    df = (pl.DataFrame({'Mean': mean_distances})
            .with_columns([
                pl.Series(f'{k} Nearest', k_smallest_columns(nearest_array, k=k).mean(axis=0))
                    for k in nearest_k
            ])
            .with_columns([
                pl.Series(f'{k} Furthest', k_largest_columns(furthest_array, k=k).mean(axis=0))
                    for k in furthest_k
            ])
            .rename({'1 Nearest': 'Min', '1 Furthest': 'Max'})
    )

    return df


def group_k_neighbors_distance(df: pl.DataFrame, features_col: str, group_col: str, metric: str = 'jaccard', n_jobs: int = 1,
                               nearest_k: Optional[List[int]] = None, furthest_k: Optional[List[int]] = None):
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

    req_cols = [features_col, group_col]
    if missing_cols := [col for col in req_cols if col not in df.columns]:
        raise KeyError(f"Missing required columns {missing_cols}")

    # Assume each entry belongs to a single group
    if isinstance(df[group_col].dtype, (pl.Int32, pl.Int64, pl.String)):
        unique_groups = set(df[group_col].to_list())
        print(f"Group column treated a Single-assignment")
        group_type = 'Single'

    # Assume each entry *might* belong to multiple groups
    elif isinstance(df[group_col].dtype, pl.List):
        unique_groups = set(chain.from_iterable(df[group_col].to_list()))
        print(f"Group column treated a Multiple-assignment")
        group_type = 'Multiple'

    else:
        raise TypeError(f"Expected the dtype of group_col to be either Int32, Int64, or String for single group"
                        f"assignment, and pl.List for multi-group assignments. Got {df[group_col].dtype} "
                        f"instead")

    results = []

    for group in unique_groups:
        if group_type == 'Single':
            query_array = np.vstack(df.filter(pl.col(group_col) == group)[features_col].to_numpy())
            ref_array = np.vstack(df.filter(pl.col(group_col) != group)[features_col].to_numpy())
        else:
            query_array = np.vstack(df.filter(pl.col(group_col).list.contains(group))[features_col].to_numpy())
            ref_array = np.vstack(df.filter(~pl.col(group_col).list.contains(group))[features_col].to_numpy())

        # if ref_array is None, calculates distance to self :)
        intra_knd = k_neighbors_distance(
            query_array=query_array,
            ref_array=None,
            metric=metric,
            n_jobs=n_jobs,
            nearest_k=nearest_k,
            furthest_k=furthest_k
        )

        intra_df = pl.DataFrame({
            "Scope": "Intra",
            "Group": group,
            "Aggregation": intra_knd.columns,
            "Values": [intra_knd[col] for col in intra_knd.columns]
        })

        inter_knd = k_neighbors_distance(
            query_array=query_array,
            ref_array=ref_array,
            metric=metric,
            n_jobs=n_jobs,
            nearest_k=nearest_k,
            furthest_k=furthest_k
        )

        inter_df = pl.DataFrame({
            "Scope": "Inter",
            "Group": group,
            "Aggregation": inter_knd.columns,
            "Values": [inter_knd[col] for col in inter_knd.columns]
        })

        results.append(pl.concat([intra_df, inter_df]))

    if not results:
        print("No results")
        return None

    return pl.concat(results)


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
