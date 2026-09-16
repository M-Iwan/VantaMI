import numpy as np
import polars as pl

from vantami.data.distance import distance_matrix


def _to_arrays(df: pl.DataFrame, features_col: str):
    """
    Convert polars Series to list of numpy arrays
    """
    if features_col not in df.columns:
        raise ValueError(f"Features column {features_col} was not found.")

    features = df[features_col].to_numpy()

    if any(array is None for array in features):
        raise ValueError(f"Features column {features_col} contains null values.")

    return [array.reshape(-1) for array in features]


def butina_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = "jaccard",
                   n_jobs: int = 1, batch_size: int = 1024, threshold: float = 0.3):
    """
    Perform Butina clustering from a thresholded pairwise neighborhood graph.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    features_col : str, optional
        Name of the column containing features. Default is ECFP.
    metric : str, optional
        Metric passed to distance_matrix. Default is "jaccard"
    n_jobs : int, optional
        Number of parallel jobs used for distance calculation.
    batch_size : int, optional
        Number of rows assigned to one distance-calculation batch. Default is 1024.
    threshold : float, optional
        Distance threshold for defining neighbors. Two entries are neighbors
        when their distance is less than or equal to this value. Default is 0.3.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an added integer "Cluster" column.
    """

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    neighbor_mask = distance_matrix(
        array_1=_to_arrays(df, features_col),
        array_2=None,
        metric=metric,
        n_jobs=n_jobs,
        batch_size=batch_size,
        return_triu=False,
        threshold=threshold,
    ).astype(bool, copy=False)

    seen = np.zeros(n_samples, dtype=bool)
    cluster_ids = np.full(n_samples, -1, dtype=np.int64)

    neighbor_counts = neighbor_mask.sum(axis=1, dtype=np.int64)

    current_cluster_id = 0

    while not np.all(seen):
        unassigned_indices = np.flatnonzero(~seen)

        candidate_counts = neighbor_counts.copy()
        candidate_counts[seen] = -1

        most_neighbors_idx = int(np.argmax(candidate_counts))

        if candidate_counts[most_neighbors_idx] == 0:
            cluster_ids[unassigned_indices] = np.arange(
                current_cluster_id,
                current_cluster_id + len(unassigned_indices),
                dtype=np.int64,
            )
            break

        cluster_members = np.flatnonzero(
            neighbor_mask[most_neighbors_idx] & ~seen
        )
        cluster_members = np.unique(np.append(cluster_members, most_neighbors_idx))

        cluster_ids[cluster_members] = current_cluster_id
        seen[cluster_members] = True

        removed_edge_counts = neighbor_mask[:, cluster_members].sum(
            axis=1,
            dtype=np.int64,
        )

        neighbor_counts -= removed_edge_counts
        neighbor_counts[seen] = -1

        current_cluster_id += 1

    return df.with_columns(pl.Series("Cluster", cluster_ids))


def murcko_cluster(df: pl.DataFrame, smiles_col: str = 'SMILES', generic: bool = False):
    """
    Cluster molecules based on their (generic) Murcko Scaffolds. Molecules for which a scaffold cannot be generated
    are assigned to a single cluster.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    smiles_col: str
        A name of the column with SMILES strings.
    generic: bool
        A flag to make Murcko scaffolds generic, i.e. only carbon atoms and single bonds

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from vantami.data.descriptors import dataframe_2_murcko
    except ImportError as exc:
        raise ImportError(f"Function < murcko_cluster > requires RDKit:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    df = dataframe_2_murcko(
        df=df,
        smiles_col=smiles_col,
        output_col="Murcko",
        generic=generic,
        n_jobs=1)

    m_df = df.select("Murcko").drop_nulls().unique().sort("Murcko").with_row_index(name="Cluster")
    df = df.join(m_df, on="Murcko", how="left")
    df = df.with_columns(pl.col("Cluster").fill_null(-1).cast(pl.Int64))

    return df


def cc_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard',
               n_jobs: int = 1, batch_size: int = 1024, threshold: float = 0.3):
    """
    Cluster molecules using connected components graphs.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int
        Number of cores to use. Default is 1.
    batch_size: int
        Number of rows to process within a batch. Default is 1024.
    threshold: float
        Distance threshold. Edge is set if distance <= threshold. Default is 0.3

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError(f"Function < cc_cluster > requires scipy:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    adj_sparse = distance_matrix(
        array_1=_to_arrays(df, features_col),
        array_2=None,
        metric=metric,
        n_jobs=n_jobs,
        batch_size=batch_size,
        return_triu=False,
        return_sparse=True,
        threshold=threshold
    )

    adj_sparse.data = np.ones_like(adj_sparse.data, dtype=np.uint8)

    _, labels = connected_components(adj_sparse, directed=False, return_labels=True)

    return df.with_columns(pl.Series('Cluster', labels))


def dbscan_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard', n_jobs: int = 1,
                   batch_size: int = 1024, threshold: float = 0.3, kwargs: dict = None):
    """
    Cluster molecules using the DBSCAN approach.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int
        Number of cores to use. Default is 1.
    batch_size: int
        Number of rows to process within a batch. Default is 1024.
    threshold: float
        Distance threshold. Edge is set if distance <= threshold. Default is 0.3
    kwargs: dict
        Additional keyword arguments passed to DBSCAN.

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import sort_graph_by_row_values
    except ImportError as exc:
        raise ImportError(f"Function < dbscan_cluster > requires sklearn:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    default_kwargs = {
        "min_samples": 5
    }

    reserved_kwargs = {
        "eps",
        "metric"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs.update(kwargs)

    distances = distance_matrix(
        array_1=_to_arrays(df, features_col),
        array_2=None,
        metric=metric,
        n_jobs=n_jobs,
        batch_size=batch_size,
        return_triu=False,
        return_sparse=True,
        threshold=threshold
    )

    zero_distances = distances.data == 0.0

    if np.any(zero_distances):
        distances.data[zero_distances] = np.nextafter(0.0, 1.0)

    distances.setdiag(np.nextafter(0.0, 1.0))
    distances.sort_indices()

    distances = sort_graph_by_row_values(
        distances,
        copy=False,
        warn_when_not_sorted=False
    )

    model = DBSCAN(
        eps=threshold,
        metric="precomputed",
        **default_kwargs
    )

    labels = model.fit_predict(distances)

    return df.with_columns(pl.Series('Cluster', labels))


def hdbscan_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard', n_jobs: int = 1,
                    batch_size: int = 1024, threshold: float = 0.3, kwargs: dict = None):
    """
    Cluster molecules using the HDBSCAN approach.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int
        Number of cores to use. Default is 1.
    batch_size: int
        Number of rows to process within a batch. Default is 1024.
    threshold: float
        Distance threshold. Edge is set if distance <= threshold. Default is 0.3
    kwargs: dict
        Additional keyword arguments passed to HDBSCAN

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise ImportError(f"Function < hdbscan_cluster > requires sklearn:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    default_kwargs = {
        "min_cluster_size": 5,
        "alpha": 1.0,
        "cluster_selection_method": "eom",
        "copy": False
    }

    reserved_kwargs = {
        "cluster_selection_epsilon",
        "metric"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs.update(kwargs)

    distances = distance_matrix(
        array_1=_to_arrays(df, features_col),
        array_2=None,
        metric=metric,
        n_jobs=n_jobs,
        batch_size=batch_size,
        return_triu=False,
    )

    model = HDBSCAN(
        cluster_selection_epsilon=threshold,
        metric="precomputed",
        **default_kwargs
    )

    labels = model.fit_predict(distances)

    return df.with_columns(pl.Series('Cluster', labels))


def agglomerative_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard', n_jobs: int = 1,
                          batch_size: int = 1024, threshold: float = 0.3, kwargs: dict = None):
    """
    Cluster molecules using the agglomerative clustering approach.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int
        Number of cores to use. Default is 1.
    batch_size: int
        Number of rows to process within a batch. Default is 1024.
    threshold: float
        Distance threshold. Edge is set if distance <= threshold. Default is 0.3
    kwargs: dict
        Additional keyword arguments passed to AgglomerativeClustering

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError(f"Function < agglomerative_cluster > requires sklearn:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    default_kwargs = {
        "linkage": "average"
    }

    reserved_kwargs = {
        "n_clusters",
        "distance_threshold",
        "metric"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs.update(kwargs)

    distances = distance_matrix(
        array_1=_to_arrays(df, features_col),
        array_2=None,
        metric=metric,
        n_jobs=n_jobs,
        batch_size=batch_size,
        return_triu=False,
    )

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        **default_kwargs
    )

    labels = model.fit_predict(distances)

    return df.with_columns(pl.Series('Cluster', labels))


def spectral_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard', n_jobs: int = 1,
                     batch_size: int = 1024, threshold: float = 0.3, n_clusters: int = 2, kwargs: dict = None):
    """
    Cluster molecules using the spectral clustering approach.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int
        Number of cores to use. Default is 1.
    batch_size: int
        Number of rows to process within a batch. Default is 1024.
    threshold: float
        Distance threshold. Edge is set if distance <= threshold. Default is 0.3
    n_clusters: int
        Number of clusters to assign. Default is 2
    kwargs: dict
        Additional keyword arguments passed to SpectralClustering.

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from sklearn.cluster import SpectralClustering
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError(f"Function < spectral_cluster > requires sklearn and scipy:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    default_kwargs = {
        "assign_labels": "kmeans"
    }

    reserved_kwargs = {
        "affinity",
        "n_clusters",
        "n_jobs"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs.update(kwargs)

    affinity = distance_matrix(
        array_1=_to_arrays(df, features_col),
        array_2=None,
        metric=metric,
        n_jobs=n_jobs,
        batch_size=batch_size,
        return_triu=False,
        threshold=threshold,
        return_sparse=True
    )

    affinity.data = 1.0 - affinity.data

    affinity.setdiag(1.0)
    affinity.eliminate_zeros()

    # current sklearn expectations
    affinity = csr_matrix(
        (
            affinity.data,
            affinity.indices.astype(np.int32, copy=False),
            affinity.indptr.astype(np.int32, copy=False),
        ),
        shape=affinity.shape
    )

    model = SpectralClustering(
        affinity="precomputed",
        n_jobs=n_jobs,
        n_clusters=n_clusters,
        **default_kwargs
    )

    labels = model.fit_predict(affinity)

    return df.with_columns(pl.Series('Cluster', labels))


def optics_cluster(df: pl.DataFrame, features_col: str = "ECFP", metric: str = 'jaccard', n_jobs: int = 1,
                   threshold: float = 0.3, cluster_method: str = "dbscan",
                   cluster_threshold: float = None, kwargs: dict = None):
    """
    Cluster molecules using the OPTICS approach.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is ECFP
    metric: str, optional
        The distance metric to use. Default is 'jaccard'.
        See scipy.spatial.distance.cdist for a list of supported metrics.
    n_jobs: int
        Number of cores to use. Default is 1.
    threshold: float
        Distance threshold. Edge is set if distance <= threshold. Default is 0.3
    cluster_method: str
        Clustering method to use. Default is 'dbscan'.
    cluster_threshold: float
        Clustering threshold to use. If cluster_method == 'dbscan' corresponds to the eps argument.
        If cluster_method == 'xi', it corresponds to the xi argument.
    kwargs: dict
        Additional keyword arguments passed to OPTICS.

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from sklearn.cluster import OPTICS
    except ImportError as exc:
        raise ImportError(f"Function < optics_cluster > requires sklearn:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    default_kwargs = {
        "min_samples": 5,
    }

    reserved_kwargs = {
        "cluster_method",
        "eps",
        "xi",
        "metric",
        "max_eps",
        "n_jobs"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs.update(kwargs)

    if cluster_method == "dbscan":
        if cluster_threshold is not None:
            default_kwargs["eps"] = cluster_threshold
    elif cluster_method == "xi":
        if cluster_threshold is not None:
            default_kwargs["xi"] = cluster_threshold
    else:
        raise ValueError("cluster_method must be one of: 'dbscan', 'xi'")

    array = np.vstack(_to_arrays(df, features_col))

    model = OPTICS(
        max_eps=threshold,
        metric=metric,
        cluster_method=cluster_method,
        n_jobs=n_jobs,
        **default_kwargs
    )

    labels = model.fit_predict(array)

    return df.with_columns(pl.Series('Cluster', labels))


def kmeans_cluster(df: pl.DataFrame, features_col: str = "RDKit", n_clusters: int = 3, kwargs: dict = None):
    """
    Cluster molecules using the KMeans approach.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame
    features_col: str
        Name of the column with features. Default is RDKit
    n_clusters: int
        Number of clusters to generate.
    kwargs: dict
        Additional keyword arguments passed to KMeans.

    Returns
    -------
    df: pl.DataFrame
    """

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(f"Function < kmeans_cluster > requires scikit-learn.:\n{exc}")

    if (n_samples := len(df)) == 0:
        raise ValueError(f"Input DataFrame is empty.")
    if n_samples == 1:
        return df.with_columns(pl.lit(0).alias("Cluster"))

    default_kwargs = {
        "init": "k-means++",
        "n_init": "auto",
        "max_iter": 300,
        "random_state": 42,
    }

    reserved_kwargs = {
        "n_clusters"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs.update(kwargs)

    array = np.vstack(_to_arrays(df, features_col))

    model = KMeans(
        n_clusters=n_clusters,
        **default_kwargs,
    )

    labels = model.fit_predict(array)

    return df.with_columns(pl.Series("Cluster", labels))
