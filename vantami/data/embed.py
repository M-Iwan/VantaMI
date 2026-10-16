from abc import ABC, abstractmethod
from typing import Optional
import warnings

import numpy as np
import numpy.typing as npt
import polars as pl


class Embedder(ABC):
    """
    Abstract base class for feature embedding.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42,
                 kwargs: Optional[dict] = None):
        """
        Parameters
        ----------
        n_components: int
            Number of embedding dimensions to keep.
        random_state: int
            Random state for reproducibility.
        kwargs: dict or None
            Additional keyword arguments passed to the underlying embedder.
        """
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError('n_components must be a positive integer.')

        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = {} if kwargs is None else kwargs.copy()

        self.embedder = None
        self.n_features = {}
        self.fitted_ = False

    def __repr__(self):
        return (f"<{self.__class__.__name__}>: "
                f"n_components: {self.n_components}, "
                f"random_state: {self.random_state}")

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    @staticmethod
    def validate_features(x_array: npt.NDArray):
        """
        Check validity of passed feature values and convert to 2D numpy array if needed.

        Parameters
        ----------
        x_array: npt.NDArray
            Numpy array corresponding to molecular features. Must be one- or two-dimensional.
        """
        if not isinstance(x_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(x_array)} instead.")

        if x_array.ndim == 0 or x_array.ndim > 2:
            raise ValueError(f"Features must be one- or two-dimensional, got {x_array.ndim} dimensions instead.")

        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        return x_array

    def reset(self):
        """
        Reset embedder attributes to its initial state.
        """
        self.embedder = None
        self.n_features = {}
        self.fitted_ = False
        return self

    @abstractmethod
    def transform(self, x_array: npt.NDArray):
        """
        Transform an input feature array using the fitted embedder.
        """
        pass

    @abstractmethod
    def fit_transform(self, x_array: npt.NDArray):
        """
        Fit the embedder and transform the input feature array.
        """
        pass

    def transform_df(self, df: pl.DataFrame, features_col: str):
        """
        Transform features in a Polars DataFrame using the fitted embedder.

        Parameters
        ----------
        df: pl.DataFrame
            DataFrame containing array-valued features.
        features_col: str
            Name of the input feature column.

        Returns
        -------
        pl.DataFrame
            DataFrame with the embedded feature column.
        """
        if df.is_empty():
            raise ValueError(f"Cannot transform an empty DataFrame")

        x_array = np.vstack(df[features_col].to_numpy())

        x_array = self.validate_features(x_array)
        x_array = self.transform(x_array)

        df = df.with_columns(
            pl.Series(features_col, [array.reshape(-1) for array in np.vsplit(x_array, x_array.shape[0])])
        )
        return df

    def fit_transform_df(self, df: pl.DataFrame, features_col: str):
        """
        Fit the embedder and transform features in a Polars DataFrame.

        Parameters
        ----------
        df: pl.DataFrame
            DataFrame containing array-valued features.
        features_col: str
            Name of the input feature column.

        Returns
        -------
        pl.DataFrame
            DataFrame with the embedded feature column.
        """
        if df.is_empty():
            raise ValueError(f"Cannot transform an empty DataFrame")

        x_array = np.vstack(df[features_col].to_numpy())

        x_array = self.validate_features(x_array)
        x_array = self.fit_transform(x_array)

        df = df.with_columns(
            pl.Series(features_col, [array.reshape(-1) for array in np.vsplit(x_array, x_array.shape[0])])
        )

        return df


class PCAEmbedder(Embedder):
    """
    A class for embedding features using Principal Component Analysis.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42,
                 kwargs: Optional[dict] = None):
        """
        Parameters
        ----------
        n_components: int
            Number of principal components to keep.
        random_state: int
            Random state for reproducibility.
        kwargs: dict or None
            Additional keyword arguments passed to sklearn.decomposition.PCA.
        """
        super().__init__(
            n_components=n_components,
            random_state=random_state,
            kwargs=kwargs
        )

    def transform(self, x_array: npt.NDArray):
        """
        Transform an input feature array using the fitted PCA embedder.

        Parameters
        ----------
        x_array: npt.NDArray
            Input feature array.

        Returns
        -------
        npt.NDArray
            Embedded feature array.
        """
        if self.fitted_ is False:
            raise RuntimeError('Embedder has not been fit yet. Please call .fit_transform() first.')

        x_array = self.validate_features(x_array)

        if x_array.shape[1] != self.n_features['input']:
            raise ValueError(
                f"Expected {self.n_features['input']} features, got {x_array.shape[1]} instead."
            )

        return self.embedder.transform(x_array)

    def fit_transform(self, x_array: npt.NDArray):
        """
        Fit PCA and transform the input feature array.

        Parameters
        ----------
        x_array: npt.NDArray
            Input feature array.

        Returns
        -------
        npt.NDArray
            Embedded feature array.
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(f"Class < PCAEmbedder > requires < sklearn > library:\n{exc}")

        reserved_kwargs = {
            "n_components",
            "random_state"
        }

        if reserved_kwargs.intersection(self.kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        x_array = self.validate_features(x_array)

        max_components = min(x_array.shape[0], x_array.shape[1])
        if self.n_components > max_components:
            warnings.warn(
                f"Requested n_components={self.n_components}, but the maximum for input shape {x_array.shape} is"
                f"{max_components}. Using n_components={max_components}.",
                UserWarning,
                stacklevel=2
            )
            self.n_components = max_components

        self.embedder = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            **self.kwargs
        )

        x_array = self.embedder.fit_transform(x_array)

        self.n_features['input'] = self.embedder.n_features_in_
        self.n_features['output'] = x_array.shape[1]
        self.fitted_ = True

        return x_array


class UMAPEmbedder(Embedder):
    """
    A class for embedding features using Uniform Manifold Approximation and Projection.
    """

    def __init__(self, n_components: int = 64, distance_threshold: float = 0.1,
                 distance_metric: str = "euclidean", n_neighbors: int = 16, random_state: int = 42,
                 kwargs: Optional[dict] = None):
        """
        Parameters
        ----------
        n_components: int
            Number of embedding dimensions to keep.
        distance_threshold: float
            Minimum distance between points in the embedded space.
        distance_metric: str
            Name of the distance metric to use.
        n_neighbors: int
            Balancing parameter for local vs global structure.
        random_state: int
            Random state for reproducibility.
        kwargs: dict or None
            Additional keyword arguments passed to umap.UMAP.
        """
        super().__init__(
            n_components=n_components,
            random_state=random_state,
            kwargs=kwargs
        )

        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric
        self.n_neighbors = n_neighbors

    def __repr__(self):
        return (f"<{self.__class__.__name__}>: "
                f"n_components: {self.n_components}, "
                f"distance_threshold: {self.distance_threshold}, "
                f"distance_metric: {self.distance_metric}, "
                f"random_state: {self.random_state}, "
                f"n_neighbors: {self.n_neighbors}>")

    def transform(self, x_array: npt.NDArray):
        """
        Transform an input feature array using the fitted UMAP embedder.

        Parameters
        ----------
        x_array: npt.NDArray
            Input feature array.

        Returns
        -------
        npt.NDArray
            Embedded feature array.
        """
        if self.fitted_ is False:
            raise RuntimeError('Embedder has not been fit yet. Please call .fit_transform() first.')

        x_array = self.validate_features(x_array)

        if x_array.shape[1] != self.n_features['input']:
            raise ValueError(
                f"Expected {self.n_features['input']} features, got {x_array.shape[1]} instead."
            )

        return self.embedder.transform(x_array)

    def fit_transform(self, x_array: npt.NDArray):
        """
        Fit UMAP and transform the input feature array.

        Parameters
        ----------
        x_array: npt.NDArray
            Input feature array.

        Returns
        -------
        npt.NDArray
            Embedded feature array.
        """
        try:
            import umap
        except ImportError as exc:
            raise ImportError(f"Class < UMAPEmbedder > requires < umap > library:\n{exc}")

        reserved_kwargs = {
            "n_components",
            "min_dist",
            "metric",
            "random_state",
            "n_neighbors"
        }

        if reserved_kwargs.intersection(self.kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")

        default_kwargs = {}
        default_kwargs.update(self.kwargs)

        x_array = self.validate_features(x_array)

        max_components = min(x_array.shape[0], x_array.shape[1])
        if self.n_components > max_components:
            warnings.warn(
                f"Requested n_components={self.n_components}, but the maximum for input shape {x_array.shape} is"
                f"{max_components}. Using n_components={max_components}.",
                UserWarning,
                stacklevel=2
            )
            self.n_components = max_components

        self.embedder = umap.UMAP(
            n_components=self.n_components,
            metric=self.distance_metric,
            min_dist=self.distance_threshold,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            **default_kwargs
        )

        x_array = self.embedder.fit_transform(x_array)

        self.n_features['input'] = self.embedder.n_features_in_
        self.n_features['output'] = x_array.shape[1]
        self.fitted_ = True

        return x_array


def get_embedder_params(embedder):
    """
    Get default embedder parameters for a feature type.

    Parameters
    ----------
    embedder: type
        Embedder class.

    Returns
    -------
    dict
        Default embedder parameters.
    """
    pca_defaults = {
        "n_components": 128,
        "random_state": 42,
    }

    umap_defaults = {
        "n_components": 128,
        "n_neighbors": 16,
        "distance_threshold": 0.1,
        "distance_metric": "cosine",
        "random_state": 42,
    }

    defaults = {
        PCAEmbedder: pca_defaults,
        UMAPEmbedder: umap_defaults,
    }

    if embedder not in defaults:
        raise TypeError(f"Embedder must be PCAEmbedder or UMAPEmbedder.")

    return defaults[embedder].copy()


def array_2_pca(array: npt.NDArray, n_components: int = 64, random_state: int = 42,
                as_list: bool = False, kwargs: dict = None):
    """
    Transform the passed array using Principal Component Analysis.

    Parameters
    ----------
    array: npt.NDArray
        2D numpy array
    n_components: int
        Number of principal components to keep
    random_state: int
        Random state for reproducibility
    as_list: bool
        A flag to return the array as a list of 1D numpy arrays
    kwargs: dict
        Other parameters passed to the PCA object.

    Returns
    -------
    (array: Union[npt.NDarray, List[npt.NDArray]], PCA)
    """

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(f"Function < array_2_pca > requires < sklearn > library:\n{exc}")

    reserved_kwargs = {
        "n_components",
        "random_state"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")
    else:
        kwargs = {}

    pca = PCA(n_components=n_components, random_state=random_state, **kwargs)
    emb = pca.fit_transform(array)

    if as_list:
        return [array.reshape(-1) for array in np.vsplit(emb, emb.shape[0])], pca
    return emb, pca


def dataframe_2_pca(df: pl.DataFrame, tdf: pl.DataFrame = None, features_col: str = "RDKit", output_col: str = None,
                    n_components: int = 64, random_state: int = 42, kwargs: dict = None):
    """
    Transform features in a DataFrame using Principal Component Analysis.
    If a second DataFrame is provided, its features are embedded using model fit on the first one.

    Parameters
    ----------
    df: pl.DataFrame
        A Polars DataFrame with features to transform. Used for fitting.
    tdf: pl.DataFrame
        Another polars DataFrame. Not used for fitting.
    features_col: str
        The name of the column with features
    output_col: str
        The name for the column with transformed features
    n_components: int
        Number of principal components to keep
    random_state: int
        Random state for reproducibility
    kwargs: dict
        Other parameters passed to the PCA object.

    Returns
    -------
    (PCA, pl.DataFrame, Optional[pl.DataFrame])
    """

    if output_col is None:
        output_col = features_col + "_PCA"

    array = np.vstack(df[features_col].to_numpy())
    emb, pca = array_2_pca(array, n_components=n_components, as_list=True, random_state=random_state, kwargs=kwargs)

    df = df.with_columns(
        pl.Series(output_col, emb)
    )

    if tdf is not None:
        tdf_emb = pca.transform(np.vstack(tdf[features_col].to_numpy()))
        tdf = tdf.with_columns(
            pl.Series(output_col, [array.reshape(-1) for array in np.vsplit(tdf_emb, tdf_emb.shape[0])])
        )

    return pca, df, tdf


def array_2_tsne(array: npt.NDArray, n_components: int = 2, perplexity: int = 30, random_state: int = 42,
                 as_list: bool = False, kwargs: dict = None):
    """
    Transform the passed array using t-Stochastic Neighbor Embedding.

    Parameters
    ----------
    array: npt.NDArray
        2D numpy array
    n_components: int
        Number of dimensions to keep
    perplexity: int
        Number of neighbors (more or less) considered by the manifold algorithm
    random_state: int
        Random state for reproducibility
    as_list: bool
        A flag to return the array as a list of 1D numpy arrays
    kwargs: dict
        Other parameters passed to the TSNE object.

    Returns
    -------
    (array: Union[npt.NDarray, List[npt.NDArray]], TSNE)
    """

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(f"Function < array_2_tsne > requires < sklearn > library:\n{exc}")

    reserved_kwargs = {
        "n_components",
        "perplexity",
        "random_state",
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")
    else:
        kwargs = {}

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, **kwargs)
    emb = tsne.fit_transform(array)

    if as_list:
        return [array.reshape(-1) for array in np.vsplit(emb, emb.shape[0])], tsne
    return emb, tsne


def dataframe_2_tsne(df: pl.DataFrame, features_col: str = "RDKit", output_col: str = None,
                     n_components: int = 2, perplexity: int = 30, random_state: int = 42, kwargs: dict = None):
    """
    Transform features in a DataFrame using t-Stochastic Neighbor Embedding.

    Parameters
    ----------
    df: pl.DataFrame
        A Polars DataFrame with features to transform. Used for fitting.
    features_col: str
        The name of the column with features
    output_col: str
        The name for the column with transformed features
    n_components: int
        Number of dimensions to keep
    perplexity: int
        Number of neighbors (more or less) considered by the manifold algorithm
    random_state: int
        Random state for reproducibility
    kwargs: dict
        Other parameters passed to the TSNE object.

    Returns
    -------
    (TSNE, pl.DataFrame, Optional[pl.DataFrame])
    """

    if output_col is None:
        output_col = features_col + "_TSNE"

    array = np.vstack(df[features_col].to_numpy())

    emb, tsne = array_2_tsne(array, n_components=n_components, perplexity=perplexity, as_list=True,
                             random_state=random_state, kwargs=kwargs)

    df = df.with_columns(
        pl.Series(output_col, emb)
    )

    return tsne, df


def array_2_umap(array: npt.NDArray, n_components: int, distance_threshold: float = 0.1,
                 distance_metric: str = "euclidean", random_state: int = 42, as_list: bool = False, kwargs: dict = None):
    """
    Transform the passed array using Uniform Manifold Approximation and Projection (UMAP).

    Parameters
    ----------
    array: npt.NDArray
        2D numpy array
    n_components: int
        Number of dimensions to keep
    distance_threshold: float
        Minimum distance between points in the embedded space
    distance_metric: str
        Name of the distance metric to use
    random_state: int
        Random state for reproducibility
    as_list: bool
        A flag to return the array as a list of 1D numpy arrays
    kwargs: dict
        Other parameters passed to the UMAP object.

    Returns
    -------
    (array: Union[npt.NDarray, List[npt.NDArray]], UMAP)
    """

    try:
        import umap
    except ImportError as exc:
        raise ImportError(f"Function < array_2_umap > requires < umap > library:\n{exc}")

    default_kwargs = {
        "n_neighbors": 16,
    }

    reserved_kwargs = {
        "n_components",
        "min_dist",
        "metric",
        "random_state"
    }

    if kwargs is not None:
        if reserved_kwargs.intersection(kwargs):
            raise ValueError(f"kwargs must not contain managed arguments: {reserved_kwargs}")
        default_kwargs.update(kwargs)

    ump = umap.UMAP(
        n_components=n_components,
        metric=distance_metric,
        min_dist=distance_threshold,
        random_state=random_state,
        **default_kwargs
    )
    emb = ump.fit_transform(array)

    if as_list:
        return [array.reshape(-1) for array in np.vsplit(emb, emb.shape[0])], ump
    return emb, ump


def dataframe_2_umap(df: pl.DataFrame, tdf: pl.DataFrame = None, features_col: str = "RDKit", output_col: str = None,
                     n_components: int = 64, distance_threshold: float = 0.1, distance_metric: str = "euclidean",
                     random_state: int = 42, kwargs: dict = None):
    """
    Transform features in a DataFrame using Uniform Manifold Approximation and Projection (UMAP).
    If a second DataFrame is provided, its features are embedded using model fit on the first one.

    Parameters
    ----------
    df: pl.DataFrame
        A Polars DataFrame with features to transform. Used for fitting.
    tdf: pl.DataFrame
        Another polars DataFrame. Not used for fitting.
    features_col: str
        The name of the column with features
    output_col: str
        The name for the column with transformed features
    n_components: int
        Number of dimensions to keep
    distance_threshold: float
        Minimum distance between points in the embedded space
    distance_metric: str
        Name of the distance metric to use
    random_state: int
        Random state for reproducibility
    kwargs: dict
        Other parameters passed to the UMAP object.

    Returns
    -------
    (UMAP, pl.DataFrame, Optional[pl.DataFrame])
    """

    if output_col is None:
        output_col = features_col + "_UMAP"

    array = np.vstack(df[features_col].to_numpy())
    emb, ump = array_2_umap(
        array, n_components=n_components, distance_threshold=distance_threshold, distance_metric=distance_metric,
        as_list=True, random_state=random_state, kwargs=kwargs)

    df = df.with_columns(
        pl.Series(output_col, emb)
    )

    if tdf is not None:
        tdf_emb = ump.transform(np.vstack(tdf[features_col].to_numpy()))
        tdf = tdf.with_columns(
            pl.Series(output_col, [array.reshape(-1) for array in np.vsplit(tdf_emb, tdf_emb.shape[0])])
        )

    return ump, df, tdf