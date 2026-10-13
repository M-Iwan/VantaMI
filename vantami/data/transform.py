import re
from math import floor
from typing import List

import numpy as np
import numpy.typing as npt
import polars as pl

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class SimpleClipper:
    def __init__(self, n_std: int = 6):
        """
        Clips values based on their mean and standard deviation.

        Parameters
        ----------
        n_std: int
            Number of standard deviations within which the data remains unclipped.
        """
        self.n_std = n_std
        self.stds = None
        self.means = None
        self.fitted_ = False

    def fit(self, x_array: npt.NDArray):
        self.stds = np.nanstd(x_array, axis=0)
        self.means = np.nanmean(x_array, axis=0)
        self.a_min = self.means - self.n_std * self.stds
        self.a_max = self.means + self.n_std * self.stds
        self.fitted_ = True
        return self

    def transform(self, x_array: npt.NDArray):
        if self.fitted_ is False:
            raise RuntimeError('Clipper has not been fit yet. Please call .fit() first.')

        return np.clip(a=x_array, a_min=self.a_min, a_max=self.a_max)

    def fit_transform(self, x_array: npt.NDArray):
        self.fit(x_array)
        return self.transform(x_array)


class MADClipper:
    def __init__(self, n_mad: int = 6):
        """
        Clips values based on their median and median absolute deviation.

        Parameters
        ----------
        n_mad: int
            Number of median absolute deviations within which the data remains unclipped.
        """
        self.n_mad = n_mad
        self.mads = None
        self.medians = None
        self.fitted_ = False

    def fit(self, x_array: npt.NDArray):
        self.medians = np.nanmedian(x_array, axis=0)
        self.mads = np.nanmedian(np.abs(x_array - self.medians), axis=0)
        self.a_min = self.medians - self.n_mad * self.mads
        self.a_max = self.medians + self.n_mad * self.mads
        self.fitted_ = True
        return self

    def transform(self, x_array: npt.NDArray):
        if self.fitted_ is False:
            raise RuntimeError('Clipper has not been fit yet. Please call .fit() first.')

        return np.clip(a=x_array, a_min=self.a_min, a_max=self.a_max)

    def fit_transform(self, x_array: npt.NDArray):
        self.fit(x_array)
        return self.transform(x_array)


class QuantileClipper:
    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995):
        """
        Clips values based on lower and upper quantiles.

        Parameters
        ----------
        lower_quantile: float
            Lower quantile used as the clipping minimum.
        upper_quantile: float
            Upper quantile used as the clipping maximum.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.fitted_ = False

    def fit(self, x_array: npt.NDArray):
        self.a_min = np.nanquantile(x_array, self.lower_quantile, axis=0)
        self.a_max = np.nanquantile(x_array, self.upper_quantile, axis=0)
        self.fitted_ = True
        return self

    def transform(self, x_array: npt.NDArray):
        if self.fitted_ is False:
            raise RuntimeError('Clipper has not been fit yet. Please call .fit() first.')

        return np.clip(a=x_array, a_min=self.a_min, a_max=self.a_max)

    def fit_transform(self, x_array: npt.NDArray):
        self.fit(x_array)
        return self.transform(x_array)


class DataTransformer:
    def __init__(self, use_masks: bool = True, use_clipper: bool = True, use_imputer: bool = True,
                 use_selector: bool = True, use_corr: bool = False, use_scaler: bool = True):
        """
        A class for preprocessing and transforming feature data prior to machine learning model training.

        Handles several data preprocessing steps including:
            - Removing features with missing values, infinities, zero variance or very large values
            - Clipping large features
            - Imputing missing values
            - Removing features with near-zero variance
            - Removing highly correlated features
            - Scaling features using robust scaling

        Parameters
        ----------
        use_masks: bool, default=True
            Whether to remove missing, infinite, very large or zero-variance features.
        use_clipper: bool, default=True
            Whether to clip features based on their standard deviation.
        use_imputer: bool, default=True
            Whether to impute missing values.
        use_selector: bool, default=True
            Whether to remove features with near-zero variance.
        use_corr: bool, default=False
            Whether to perform correlation analysis and remove highly correlated features.
        use_scaler: bool, default=True
            Whether to scale features using RobustScaler.
        """
        self.masks = {}
        self.clipper = None
        self.imputer = None
        self.selector = None
        self.scaler = None
        self.n_features = {}
        self.use_masks = use_masks
        self.use_clipper = use_clipper
        self.use_imputer = use_imputer
        self.use_selector = use_selector
        self.use_corr = use_corr  # goes with masks, no separate attribute
        self.use_scaler = use_scaler
        self.fitted_ = False
        self.masks_fit = False
        self.clipper_fit = False
        self.imputer_fit = False
        self.selector_fit = False
        self.corr_fit = False
        self.scaler_fit = False
        self.frac_missing_ = 0.05
        self.frac_large_ = 0.05
        self.large_threshold_ = 1e12
        self.random_state_ = 42

    def __repr__(self):
        return (f"<{self.__class__.__name__}>: use_masks: {self.use_masks}, use_clipper: {self.use_clipper}, "
                f"use_imputer: {self.use_imputer}, use_selector: {self.use_selector}, "
                f"use_corr: {self.use_corr}, use_scaler: {self.use_scaler}>")

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    @staticmethod
    def validate_features(x_array: npt.NDArray):
        """
        Check validity of passed feature values and convert to 2D numpy array if needed.
        """
        if not isinstance(x_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(x_array)} instead.")

        if x_array.ndim == 0 or x_array.ndim > 2:
            raise ValueError(f"Features must be one- or two-dimensional, got {x_array.ndim} dimensions instead.")

        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        return x_array

    @staticmethod
    def validate_targets(y_array: npt.NDArray):
        """
        Check validity of passed target values and convert to 1D numpy array if needed.
        """
        if not isinstance(y_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(y_array)} instead.")

        if (y_array.ndim == 2 and y_array.shape[1] > 1) or y_array.ndim > 2:
            raise ValueError(f"Targets must be convertible to 1D array, got {y_array.ndim} dimensions instead.")

        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.reshape(-1)

        return y_array

    def reset(self):
        """
        Reset the fitted objects while retaining the set  parameters. Returns self.
        """
        self.masks = {}
        self.clipper = None
        self.imputer = None
        self.selector = None
        self.scaler = None
        self.n_features = {}
        self.fitted_ = False
        self.masks_fit = False
        self.clipper_fit = False
        self.imputer_fit = False
        self.selector_fit = False
        self.corr_fit = False
        self.scaler_fit = False
        return self

    def fit_masks(self, x_array: npt.NDArray, update: bool = False):
        """
        Prepare numpy masks for filtering out problematic features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to analyze for masks creation.
        update: bool, default=False
            Whether to update existing masks if already fit.
        """

        if self.masks_fit and not update:
            raise RuntimeError('Cannot fit masks twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)

        self.masks['missing'] = np.isnan(x_array).mean(axis=0) > self.frac_missing_
        self.masks['large'] = (np.abs(x_array) > self.large_threshold_).mean(axis=0) > self.frac_large_
        self.masks['infinite'] = np.isinf(x_array).any(axis=0)
        self.masks['zero'] = np.nanvar(x_array, axis=0) == 0

        self.masks['combined'] = ~(self.masks['missing'] | self.masks['infinite'] | self.masks['zero'] | self.masks['large'])
        self.n_features['start'] = x_array.shape[1]
        self.n_features['masks'] = int(self.masks['combined'].sum())
        self.masks_fit = True
        if self.n_features["masks"] == 0:
            raise ValueError("No features remain after applying masks.")

    def apply_masks(self, x_array: npt.NDArray):
        """
        Apply previously fitted masks to filter out problematic features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to filter using masks.

        Returns
        --------
        npt.NDArray
            Filtered feature array.
        """

        if not self.masks_fit:
            raise RuntimeError('Masks have not been fit yet. Please call .fit_masks() first.')

        x_array = self.validate_features(x_array)
        x_array = x_array[:, self.masks['combined']]

        return x_array

    def fit_corr(self, x_array: npt.NDArray, threshold: float = 0.9, frac: float = 1.0, update: bool = False):
        """
        Prepare a feature correlation mask to remove highly correlated features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to analyze for correlation.
        threshold: float, default=0.9
            Correlation threshold above which features are considered highly correlated.
        frac: float, default=1.0
            Fraction of samples to use for correlation analysis.
        update: bool, default=False
            Whether to update existing correlation mask if already fit.
        """

        if self.corr_fit and not update:
            raise RuntimeError('Cannot fit correlation mask twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.masks['correlation'] = np.ones(x_array.shape[1], dtype=bool)

        if frac < 1.0:
            if frac <= 0.0 or frac > 1.0:
                raise ValueError(f"Fraction {frac} must be between 0 and 1.")

            sample_size = max(floor(x_array.shape[0] * frac), 2)
            rng = np.random.default_rng(seed=self.random_state_)
            sample_indices = rng.choice(x_array.shape[0], size=sample_size, replace=False)
            x_array = x_array[sample_indices, :]

        correlation_matrix = np.abs(np.triu(np.corrcoef(x_array.T), k=1))
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        rows, cols = np.where(correlation_matrix >= threshold)

        if len(rows) == 0:  # no highly correlated features
            self.corr_fit = True
            return

        df = pl.DataFrame({
            'Idx1': rows,
            'Idx2': cols,
            'Correlation': [correlation_matrix[row, col] for row, col in zip(rows, cols)]
        }).sort('Correlation', descending=True)

        while len(df) > 0:

            counts = (pl.concat([
                df[['Idx1', 'Correlation']].rename({'Idx1': 'Idx'}),
                df[['Idx2', 'Correlation']].rename({'Idx2': 'Idx'})
            ])
            .group_by('Idx')
            .agg([
                pl.col('Correlation').len().alias('Count'),
                pl.col('Correlation').mean().round(5).alias('Mean_Correlation')
            ])
            .sort(['Count', 'Mean_Correlation'], descending=[True, True])
            )

            max_corr_idx = counts['Idx'].item(0)
            self.masks['correlation'][max_corr_idx] = False

            df = df.filter(
                (pl.col('Idx1') != max_corr_idx) &
                (pl.col('Idx2') != max_corr_idx)
            )

        self.corr_fit = True
        return

    def apply_corr(self, x_array: npt.NDArray):
        """
        Apply previously fitted correlation mask to remove highly correlated features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to filter using correlation mask.

        Returns
        --------
        npt.NDArray
            Filtered feature array with highly correlated features removed.
        """

        if not self.corr_fit:
            raise RuntimeError('Correlation mask has not been fit yet. Please call .fit_corr() first.')

        x_array = self.validate_features(x_array)
        x_array = x_array[:, self.masks['correlation']]
        self.n_features['correlation'] = x_array.shape[1]

        return x_array

    def fit_clipper(self, x_array: npt.NDArray, update: bool = False):
        """
        Fit a SimpleImputer for handling unstable values.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to fit the clipper on.
        update: bool, default=False
            Whether to update existing clipper if already fit.
        """
        if self.clipper_fit and not update:
            raise RuntimeError('Cannot fit clipper twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.clipper = QuantileClipper(lower_quantile=0.01, upper_quantile=0.99)
        self.clipper.fit(x_array)
        self.clipper_fit = True

    def apply_clipper(self, x_array: npt.NDArray):
        """
        Apply previously fitted Clipper to handle unstable values.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array with potentially unstable values.

        Returns
        --------
        npt.NDArray
            Feature array with clipped values.
        """
        if not self.clipper_fit:
            raise RuntimeError('Clipper has not been fit yet. Please call .fit_clipper first.')

        x_array = self.validate_features(x_array)
        x_array = self.clipper.transform(x_array)
        return x_array

    def fit_imputer(self, x_array: npt.NDArray, update: bool = False):
        """
        Fit a SimpleImputer for handling missing values.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to fit the imputer on.
        update: bool, default=False
            Whether to update existing imputer if already fit.
        """

        if self.imputer_fit and not update:
            raise RuntimeError('Cannot fit imputer twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(x_array)
        self.imputer_fit = True

    def apply_imputer(self, x_array: npt.NDArray):
        """
        Apply previously fitted imputer to handle missing values.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array with potential missing values.

        Returns
        --------
        npt.NDArray
            Feature array with imputed values.
        """

        if not self.imputer_fit:
            raise RuntimeError('Imputer has not been fit yet. Please call .fit_imputer first.')

        x_array = self.validate_features(x_array)
        x_array = self.imputer.transform(x_array)
        return x_array

    def fit_selector(self, x_array: npt.NDArray, selector_threshold: float = 1e-3, update: bool = False):
        """
        Fit a VarianceThreshold selector to remove low-variance features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to fit the selector on.
        selector_threshold: float, default=1e-3
            Variance threshold below which features are removed.
        update: bool, default=False
            Whether to update existing selector if already fit.
        """

        if self.selector_fit and not update:
            raise RuntimeError('Cannot fit selector twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.selector = VarianceThreshold(selector_threshold)
        self.selector.fit(x_array)
        self.selector_fit = True

    def apply_selector(self, x_array: npt.NDArray):
        """
        Apply previously fitted selector to remove low-variance features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to filter using variance selector.

        Returns
        --------
        npt.NDArray
            Filtered feature array with low-variance features removed.
        """

        if not self.selector_fit:
            raise RuntimeError('Selector has not been fit yet. Please call .fit_selector first.')

        x_array = self.validate_features(x_array)
        x_array = self.selector.transform(x_array)
        self.n_features['selector'] = x_array.shape[1]
        return x_array

    def fit_scaler(self, x_array: npt.NDArray, update: bool = False):
        """
        Fit a RobustScaler for scaling features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to fit the scaler on.
        update: bool, default=False
            Whether to update existing scaler if already fit.
        """

        if self.scaler_fit and not update:
            raise RuntimeError('Cannot fit scaler twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.scaler = RobustScaler(unit_variance=True, quantile_range=(5, 95))
        self.scaler.fit(x_array)
        self.scaler_fit = True

    def apply_scaler(self, x_array: npt.NDArray):
        """
        Apply previously fitted scaler to scale features.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to scale.

        Returns
        --------
        npt.NDArray
            Scaled feature array.
        """

        if not self.scaler_fit:
            raise RuntimeError('Scaler has not been fit yet. Please call .fit_scaler first.')

        x_array = self.validate_features(x_array)
        x_array = self.scaler.transform(x_array)
        return x_array

    def transform(self, x_array: npt.NDArray):
        """
        Transform data by applying all enabled preprocessing steps.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to transform.

        Returns
        --------
        npt.NDArray
            Transformed feature array.
        """
        if not self.fitted_:
            raise RuntimeError(f"DataTransformer has not been fit yet. Please call .fit_transform() first.")

        x_array = self.validate_features(x_array)

        if self.use_masks:
            x_array = self.apply_masks(x_array)
        if self.use_clipper:
            x_array = self.apply_clipper(x_array)
        if self.use_imputer:
            x_array = self.apply_imputer(x_array)
        if self.use_selector:
            x_array = self.apply_selector(x_array)
        if self.use_corr:
            x_array = self.apply_corr(x_array)
        if self.use_scaler:
            x_array = self.apply_scaler(x_array)

        return x_array

    def fit_transform(self, x_array: npt.NDArray):
        """
        Fit all enabled preprocessing steps and transform data accordingly.

        Parameters
        -----------
        x_array: npt.NDArray
            Input feature array to fit on and transform.

        Returns
        --------
        npt.NDArray
            Transformed feature array.
        """
        if self.fitted_:
            raise RuntimeError("DataTransformer has already been fitted. Create a new transformer or call .reset()")

        x_array = self.validate_features(x_array)

        if self.use_masks:
            if not self.masks_fit:
                self.fit_masks(x_array)
            x_array = self.apply_masks(x_array)

        if self.use_clipper:
            if not self.clipper_fit:
                self.fit_clipper(x_array)
            x_array = self.apply_clipper(x_array)

        if self.use_imputer:
            if not self.imputer_fit:
                self.fit_imputer(x_array)
            x_array = self.apply_imputer(x_array)

        if self.use_selector:
            if not self.selector_fit:
                self.fit_selector(x_array)
            x_array = self.apply_selector(x_array)

        if self.use_corr:
            if not self.corr_fit:
                self.fit_corr(x_array)
            x_array = self.apply_corr(x_array)

        if self.use_scaler:
            if not self.scaler_fit:
                self.fit_scaler(x_array)
            x_array = self.apply_scaler(x_array)

        self.fitted_ = True
        return x_array

    def transform_df(self, df: pl.DataFrame, features_col: str):
        """
        Transform data in a DataFrame by applying all enabled preprocessing steps.

        Parameters
        -----------
        df: pl.DataFrame
            A Polars DataFrame with features to transform.
        features_col: str
            The name of the column with features.

        Returns
        --------
        df: pl.DataFrame
            Initial DataFrame with transformed features.
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
        Fit and transform data in a DataFrame by applying all enabled preprocessing steps.

        Parameters
        -----------
        df: pl.DataFrame
            A Polars DataFrame with features to transform.
        features_col: str
            The name of the column with features.

        Returns
        --------
        df: pl.DataFrame
            Initial DataFrame with transformed features.
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


def get_transformer_params(features: str):
    """
    Get recommended transformer parameters based on feature type.

    Parameters
    -----------
    features: str
        Name of the feature type to get parameters for.

    Returns
    --------
    dict
        Dictionary of recommended parameter settings for DataTransformer.
    """

    continuous = {
        'use_masks': True,
        'use_corr': True,
        'use_imputer': True,
        'use_selector': True,
        'use_scaler': True,
        'use_clipper': True,
    }

    binary = {
        'use_masks': True,
        'use_corr': True,
        'use_imputer': True,
        'use_selector': True,
        'use_scaler': False,
        'use_clipper': True,
    }

    integer = {
        'use_masks': True,
        'use_corr': True,
        'use_imputer': False,
        'use_selector': True,
        'use_scaler': False,
        'use_clipper': False,
    }

    features_mapping = {
        'AtomPair': binary,
        'Daylight': binary,
        'ECFP': binary,
        'MACCS': binary,
        'Klek': binary,
        'Mordred': binary,
        'CDDD': continuous,
        'ChemBERTa': continuous,
        'RDKit': continuous,
        'CDDD_internal': continuous,
        'MiniMol': continuous,
        'MAP': integer,
        'MAPC': integer,
        'AtomPairCount': integer,
        'DaylightCount': integer,
        'ECFPCount': integer,
        "MiniLM": continuous,
        "Qwen3": continuous,
        "MolEncoder": continuous
    }

    params = features_mapping.get(features, None)
    if params is None:
        raise KeyError(f'No parameters defined for < {features} >. Please, update the < get_transformer_params > '
                       f'function at novami.data.transform')

    return params


def merge_arrays_dataframe(df: pl.DataFrame, input_col: List[str], output_col: str):
    """
    Merge numpy arrays held in polars columns of Array type.

    Parameters
    ----------
    df: pl.DataFrame
        Polars dataframe with arrays to merge
    input_col: List[str]
        Names of columns holding arrays
    output_col: str
        Name for the output column

    Returns
    -------
    pl.DataFrame
    """
    if missing_cols := [col for col in input_col if col not in df.columns]:
        raise ValueError(f"Columns missing from the DataFrame: {missing_cols}")

    if output_col in df.columns:
        print(f"Overwriting output column: {output_col}")

    dtypes = {str(df[col].dtype).split(",")[0].split("(")[-1]
              for col in input_col}

    precisions = sorted({int(re.search(r"(\d+)", dt).group(1)) for dt in dtypes})

    has_float = any(dtype.startswith("Float") for dtype in dtypes)
    has_int = any(dtype.startswith("Int") for dtype in dtypes)
    has_uint = any(dtype.startswith("UInt") for dtype in dtypes)

    req_precision = np.max(precisions)

    if has_float:
        target_dtype = getattr(pl, f"Float{req_precision}")
    elif has_int and has_uint:
        target_dtype = getattr(pl, f"Int{req_precision}")
    elif has_int:
        target_dtype = getattr(pl, f"Int{req_precision}")
    else:
        target_dtype = getattr(pl, f"UInt{req_precision}")

    df = df.with_columns(
        pl.concat_arr([
            pl.col(col).cast(pl.Array(target_dtype, df.schema[col].shape[0]))
            for col in input_col
        ]).alias(output_col)
    )

    return df