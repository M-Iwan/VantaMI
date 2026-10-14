import inspect
import joblib
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import stats
from scipy.special import expit
from sklearn.metrics import (confusion_matrix, roc_auc_score, r2_score, mean_absolute_error, root_mean_squared_error,
                             average_precision_score, brier_score_loss)

from vantami.data.transform import DataTransformer
from vantami.data.embed import Embedder
from vantami.ml.score import ece_score
from vantami.stats.utils import round_to_significant as rts

Transformer = Union[DataTransformer, Embedder]
TransformerSequence = List[Transformer]
TransformerType = Dict[str, Optional[Union[Transformer, TransformerSequence]]]


class Unit(ABC):
    """
    A wrapper for classical machine learning models that provides an end-to-end pipeline for feature selection,
    data preprocessing, model training, and inference.

    This abstract base class provides a framework for building standardized machine learning workflows with
    automatic data validation, feature selection, and scaling capabilities.
    Each transformation stage can be individually enabled or disabled.

    Parameters
    ----------
    model: object
        A scikit-learn compatible model with at least .fit() and .predict() methods.
    transformer: Dict[str, Union[Transformer, List[Transformer]]]
        Transformer(s) responsible for pre-processing training data, connected to input features.

    Methods
    -------
    fit(x_array, y_array, sample_weight=None, groups=None)
        Fits the preprocessing pipeline and the underlying model on training data.

    predict(x_array)
        Abstract method to be implemented by subclasses for making predictions.

    score(x_array, y_array, sample_weight=None, groups=None)
        Abstract method to be implemented by subclasses for model evaluation.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com

    Changes
    -------
    Added in version 0.1.2
    Imputation added in version 0.1.3
    Correlation analysis added in version 0.1.4
    Group scoring added in version 0.2.6
    Multi-modality/multi-transformations added in version 0.5.0
    """

    def __init__(self, model, transformer: Optional[TransformerType] = None):
        """
        Initialize the Unit object.

        Parameters
        ----------
        model: object
            A scikit-learn compatible model with at least .fit() and .predict() methods.
        transformer: Optional[TransformerType]
            Mapping from feature-group names to transformers, lists of transformers, or None.
            If None, all feature groups are passed through without preprocessing.
        """
        self.model = model
        self.name = self.model.__class__.__name__
        self.no_transformer = transformer is None

        if self.no_transformer:
            self.transformer = {}
            self.input_features = None
        else:
            self.transformer = {
                name: (
                    [] if trans is None else [trans]
                    if isinstance(trans, (DataTransformer, Embedder))
                    else list(trans)
                )
                for name, trans in transformer.items()
            }

            if not self.transformer:
                raise ValueError("Transformer must be None or contain at least one feature group")

            self.input_features = tuple(self.transformer)

        self.metrics = {}
        self._model_fit = False
        self._has_weights = 'sample_weight' in inspect.signature(self.model.fit).parameters
        self._task = None

    def __repr__(self):
        return f"<{self.__class__.__name__} using {self.name}>"

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

    @staticmethod
    def validate_targets(y_array: npt.NDArray):
        """
        Check validity of passed target values and convert to 1D numpy array if needed.

        Parameters
        ----------
        y_array: npt.NDArray
            Numpy array corresponding to target values. Must be one dimensional.
        """
        if not isinstance(y_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(y_array)} instead.")

        if y_array.ndim == 0 or (y_array.ndim == 2 and y_array.shape[1] > 1) or y_array.ndim > 2:
            raise ValueError(f"Targets must be convertible to 1D array, got {y_array.ndim} dimensions instead.")

        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.reshape(-1)

        return y_array

    @staticmethod
    def validate_length(arrays: List[npt.NDArray]):
        """
        Check if all passed arrays contain the same number of samples.

        Parameters
        ----------
        arrays: List[npt.NDArray]
            A list of numpy arrays to check

        Returns
        -------
        bool
        """
        return len(set(array.shape[0] for array in arrays)) == 1

    def apply_transformers(self, x_array: Dict[str, npt.NDArray], fit: bool = False):
        """
        Fit and apply all transformations to the passed data and concatenate the results.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Dictionary of features encoded as numpy arrays.
        fit: bool
            Flag to fit the transformers.

        Returns
        -------
        npt.NDArray
        """
        if not isinstance(x_array, dict):
            raise TypeError(f"Expected a dictionary of feature arrays, got {type(x_array)} instead.")

        if not x_array:
            raise ValueError("At least one feature group must be provided.")

        if not self.validate_length(list(x_array.values())):
            raise ValueError(f"Passed features must correspond to the same number of samples")

        # No transformers passed -> no preprocessing
        if self.no_transformer and self.input_features is None:
            self.input_features = tuple(x_array)
            self.transformer = {
                name: []
                for name in self.input_features
            }

        missing_features = set(self.input_features).difference(set(x_array.keys()))
        if missing_features:
            raise ValueError(f"Missing features in passed data: {missing_features}")

        extra_features = set(x_array.keys()).difference(set(self.input_features))
        if extra_features:
            raise ValueError(f"Extra features in passed data: {extra_features}")

        x_arrays = []

        for name in self.input_features:
            sub_array = self.validate_features(x_array[name])
            for transformer in self.transformer[name]:
                sub_array = transformer.fit_transform(sub_array) if fit else transformer.transform(sub_array)
            x_arrays.append(sub_array)

        all_same_lengths = len(set(sub_array.shape[0] for sub_array in x_arrays)) == 1
        if not all_same_lengths:
            raise ValueError(f"Passed arrays have different number of samples")

        x_array = self.validate_features(np.concatenate(x_arrays, axis=1))
        return x_array

    def fit(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
            groups: Optional[npt.NDArray] = None, refit: bool = False):
        """
        Transform data and fit model. Groups parameter is ignored during fitting and exists for compatibility.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Dictionary of features encoded as numpy arrays.
        y_true: npt.NDArray
            Numpy array corresponding to target values.
        sample_weight: Optional[npt.NDArray]
            Target values weights.
        groups: Optional[npt.NDArray]
            Parameter included for compatibility.
        refit: bool
            An option to refit the model to new data.
        """
        if self._model_fit and not refit:
            raise RuntimeError("Model already fitted. Pass refit=True to overwrite this behavior.")

        if self._model_fit and refit:
            self.reset_transformers()
            self.metrics = {}

        x_array = self.apply_transformers(x_array, fit=True)
        y_true = self.validate_targets(y_true)

        if x_array.shape[0] != y_true.shape[0]:
            raise ValueError(f"Features and target sample counts must match.")

        if sample_weight is not None:
            sample_weight = self.validate_targets(sample_weight)
            if sample_weight.shape[0] != y_true.shape[0]:
                raise ValueError("Sample weights and targets must have matching lengths.")

        if self._has_weights and sample_weight is not None:
            self.model.fit(x_array, y_true, sample_weight=sample_weight)
        else:
            self.model.fit(x_array, y_true)

        self._model_fit = True
        return self

    def predict(self, x_array: Dict[str, npt.NDArray]):
        """
        Predict target values for new data.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Dictionary of features encoded as numpy arrays.

        Returns
        -------
        predictions: npt.NDArray
        """
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')

        x_array = self.apply_transformers(x_array, fit=False)
        predictions = self.model.predict(x_array).reshape(-1) # no support for multitask predictions

        return predictions

    def reset_transformers(self):
        """
        Reset all transformers to their initial states to permit refitting.
        """
        for name, transformers in self.transformers.items():
            for transformer in transformers:
                transformer.reset()

    @abstractmethod
    def score(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
              groups: Optional[npt.NDArray] = None, eval_type: Optional[str] = None):
        """
        Predict and score the quality of predictions.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Dictionary of features encoded as numpy arrays.
        y_true: npt.NDArray
            Numpy array corresponding to known target values.
        sample_weight: Optional[npt.NDArray]
            Target values weights.
        groups: Optional[npt.NDArray]
            Grouping scheme. If passed, performance metrics are calculated separately for each group.
        eval_type: str
            Name for persistent metrics storage in self.metrics
        """
        pass

    def save(self, path: str):
        """
        Save the model.

        Parameters
        ----------
        path: str
            Path for saving model.
        """
        joblib.dump(self, path, compress=('xz', 5))

    @staticmethod
    def load(path: str):
        """
        Load the model.

        Parameters
        ----------
        path: str
            Path for loading model.
        """
        return joblib.load(path)


class ClassifierUnit(Unit):
    """
    Unit class for classification that additionally implements predict_proba. Multiclass predictions not yet supported.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Added in VantaMI version: 0.1.2
    """
    def __init__(self, model, transformer: Optional[TransformerType] = None, use_proba: bool = True):
        """
        Initialize a classification unit with a model and preprocessing transformers.

        Parameters
        ----------
        model: object
            Scikit-learn compatible classification model implementing .fit() and .predict().
        transformer: Optional[TransformerType]
            Mapping from feature-group names to transformers, lists of transformers, or None.
            If None, all feature groups are passed through without preprocessing.
        use_proba: bool, default=True
            Whether probability-based predictions and probability-based metrics should be used.
        """

        super().__init__(model=model, transformer=transformer)

        self.use_proba = use_proba
        self._has_proba = hasattr(self.model, 'predict_proba')
        self._has_decision = hasattr(self.model, 'decision_function')
        self._task = 'Classification'

    def predict_proba(self, x_array: Dict[str, npt.NDArray]):
        """
        Predict positive-class probabilities or decision scores for input features.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.

        Returns
        -------
        npt.NDArray
            One-dimensional array containing a score or probability for the positive class.
        """
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')
        if not self.use_proba:
            raise RuntimeError('Probability prediction is disabled. It can be overridden by setting .use_proba = True')

        x_array = self.apply_transformers(x_array, fit=False)

        if self._has_proba:
            y_score = self.model.predict_proba(x_array)[:, 1]
        elif self._has_decision:
            y_score = expit(self.model.decision_function(x_array))
        else:
            raise RuntimeError(f'{self.name} supports neither decision_function nor predict_proba, '
                               f'but .use_proba was set to True')

        return y_score.reshape(-1)

    def evaluate(self, y_true: npt.NDArray, y_pred: npt.NDArray, y_score: npt.NDArray,
                 sample_weight: Optional[npt.NDArray] = None):
        """
        Calculate classification performance metrics from known labels and predictions.

        Parameters
        ----------
        y_true: npt.NDArray
            One-dimensional array containing true binary target labels.
        y_pred: npt.NDArray
            One-dimensional array containing predicted binary target labels.
        y_score: npt.NDArray or None
            One-dimensional array containing positive-class probabilities or scores.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.

        Returns
        -------
        pl.DataFrame
            Long-format table containing classification metric names and values.
        """

        def safe_div(numerator, denominator, default_=0.0):
            return numerator / denominator if denominator != 0 else default_

        y_true = self.validate_targets(y_true)
        y_pred = self.validate_targets(y_pred)

        labels = np.unique(np.concatenate([y_true, y_pred]))
        true_labels = np.unique(y_true)

        if len(labels) != 2 or len(true_labels) != 2:
            return pl.DataFrame(schema={"Metric": pl.String, "Value": pl.Float64})

        sample_weight = self.validate_targets(sample_weight) if sample_weight is not None \
            else np.ones(y_true.shape[0], dtype=float)

        conf_mat = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)

        TN, FP, FN, TP = conf_mat.ravel()

        rec = safe_div(TP, TP + FN)
        spec = safe_div(TN, TN + FP)

        values = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'Accuracy': rts(safe_div(TP + TN, TP + TN + FP + FN), 5),
            'Recall': rts(rec, 5),
            'Specificity': rts(spec, 5),
            'Precision': rts(safe_div(TP, TP + FP), 5),
            'Balanced Accuracy': rts((rec + spec) / 2, 5),
            'GeomRS': rts(np.sqrt(rec * spec), 5),
            'HarmRS': rts(safe_div(2 * rec * spec, rec + spec), 5),
            'F1 Score': rts(safe_div(2 * TP, 2 * TP + FP + FN), 5),
            'MCC': rts(safe_div((TP * TN) - (FP * FN), np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))), 5),
        }

        if y_score is not None:
            y_score = self.validate_targets(y_score)

            values.update({
                "ROC AUC": rts(roc_auc_score(y_true, y_score, sample_weight=sample_weight), 5),
                'PRC AUC': rts(average_precision_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight), 5),
                'Brier': rts(brier_score_loss(y_true=y_true, y_score=y_score, sample_weight=sample_weight), 5),
                'ECE': rts(ece_score(y_true=y_true, y_score=y_score, sample_weight=sample_weight), 5)
            })


        df = pl.DataFrame(values).unpivot(variable_name='Metric', value_name='Value')
        return df

    def score(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
              groups: Optional[npt.NDArray] = None, eval_type: Optional[str] = None):
        """
        Predict labels and calculate overall and optional group-specific classification metrics.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.
        y_true: npt.NDArray
            One-dimensional array containing true binary target labels.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.
        groups: Optional[npt.NDArray]
            One-dimensional array defining groups for separate metric calculation.
        eval_type: str
            Name for persistent metrics storage in self.metrics

        Returns
        -------
        pl.DataFrame
            Long-format table containing overall and, when requested, group-specific metrics.
        """
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')

        y_true = self.validate_targets(y_true)
        y_pred = self.predict(x_array)

        if self.use_proba and (self._has_proba or self._has_decision):
            y_score = self.predict_proba(x_array)
        else:
            y_score = None

        metrics = (self.evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = self.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true = y_true[mask]
                    g_y_pred = y_pred[mask]
                    g_y_score = y_score[mask] if y_score is not None else None
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = self.evaluate(y_true=g_y_true, y_pred=g_y_pred, y_score=g_y_score, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        if eval_type is not None:
            self.metrics[eval_type] = metrics

        return metrics


class RegressorUnit(Unit):
    """
    Unit class for regression.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Added in VantaMI version: 0.1.2
    """
    def __init__(self, model, transformer: Optional[TransformerType] = None):
        """
        Initialize a regression unit with a model and preprocessing transformers.

        Parameters
        ----------
        model: object
            Scikit-learn compatible regression model implementing .fit() and .predict().
        transformer: Optional[TransformerType]
            Mapping from feature-group names to transformers, lists of transformers, or None.
            If None, all feature groups are passed through without preprocessing.
        """

        super().__init__(model=model, transformer=transformer)
        self._task = 'Regression'

    def evaluate(self, y_true: npt.NDArray, y_pred: npt.NDArray, sample_weight: Optional[npt.NDArray] = None):
        """
        Calculate regression performance metrics from known and predicted target values.

        Parameters
        ----------
        y_true: npt.NDArray
            One-dimensional array containing true target values.
        y_pred: npt.NDArray
            One-dimensional array containing predicted target values.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.

        Returns
        -------
        pl.DataFrame
            Long-format table containing regression metric names and values.
        """

        y_true = self.validate_targets(y_true)
        y_pred = self.validate_targets(y_pred)

        sample_weight = self.validate_targets(sample_weight) if sample_weight is not None \
            else np.ones(y_true.shape[0], dtype=float)

        values = {
            'R2': rts(r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight), 5),
            'MAE': rts(mean_absolute_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight), 5),
            'RMSE': rts(root_mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight), 5)
        }

        df = pl.DataFrame(values).unpivot(variable_name='Metric', value_name='Value')
        return df

    def score(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
              groups: Optional[npt.NDArray] = None, eval_type: Optional[str] = None):
        """
        Predict target values and calculate overall and optional group-specific regression metrics.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.
        y_true: npt.NDArray
            One-dimensional array containing true target values.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.
        groups: Optional[npt.NDArray]
            One-dimensional array defining groups for separate metric calculation.
        eval_type: str
            Name for persistent metrics storage in self.metrics

        Returns
        -------
        pl.DataFrame
            Long-format table containing overall and, when requested, group-specific metrics.
        """
        if not self._model_fit:
            raise RuntimeError('Model has not been fit yet. Please call .fit first.')

        y_true = self.validate_targets(y_true)
        y_pred = self.predict(x_array)

        metrics = (self.evaluate(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = self.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true, g_y_pred = y_true[mask], y_pred[mask]
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = self.evaluate(y_true=g_y_true, y_pred=g_y_pred, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        if eval_type is not None:
            self.metrics[eval_type] = metrics

        return metrics


class Ensemble(ABC):
    """
    Base class for managing ensembles of Unit models.

    Parameters
    ----------
    units: List[Unit], optional
        Pre-trained Unit instances to include in the ensemble.
    voting: str, default='soft'
        Strategy for combining predictions
    weights: npt.NDArray, optional
        Weights for each unit when using weighted voting. Must be a 1D array or
        a column vector with length matching the number of units.

    Attributes
    ----------
    units: List[Unit]
        The collection of Unit instances in the ensemble.
    voting: str
        The voting strategy used for combining predictions.
    raw_weights: Optional[npt.NDArray]
        Not-normalized weights for each unit.
    weights: Optional[npt.NDArray]
        Normalized weights for each unit (sums to 1.0).
    unit_type: type
        The type of the units in the ensemble (all units must be of the same type).
    metrics: dict
        Dictionary storing evaluation metrics for the ensemble.
    _has_units: bool
        Whether the ensemble contains any units.
    _has_weights: bool
        Whether weights have been provided for weighted voting.

    Methods
    -------
    add_unit(unit)
        Add a Unit to the ensemble.

    aggregate_metrics()
        Calculate average metrics across all units in the ensemble.

    predict(x_array)
        Make predictions using the ensemble. Must be implemented by subclasses.

    score(x_array, y_true, sample_weight=None, groups=None)
        Evaluate the ensemble on test data. Must be implemented by subclasses.

    save(path)
        Save the ensemble to a file.

    load(path)
        Load an ensemble from a file.

    Notes
    -----
    Author: Mateusz Iwan
    Email: mateusz.iwan@hotmail.com
    Added in VantaMI version: 0.1.2
    """

    def __init__(self, units: Optional[List[Unit]] = None, voting: str = 'soft', weights: Optional[npt.NDArray] = None):
        """
        Initialize an ensemble of compatible Unit models.

        Parameters
        ----------
        units: Optional[List[Unit]], default=None
            Pre-trained Unit instances to include in the ensemble.
        voting: str, default='soft'
            Strategy used to combine predictions from individual units.
        weights: Optional[npt.NDArray],
            One-dimensional array or column vector of weights corresponding to the units.
        """
        self.units = [] if units is None else list(units)
        self.voting = voting
        self.raw_weights = None
        self.weights = None
        self.unit_type = None
        self.metrics = {}
        self.aggregated_metrics = None
        self._has_units = len(self.units) > 0
        self._has_weights = False
        self._task = None

        if weights is not None:
            if not isinstance(weights, np.ndarray):
                raise TypeError('Weights must be a numpy array')

            if weights.ndim > 2:
                raise ValueError('Weights must be flattenable to a 1D array')
            if weights.ndim == 2:
                if weights.shape[1] == 1:
                    weights = weights.reshape(-1)
                else:
                    raise ValueError('2D array was passed, but the 2nd dimension is greater than 1.')
            if weights.shape[0] != len(self.units):
                raise ValueError('Number of units does not match the number of passed weights.')

            if not np.all(np.isfinite(weights)):
                raise ValueError('All weights must be finite.')

            if np.any(weights < 0):
                raise ValueError('All weights must be non-negative.')

            if np.sum(np.abs(weights)) == 0:
                raise ValueError('Weights cannot all be zeroes.')

            self.raw_weights = weights.astype(float, copy=True)
            self.weights = self.raw_weights / self.raw_weights.sum()
            self._has_weights = True

        if self._has_units:
            if not all(isinstance(unit, Unit) for unit in self.units):
                raise TypeError('All units must be of type Unit')

            unit_types = {type(unit) for unit in self.units}
            if len(unit_types) != 1:
                raise TypeError('All units must be of the same Unit subclass.')

            self.unit_type = type(self.units[0])

    @staticmethod
    def validate_targets(y_array: npt.NDArray):
        """
        Check validity of passed target values and convert to 1D numpy array if needed.

        Parameters
        ----------
        y_array: npt.NDArray
            Numpy array corresponding to target values. Must be one dimensional.
        """
        if not isinstance(y_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(y_array)} instead.")

        if y_array.ndim == 0 or (y_array.ndim == 2 and y_array.shape[1] > 1) or y_array.ndim > 2:
            raise ValueError(f"Targets must be convertible to 1D array, got {y_array.ndim} dimensions instead.")

        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.reshape(-1)

        return y_array

    def add_unit(self, unit: Unit, weight: Optional[float] = None):
        """
        Add a compatible Unit instance to the ensemble.

        Parameters
        ----------
        unit: Unit
            Unit instance to add. Its concrete type must match existing units.
        weight: Optional[float], default=None
            Initial unnormalized weight assigned to the added unit. A weight must
            be supplied when the ensemble already uses weighted voting.

        Returns
        -------
        Ensemble
            Ensemble instance containing the added unit.
        """
        if not isinstance(unit, Unit):
            raise TypeError('Unit must be of type Unit')

        if not self._has_units:
            self.units.append(unit)
            self.unit_type = type(unit)
            self._has_units = True

            if weight is not None:
                self.raw_weights = np.array([weight], dtype=float)
                self.weights = np.array([1.0], dtype=float)
                self._has_weights = True

            return self

        if type(unit) is not self.unit_type:
            raise TypeError(f'Units must be of type {self.unit_type.__name__}, received {type(unit).__name__} instead')

        if self._has_weights and weight is None:
            raise ValueError("A weight must be provided when adding a unit to a weighted ensemble")

        self.units.append(unit)
        if self._has_weights:
            self.raw_weights = np.append(self.raw_weights, float(weight))
            self.weights = self.raw_weights / self.raw_weights.sum()

        return self

    def aggregate_metrics(self):
        """
        Aggregate stored metrics across all units in the ensemble.

        Parameters
        ----------
        None

        Returns
        -------
        Optional[pl.DataFrame]
            Aggregated metric table containing mean, standard deviation, minimum,
            maximum, and median values across units. Returns None when no stored
            unit metrics are available.
        """
        metrics = []

        if not self._has_units:
            return

        for idx, unit in enumerate(self.units):
            for evaluation_type, metrics_df in unit.metrics.items():
                df = metrics_df.with_columns([
                    pl.lit(idx).alias('Unit'),
                    pl.lit(evaluation_type).alias('Set')
                ])
                metrics.append(df)

        if not metrics:
            self.aggregated_metrics = None
            return None

        self.aggregated_metrics = (pl.concat(metrics)
                        .group_by(['Set', 'Metric', 'Group'])
                        .agg([
                            pl.col("Value").mean().alias("Mean"),
                            pl.col("Value").std().alias("SD"),
                            pl.col("Value").min().alias("Min"),
                            pl.col("Value").max().alias("Max"),
                            pl.col("Value").median().alias("Median")
                        ])
                        .sort(['Set', 'Metric', 'Group'])
        )

        return self.aggregated_metrics

    @abstractmethod
    def predict(self, x_array: Dict[str, npt.NDArray]) -> npt.NDArray:
        """
        Generate predictions by combining predictions from all units.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.

        Returns
        -------
        npt.NDArray
            One-dimensional array containing ensemble predictions.
        """
        pass

    @abstractmethod
    def score(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
              groups: Optional[npt.NDArray] = None, eval_type: Optional[str] = None):
        """
        Evaluate ensemble predictions using known target values.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.
        y_true: npt.NDArray
            One-dimensional array containing known target values.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.
        groups: Optional[npt.NDArray]
            One-dimensional array defining groups for separate metric calculation.
        eval_type: str
            Name for persistent metrics storage in self.metrics

        Returns
        -------
        pl.DataFrame
            Long-format table containing overall and optional group-specific metrics.
        """
        pass

    def save(self, path: str):
        """
        Save the ensemble to a compressed file.

        Parameters
        ----------
        path: str
            Destination path for the saved ensemble.
        """
        joblib.dump(self, path, compress=('xz', 5))

    @staticmethod
    def load(path: str):
        """
        Load a previously saved ensemble from a file.

        Parameters
        ----------
        path: str
            Path to the saved ensemble file.

        Returns
        -------
        Ensemble
            Deserialized ensemble instance.
        """
        return joblib.load(path)


class ClassifierEnsemble(Ensemble):
    """
    For ClassifierEnsemble, "voting" governs only the .predict_proba method.
    Regardless of its setting, the .predict method ALWAYS returns the mode of predictions.
    For the .predict_proba method:
        - 'soft': returns the mean value
        - 'weighted': returns the weighted average
    """
    def __init__(self, units: Optional[List[ClassifierUnit]] = None, voting: str = "soft",
                 weights: Optional[npt.NDArray] = None):
        """
        Initialize an ensemble of classification units.

        Parameters
        ----------
        units: Optional[List[ClassifierUnit]]
            Pre-trained classification units to include in the ensemble.
        voting: str, default='soft'
            Probability-combination strategy. Supported values are 'soft' and 'weighted'.
        weights: Optional[npt.NDArray]
            One-dimensional array or column vector of weights corresponding to the units.
        """
        if voting not in {"soft", "weighted"}:
            raise ValueError('Voting must be either "soft" or "weighted".')

        if units is not None and not all(isinstance(unit, ClassifierUnit) for unit in units):
            raise TypeError("ClassifierEnsemble requires ClassifierUnit instances.")

        super().__init__(units=units, voting=voting, weights=weights)
        self._task = 'Classification'

    def add_unit(self, unit: ClassifierUnit, weight: Optional[float] = None):
        """
        Add a classification unit to the ensemble.

        Parameters
        ----------
        unit: ClassifierUnit
            Classification unit to add. Its concrete type must match existing units.

        Returns
        -------
        ClassifierEnsemble
            Ensemble instance containing the added classification unit.
        """
        if not isinstance(unit, ClassifierUnit):
            raise TypeError("ClassifierEnsemble requires ClassifierUnit instances.")
        return super().add_unit(unit, weight=weight)

    def predict(self, x_array: Dict[str, npt.NDArray]) -> npt.NDArray:
        """
        Predict class labels using the mode of unit-level predictions.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.

        Returns
        -------
        npt.NDArray
            One-dimensional array containing ensemble class predictions.
        """
        if not self._has_units:
            raise RuntimeError("No units are available for prediction.")

        predictions = [unit.predict(x_array).reshape(-1, 1) for unit in self.units]
        return stats.mode(np.hstack(predictions), axis=1, keepdims=False).mode.reshape(-1)

    def predict_proba(self, x_array: Dict[str, npt.NDArray]) -> npt.NDArray:
        """
        Predict positive-class probabilities by combining unit-level probabilities.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.

        Returns
        -------
        npt.NDArray
            One-dimensional array containing positive-class probabilities or scores.
        """
        if not self._has_units:
            raise RuntimeError("No units are available for prediction.")

        if self.voting == 'weighted' and self.weights is None:
            raise ValueError('Voting was set to "weighted" but self.weights were not set.')

        probs = [unit.predict_proba(x_array).reshape(-1, 1) for unit in self.units]

        if self.voting == 'soft':
            return np.mean(np.hstack(probs), axis=1)

        elif self.voting == 'weighted':
            if len(self.weights) != len(self.units):
                raise ValueError('Number of Units does not match the number of passed weights.')
            return np.average(np.hstack(probs), axis=1, weights=self.weights)
        else:
            raise ValueError('Voting must be either "soft" or "weighted"')

    def score(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
              groups: Optional[npt.NDArray] = None, eval_type: Optional[str] = None):
        """
        Calculate overall and optional group-specific classification metrics.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.
        y_true: npt.NDArray
            One-dimensional array containing known binary target labels.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.
        groups: Optional[npt.NDArray]
            One-dimensional array defining groups for separate metric calculation.
        eval_type: str
            Name for persistent metrics storage in self.metrics

        Returns
        -------
        pl.DataFrame
            Long-format table containing overall and optional group-specific classification metrics.
        """
        if not self._has_units:
            raise RuntimeError("No Units to score")

        y_pred = self.predict(x_array)

        if all(unit.use_proba and (unit._has_proba or unit._has_decision) for unit in self.units):
            y_score = self.predict_proba(x_array)
        else:
            y_score = None

        cls_unit = self.units[0]

        metrics = (cls_unit.evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = self.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:
                    g_y_true, g_y_pred = y_true[mask], y_pred[mask]
                    g_y_score = y_score[mask] if y_score is not None else None
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = cls_unit.evaluate(
                        y_true=g_y_true,
                        y_pred=g_y_pred,
                        y_score=g_y_score,
                        sample_weight=g_sample_weight
                    )
                    g_metrics = g_metrics.with_columns(
                        pl.lit(group_id).alias('Group')
                    )
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        if eval_type is not None:
            self.metrics[eval_type] = metrics

        return metrics


class RegressorEnsemble(Ensemble):
    """
    For RegressorEnsemble, "voting" gives the following behavior:
        - 'hard': returns the median value
        - 'soft': returns the mean value
        - 'weighted': returns the weighted average
    """
    def __init__(self, units: Optional[List[RegressorUnit]] = None, voting: str = "soft",
                 weights: Optional[npt.NDArray] = None):
        """
        Initialize an ensemble of regression units.

        Parameters
        ----------
        units: Optional[List[RegressorUnit]]
            Pre-trained regression units to include in the ensemble.
        voting: str, default='soft'
            Prediction-combination strategy. Supported values are 'hard', 'soft', and 'weighted'.
        weights: Optional[npt.NDArray]
            One-dimensional array or column vector of weights corresponding to the units.
        """
        if voting not in {"hard", "soft", "weighted"}:
            raise ValueError('Voting must be either "hard", "soft", or "weighted".')

        if units is not None and not all(isinstance(unit, RegressorUnit) for unit in units):
            raise TypeError("RegressorEnsemble requires RegressorUnit instances.")

        super().__init__(units=units, voting=voting, weights=weights)
        self._task = 'Regression'

    def add_unit(self, unit: RegressorUnit, weight: Optional[float] = None):
        """
        Add a regression unit to the ensemble.

        Parameters
        ----------
        unit: RegressorUnit
            Regression unit to add. Its concrete type must match existing units.

        Returns
        -------
        RegressorEnsemble
            Ensemble instance containing the added regression unit.
        """
        if not isinstance(unit, RegressorUnit):
            raise TypeError(f"RegressorEnsemble requires RegressorUnit instances. Got {type(unit)} instead.")
        return super().add_unit(unit, weight=weight)

    def predict(self, x_array: Dict[str, npt.NDArray]) -> npt.NDArray:
        """
        Predict target values by combining predictions from regression units.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.

        Returns
        -------
        npt.NDArray
            One-dimensional array containing ensemble target predictions.
        """
        if not self._has_units:
            raise RuntimeError("No units are available for prediction.")

        if self.voting == 'weighted' and self.weights is None:
            raise ValueError('Voting was set to "weighted" but self.weights were not set.')

        predictions = [unit.predict(x_array).reshape(-1, 1) for unit in self.units]

        if self.voting == 'hard':
            return np.median(np.hstack(predictions), axis=1)

        elif self.voting == 'soft':
            return np.mean(np.hstack(predictions), axis=1)

        elif self.voting == 'weighted':
            if len(self.weights) != len(self.units):
                raise ValueError('Number of Units does not match the number of passed weights.')
            return np.average(np.hstack(predictions), axis=1, weights=self.weights)
        else:
            raise ValueError('Voting must be either "hard", "soft" or "weighted"')


    def score(self, x_array: Dict[str, npt.NDArray], y_true: npt.NDArray, sample_weight: Optional[npt.NDArray] = None,
              groups: Optional[npt.NDArray] = None, eval_type: Optional[str] = None):
        """
        Calculate overall and optional group-specific regression metrics.

        Parameters
        ----------
        x_array: Dict[str, npt.NDArray]
            Mapping from feature-group names to one- or two-dimensional feature arrays.
        y_true: npt.NDArray
            One-dimensional array containing known target values.
        sample_weight: Optional[npt.NDArray]
            One-dimensional array of non-negative sample weights.
        groups: Optional[npt.NDArray]
            One-dimensional array defining groups for separate metric calculation.
        eval_type: str
            Name for persistent metrics storage in self.metrics

        Returns
        -------
        pl.DataFrame
            Long-format table containing overall and optional group-specific regression metrics.
        """
        if not self._has_units:
            raise RuntimeError("No units are available for prediction.")

        y_pred = self.predict(x_array)

        reg_unit = self.units[0]

        metrics = (reg_unit.evaluate(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
                   .with_columns(pl.lit('Overall').alias('Group')))

        if groups is not None:
            groups = self.validate_targets(groups)  # groups should be 1D numpy array, same as targets
            unique_groups = np.unique(groups)
            group_metrics = [metrics]
            for group_id in unique_groups:
                mask = (groups == group_id)
                if np.sum(mask) > 0:  # If for some reason a group is empty...
                    g_y_true = y_true[mask]
                    g_y_pred = y_pred[mask]
                    g_sample_weight = sample_weight[mask] if sample_weight is not None else None
                    g_metrics = reg_unit.evaluate(y_true=g_y_true, y_pred=g_y_pred, sample_weight=g_sample_weight)
                    g_metrics = g_metrics.with_columns(pl.lit(group_id).alias('Group'))
                    group_metrics.append(g_metrics)

            if group_metrics:
                metrics = pl.concat(group_metrics, how='vertical_relaxed')

        if eval_type is not None:
            self.metrics[eval_type] = metrics

        return metrics
