import importlib
import inspect
import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import polars as pl

from vantami.data.transform import DataTransformer, get_transformer_params
from vantami.data.embed import PCAEmbedder, UMAPEmbedder, get_embedder_params
from vantami.ml.params import *
from vantami.ml.models import RegressorUnit, ClassifierUnit


Transformer = Union[DataTransformer, Embedder]
TransformerSequence = List[Transformer]
TransformerType = Dict[str, Optional[Union[Transformer, TransformerSequence]]]


def log(message: str):
    log_time = str(datetime.datetime.now()).split('.')[0]
    print(f'{log_time}: {message}')


def prepare_feature_transformers(features: List[str], config: Optional[Dict] = None) -> TransformerType:
    """
    TODO: Write basic checks for correctness of passed config.
    Create feature-specific transformation pipelines.

    Parameters
    ----------
    features: List[str]
        Ordered feature names.
    config: Optional[Dict]
        Feature-specific transformation configuration.

    Returns
    -------
    TransformerType
        Mapping from feature names to transformation pipelines.
    """
    config = {} if config is None else config
    transformers = {}

    if isinstance(features, str):
        features = [features]

    default_transformers = {
        "transformer": DataTransformer,
        "pca": PCAEmbedder,
        "umap": UMAPEmbedder,
    }

    for feature in features:
        feature_config = config.get(feature, "transformer")

        if feature_config is None:
            transformers[feature] = None
            continue

        if feature_config in default_transformers:
            transformer_class = default_transformers[feature_config]

            if transformer_class is DataTransformer:
                params = get_transformer_params(feature)
            else:
                params = get_embedder_params(feature, transformer_class)

            transformers[feature] = transformer_class(**params)
            continue

        pipeline = []

        for transformer_class, transformer_config in feature_config:
            if transformer_class is DataTransformer:
                params = get_transformer_params(feature)
            else:
                params = get_embedder_params(feature, transformer_class)

            params.update(transformer_config)
            pipeline.append(transformer_class(**params))

        transformers[feature] = pipeline

    return transformers


def prepare_unit(model_name: str, model_params: Dict, features: List[str], task: str, n_jobs: int, config: Optional[Dict] = None):
    """
    Prepare a Unit with feature-specific transformation pipelines.

    Parameters
    ----------
    model_name: str
        Name of an sklearn-compatible model.
    model_params: Dict
        Hyperparameters passed to the model constructor.
    features: List[str]
        Ordered names of feature groups.
    task: str
        Training task: "classification" or "regression".
    n_jobs: int
        Number of CPUs to use during training.
    config: Optional[Dict]
        Mapping from feature names to transformation pipelines or None.

    Returns
    -------
    Unit
        Configured ClassifierUnit or RegressorUnit.
    """
    model_class = get_model(model_name)
    model_params = add_n_jobs(model_params, model_class, n_jobs)
    model = model_class(**model_params)

    transformer = prepare_feature_transformers(
        features=features,
        config=config,
    )

    if task == "classification":
        return ClassifierUnit(
            model=model,
            transformer=transformer,
            use_proba=True,
        )

    if task == "regression":
        return RegressorUnit(
            model=model,
            transformer=transformer,
        )

    raise ValueError(f'Task should be either "classification" or "regression", got < {task} > instead.')


def get_model(model_name: str):

    model_mapping = {
        'XGBClassifier': ('xgboost', 'XGBClassifier'),
        'XGBRegressor': ('xgboost', 'XGBRegressor'),

        'LGBMClassifier': ('lightgbm', 'LGBMClassifier'),
        'LGBMRegressor': ('lightgbm', 'LGBMRegressor'),

        'CatBoostClassifier': ('catboost', 'CatBoostClassifier'),
        'CatBoostRegressor': ('catboost', 'CatBoostRegressor'),

        'RandomForestClassifier': ('sklearn.ensemble', 'RandomForestClassifier'),
        'RandomForestRegressor': ('sklearn.ensemble', 'RandomForestRegressor'),
        'SVC': ('sklearn.svm', 'SVC'),
        'SVR': ('sklearn.svm', 'SVR'),
        'SGDRegressor': ('sklearn.linear_model', 'SGDRegressor'),
        'LogisticRegression': ('sklearn.linear_model', 'LogisticRegression'),
        'KNeighborsClassifier': ('sklearn.neighbors', 'KNeighborsClassifier'),
        'KNeighborsRegressor': ('sklearn.neighbors', 'KNeighborsRegressor'),
    }

    if model_name not in model_mapping.keys():
        raise ValueError(f'Unknown model: < {model_name} >. Please add it to get_model function '
                         f'in novami.ml.utils')

    module_name, class_name = model_mapping.get(model_name)
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(f"Could not import < {model_name} > from < {module_name} >\n> {e}")


def add_n_jobs(hparams: Dict, model_class: object, n_jobs: int = 1):
    """
    Add a parameter controlling the number of used CPUs to model hyperparameters.

    Parameters
    ----------
    hparams: Dict
        A dictionary with hyperparameters to use.
    model_class: object
        Model's class
    n_jobs: int
        Number of CPUs to use. Default is 1.

    Returns
    -------
    hparams: Dict
    """
    init_params = inspect.signature(model_class.__init__).parameters
    hparams = dict(hparams)

    if 'thread_count' in init_params:
        hparams['thread_count'] = n_jobs

    elif 'n_jobs' in init_params:
        hparams['n_jobs'] = n_jobs

    else:
        pass

    return hparams


def value_2_ohe(value: str, levels: Tuple[str]):
    """
    One hot encode a value, i.e., return its position in levels. Returns all-zero array if None.

    Parameters
    ----------
    value: str
        Value to encode
    levels: Tuple[str]
        Pre-defined levels used for construction of array.

    Returns
    -------
    npt.NDArray[np.uint8]
    """
    array = np.zeros(len(levels), np.uint8)
    if value is None:
        return array

    if not isinstance(value, str):
        raise TypeError(f"Expected the < value > to be str, got {type(value)} instead.")

    if len(levels) == 0:
        return np.empty(shape=0, dtype=np.uint8)

    if value not in levels:
        raise ValueError(f"The value < {value} > is not present in the levels.")

    array[levels.index(value)] = 1
    return array


def dataframe_2_ohe(df: pl.DataFrame, input_col: Union[str, List[str]], output_col: str,
                    levels: Optional[Dict[str, Tuple[str]]] = None):
    """
    One hot encode column(s) of a DataFrame. Missing (None) values are encoded as all-zeros.
    If levels are passed, they they used to build the array, otherwise each not-None level is treated as a separate level.

    Parameters
    ----------
    df: pl.DataFrame
        Polars dataframe with data
    input_col: Union[str, List[str]]
        Name of the column(s) to encode.
    output_col: str
        Name for the output column
    levels: Optional[Dict[str, List[str]]]
        Pre-defined levels used for construction of arrays.

    Returns
    -------
    pl.DataFrame
    """
    if isinstance(input_col, str):
        input_col = [input_col]

    if missing_col := [col for col in input_col if col not in df.columns]:
        raise ValueError(f"The following column(s) are not present in the dataframe: < {missing_col} >")

    if output_col in df.columns:
        raise ValueError(f"Column < {output_col} > already exists in the dataframe.")

    if levels is None:
        levels = {col: sorted(df[col].drop_nulls().unique()) for col in input_col}

    levels_shape = sum([len(val) for val in levels.values()])

    df = df.with_columns(
        pl.struct(input_col).map_elements(
            lambda row: np.concat(
                [value_2_ohe(row[col], levels=levels[col]) for col in input_col]
            ),
            return_dtype=pl.Array(pl.UInt8, shape=levels_shape)
        ).alias(output_col)
    )

    return df