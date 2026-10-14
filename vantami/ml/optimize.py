import inspect
from itertools import product
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import optuna

from vantami.ml.evaluate import kf_evaluate, tt_evaluate
from vantami.ml.models import Unit, Ensemble, ClassifierUnit, RegressorUnit, ClassifierEnsemble, RegressorEnsemble
from vantami.ml.score import average_scores, score_regression_model, score_classification_model
from vantami.ml.utils import *
from vantami.data.manager import KFoldManager


def random_parameters(params: dict):
    selected = dict()
    for param, values in params.items():
        selected[param] = values[np.random.randint(0, len(values))]
    return selected


def grid_parameters(params: dict):
    combinations = list(product(*params.values()))
    grid = [{key: value for key, value in zip(params.keys(), combination)} for combination in combinations]

    return grid


def random_hyperparameter_search(model_class, df: pd.DataFrame, features_col: str, target_col: str, parameters: dict,
                                 n_iter: int = 100, task: str = 'classification', library: str = 'sklearn',
                                 evaluation: str = 'kf'):
    """
    If evaluation = 'ts' is used, the dataframe must hold a 'Dataset' column with 'train'/'test'
    literal strings that are used to split it into training/evaluation sets

    Notes
    -----
    The function is now somewhat deprecated and not integrated with the rest of the code.
    """
    print("Function < random_hyperparameter_search > has not been updated in a long time and likely doesn't work")
    raise NotImplementedError

    results = []

    for iteration in range(n_iter):
        current_params = random_parameters(parameters[model_class.__name__])

        try:
            model = model_class(**current_params)

            if evaluation == 'kf':
                scores, _ = kf_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'bs':
                scores, _ = bootstrap_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'ts':

                assert 'Dataset' in df.columns
                dataset_values = df['Dataset'].unique()
                assert all(['train' in dataset_values, 'test' in dataset_values, len(dataset_values) == 2])

                scores, _ = tt_evaluate(model, df, features_col, target_col, task, library, split_column='Dataset')
            else:
                raise ValueError(f'Expected < evaluation > to be either kf, bs, or ts, got < {evaluation} > instead)')
            av_scores = average_scores(scores)

            results.append({'parameters': current_params, 'scores': av_scores})

        except ValueError:  # skip illegal combinations of parameters
            pass

    return results


def grid_hyperparameter_search(model_class, df: pd.DataFrame, features_col: str, target_col: str, parameters: dict,
                               task: str = 'classification', library: str = 'sklearn', evaluation: str = 'kf'):
    """
    Notes
    -----
    The function is now somewhat deprecated and not integrated with the rest of the code.
    """
    print("Function < grid_hyperparameter_search > has not been updated in a long time and likely doesn't work")
    raise NotImplementedError
    results = []

    grid = grid_parameters(parameters[model_class.__name__])
    total_number = len(grid)

    print(f'> Total number of models to be evaluated: {total_number} <')

    for current_param in grid:
        try:
            model = model_class(**current_param)

            if evaluation == 'kf':
                scores, _ = kf_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'bs':
                scores, _ = bootstrap_evaluate(model, df, features_col, target_col, task, library)
            elif evaluation == 'ts':

                assert 'Dataset' in df.columns
                dataset_values = df['Dataset'].unique()
                assert all(['train' in dataset_values, 'test' in dataset_values, len(dataset_values) == 2])

                scores, _ = test_evaluate(model, df, features_col, target_col, task, library, split_column='Dataset')
            else:
                raise ValueError(f'Expected < evaluation > to be either kf or bs, got < {evaluation} > instead)')

            av_scores = average_scores(scores)

            results.append({'parameters': current_param, 'scores': av_scores})

        except ValueError:  # skip illegal combinations of parameters
            pass

    return results


def train_ensemble(unit: Unit, dataset_manager: KFoldManager, task: str = 'regression'):
    """
    Train an ensemble model based on dataset with KFold split.

    Parameters
    ----------
    unit: Unit
        A Unit to use for training.
    dataset_manager: KFoldManager
        Dataset manager
    task: str
        Regression or classification

    Returns
    -------
    ensemble: Ensemble
    """

    ensemble = {'regression': RegressorEnsemble, 'classification': ClassifierEnsemble}.get(task)()

    for fold in dataset_manager.folds:
        train_data = dataset_manager.get_train_data(fold)
        eval_data = dataset_manager.get_eval_data(fold)

        funit = deepcopy(unit)
        funit.fit(**train_data)
        _ = funit.score(**train_data, eval_type="Training")
        _ = funit.score(**eval_data, eval_type="Validation")
        ensemble.add_unit(funit)

    ensemble.aggregate_metrics()
    return ensemble


def optuna_search(model_name: str, dataset_manager: KFoldManager, optimization_metric: str = 'RMSE',
                  task: str = 'regression', n_trials: int = 64, n_jobs: int = 1, config: Dict[str, Any] = None):

    # select optimization direction
    if optimization_metric in ['Recall', 'Accuracy', 'ROC AUC', 'Precision', 'F1 Score',
                               'MCC', 'R2', 'Balanced Accuracy', 'Specificity', 'PRC AUC']:
        direction = 'maximize'
    else:
        direction = 'minimize'

    features = dataset_manager.features_col

    best_ensemble = None
    best_params = None
    best_metric = float('-inf') if direction == 'maximize' else float('inf')

    def objective(trial: optuna.trial.Trial):
        nonlocal best_ensemble, best_params, best_metric

        model_params_fn = get_params_function(model_name)
        model_params = model_params_fn(trial)

        try:
            unit = prepare_unit(
                model_name=model_name,
                model_params=model_params,
                features=features,
                task=task,
                n_jobs=n_jobs,
                config=config
            )

            ensemble = train_ensemble(
                unit=unit,
                dataset_manager=dataset_manager,
                task=task
            )

            metric_df = ensemble.aggregated_metrics_metrics.filter(
                pl.col("Set") == "Validation",
                pl.col("Metric") == optimization_metric,
                pl.col("Group") == "Overall"
            )

            if metric_df.is_empty():
                raise ValueError(f"Metric {optimization_metric} was not produced for Validation/Overall")

            metric = metric_df.select("Mean").item()
            if not np.isfinite(metric):
                raise ValueError(f"Trial produced a non-finite {optimization_metric}: {metric}")

        except (ValueError, TypeError, FloatingPointError) as exc:
            raise optuna.TrialPruned(f"Trial {trial.number} pruned: {exc}") from exc

        is_better = (direction == 'maximize' and metric > best_metric) or \
                    (direction == 'minimize' and metric < best_metric)

        if is_better:
            best_metric = metric
            best_params = model_params.copy()
            best_ensemble = ensemble

        return metric

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

    if not completed_trials or best_ensemble is None or best_params is None:
        raise RuntimeError(f"Optuna search completed without any successful trials.")

    return study, best_ensemble, best_params
