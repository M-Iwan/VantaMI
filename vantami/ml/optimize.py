from copy import deepcopy
from typing import Literal, TypeAlias

from vantami.ml.models import Unit, Ensemble, ClassifierEnsemble, RegressorEnsemble
from vantami.ml.utils import *
from vantami.ml.params import *
from vantami.data.manager import KFoldManager


TaskType: TypeAlias = Literal["regression", "classification"]


def train_ensemble(unit: Unit, dataset_manager: KFoldManager, task: TaskType = "regression"):
    """
    Train an ensemble model based on dataset with KFold split.

    Parameters
    ----------
    unit: Unit
        A Unit to use for training.
    dataset_manager: KFoldManager
        Dataset manager
    task: TaskType
        Regression or classification

    Returns
    -------
    ensemble: Ensemble
    """

    ensemble_class = {"regression": RegressorEnsemble, "classification": ClassifierEnsemble}.get(task)

    if ensemble_class is None:
        raise ValueError(f"Task should be either `regression` or `classification`, got < {task} > instead.")

    ensemble = ensemble_class()

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


def optuna_search(model_name: str, dataset_manager: KFoldManager, optimization_metric: str = "RMSE",
                  task: TaskType = "regression", n_trials: int = 64, n_jobs: int = 1,
                  config: Dict[str, Any] | None = None):
    """
    Find optimal hyperparameters for a given model and data using Optuna

    Parameters
    ----------
    model_name: str
        Model of the name to use. Must be available in ml.params
    dataset_manager: KFoldManager
        Initialized KFoldManager object
    optimization_metric: str
        Metric used to select the best hyperparameters
    task: Literal["regression", "classification"]
        Either `regression` or `classification`
    n_trials: int
        Number of trials to run
    n_jobs: int
        Number of CPUs to use for model training
    config: Dict[str, Any] | None
        Configuration dictionary for Unit

    Returns
    -------
    study, best_ensemble, best_params
    """
    if optimization_metric in ["Recall", "Accuracy", "ROC AUC", "Precision", "F1 Score",
                               "MCC", "R2", "Balanced Accuracy", "Specificity", "PRC AUC"]:
        direction = "maximize"
    else:
        direction = "minimize"

    features = dataset_manager.features_col

    best_ensemble = None
    best_params = None
    best_metric = float("-inf") if direction == "maximize" else float("inf")

    def objective(trial: optuna.trial.Trial):
        nonlocal best_ensemble, best_params, best_metric

        model_params_fn = get_optuna_params_function(model_name)
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

            metric_df = ensemble.aggregated_metrics.filter(
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

        is_better = (direction == "maximize" and metric > best_metric) or \
                    (direction == "minimize" and metric < best_metric)

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
