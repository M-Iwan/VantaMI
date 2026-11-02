from typing import Dict, Any, List
import numpy as np
import optuna


# ============== #
# Random Section #
# ============== #


def _random_int(low: int, high: int, scale: str = "uniform", size: int = 64) -> np.ndarray:
    """
    Randomly sample integers between "low" and "high" (inclusive).
    """
    if low > high:
        raise ValueError("`low` must be less than or equal to `high`.")

    if scale == "uniform":
        return np.random.randint(low=low, high=high + 1, size=size)

    if scale in {"log", "log10"}:
        if low <= 0 or high <= 0:
            raise ValueError("Log-scale sampling requires positive `low` and `high`.")

        samples = np.random.uniform(low=np.log10(low), high=np.log10(high), size=size)
        return np.clip(np.rint(10 ** samples), low, high).astype(int)

    raise ValueError(f"Unsupported scale < {scale} >. Use 'uniform', 'log', or 'log10'.")


def _random_float(low: float, high: float, scale: str = "uniform", size: int = 64) -> np.ndarray:
    """
    Randomly sample floats between "low" and "high".
    """
    if low > high:
        raise ValueError("`low` must be less than or equal to `high`.")

    if scale == "uniform":
        return np.random.uniform(low=low, high=high, size=size)

    if scale in {"log", "log10"}:
        if low <= 0 or high <= 0:
            raise ValueError("Log-scale sampling requires positive `low` and `high`.")

        return 10 ** np.random.uniform(low=np.log10(low), high=np.log10(high), size=size)

    raise ValueError(f"Unsupported scale < {scale} >. Use 'uniform', 'log', or 'log10'.")


def _random_categorical(values: List[Any]) -> Any:
    """
    Randomly select one item from a list of categorical values.
    """
    if not values:
        raise ValueError("`values` must contain at least one item.")

    return values[np.random.randint(len(values))]


def get_random_params_xgb_classifier(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for XGBClassifier.
    """
    return [{
            "n_estimators": int(_random_int(100, 1000, size=1)[0]),
            "max_depth": int(_random_int(2, 12, size=1)[0]),
            "learning_rate": float(_random_float(5e-3, 1e-1, scale="log10", size=1)[0]),
            "subsample": float(_random_float(0.5, 1.0, size=1)[0]),
            "colsample_bytree": float(_random_float(0.5, 1.0, size=1)[0]),
            "gamma": float(_random_float(0.0, 5.0, size=1)[0]),
        } for _ in range(size)]


def get_random_params_xgb_regressor(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for XGBRegressor.
    """
    return [{
            "n_estimators": int(_random_int(100, 1000, size=1)[0]),
            "max_depth": int(_random_int(2, 12, size=1)[0]),
            "learning_rate": float(_random_float(5e-3, 1e-1, scale="log10", size=1)[0]),
            "subsample": float(_random_float(0.5, 1.0, size=1)[0]),
            "colsample_bytree": float(_random_float(0.5, 1.0, size=1)[0]),
            "gamma": float(_random_float(0.0, 5.0, size=1)[0]),
        } for _ in range(size)]


def get_random_params_lgbm_classifier(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for LGBMClassifier.
    """
    params_list = []

    for _ in range(size):
        boosting_type = _random_categorical(["gbdt", "dart", "goss"])

        params = {
            "learning_rate": float(_random_float(5e-3, 1e-1, scale="log10", size=1)[0]),
            "n_estimators": int(_random_int(100, 1000, size=1)[0]),
            "num_leaves": int(_random_int(2, 256, size=1)[0]),
            "boosting_type": boosting_type,
            "max_depth": int(_random_int(2, 12, size=1)[0]),
            "colsample_bytree": float(_random_float(0.5, 1.0, size=1)[0]),
        }

        if boosting_type in {"gbdt", "dart"}:
            params["subsample"] = float(_random_float(0.5, 1.0, size=1)[0])

        if boosting_type == "dart":
            params["drop_rate"] = float(_random_float(0.1, 0.5, size=1)[0])
            params["skip_drop"] = float(_random_float(0.1, 0.5, size=1)[0])

        if boosting_type == "goss":
            params["top_rate"] = float(_random_float(0.1, 0.3, size=1)[0])
            params["other_rate"] = float(_random_float(0.05, 0.2, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_lgbm_regressor(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for LGBMRegressor.
    """
    params_list = []

    for _ in range(size):
        boosting_type = _random_categorical(["gbdt", "dart", "goss"])

        params = {
            "learning_rate": float(_random_float(5e-3, 1e-1, scale="log10", size=1)[0]),
            "n_estimators": int(_random_int(100, 1000, size=1)[0]),
            "num_leaves": int(_random_int(2, 256, size=1)[0]),
            "boosting_type": boosting_type,
            "max_depth": int(_random_int(2, 12, size=1)[0]),
            "colsample_bytree": float(_random_float(0.5, 1.0, size=1)[0]),
        }

        if boosting_type in {"gbdt", "dart"}:
            params["subsample"] = float(_random_float(0.5, 1.0, size=1)[0])

        if boosting_type == "dart":
            params["drop_rate"] = float(_random_float(0.1, 0.5, size=1)[0])
            params["skip_drop"] = float(_random_float(0.1, 0.5, size=1)[0])

        if boosting_type == "goss":
            params["top_rate"] = float(_random_float(0.1, 0.3, size=1)[0])
            params["other_rate"] = float(_random_float(0.05, 0.2, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_catboost_classifier(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for CatBoostClassifier.
    """
    return [{
            "iterations": int(_random_int(100, 1000, size=1)[0]),
            "depth": int(_random_int(2, 12, size=1)[0]),
            "learning_rate": float(_random_float(5e-3, 1e-1, scale="log10", size=1)[0]),
            "l2_leaf_reg": float(_random_float(0.0, 5.0, size=1)[0]),
            "subsample": float(_random_float(0.5, 1.0, size=1)[0]),
            "colsample_bylevel": float(_random_float(0.5, 1.0, size=1)[0]),
            "grow_policy": _random_categorical(["SymmetricTree", "Depthwise", "Lossguide"]),
        } for _ in range(size)]


def get_random_params_catboost_regressor(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for CatBoostRegressor.
    """
    return [{
            "iterations": int(_random_int(100, 1000, size=1)[0]),
            "depth": int(_random_int(2, 12, size=1)[0]),
            "learning_rate": float(_random_float(5e-3, 1e-1, scale="log10", size=1)[0]),
            "l2_leaf_reg": float(_random_float(0.0, 5.0, size=1)[0]),
            "subsample": float(_random_float(0.5, 1.0, size=1)[0]),
            "colsample_bylevel": float(_random_float(0.5, 1.0, size=1)[0]),
            "grow_policy": _random_categorical(["SymmetricTree", "Depthwise", "Lossguide"]),
        } for _ in range(size)]


def get_random_params_random_forest_classifier(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for RandomForestClassifier.
    """
    return [{
            "n_estimators": int(_random_int(100, 1000, size=1)[0]),
            "max_depth": int(_random_int(2, 20, size=1)[0]),
            "max_features": _random_categorical(["sqrt", "log2", None]),
            "criterion": _random_categorical(["gini", "entropy", "log_loss"]),
            "min_samples_leaf": int(_random_int(1, 10, size=1)[0]),
            "min_samples_split": int(_random_int(2, 10, size=1)[0]),
        } for _ in range(size)]


def get_random_params_random_forest_regressor(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for RandomForestRegressor.
    """
    return [{
            "n_estimators": int(_random_int(100, 1000, size=1)[0]),
            "max_depth": int(_random_int(2, 20, size=1)[0]),
            "max_features": _random_categorical(["sqrt", "log2", None]),
            "criterion": _random_categorical(["squared_error", "poisson"]),
            "min_samples_leaf": int(_random_int(1, 10, size=1)[0]),
            "min_samples_split": int(_random_int(2, 10, size=1)[0]),
        } for _ in range(size)]


def _get_random_params_svm(size: int = 64, is_regression: bool = False) -> List[Dict]:
    """
    Shared random-search sampling logic for SVC and SVR.
    """
    params_list = []

    for _ in range(size):
        kernel = _random_categorical(["linear", "poly", "rbf", "sigmoid"])

        params = {
            "C": float(_random_float(1e-2, 1e2, scale="log10", size=1)[0]),
            "kernel": kernel,
        }

        if is_regression:
            params["epsilon"] = float(_random_float(1e-2, 1.0, scale="log10", size=1)[0])

        if kernel in {"poly", "rbf", "sigmoid"}:
            gamma_low = 1e-3 if is_regression else 1e-2
            params["gamma"] = float(_random_float(gamma_low, 1.0, scale="log10", size=1)[0])

        if kernel == "poly":
            params["degree"] = int(_random_int(2, 5, size=1)[0])

        if kernel in {"poly", "sigmoid"}:
            params["coef0"] = float(_random_float(0.0, 1.0, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_svc(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for SVC.
    """
    params_list = []

    for _ in range(size):
        kernel = _random_categorical(["linear", "poly", "rbf", "sigmoid"])

        params = {
            "C": float(_random_float(1e-2, 1e2, scale="log10", size=1)[0]),
            "kernel": kernel,
        }

        if kernel in {"poly", "rbf", "sigmoid"}:
            gamma_low = 1e-3 if is_regression else 1e-2
            params["gamma"] = float(_random_float(gamma_low, 1.0, scale="log10", size=1)[0])

        if kernel == "poly":
            params["degree"] = int(_random_int(2, 5, size=1)[0])

        if kernel in {"poly", "sigmoid"}:
            params["coef0"] = float(_random_float(0.0, 1.0, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_svr(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for SVR.
    """
    params_list = []

    for _ in range(size):
        kernel = _random_categorical(["linear", "poly", "rbf", "sigmoid"])

        params = {
            "C": float(_random_float(1e-2, 1e2, scale="log10", size=1)[0]),
            "kernel": kernel,
        }

        params["epsilon"] = float(_random_float(1e-2, 1.0, scale="log10", size=1)[0])

        if kernel in {"poly", "rbf", "sigmoid"}:
            gamma_low = 1e-3 if is_regression else 1e-2
            params["gamma"] = float(_random_float(gamma_low, 1.0, scale="log10", size=1)[0])

        if kernel == "poly":
            params["degree"] = int(_random_int(2, 5, size=1)[0])

        if kernel in {"poly", "sigmoid"}:
            params["coef0"] = float(_random_float(0.0, 1.0, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_sgd_regressor(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for SGDRegressor.
    """
    params_list = []

    for _ in range(size):
        loss = _random_categorical(["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
        penalty = _random_categorical(["l2", "l1", "elasticnet", None])
        learning_rate = _random_categorical(["constant", "optimal", "invscaling", "adaptive"])

        params = {
            "loss": loss,
            "penalty": penalty,
            "alpha": float(_random_float(1e-5, 1.0, scale="log10", size=1)[0]),
            "learning_rate": learning_rate,
            "max_iter": int(_random_int(500, 2000, size=1)[0]),
            "tol": float(_random_float(1e-4, 1e-2, scale="log10", size=1)[0]),
        }

        if penalty == "elasticnet":
            params["l1_ratio"] = float(_random_float(0.0, 1.0, size=1)[0])

        if loss in {"huber", "epsilon_insensitive", "squared_epsilon_insensitive"}:
            params["epsilon"] = float(_random_float(0.01, 0.5, size=1)[0])

        if learning_rate in {"constant", "invscaling", "adaptive"}:
            params["eta0"] = float(_random_float(1e-3, 1e-1, scale="log10", size=1)[0])

        if learning_rate == "invscaling":
            params["power_t"] = float(_random_float(0.1, 0.5, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_logistic_regression(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for LogisticRegression.
    """
    valid_combinations = ["lbfgs:l2", "lbfgs:None", "liblinear:l1", "liblinear:l2", "newton-cg:l2", "newton-cg:None",
                          "newton-cholesky:l2", "newton-cholesky:None", "sag:l2", "sag:None", "saga:elasticnet",
                          "saga:l1", "saga:l2", "saga:None"]

    params_list = []

    for _ in range(size):
        solver_penalty = _random_categorical(valid_combinations)
        solver, penalty = solver_penalty.split(":")
        penalty = None if penalty == "None" else penalty

        params = {
            "solver": solver,
            "penalty": penalty,
            "C": float(_random_float(1e-3, 10.0, scale="log10", size=1)[0]),
            "max_iter": 1024,
        }

        if penalty == "elasticnet":
            params["l1_ratio"] = float(_random_float(0.0, 1.0, size=1)[0])

        params_list.append(params)

    return params_list


def get_random_params_knn_classifier(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for KNeighborsClassifier.
    """
    return [{
        "n_neighbors": int(_random_int(1, 25, size=1)[0]),
        "weights": _random_categorical(["uniform", "distance"]),
        "algorithm": _random_categorical(["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": int(_random_int(10, 100, size=1)[0]),
        "p": int(_random_int(1, 2, size=1)[0]),
    } for _ in range(size)]



def get_random_params_knn_regressor(size: int = 64) -> List[Dict]:
    """
    Sample valid random-search configurations for KNeighborsRegressor.
    """
    return [{
        "n_neighbors": int(_random_int(1, 25, size=1)[0]),
        "weights": _random_categorical(["uniform", "distance"]),
        "algorithm": _random_categorical(["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": int(_random_int(10, 100, size=1)[0]),
        "p": int(_random_int(1, 2, size=1)[0]),
    } for _ in range(size)]


def get_random_params_function(model_name: str):
    """
    Get the random parameter functions for a model.

    Parameters
    ----------
    model_name: str
        Name of the model.

    Returns
    -------
    callable
        Function used to draw random parameters for a given model.
    """
    functions = {
        'XGBClassifier': get_random_params_xgb_classifier,
        'XGBRegressor': get_random_params_xgb_regressor,
        'LGBMClassifier': get_random_params_lgbm_classifier,
        'LGBMRegressor': get_random_params_lgbm_regressor,
        'CatBoostClassifier': get_random_params_catboost_classifier,
        'CatBoostRegressor': get_random_params_catboost_regressor,
        'RandomForestClassifier': get_random_params_random_forest_classifier,
        'RandomForestRegressor': get_random_params_random_forest_regressor,
        'SVC': get_random_params_svc,
        'SVR': get_random_params_svr,
        'SGDRegressor': get_random_params_sgd_regressor,
        'LogisticRegression': get_random_params_logistic_regression,
        'KNeighborsClassifier': get_random_params_knn_classifier,
        'KNeighborsRegressor': get_random_params_knn_regressor,
    }

    model_fn = functions.get(model_name, None)
    if model_fn is None:
        raise KeyError(f"No parameters function defined for < {model_name} >. "
                       f"Please update the < get_optuna_params_function > in novami.ml.params")

    return model_fn


# ============== #
# Optuna Section #
# ============== #


def get_optuna_params_function(model_name: str):
    """
    Get the Optuna parameter suggestion function for a model.

    Parameters
    ----------
    model_name: str
        Name of the model.

    Returns
    -------
    callable
        Function used to suggest model parameters for an Optuna trial.
    """
    functions = {
        'XGBClassifier': get_optuna_params_xgb_classifier,
        'XGBRegressor': get_optuna_params_xgb_regressor,
        'LGBMClassifier': get_optuna_params_lgbm_classifier,
        'LGBMRegressor': get_optuna_params_lgbm_regressor,
        'CatBoostClassifier': get_optuna_params_catboost_classifier,
        'CatBoostRegressor': get_optuna_params_catboost_regressor,
        'RandomForestClassifier': get_optuna_params_random_forest_classifier,
        'RandomForestRegressor': get_optuna_params_random_forest_regressor,
        'SVC': get_optuna_params_svc,
        'SVR': get_optuna_params_svr,
        'SGDRegressor': get_optuna_params_sgd_regressor,
        'LogisticRegression': get_optuna_params_logistic_regression,
        'KNeighborsClassifier': get_optuna_params_knn_classifier,
        'KNeighborsRegressor': get_optuna_params_knn_regressor,
    }

    model_fn = functions.get(model_name, None)
    if model_fn is None:
        raise KeyError(f"No parameters function defined for < {model_name} >. "
                       f"Please update the < get_optuna_params_function > in novami.ml.params")

    return model_fn


def get_optuna_params_xgb_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for XGBClassifier for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),

        # medium-impact parameters
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),

        # low-impact parameters
        #'max_leaves': trial.suggest_int('max_leaves', 0, 12),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        #'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }


def get_optuna_params_xgb_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for XGBRegressor for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),

        # medium-impact parameters
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),

        # low-impact parameters
        #'max_leaves': trial.suggest_int('max_leaves', 0, 12),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        #'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
    }


def get_optuna_params_lgbm_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for LGBMRegressor for optuna, including boosting types
    """

    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])

    params = {
        # high-impact parameters
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'boosting_type': boosting_type,

        # medium-impact parameters
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        # low-impact parameters
        #'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        #'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        #'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        #'min_split_gain': trial.suggest_float('min_split_gain', 0, 5),
    }

    if boosting_type == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)

    if boosting_type == 'goss':
        params.pop('subsample', None)  # Not used by GOSS
        params['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.3)
        params['other_rate'] = trial.suggest_float('other_rate', 0.05, 0.2)

    return params


def get_optuna_params_lgbm_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for LGBMClassifier for Optuna, including boosting types.
    """
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])

    params = {
        # high-impact parameters
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=25),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "boosting_type": boosting_type,

        # medium-impact parameters
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # low-impact parameters
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        # "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        # "min_split_gain": trial.suggest_float("min_split_gain", 0, 5),
    }

    if boosting_type == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 0.5)

    if boosting_type == "goss":
        # GOSS does not use bagging/subsampling.
        params.pop("subsample", None)
        params["top_rate"] = trial.suggest_float("top_rate", 0.1, 0.3)
        params["other_rate"] = trial.suggest_float("other_rate", 0.05, 0.2)

    return params


def get_optuna_params_catboost_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for CatBoostClassifier for Optuna.
    """
    return {
        # high-impact parameters
        "iterations": trial.suggest_int("iterations", 100, 1000, step=25),
        "depth": trial.suggest_int("depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),

        # medium-impact parameters
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),

        # low-impact parameters
        # "random_strength": trial.suggest_float("random_strength", 0, 5),
        # "bagging_temperature": trial.suggest_float(
        #     "bagging_temperature", 0, 10
        # ),
        # "border_count": trial.suggest_int("border_count", 32, 255),
        # "min_data_in_leaf": trial.suggest_int(
        #     "min_data_in_leaf", 1, 100
        # ),
    }


def get_optuna_params_catboost_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for CatBoostRegressor for optuna
    """
    return {
        # high-impact parameters
        'iterations': trial.suggest_int('iterations', 100, 1000, step=25),
        'depth': trial.suggest_int('depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),

        # medium-impact parameters
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),

        # low-impact parameters
        #'random_strength': trial.suggest_float('random_strength', 0, 5),
        #'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        #'border_count': trial.suggest_int('border_count', 32, 255),
        #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    }


def get_optuna_params_random_forest_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for RandomForestClassifier for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),

        # medium-impact parameters
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),

        # low-impact parameters
        #'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 0.05, log=True)
    }


def get_optuna_params_random_forest_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for RandomForestRegressor for optuna
    """
    return {
        # high-impact parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),

        # medium-impact parameters
        'criterion': trial.suggest_categorical('criterion', ['squared_error', 'poisson']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),

        # low-impact parameters
        # 'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 0.05, log=True)
    }


def get_optuna_params_svc(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for SVC for optuna
    """
    # high-impact parameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel
    }

    # medium-impact parameters
    if kernel in ['poly', 'rbf', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-2, 1e0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    # low-impact parameters
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def get_optuna_params_svr(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for SVR for optuna
    """
    # high-impact parameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel,
        'epsilon': trial.suggest_float('epsilon', 1e-2, 1e0, log=True)
    }

    # medium-impact parameters
    if kernel in ['poly', 'rbf', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    # low-impact parameters
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def get_optuna_params_sgd_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for StochasticGradientDescentRegressor for optuna.
    Equivalent of testing LinearRegression, Lasso, Ridge, and ElasticNet
    """
    loss = trial.suggest_categorical('loss', [
        'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
    ])

    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None])

    params = {
        # high-impact parameters
        'loss': loss,
        'penalty': penalty,
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),

        # medium-impact parameters
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),

        # low-impact parameters
        'max_iter': trial.suggest_int('max_iter', 500, 2000),
        'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
    }

    if penalty == 'elasticnet': # medium
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    if loss in ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']: # low
        params['epsilon'] = trial.suggest_float('epsilon', 0.01, 0.5)

    if params['learning_rate'] in ['constant', 'invscaling', 'adaptive']: # low
        params['eta0'] = trial.suggest_float('eta0', 0.001, 0.1, log=True)

    if params['learning_rate'] == 'invscaling': # low
        params['power_t'] = trial.suggest_float('power_t', 0.1, 0.5)

    return params


def get_optuna_params_logistic_regression(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for LogisticRegression for optuna
    """
    valid_combinations = [
        'lbfgs:l2', 'lbfgs:None',
        'liblinear:l1', 'liblinear:l2',
        'newton-cg:l2', 'newton-cg:None',
        'newton-cholesky:l2', 'newton-cholesky:None',
        'sag:l2', 'sag:None', 'saga:elasticnet',
        'saga:l1', 'saga:l2', 'saga:None'
    ]
    combination = trial.suggest_categorical('solver_penalty', valid_combinations)
    solver, penalty = combination.split(':')
    penalty = None if penalty == 'None' else penalty

    params = {
        # high-impact parameters
        'solver': solver,
        'penalty': penalty,
        'C': trial.suggest_float('C', 0.001, 10, log=True),

        # medium-impact parameters
        'max_iter': 1024,
    }

    if penalty == 'elasticnet': # medium
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    return params


def get_optuna_params_knn_classifier(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for KNeighborsClassifier for optuna
    """
    params = {
        # high-impact parameters
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),

        # medium-impact parameters
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),

        # low-impact parameters
        'p': trial.suggest_int('p', 1, 2)
    }

    return params

def get_optuna_params_knn_regressor(trial: optuna.trial.Trial) -> Dict:
    """
    Suggest values for KNeighborsRegressor for optuna
    """
    params = {
        # high-impact parameters
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),

        # medium-impact parameters
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),

        # low-impact parameters
        'p': trial.suggest_int('p', 1, 2)
    }

    return params
