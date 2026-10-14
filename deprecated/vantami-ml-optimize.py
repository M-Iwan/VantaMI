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