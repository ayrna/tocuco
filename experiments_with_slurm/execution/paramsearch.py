from itertools import product


def get_gridsearch_params(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in product(*values)]


def get_randomizedsearch_params(param_distributions, n_iter=10, *, random_state=0):
    total_combinations = 1
    for values in param_distributions.values():
        total_combinations *= len(values)

    if total_combinations <= n_iter:
        from sklearn.model_selection import ParameterGrid

        return list(ParameterGrid(param_distributions))
    else:
        from sklearn.model_selection import ParameterSampler

        return list(
            ParameterSampler(
                param_distributions, n_iter=n_iter, random_state=random_state
            )
        )
