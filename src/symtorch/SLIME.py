import numpy as np
from pysr import PySRRegressor
from sympy import lambdify
from sklearn.neighbors import NearestNeighbors

DEFAULT_PYSR_PARAMS = {
    "binary_operators": ["+", "*"],
    "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
    "extra_sympy_mappings": {"inv": lambda x: 1/x},
    "niterations": 400,
    "complexity_of_operators": {"sin": 3, "exp": 3}
}

def regressor_to_function(regressor, complexity=None):
    if complexity is None:
        best_str = regressor.get_best()["equation"]
        expr = regressor.equations_.loc[regressor.equations_["equation"] == best_str, "sympy_format"].values[0]
    else:
        matching_rows = regressor.equations_[regressor.equations_["complexity"] == complexity]
        if matching_rows.empty:
            available_complexities = sorted(regressor.equations_["complexity"].unique())
            raise ValueError(f"No equation found with complexity {complexity}. Available complexities: {available_complexities}")
        expr = matching_rows["sympy_format"].values[0]

    vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))
    try:
        f = lambdify(vars_sorted, expr, "numpy")
        return f, vars_sorted
    except Exception as e:
        raise RuntimeError(f"Could not create lambdify function: {e}")


def SLIME(f, inputs, x=None, num_synthetic=0, var=None, J_neighbours=10, real_weighting=1.0, pysr_params=None, fit_params=None, nn_metric = 'euclidean'):
    # Validate real_weighting can only be used with synthetic samples
    if real_weighting != 1.0 and num_synthetic == 0:
        import warnings
        warnings.warn("real_weighting can only be modified when num_synthetic > 0. Reverting real_weighting to 1.0", UserWarning)
        real_weighting = 1.0

    if x is not None:
        if num_synthetic == 0:
            raise ValueError("Need to set num_synthetic to non-zero if x is specified.")

        # Validate J_neighbours
        if J_neighbours >= len(inputs):
            raise ValueError(f"J_neighbours ({J_neighbours}) must be less than len(inputs) ({len(inputs)})")

        # Use NearestNeighbors to find J nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=J_neighbours, metric=nn_metric).fit(inputs)
        _, indices = nbrs.kneighbors(x.reshape(1, -1))

        # Get the J nearest neighbors
        real_inputs = inputs[indices[0]]

        if var is None:
            var = np.var(real_inputs, axis=0, ddof=1)

        # Use num_synthetic directly as the number of synthetic samples
        samples = np.random.normal(loc=x, scale=np.sqrt(var), size=(num_synthetic, len(x)))
        sr_inputs = np.concatenate([real_inputs, samples], axis=0)

        print(f"Fitting SLIME with {len(sr_inputs)} points including {len(real_inputs)} real points and {num_synthetic} Gaussian sampled points.")
    else:
        print("Fitting SLIME with all inputs provided.")
        real_inputs = inputs
        sr_inputs = inputs
        samples = None

    sr_targets = f(sr_inputs)

    if pysr_params is None:
        pysr_params = {}
    final_pysr_params = {**DEFAULT_PYSR_PARAMS, **pysr_params}

    # Implement custom weighted loss if we have synthetic samples
    if x is not None and samples is not None:
        # Calculate Gaussian kernel weights for synthetic samples: pi(x) = exp(-(x_i - mu)^2 / sigma^2)
        # where mu = x (the point of interest) and sigma^2 = var
        synthetic_distances_sq = np.sum((samples - x)**2 / var, axis=1)
        gaussian_weights = np.exp(-synthetic_distances_sq)

        # Create weight vector: real samples get real_weighting, synthetic samples get gaussian_weights
        num_real = len(real_inputs)
        weights = np.concatenate([
            np.full(num_real, real_weighting),
            gaussian_weights
        ])

        # Pass weights through fit_params for PySR
        if fit_params is None:
            fit_params = {}
        pysr_params['weights'] = weights
        pysr_params['elementwise_loss'] = "f(x,y,w) = w * abs(x-y)^2"

    pysr_model = PySRRegressor(**final_pysr_params)

    if fit_params is None:
        fit_params = {}

    regressor = pysr_model.fit(sr_inputs, sr_targets, **fit_params)

    return regressor