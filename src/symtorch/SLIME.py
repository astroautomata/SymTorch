"""
SymTorch SLIME Module

This module implements SLIME (SupraLocal Interpretable Model Agnostic Explanations),
a model interpretability technique that extends LIME by using symbolic regression
instead of linear models for local approximations.
"""
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
    """
    Convert a PySR regressor to a callable Python function.

    This helper function extracts the symbolic equation from a fitted PySRRegressor
    and converts it to a numpy-compatible lambda function that can be called directly.

    Args:
        regressor (PySRRegressor): A fitted PySR regressor containing discovered equations
        complexity (int, optional): Specific complexity level of equation to extract.
                                   If None, uses the best equation according to PySR scoring.

    Returns:
        tuple: A tuple containing:
            - f (callable): Numpy-compatible lambda function that evaluates the symbolic expression
            - vars_sorted (list): List of sympy symbols representing the input variables,
                                sorted alphabetically (e.g., [x0, x1, x2])

    Raises:
        ValueError: If specified complexity level is not found in the equation set
        RuntimeError: If the symbolic expression cannot be converted to a lambda function

    Example:
        >>> regressor = SLIME(f=model.predict, inputs=X_train, x=x0, J_neighbours=10)
        >>> slime_func, variables = regressor_to_function(regressor)
        >>> # Call the function with individual feature values
        >>> prediction = slime_func(x0[0], x0[1], x0[2])
    """
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


def SLIME(f, inputs, x=None, num_synthetic=0, var=None, J_neighbours=10, real_weighting=1.0, pysr_params=None, fit_params=None, nn_metric='euclidean'):
    """
    Fit a SLIME (SupraLocal Interpretable Model Agnostic Explanations) model.

    SLIME is a model interpretability technique that extends LIME (Local Interpretable
    Model Agnostic Explanations) by using symbolic regression instead of linear models
    to approximate black-box model behavior in a local region around a point of interest.

    The algorithm works as follows:
    1. Define a local neighborhood around point x₀ using J nearest neighbors
    2. Optionally generate synthetic samples around x₀ using Gaussian perturbations
    3. Evaluate the black-box model f(x) on all neighborhood points
    4. Fit a symbolic regression model that minimizes:

       g*(x) = argmin_g ∑(synthetic) π(z_i)[f(z_i) - g(z_i)]² + M ∑(neighbors) [f(z_j) - g(z_j)]²

       where π(z_i) = exp(-||x₀ - z_i||²/σ²) is a Gaussian proximity kernel

    SLIME offers better interpretability than LIME by discovering closed-form analytic
    expressions that may be nonlinear, while maintaining the same local approximation
    guarantee.

    Args:
        f (callable): Black-box model function to approximate. Should accept numpy array
                     of shape (n_samples, n_features) and return predictions.
        inputs (np.ndarray): Dataset of input samples, shape (n_samples, n_features).
                            Used to find nearest neighbors.
        x (np.ndarray, optional): Point of interest around which to build local approximation,
                                 shape (n_features,). If None, uses all inputs without
                                 neighborhood selection.
        num_synthetic (int, optional): Number of synthetic samples to generate around x
                                      using Gaussian perturbations. Default is 0.
                                      Must be > 0 if x is specified.
        var (np.ndarray, optional): Variance for Gaussian perturbations, shape (n_features,).
                                   If None, computed from J nearest neighbors.
        J_neighbours (int, optional): Number of nearest neighbors to include in the
                                     local neighborhood. Default is 10.
        real_weighting (float, optional): Weight M for real neighbor samples in the loss
                                         function. Only applies when num_synthetic > 0.
                                         Default is 1.0.
        pysr_params (dict, optional): Custom parameters for PySRRegressor. These override
                                     DEFAULT_PYSR_PARAMS. Common parameters include:
                                     - 'niterations': Number of iterations (default 400)
                                     - 'binary_operators': List of binary operators
                                     - 'unary_operators': List of unary operators
                                     - 'complexity_of_operators': Complexity penalties
        fit_params (dict, optional): Additional parameters to pass to PySRRegressor.fit()
        nn_metric (str, optional): Distance metric for nearest neighbor search.
                                  Any metric supported by sklearn.neighbors.NearestNeighbors.
                                  Default is 'euclidean'.

    Returns:
        PySRRegressor: Fitted symbolic regression model. Use regressor_to_function()
                      to convert to a callable function, or access equations via
                      regressor.equations_ DataFrame.

    Raises:
        ValueError: If x is specified but num_synthetic is 0
        ValueError: If J_neighbours >= len(inputs)
        UserWarning: If real_weighting != 1.0 but num_synthetic is 0 (reverts to 1.0)

    Example:
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from symtorch.SLIME import SLIME, regressor_to_function
        >>>
        >>> # Train a black-box model
        >>> xgb_model = GradientBoostingRegressor().fit(X_train, y_train)
        >>>
        >>> # Define model wrapper
        >>> def f(inputs):
        ...     return xgb_model.predict(inputs)
        >>>
        >>> # Fit SLIME around a specific point
        >>> x0 = X_test[0]  # Point of interest
        >>> regressor = SLIME(
        ...     f=f,
        ...     inputs=X_test,
        ...     x=x0,
        ...     num_synthetic=5000,
        ...     J_neighbours=10,
        ...     pysr_params={
        ...         'niterations': 500,
        ...         'binary_operators': ['+', '*', '-', '/'],
        ...         'unary_operators': ['sin', 'cos', 'exp', 'log', 'sqrt'],
        ...     }
        ... )
        >>>
        >>> # Convert to callable function
        >>> slime_func, variables = regressor_to_function(regressor)
        >>> print(f"Best equation: {regressor.get_best()['equation']}")
        >>> print(f"Uses variables: {variables}")
        >>>
        >>> # Make predictions
        >>> var_indices = [int(str(v).replace('x', '')) for v in variables]
        >>> prediction = slime_func(*[x0[i] for i in var_indices])

    References:
        Fong, Motani (2025). "SLIME: SupraLocal Interpretable Model Agnostic Explanations"
        ACM Conference, https://dl.acm.org/doi/10.1145/3712255.3726721

        Ribeiro et al. (2016). "LIME: Local Interpretable Model Agnostic Explanations"
        https://arxiv.org/pdf/1602.04938
    """
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