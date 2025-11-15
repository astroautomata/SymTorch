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
import warnings

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

    Args:
        regressor (PySRRegressor): A fitted PySR regressor containing discovered equations
        complexity (int, optional): Specific complexity level of equation to extract.
                                   If None, uses the best equation according to PySR scoring.

    Returns:
        tuple: A tuple containing:
            - f (callable): Numpy-compatible lambda function that evaluates the symbolic expression
            - vars_sorted (list): List of sympy symbols representing the input variables

    Raises:
        ValueError: If specified complexity level is not found in the equation set
        RuntimeError: If the symbolic expression cannot be converted to a lambda function
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


class SLIMEModel:
    """
    SLIME (SupraLocal Interpretable Model Agnostic Explanations) model.

    SLIME extends LIME by using symbolic regression instead of linear models
    to approximate black-box model behavior in a local region.

    Parameters:
        J_neighbours (int): Number of nearest neighbors. Default is 10.
        num_synthetic (int): Number of synthetic samples. Default is 0.
        real_weighting (float): Weight for real samples. Default is 1.0.
        nn_metric (str): Distance metric. Default is 'euclidean'.
        pysr_params (dict): Custom PySR parameters.

    Attributes (set after fit):
        x0_: Point of interest
        real_inputs_: Real nearest neighbor inputs
        synthetic_samples_: Synthetic samples
        weights_: Sample weights
        var_: Variance for perturbations
        regressor_: Fitted PySRRegressor
        equations_: DataFrame of discovered equations
    """

    def __init__(self, J_neighbours=10, num_synthetic=0, real_weighting=1.0,
                 nn_metric='euclidean', var = None, pysr_params=None):
        self.J_neighbours = J_neighbours
        self.num_synthetic = num_synthetic
        self.real_weighting = real_weighting
        self.nn_metric = nn_metric
        self.pysr_params = pysr_params if pysr_params is not None else {}

        # Set during fit
        self.x0_ = None
        self.real_inputs_ = None
        self.synthetic_samples_ = None
        self.weights_ = None
        self.regressor_ = None

    def fit(self, f, inputs, x=None, var=None, fit_params=None):
        """
        Fit SLIME model to approximate function f around point x.

        Args:
            f (callable): Black-box model function
            inputs (np.ndarray): Dataset of input samples
            x (np.ndarray, optional): Point of interest
            var (np.ndarray, optional): Variance for perturbations
            fit_params (dict, optional): Additional fit parameters

        Returns:
            self
        """
        if self.real_weighting != 1.0 and self.num_synthetic == 0:
            warnings.warn("real_weighting only works with num_synthetic > 0. Setting to 1.0", UserWarning)
            self.real_weighting = 1.0

        if x is not None:
            if self.num_synthetic == 0:
                raise ValueError("num_synthetic must be > 0 when x is specified")
            if self.J_neighbours >= len(inputs):
                raise ValueError(f"J_neighbours ({self.J_neighbours}) must be < len(inputs) ({len(inputs)})")

            self.x0_ = x

            # Find nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=self.J_neighbours, metric=self.nn_metric).fit(inputs)
            _, indices = nbrs.kneighbors(x.reshape(1, -1))
            self.real_inputs_ = inputs[indices[0]]

            # Compute variance (add small epsilon to avoid division by zero)
            if var is None:
                self.var_ = np.var(self.real_inputs_, axis=0, ddof=1)/2
                # Add small epsilon to avoid zero variance
                self.var_ = np.maximum(self.var_, 1e-8)
            else:
                self.var_ = var

            # Generate synthetic samples (ensure float64 type)
            self.synthetic_samples_ = np.random.normal(loc=x, scale=np.sqrt(self.var_),
                                                       size=(self.num_synthetic, len(x))).astype(np.float64)
            sr_inputs = np.concatenate([self.real_inputs_, self.synthetic_samples_], axis=0).astype(np.float64)

            print(f"Fitting SLIME with {len(sr_inputs)} points ({len(self.real_inputs_)} real + {self.num_synthetic} synthetic)")
        else:
            print("Fitting SLIME with all inputs")
            self.x0_ = None
            self.real_inputs_ = inputs
            self.synthetic_samples_ = None
            sr_inputs = np.asarray(inputs, dtype=np.float64)

        sr_targets = f(sr_inputs).astype(np.float64)

        # Check for NaN or Inf values
        if np.isnan(sr_inputs).any():
            raise ValueError("NaN values detected in input data")
        if np.isnan(sr_targets).any():
            raise ValueError("NaN values detected in target predictions from black-box function")
        if np.isinf(sr_targets).any():
            raise ValueError("Inf values detected in target predictions from black-box function")

        final_pysr_params = {**DEFAULT_PYSR_PARAMS, **self.pysr_params}

        # Prepare fit parameters
        if fit_params is None:
            fit_params = {}

        # Add weights if synthetic samples exist
        if x is not None and self.synthetic_samples_ is not None:
            synthetic_distances_sq = np.sum((self.synthetic_samples_ - x)**2 / self.var_, axis=1)
            gaussian_weights = np.exp(-synthetic_distances_sq).astype(np.float64)
            self.weights_ = np.concatenate([
                np.full(len(self.real_inputs_), self.real_weighting, dtype=np.float64),
                gaussian_weights
            ])
            final_pysr_params['elementwise_loss'] = "loss(prediction, target, weight) = weight * (prediction - target)^2"
            fit_params['weights'] = self.weights_

        # Fit symbolic regression
        self.regressor_ = PySRRegressor(**final_pysr_params).fit(sr_inputs, sr_targets, **fit_params)

        return self

    def predict(self, X, complexity=None):
        """
        Predict using the discovered symbolic equation.

        Args:
            X (np.ndarray): Input samples
            complexity (int, optional): Equation complexity to use

        Returns:
            np.ndarray: Predictions
        """
        if self.regressor_ is None:
            raise ValueError("Model not fitted. Call fit() first")

        equation_func, equation_vars = regressor_to_function(self.regressor_, complexity=complexity)
        var_indices = [int(str(v).replace('x', '')) for v in equation_vars]

        X = np.atleast_2d(X)
        X_features = [X[:, idx] for idx in var_indices]

        return equation_func(*X_features)

    def get_equation(self, complexity=None):
        """Get the symbolic equation as a string."""
        if self.regressor_ is None:
            raise ValueError("Model not fitted. Call fit() first")

        if complexity is None:
            return self.regressor_.get_best()["equation"]
        else:
            matching = self.regressor_.equations_[self.regressor_.equations_["complexity"] == complexity]
            if matching.empty:
                avail = sorted(self.regressor_.equations_["complexity"].unique())
                raise ValueError(f"No equation with complexity {complexity}. Available: {avail}")
            return matching["equation"].values[0]

    @property
    def equations_(self):
        """Access the PySR equations DataFrame."""
        if self.regressor_ is None:
            raise ValueError("Model not fitted. Call fit() first")
        return self.regressor_.equations_
