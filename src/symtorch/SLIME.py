import numpy as np
from pysr import PySRRegressor
from sympy import lambdify

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


def SLIME(f, inputs, x=None, p_synthetic=0, var=None, pysr_params=None, fit_params=None):
    if x is not None:
        if p_synthetic == 0:
            raise ValueError("Need to set p_synthetic to non-zero if x is specified.")
        if var is None:
            var = np.var(inputs, axis=0, ddof=1)

        N = int(p_synthetic * len(inputs) / (1 - p_synthetic))
        samples = np.random.normal(loc=x, scale=np.sqrt(var), size=(N, len(x)))
        sr_inputs = np.concatenate([inputs, samples], axis=0)
    else:
        sr_inputs = inputs

    sr_targets = f(sr_inputs)

    if pysr_params is None:
        pysr_params = {}
    final_pysr_params = {**DEFAULT_PYSR_PARAMS, **pysr_params}

    pysr_model = PySRRegressor(**final_pysr_params)

    if fit_params is None:
        fit_params = {}

    regressor = pysr_model.fit(sr_inputs, sr_targets, **fit_params)

    return regressor