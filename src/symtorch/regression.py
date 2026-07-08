"""PySR interface: default parameters, parameter merging, and equation fitting."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from pysr import PySRRegressor

logger = logging.getLogger(__name__)

DEFAULT_SR_PARAMS = {
    "binary_operators": ["+", "*"],
    "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
    "extra_sympy_mappings": {"inv": lambda x: 1 / x},
    "niterations": 400,
    "complexity_of_operators": {"sin": 3, "exp": 3},
}


def create_sr_params(
    block_name: str,
    save_path: Optional[str],
    run_id: str,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge DEFAULT_SR_PARAMS with output location, run id, and user overrides."""
    output_name = f"SR_output/{block_name}"
    if save_path is not None:
        output_name = f"{save_path}/{block_name}"

    base_params = {**DEFAULT_SR_PARAMS, "output_directory": output_name, "run_id": run_id}
    if custom_params:
        base_params.update(custom_params)
    return base_params


def fit_single_dimension(
    X: np.ndarray,
    y: np.ndarray,
    block_name: str,
    save_path: Optional[str],
    dim: int,
    sr_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    timestamp: int,
) -> PySRRegressor:
    """Fit one PySRRegressor on a single target column (dedicated search)."""
    run_id = f"dim{dim}_{timestamp}"
    final_sr_params = create_sr_params(block_name, save_path, run_id, sr_params)
    regressor = PySRRegressor(**final_sr_params)
    regressor.fit(X, y, **dict(fit_params))
    return regressor


class DimensionView:
    """Per-dimension view onto a (possibly multi-output) PySRRegressor.

    Presents the single-dimension surface that SymbolicModel and user code
    consume — `equations_` as a DataFrame, `get_best()` as a single row,
    `predict()` as a 1-D array — while delegating everything else to the
    shared underlying regressor.
    """

    def __init__(self, regressor, column: int):
        self._regressor = regressor
        self._column = column

    @property
    def equations_(self):
        eqs = self._regressor.equations_
        return eqs[self._column] if isinstance(eqs, list) else eqs

    def get_best(self):
        best = self._regressor.get_best()
        return best[self._column] if isinstance(best, list) else best

    def predict(self, X):
        pred = np.asarray(self._regressor.predict(X))
        return pred[:, self._column] if pred.ndim > 1 else pred

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._regressor, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def fit_all_dimensions(X, Y, dims, block_name, save_path, sr_params, fit_params, timestamp):
    """Fit ALL requested dimensions in one multi-output PySR search.

    Returns {original_dim_index: DimensionView} so per-dimension access is
    preserved. A 1-D `weights` entry in fit_params (SLIME) is tiled to match
    Y's shape, as PySR requires weights shaped like the target matrix.
    """
    run_id = f"alldims_{timestamp}"
    final_sr_params = create_sr_params(block_name, save_path, run_id, sr_params)

    final_fit_params = dict(fit_params)
    weights = final_fit_params.get("weights")
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim == 1 and Y.ndim == 2 and Y.shape[1] > 1:
            final_fit_params["weights"] = np.tile(weights[:, None], (1, Y.shape[1]))

    regressor = PySRRegressor(**final_sr_params)
    regressor.fit(X, Y, **final_fit_params)
    return {dim: DimensionView(regressor, col) for col, dim in enumerate(dims)}
