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
