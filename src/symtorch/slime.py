"""SLIME local-interpretability sampling: nearest neighbors + Gaussian synthetic points."""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

DEFAULT_SLIME_PARAMS = {
    "x": None,  # Point of interest for local explanation
    "J_nn": 10,  # Number of nearest neighbors
    "num_synthetic": 100,  # Number of synthetic samples
    "real_weighting": 1.0,  # Weight for real samples vs synthetic
    "nn_metric": "euclidean",  # Distance metric for nearest neighbors
    "var": None,  # Variance for perturbations (auto-computed if None)
}


def merge_slime_params(slime_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**DEFAULT_SLIME_PARAMS, **(slime_params or {})}


def apply_slime_sampling(inputs_np, function_to_call, slime_params, sr_params, fit_params):
    """
    Apply SLIME sampling to create a local dataset around a point of interest.

    Args:
        inputs_np (np.ndarray): Input data
        function_to_call (Callable): Function to evaluate outputs (block or callable)
        slime_params (Dict): SLIME parameters
        sr_params (Dict): SR parameters (will be modified with weighted loss)
        fit_params (Dict): Fit parameters (will be modified with weights)

    Returns:
        tuple: (sampled_inputs, sampled_outputs, updated_sr_params, updated_fit_params)
    """
    # Merge default SLIME params with user-provided params
    final_slime_params = merge_slime_params(slime_params)

    x0 = final_slime_params["x"]
    J_nn = final_slime_params["J_nn"]
    num_synthetic = final_slime_params["num_synthetic"]
    real_weighting = final_slime_params["real_weighting"]
    nn_metric = final_slime_params["nn_metric"]
    var = final_slime_params["var"]

    # Validation
    if real_weighting != 1.0 and num_synthetic == 0:
        warnings.warn("real_weighting only works with num_synthetic > 0. Setting to 1.0", UserWarning)
        real_weighting = 1.0

    if x0 is not None:
        if num_synthetic == 0:
            raise ValueError("num_synthetic must be > 0 when x is specified in SLIME mode")
        if J_nn >= len(inputs_np):
            raise ValueError(f"J_nn ({J_nn}) must be < len(inputs) ({len(inputs_np)})")

        # Convert x0 to numpy if needed
        if isinstance(x0, torch.Tensor):
            x0 = x0.detach().cpu().numpy()
        x0 = np.array(x0)

        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=J_nn, metric=nn_metric).fit(inputs_np)
        _, indices = nbrs.kneighbors(x0.reshape(1, -1))
        real_inputs = inputs_np[indices[0]]

        # Compute variance
        if var is None:
            var_computed = np.var(real_inputs, axis=0, ddof=1) / 2
            var_computed = np.maximum(var_computed, 1e-8)  # Avoid zero variance
        else:
            var_computed = var

        # Generate synthetic samples
        synthetic_samples = np.random.normal(loc=x0, scale=np.sqrt(var_computed), size=(num_synthetic, len(x0))).astype(
            np.float64
        )

        # Combine real and synthetic inputs
        sr_inputs_slime = np.concatenate([real_inputs, synthetic_samples], axis=0).astype(np.float64)

        # Get outputs for SLIME samples
        slime_outputs = function_to_call(sr_inputs_slime)

        # Prepare weights
        synthetic_distances_sq = np.sum((synthetic_samples - x0) ** 2 / var_computed, axis=1)
        gaussian_weights = np.exp(-synthetic_distances_sq).astype(np.float64)
        slime_weights = np.concatenate([np.full(len(real_inputs), real_weighting, dtype=np.float64), gaussian_weights])

        # Update sr_params with weighted loss
        if sr_params is None:
            sr_params = {}
        sr_params = sr_params.copy()
        sr_params["elementwise_loss"] = "loss(prediction, target, weight) = weight * (prediction - target)^2"

        # Update fit_params with weights
        if fit_params is None:
            fit_params = {}
        fit_params = fit_params.copy()
        fit_params["weights"] = slime_weights

        logger.info(
            f"🔍 SLIME mode: Using {len(sr_inputs_slime)} points ({len(real_inputs)} real + {num_synthetic} synthetic)"
        )
        logger.info(f"   Point of interest: {x0}")

        return sr_inputs_slime, slime_outputs, sr_params, fit_params
    else:
        # Global SLIME (no local focus)
        logger.info("🔍 SLIME mode: Global (no local focus point)")
        return inputs_np, function_to_call(inputs_np), sr_params, fit_params
