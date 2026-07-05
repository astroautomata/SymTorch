"""I/O caching for distill calls: avoid redundant forward passes across re-runs."""

import numpy as np
import torch

from .slime import merge_slime_params


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def check_cache_hit(cache, inputs, parent_model, SLIME, slime_params):
    """Return (hit, sr_inputs, sr_outputs); sr_* are None on a miss."""
    if cache is None:
        return False, None, None

    inputs_np = to_numpy(inputs)
    if not np.array_equal(inputs_np, cache["inputs"]):
        return False, None, None
    if cache["parent_model"] is not parent_model:
        return False, None, None

    if SLIME:
        final_slime_params = merge_slime_params(slime_params)
        cached_slime_params = cache["slime_params"]
        for key in final_slime_params:
            if key == "x":
                cached_x = cached_slime_params.get("x")
                current_x = final_slime_params.get("x")
                if isinstance(cached_x, torch.Tensor):
                    cached_x = cached_x.detach().cpu().numpy()
                if isinstance(current_x, torch.Tensor):
                    current_x = current_x.detach().cpu().numpy()
                if cached_x is None and current_x is None:
                    continue
                if cached_x is None or current_x is None:
                    return False, None, None
                if not np.array_equal(np.array(cached_x), np.array(current_x)):
                    return False, None, None
            elif cached_slime_params.get(key) != final_slime_params.get(key):
                return False, None, None

    return True, cache["sr_inputs"], cache["sr_outputs"]


def build_cache_entry(
    inputs, sr_inputs, sr_outputs, parent_model, slime_params=None, slime_weights=None, slime_loss=None
):
    entry = {
        "inputs": to_numpy(inputs),
        "sr_inputs": sr_inputs,
        "sr_outputs": to_numpy(sr_outputs),
        "parent_model": parent_model,
    }
    if slime_params is not None:
        entry["slime_params"] = merge_slime_params(slime_params)
        entry["slime_weights"] = slime_weights
        entry["slime_loss"] = slime_loss
    return entry
