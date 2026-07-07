"""Distill pipeline pieces: block-level I/O resolution and variable transforms."""

import logging
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .caching import to_numpy

logger = logging.getLogger(__name__)


@contextmanager
def capture_layer_io(block, parent_model, inputs):
    """Capture (inputs, outputs) of `block` during a parent_model forward pass."""
    layer_inputs, layer_outputs = [], []

    def hook_fn(module, input, output):
        if module is block:
            layer_inputs.append(input[0].clone())
            layer_outputs.append(output.clone())

    hook = block.register_forward_hook(hook_fn)
    try:
        parent_model.eval()
        with torch.no_grad():
            _ = parent_model(inputs)
        yield layer_inputs, layer_outputs
    finally:
        hook.remove()


def resolve_io(block, inputs, parent_model):
    """Resolve block-level (raw_inputs, raw_outputs, eval_fn) for any mode.

    raw_inputs/raw_outputs keep their native type (torch tensors for modules)
    so user variable_transforms written against torch still work.
    eval_fn maps a numpy array through the block, returning numpy (SLIME).
    """
    if isinstance(block, nn.Module):
        if parent_model is not None:
            with capture_layer_io(block, parent_model, inputs) as (layer_inputs, layer_outputs):
                pass
            if not layer_inputs or not layer_outputs:
                raise RuntimeError(
                    "Failed to capture intermediate activations. "
                    "Ensure parent_model contains this SymbolicModel instance."
                )
            raw_inputs, raw_outputs = layer_inputs[0], layer_outputs[0]
        else:
            block.eval()
            with torch.no_grad():
                raw_outputs = block(inputs)
            raw_inputs = inputs

        device = raw_inputs.device if isinstance(raw_inputs, torch.Tensor) else "cpu"

        def eval_fn(arr):
            tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            block.eval()
            with torch.no_grad():
                return to_numpy(block(tensor))

        return raw_inputs, raw_outputs, eval_fn

    raw_outputs = block(inputs)

    def eval_fn(arr):
        return to_numpy(block(arr))

    return inputs, raw_outputs, eval_fn


def apply_variable_transforms(raw_inputs, variable_transforms, variable_names):
    """Apply feature-engineering transforms, returning an (N, n_transforms) numpy matrix."""
    if variable_names is not None and len(variable_names) != len(variable_transforms):
        raise ValueError(
            f"Length of variable_names ({len(variable_names)}) must match "
            f"length of variable_transforms ({len(variable_transforms)})"
        )

    columns = []
    for i, transform_func in enumerate(variable_transforms):
        try:
            col = to_numpy(transform_func(raw_inputs))
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error applying transformation {i}: {e}")
        if col.ndim > 1:
            col = col.flatten()
        columns.append(col)

    logger.info(f"🔄 Applied {len(variable_transforms)} variable transformations")
    if variable_names:
        logger.info(f"   Variable names: {variable_names}")
    return np.column_stack(columns)
