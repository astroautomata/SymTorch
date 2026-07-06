"""Symbolic expression handling: selection, lambdification, variable mapping."""
import logging
from typing import List, Optional

import torch
from sympy import lambdify

logger = logging.getLogger(__name__)


def select_expression(regressor, complexity: Optional[int] = None):
    if complexity is None:
        best_str = regressor.get_best()["equation"]
        return regressor.equations_.loc[
            regressor.equations_["equation"] == best_str, "sympy_format"
        ].values[0]
    matching_rows = regressor.equations_[regressor.equations_["complexity"] == complexity]
    if matching_rows.empty:
        available = sorted(regressor.equations_["complexity"].unique())
        logger.warning(f"⚠️ No equation found with complexity {complexity}. Available: {available}")
        return None
    return matching_rows["sympy_format"].values[0]


def expression_to_callable(expr):
    vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))
    try:
        return lambdify(vars_sorted, expr, "torch"), vars_sorted
    except Exception as e:
        raise RuntimeError(f"Could not create lambdify function: {e}")


def map_variables_to_indices(vars_sorted: List, variable_names, variable_transforms, dim: int) -> List[int]:
    """
    Map symbolic variables to their corresponding indices.
    Used during the forward pass when the model is in equation mode to determine
    which input columns/transforms to extract and pass to each discovered symbolic equation.

    Args:
        vars_sorted (List): List of symbolic variables from equation
        variable_names: Custom variable names (or None)
        variable_transforms: Variable transform callables (or None)
        dim (int): Output dimension being processed

    Returns:
        List[int]: List of variable indices

    Raises:
        ValueError: If variables cannot be mapped to indices
    """
    var_indices = []

    for var in vars_sorted:
        var_str = str(var)
        idx = None

        # Try to match with custom variable names first
        if variable_names:
            try:
                idx = variable_names.index(var_str)
            except ValueError:
                pass  # Variable not found in custom names, try other methods

        # If not found in custom names, try default x0, x1, etc. format
        if idx is None and var_str.startswith("x"):
            try:
                idx = int(var_str[1:])
                # With transforms, validate index is within range
                if variable_transforms is not None:
                    if idx >= len(variable_transforms):
                        raise ValueError(
                            f"Variable {var_str} index {idx} exceeds available transforms ({len(variable_transforms)}) for dimension {dim}"
                        )
            except ValueError as e:
                if "exceeds available transforms" in str(e):
                    raise e
                pass  # Not a valid x-numbered variable

        if idx is None:
            error_msg = f"Could not map variable '{var_str}' for dimension {dim}"
            if variable_names:
                error_msg += f"\n   Available custom names: {variable_names}"
            if variable_transforms is not None:
                error_msg += f"\n   Available transforms: {len(variable_transforms)}"
            else:
                error_msg += "\n   Expected format: x0, x1, x2, etc."
            raise ValueError(error_msg)

        var_indices.append(idx)

    return var_indices


def extract_variables_for_equation(
    x: torch.Tensor, var_indices: List[int], variable_transforms, dim: int
) -> List[torch.Tensor]:
    """
    Extract and transform variables needed for a specific equation dimension.
    Each output dimension may only depend on a subset of the input variables.

    Args:
        x (torch.Tensor): Input tensor
        var_indices (List[int]): List of variable indices needed
        variable_transforms: Variable transform callables (or None)
        dim (int): Output dimension being processed

    Returns:
        List[torch.Tensor]: List of extracted/transformed variables

    Raises:
        ValueError: If required variables/transforms are not available
    """
    selected_inputs = []

    if variable_transforms is not None:
        # Apply transformations and select needed variables
        for idx in var_indices:
            if idx < len(variable_transforms):
                transformed_var = variable_transforms[idx](x)
                if transformed_var.dim() > 1:
                    transformed_var = transformed_var.flatten()
                selected_inputs.append(transformed_var)
            else:
                raise ValueError(
                    f"Equation for dimension {dim} requires transform {idx} but only {len(variable_transforms)} transforms available"
                )
    else:
        # Original behavior - extract by column index
        for idx in var_indices:
            if idx < x.shape[1]:
                selected_inputs.append(x[:, idx])
            else:
                raise ValueError(
                    f"Equation for dimension {dim} requires variable x{idx} but input only has {x.shape[1]} dimensions"
                )

    return selected_inputs
