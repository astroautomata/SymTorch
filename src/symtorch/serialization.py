"""state_dict save/load for SymbolicModel extras (regressors, metadata, transforms)."""

import copy
import logging
import warnings

import dill
import torch

logger = logging.getLogger(__name__)


def save_symtorch_state(model, destination, prefix):
    # Note: We DO save _original_block if it exists (needed for switch_to_block())
    # Save metadata
    metadata = {
        "block_name": model.block_name,
        "output_dims": getattr(model, "output_dims", None),
        "_variable_names": getattr(model, "_variable_names", None),
        "_using_equation": getattr(model, "_using_equation", False),
        "_equation_vars": getattr(model, "_equation_vars", {}),
    }

    # Try to serialize variable transforms with dill
    if hasattr(model, "_variable_transforms") and model._variable_transforms is not None:
        try:
            metadata["_variable_transforms"] = dill.dumps(model._variable_transforms)
            metadata["_variable_transforms_serialized"] = True
        except Exception as e:
            warnings.warn(
                f"Could not serialize variable transforms for '{model.block_name}': {e}. "
                "Transforms will need to be re-provided after loading."
            )
            metadata["_variable_transforms_serialized"] = False
    else:
        metadata["_variable_transforms_serialized"] = False

    # Add pruning metadata if present
    if hasattr(model, "pruning_schedule") and model.pruning_schedule is not None:
        metadata.update(
            {
                "initial_dim": model.initial_dim,
                "target_dim": model.target_dim,
                "current_dim": model.current_dim,
                "pruning_schedule": model.pruning_schedule,
            }
        )

    destination[prefix + "_symtorch_metadata"] = metadata

    # Save PySR regressors (serialize with dill)
    if hasattr(model, "pysr_regressor") and model.pysr_regressor:
        for dim, regressor in model.pysr_regressor.items():
            key = f"_pysr_regressor_dim_{dim}"
            try:
                destination[prefix + key] = dill.dumps(regressor)
            except Exception as e:
                warnings.warn(f"Could not serialize PySR regressor for dimension {dim}: {e}")

    # Save SLIME PySR regressors
    if hasattr(model, "SLIME_pysr_regressor") and model.SLIME_pysr_regressor:
        for dim, regressor in model.SLIME_pysr_regressor.items():
            key = f"_slime_regressor_dim_{dim}"
            try:
                destination[prefix + key] = dill.dumps(regressor)
            except Exception as e:
                warnings.warn(f"Could not serialize SLIME regressor for dimension {dim}: {e}")

    # Store list of regressor dimensions for easier reconstruction
    destination[prefix + "_pysr_dims"] = list(model.pysr_regressor.keys()) if hasattr(model, "pysr_regressor") else []
    destination[prefix + "_slime_dims"] = (
        list(model.SLIME_pysr_regressor.keys()) if hasattr(model, "SLIME_pysr_regressor") else []
    )


def load_symtorch_extras(model, state_dict, prefix, error_msgs):
    # Load metadata first
    metadata_key = prefix + "_symtorch_metadata"
    if metadata_key in state_dict:
        metadata = state_dict.pop(metadata_key)

        # Restore basic metadata
        model.block_name = metadata.get("block_name", model.block_name)
        model.output_dims = metadata.get("output_dims")
        model._variable_names = metadata.get("_variable_names")
        model._using_equation = metadata.get("_using_equation", False)
        model._equation_vars = metadata.get("_equation_vars", {})

        # Restore pruning metadata if present
        if "initial_dim" in metadata:
            model.initial_dim = metadata["initial_dim"]
            model.target_dim = metadata["target_dim"]
            model.current_dim = metadata["current_dim"]
            model.pruning_schedule = metadata["pruning_schedule"]

            # Register pruning_mask buffer if not already registered
            # This allows loading models with pruning without calling setup_pruning first
            if not hasattr(model, "pruning_mask"):
                model.register_buffer("pruning_mask", torch.ones(model.initial_dim, dtype=torch.bool))

        # Restore variable transforms if they were serialized
        if metadata.get("_variable_transforms_serialized", False):
            try:
                model._variable_transforms = dill.loads(metadata["_variable_transforms"])
            except Exception as e:
                warnings.warn(
                    f"Could not deserialize variable transforms for '{model.block_name}': {e}. "
                    "You must re-provide variable_transforms if you need to use equation mode."
                )
                model._variable_transforms = None
        else:
            model._variable_transforms = None

    # Load PySR regressors
    pysr_dims_key = prefix + "_pysr_dims"
    if pysr_dims_key in state_dict:
        pysr_dims = state_dict.pop(pysr_dims_key)
        model.pysr_regressor = {}

        for dim in pysr_dims:
            key = prefix + f"_pysr_regressor_dim_{dim}"
            if key in state_dict:
                try:
                    model.pysr_regressor[dim] = dill.loads(state_dict.pop(key))
                except Exception as e:
                    error_msgs.append(f"Could not load PySR regressor for dimension {dim}: {e}")
    else:
        model.pysr_regressor = {}

    # Load SLIME regressors
    slime_dims_key = prefix + "_slime_dims"
    if slime_dims_key in state_dict:
        slime_dims = state_dict.pop(slime_dims_key)
        model.SLIME_pysr_regressor = {}

        for dim in slime_dims:
            key = prefix + f"_slime_regressor_dim_{dim}"
            if key in state_dict:
                try:
                    model.SLIME_pysr_regressor[dim] = dill.loads(state_dict.pop(key))
                except Exception as e:
                    error_msgs.append(f"Could not load SLIME regressor for dimension {dim}: {e}")
    else:
        model.SLIME_pysr_regressor = {}

    # Initialize cache as None (not serialized)
    model.distill_data = None
    model.distill_data_slime = None

    # Check if state_dict contains _original_block (means model was in equation mode)
    has_original_block = any(key.startswith(prefix + "_original_block.") for key in state_dict.keys())

    # _original_block IS saved automatically by PyTorch (it's an nn.Module).
    # We create a placeholder here BEFORE calling parent's load_state_dict so PyTorch
    # knows where to load the saved _original_block weights. Without this, strict mode
    # would complain about unexpected keys in the state dict.
    if has_original_block and model._using_equation:
        # Create a placeholder _original_block that will be populated by parent's load
        model._original_block = copy.deepcopy(model.symtorch_block)
