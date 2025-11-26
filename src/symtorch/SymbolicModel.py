"""
SymTorch SymbolicModel Module

This module provides a wrapper for components of (or whole) ML models that adds symbolic regression
capabilities using PySR (Python Symbolic Regression).
"""
import warnings
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
from pysr import *
import torch 
import torch.nn as nn
import time
import sympy
from sympy import lambdify
import numpy as np
import os
import pickle
from typing import List, Callable, Optional, Union, Dict, Any
from contextlib import contextmanager


class SymbolicModel(nn.Module):

    # Default PySR parameters
    DEFAULT_SR_PARAMS = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "niterations": 400,
        "complexity_of_operators": {"sin": 3, "exp": 3}
    }

    def __init__(self, block: Union[nn.Module, Callable], block_name: str = None):

        super().__init__()
        self.symtorch_block = block 
        self.block_name = block_name or f"block_{id(self)}"

        if not block_name:
            print(f"No name specified for this block. Label is {self.block_name}.")

        self.pysr_regressor = {}

    def _create_sr_params(self, save_path: str, run_id: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create SR parameters by merging defaults with custom parameters.
        
        Args:
            save_path (str): Output directory path for SR results
            run_id (str): Unique run identifier
            custom_params (Dict[str, Any], optional): Custom parameters to override defaults
            
        Returns:
            Dict[str, Any]: Final SR parameters for PySRRegressor
        """
        output_name = f"SR_output/{self.block_name}"
        if save_path is not None:
            output_name = f"{save_path}/{self.block_name}"
        
        base_params = {
            **self.DEFAULT_SR_PARAMS,
            "output_directory": output_name,
            "run_id": run_id
        }
        
        if custom_params:
            base_params.update(custom_params)
            
        return base_params
    
    @contextmanager
    def _capture_layer_output(self, parent_model, inputs):
        """
        Context manager to capture inputs and outputs from this layer.

        Args:
            parent_model (nn.Module): Parent model containing this SymbolicMLP instance
            inputs (torch.Tensor): Input tensor to pass through parent model

        Yields:
            tuple: (layer_inputs, layer_outputs) lists containing captured tensors
        """
        layer_inputs = []
        layer_outputs = []
        
        def hook_fn(module, input, output):
            if module is self.symtorch_block: # Only captures layer data for the layers we want to distil
                layer_inputs.append(input[0].clone())
                layer_outputs.append(output.clone())
        
        # Register forward hook
        hook = self.symtorch_block.register_forward_hook(hook_fn)
        
        try:
            # Run parent model to capture intermediate activations
            parent_model.eval()
            with torch.no_grad():
                _ = parent_model(inputs)
            
            yield layer_inputs, layer_outputs
        finally:
            # Always remove hook
            hook.remove()
    
    def _extract_variables_for_equation(self, x: torch.Tensor, var_indices: List[int], dim: int) -> List[torch.Tensor]:
        """
        Extract and transform variables needed for a specific equation dimension.
        Each output dimension may only depend on a subset of the input variables.
        
        Args:
            x (torch.Tensor): Input tensor
            var_indices (List[int]): List of variable indices needed
            dim (int): Output dimension being processed
            
        Returns:
            List[torch.Tensor]: List of extracted/transformed variables
            
        Raises:
            ValueError: If required variables/transforms are not available
        """
        selected_inputs = []
        
        if hasattr(self, '_variable_transforms') and self._variable_transforms is not None:
            # Apply transformations and select needed variables
            for idx in var_indices:
                if idx < len(self._variable_transforms):
                    transformed_var = self._variable_transforms[idx](x)
                    if transformed_var.dim() > 1:
                        transformed_var = transformed_var.flatten()
                    selected_inputs.append(transformed_var)
                else:
                    raise ValueError(f"Equation for dimension {dim} requires transform {idx} but only {len(self._variable_transforms)} transforms available")
        else:
            # Original behavior - extract by column index
            for idx in var_indices:
                if idx < x.shape[1]:
                    selected_inputs.append(x[:, idx])
                else:
                    raise ValueError(f"Equation for dimension {dim} requires variable x{idx} but input only has {x.shape[1]} dimensions")
        
        return selected_inputs
    
    def _map_variables_to_indices(self, vars_sorted: List, dim: int) -> List[int]:
        """
        Map symbolic variables to their corresponding indices.
        Method used during the forward pass when the model is in equation mode to determine 
        which input columns/transforms to extract and pass to each discovered symbolic equation.
        
        Args:
            vars_sorted (List): List of symbolic variables from equation
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
            if hasattr(self, '_variable_names') and self._variable_names:
                try:
                    idx = self._variable_names.index(var_str)
                except ValueError:
                    pass  # Variable not found in custom names, try other methods
            
            # If not found in custom names, try default x0, x1, etc. format
            if idx is None and var_str.startswith('x'):
                try:
                    idx = int(var_str[1:])
                    # With transforms, validate index is within range
                    if hasattr(self, '_variable_transforms') and self._variable_transforms is not None:
                        if idx >= len(self._variable_transforms):
                            raise ValueError(f"Variable {var_str} index {idx} exceeds available transforms ({len(self._variable_transforms)}) for dimension {dim}")
                except ValueError as e:
                    if "exceeds available transforms" in str(e):
                        raise e
                    pass  # Not a valid x-numbered variable
            
            if idx is None:
                error_msg = f"Could not map variable '{var_str}' for dimension {dim}"
                if hasattr(self, '_variable_names') and self._variable_names:
                    error_msg += f"\n   Available custom names: {self._variable_names}"
                if hasattr(self, '_variable_transforms') and self._variable_transforms is not None:
                    error_msg += f"\n   Available transforms: {len(self._variable_transforms)}"
                else:
                    error_msg += f"\n   Expected format: x0, x1, x2, etc."
                raise ValueError(error_msg)
            
            var_indices.append(idx)
        
        return var_indices
    
    def distill(self, inputs, output_dim: int = None, parent_model=None,
                 variable_transforms: Optional[List[Callable]] = None,
                 save_path: str = None,
                 sr_params: Optional[Dict[str, Any]] = None,
                 fit_params: Optional[Dict[str, Any]] = None):
        
        if isinstance(self.symtorch_block, Callable) and not isinstance(self.symtorch_block, nn.Module) and parent_model is not None:
            raise ValueError(
                "Cannot use parent_model with Callable functions. "
                "Hooks are only supported for nn.Module objects. "
                "Please call distill() without parent_model argument and pass inputs directly to the function."
            )
        # Extract inputs and outputs at this layer level
        if isinstance(self.symtorch_block, nn.Module):

            if parent_model is not None:
                with self._capture_layer_output(parent_model, inputs) as (layer_inputs, layer_outputs):
                    pass
                
                # Use captured intermediate data
                if layer_inputs and layer_outputs:
                    actual_inputs = layer_inputs[0]
                    output = layer_outputs[0]
                else:
                    raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicModel instance.")
        
            else:
                # Original behavior - use block directly 
                actual_inputs = inputs
                self.symtorch_block.eval()
                with torch.no_grad():
                    output = self.symtorch_block(inputs)

            # Extract fit parameters
            if fit_params is None:
                fit_params = {}

            variable_names = fit_params.get('variable_names', None)

            # Extract sr_params with defaults
            if sr_params is None:
                sr_params = {}
            
            # Apply variable transformations if provided
            if variable_transforms is not None:
                # Validate inputs - variable_names is optional
                if variable_names is not None and len(variable_names) != len(variable_transforms):
                    raise ValueError(f"Length of variable_names ({len(variable_names)}) must match length of variable_transforms ({len(variable_transforms)})")
                
                # Apply transformations
                transformed_inputs = []
                for i, transform_func in enumerate(variable_transforms):
                    try:
                        transformed_var = transform_func(actual_inputs)
                        # Ensure the result is 1D (batch_size,)
                        if transformed_var.dim() > 1:
                            transformed_var = transformed_var.flatten()
                        transformed_inputs.append(transformed_var.detach().cpu().numpy())
                    except Exception as e:
                        raise ValueError(f"Error applying transformation {i}: {e}")
                
                # Stack transformed variables into input matrix
                actual_inputs_numpy = np.column_stack(transformed_inputs)
                
                # Store transformation info for later use in switch_to_equation
                self._variable_transforms = variable_transforms
                self._variable_names = variable_names
                
                print(f"🔄 Applied {len(variable_transforms)} variable transformations")
                if variable_names:
                    print(f"   Variable names: {variable_names}")
            else:
                # Use original inputs
                actual_inputs_numpy = actual_inputs.detach().cpu().numpy()
                self._variable_transforms = None
                # Still store variable names even without transforms for switch_to_equation
                self._variable_names = variable_names

            timestamp = int(time.time())

            output_dims = output.shape[1] # Number of output dimensions
            self.output_dims = output_dims # Save this 

            pysr_regressors = {}

            if not output_dim:
                #If output dimension is not specified, run SR on all dims

                for dim in range(output_dims):

                    print(f"🛠️ Running SR on output dimension {dim} of {output_dims-1}")
            
                    run_id = f"dim{dim}_{timestamp}"
                    final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
                    regressor = PySRRegressor(**final_sr_params)

                    # Prepare fit arguments
                    fit_args = [actual_inputs_numpy, output.detach()[:, dim].cpu().numpy()]
                    final_fit_params = dict(fit_params)  # Copy to avoid modifying original
                    
                    regressor.fit(*fit_args, **final_fit_params)

                    pysr_regressors[dim] = regressor

                    print(f"💡Best equation for output {dim} found to be {regressor.get_best()['equation']}.")
            
            else:
                
                print(f"🛠️ Running SR on output dimension {output_dim}.")

                run_id = f"dim{output_dim}_{timestamp}"
                final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
                regressor = PySRRegressor(**final_sr_params)

                # Prepare fit arguments
                fit_args = [actual_inputs_numpy, output.detach()[:, output_dim].cpu().numpy()]
                final_fit_params = dict(fit_params)  # Copy to avoid modifying original
                
                regressor.fit(*fit_args, **final_fit_params)
                pysr_regressors[output_dim] = regressor

                print(f"💡Best equation for output {output_dim} found to be {regressor.get_best()['equation']}.")

            print(f"❤️ SR on {self.block_name} complete.")
            self.pysr_regressor = self.pysr_regressor | pysr_regressors
            
            # For backward compatibility, return the regressor or dict of regressors
            if output_dim is not None:
                return pysr_regressors[output_dim]
            else:
                return pysr_regressors
            
        else: #code for Callable function
            # Extract fit parameters
            if fit_params is None:
                fit_params = {}

            variable_names = fit_params.get('variable_names', None)

            # Extract sr_params with defaults
            if sr_params is None:
                sr_params = {}

            # Convert inputs to numpy if needed
            if hasattr(inputs, 'detach'):  # torch tensor
                inputs_np = inputs.detach().cpu().numpy()
            else:
                inputs_np = np.array(inputs)

            # Get outputs from the black-box function
            outputs_raw = self.symtorch_block(inputs)
            if hasattr(outputs_raw, 'detach'):  # torch tensor
                outputs_np = outputs_raw.detach().cpu().numpy()
            else:
                outputs_np = np.array(outputs_raw)

            # Apply variable transformations if provided
            if variable_transforms is not None:
                # Validate inputs
                if variable_names is not None and len(variable_names) != len(variable_transforms):
                    raise ValueError(f"Length of variable_names ({len(variable_names)}) must match length of variable_transforms ({len(variable_transforms)})")

                # Apply transformations
                transformed_inputs = []
                for i, transform_func in enumerate(variable_transforms):
                    try:
                        # Handle both numpy and torch inputs
                        if isinstance(inputs, torch.Tensor):
                            transformed_var = transform_func(inputs)
                            if hasattr(transformed_var, 'detach'):
                                transformed_var = transformed_var.detach().cpu().numpy()
                            else:
                                transformed_var = np.array(transformed_var)
                        else:
                            transformed_var = transform_func(inputs_np)
                            transformed_var = np.array(transformed_var)

                        # Ensure the result is 1D (batch_size,)
                        if transformed_var.ndim > 1:
                            transformed_var = transformed_var.flatten()
                        transformed_inputs.append(transformed_var)
                    except Exception as e:
                        raise ValueError(f"Error applying transformation {i}: {e}")

                # Stack transformed variables into input matrix
                inputs_np = np.column_stack(transformed_inputs)

                # Store transformation info for later use in switch_to_equation
                self._variable_transforms = variable_transforms
                self._variable_names = variable_names

                print(f"🔄 Applied {len(variable_transforms)} variable transformations")
                if variable_names:
                    print(f"   Variable names: {variable_names}")
            else:
                # No transforms used
                self._variable_transforms = None
                # Still store variable names even without transforms for switch_to_equation
                self._variable_names = variable_names

            # Handle both 1D and 2D outputs
            if outputs_np.ndim == 1:
                outputs_np = outputs_np.reshape(-1, 1)

            output_dims = outputs_np.shape[1]  # Number of output dimensions
            self.output_dims = output_dims  # Save this
            timestamp = int(time.time())

            # Use dict for consistency with nn.Module branch
            pysr_regressors = {}

            if output_dim is None:
                # Run on all output dimensions
                for dim in range(output_dims):
                    print(f"🛠️ Running SR on output dimension {dim} of {output_dims-1}")

                    run_id = f"dim{dim}_{timestamp}"
                    final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
                    regressor = PySRRegressor(**final_sr_params)

                    # Prepare fit arguments
                    fit_args = [inputs_np, outputs_np[:, dim]]
                    final_fit_params = dict(fit_params)  # Copy to avoid modifying original

                    regressor.fit(*fit_args, **final_fit_params)

                    pysr_regressors[dim] = regressor

                    print(f"💡Best equation for output {dim} found to be {regressor.get_best()['equation']}.")

            else:
                # Run on specific output dimension
                if output_dim >= output_dims:
                    raise ValueError(f"output_dim {output_dim} is out of range for outputs with {output_dims} dimensions")

                print(f"🛠️ Running SR on output dimension {output_dim}.")

                run_id = f"dim{output_dim}_{timestamp}"
                final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
                regressor = PySRRegressor(**final_sr_params)

                # Prepare fit arguments
                fit_args = [inputs_np, outputs_np[:, output_dim]]
                final_fit_params = dict(fit_params)  # Copy to avoid modifying original

                regressor.fit(*fit_args, **final_fit_params)

                pysr_regressors[output_dim] = regressor

                print(f"💡Best equation for output {output_dim} found to be {regressor.get_best()['equation']}.")

            print(f"❤️ SR on {self.block_name} complete.")
            self.pysr_regressor = self.pysr_regressor | pysr_regressors

            # For backward compatibility, return the regressor or dict of regressors
            if output_dim is not None:
                return pysr_regressors[output_dim]
            else:
                return pysr_regressors
