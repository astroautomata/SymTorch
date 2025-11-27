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
from typing import Literal
import math


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
                    full_output = layer_outputs[0]
                else:
                    raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicModel instance.")

            else:
                # Original behavior - use block directly
                actual_inputs = inputs
                self.symtorch_block.eval()
                with torch.no_grad():
                    full_output = self.symtorch_block(inputs)

            # Check if pruning is enabled and filter to active dimensions
            if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                active_dims = self.get_active_dimensions()
                if not active_dims:
                    print("❗No active dimensions to distill!")
                    return {}

                # Filter to active dimensions only
                output = full_output[:, self.pruning_mask]

                # Filter active dimensions based on output_dim parameter
                if output_dim is not None:
                    if output_dim not in active_dims:
                        print(f"❗Requested output dimension {output_dim} is not active. Active dimensions: {active_dims}")
                        return {}
                    target_dims = [output_dim]
                else:
                    target_dims = active_dims
            else:
                # No pruning - use full output
                output = full_output
                target_dims = None  # Will process all dimensions

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
                
                # Store transformation info for later use in switch_to_symbolic
                self._variable_transforms = variable_transforms
                self._variable_names = variable_names
                
                print(f"🔄 Applied {len(variable_transforms)} variable transformations")
                if variable_names:
                    print(f"   Variable names: {variable_names}")
            else:
                # Use original inputs
                actual_inputs_numpy = actual_inputs.detach().cpu().numpy()
                self._variable_transforms = None
                # Still store variable names even without transforms for switch_to_symbolic
                self._variable_names = variable_names

            timestamp = int(time.time())

            pysr_regressors = {}

            # Handle pruning mode or standard mode
            if target_dims is not None:
                # Pruning mode - target_dims contains the list of active dimensions to process
                # Set output_dims to initial_dim for compatibility
                self.output_dims = self.initial_dim

                for i, dim_idx in enumerate(target_dims):
                    print(f"🛠️ Running SR on active dimension {dim_idx} ({i+1}/{len(target_dims)})")

                    run_id = f"dim{dim_idx}_{timestamp}"
                    final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
                    regressor = PySRRegressor(**final_sr_params)

                    # Find the index of this dimension in the active output
                    active_dims = self.get_active_dimensions()
                    active_dim_index = active_dims.index(dim_idx)

                    # Prepare fit arguments
                    fit_args = [actual_inputs_numpy, output[:, active_dim_index].detach().cpu().numpy()]
                    final_fit_params = dict(fit_params)  # Copy to avoid modifying original

                    regressor.fit(*fit_args, **final_fit_params)

                    pysr_regressors[dim_idx] = regressor

                    print(f"💡Best equation for active dimension {dim_idx}: {regressor.get_best()['equation']}.")

                print(f"❤️ SR on {self.block_name} active dimensions complete.")
            else:
                # Standard mode - no pruning
                output_dims = output.shape[1]  # Number of output dimensions
                self.output_dims = output_dims  # Save this

                if not output_dim:
                    # If output dimension is not specified, run SR on all dims
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
                return pysr_regressors.get(output_dim)
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

                # Store transformation info for later use in switch_to_symbolic
                self._variable_transforms = variable_transforms
                self._variable_names = variable_names

                print(f"🔄 Applied {len(variable_transforms)} variable transformations")
                if variable_names:
                    print(f"   Variable names: {variable_names}")
            else:
                # No transforms used
                self._variable_transforms = None
                # Still store variable names even without transforms for switch_to_symbolic
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
            
    def _get_equation(self, dim, complexity: int = None):
        """
        Extract symbolic equation function from fitted regressor.
        
        Converts the symbolic expression from PySR into a callable function
        that can be used for prediction.
        
        Args:
            dim (int): Output dimension to get equation for.
            complexity (int, optional): Specific complexity level to retrieve.
                                      If None, returns the best overall equation.
                                      
        Returns:
            tuple or None: (equation_function, sorted_variables) if successful,
                          None if no equation found or complexity not available
                          

        Note:
            This is an internal method. Use switch_to_symbolic() for public API.
        """
        if not hasattr(self, 'pysr_regressor') or self.pysr_regressor is None:
            print("❗No equations found for this block yet. You need to first run .distill to find the best equation to fit this block.")
            return None
        if dim not in self.pysr_regressor:
            print(f"❗No equation found for output dimension {dim}. You need to first run .distill.")
            return None

        regressor = self.pysr_regressor[dim]
        
        if complexity is None:
            best_str = regressor.get_best()["equation"] 
            expr = regressor.equations_.loc[regressor.equations_["equation"] == best_str, "sympy_format"].values[0]
        else:
            matching_rows = regressor.equations_[regressor.equations_["complexity"] == complexity]
            if matching_rows.empty:
                available_complexities = sorted(regressor.equations_["complexity"].unique())
                print(f"⚠️ Warning: No equation found with complexity {complexity} for dimension {dim}. Available complexities: {available_complexities}")
                return None
            expr = matching_rows["sympy_format"].values[0]

        vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))
        try:
            f = lambdify(vars_sorted, expr, "numpy")
            return f, vars_sorted
        except Exception as e:
            print(f"⚠️ Warning: Could not create lambdify function for dimension {dim}: {e}")
            return None

    def switch_to_symbolic(self, complexity: list = None):
        """
        Switch the forward pass from model block to symbolic equations for all output dimensions.

        After calling this method, the model will use the discovered symbolic
        expressions instead of the neural network for forward passes.

        For pruned models, only active dimensions need equations. Inactive dimensions
        will output zeros.

        Args:
            complexity (list, optional): Specific complexity levels to use for each dimension.
                                      If None, uses the best overall equation for each dimension.

        Example:
            >>> model.switch_to_symbolic(complexity=5)

        """
        if not hasattr(self, 'pysr_regressor') or not self.pysr_regressor:
            print("❗No equations found for this block yet. You need to first run .distill.")
            return

        if not hasattr(self, 'output_dims'):
            print("❗No output dimension information found. You need to first run .distill.")
            return

        # Check if pruning is enabled
        if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
            # Pruning mode - only need equations for active dimensions
            active_dims = self.get_active_dimensions()
            if not active_dims:
                print("❗No active dimensions to switch to equations.")
                return

            # Check that we have equations for all active dimensions
            missing_dims = []
            for dim in active_dims:
                if dim not in self.pysr_regressor:
                    missing_dims.append(dim)

            if missing_dims:
                print(f"❗Missing equations for active dimensions {missing_dims}. You need to run .distill on all active dimensions first.")
                return

            dimensions_to_process = active_dims
        else:
            # Standard mode - need equations for all dimensions
            missing_dims = []
            for dim in range(self.output_dims):
                if dim not in self.pysr_regressor:
                    missing_dims.append(dim)

            if missing_dims:
                print(f"❗Missing equations for dimensions {missing_dims}. You need to run .distill on all output dimensions first.")
                print(f"Available dimensions: {list(self.pysr_regressor.keys())}")
                print(f"Required dimensions: {list(range(self.output_dims))}")
                return

            dimensions_to_process = list(range(self.output_dims))

        # Store original block for potential restoration
        if not hasattr(self, '_original_block'):
            self._original_block = self.symtorch_block

        # Get equations for dimensions to process
        equation_funcs = {}
        equation_vars = {}
        equation_strs = {}

        for i, dim in enumerate(dimensions_to_process):
            # Get complexity for this specific dimension
            dim_complexity = None
            if complexity is not None:
                if isinstance(complexity, list):
                    if i < len(complexity):
                        dim_complexity = complexity[i]
                    else:
                        print(f"⚠️ Warning: Not enough complexity values provided. Using default for dimension {dim}")
                else:
                    # If complexity is a single value, use it for all dimensions
                    dim_complexity = complexity

            result = self._get_equation(dim, dim_complexity)
            if result is None:
                print(f"⚠️ Failed to get equation for dimension {dim}")
                return

            f, vars_sorted = result

            # Map variables to indices using helper method
            var_indices = self._map_variables_to_indices(vars_sorted, dim)

            equation_funcs[dim] = f
            equation_vars[dim] = var_indices

            # Get equation string for display
            regressor = self.pysr_regressor[dim]
            if dim_complexity is None:
                equation_strs[dim] = regressor.get_best()["equation"]
            else:
                matching_rows = regressor.equations_[regressor.equations_["complexity"] == dim_complexity]
                equation_strs[dim] = matching_rows["equation"].values[0]

        # Store the equation information
        self._equation_funcs = equation_funcs
        self._equation_vars = equation_vars
        self._using_equation = True

        # Print success messages
        if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
            print(f"✅ Successfully switched {self.block_name} to symbolic equations for {len(dimensions_to_process)} active dimensions:")
        else:
            print(f"✅ Successfully switched {self.block_name} to symbolic equations for all {len(dimensions_to_process)} dimensions:")

        for dim in dimensions_to_process:
            print(f"   Dimension {dim}: {equation_strs[dim]}")

            # Display variable names properly
            var_names_display = []
            if hasattr(self, '_variable_names') and self._variable_names is not None:
                # Use custom variable names
                for idx in equation_vars[dim]:
                    if idx < len(self._variable_names):
                        var_names_display.append(self._variable_names[idx])
                    else:
                        var_names_display.append(f"transform_{idx}")
            else:
                # Use default x0, x1, etc. format
                var_names_display = [f'x{i}' for i in equation_vars[dim]]

            print(f"   Variables: {var_names_display}")

        if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
            print(f"🎯 Active dimensions {dimensions_to_process} now using symbolic equations.")
            print(f"🔒 Inactive dimensions will output zeros.")
        else:
            print(f"🎯 All {len(dimensions_to_process)} output dimensions now using symbolic equations.")

    def get_symbolic_function(self, dim: int = 0, complexity: int = None):

        if not hasattr(self, 'pysr_regressor') or not self.pysr_regressor:
            raise ValueError("No equations found. Run .distill() first.")

        if not hasattr(self, 'output_dims'):
            raise ValueError("No output dimension information found. Run .distill() first.")

        # If only one output dimension, default to dim=0
        if self.output_dims == 1:
            dim = 0
        elif dim >= self.output_dims:
            raise ValueError(f"Dimension {dim} out of range. Model has {self.output_dims} output dimensions (0-{self.output_dims-1})")

        if dim not in self.pysr_regressor:
            raise ValueError(f"No equation found for dimension {dim}. Available dimensions: {list(self.pysr_regressor.keys())}")

        regressor = self.pysr_regressor[dim]

        # Get the equation at specified complexity or best equation
        if complexity is None:
            best_str = regressor.get_best()["equation"]
            expr = regressor.equations_.loc[regressor.equations_["equation"] == best_str, "sympy_format"].values[0]
        else:
            matching_rows = regressor.equations_[regressor.equations_["complexity"] == complexity]
            if matching_rows.empty:
                available_complexities = sorted(regressor.equations_["complexity"].unique())
                raise ValueError(f"No equation with complexity {complexity} for dimension {dim}. Available complexities: {available_complexities}")
            expr = matching_rows["sympy_format"].values[0]

        vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))

        try:
            f = lambdify(vars_sorted, expr, "numpy")
        except Exception as e:
            raise RuntimeError(f"Could not create lambdify function for dimension {dim}: {e}")

        # Create a wrapper function that handles variable extraction
        def symbolic_func(x):
            if isinstance(x, torch.Tensor):
                x_tensor = x
            else:
                x_tensor = torch.tensor(x, dtype=torch.float32)

            # Map variables to indices
            var_indices = self._map_variables_to_indices(vars_sorted, dim)

            # Extract variables
            selected_inputs = self._extract_variables_for_equation(x_tensor, var_indices, dim)

            # Convert to numpy
            numpy_inputs = [inp.detach().cpu().numpy() if hasattr(inp, 'detach') else np.array(inp) for inp in selected_inputs]

            # Evaluate the equation
            result = f(*numpy_inputs)

            return result

        return symbolic_func

    def show_symbolic_expression(self, dim = None, complexity = None):

        if not hasattr(self, 'pysr_regressor') or not self.pysr_regressor:
            print("❗No equations found for this block yet. You need to first run .distill.")
            return

        if not hasattr(self, 'output_dims'):
            print("❗No output dimension information found. You need to first run .distill.")
            return

        # Convert single values to lists
        if isinstance(dim, int):
            dims_to_show = [dim]
        elif dim is None:
            # For pruned models, show only active dimensions by default
            if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                dims_to_show = self.get_active_dimensions()
                if dims_to_show:
                    print(f"ℹ️ Showing expressions for {len(dims_to_show)} active dimensions (out of {self.output_dims} total)")
            else:
                dims_to_show = list(range(self.output_dims))
        else:
            dims_to_show = dim

        # Show all equations for specified dimensions
        if complexity is None:
            for i in dims_to_show:
                if i not in self.pysr_regressor:
                    print(f"❌ No expression distilled for output dimension {i}.")
                    continue
                regressor = self.pysr_regressor[i]
                print(f"\n➡️ Symbolic expressions for output dimension {i}:")
                print(regressor.equations_)
                best_equation = regressor.get_best()
                print(f"🏆 Best: {best_equation['equation']} (loss: {best_equation['loss']:.6e})")

        # Show specific complexity for each dimension
        else:
            if isinstance(complexity, int):
                complexities = [complexity] * len(dims_to_show)
            else:
                complexities = complexity

            if len(complexities) != len(dims_to_show):
                print(f"❗Complexity list length ({len(complexities)}) must match dimension list length ({len(dims_to_show)})")
                return

            for i, comp in zip(dims_to_show, complexities):
                if i not in self.pysr_regressor:
                    print(f"❌ No expression distilled for output dimension {i}.")
                    continue

                regressor = self.pysr_regressor[i]
                matching_rows = regressor.equations_[regressor.equations_["complexity"] == comp]

                if matching_rows.empty:
                    available = sorted(regressor.equations_["complexity"].unique())
                    print(f"❌ No equation with complexity {comp} for dimension {i}. Available: {available}")
                    continue

                print(f"\n➡️ Dimension {i} - Complexity {comp}:")
                print(f"   {matching_rows['equation'].values[0]} (loss: {matching_rows['loss'].values[0]:.6e})")


                

    def switch_to_block(self):
        """
        Switch back to using the original model block for forward passes.
        
        Restores the neural network as the primary forward pass mechanism,
        reverting any previous switch_to_symbolic() call.
            
        Example:
            >>> model.switch_to_symbolic()  # Use symbolic equation
            >>> # ... do some analysis ...
            >>> model.switch_to_block()       # Switch back to neural network
        """
        self._using_equation = False
        if hasattr(self, '_original_block'):
            self.symtorch_block = self._original_block
        print(f"✅ Switched {self.block_name} back to block")

    def setup_pruning(self, initial_dim: int, target_dim: int, total_epochs: int,
                      end_epoch_frac: float = 0.5,
                      decay_rate: Literal['cosine', 'exp', 'linear'] = 'exp'):
        
        if not isinstance(self.symtorch_block, nn.Module):
            assert "❌ Pruning only works on PyTorch MLPs, not callable functions."

        self.initial_dim = initial_dim
        self.current_dim = initial_dim
        self.target_dim = target_dim 

        self._set_pruning_schedule = self._set_pruning_schedule(total_epochs, decay_rate, end_epoch_frac)
        self.register_buffer('pruning_mask', torch.ones(self.current_dim, dtype=torch.bool))

        print(f"✅ Pruning successfully set up for MLP {self.block_name}.")

        return None


    def _set_pruning_schedule(self, total_epochs: int, decay_rate: str = 'cosine', end_epoch_frac: float = 0.5):
        
        prune_end_epoch = int(end_epoch_frac * total_epochs)
        prune_epochs = prune_end_epoch

        dims_to_prune = self.initial_dim - self.target_dim
        schedule_dict = {}

        #different pruning schedules
        #exponential decay
        if decay_rate == 'exp':
            decay_rate = 3.0
            max_decay = 1 - math.exp(-decay_rate)

            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                raw_decay = 1 - math.exp(-decay_rate * progress)
                decay_factor = raw_decay / max_decay

                dims_pruned = math.ceil(dims_to_prune * decay_factor)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #linear decay
        elif decay_rate == 'linear':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                dims_pruned = math.ceil(dims_to_prune * progress)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #cosine decay
        elif decay_rate == 'cosine':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                dims_pruned = math.ceil(dims_to_prune * (1 - cosine_decay))
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #keep target_dim for the last part of training
        for epoch in range(prune_end_epoch, total_epochs):
            schedule_dict[epoch] = self.target_dim

        return schedule_dict
    
    def prune(self, epoch: int, sample_data: torch.Tensor, parent_model=None):
        """
        Perform pruning for the current epoch based on the pruning schedule.
        
        Evaluates the importance of each output dimension by computing the standard deviation
        of activations across the sample data. Retains the most important dimensions according
        to the current epoch's target dimensionality.
        
        Args:
            epoch (int): Current training epoch
            sample_data (torch.Tensor): Sample input data to evaluate dimension importance.
                                       Typically a subset of validation data.
            parent_model (nn.Module, optional): The parent model containing this PruningMLP instance.
                                              If provided, will trace intermediate activations to get
                                              the actual outputs at this layer level for importance evaluation.
                                       
        Note:
            This method should be called during each training epoch. If the current epoch
            is not in the pruning schedule, no pruning is performed.
        """

        if epoch not in self.pruning_schedule:
            return
        if self.pruning_schedule is None:
            assert 'Pruning has not been set up for this block.'
            
        target_dims = self.pruning_schedule[epoch]
        
        with torch.no_grad():
            # Extract outputs at this layer level for importance evaluation
            if parent_model is not None:
                with self._capture_layer_output(parent_model, sample_data) as (_, layer_outputs):
                    pass
                
                # Use captured intermediate data
                if layer_outputs:
                    output_array = layer_outputs[0]
                else:
                    raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicMLP instance.")
            else:
                # Original behavior - use MLP directly
                output_array = self.InterpretSR_MLP(sample_data)

            output_importance = output_array.std(dim=0)
            most_important = torch.argsort(output_importance)[-target_dims:]
            
            new_mask = torch.zeros_like(self.pruning_mask)
            new_mask[most_important] = True
            # Update the registered buffer (this maintains device consistency)
            self.pruning_mask.data = new_mask.data
            self.current_dim = target_dims

    def get_active_dimensions(self):
        """
        Get indices of currently active (non-masked) dimensions.

        Returns:
            list: List of integer indices for dimensions that are currently active
                 (not pruned/masked)

        Example:
            >>> active_dims = pruned_mlp.get_active_dimensions()
            >>> print(f"Active dimensions: {active_dims}")
            Active dimensions: [5, 12, 18]
        """
        if not hasattr(self, 'pruning_mask') or self.pruning_mask is None:
            raise RuntimeError("Pruning has not been set up for this block. Call setup_pruning() first.")

        return torch.where(self.pruning_mask)[0].tolist()

    def forward(self, x):
        """
        Forward pass through the model.

        Automatically switches between block and symbolic equations based on current mode.
        When using symbolic equation mode, evaluates each output dimension separately
        using its corresponding symbolic expression.

        This method works for both nn.Module and Callable function blocks, handling
        type conversions automatically.

        If pruning is enabled, applies pruning mask to enforce zero outputs for inactive dimensions.

        Args:
            x (torch.Tensor or numpy.ndarray): Input data of shape (batch_size, input_dim)

        Returns:
            Same type as input: Output data of shape (batch_size, output_dim)
                              - torch.Tensor if input is torch.Tensor
                              - numpy.ndarray if input is numpy.ndarray

        Raises:
            ValueError: If symbolic equations require variables not present in input
        """
        if hasattr(self, '_using_equation') and self._using_equation:
            # Track input type to return matching output type
            is_torch_input = isinstance(x, torch.Tensor)

            # Convert to torch tensor if needed for equation evaluation
            if not is_torch_input:
                x_torch = torch.tensor(x, dtype=torch.float32)
            else:
                x_torch = x

            batch_size = x_torch.shape[0]

            # Check if pruning is enabled
            if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                # For pruning mode, initialize output with zeros for all dimensions
                output = torch.zeros(batch_size, self.initial_dim, dtype=x_torch.dtype, device=x_torch.device)

                # Fill in only active dimensions with symbolic equations
                active_dims = self.get_active_dimensions()
                for dim in active_dims:
                    if dim in self._equation_funcs:
                        equation_func = self._equation_funcs[dim]
                        var_indices = self._equation_vars[dim]

                        # Extract variables needed for this dimension
                        selected_inputs = self._extract_variables_for_equation(x_torch, var_indices, dim)

                        # Convert to numpy for the equation function
                        numpy_inputs = [inp.detach().cpu().numpy() for inp in selected_inputs]

                        # Evaluate the equation for this dimension
                        result = equation_func(*numpy_inputs)

                        # Convert back to torch tensor with same device/dtype as input
                        result_tensor = torch.tensor(result, dtype=x_torch.dtype, device=x_torch.device)

                        # Ensure result is 1D (batch_size,)
                        if result_tensor.dim() == 0:
                            result_tensor = result_tensor.expand(batch_size)
                        elif result_tensor.dim() > 1:
                            result_tensor = result_tensor.flatten()

                        output[:, dim] = result_tensor

                # Apply pruning mask to ensure inactive dimensions are zero
                result_tensor = output * self.pruning_mask
            else:
                # Standard mode without pruning
                output_dims = len(self._equation_funcs)

                # Initialize output tensor
                outputs = []

                # Evaluate each dimension separately
                for dim in range(output_dims):
                    equation_func = self._equation_funcs[dim]
                    var_indices = self._equation_vars[dim]

                    # Extract variables needed for this dimension
                    selected_inputs = self._extract_variables_for_equation(x_torch, var_indices, dim)

                    # Convert to numpy for the equation function
                    numpy_inputs = [inp.detach().cpu().numpy() for inp in selected_inputs]

                    # Evaluate the equation for this dimension
                    result = equation_func(*numpy_inputs)

                    # Convert back to torch tensor with same device/dtype as input
                    result_tensor = torch.tensor(result, dtype=x_torch.dtype, device=x_torch.device)

                    # Ensure result is 1D (batch_size,)
                    if result_tensor.dim() == 0:
                        result_tensor = result_tensor.expand(batch_size)
                    elif result_tensor.dim() > 1:
                        result_tensor = result_tensor.flatten()

                    outputs.append(result_tensor)

                # Stack all dimensions to create (batch_size, output_dim) tensor
                result_tensor = torch.stack(outputs, dim=1)

            # Return in same type as input
            if is_torch_input:
                return result_tensor
            else:
                return result_tensor.detach().cpu().numpy()
        else:
            # For nn.Module, call directly
            if isinstance(self.symtorch_block, nn.Module):
                output = self.symtorch_block(x)
                # Apply pruning mask if enabled
                if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                    output = output * self.pruning_mask
                return output
            else:
                # For Callable functions, handle input type appropriately
                is_torch_input = isinstance(x, torch.Tensor)

                if is_torch_input:
                    # Convert torch tensor to numpy for callable function
                    x_np = x.detach().cpu().numpy()
                    output = self.symtorch_block(x_np)

                    # Convert output back to torch tensor
                    if hasattr(output, 'detach'):  # Already a torch tensor
                        output = output.to(x.device)
                    else:
                        output = torch.tensor(output, dtype=x.dtype, device=x.device)

                    # Apply pruning mask if enabled
                    if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                        output = output * self.pruning_mask
                    return output
                else:
                    # Input is already numpy, call directly and return numpy
                    output = self.symtorch_block(x)
                    # Apply pruning mask if enabled (convert to numpy if needed)
                    if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                        if not isinstance(output, torch.Tensor):
                            output = torch.tensor(output, dtype=torch.float32)
                        output = output * self.pruning_mask
                        output = output.numpy()
                    return output
