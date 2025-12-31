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
import dill
from typing import List, Callable, Optional, Union, Dict, Any
from contextlib import contextmanager
from typing import Literal
import math
from sklearn.neighbors import NearestNeighbors


# TODO: switch to using a logger.
# TODO: break up this class using composition?
# TODO: integrate dim reduction workflow (e.g., pca, proj. layer training, etc...)
class SymbolicModel(nn.Module):

    # Default PySR parameters
    DEFAULT_SR_PARAMS = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "niterations": 400,
        "complexity_of_operators": {"sin": 3, "exp": 3}
    }

    # Default SLIME parameters
    DEFAULT_SLIME_PARAMS = {
        "x": None,                  # Point of interest for local explanation
        "J_nn": 10,                 # Number of nearest neighbors
        "num_synthetic": 100,       # Number of synthetic samples
        "real_weighting": 1.0,      # Weight for real samples vs synthetic
        "nn_metric": 'euclidean',   # Distance metric for nearest neighbors
        "var": None                 # Variance for perturbations (auto-computed if None)
    }

    def __init__(self, block: Union[nn.Module, Callable], block_name: str = None):
        """
        Initialize a SymbolicModel wrapper for symbolic regression.

        Creates a unified wrapper that can perform symbolic regression on either
        PyTorch nn.Module layers or any callable function. This is the entry point
        for all SymTorch functionality including layer-level analysis, model-agnostic
        symbolic regression, SLIME local interpretability, and pruning.

        Args:
            block (Union[nn.Module, Callable]): The component to wrap. Can be:
                - A PyTorch nn.Module (e.g., nn.Linear, custom layer) for layer-level mode
                - Any callable function for model-agnostic mode (PyTorch models,
                  scikit-learn models, TensorFlow models, pure Python functions)
            block_name (str, optional): Human-readable identifier for this block.
                If None, generates a unique name based on object ID.

        Examples:
            >>> # Layer-level mode: Wrap a PyTorch layer
            >>> import torch.nn as nn
            >>> layer = nn.Linear(10, 5)
            >>> symbolic_layer = SymbolicModel(layer, block_name='hidden_layer_1')

            >>> # Model-agnostic mode: Wrap a callable function
            >>> def my_function(x):
            ...     return x[:, 0]**2 + 3*np.sin(x[:, 1])
            >>> symbolic_func = SymbolicModel(my_function, block_name='my_func')

            >>> # Model-agnostic mode: Wrap a scikit-learn model's predict method
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> rf = RandomForestRegressor().fit(X_train, y_train)
            >>> symbolic_rf = SymbolicModel(rf.predict, block_name='rf_model')

        Save/Load:
            SymbolicModel supports PyTorch's standard save/load mechanisms:

            >>> # Save model state (recommended)
            >>> torch.save(model.state_dict(), 'model.pth')
            >>>
            >>> # Load model state
            >>> model = SymbolicModel(architecture, block_name='my_model')
            >>> model.load_state_dict(torch.load('model.pth'))
            >>>
            >>> # Full model save/load also works
            >>> torch.save(model, 'full_model.pth')
            >>> model = torch.load('full_model.pth', weights_only=False)
        """

        super().__init__()
        self.symtorch_block = block
        self.block_name = block_name or f"block_{id(self)}"

        if not block_name:
            print(f"No name specified for this block. Label is {self.block_name}.")

        self.pysr_regressor = {}
        self.SLIME_pysr_regressor = {}

        # I/O caching for distill
        self.distill_data = None  # Cache for standard distill
        self.distill_data_slime = None  # Cache for SLIME distill

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

    def _check_cache_hit(self, inputs, parent_model, SLIME, slime_params):
        """
        Check if we can use cached I/O data from a previous distill call.

        Args:
            inputs: Input data for distill
            parent_model: Parent model (or None)
            SLIME (bool): Whether SLIME mode is enabled
            slime_params (Dict): SLIME parameters

        Returns:
            tuple: (cache_hit, cached_inputs, cached_outputs) where cache_hit is bool,
                   and cached_inputs/outputs are numpy arrays if hit, else None
        """
        # Convert inputs to numpy for comparison
        if hasattr(inputs, 'detach'):  # torch tensor
            inputs_np = inputs.detach().cpu().numpy()
        else:
            inputs_np = np.array(inputs)

        # Select appropriate cache
        if SLIME:
            cache = self.distill_data_slime
        else:
            cache = self.distill_data

        # If no cache exists, return miss
        if cache is None:
            return False, None, None

        # Check if inputs match
        cached_inputs = cache['inputs']
        if not np.array_equal(inputs_np, cached_inputs):
            return False, None, None

        # Check if parent_model matches (both None or both same object)
        if cache['parent_model'] is not parent_model:
            return False, None, None

        # For SLIME mode, also check if slime_params match
        if SLIME:
            # Merge with defaults to ensure complete comparison
            final_slime_params = {**self.DEFAULT_SLIME_PARAMS}
            if slime_params is not None:
                final_slime_params.update(slime_params)

            cached_slime_params = cache['slime_params']

            # Compare all SLIME params except 'x' (which needs special handling for numpy arrays)
            for key in final_slime_params:
                if key == 'x':
                    # Handle numpy array comparison for point of interest
                    cached_x = cached_slime_params.get('x')
                    current_x = final_slime_params.get('x')

                    # Convert to numpy if needed
                    if isinstance(cached_x, torch.Tensor):
                        cached_x = cached_x.detach().cpu().numpy()
                    if isinstance(current_x, torch.Tensor):
                        current_x = current_x.detach().cpu().numpy()

                    # Check if both are None or both are equal arrays
                    if cached_x is None and current_x is None:
                        continue
                    elif cached_x is None or current_x is None:
                        return False, None, None
                    elif not np.array_equal(np.array(cached_x), np.array(current_x)):
                        return False, None, None
                else:
                    if cached_slime_params.get(key) != final_slime_params.get(key):
                        return False, None, None

        # Cache hit!
        return True, cache['sr_inputs'], cache['sr_outputs']

    def _apply_slime_sampling(self, inputs_np, function_to_call, slime_params, sr_params, fit_params):
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
        final_slime_params = {**self.DEFAULT_SLIME_PARAMS}
        if slime_params is not None:
            final_slime_params.update(slime_params)

        x0 = final_slime_params['x']
        J_nn = final_slime_params['J_nn']
        num_synthetic = final_slime_params['num_synthetic']
        real_weighting = final_slime_params['real_weighting']
        nn_metric = final_slime_params['nn_metric']
        var = final_slime_params['var']

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
            synthetic_samples = np.random.normal(
                loc=x0,
                scale=np.sqrt(var_computed),
                size=(num_synthetic, len(x0))
            ).astype(np.float64)

            # Combine real and synthetic inputs
            sr_inputs_slime = np.concatenate([real_inputs, synthetic_samples], axis=0).astype(np.float64)

            # Get outputs for SLIME samples
            slime_outputs = function_to_call(sr_inputs_slime)

            # Prepare weights
            synthetic_distances_sq = np.sum((synthetic_samples - x0)**2 / var_computed, axis=1)
            gaussian_weights = np.exp(-synthetic_distances_sq).astype(np.float64)
            slime_weights = np.concatenate([
                np.full(len(real_inputs), real_weighting, dtype=np.float64),
                gaussian_weights
            ])

            # Update sr_params with weighted loss
            if sr_params is None:
                sr_params = {}
            sr_params = sr_params.copy()
            sr_params['elementwise_loss'] = "loss(prediction, target, weight) = weight * (prediction - target)^2"

            # Update fit_params with weights
            if fit_params is None:
                fit_params = {}
            fit_params = fit_params.copy()
            fit_params['weights'] = slime_weights

            print(f"🔍 SLIME mode: Using {len(sr_inputs_slime)} points ({len(real_inputs)} real + {num_synthetic} synthetic)")
            print(f"   Point of interest: {x0}")

            return sr_inputs_slime, slime_outputs, sr_params, fit_params
        else:
            # Global SLIME (no local focus)
            print("🔍 SLIME mode: Global (no local focus point)")
            return inputs_np, function_to_call(inputs_np), sr_params, fit_params

    def distill(self, inputs, output_dim: int = None, parent_model=None,
                 variable_transforms: Optional[List[Callable]] = None,
                 save_path: str = None,
                 sr_params: Optional[Dict[str, Any]] = None,
                 fit_params: Optional[Dict[str, Any]] = None,
                 SLIME: bool = False,
                 slime_params: Optional[Dict[str, Any]] = None):
        """
        Perform symbolic regression to discover symbolic equations.

        This is the main method for extracting symbolic representations from neural networks
        or arbitrary functions. It uses PySR (Python Symbolic Regression) to find mathematical
        expressions that approximate the behavior of the wrapped block or function.

        The method supports multiple operational modes:
        - Layer-level mode: Analyze intermediate activations within a parent model
        - Model-agnostic mode: Analyze any callable function directly
        - SLIME mode: Local interpretability around specific data points
        - Pruning mode: Symbolic regression on only active dimensions

        Args:
            inputs (torch.Tensor or np.ndarray): Input data for symbolic regression.
                - For layer-level mode with parent_model: inputs to the parent model
                - For direct mode: inputs to the block/function itself
                Shape: (num_samples, input_dim)
            output_dim (int, optional): Specific output dimension to process.
                If None, processes all output dimensions. Useful for incremental analysis.
            parent_model (nn.Module, optional): Parent model containing this layer.
                Required for layer-level mode to capture intermediate activations.
                Must be None for callable functions (non-nn.Module blocks).
            variable_transforms (List[Callable], optional): List of transformation functions
                to apply to inputs before symbolic regression. Each function should take
                inputs and return a 1D tensor/array. Useful for feature engineering.
            save_path (str, optional): Directory path to save PySR outputs.
                If None, saves to 'SR_output/{block_name}'.
            sr_params (Dict[str, Any], optional): Custom PySR parameters to override defaults.
                Common parameters:
                - 'niterations': Number of iterations (default: 400)
                - 'binary_operators': List of binary ops (default: ["+", "*"])
                - 'unary_operators': List of unary ops (default: ["inv(x) = 1/x", "sin", "exp"])
                - 'complexity_of_operators': Complexity constraints (default: {"sin": 3, "exp": 3})
            fit_params (Dict[str, Any], optional): Parameters passed to PySRRegressor.fit().
                - 'variable_names': List of custom names for input variables
                - 'weights': Sample weights for weighted regression
            SLIME (bool, optional): Enable SLIME mode for local interpretability.
                Default: False. When True, focuses regression around specific points.
            slime_params (Dict[str, Any], optional): SLIME configuration parameters.
                - 'x': Point of interest for local explanation (np.ndarray or None for global)
                - 'J_nn': Number of nearest neighbors (default: 10)
                - 'num_synthetic': Number of synthetic samples (default: 100)
                - 'real_weighting': Weight for real vs synthetic samples (default: 1.0)
                - 'nn_metric': Distance metric (default: 'euclidean')
                - 'var': Variance for perturbations (default: auto-computed)

        Returns:
            Union[PySRRegressor, Dict[int, PySRRegressor]]:
                - If output_dim is specified: Single PySRRegressor for that dimension
                - If output_dim is None: Dictionary mapping dimension indices to PySRRegressors

        Raises:
            ValueError: If parent_model is provided with a Callable (non-nn.Module) block
            ValueError: If variable_transforms length doesn't match variable_names length
            ValueError: If SLIME mode with point of interest but num_synthetic=0
            RuntimeError: If layer-level mode fails to capture intermediate activations

        Examples:
            >>> # Layer-level mode: Analyze a hidden layer within a parent model
            >>> model = MyNeuralNetwork()
            >>> symbolic_layer = SymbolicModel(model.hidden_layer, block_name='layer_1')
            >>> symbolic_layer.distill(training_data, parent_model=model)
            >>> symbolic_layer.show_symbolic_expression()

            >>> # Model-agnostic mode: Analyze a function directly
            >>> def f(x):
            ...     return x[:, 0]**2 + 3*np.sin(x[:, 1])
            >>> symbolic_func = SymbolicModel(f, block_name='my_func')
            >>> symbolic_func.distill(training_data)
            >>> symbolic_func.switch_to_symbolic()

            >>> # SLIME mode: Local explanation around a specific point
            >>> x0 = np.array([1.0, 2.0])
            >>> slime_params = {'x': x0, 'J_nn': 10, 'num_synthetic': 100}
            >>> symbolic_func.distill(training_data, SLIME=True, slime_params=slime_params)
            >>> symbolic_func.show_symbolic_expression(SLIME=True)

            >>> # With custom variable transforms and names
            >>> transforms = [
            ...     lambda x: x[:, 0] + x[:, 1],  # Sum of first two features
            ...     lambda x: x[:, 0] * x[:, 1],  # Product of first two features
            ...     lambda x: torch.sin(x[:, 2])  # Sin of third feature
            ... ]
            >>> fit_params = {'variable_names': ['sum_01', 'prod_01', 'sin_2']}
            >>> symbolic_layer.distill(data, variable_transforms=transforms, fit_params=fit_params)

            >>> # With custom SR parameters
            >>> sr_params = {
            ...     'niterations': 1000,
            ...     'binary_operators': ["+", "*", "-", "/"],
            ...     'complexity_of_operators': {"sin": 5, "exp": 5}
            ... }
            >>> symbolic_func.distill(data, sr_params=sr_params)

            >>> # Process only a specific output dimension
            >>> symbolic_layer.distill(data, output_dim=2, parent_model=model)
        """

        if isinstance(self.symtorch_block, Callable) and not isinstance(self.symtorch_block, nn.Module) and parent_model is not None:
            raise ValueError(
                "Cannot use parent_model with Callable functions. "
                "Hooks are only supported for nn.Module objects. "
                "Please call distill() without parent_model argument and pass inputs directly to the function."
            )

        # Check cache for I/O data
        cache_hit, cached_sr_inputs, cached_sr_outputs = self._check_cache_hit(inputs, parent_model, SLIME, slime_params)

        if cache_hit:
            print(f"🔄 Cache hit! Reusing I/O data from previous distill call.")
            actual_inputs_numpy = cached_sr_inputs
            # cached_sr_outputs is already numpy array
            if SLIME or (hasattr(cached_sr_outputs, 'ndim') and cached_sr_outputs.ndim == 1):
                output = cached_sr_outputs
            else:
                # Convert back to torch for processing
                output = torch.tensor(cached_sr_outputs, dtype=torch.float32)
            skip_io_extraction = True
        else:
            skip_io_extraction = False

        # Extract inputs and outputs at this layer level
        if isinstance(self.symtorch_block, nn.Module):
            # Extract fit parameters (needed for both cache hit and miss)
            if fit_params is None:
                fit_params = {}

            variable_names = fit_params.get('variable_names', None)

            # Extract sr_params with defaults (needed for both cache hit and miss)
            if sr_params is None:
                sr_params = {}

            if not skip_io_extraction:
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

                # Apply SLIME sampling if enabled
                if SLIME:
                    # Create function that evaluates the block
                    def eval_block(inputs_array):
                        inputs_tensor = torch.tensor(inputs_array, dtype=torch.float32, device=actual_inputs.device)
                        self.symtorch_block.eval()
                        with torch.no_grad():
                            return self.symtorch_block(inputs_tensor)

                    actual_inputs_numpy, output, sr_params, fit_params = self._apply_slime_sampling(
                        actual_inputs_numpy, eval_block, slime_params, sr_params, fit_params
                    )

                # Store cache for future distill calls
                # Convert inputs to numpy for cache storage
                if hasattr(inputs, 'detach'):
                    inputs_cache = inputs.detach().cpu().numpy()
                else:
                    inputs_cache = np.array(inputs)

                # Convert output to numpy for cache storage
                if hasattr(output, 'detach'):
                    output_cache = output.detach().cpu().numpy()
                else:
                    output_cache = np.array(output)

                # Store in appropriate cache
                cache_data = {
                    'inputs': inputs_cache,
                    'sr_inputs': actual_inputs_numpy,
                    'sr_outputs': output_cache,
                    'parent_model': parent_model
                }

                if SLIME:
                    # Merge with defaults for complete storage
                    final_slime_params = {**self.DEFAULT_SLIME_PARAMS}
                    if slime_params is not None:
                        final_slime_params.update(slime_params)
                    cache_data['slime_params'] = final_slime_params
                    self.distill_data_slime = cache_data
                else:
                    self.distill_data = cache_data
            else:
                # Using cached data - set target_dims based on cached output shape
                if hasattr(output, 'shape') and len(output.shape) > 1:
                    # Reconstruct target_dims from cache
                    if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
                        target_dims = self.get_active_dimensions()
                    else:
                        target_dims = None

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

            # Store in appropriate dictionary
            if SLIME:
                self.SLIME_pysr_regressor = self.SLIME_pysr_regressor | pysr_regressors
            else:
                self.pysr_regressor = self.pysr_regressor | pysr_regressors

            # For backward compatibility, return the regressor or dict of regressors
            if output_dim is not None:
                return pysr_regressors.get(output_dim)
            else:
                return pysr_regressors
            
        else: #code for Callable function
            # Extract fit parameters (needed for both cache hit and miss)
            if fit_params is None:
                fit_params = {}

            variable_names = fit_params.get('variable_names', None)

            # Extract sr_params with defaults (needed for both cache hit and miss)
            if sr_params is None:
                sr_params = {}

            if not skip_io_extraction:

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

                # Apply SLIME sampling if enabled
                if SLIME:
                    # Create function that evaluates the callable
                    def eval_callable(inputs_array):
                        outputs_raw = self.symtorch_block(inputs_array)
                        if hasattr(outputs_raw, 'detach'):  # torch tensor
                            return outputs_raw.detach().cpu().numpy()
                        else:
                            return np.array(outputs_raw)

                    inputs_np, outputs_np, sr_params, fit_params = self._apply_slime_sampling(
                        inputs_np, eval_callable, slime_params, sr_params, fit_params
                    )

                # Handle both 1D and 2D outputs
                if outputs_np.ndim == 1:
                    outputs_np = outputs_np.reshape(-1, 1)

                # Store cache for future distill calls
                # Convert inputs to numpy for cache storage
                if hasattr(inputs, 'detach'):
                    inputs_cache = inputs.detach().cpu().numpy()
                else:
                    inputs_cache = np.array(inputs)

                # Store in appropriate cache
                cache_data = {
                    'inputs': inputs_cache,
                    'sr_inputs': inputs_np,
                    'sr_outputs': outputs_np,
                    'parent_model': parent_model
                }

                if SLIME:
                    # Merge with defaults for complete storage
                    final_slime_params = {**self.DEFAULT_SLIME_PARAMS}
                    if slime_params is not None:
                        final_slime_params.update(slime_params)
                    cache_data['slime_params'] = final_slime_params
                    self.distill_data_slime = cache_data
                else:
                    self.distill_data = cache_data
            else:
                # Using cached data
                inputs_np = actual_inputs_numpy
                outputs_np = output

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

            # Store in appropriate dictionary
            if SLIME:
                self.SLIME_pysr_regressor = self.SLIME_pysr_regressor | pysr_regressors
            else:
                self.pysr_regressor = self.pysr_regressor | pysr_regressors

            # For backward compatibility, return the regressor or dict of regressors
            if output_dim is not None:
                return pysr_regressors[output_dim]
            else:
                return pysr_regressors
            
    def _get_equation(self, dim, complexity: int = None, SLIME: bool = False):
        """
        Extract symbolic equation function from fitted regressor.

        Converts the symbolic expression from PySR into a callable function
        that can be used for prediction.

        Args:
            dim (int): Output dimension to get equation for.
            complexity (int, optional): Specific complexity level to retrieve.
                                      If None, returns the best overall equation.
            SLIME (bool, optional): If True, use SLIME regressor instead of standard regressor.

        Returns:
            tuple or None: (equation_function, sorted_variables) if successful,
                          None if no equation found or complexity not available


        Note:
            This is an internal method. Use switch_to_symbolic() for public API.
        """
        # Select appropriate regressor dictionary
        if SLIME:
            regressor_dict = self.SLIME_pysr_regressor
            mode_name = "SLIME"
        else:
            regressor_dict = self.pysr_regressor
            mode_name = "standard"

        if not hasattr(self, regressor_dict.__class__.__name__.replace('dict', 'pysr_regressor')) or regressor_dict is None:
            print(f"❗No {mode_name} equations found for this block yet. You need to first run .distill with SLIME={SLIME}.")
            return None
        if dim not in regressor_dict:
            print(f"❗No {mode_name} equation found for output dimension {dim}. You need to first run .distill with SLIME={SLIME}.")
            return None

        regressor = regressor_dict[dim]
        
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
            f = lambdify(vars_sorted, expr, "torch")
            return f, vars_sorted
        except Exception as e:
            print(f"⚠️ Warning: Could not create lambdify function for dimension {dim}: {e}")
            return None

    def switch_to_symbolic(self, complexity: list = None, SLIME: bool = False):
        """
        Switch the forward pass from model block to symbolic equations for all output dimensions.

        After calling this method, the model will use the discovered symbolic
        expressions instead of the neural network for forward passes.

        For pruned models, only active dimensions need equations. Inactive dimensions
        will output zeros.

        Args:
            complexity (list, optional): Specific complexity levels to use for each dimension.
                                      If None, uses the best overall equation for each dimension.
            SLIME (bool, optional): If True, use SLIME equations instead of standard equations.

        Example:
            >>> model.switch_to_symbolic(complexity=5)
            >>> model.switch_to_symbolic(SLIME=True)

        """
        # Select appropriate regressor dictionary
        if SLIME:
            regressor_dict = self.SLIME_pysr_regressor
            mode_name = "SLIME"
        else:
            regressor_dict = self.pysr_regressor
            mode_name = "standard"

        if not regressor_dict:
            print(f"❗No {mode_name} equations found for this block yet. You need to first run .distill with SLIME={SLIME}.")
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
                if dim not in regressor_dict:
                    missing_dims.append(dim)

            if missing_dims:
                print(f"❗Missing {mode_name} equations for active dimensions {missing_dims}. You need to run .distill with SLIME={SLIME} on all active dimensions first.")
                return

            dimensions_to_process = active_dims
        else:
            # Standard mode - need equations for all dimensions
            missing_dims = []
            for dim in range(self.output_dims):
                if dim not in regressor_dict:
                    missing_dims.append(dim)

            if missing_dims:
                print(f"❗Missing {mode_name} equations for dimensions {missing_dims}. You need to run .distill with SLIME={SLIME} on all output dimensions first.")
                print(f"Available dimensions: {list(regressor_dict.keys())}")
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

            result = self._get_equation(dim, dim_complexity, SLIME=SLIME)
            if result is None:
                print(f"⚠️ Failed to get equation for dimension {dim}")
                return

            f, vars_sorted = result

            # Map variables to indices using helper method
            var_indices = self._map_variables_to_indices(vars_sorted, dim)

            equation_funcs[dim] = f
            equation_vars[dim] = var_indices

            # Get equation string for display
            regressor = regressor_dict[dim]
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
        mode_label = f"{mode_name} " if SLIME else ""
        if hasattr(self, 'pruning_mask') and self.pruning_mask is not None:
            print(f"✅ Successfully switched {self.block_name} to {mode_label}symbolic equations for {len(dimensions_to_process)} active dimensions:")
        else:
            print(f"✅ Successfully switched {self.block_name} to {mode_label}symbolic equations for all {len(dimensions_to_process)} dimensions:")

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
            print(f"🎯 Active dimensions {dimensions_to_process} now using {mode_label}symbolic equations.")
            print(f"🔒 Inactive dimensions will output zeros.")
        else:
            print(f"🎯 All {len(dimensions_to_process)} output dimensions now using {mode_label}symbolic equations.")

        # TODO: Make torch compiling optional for user.
        # Apply torch.compile() optimization if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            print("🚀 Compiling forward pass with torch.compile() for GPU optimization...")
            try:
                # Compile with fullgraph=False to allow dynamic control flow
                # mode="reduce-overhead" optimizes for repeated calls
                self._original_forward = self.forward
                self.forward = torch.compile(self.forward, mode="reduce-overhead", fullgraph=False)
                print("✅ Forward pass compiled successfully")
            except Exception as e:
                print(f"⚠️ torch.compile() failed: {e}. Continuing without compilation.")
                # Forward pass will still work, just without compilation optimization

    def get_symbolic_function(self, dim: int = 0, complexity: int = None, SLIME: bool = False):
        """
        Get a callable Python function for a specific output dimension's symbolic equation.

        Returns a standalone Python function that evaluates the discovered symbolic expression
        for a given output dimension. This function can be used independently of the SymbolicModel
        for predictions, analysis, or integration into other code.

        The returned function automatically handles variable extraction and transformation based
        on the configuration used during distill().

        Args:
            dim (int, optional): Output dimension to retrieve function for. Default: 0.
                For models with only one output dimension, dim=0 is automatically used.
            complexity (int, optional): Specific complexity level to retrieve.
                If None, returns the best overall equation discovered by PySR.
                Use this to get simpler or more complex versions of the equation.
            SLIME (bool, optional): If True, retrieve SLIME equation instead of standard equation.
                Default: False. Must have run distill(SLIME=True) first.

        Returns:
            Callable: A function that takes input data (torch.Tensor or np.ndarray) and returns
                predictions as np.ndarray. The function signature is: f(x) -> np.ndarray

        Raises:
            ValueError: If no equations found (distill() not called yet)
            ValueError: If dimension is out of range
            ValueError: If requested dimension doesn't have an equation
            ValueError: If requested complexity level doesn't exist
            RuntimeError: If lambdify fails to create the function

        Examples:
            >>> # Get the symbolic function for dimension 0
            >>> symbolic_model.distill(training_data)
            >>> sym_func = symbolic_model.get_symbolic_function(dim=0)
            >>> predictions = sym_func(test_data)

            >>> # Get a simpler equation at lower complexity
            >>> sym_func_simple = symbolic_model.get_symbolic_function(dim=0, complexity=3)
            >>> simple_predictions = sym_func_simple(test_data)

            >>> # Get SLIME local explanation function
            >>> slime_params = {'x': np.array([1.0, 2.0]), 'J_nn': 10, 'num_synthetic': 100}
            >>> symbolic_model.distill(data, SLIME=True, slime_params=slime_params)
            >>> local_func = symbolic_model.get_symbolic_function(dim=0, SLIME=True)
            >>> local_predictions = local_func(test_data)

            >>> # Use the function independently
            >>> import numpy as np
            >>> test_input = np.random.randn(100, 5)
            >>> output = sym_func(test_input)  # Works with numpy arrays
            >>>
            >>> import torch
            >>> test_tensor = torch.randn(100, 5)
            >>> output = sym_func(test_tensor)  # Also works with torch tensors

            >>> # For multi-output models, get functions for each dimension
            >>> functions = []
            >>> for dim in range(model.output_dims):
            ...     functions.append(symbolic_model.get_symbolic_function(dim=dim))
            >>> outputs = [f(test_data) for f in functions]
        """

        # Select appropriate regressor dictionary
        if SLIME:
            regressor_dict = self.SLIME_pysr_regressor
            mode_name = "SLIME"
        else:
            regressor_dict = self.pysr_regressor
            mode_name = "standard"

        if not regressor_dict:
            raise ValueError(f"No {mode_name} equations found. Run .distill(SLIME={SLIME}) first.")

        if not hasattr(self, 'output_dims'):
            raise ValueError("No output dimension information found. Run .distill() first.")

        # If only one output dimension, default to dim=0
        if self.output_dims == 1:
            dim = 0
        elif dim >= self.output_dims:
            raise ValueError(f"Dimension {dim} out of range. Model has {self.output_dims} output dimensions (0-{self.output_dims-1})")

        if dim not in regressor_dict:
            raise ValueError(f"No {mode_name} equation found for dimension {dim}. Available dimensions: {list(regressor_dict.keys())}")

        regressor = regressor_dict[dim]

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
            f = lambdify(vars_sorted, expr, "torch")
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

            # Evaluate the equation (torch backend, stays on device)
            result = f(*selected_inputs)

            # Convert to numpy only for output (API compatibility)
            if isinstance(result, torch.Tensor):
                return result.detach().cpu().numpy()
            return result

        return symbolic_func

    def show_symbolic_expression(self, dim = None, complexity = None, SLIME: bool = False):
        """
        Display the discovered symbolic expressions for output dimensions.

        Prints the symbolic equations discovered by PySR in a human-readable format.
        Can show all equations at all complexity levels or specific equations at specific
        complexity levels. Useful for inspecting and comparing different symbolic approximations.

        Args:
            dim (int, list, or None, optional): Dimension(s) to display.
                - None: Show all dimensions (or all active dimensions if pruning is enabled)
                - int: Show only the specified dimension
                - list: Show multiple specified dimensions
                Default: None (show all)
            complexity (int, list, or None, optional): Complexity level(s) to display.
                - None: Show all equations at all complexity levels plus the best equation
                - int: Show equation at this specific complexity for all specified dimensions
                - list: Show equations at specified complexities (must match length of dim list)
                Default: None (show all)
            SLIME (bool, optional): If True, show SLIME equations instead of standard equations.
                Default: False. Must have run distill(SLIME=True) first.

        Returns:
            None: This method prints to console and does not return a value.

        Examples:
            >>> # Show all equations for all dimensions
            >>> symbolic_model.distill(training_data)
            >>> symbolic_model.show_symbolic_expression()

            >>> # Show equations for a specific dimension
            >>> symbolic_model.show_symbolic_expression(dim=0)

            >>> # Show equation at specific complexity for dimension 0
            >>> symbolic_model.show_symbolic_expression(dim=0, complexity=5)

            >>> # Show equations for multiple dimensions at different complexities
            >>> symbolic_model.show_symbolic_expression(dim=[0, 1, 2], complexity=[3, 5, 4])

            >>> # Show SLIME local interpretability equations
            >>> slime_params = {'x': np.array([1.0, 2.0]), 'J_nn': 10, 'num_synthetic': 100}
            >>> symbolic_model.distill(data, SLIME=True, slime_params=slime_params)
            >>> symbolic_model.show_symbolic_expression(SLIME=True)

            >>> # For pruned models, shows only active dimensions by default
            >>> symbolic_model.setup_pruning(initial_dim=64, target_dim=8, total_steps=10000)
            >>> # ... training with pruning ...
            >>> symbolic_model.distill(data)
            >>> symbolic_model.show_symbolic_expression()  # Shows only 8 active dimensions

            >>> # Show specific dimensions for a multi-output model
            >>> symbolic_model.show_symbolic_expression(dim=[0, 2, 5])

            >>> # Compare equations at different complexity levels
            >>> for c in [3, 5, 7]:
            ...     print(f"\nComplexity {c}:")
            ...     symbolic_model.show_symbolic_expression(dim=0, complexity=c)
        """

        # Select appropriate regressor dictionary
        if SLIME:
            regressor_dict = self.SLIME_pysr_regressor
            mode_name = "SLIME"
        else:
            regressor_dict = self.pysr_regressor
            mode_name = "standard"

        if not regressor_dict:
            print(f"❗No {mode_name} equations found for this block yet. You need to first run .distill with SLIME={SLIME}.")
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
                    print(f"ℹ️ Showing {mode_name} expressions for {len(dims_to_show)} active dimensions (out of {self.output_dims} total)")
            else:
                dims_to_show = list(range(self.output_dims))
        else:
            dims_to_show = dim

        # Show all equations for specified dimensions
        if complexity is None:
            for i in dims_to_show:
                if i not in regressor_dict:
                    print(f"❌ No {mode_name} expression distilled for output dimension {i}.")
                    continue
                regressor = regressor_dict[i]
                print(f"\n➡️ {mode_name.capitalize()} symbolic expressions for output dimension {i}:")
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
                if i not in regressor_dict:
                    print(f"❌ No {mode_name} expression distilled for output dimension {i}.")
                    continue

                regressor = regressor_dict[i]
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

        # Restore original block if it was saved
        if hasattr(self, '_original_block'):
            self.symtorch_block = self._original_block

        print(f"✅ Switched {self.block_name} back to block")

    def setup_pruning(self, initial_dim: int, target_dim: int, total_steps: int,
                      end_step_frac: float = 0.5,
                      decay_rate: Literal['cosine', 'exp', 'linear'] = 'exp'):
        """
        Set up pruning schedule for progressive dimensionality reduction on a per-step basis.

        Creates a schedule that progressively reduces dimensions from initial_dim to target_dim
        over the specified fraction of training steps using the chosen decay strategy.

        Args:
            initial_dim (int): Initial output dimensionality before pruning
            target_dim (int): Target output dimensionality after pruning
            total_steps (int): Total number of training steps
            end_step_frac (float, optional): Fraction of total steps to complete pruning by.
                                            Defaults to 0.5 (pruning ends halfway through training)
            decay_rate (str, optional): Pruning schedule type. Options:
                                      - 'exp': Exponential decay schedule (default)
                                      - 'linear': Linear reduction schedule
                                      - 'cosine': Cosine annealing schedule

        Example:
            >>> model.block.setup_pruning(initial_dim=64, target_dim=8, total_steps=10000)
        """

        if not isinstance(self.symtorch_block, nn.Module):
            raise ValueError("❌ Pruning only works on PyTorch MLPs, not callable functions.")

        self.initial_dim = initial_dim
        self.current_dim = initial_dim
        self.target_dim = target_dim

        self.pruning_schedule = self._set_pruning_schedule(total_steps, decay_rate, end_step_frac)
        self.register_buffer('pruning_mask', torch.ones(self.current_dim, dtype=torch.bool))

        print(f"✅ Pruning successfully set up for block {self.block_name}.")
        print(f"   Initial dimensions: {initial_dim}")
        print(f"   Target dimensions: {target_dim}")
        print(f"   Total steps: {total_steps}")
        print(f"   Pruning will complete at step {int(end_step_frac * total_steps)}")

        return None


    def _set_pruning_schedule(self, total_steps: int, decay_rate: str = 'cosine', end_step_frac: float = 0.5):
        """
        Create step-based pruning schedule.

        Args:
            total_steps (int): Total number of training steps
            decay_rate (str): Type of decay schedule ('exp', 'linear', 'cosine')
            end_step_frac (float): Fraction of steps to complete pruning by

        Returns:
            dict: Mapping from step number to target dimensions
        """

        prune_end_step = int(end_step_frac * total_steps)
        prune_steps = prune_end_step

        dims_to_prune = self.initial_dim - self.target_dim
        schedule_dict = {}

        # Different pruning schedules
        # Exponential decay
        if decay_rate == 'exp':
            decay_rate_val = 3.0
            max_decay = 1 - math.exp(-decay_rate_val)

            for step in range(prune_end_step):
                progress = step / prune_steps
                raw_decay = 1 - math.exp(-decay_rate_val * progress)
                decay_factor = raw_decay / max_decay

                dims_pruned = math.ceil(dims_to_prune * decay_factor)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[step] = target_dims

        # Linear decay
        elif decay_rate == 'linear':
            for step in range(prune_end_step):
                progress = step / prune_steps
                dims_pruned = math.ceil(dims_to_prune * progress)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[step] = target_dims

        # Cosine decay
        elif decay_rate == 'cosine':
            for step in range(prune_end_step):
                progress = step / prune_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                dims_pruned = math.ceil(dims_to_prune * (1 - cosine_decay))
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[step] = target_dims

        # Keep target_dim for the last part of training
        for step in range(prune_end_step, total_steps):
            schedule_dict[step] = self.target_dim

        return schedule_dict
    
    def prune(self, step: int, sample_data: torch.Tensor, parent_model=None):
        """
        Perform pruning for the current training step based on the pruning schedule.

        Evaluates the importance of each output dimension by computing the standard deviation
        of activations across the sample data. Retains the most important dimensions according
        to the current step's target dimensionality.

        Args:
            step (int): Current training step
            sample_data (torch.Tensor): Sample input data to evaluate dimension importance.
                                       Typically a subset of validation data.
            parent_model (nn.Module, optional): The parent model containing this SymbolicModel instance.
                                              If provided, will trace intermediate activations to get
                                              the actual outputs at this layer level for importance evaluation.

        Note:
            This method should be called during training steps. If the current step
            is not in the pruning schedule, no pruning is performed.

        Example:
            >>> for step in range(total_steps):
            >>>     # ... training code ...
            >>>     if step % prune_every == 0:
            >>>         model.block.prune(step, validation_data)
        """

        if not hasattr(self, 'pruning_schedule') or self.pruning_schedule is None:
            raise RuntimeError('Pruning schedule is not set. Call setup_pruning() first.')

        if step not in self.pruning_schedule:
            return

        target_dims = self.pruning_schedule[step]

        with torch.no_grad():
            # Extract outputs at this layer level for importance evaluation
            if parent_model is not None:
                with self._capture_layer_output(parent_model, sample_data) as (_, layer_outputs):
                    pass

                # Use captured intermediate data
                if layer_outputs:
                    output_array = layer_outputs[0]
                else:
                    raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicModel instance.")
            else:
                # Original behavior - use block directly
                self.symtorch_block.eval()
                output_array = self.symtorch_block(sample_data)

            output_importance = output_array.std(dim=0)
            most_important = torch.argsort(output_importance, descending=True)[:target_dims]

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

                        # Evaluate the equation for this dimension (torch backend, stays on device)
                        result = equation_func(*selected_inputs)

                        # Convert to tensor if needed (torch backend may return Python scalars for constants)
                        if not isinstance(result, torch.Tensor):
                            result = torch.tensor(result, dtype=x_torch.dtype, device=x_torch.device)

                        # Ensure result is 1D (batch_size,)
                        if result.dim() == 0:
                            result = result.expand(batch_size)
                        elif result.dim() > 1:
                            result = result.flatten()

                        output[:, dim] = result

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

                    # Evaluate the equation for this dimension (torch backend, stays on device)
                    result = equation_func(*selected_inputs)

                    # Convert to tensor if needed (torch backend may return Python scalars for constants)
                    if not isinstance(result, torch.Tensor):
                        result = torch.tensor(result, dtype=x_torch.dtype, device=x_torch.device)

                    # Ensure result is 1D (batch_size,)
                    if result.dim() == 0:
                        result = result.expand(batch_size)
                    elif result.dim() > 1:
                        result = result.flatten()

                    outputs.append(result)

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

    def clear_cache(self):
        """
        Clear cached I/O data from previous distill calls.

        This method removes all cached input/output data that was stored during
        previous distill() calls. Use this when you want to force a fresh forward
        pass and data extraction on the next distill() call, or to free up memory.

        The cache is used to avoid redundant forward passes when running distill()
        multiple times with the same inputs. Clearing the cache ensures that the
        next distill() call will perform a fresh forward pass through the model/function.

        Examples:
            >>> # First distill call - performs forward pass and caches data
            >>> model.distill(training_data)

            >>> # Second distill call with same data - uses cache
            >>> model.distill(training_data)  # Prints "Cache hit!"

            >>> # Clear the cache
            >>> model.clear_cache()

            >>> # Next distill call will perform fresh forward pass
            >>> model.distill(training_data)  # No cache hit message

            >>> # Clear cache to free memory after distillation
            >>> model.distill(large_dataset)
            >>> model.clear_cache()  # Free up memory used by cached data
        """
        self.distill_data = None
        self.distill_data_slime = None
        print(f"✅ Cache cleared for {self.block_name}.")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Save SymbolicModel state to state dict using PyTorch's built-in mechanism.

        This method is automatically called by state_dict() and enables users to save
        models using standard PyTorch patterns:
            torch.save(model.state_dict(), 'model.pth')

        Saves:
            - PyTorch parameters and buffers (handled by parent class)
            - Metadata (block_name, output_dims, etc.)
            - PySR regressors (serialized with dill)
            - SLIME regressors (serialized with dill)
            - Pruning state
            - Equation mode state
            - Variable transforms (serialized with dill)

        Note:
            Variable transforms (_variable_transforms) are serialized using dill.
            If serialization fails, a warning is issued and transforms will need
            to be re-provided after loading.
        """
        # Call parent to save parameters and buffers (including pruning_mask)
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # Note: We DO save _original_block if it exists (needed for switch_to_block())
        # Save metadata
        metadata = {
            'block_name': self.block_name,
            'output_dims': getattr(self, 'output_dims', None),
            '_variable_names': getattr(self, '_variable_names', None),
            '_using_equation': getattr(self, '_using_equation', False),
            '_equation_vars': getattr(self, '_equation_vars', {}),
        }

        # Try to serialize variable transforms with dill
        if hasattr(self, '_variable_transforms') and self._variable_transforms is not None:
            try:
                metadata['_variable_transforms'] = dill.dumps(self._variable_transforms)
                metadata['_variable_transforms_serialized'] = True
            except Exception as e:
                warnings.warn(
                    f"Could not serialize variable transforms for '{self.block_name}': {e}. "
                    "Transforms will need to be re-provided after loading."
                )
                metadata['_variable_transforms_serialized'] = False
        else:
            metadata['_variable_transforms_serialized'] = False

        # Add pruning metadata if present
        if hasattr(self, 'pruning_schedule') and self.pruning_schedule is not None:
            metadata.update({
                'initial_dim': self.initial_dim,
                'target_dim': self.target_dim,
                'current_dim': self.current_dim,
                'pruning_schedule': self.pruning_schedule,
            })

        destination[prefix + '_symtorch_metadata'] = metadata

        # Save PySR regressors (serialize with dill)
        if hasattr(self, 'pysr_regressor') and self.pysr_regressor:
            for dim, regressor in self.pysr_regressor.items():
                key = f'_pysr_regressor_dim_{dim}'
                try:
                    destination[prefix + key] = dill.dumps(regressor)
                except Exception as e:
                    warnings.warn(f"Could not serialize PySR regressor for dimension {dim}: {e}")

        # Save SLIME PySR regressors
        if hasattr(self, 'SLIME_pysr_regressor') and self.SLIME_pysr_regressor:
            for dim, regressor in self.SLIME_pysr_regressor.items():
                key = f'_slime_regressor_dim_{dim}'
                try:
                    destination[prefix + key] = dill.dumps(regressor)
                except Exception as e:
                    warnings.warn(f"Could not serialize SLIME regressor for dimension {dim}: {e}")

        # Store list of regressor dimensions for easier reconstruction
        destination[prefix + '_pysr_dims'] = list(self.pysr_regressor.keys()) if hasattr(self, 'pysr_regressor') else []
        destination[prefix + '_slime_dims'] = list(self.SLIME_pysr_regressor.keys()) if hasattr(self, 'SLIME_pysr_regressor') else []

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Load SymbolicModel state from state dict using PyTorch's built-in mechanism.

        This method is automatically called by load_state_dict() and enables users to load
        models using standard PyTorch patterns:
            model.load_state_dict(torch.load('model.pth'))

        Restores:
            - PyTorch parameters and buffers (handled by parent class)
            - Metadata
            - PySR regressors
            - SLIME regressors
            - Pruning state
            - Equation functions (rebuilt from regressors)
            - Variable transforms (deserialized with dill)

        Note:
            Variable transforms are restored from dill serialization if available.
            If deserialization fails or transforms weren't serialized, they must
            be re-provided by the user if needed for equation mode.
        """
        # Load metadata first
        metadata_key = prefix + '_symtorch_metadata'
        if metadata_key in state_dict:
            metadata = state_dict.pop(metadata_key)

            # Restore basic metadata
            self.block_name = metadata.get('block_name', self.block_name)
            self.output_dims = metadata.get('output_dims')
            self._variable_names = metadata.get('_variable_names')
            self._using_equation = metadata.get('_using_equation', False)
            self._equation_vars = metadata.get('_equation_vars', {})

            # Restore pruning metadata if present
            if 'initial_dim' in metadata:
                self.initial_dim = metadata['initial_dim']
                self.target_dim = metadata['target_dim']
                self.current_dim = metadata['current_dim']
                self.pruning_schedule = metadata['pruning_schedule']

                # Register pruning_mask buffer if not already registered
                # This allows loading models with pruning without calling setup_pruning first
                if not hasattr(self, 'pruning_mask'):
                    self.register_buffer('pruning_mask', torch.ones(self.initial_dim, dtype=torch.bool))

            # Restore variable transforms if they were serialized
            if metadata.get('_variable_transforms_serialized', False):
                try:
                    self._variable_transforms = dill.loads(metadata['_variable_transforms'])
                except Exception as e:
                    warnings.warn(
                        f"Could not deserialize variable transforms for '{self.block_name}': {e}. "
                        "You must re-provide variable_transforms if you need to use equation mode."
                    )
                    self._variable_transforms = None
            else:
                self._variable_transforms = None

        # Load PySR regressors
        pysr_dims_key = prefix + '_pysr_dims'
        if pysr_dims_key in state_dict:
            pysr_dims = state_dict.pop(pysr_dims_key)
            self.pysr_regressor = {}

            for dim in pysr_dims:
                key = prefix + f'_pysr_regressor_dim_{dim}'
                if key in state_dict:
                    try:
                        self.pysr_regressor[dim] = dill.loads(state_dict.pop(key))
                    except Exception as e:
                        error_msgs.append(f"Could not load PySR regressor for dimension {dim}: {e}")
        else:
            self.pysr_regressor = {}

        # Load SLIME regressors
        slime_dims_key = prefix + '_slime_dims'
        if slime_dims_key in state_dict:
            slime_dims = state_dict.pop(slime_dims_key)
            self.SLIME_pysr_regressor = {}

            for dim in slime_dims:
                key = prefix + f'_slime_regressor_dim_{dim}'
                if key in state_dict:
                    try:
                        self.SLIME_pysr_regressor[dim] = dill.loads(state_dict.pop(key))
                    except Exception as e:
                        error_msgs.append(f"Could not load SLIME regressor for dimension {dim}: {e}")
        else:
            self.SLIME_pysr_regressor = {}

        # Initialize cache as None (not serialized)
        self.distill_data = None
        self.distill_data_slime = None

        # Check if state_dict contains _original_block (means model was in equation mode)
        has_original_block = any(key.startswith(prefix + '_original_block.') for key in state_dict.keys())

        # _original_block IS saved automatically by PyTorch (it's an nn.Module).
        # We create a placeholder here BEFORE calling parent's load_state_dict so PyTorch
        # knows where to load the saved _original_block weights. Without this, strict mode
        # would complain about unexpected keys in the state dict.
        if has_original_block and self._using_equation:
            import copy
            # Create a placeholder _original_block that will be populated by parent's load
            self._original_block = copy.deepcopy(self.symtorch_block)

        # Call parent to load parameters and buffers (including _original_block if present)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

        # Rebuild equation functions if model was in equation mode
        if self._using_equation and self._equation_vars:
            try:
                self._rebuild_equation_funcs()
            except Exception as e:
                warnings.warn(
                    f"Model was saved in equation mode but equations could not be rebuilt: {e}. "
                    f"Switching to block mode."
                )
                self._using_equation = False
                self._equation_funcs = {}

    def _rebuild_equation_funcs(self):
        """
        Rebuild lambdified equation functions from loaded PySR regressors.

        Called during load_state_dict when model was saved in equation mode.
        Attempts to reconstruct _equation_funcs from the stored regressors.

        Raises:
            RuntimeError: If equations cannot be rebuilt from regressors
        """
        if not hasattr(self, '_equation_vars') or not self._equation_vars:
            raise RuntimeError("Cannot rebuild equations: _equation_vars not found")

        self._equation_funcs = {}

        for dim, var_indices in self._equation_vars.items():
            # Get equation from regressor
            result = self._get_equation(dim, complexity=None, SLIME=False)
            if result is None:
                raise RuntimeError(f"Cannot rebuild equation for dimension {dim}")

            equation_func, vars_sorted = result
            self._equation_funcs[dim] = equation_func
