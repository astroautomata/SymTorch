"""
InterpretSR SymbolicMLP Module

This module provides a wrapper for PyTorch MLP models that adds symbolic regression
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

class SymbolicMLP(nn.Module):
    """
    A PyTorch module wrapper that adds symbolic regression capabilities to MLPs.
    
    This class wraps any PyTorch MLP (Multi-Layer Perceptron) and provides methods
    to discover symbolic expressions that approximate the learned neural network
    behavior using genetic algorithms supported by PySR.
    
    The wrapper maintains full compatibility with PyTorch's training pipeline while
    adding interpretability features through symbolic regression.
    
    Attributes:
        InterpretSR_MLP (nn.Module): The wrapped PyTorch MLP model
        mlp_name (str): Human-readable name for the MLP instance
        pysr_regressor (PySRRegressor): The fitted symbolic regression model
        
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from symtorch import SymbolicMLP
        >>>
        >>> # Create a model
        >>> class SimpleModel(nn.Module):
                def __init__(self, input_dim, output_dim, hidden_dim = 64):
                    super(SimpleModel, self).__init__()
                    mlp = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, output_dim)
                    )
                    self.mlp = mlp
        >>> model = SimpleModel(input_dim=5, output_dim=1) # Initialise the model
        >>> # Train the model normally
        >>> model = training_function(model, dataloader, num_steps)
        >>>
        >>> # Wrap the mlp with the SymbolicMLP wrapper
        >>> model.mlp = SymbolicMLP(model.mlp, mlp_name = "Sequential") # Wrap the mlp 
        >>> # Apply symbolic regression to the inputs and outputs of the MLP
        >>> regressor = model.mlp.distill(inputs)
        >>> 
        >>> # Switch to using the symbolic equation instead of the MLP in the forwards 
            pass of your deep learning model
        >>> model.switch_to_equation()
        >>> # Switch back to using the MLP in the forwards pass
        >>> model.switch_to_mlp()
    """
    
    # Default PySR parameters
    DEFAULT_SR_PARAMS = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "niterations": 400,
        "complexity_of_operators": {"sin": 3, "exp": 3}
    }
    
    def __init__(self, mlp: nn.Module, mlp_name: str = None):
        """
        Initialise the SymbolicMLP wrapper.

        Args:
            mlp (nn.Module): The PyTorch MLP model to wrap
            mlp_name (str, optional): Human-readable name for this MLP instance.
                                    If None, generates a unique name based on object ID.
        """
        super().__init__()
        self.InterpretSR_MLP = mlp
        self.mlp_name = mlp_name or f"mlp_{id(self)}"
        if not mlp_name: 
            print(f"➡️ No MLP name specified. MLP label is {self.mlp_name}.")
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
        output_name = f"SR_output/{self.mlp_name}"
        if save_path is not None:
            output_name = f"{save_path}/{self.mlp_name}"
        
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
            if module is self.InterpretSR_MLP: # Only captures layer data for the layers we want to distil
                layer_inputs.append(input[0].clone())
                layer_outputs.append(output.clone())
        
        # Register forward hook
        hook = self.InterpretSR_MLP.register_forward_hook(hook_fn)
        
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
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Automatically switches between MLP and symbolic equations based on current mode.
        When using symbolic equation mode, evaluates each output dimension separately
        using its corresponding symbolic expression.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If symbolic equations require variables not present in input
        """
        if hasattr(self, '_using_equation') and self._using_equation:
            batch_size = x.shape[0]
            output_dims = len(self._equation_funcs)
            
            # Initialize output tensor
            outputs = []
            
            # Evaluate each dimension separately
            for dim in range(output_dims):
                equation_func = self._equation_funcs[dim]
                var_indices = self._equation_vars[dim]
                
                # Extract variables needed for this dimension
                selected_inputs = self._extract_variables_for_equation(x, var_indices, dim)
                
                # Convert to numpy for the equation function
                numpy_inputs = [inp.detach().cpu().numpy() for inp in selected_inputs]
                
                # Evaluate the equation for this dimension
                result = equation_func(*numpy_inputs)
                
                # Convert back to torch tensor with same device/dtype as input
                result_tensor = torch.tensor(result, dtype=x.dtype, device=x.device)
                
                # Ensure result is 1D (batch_size,)
                if result_tensor.dim() == 0:
                    result_tensor = result_tensor.expand(batch_size)
                elif result_tensor.dim() > 1:
                    result_tensor = result_tensor.flatten()
                
                outputs.append(result_tensor)
            
            # Stack all dimensions to create (batch_size, output_dim) tensor
            result_tensor = torch.stack(outputs, dim=1)
            
            return result_tensor
        else:
            return self.InterpretSR_MLP(x)

    def distill(self, inputs, output_dim: int = None, parent_model=None,
                 variable_transforms: Optional[List[Callable]] = None,
                 save_path: str = None,
                 sr_params: Optional[Dict[str, Any]] = None,
                 fit_params: Optional[Dict[str, Any]] = None):
        """
        Discover symbolic expressions that approximate the MLP's behavior.
        
        Uses PySR to find mathematical expressions that best fit the input-output relationship learned by the neural network.
        
        Args:
            inputs (torch.Tensor): Input data for symbolic regression fitting
            output_dim(torch.Tensor): The output dimension to run PySR on. If None, PySR run on all outputs. Default: None.
            parent_model (nn.Module, optional): The parent model containing this SymbolicMLP instance.
                                              If provided, will trace intermediate activations to get
                                              the actual inputs/outputs at this layer level.
            variable_transforms (List[Callable], optional): List of functions to transform input variables.
                                                           Each function should take the full input tensor and return
                                                           a transformed tensor. Example: [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2]
            save_path (str, optional): Custom base directory for PySR outputs.
                                     If None, uses default "SR_output/" directory.
                                     Example: "/custom/output/path"
            sr_params (Dict[str, Any], optional): Parameters passed to PySRRegressor. Defaults:
                - binary_operators (list): ["+", "*"]
                - unary_operators (list): ["inv(x) = 1/x", "sin", "exp"]
                - niterations (int): 400
                - output_directory (str): "{save_path}/{mlp_name}" or "SR_output/{mlp_name}" # Where PySR outputs are stored
                - run_id (str): "{timestamp}" # Where PySR outputs of a specific run are stored
                To see more information on the possible inputs to the PySRRegressor, please see
                the PySR documentation.
            fit_params (Dict[str, Any], optional): Parameters passed to the regressor.fit() method. Defaults:
                - variable_names (List[str]): Custom names for variables if variable_transforms is used.
                                             If provided, must match the length of variable_transforms.
                                             Example: ["x0_minus_x1", "x2_squared"]
                
        Returns:
            PySRRegressor: Fitted symbolic regression model
            
        Example:
            >>> # Basic usage
            >>> model.mlp.distill(sample_inputs, 
            ...                            sr_params={'niterations': 1000})
            
            >>> # With variable transformations
            >>> transforms = [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2, lambda x: torch.sin(x[:, 3])]
            >>> names = ["x0_minus_x1", "x2_squared", "sin_x3"]
            >>> model.distill(train_inputs, 
            ...               variable_transforms=transforms, 
            ...               fit_params={'variable_names': names})
        """

        # Extract inputs and outputs at this layer level
        if parent_model is not None:
            with self._capture_layer_output(parent_model, inputs) as (layer_inputs, layer_outputs):
                pass
            
            # Use captured intermediate data
            if layer_inputs and layer_outputs:
                actual_inputs = layer_inputs[0]
                output = layer_outputs[0]
            else:
                raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicMLP instance.")
        else:
            # Original behavior - use MLP directly
            actual_inputs = inputs
            self.InterpretSR_MLP.eval()
            with torch.no_grad():
                output = self.InterpretSR_MLP(inputs)

        # Extract fit parameters
        if fit_params is None:
            fit_params = {}
        
        variable_names = fit_params.get('variable_names', None)
        
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
        
        # Extract sr_params with defaults
        if sr_params is None:
            sr_params = {}

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
            
        print(f"❤️ SR on {self.mlp_name} complete.")
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
            This is an internal method. Use switch_to_equation() for public API.
        """
        if not hasattr(self, 'pysr_regressor') or self.pysr_regressor is None:
            print("❗No equations found for this MLP yet. You need to first run .distill to find the best equation to fit this MLP.")
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

    def switch_to_equation(self, complexity: list = None):
        """
        Switch the forward pass from MLP to symbolic equations for all output dimensions.
        
        After calling this method, the model will use the discovered symbolic
        expressions instead of the neural network for forward passes. This requires
        equations to be available for ALL output dimensions.
        
        Args:
            complexity (list, optional): Specific complexity levels to use for each dimension.
                                      If None, uses the best overall equation for each dimension.
            
        Example:
            >>> model.switch_to_equation(complexity=5)

        """
        if not hasattr(self, 'pysr_regressor') or not self.pysr_regressor:
            print("❗No equations found for this MLP yet. You need to first run .distill.")
            return
        
        if not hasattr(self, 'output_dims'):
            print("❗No output dimension information found. You need to first run .distill.")
            return
        
        # Check that we have equations for all output dimensions
        missing_dims = []
        for dim in range(self.output_dims):
            if dim not in self.pysr_regressor:
                missing_dims.append(dim)
        
        if missing_dims:
            print(f"❗Missing equations for dimensions {missing_dims}. You need to run .distill on all output dimensions first.")
            print(f"Available dimensions: {list(self.pysr_regressor.keys())}")
            print(f"Required dimensions: {list(range(self.output_dims))}")
            return
        
        # Store original MLP for potential restoration
        if not hasattr(self, '_original_mlp'):
            self._original_mlp = self.InterpretSR_MLP
        
        # Get equations for all dimensions
        equation_funcs = {}
        equation_vars = {}
        equation_strs = {}
        
        for dim in range(self.output_dims):
            # Get complexity for this specific dimension
            dim_complexity = None
            if complexity is not None:
                if isinstance(complexity, list):
                    if dim < len(complexity):
                        dim_complexity = complexity[dim]
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
        print(f"✅ Successfully switched {self.mlp_name} to symbolic equations for all {self.output_dims} dimensions:")
        for dim in range(self.output_dims):
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
        
        print(f"🎯 All {self.output_dims} output dimensions now using symbolic equations.")
   
    def switch_to_mlp(self):
        """
        Switch back to using the original MLP for forward passes.
        
        Restores the neural network as the primary forward pass mechanism,
        reverting any previous switch_to_equation() call.
            
        Example:
            >>> model.switch_to_equation()  # Use symbolic equation
            >>> # ... do some analysis ...
            >>> model.switch_to_mlp()       # Switch back to neural network
        """
        self._using_equation = False
        if hasattr(self, '_original_mlp'):
            self.InterpretSR_MLP = self._original_mlp
        print(f"✅ Switched {self.mlp_name} back to MLP")

    def save_model(self, save_path: str, save_pytorch: bool = True, save_regressors: bool = True):
        """
        Save the SymbolicMLP model including PyTorch weights and PySR regressors.

        Creates a comprehensive save that includes:
        - PyTorch model state dict (if save_pytorch=True)
        - All fitted PySR regressors (if save_regressors=True)
        - Model metadata and configuration
        - Variable transforms and names if used

        Args:
            save_path (str): Base path for saving (without extension)
            save_pytorch (bool, optional): Whether to save PyTorch model state. Defaults to True.
            save_regressors (bool, optional): Whether to save PySR regressors. Defaults to True.

        Example:
            >>> model.mlp = SymbolicMLP(model.mlp, mlp_name="encoder")
            >>> # ... train and run distill ...
            >>> model.mlp.save_model("./saved_models/my_model")
            
        Note:
            This creates multiple files:
            - {save_path}_pytorch.pth: PyTorch model state
            - {save_path}_metadata.pkl: Model configuration and metadata
            - {save_path}_regressor_dim{i}.pkl: Individual PySR regressors (one per dimension)
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        saved_files = []
        
        # Save PyTorch model state
        if save_pytorch:
            pytorch_path = f"{save_path}_pytorch.pth"
            torch.save(self.InterpretSR_MLP.state_dict(), pytorch_path)
            saved_files.append(pytorch_path)
            print(f"✅ Saved PyTorch model state to {pytorch_path}")
        
        # Save model metadata
        metadata = {
            'mlp_name': self.mlp_name,
            'output_dims': getattr(self, 'output_dims', None),
            'variable_transforms_available': hasattr(self, '_variable_transforms') and self._variable_transforms is not None,
            'variable_names': getattr(self, '_variable_names', None),
            'using_equation': getattr(self, '_using_equation', False),
            'class_name': self.__class__.__name__,
            'equation_vars': getattr(self, '_equation_vars', {}),
            'regressor_dimensions': list(self.pysr_regressor.keys()) if hasattr(self, 'pysr_regressor') else []
        }
        
        metadata_path = f"{save_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        saved_files.append(metadata_path)
        print(f"✅ Saved model metadata to {metadata_path}")
        
        # Save PySR regressors
        if save_regressors and hasattr(self, 'pysr_regressor') and self.pysr_regressor:
            regressor_files = []
            for dim, regressor in self.pysr_regressor.items():
                regressor_path = f"{save_path}_regressor_dim{dim}.pkl"
                try:
                    # Use PySR's built-in pickling support
                    with open(regressor_path, 'wb') as f:
                        pickle.dump(regressor, f)
                    regressor_files.append(regressor_path)
                    saved_files.append(regressor_path)
                    print(f"✅ Saved regressor for dimension {dim} to {regressor_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save regressor for dimension {dim}: {e}")
            
            if regressor_files:
                print(f"✅ Saved {len(regressor_files)} PySR regressors")
        elif save_regressors:
            print("ℹ️ No PySR regressors found to save")
        
        print(f"🎯 Model save complete. Created {len(saved_files)} files with base name: {save_path}")
        return saved_files

    @classmethod
    def load_model(cls, save_path: str, mlp_architecture: nn.Module = None, device: str = 'cpu'):
        """
        Load a previously saved SymbolicMLP model with all components.

        Reconstructs the complete SymbolicMLP instance including:
        - PyTorch model weights (requires architecture)
        - All fitted PySR regressors
        - Model metadata and configuration
        - Variable transforms setup

        Args:
            save_path (str): Base path used during saving (without extension)
            mlp_architecture (nn.Module, optional): PyTorch model architecture to load weights into.
                                                   If None, only metadata and regressors are loaded.
            device (str, optional): Device to load tensors to ('cpu', 'cuda', etc.). Defaults to 'cpu'.

        Returns:
            SymbolicMLP: Reconstructed SymbolicMLP instance with loaded components

        Example:
            >>> # Create same architecture as original
            >>> mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 1))
            >>> loaded_model = SymbolicMLP.load_model("./saved_models/my_model", mlp)
            >>> # Model ready to use with equations
            >>> loaded_model.switch_to_equation()
            
        Note:
            The mlp_architecture must match the original architecture exactly for weight loading.
            If architecture is not provided, returns a model instance with regressors but no PyTorch weights.
        """
        # Load metadata first
        metadata_path = f"{save_path}_metadata.pkl"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📂 Loading {metadata['class_name']} model: {metadata['mlp_name']}")
        
        # Create instance based on class type
        if metadata['class_name'] == 'PruningMLP':
            # Import here to avoid circular imports
            from .toolkit import PruningMLP

            # Need to determine dimensions for PruningMLP
            if mlp_architecture is None:
                raise ValueError("mlp_architecture is required when loading PruningMLP")

            # Try to infer dimensions from saved state
            output_dims = metadata.get('output_dims', None)
            if output_dims is None:
                raise ValueError("Cannot determine output dimensions for PruningMLP")

            # Create minimal PruningMLP - dimensions will be updated from saved state
            instance = PruningMLP(mlp_architecture,
                                  initial_dim=output_dims,
                                  target_dim=1,  # Will be updated
                                  mlp_name=metadata['mlp_name'])
        else:
            # Standard SymbolicMLP
            instance = cls(mlp_architecture or nn.Identity(), metadata['mlp_name'])
        
        # Load PyTorch weights if available and architecture provided
        pytorch_path = f"{save_path}_pytorch.pth"
        if os.path.exists(pytorch_path) and mlp_architecture is not None:
            state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
            instance.InterpretSR_MLP.load_state_dict(state_dict)
            instance.InterpretSR_MLP.eval()  # Ensure model is in eval mode after loading
            print(f"✅ Loaded PyTorch weights from {pytorch_path}")
        elif mlp_architecture is not None:
            print(f"⚠️ PyTorch weights file not found: {pytorch_path}")
        
        # Restore metadata
        instance.output_dims = metadata.get('output_dims')
        instance._variable_names = metadata.get('variable_names')
        instance._using_equation = metadata.get('using_equation', False)
        instance._equation_vars = metadata.get('equation_vars', {})
        
        # Load PySR regressors
        regressor_dims = metadata.get('regressor_dimensions', [])
        instance.pysr_regressor = {}
        equation_funcs = {}
        
        loaded_regressors = 0
        for dim in regressor_dims:
            regressor_path = f"{save_path}_regressor_dim{dim}.pkl"
            if os.path.exists(regressor_path):
                try:
                    with open(regressor_path, 'rb') as f:
                        regressor = pickle.load(f)
                    instance.pysr_regressor[dim] = regressor
                    
                    # Rebuild equation function if model was using equations
                    if instance._using_equation and dim in instance._equation_vars:
                        try:
                            result = instance._get_equation(dim)
                            if result is not None:
                                equation_funcs[dim] = result[0]
                        except Exception as e:
                            print(f"⚠️ Warning: Could not rebuild equation for dimension {dim}: {e}")
                            # Don't use equations for this dimension if we can't rebuild it
                            pass
                    
                    loaded_regressors += 1
                    print(f"✅ Loaded regressor for dimension {dim}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load regressor for dimension {dim}: {e}")
            else:
                print(f"⚠️ Warning: Regressor file not found for dimension {dim}: {regressor_path}")
        
        if loaded_regressors > 0:
            print(f"✅ Loaded {loaded_regressors} PySR regressors")
            
            # Restore equation functions if model was using equations
            if instance._using_equation and equation_funcs:
                # Check if we have equation functions for all required dimensions
                required_dims = set(instance._equation_vars.keys())
                available_dims = set(equation_funcs.keys())
                
                if required_dims == available_dims:
                    instance._equation_funcs = equation_funcs
                    print(f"✅ Restored symbolic equation functions for {len(equation_funcs)} dimensions")
                else:
                    missing_dims = required_dims - available_dims
                    print(f"⚠️ Could not restore equations for dimensions {missing_dims}, switching to MLP mode")
                    instance._using_equation = False
            elif instance._using_equation:
                print("⚠️ Model was saved in equation mode but no equations could be restored, switching to MLP mode")
                instance._using_equation = False
        else:
            print("ℹ️ No PySR regressors found to load")
        
        print(f"🎯 Model loading complete: {metadata['mlp_name']}")
        return instance

    def get_importance(self, sample_data: torch.Tensor, parent_model=None):
        """
        Get ordered list of output dimensions from most to least important.

        Evaluates importance by computing standard deviation across sample data,
        with higher standard deviation indicating higher importance.

        Args:
            sample_data (torch.Tensor): Sample input data to evaluate dimension importance.
                                       Typically a subset of validation data.
            parent_model (nn.Module, optional): The parent model containing this SymbolicMLP instance.
                                              If provided, will trace intermediate activations to get
                                              the actual outputs at this layer level for importance evaluation.
                                              
        Returns:
            dict: Dictionary with keys:
                - 'importance': List of dimension indices ordered from most important to least important
                - 'std': List of standard deviation values corresponding to the ordered dimensions
            
        Example:
            >>> result = model.mlp.get_importance(validation_data)
            >>> print(f"Most important dimension: {result['importance'][0]} (std: {result['std'][0]})")
            >>> print(f"Least important dimension: {result['importance'][-1]} (std: {result['std'][-1]})")
        """
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
                self.InterpretSR_MLP.eval()
                output_array = self.InterpretSR_MLP(sample_data)

            # Calculate importance based on standard deviation
            output_importance = output_array.std(dim=0)
            # Sort from most important to least important (descending order)
            importance_indices = torch.argsort(output_importance, descending=True)
            
            # Get ordered importance values and dimension indices
            importance_order = importance_indices.tolist()
            std_values = output_importance[importance_indices].tolist()
            
            return {
                'importance': importance_order,
                'std': std_values
            }