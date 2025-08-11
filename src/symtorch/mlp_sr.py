"""
InterpretSR MLP_SR Module

This module provides a wrapper for PyTorch MLP models that adds symbolic regression
capabilities using PySR (Python Symbolic Regression).
"""

from pysr import *
import torch 
import torch.nn as nn
import time
import sympy
from sympy import lambdify
import numpy as np
from typing import List, Callable, Optional, Union, Dict, Any

class MLP_SR(nn.Module):
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
        >>> from interpretsr.mlp_sr import MLP_SR
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
                    with MLP_SR and provide a label
        >>> model = SimpleModel(input_dim=5, output_dim=1) # Initialise the model
        >>> # Train the model normally
        >>> model = training_function(model, dataloader, num_steps)
        >>> 
        >>> # Wrap the mlp with the MLP_SR wrapper
        >>> model.mlp = MLP_SR(model.mlp, mlp_name = "Sequential") # Wrap the mlp 
        >>> # Apply symbolic regression to the inputs and outputs of the MLP
        >>> regressor = model.mlp.interpret(inputs)
        >>> 
        >>> # Switch to using the symbolic equation instead of the MLP in the forwards 
            pass of your deep learning model
        >>> model.switch_to_equation()
        >>> # Switch back to using the MLP in the forwards pass
        >>> model.switch_to_mlp()
    """
    
    def __init__(self, mlp: nn.Module, mlp_name: str = None):
        """
        Initialise the MLP_SR wrapper.
        
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

    def interpret(self, inputs, output_dim: int = None, parent_model=None, 
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
            parent_model (nn.Module, optional): The parent model containing this MLP_SR instance.
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
            >>> regressor = model.interpret(train_inputs, 
            ...                            sr_params={'niterations': 1000})
            >>> print(regressor.get_best()['equation'])
            
            >>> # With variable transformations
            >>> transforms = [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2, lambda x: torch.sin(x[:, 3])]
            >>> names = ["x0_minus_x1", "x2_squared", "sin_x3"]
            >>> regressor = model.interpret(train_inputs, 
            ...                            variable_transforms=transforms, 
            ...                            fit_params={'variable_names': names})
        """

        # Extract inputs and outputs at this layer level
        if parent_model is not None:
            # Use forward hooks to capture inputs/outputs at this specific layer
            layer_inputs = []
            layer_outputs = []
            
            def hook_fn(module, input, output):
                if module is self.InterpretSR_MLP:
                    layer_inputs.append(input[0].clone())
                    layer_outputs.append(output.clone())
            
            # Register forward hook
            hook = self.InterpretSR_MLP.register_forward_hook(hook_fn)
            
            # Run parent model to capture intermediate activations
            parent_model.eval()
            with torch.no_grad():
                _ = parent_model(inputs)
            
            # Remove hook
            hook.remove()
            
            # Use captured intermediate data
            if layer_inputs and layer_outputs:
                actual_inputs = layer_inputs[0]
                output = layer_outputs[0]
            else:
                raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this MLP_SR instance.")
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
            # Validate inputs
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
            self._variable_names = None

        timestamp = int(time.time())

        output_dims = output.shape[1] # Number of output dimensions
        self.output_dims = output_dims # Save this 

        pysr_regressors = {}
        
        # Extract sr_params with defaults
        if sr_params is None:
            sr_params = {}

        if not output_dim:

            for dim in range(output_dims):

                print(f"🛠️ Running SR on output dimension {dim} of {output_dims-1}")
        
                run_id = f"dim{dim}_{timestamp}"
                output_name = f"SR_output/{self.mlp_name}"

                if save_path is not None:
                    output_name = f"{save_path}/{self.mlp_name}"
                
                default_sr_params = {
                    "binary_operators": ["+", "*"],
                    "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
                    "extra_sympy_mappings": {"inv": lambda x: 1/x},
                    "niterations": 400,
                    "complexity_of_operators": {"sin": 3, "exp":3},
                    "output_directory": output_name,
                    "run_id": run_id
                }
                
                
                final_sr_params = {**default_sr_params, **sr_params}
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
            output_name = f"SR_output/{self.mlp_name}"

            if save_path is not None:
                output_name = f"{save_path}/{self.mlp_name}"
            
            default_sr_params = {
                "binary_operators": ["+", "*"],
                "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
                "extra_sympy_mappings": {"inv": lambda x: 1/x},
                "niterations": 400,
                "complexity_of_operators": {"sin": 3, "exp":3},
                "output_directory": output_name,
                "run_id": run_id
            }
                
            final_sr_params = {**default_sr_params, **sr_params}
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
            print("❗No equations found for this MLP yet. You need to first run .interpret to find the best equation to fit this MLP.")
            return None
        if dim not in self.pysr_regressor:
            print(f"❗No equation found for output dimension {dim}. You need to first run .interpret.")
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
        f = lambdify(vars_sorted, expr, "numpy")
        return f, vars_sorted

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
            print("❗No equations found for this MLP yet. You need to first run .interpret.")
            return
        
        if not hasattr(self, 'output_dims'):
            print("❗No output dimension information found. You need to first run .interpret.")
            return
        
        # Check that we have equations for all output dimensions
        missing_dims = []
        for dim in range(self.output_dims):
            if dim not in self.pysr_regressor:
                missing_dims.append(dim)
        
        if missing_dims:
            print(f"❗Missing equations for dimensions {missing_dims}. You need to run .interpret on all output dimensions first.")
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
            
            # Handle variable indices based on whether transformations were used
            if hasattr(self, '_variable_transforms') and self._variable_transforms is not None:
                # With transformations, variables are named by custom names or transform indices
                var_indices = []
                for var in vars_sorted:
                    var_str = str(var)
                    if self._variable_names:
                        # Find the index based on custom variable names
                        try:
                            idx = self._variable_names.index(var_str)
                            var_indices.append(idx)
                        except ValueError:
                            print(f"⚠️ Warning: Variable {var_str} not found in variable_names for dimension {dim}")
                            return
                    else:
                        # Variables named as x0, x1, etc. based on transform index
                        if var_str.startswith('x'):
                            try:
                                idx = int(var_str[1:])
                                var_indices.append(idx)
                            except ValueError:
                                print(f"⚠️ Warning: Could not parse variable {var_str} for dimension {dim}")
                                return
                        else:
                            print(f"⚠️ Warning: Unexpected variable format {var_str} for dimension {dim}")
                            return
            else:
                # Original behavior for non-transformed variables
                var_indices = []
                for var in vars_sorted:
                    var_str = str(var)
                    if var_str.startswith('x'):
                        try:
                            idx = int(var_str[1:])
                            var_indices.append(idx)
                        except ValueError:
                            print(f"⚠️ Warning: Could not parse variable {var_str} for dimension {dim}")
                            return
                    else:
                        print(f"⚠️ Warning: Unexpected variable format {var_str} for dimension {dim}")
                        return
            
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
        if hasattr(self, '_original_mlp'):
            self.InterpretSR_MLP = self._original_mlp
            self._using_equation = False
            print(f"✅ Switched {self.mlp_name} back to MLP")
        else:
            print("❗ No original MLP stored to switch back to")