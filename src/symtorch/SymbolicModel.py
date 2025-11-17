"""
InterpretSR SymbolicModel Module

This module provides a wrapper for entire PyTorch models that adds symbolic regression
capabilities using PySR (Python Symbolic Regression).
"""
import warnings
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
from pysr import PySRRegressor
import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Callable, Optional, Dict, Any


class SymbolicModel(nn.Module):
    """
    A PyTorch module wrapper that adds symbolic regression capabilities to entire models.

    This class wraps any PyTorch model and provides a method to discover symbolic expressions
    that approximate the model's input-output behavior using PySR. Unlike SymbolicMLP, this
    operates on the entire model rather than individual layers.

    Attributes:
        model (nn.Module): The wrapped PyTorch model
        model_name (str): Human-readable name for the model instance
        pysr_regressor (dict): Dictionary of fitted symbolic regression models per output dimension

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from symtorch import SymbolicModel
        >>>
        >>> # Create and train your model
        >>> model = MyComplexModel(input_dim=5, output_dim=2)
        >>> model = training_function(model, dataloader, num_steps)
        >>>
        >>> # Wrap the model with SymbolicModel
        >>> symbolic_model = SymbolicModel(model, model_name="MyModel")
        >>> # Apply symbolic regression to the inputs and outputs
        >>> regressor = symbolic_model.distill(sample_inputs)
    """

    # Default PySR parameters (same as SymbolicMLP)
    DEFAULT_SR_PARAMS = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "niterations": 400,
        "complexity_of_operators": {"sin": 3, "exp": 3}
    }

    def __init__(self, model: nn.Module, model_name: str = None):
        """
        Initialize the SymbolicModel wrapper.

        Args:
            model (nn.Module): The PyTorch model to wrap
            model_name (str, optional): Human-readable name for this model instance.
                                       If None, generates a unique name based on object ID.
        """
        super().__init__()
        self.model = model
        self.model_name = model_name or f"model_{id(self)}"
        if not model_name:
            print(f"No model name specified. Model label is {self.model_name}.")
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
        output_name = f"SR_output/{self.model_name}"
        if save_path is not None:
            output_name = f"{save_path}/{self.model_name}"

        base_params = {
            **self.DEFAULT_SR_PARAMS,
            "output_directory": output_name,
            "run_id": run_id
        }

        if custom_params:
            base_params.update(custom_params)

        return base_params

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor from the wrapped model
        """
        return self.model(x)

    def distill(self, inputs, output_dim: int = None,
                variable_transforms: Optional[List[Callable]] = None,
                save_path: str = None,
                sr_params: Optional[Dict[str, Any]] = None,
                fit_params: Optional[Dict[str, Any]] = None):
        """
        Discover symbolic expressions that approximate the entire model's behavior.

        Uses PySR to find mathematical expressions that best fit the model's input-output
        relationship. Unlike SymbolicMLP.distill(), this operates on the entire model
        without needing a parent_model parameter.

        Args:
            inputs (torch.Tensor): Input data for symbolic regression fitting
            output_dim (int, optional): The output dimension to run PySR on. If None, PySR runs on all outputs. Default: None.
            variable_transforms (List[Callable], optional): List of functions to transform input variables.
                                                           Each function should take the full input tensor and return
                                                           a transformed tensor. Example: [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2]
            save_path (str, optional): Custom base directory for PySR outputs.
                                     If None, uses default "SR_output/" directory.
            sr_params (Dict[str, Any], optional): Parameters passed to PySRRegressor. Defaults:
                - binary_operators (list): ["+", "*"]
                - unary_operators (list): ["inv(x) = 1/x", "sin", "exp"]
                - niterations (int): 400
                - output_directory (str): "{save_path}/{model_name}" or "SR_output/{model_name}"
                - run_id (str): "{timestamp}"
            fit_params (Dict[str, Any], optional): Parameters passed to the regressor.fit() method. Defaults:
                - variable_names (List[str]): Custom names for variables if variable_transforms is used.
                                             If provided, must match the length of variable_transforms.

        Returns:
            PySRRegressor or dict: Fitted symbolic regression model(s)

        Example:
            >>> # Basic usage
            >>> symbolic_model.distill(sample_inputs, sr_params={'niterations': 1000})

            >>> # With variable transformations
            >>> transforms = [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2]
            >>> names = ["x0_minus_x1", "x2_squared"]
            >>> symbolic_model.distill(train_inputs,
            ...                        variable_transforms=transforms,
            ...                        fit_params={'variable_names': names})
        """
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs)

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
                    transformed_var = transform_func(inputs)
                    # Ensure the result is 1D (batch_size,)
                    if transformed_var.dim() > 1:
                        transformed_var = transformed_var.flatten()
                    transformed_inputs.append(transformed_var.detach().cpu().numpy())
                except Exception as e:
                    raise ValueError(f"Error applying transformation {i}: {e}")

            # Stack transformed variables into input matrix
            inputs_numpy = np.column_stack(transformed_inputs)

            print(f"Applied {len(variable_transforms)} variable transformations")
            if variable_names:
                print(f"Variable names: {variable_names}")
        else:
            # Use original inputs
            inputs_numpy = inputs.detach().cpu().numpy()

        timestamp = int(time.time())

        # Handle both 1D and 2D outputs
        if output.dim() == 1:
            output = output.unsqueeze(1)

        output_dims = output.shape[1]  # Number of output dimensions
        self.output_dims = output_dims  # Save this

        pysr_regressors = {}

        # Extract sr_params with defaults
        if sr_params is None:
            sr_params = {}

        if not output_dim:
            # If output dimension is not specified, run SR on all dims
            for dim in range(output_dims):
                print(f"=Running SR on output dimension {dim} of {output_dims-1}")

                run_id = f"dim{dim}_{timestamp}"
                final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
                regressor = PySRRegressor(**final_sr_params)

                # Prepare fit arguments
                fit_args = [inputs_numpy, output.detach()[:, dim].cpu().numpy()]
                final_fit_params = dict(fit_params)  # Copy to avoid modifying original

                regressor.fit(*fit_args, **final_fit_params)

                pysr_regressors[dim] = regressor

                print(f"Best equation for output {dim} found to be {regressor.get_best()['equation']}.")

        else:
            print(f"Running SR on output dimension {output_dim}.")

            run_id = f"dim{output_dim}_{timestamp}"
            final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
            regressor = PySRRegressor(**final_sr_params)

            # Prepare fit arguments
            fit_args = [inputs_numpy, output.detach()[:, output_dim].cpu().numpy()]
            final_fit_params = dict(fit_params)  # Copy to avoid modifying original

            regressor.fit(*fit_args, **final_fit_params)
            pysr_regressors[output_dim] = regressor

            print(f"Best equation for output {output_dim} found to be {regressor.get_best()['equation']}.")

        print(f"SR on {self.model_name} complete.")
        self.pysr_regressor = self.pysr_regressor | pysr_regressors

        # For backward compatibility, return the regressor or dict of regressors
        if output_dim is not None:
            return pysr_regressors[output_dim]
        else:
            return pysr_regressors
