"""
InterpretSR SymbolicModel Module

This module provides a model-agnostic symbolic regression wrapper that can work with
any callable function using PySR (Python Symbolic Regression).
"""
import warnings
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
from pysr import PySRRegressor
import torch
import torch.nn as nn
import time
import numpy as np
from typing import List, Callable, Optional, Dict, Any, Union


class SymbolicModel:
    """
    A model-agnostic symbolic regression wrapper for discovering symbolic expressions.

    This class works with any callable function (PyTorch models, scikit-learn models,
    TensorFlow models, or any Python function) to discover symbolic expressions that
    approximate the function's input-output behavior using PySR.

    Attributes:
        pysr_regressor (list): List of fitted PySRRegressor objects per output dimension
        equations_ (list): List of equation DataFrames from PySR for each output dimension
        sr_params (dict): PySR parameters used for fitting

    Example:
        >>> from symtorch import SymbolicModel
        >>> import numpy as np
        >>>
        >>> # Define any callable function
        >>> def my_function(x):
        ...     return x[:, 0]**2 + 3*np.sin(x[:, 4]) - 4
        >>>
        >>> # Fit symbolic regression
        >>> symbolic_model = SymbolicModel()
        >>> symbolic_model.fit(my_function, training_data)
        >>>
        >>> # Get the discovered equation
        >>> equation = symbolic_model.get_equation()
        >>> print(equation)
        >>>
        >>> # Make predictions with the symbolic equation
        >>> predictions = symbolic_model.predict(test_data)

    Example with PyTorch:
        >>> import torch
        >>>
        >>> # Use with any PyTorch model
        >>> pytorch_model = MyComplexModel()
        >>>
        >>> def f(x):
        ...     with torch.no_grad():
        ...         return pytorch_model(torch.tensor(x, dtype=torch.float32))
        >>>
        >>> symbolic_model = SymbolicModel(sr_params={'niterations': 500})
        >>> symbolic_model.fit(f, training_data)
    """

    # Default PySR parameters
    DEFAULT_SR_PARAMS = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "niterations": 400,
        "complexity_of_operators": {"sin": 3, "exp": 3}
    }

    def __init__(self, sr_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the SymbolicModel.

        Args:
            sr_params (Dict[str, Any], optional): Custom PySR parameters to override defaults.
                Common parameters include:
                - niterations (int): Number of iterations for PySR (default: 400)
                - binary_operators (list): Binary operators to use (default: ["+", "*"])
                - unary_operators (list): Unary operators to use (default: ["inv(x) = 1/x", "sin", "exp"])
                - complexity_of_operators (dict): Complexity penalties for operators
        """
        self.sr_params = sr_params or {}
        self.pysr_regressor = None
        self.equations_ = None

    def _create_sr_params(self, save_path: Optional[str], run_id: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create SR parameters by merging defaults with custom parameters.

        Args:
            save_path (str, optional): Output directory path for SR results
            run_id (str): Unique run identifier
            custom_params (Dict[str, Any], optional): Custom parameters to override defaults

        Returns:
            Dict[str, Any]: Final SR parameters for PySRRegressor
        """
        output_name = "SR_output/SymbolicModel"
        if save_path is not None:
            output_name = save_path

        base_params = {
            **self.DEFAULT_SR_PARAMS,
            "output_directory": output_name,
            "run_id": run_id
        }

        if custom_params:
            base_params.update(custom_params)

        return base_params

    def fit(self, f: Callable, inputs: Union[np.ndarray, torch.Tensor],
            outputs: Optional[Union[np.ndarray, torch.Tensor]] = None,
            output_dim: Optional[int] = None,
            variable_transforms: Optional[List[Callable]] = None,
            variable_names: Optional[List[str]] = None,
            save_path: Optional[str] = None,
            **kwargs):
        """
        Fit symbolic regression to approximate a black-box function.

        Args:
            f (Callable): Black-box function that takes inputs and returns outputs.
                         Can be a model, function, or any callable.
            inputs (np.ndarray or torch.Tensor): Input data for symbolic regression fitting
            outputs (np.ndarray or torch.Tensor, optional): Pre-computed outputs. If None, will call f(inputs)
            output_dim (int, optional): Specific output dimension to run PySR on. If None, runs on all outputs.
            variable_transforms (List[Callable], optional): List of functions to transform input variables.
                                                           Each function should take the full input tensor and return
                                                           a transformed tensor. Example: [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2]
            variable_names (List[str], optional): Custom names for variables. If variable_transforms is used,
                                                 must match its length.
            save_path (str, optional): Custom directory for PySR outputs. If None, uses "SR_output/SymbolicModel"
            **kwargs: Additional parameters to override default PySR parameters

        Returns:
            self: Returns self for method chaining

        Example:
            >>> # Basic usage with any function
            >>> def f(x):
            ...     return x[:, 0]**2 + 3*np.sin(x[:, 4]) - 4
            >>>
            >>> symbolic_model = SymbolicModel()
            >>> symbolic_model.fit(f, training_data)
            >>> equation = symbolic_model.get_equation()

            >>> # With PyTorch model
            >>> pytorch_model = MyModel()
            >>> def f(x):
            ...     with torch.no_grad():
            ...         return pytorch_model(torch.tensor(x, dtype=torch.float32))
            >>>
            >>> symbolic_model.fit(f, training_data, niterations=1000)

            >>> # With variable transformations
            >>> transforms = [lambda x: x[:, 0] - x[:, 1], lambda x: x[:, 2]**2]
            >>> names = ["x0_minus_x1", "x2_squared"]
            >>> symbolic_model.fit(f, training_data,
            ...                   variable_transforms=transforms,
            ...                   variable_names=names)
        """
        # Convert inputs to numpy if needed
        if hasattr(inputs, 'detach'):  # torch tensor
            inputs_np = inputs.detach().cpu().numpy()
        else:
            inputs_np = np.array(inputs)

        # Get outputs from the black-box function
        if outputs is None:
            outputs_raw = f(inputs)
            if hasattr(outputs_raw, 'detach'):  # torch tensor
                outputs_np = outputs_raw.detach().cpu().numpy()
            else:
                outputs_np = np.array(outputs_raw)
        else:
            if hasattr(outputs, 'detach'):
                outputs_np = outputs.detach().cpu().numpy()
            else:
                outputs_np = np.array(outputs)

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

            print(f"Applied {len(variable_transforms)} variable transformations")
            if variable_names:
                print(f"Variable names: {variable_names}")

        # Handle both 1D and 2D outputs
        if outputs_np.ndim == 1:
            outputs_np = outputs_np.reshape(-1, 1)

        output_dims = outputs_np.shape[1]  # Number of output dimensions
        timestamp = int(time.time())

        # Merge SR parameters
        final_sr_params_base = {**self.sr_params, **kwargs}

        # Fit symbolic regression
        self.pysr_regressor = []
        self.equations_ = []

        if output_dim is None:
            # Run on all output dimensions
            for dim in range(output_dims):
                print(f"Running SR on output dimension {dim} of {output_dims-1}")

                run_id = f"dim{dim}_{timestamp}"
                final_sr_params = self._create_sr_params(save_path, run_id, final_sr_params_base)
                regressor = PySRRegressor(**final_sr_params)

                # Fit with optional variable names
                if variable_names:
                    regressor.fit(inputs_np, outputs_np[:, dim], variable_names=variable_names)
                else:
                    regressor.fit(inputs_np, outputs_np[:, dim])

                self.pysr_regressor.append(regressor)
                self.equations_.append(regressor.equations_)

                print(f"Best equation for output {dim}: {regressor.get_best()['equation']}")

        else:
            # Run on specific output dimension
            if output_dim >= output_dims:
                raise ValueError(f"output_dim {output_dim} is out of range for outputs with {output_dims} dimensions")

            print(f"Running SR on output dimension {output_dim}")

            run_id = f"dim{output_dim}_{timestamp}"
            final_sr_params = self._create_sr_params(save_path, run_id, final_sr_params_base)
            regressor = PySRRegressor(**final_sr_params)

            # Fit with optional variable names
            if variable_names:
                regressor.fit(inputs_np, outputs_np[:, output_dim], variable_names=variable_names)
            else:
                regressor.fit(inputs_np, outputs_np[:, output_dim])

            # Store as single-element list for consistency
            self.pysr_regressor = [regressor]
            self.equations_ = [regressor.equations_]

            print(f"Best equation for output {output_dim}: {regressor.get_best()['equation']}")

        print("Symbolic regression complete.")
        return self

    def predict(self, inputs: Union[np.ndarray, torch.Tensor], complexity: Optional[int] = None) -> np.ndarray:
        """
        Predict using the discovered symbolic equations.

        Args:
            inputs (np.ndarray or torch.Tensor): Input data for prediction
            complexity (int, optional): Specific complexity value to use from the Pareto frontier.
                                       If None, uses the best equation.
                                       Note: This filters by complexity value (e.g., complexity=5),
                                       not by row index.

        Returns:
            np.ndarray: Predictions from symbolic equations. Shape: (n_samples, n_outputs)

        Raises:
            ValueError: If model has not been fitted yet or complexity not found

        Example:
            >>> symbolic_model.fit(f, training_data)
            >>> predictions = symbolic_model.predict(test_data)
            >>>
            >>> # Use equation with complexity=5
            >>> predictions = symbolic_model.predict(test_data, complexity=5)
        """
        if self.pysr_regressor is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Convert to numpy
        if hasattr(inputs, 'detach'):
            inputs_np = inputs.detach().cpu().numpy()
        else:
            inputs_np = np.array(inputs)

        # Predict with each regressor
        predictions = []
        for idx, regressor in enumerate(self.pysr_regressor):
            if complexity is not None:
                # Find the row index for the given complexity value
                matching = self.equations_[idx][self.equations_[idx]["complexity"] == complexity]
                if matching.empty:
                    avail = sorted(self.equations_[idx]["complexity"].unique())
                    raise ValueError(f"No equation with complexity {complexity} for output {idx}. Available: {avail}")

                # Get the DataFrame index (row number) for this complexity
                row_idx = matching.index[0]
                pred = regressor.predict(inputs_np, index=row_idx)
            else:
                pred = regressor.predict(inputs_np)
            predictions.append(pred)

        # Stack predictions
        if len(predictions) == 1:
            return predictions[0]
        else:
            return np.column_stack(predictions)

    def get_equation(self, output_idx: int = 0, complexity: Optional[int] = None) -> str:
        """
        Get the symbolic equation for a specific output dimension.

        Args:
            output_idx (int, optional): Index of the output dimension (default: 0)
            complexity (int, optional): Specific complexity value from the Pareto frontier.
                                       If None, returns the best equation.
                                       Note: This filters by complexity value (e.g., complexity=5),
                                       not by row index.

        Returns:
            str: The symbolic equation as a string

        Raises:
            ValueError: If model has not been fitted yet, output_idx is out of range,
                       or complexity not found

        Example:
            >>> symbolic_model.fit(f, training_data)
            >>> equation = symbolic_model.get_equation()
            >>> print(f"Discovered equation: {equation}")
            >>>
            >>> # For multi-output models
            >>> eq0 = symbolic_model.get_equation(output_idx=0)
            >>> eq1 = symbolic_model.get_equation(output_idx=1)
            >>>
            >>> # Get equation with complexity=3
            >>> simple_eq = symbolic_model.get_equation(complexity=3)
        """
        if self.pysr_regressor is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if output_idx >= len(self.pysr_regressor):
            raise ValueError(f"output_idx {output_idx} is out of range. Model has {len(self.pysr_regressor)} outputs.")

        if complexity is not None:
            # Filter by actual complexity value, not index
            matching = self.equations_[output_idx][self.equations_[output_idx]["complexity"] == complexity]
            if matching.empty:
                avail = sorted(self.equations_[output_idx]["complexity"].unique())
                raise ValueError(f"No equation with complexity {complexity}. Available: {avail}")
            return matching["equation"].values[0]
        else:
            return self.pysr_regressor[output_idx].get_best()['equation']
