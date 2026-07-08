"""
SymTorch SymbolicModel Module

This module provides a wrapper for components of (or whole) ML models that adds symbolic regression
capabilities using PySR (Python Symbolic Regression).
"""

# Warnings configuration
import warnings

warnings.filterwarnings("ignore", message="torch was imported before juliacall")  # noqa: E402

# Standard library
import logging  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Dict, List, Literal, Optional, Union  # noqa: E402

# Third-party libraries
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from . import caching, distillation, equations, pruning, regression, serialization, slime  # noqa: E402

# Logger initialization
logger = logging.getLogger(__name__)


# TODO: break up this class using composition?
# TODO: integrate dim reduction workflow (e.g., pca, proj. layer training, etc...)
class SymbolicModel(nn.Module):
    # Default PySR parameters
    DEFAULT_SR_PARAMS = regression.DEFAULT_SR_PARAMS

    # Default SLIME parameters
    DEFAULT_SLIME_PARAMS = slime.DEFAULT_SLIME_PARAMS

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
            logger.info(f"No name specified for this block. Label is {self.block_name}.")

        self.pysr_regressor = {}
        self.SLIME_pysr_regressor = {}

        # I/O caching for distill
        self.distill_data = None  # Cache for standard distill
        self.distill_data_slime = None  # Cache for SLIME distill

    def _create_sr_params(
        self, save_path: str, run_id: str, custom_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create SR parameters by merging defaults with custom parameters.

        Args:
            save_path (str): Output directory path for SR results
            run_id (str): Unique run identifier
            custom_params (Dict[str, Any], optional): Custom parameters to override defaults

        Returns:
            Dict[str, Any]: Final SR parameters for PySRRegressor
        """
        return regression.create_sr_params(self.block_name, save_path, run_id, custom_params)

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
        return equations.extract_variables_for_equation(
            x,
            var_indices,
            getattr(self, "_variable_transforms", None),
            dim,
        )

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
        return equations.map_variables_to_indices(
            vars_sorted,
            getattr(self, "_variable_names", None),
            getattr(self, "_variable_transforms", None),
            dim,
        )

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
        cache = self.distill_data_slime if SLIME else self.distill_data
        return caching.check_cache_hit(cache, inputs, parent_model, SLIME, slime_params)

    def _apply_slime_sampling(self, inputs_np, function_to_call, slime_params, sr_params, fit_params):
        return slime.apply_slime_sampling(inputs_np, function_to_call, slime_params, sr_params, fit_params)

    def distill(
        self,
        inputs,
        output_dim: int = None,
        parent_model=None,
        variable_transforms: Optional[List[Callable]] = None,
        save_path: str = None,
        sr_params: Optional[Dict[str, Any]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
        SLIME: bool = False,
        slime_params: Optional[Dict[str, Any]] = None,
    ):
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

        if not isinstance(self.symtorch_block, nn.Module) and parent_model is not None:
            raise ValueError(
                "Cannot use parent_model with Callable functions. "
                "Hooks are only supported for nn.Module objects. "
                "Please call distill() without parent_model argument and pass inputs directly to the function."
            )

        sr_params = dict(sr_params) if sr_params else {}
        fit_params = dict(fit_params) if fit_params else {}
        variable_names = fit_params.get("variable_names", None)

        # --- Stage 1+2: I/O resolution, with cache short-circuit ---
        cache = self.distill_data_slime if SLIME else self.distill_data
        cache_hit, sr_inputs, sr_outputs = caching.check_cache_hit(cache, inputs, parent_model, SLIME, slime_params)
        if cache_hit:
            logger.info("🔄 Cache hit! Reusing I/O data from previous distill call.")
            if SLIME:
                # Re-apply the cached SLIME weighting so refits stay weighted
                if cache.get("slime_loss"):
                    sr_params.setdefault("elementwise_loss", cache["slime_loss"])
                if cache.get("slime_weights") is not None:
                    fit_params.setdefault("weights", cache["slime_weights"])
        else:
            raw_inputs, raw_outputs, eval_fn = distillation.resolve_io(self.symtorch_block, inputs, parent_model)

            # --- Stage 3: variable transforms ---
            if variable_transforms is not None:
                sr_inputs = distillation.apply_variable_transforms(raw_inputs, variable_transforms, variable_names)
                self._variable_transforms = variable_transforms
            else:
                sr_inputs = caching.to_numpy(raw_inputs)
                self._variable_transforms = None
            self._variable_names = variable_names
            sr_outputs = caching.to_numpy(raw_outputs)

            # --- Stage 4: SLIME sampling ---
            if SLIME:
                sr_inputs, slime_outputs, sr_params, fit_params = self._apply_slime_sampling(
                    sr_inputs, eval_fn, slime_params, sr_params, fit_params
                )
                sr_outputs = caching.to_numpy(slime_outputs)

            if sr_outputs.ndim == 1:
                sr_outputs = sr_outputs.reshape(-1, 1)

            # --- Stage 5a: pruning mask (mask before caching, as before) ---
            if getattr(self, "pruning_mask", None) is not None:
                sr_outputs = sr_outputs[:, caching.to_numpy(self.pruning_mask).astype(bool)]

            entry = caching.build_cache_entry(
                inputs,
                sr_inputs,
                sr_outputs,
                parent_model,
                slime_params=(slime_params or {}) if SLIME else None,
                slime_weights=fit_params.get("weights") if SLIME else None,
                slime_loss=sr_params.get("elementwise_loss") if SLIME else None,
            )
            if SLIME:
                self.distill_data_slime = entry
            else:
                self.distill_data = entry

        # --- Stage 5b: dimension selection ---
        if getattr(self, "pruning_mask", None) is not None:
            active_dims = self.get_active_dimensions()
            if not active_dims:
                logger.warning("❗No active dimensions to distill!")
                return {}
            self.output_dims = self.initial_dim
            if output_dim is not None:
                if output_dim not in active_dims:
                    logger.warning(
                        f"❗Requested output dimension {output_dim} is not active. Active dimensions: {active_dims}"
                    )
                    return {}
                dims = [output_dim]
            else:
                dims = active_dims
            columns = [active_dims.index(d) for d in dims]
        else:
            n_out = sr_outputs.shape[1]
            self.output_dims = n_out
            if output_dim is not None:
                if output_dim >= n_out:
                    raise ValueError(f"output_dim {output_dim} is out of range for outputs with {n_out} dimensions")
                dims = [output_dim]
                columns = [output_dim]
            else:
                dims = list(range(n_out))
                columns = dims

        # --- Stage 6: fit ---
        timestamp = int(time.time())
        if len(dims) == 1:
            dim = dims[0]
            logger.info(f"🛠️ Running SR on output dimension {dim}.")
            regressor = regression.fit_single_dimension(
                sr_inputs,
                sr_outputs[:, columns[0]],
                self.block_name,
                save_path,
                dim,
                sr_params,
                fit_params,
                timestamp,
            )
            pysr_regressors = {dim: regressor}
        else:
            logger.info(f"🛠️ Running multi-output SR on {len(dims)} output dimensions")
            pysr_regressors = regression.fit_all_dimensions(
                sr_inputs,
                sr_outputs[:, columns],
                dims,
                self.block_name,
                save_path,
                sr_params,
                fit_params,
                timestamp,
            )

        for dim in dims:
            logger.info(f"💡Best equation for output {dim} found to be {pysr_regressors[dim].get_best()['equation']}.")
        logger.info(f"❤️ SR on {self.block_name} complete.")

        # --- Stage 7: store ---
        if SLIME:
            self.SLIME_pysr_regressor = self.SLIME_pysr_regressor | pysr_regressors
        else:
            self.pysr_regressor = self.pysr_regressor | pysr_regressors

        if output_dim is not None:
            return pysr_regressors.get(output_dim)
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

        if (
            not hasattr(self, regressor_dict.__class__.__name__.replace("dict", "pysr_regressor"))
            or regressor_dict is None
        ):
            logger.error(
                f"❗No {mode_name} equations found for this block yet. You need to first run .distill with SLIME={SLIME}."
            )
            return None
        if dim not in regressor_dict:
            logger.error(
                f"❗No {mode_name} equation found for output dimension {dim}. You need to first run .distill with SLIME={SLIME}."
            )
            return None

        regressor = regressor_dict[dim]

        expr = equations.select_expression(regressor, complexity)
        if expr is None:
            return None

        try:
            f, vars_sorted = equations.expression_to_callable(expr)
            return f, vars_sorted
        except RuntimeError as e:
            logger.warning(f"⚠️ Warning: Could not create lambdify function for dimension {dim}: {e}")
            return None

    def switch_to_symbolic(self, complexity: list = None, SLIME: bool = False, compile: bool = False):
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
            compile (bool, optional): If True, wrap the symbolic forward pass with
                torch.compile() (PyTorch 2.0+). Default: False.

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
            raise RuntimeError(
                f"No {mode_name} equations found for this block yet. You need to first run .distill with SLIME={SLIME}."
            )

        if not hasattr(self, "output_dims"):
            raise RuntimeError("No output dimension information found. You need to first run .distill.")

        # Check if pruning is enabled
        if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
            # Pruning mode - only need equations for active dimensions
            active_dims = self.get_active_dimensions()
            if not active_dims:
                raise RuntimeError("No active dimensions to switch to equations.")

            # Check that we have equations for all active dimensions
            missing_dims = []
            for dim in active_dims:
                if dim not in regressor_dict:
                    missing_dims.append(dim)

            if missing_dims:
                raise RuntimeError(
                    f"Missing {mode_name} equations for active dimensions {missing_dims}. You need to run .distill with SLIME={SLIME} on all active dimensions first."
                )

            dimensions_to_process = active_dims
        else:
            # Standard mode - need equations for all dimensions
            missing_dims = []
            for dim in range(self.output_dims):
                if dim not in regressor_dict:
                    missing_dims.append(dim)

            if missing_dims:
                logger.error(f"Available dimensions: {list(regressor_dict.keys())}")
                logger.error(f"Required dimensions: {list(range(self.output_dims))}")
                raise RuntimeError(
                    f"Missing {mode_name} equations for dimensions {missing_dims}. You need to run .distill with SLIME={SLIME} on all output dimensions first."
                )

            dimensions_to_process = list(range(self.output_dims))

        # Store original block for potential restoration
        if not hasattr(self, "_original_block"):
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
                        logger.warning(
                            f"⚠️ Warning: Not enough complexity values provided. Using default for dimension {dim}"
                        )
                else:
                    # If complexity is a single value, use it for all dimensions
                    dim_complexity = complexity

            result = self._get_equation(dim, dim_complexity, SLIME=SLIME)
            if result is None:
                if dim_complexity is not None:
                    raise ValueError(f"No equation with complexity {dim_complexity} for dimension {dim}.")
                raise RuntimeError(f"Failed to get equation for dimension {dim}")

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
        if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
            logger.info(
                f"✅ Successfully switched {self.block_name} to {mode_label}symbolic equations for {len(dimensions_to_process)} active dimensions:"
            )
        else:
            logger.info(
                f"✅ Successfully switched {self.block_name} to {mode_label}symbolic equations for all {len(dimensions_to_process)} dimensions:"
            )

        for dim in dimensions_to_process:
            logger.info(f"   Dimension {dim}: {equation_strs[dim]}")

            # Display variable names properly
            var_names_display = []
            if hasattr(self, "_variable_names") and self._variable_names is not None:
                # Use custom variable names
                for idx in equation_vars[dim]:
                    if idx < len(self._variable_names):
                        var_names_display.append(self._variable_names[idx])
                    else:
                        var_names_display.append(f"transform_{idx}")
            else:
                # Use default x0, x1, etc. format
                var_names_display = [f"x{i}" for i in equation_vars[dim]]

            logger.info(f"   Variables: {var_names_display}")

        if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
            logger.info(f"🎯 Active dimensions {dimensions_to_process} now using {mode_label}symbolic equations.")
            logger.info("🔒 Inactive dimensions will output zeros.")
        else:
            logger.info(
                f"🎯 All {len(dimensions_to_process)} output dimensions now using {mode_label}symbolic equations."
            )

        # Apply torch.compile() optimization if requested (PyTorch 2.0+)
        if compile and hasattr(torch, "compile"):
            logger.info("🚀 Compiling forward pass with torch.compile() for GPU optimization...")
            try:
                # Compile with fullgraph=False to allow dynamic control flow
                # mode="reduce-overhead" optimizes for repeated calls
                self._original_forward = self.forward
                self.forward = torch.compile(self.forward, mode="reduce-overhead", fullgraph=False)
                logger.info("✅ Forward pass compiled successfully")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile() failed: {e}. Continuing without compilation.")
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

        if not hasattr(self, "output_dims"):
            raise ValueError("No output dimension information found. Run .distill() first.")

        # If only one output dimension, default to dim=0
        if self.output_dims == 1:
            dim = 0
        elif dim >= self.output_dims:
            raise ValueError(
                f"Dimension {dim} out of range. Model has {self.output_dims} output dimensions (0-{self.output_dims - 1})"
            )

        if dim not in regressor_dict:
            raise ValueError(
                f"No {mode_name} equation found for dimension {dim}. Available dimensions: {list(regressor_dict.keys())}"
            )

        regressor = regressor_dict[dim]

        # Get the equation at specified complexity or best equation
        expr = equations.select_expression(regressor, complexity)
        if expr is None:
            available_complexities = sorted(regressor.equations_["complexity"].unique())
            raise ValueError(
                f"No equation with complexity {complexity} for dimension {dim}. Available complexities: {available_complexities}"
            )

        f, vars_sorted = equations.expression_to_callable(expr)

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

    def show_symbolic_expression(self, dim=None, complexity=None, SLIME: bool = False):
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
            print(
                f"❗No {mode_name} equations found for this block yet. You need to first run .distill with SLIME={SLIME}."
            )
            return

        if not hasattr(self, "output_dims"):
            print("❗No output dimension information found. You need to first run .distill.")
            return

        # Convert single values to lists
        if isinstance(dim, int):
            dims_to_show = [dim]
        elif dim is None:
            # For pruned models, show only active dimensions by default
            if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
                dims_to_show = self.get_active_dimensions()
                if dims_to_show:
                    print(
                        f"ℹ️ Showing {mode_name} expressions for {len(dims_to_show)} active dimensions (out of {self.output_dims} total)"
                    )
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
                print(
                    f"❗Complexity list length ({len(complexities)}) must match dimension list length ({len(dims_to_show)})"
                )
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
        if hasattr(self, "_original_block"):
            self.symtorch_block = self._original_block

        logger.info(f"✅ Switched {self.block_name} back to block")

    def setup_pruning(
        self,
        initial_dim: int,
        target_dim: int,
        total_steps: int,
        end_step_frac: float = 0.5,
        decay_rate: Literal["cosine", "exp", "linear"] = "exp",
    ):
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
        self.register_buffer("pruning_mask", torch.ones(self.current_dim, dtype=torch.bool))

        logger.info(f"✅ Pruning successfully set up for block {self.block_name}.")
        logger.info(f"   Initial dimensions: {initial_dim}")
        logger.info(f"   Target dimensions: {target_dim}")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Pruning will complete at step {int(end_step_frac * total_steps)}")

        return None

    def _set_pruning_schedule(self, total_steps: int, decay_rate: str = "cosine", end_step_frac: float = 0.5):
        """
        Create step-based pruning schedule.

        Args:
            total_steps (int): Total number of training steps
            decay_rate (str): Type of decay schedule ('exp', 'linear', 'cosine')
            end_step_frac (float): Fraction of steps to complete pruning by

        Returns:
            dict: Mapping from step number to target dimensions
        """
        return pruning.make_pruning_schedule(self.initial_dim, self.target_dim, total_steps, decay_rate, end_step_frac)

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

        if not hasattr(self, "pruning_schedule") or self.pruning_schedule is None:
            raise RuntimeError("Pruning schedule is not set. Call setup_pruning() first.")

        if step not in self.pruning_schedule:
            return

        target_dims = self.pruning_schedule[step]

        with torch.no_grad():
            # Extract outputs at this layer level for importance evaluation
            if parent_model is not None:
                with distillation.capture_layer_io(self.symtorch_block, parent_model, sample_data) as (
                    _,
                    layer_outputs,
                ):
                    pass

                # Use captured intermediate data
                if layer_outputs:
                    output_array = layer_outputs[0]
                else:
                    raise RuntimeError(
                        "Failed to capture intermediate activations. Ensure parent_model contains this SymbolicModel instance."
                    )
            else:
                # Original behavior - use block directly
                self.symtorch_block.eval()
                output_array = self.symtorch_block(sample_data)

            most_important = pruning.rank_dimensions(output_array, target_dims)

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
        if not hasattr(self, "pruning_mask") or self.pruning_mask is None:
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
        if hasattr(self, "_using_equation") and self._using_equation:
            # Track input type to return matching output type
            is_torch_input = isinstance(x, torch.Tensor)

            # Convert to torch tensor if needed for equation evaluation
            if not is_torch_input:
                x_torch = torch.tensor(x, dtype=torch.float32)
            else:
                x_torch = x

            batch_size = x_torch.shape[0]

            # Check if pruning is enabled
            if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
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
                if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
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
                    if hasattr(output, "detach"):  # Already a torch tensor
                        output = output.to(x.device)
                    else:
                        output = torch.tensor(output, dtype=x.dtype, device=x.device)

                    # Apply pruning mask if enabled
                    if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
                        output = output * self.pruning_mask
                    return output
                else:
                    # Input is already numpy, call directly and return numpy
                    output = self.symtorch_block(x)
                    # Apply pruning mask if enabled (convert to numpy if needed)
                    if hasattr(self, "pruning_mask") and self.pruning_mask is not None:
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
        logger.info(f"✅ Cache cleared for {self.block_name}.")

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
        serialization.save_symtorch_state(self, destination, prefix)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
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
        serialization.load_symtorch_extras(self, state_dict, prefix, error_msgs)

        # Call parent to load parameters and buffers (including _original_block if present)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
        if not hasattr(self, "_equation_vars") or not self._equation_vars:
            raise RuntimeError("Cannot rebuild equations: _equation_vars not found")

        self._equation_funcs = {}

        for dim, var_indices in self._equation_vars.items():
            # Get equation from regressor
            result = self._get_equation(dim, complexity=None, SLIME=False)
            if result is None:
                raise RuntimeError(f"Cannot rebuild equation for dimension {dim}")

            equation_func, vars_sorted = result
            self._equation_funcs[dim] = equation_func
