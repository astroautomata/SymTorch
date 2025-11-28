"""
Unit tests for SymbolicModel class using pytest.

This module provides comprehensive test coverage for the unified SymbolicModel class,
including all operational modes: layer-level, model-agnostic, SLIME, and pruning.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import shutil
import os
import warnings
from unittest.mock import Mock, MagicMock, patch
import sys
import pandas as pd
import sympy

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symtorch import SymbolicModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_layer():
    """Simple linear layer for testing."""
    return nn.Linear(5, 3)


@pytest.fixture
def symbolic_model(simple_layer):
    """Basic SymbolicModel with nn.Module."""
    return SymbolicModel(simple_layer, block_name="test_model")


@pytest.fixture
def simple_callable():
    """Simple callable function for testing."""
    def func(x):
        return x[:, 0]**2 + x[:, 1]
    return func


@pytest.fixture
def fast_sr_params():
    """Minimal SR parameters for fast testing."""
    return {
        "niterations": 10,
        "populations": 2,
        "population_size": 10
    }


@pytest.fixture
def sample_inputs():
    """Sample input data."""
    return torch.randn(50, 5)


@pytest.fixture
def sample_inputs_np():
    """Sample input data as numpy."""
    return np.random.randn(50, 5)


@pytest.fixture
def mock_pysr_regressor():
    """Mock PySR regressor for testing."""
    mock_reg = MagicMock()
    mock_reg.get_best.return_value = {"equation": "x0 + x1", "loss": 0.01}
    mock_reg.equations_ = pd.DataFrame({
        "equation": ["x0 + x1"],
        "sympy_format": [sympy.sympify("x0 + x1")],
        "complexity": [3],
        "loss": [0.01]
    })
    return mock_reg


# ============================================================================
# Test Initialization
# ============================================================================

class TestInitialization:
    """Tests for SymbolicModel initialization."""

    def test_init_with_nn_module(self, simple_layer):
        """Test initialization with a PyTorch nn.Module."""
        model = SymbolicModel(simple_layer, block_name="test_layer")

        assert model.block_name == "test_layer"
        assert isinstance(model.symtorch_block, nn.Module)
        assert model.pysr_regressor == {}
        assert model.SLIME_pysr_regressor == {}

    def test_init_with_callable(self, simple_callable):
        """Test initialization with a callable function."""
        model = SymbolicModel(simple_callable, block_name="test_func")

        assert model.block_name == "test_func"
        assert callable(model.symtorch_block)
        assert model.pysr_regressor == {}

    def test_init_auto_name_generation(self, simple_layer):
        """Test automatic block name generation when none provided."""
        model = SymbolicModel(simple_layer)

        assert model.block_name.startswith("block_")
        assert model.block_name is not None

    def test_default_params_exist(self):
        """Test that default parameters are defined."""
        assert isinstance(SymbolicModel.DEFAULT_SR_PARAMS, dict)
        assert isinstance(SymbolicModel.DEFAULT_SLIME_PARAMS, dict)

        # Check key parameters exist
        assert "binary_operators" in SymbolicModel.DEFAULT_SR_PARAMS
        assert "niterations" in SymbolicModel.DEFAULT_SR_PARAMS
        assert "x" in SymbolicModel.DEFAULT_SLIME_PARAMS
        assert "J_nn" in SymbolicModel.DEFAULT_SLIME_PARAMS


# ============================================================================
# Test Helper Methods
# ============================================================================

class TestHelperMethods:
    """Tests for SymbolicModel internal helper methods."""

    def test_create_sr_params_default(self, symbolic_model):
        """Test _create_sr_params with default settings."""
        params = symbolic_model._create_sr_params(
            save_path=None,
            run_id="test_run",
            custom_params=None
        )

        assert "binary_operators" in params
        assert "output_directory" in params
        assert "run_id" in params
        assert params["run_id"] == "test_run"
        assert "test_model" in params["output_directory"]

    def test_create_sr_params_custom(self, symbolic_model):
        """Test _create_sr_params with custom parameters."""
        custom = {"niterations": 100, "parsimony": 0.01}
        params = symbolic_model._create_sr_params(
            save_path="/tmp/test",
            run_id="custom_run",
            custom_params=custom
        )

        assert params["niterations"] == 100
        assert params["parsimony"] == 0.01
        assert "/tmp/test" in params["output_directory"]

    def test_map_variables_to_indices_standard(self, symbolic_model):
        """Test _map_variables_to_indices with standard x0, x1 format."""
        vars_list = [sympy.Symbol('x0'), sympy.Symbol('x2'), sympy.Symbol('x1')]

        indices = symbolic_model._map_variables_to_indices(vars_list, dim=0)

        assert indices == [0, 2, 1]

    def test_map_variables_to_indices_custom_names(self, symbolic_model):
        """Test _map_variables_to_indices with custom variable names."""
        symbolic_model._variable_names = ['alpha', 'beta', 'gamma']
        vars_list = [sympy.Symbol('beta'), sympy.Symbol('gamma')]

        indices = symbolic_model._map_variables_to_indices(vars_list, dim=0)

        assert indices == [1, 2]

    def test_map_variables_to_indices_error(self, symbolic_model):
        """Test _map_variables_to_indices with unmappable variable."""
        vars_list = [sympy.Symbol('unknown_var')]

        with pytest.raises(ValueError, match="Could not map variable"):
            symbolic_model._map_variables_to_indices(vars_list, dim=0)

    def test_extract_variables_standard(self, symbolic_model):
        """Test _extract_variables_for_equation with standard column extraction."""
        x = torch.randn(10, 5)
        var_indices = [0, 2, 4]

        selected = symbolic_model._extract_variables_for_equation(x, var_indices, dim=0)

        assert len(selected) == 3
        assert torch.allclose(selected[0], x[:, 0])
        assert torch.allclose(selected[1], x[:, 2])
        assert torch.allclose(selected[2], x[:, 4])

    def test_extract_variables_with_transforms(self, symbolic_model):
        """Test _extract_variables_for_equation with transformations."""
        x = torch.randn(10, 5)

        # Define simple transformations
        symbolic_model._variable_transforms = [
            lambda inp: inp[:, 0],
            lambda inp: inp[:, 1] ** 2,
            lambda inp: torch.sin(inp[:, 2])
        ]

        var_indices = [0, 2]
        selected = symbolic_model._extract_variables_for_equation(x, var_indices, dim=0)

        assert len(selected) == 2
        assert torch.allclose(selected[0], x[:, 0])
        assert torch.allclose(selected[1], torch.sin(x[:, 2]))

    def test_extract_variables_out_of_range_error(self, symbolic_model):
        """Test _extract_variables_for_equation with out-of-range index."""
        x = torch.randn(10, 5)
        var_indices = [0, 10]  # Index 10 is out of range

        with pytest.raises(ValueError, match="requires variable"):
            symbolic_model._extract_variables_for_equation(x, var_indices, dim=0)


# ============================================================================
# Test Distill with nn.Module
# ============================================================================

class TestDistillWithModule:
    """Tests for distill() method with nn.Module blocks."""

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_basic_module(self, mock_pysr_class, symbolic_model, sample_inputs, fast_sr_params, mock_pysr_regressor):
        """Test basic distill() call with nn.Module."""
        mock_pysr_class.return_value = mock_pysr_regressor

        result = symbolic_model.distill(sample_inputs, sr_params=fast_sr_params)

        # Check that regressors were created for all output dimensions
        assert len(symbolic_model.pysr_regressor) == 3
        assert 0 in symbolic_model.pysr_regressor
        assert 1 in symbolic_model.pysr_regressor
        assert 2 in symbolic_model.pysr_regressor

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_specific_dimension(self, mock_pysr_class, symbolic_model, sample_inputs, fast_sr_params):
        """Test distill() with specific output dimension."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0**2", "loss": 0.001}
        mock_pysr_class.return_value = mock_reg

        result = symbolic_model.distill(sample_inputs, output_dim=1, sr_params=fast_sr_params)

        # Check that only dimension 1 was fitted
        assert 1 in symbolic_model.pysr_regressor
        assert len(symbolic_model.pysr_regressor) == 1

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_with_parent_model(self, mock_pysr_class, sample_inputs, fast_sr_params, mock_pysr_regressor):
        """Test distill() with parent_model for layer-level analysis."""
        mock_pysr_class.return_value = mock_pysr_regressor

        # Create a simple parent model
        layer = nn.Linear(5, 3)
        symbolic_layer = SymbolicModel(layer, block_name="symbolic_layer")

        class ParentModel(nn.Module):
            def __init__(self, sym_layer):
                super().__init__()
                self.layer1 = nn.Linear(10, 5)
                self.symbolic_layer = sym_layer
                self.layer2 = nn.Linear(3, 2)

            def forward(self, x):
                x = self.layer1(x)
                x = self.symbolic_layer(x)
                x = self.layer2(x)
                return x

        parent = ParentModel(symbolic_layer)
        inputs = torch.randn(50, 10)

        result = symbolic_layer.distill(inputs, parent_model=parent, sr_params=fast_sr_params)

        # Verify regressors were created
        assert len(symbolic_layer.pysr_regressor) > 0

    def test_distill_callable_with_parent_error(self, simple_callable, sample_inputs):
        """Test that using parent_model with Callable raises error."""
        model = SymbolicModel(simple_callable, block_name="func")
        parent = nn.Linear(5, 5)

        with pytest.raises(ValueError, match="Cannot use parent_model with Callable"):
            model.distill(sample_inputs, parent_model=parent)


# ============================================================================
# Test Distill with Callable
# ============================================================================

class TestDistillWithCallable:
    """Tests for distill() method with Callable functions."""

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_numpy_function(self, mock_pysr_class, sample_inputs_np, fast_sr_params):
        """Test distill() with numpy function."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0**2 + x1", "loss": 0.001}
        mock_pysr_class.return_value = mock_reg

        def numpy_func(x):
            return x[:, 0]**2 + x[:, 1]

        model = SymbolicModel(numpy_func, block_name="numpy_test")

        result = model.distill(sample_inputs_np, sr_params=fast_sr_params)

        # Check that regressor was created
        assert 0 in model.pysr_regressor

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_torch_function(self, mock_pysr_class, sample_inputs_np, fast_sr_params):
        """Test distill() with torch function returning tensor."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "sin(x0)", "loss": 0.001}
        mock_pysr_class.return_value = mock_reg

        def torch_func(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            return torch.sin(x_tensor[:, 0])

        model = SymbolicModel(torch_func, block_name="torch_test")

        result = model.distill(sample_inputs_np, sr_params=fast_sr_params)

        assert 0 in model.pysr_regressor

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_multi_output_callable(self, mock_pysr_class, sample_inputs_np, fast_sr_params, mock_pysr_regressor):
        """Test distill() with callable returning multiple outputs."""
        mock_pysr_class.return_value = mock_pysr_regressor

        def multi_output(x):
            return np.column_stack([x[:, 0]**2, x[:, 1] + x[:, 2]])

        model = SymbolicModel(multi_output, block_name="multi_test")

        result = model.distill(sample_inputs_np, sr_params=fast_sr_params)

        # Should have created regressors for both outputs
        assert len(model.pysr_regressor) == 2
        assert 0 in model.pysr_regressor
        assert 1 in model.pysr_regressor


# ============================================================================
# Test SLIME Functionality
# ============================================================================

class TestSLIME:
    """Tests for SLIME functionality."""

    def test_apply_slime_sampling_with_point(self, simple_callable):
        """Test _apply_slime_sampling with point of interest."""
        model = SymbolicModel(simple_callable, block_name="slime_test")

        inputs = np.random.randn(100, 5)
        x0 = inputs[0]  # Point of interest

        slime_params = {
            'x': x0,
            'J_nn': 10,
            'num_synthetic': 50,
            'real_weighting': 1.0
        }

        sr_params = {}
        fit_params = {}

        result_inputs, result_outputs, updated_sr, updated_fit = model._apply_slime_sampling(
            inputs, simple_callable, slime_params, sr_params, fit_params
        )

        # Check that we got real + synthetic samples
        assert len(result_inputs) == 60  # 10 real + 50 synthetic
        assert 'weights' in updated_fit
        assert len(updated_fit['weights']) == 60
        assert 'elementwise_loss' in updated_sr

    def test_apply_slime_sampling_global(self, simple_callable):
        """Test _apply_slime_sampling without point of interest (global)."""
        model = SymbolicModel(simple_callable, block_name="slime_global")

        inputs = np.random.randn(100, 5)

        slime_params = {
            'x': None,
            'J_nn': 0,
            'num_synthetic': 0
        }

        sr_params = {}
        fit_params = {}

        result_inputs, result_outputs, updated_sr, updated_fit = model._apply_slime_sampling(
            inputs, simple_callable, slime_params, sr_params, fit_params
        )

        # Should return original inputs unchanged
        assert len(result_inputs) == 100
        np.testing.assert_array_equal(result_inputs, inputs)

    def test_apply_slime_validation_errors(self, simple_callable):
        """Test _apply_slime_sampling validation errors."""
        model = SymbolicModel(simple_callable, block_name="slime_error")
        inputs = np.random.randn(100, 5)

        # Test: x specified but num_synthetic is 0
        slime_params = {
            'x': inputs[0],
            'J_nn': 10,
            'num_synthetic': 0
        }

        with pytest.raises(ValueError, match="num_synthetic must be > 0"):
            model._apply_slime_sampling(inputs, simple_callable, slime_params, {}, {})

        # Test: J_nn >= len(inputs)
        slime_params = {
            'x': inputs[0],
            'J_nn': 100,  # >= len(inputs)
            'num_synthetic': 10
        }

        with pytest.raises(ValueError, match="J_nn"):
            model._apply_slime_sampling(inputs, simple_callable, slime_params, {}, {})

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_with_slime_mode(self, mock_pysr_class, sample_inputs_np, fast_sr_params):
        """Test distill() with SLIME=True."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0**2", "loss": 0.001}
        mock_pysr_class.return_value = mock_reg

        def test_func(x):
            return x[:, 0]**2

        model = SymbolicModel(test_func, block_name="slime_distill")

        slime_params = {
            'x': sample_inputs_np[0],
            'J_nn': 10,
            'num_synthetic': 50
        }

        result = model.distill(
            sample_inputs_np,
            SLIME=True,
            slime_params=slime_params,
            sr_params=fast_sr_params
        )

        # Check that SLIME regressor was created
        assert 0 in model.SLIME_pysr_regressor
        # Standard regressor should still be empty
        assert len(model.pysr_regressor) == 0


# ============================================================================
# Test Equation Switching
# ============================================================================

class TestEquationSwitching:
    """Tests for switching between block and symbolic modes."""

    def test_switch_to_symbolic_without_distill(self, symbolic_model):
        """Test switch_to_symbolic() before running distill()."""
        # Should print error and return early
        symbolic_model.switch_to_symbolic()

        # Model should not be in equation mode
        assert not hasattr(symbolic_model, '_using_equation')

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_switch_to_symbolic_success(self, mock_pysr_class, symbolic_model, sample_inputs):
        """Test successful switch_to_symbolic()."""
        # Mock PySR to return valid equation
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0", "loss": 0.01}
        mock_reg.equations_ = pd.DataFrame({
            "equation": ["x0"],
            "sympy_format": [sympy.sympify("x0")],
            "complexity": [1],
            "loss": [0.01]
        })

        mock_pysr_class.return_value = mock_reg

        symbolic_model.distill(sample_inputs)

        # Now switch to symbolic
        symbolic_model.switch_to_symbolic()

        assert symbolic_model._using_equation
        assert symbolic_model._equation_funcs is not None
        assert len(symbolic_model._equation_funcs) == 3

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_switch_to_block(self, mock_pysr_class, symbolic_model, sample_inputs):
        """Test switch_to_block() to restore original block."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0", "loss": 0.01}
        mock_reg.equations_ = pd.DataFrame({
            "equation": ["x0"],
            "sympy_format": [sympy.sympify("x0")],
            "complexity": [1],
            "loss": [0.01]
        })

        mock_pysr_class.return_value = mock_reg

        symbolic_model.distill(sample_inputs)
        symbolic_model.switch_to_symbolic()

        # Switch back to block
        symbolic_model.switch_to_block()

        assert not symbolic_model._using_equation

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_get_symbolic_function(self, mock_pysr_class, symbolic_model, sample_inputs):
        """Test get_symbolic_function() method."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0**2", "loss": 0.001}
        mock_reg.equations_ = pd.DataFrame({
            "equation": ["x0**2"],
            "sympy_format": [sympy.sympify("x0**2")],
            "complexity": [3],
            "loss": [0.001]
        })

        mock_pysr_class.return_value = mock_reg

        symbolic_model.distill(sample_inputs, output_dim=0)

        # Get symbolic function for dimension 0
        func = symbolic_model.get_symbolic_function(dim=0)

        # Test the function
        test_input = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = func(test_input)

        # Should be approximately x0**2 = 1.0
        assert result is not None


# ============================================================================
# Test Pruning Functionality
# ============================================================================

class TestPruning:
    """Tests for pruning functionality."""

    def test_setup_pruning(self, symbolic_model):
        """Test setup_pruning() method."""
        symbolic_model.setup_pruning(
            initial_dim=10,
            target_dim=3,
            total_steps=1000,
            end_step_frac=0.5,
            decay_rate='exp'
        )

        assert symbolic_model.initial_dim == 10
        assert symbolic_model.target_dim == 3
        assert symbolic_model.current_dim == 10
        assert symbolic_model.pruning_schedule is not None
        assert symbolic_model.pruning_mask is not None
        assert symbolic_model.pruning_mask.sum().item() == 10

    def test_setup_pruning_callable_error(self, simple_callable):
        """Test that setup_pruning() raises error for Callable."""
        model = SymbolicModel(simple_callable, block_name="callable")

        with pytest.raises(ValueError, match="Pruning only works on PyTorch MLPs"):
            model.setup_pruning(10, 3, 1000)

    def test_set_pruning_schedule_exp(self):
        """Test exponential pruning schedule."""
        layer = nn.Linear(5, 10)
        model = SymbolicModel(layer, block_name="prune_exp")
        model.setup_pruning(10, 3, 1000, decay_rate='exp')

        schedule = model.pruning_schedule

        # Check schedule properties
        assert schedule[0] == 10  # Starts at initial_dim
        assert schedule[999] == 3  # Ends at target_dim
        # Dimensions should decrease monotonically
        prev_dim = 10
        for step in range(500):
            assert schedule[step] <= prev_dim
            prev_dim = schedule[step]

    def test_set_pruning_schedule_linear(self):
        """Test linear pruning schedule."""
        layer = nn.Linear(5, 10)
        model = SymbolicModel(layer, block_name="prune_linear")
        model.setup_pruning(10, 3, 1000, decay_rate='linear')

        schedule = model.pruning_schedule

        assert schedule[0] == 10
        assert schedule[999] == 3

    def test_set_pruning_schedule_cosine(self):
        """Test cosine pruning schedule."""
        layer = nn.Linear(5, 10)
        model = SymbolicModel(layer, block_name="prune_cosine")
        model.setup_pruning(10, 3, 1000, decay_rate='cosine')

        schedule = model.pruning_schedule

        assert schedule[0] == 10
        assert schedule[999] == 3

    def test_prune_method(self):
        """Test prune() method."""
        layer = nn.Linear(5, 10)
        model = SymbolicModel(layer, block_name="prune_method")
        model.setup_pruning(10, 5, 1000)

        # Create sample data
        sample_data = torch.randn(50, 5)

        # Prune at step 100
        model.prune(step=100, sample_data=sample_data)

        # Check that pruning occurred
        active_dims = model.get_active_dimensions()
        expected_dims = model.pruning_schedule[100]
        assert len(active_dims) == expected_dims

    def test_prune_without_setup_error(self, symbolic_model):
        """Test prune() raises error without setup."""
        sample_data = torch.randn(50, 5)

        with pytest.raises(RuntimeError, match="Pruning schedule is not set"):
            symbolic_model.prune(step=100, sample_data=sample_data)

    def test_get_active_dimensions(self):
        """Test get_active_dimensions() method."""
        layer = nn.Linear(5, 10)
        model = SymbolicModel(layer, block_name="active_dims")
        model.setup_pruning(10, 5, 1000)

        # Initially all dimensions should be active
        active = model.get_active_dimensions()
        assert len(active) == 10

        # After pruning, fewer should be active
        sample_data = torch.randn(50, 5)
        model.prune(step=250, sample_data=sample_data)

        active_after = model.get_active_dimensions()
        assert len(active_after) < 10

    def test_get_active_dimensions_without_setup_error(self, symbolic_model):
        """Test get_active_dimensions() raises error without setup."""
        with pytest.raises(RuntimeError, match="Pruning has not been set up"):
            symbolic_model.get_active_dimensions()


# ============================================================================
# Test Forward Method
# ============================================================================

class TestForward:
    """Tests for forward() method in different modes."""

    def test_forward_block_mode_module(self, symbolic_model):
        """Test forward() in block mode with nn.Module."""
        inputs = torch.randn(10, 5)

        output = symbolic_model(inputs)

        assert output.shape == (10, 3)
        assert isinstance(output, torch.Tensor)

    def test_forward_block_mode_callable(self, simple_callable):
        """Test forward() in block mode with Callable."""
        model = SymbolicModel(simple_callable, block_name="callable_forward")

        # Test with torch tensor
        inputs_torch = torch.randn(10, 5)
        output = model(inputs_torch)
        assert isinstance(output, torch.Tensor)

        # Test with numpy array
        inputs_np = np.random.randn(10, 5)
        output_np = model(inputs_np)
        assert isinstance(output_np, np.ndarray)

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_forward_symbolic_mode(self, mock_pysr_class, symbolic_model, sample_inputs):
        """Test forward() in symbolic equation mode."""
        # Setup mock regressor
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0", "loss": 0.001}
        mock_reg.equations_ = pd.DataFrame({
            "equation": ["x0"],
            "sympy_format": [sympy.sympify("x0")],
            "complexity": [1],
            "loss": [0.001]
        })

        mock_pysr_class.return_value = mock_reg

        symbolic_model.distill(sample_inputs)
        symbolic_model.switch_to_symbolic()

        # Test forward pass
        test_inputs = torch.randn(10, 5)
        output = symbolic_model(test_inputs)

        assert output.shape == (10, 3)

    def test_forward_with_pruning_mask(self):
        """Test forward() with pruning mask applied."""
        layer = nn.Linear(5, 3)
        model = SymbolicModel(layer, block_name="prune_forward")
        model.setup_pruning(initial_dim=3, target_dim=2, total_steps=1000)

        inputs = torch.randn(10, 5)
        sample_data = torch.randn(50, 5)

        # Prune to reduce dimensions
        model.prune(step=500, sample_data=sample_data)

        # Forward pass should still work
        output = model(inputs)

        # Some dimensions should be zero
        active_dims = model.get_active_dimensions()
        assert len(active_dims) == 2


# ============================================================================
# Test Variable Transformations
# ============================================================================

class TestVariableTransforms:
    """Tests for variable transformation functionality."""

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_with_transforms_module(self, mock_pysr_class, sample_inputs, fast_sr_params, mock_pysr_regressor):
        """Test distill() with variable transforms for nn.Module."""
        mock_pysr_class.return_value = mock_pysr_regressor

        layer = nn.Linear(5, 2)
        model = SymbolicModel(layer, block_name="transform_test")

        # Define transforms
        transforms = [
            lambda x: x[:, 0],
            lambda x: x[:, 1] ** 2,
            lambda x: torch.sin(x[:, 2])
        ]

        variable_names = ['x', 'x_squared', 'sin_x']

        result = model.distill(
            sample_inputs,
            variable_transforms=transforms,
            sr_params=fast_sr_params,
            fit_params={'variable_names': variable_names}
        )

        # Check that transforms were stored
        assert model._variable_transforms is not None
        assert len(model._variable_transforms) == 3
        assert model._variable_names == variable_names

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_with_transforms_callable(self, mock_pysr_class, sample_inputs_np, fast_sr_params):
        """Test distill() with variable transforms for Callable."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0", "loss": 0.001}
        mock_pysr_class.return_value = mock_reg

        def my_func(x):
            if isinstance(x, np.ndarray):
                return x[:, 0] ** 2
            return x[:, 0].detach().cpu().numpy() ** 2

        model = SymbolicModel(my_func, block_name="callable_transform")

        transforms = [
            lambda x: x[:, 0] if isinstance(x, torch.Tensor) else x[:, 0],
            lambda x: x[:, 1] ** 2 if isinstance(x, torch.Tensor) else x[:, 1] ** 2
        ]

        result = model.distill(
            sample_inputs_np,
            variable_transforms=transforms,
            sr_params=fast_sr_params
        )

        assert model._variable_transforms is not None

    def test_distill_transform_length_mismatch_error(self, sample_inputs):
        """Test error when variable_names length doesn't match transforms."""
        layer = nn.Linear(5, 2)
        model = SymbolicModel(layer, block_name="error_test")

        transforms = [lambda x: x[:, 0], lambda x: x[:, 1]]
        variable_names = ['x', 'y', 'z']  # Wrong length

        with pytest.raises(ValueError, match="must match length"):
            model.distill(
                sample_inputs,
                variable_transforms=transforms,
                fit_params={'variable_names': variable_names}
            )


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_show_symbolic_expression_without_distill(self, symbolic_model):
        """Test show_symbolic_expression() before distill()."""
        # Should print error message and return early (no exception)
        symbolic_model.show_symbolic_expression()
        # No assertion needed, just verify it doesn't crash

    def test_get_symbolic_function_without_distill(self, symbolic_model):
        """Test get_symbolic_function() before distill()."""
        with pytest.raises(ValueError, match="No"):
            symbolic_model.get_symbolic_function(dim=0)

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_get_symbolic_function_invalid_dimension(self, mock_pysr_class, symbolic_model, sample_inputs):
        """Test get_symbolic_function() with invalid dimension."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0", "loss": 0.001}
        mock_pysr_class.return_value = mock_reg

        # First distill to set output_dims
        symbolic_model.distill(sample_inputs, output_dim=0)

        # Now test with invalid dimension
        with pytest.raises(ValueError, match="out of range"):
            symbolic_model.get_symbolic_function(dim=10)

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_distill_output_dim_out_of_range(self, mock_pysr_class, sample_inputs_np):
        """Test distill() with output_dim out of range for Callable."""
        def my_func(x):
            return x[:, 0:2]  # Returns 2 outputs

        model = SymbolicModel(my_func, block_name="range_test")

        with pytest.raises(ValueError, match="out of range"):
            model.distill(sample_inputs_np, output_dim=5)

    def test_slime_mode_switch_consistency(self):
        """Test that SLIME and standard modes are kept separate."""
        layer = nn.Linear(5, 2)
        model = SymbolicModel(layer, block_name="slime_consistency")

        # Both dictionaries should be independent
        assert len(model.pysr_regressor) == 0
        assert len(model.SLIME_pysr_regressor) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_complete_workflow_module(self, mock_pysr_class, sample_inputs):
        """Test complete workflow: distill -> switch -> forward."""
        # Setup mock
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0", "loss": 0.001}
        mock_reg.equations_ = pd.DataFrame({
            "equation": ["x0"],
            "sympy_format": [sympy.sympify("x0")],
            "complexity": [1],
            "loss": [0.001]
        })

        mock_pysr_class.return_value = mock_reg

        # Create model and workflow
        layer = nn.Linear(5, 3)
        model = SymbolicModel(layer, block_name="integration_test")

        # Distill
        model.distill(sample_inputs)

        # Switch to symbolic
        model.switch_to_symbolic()

        # Test forward
        test_inputs = torch.randn(10, 5)
        output = model(test_inputs)

        assert output.shape == (10, 3)

        # Switch back to block
        model.switch_to_block()
        output_block = model(test_inputs)

        assert output_block.shape == (10, 3)

    @patch('symtorch.SymbolicModel.PySRRegressor')
    def test_complete_workflow_callable(self, mock_pysr_class, sample_inputs_np):
        """Test complete workflow with callable function."""
        mock_reg = MagicMock()
        mock_reg.get_best.return_value = {"equation": "x0**2", "loss": 0.001}
        mock_reg.equations_ = pd.DataFrame({
            "equation": ["x0**2"],
            "sympy_format": [sympy.sympify("x0**2")],
            "complexity": [3],
            "loss": [0.001]
        })

        mock_pysr_class.return_value = mock_reg

        def my_func(x):
            return x[:, 0] ** 2

        model = SymbolicModel(my_func, block_name="callable_integration")

        # Distill
        model.distill(sample_inputs_np)

        # Get symbolic function
        sym_func = model.get_symbolic_function(dim=0)

        # Test it
        test_input = np.array([[2.0, 1.0, 1.0, 1.0, 1.0]])
        result = sym_func(test_input)

        assert result is not None
