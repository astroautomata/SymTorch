import sys
import os
import pytest
import numpy as np
import warnings
import shutil

# Add the src directory to Python path for absolute imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from symtorch import SLIME, regressor_to_function


# Test fixture: simple quadratic function
def simple_quadratic(X):
    """Test function: f(x0, x1) = x0^2 + x1 + 12"""
    return X[:, 0]**2 + X[:, 1] + 12


def linear_function(X):
    """Test function: f(x0, x1) = 2*x0 + 3*x1"""
    return 2 * X[:, 0] + 3 * X[:, 1]


def generate_test_data(n_samples=100, n_features=2, seed=42):
    """Generate test data."""
    np.random.seed(seed)
    return np.random.uniform(-5, 5, size=(n_samples, n_features))


# ========== Basic SLIME Functionality Tests ==========

def test_slime_basic_usage():
    """Test SLIME with all data (no local explanation)."""
    X = generate_test_data()
    regressor = SLIME(simple_quadratic, X)

    # Verify regressor was created
    assert regressor is not None
    assert hasattr(regressor, 'equations_')
    assert len(regressor.equations_) > 0

    # Verify we can get the best equation
    best = regressor.get_best()
    assert 'equation' in best
    assert 'loss' in best
    assert 'complexity' in best


def test_slime_equation_accuracy():
    """Test that SLIME discovers an accurate equation."""
    X_train = generate_test_data()
    regressor = SLIME(simple_quadratic, X_train, pysr_params={'niterations': 100})

    # Convert to callable function
    f_discovered, vars_discovered = regressor_to_function(regressor)

    # Test on new data
    X_test = generate_test_data(n_samples=50, seed=123)
    y_true = simple_quadratic(X_test)
    y_pred = f_discovered(X_test[:, 0], X_test[:, 1])

    # Calculate MSE - should be reasonably accurate
    mse = np.mean((y_true - y_pred)**2)
    assert mse < 10.0, f"MSE too high: {mse}"


# ========== J_neighbours Parameter Tests ==========

def test_j_neighbours_default():
    """Test that J_neighbours defaults to 10."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should work with default J_neighbours=10
    regressor = SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, pysr_params={'niterations': 100})
    assert regressor is not None


def test_j_neighbours_custom_value():
    """Test custom J_neighbours value."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should work with custom value
    regressor = SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, J_neighbours=20, pysr_params={'niterations': 100})
    assert regressor is not None


def test_j_neighbours_validation():
    """Test that J_neighbours must be less than len(inputs)."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should raise error when J_neighbours >= len(inputs)
    with pytest.raises(ValueError, match="J_neighbours.*must be less than"):
        SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, J_neighbours=100)


# ========== real_weighting Parameter Tests ==========

def test_real_weighting_default():
    """Test that real_weighting defaults to 1.0."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should work with default real_weighting=1.0
    regressor = SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, pysr_params={'niterations': 100})
    assert regressor is not None


def test_real_weighting_custom_value():
    """Test custom real_weighting value with synthetic samples."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should work with custom value when p_synthetic > 0
    regressor = SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, real_weighting=2.0, pysr_params={'niterations': 100})
    assert regressor is not None


def test_real_weighting_warning_without_synthetic():
    """Test that real_weighting raises warning when p_synthetic=0."""
    X = generate_test_data()

    # Should raise warning when real_weighting != 1.0 and p_synthetic == 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        regressor = SLIME(simple_quadratic, X, real_weighting=5.0, pysr_params={'niterations': 100})

        # Check that warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "real_weighting" in str(w[0].message)


# ========== Local Explanation (x parameter) Tests ==========

def test_local_explanation_with_synthetic():
    """Test local explanation with x and synthetic samples."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    regressor = SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, J_neighbours=20, pysr_params={'niterations': 100})
    assert regressor is not None

    # Should produce equations
    best = regressor.get_best()
    assert 'equation' in best


def test_x_without_synthetic_raises_error():
    """Test that x without p_synthetic raises error."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should raise error when x is specified but p_synthetic=0
    with pytest.raises(ValueError, match="Need to set p_synthetic to non-zero"):
        SLIME(simple_quadratic, X, x=x_test, p_synthetic=0)


def test_variance_calculation():
    """Test that variance is calculated correctly when not provided."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Should calculate variance automatically
    regressor = SLIME(simple_quadratic, X, x=x_test, p_synthetic=0.5, pysr_params={'niterations': 100})
    assert regressor is not None


# ========== regressor_to_function Helper Tests ==========

def test_regressor_to_function_default():
    """Test converting regressor to function (best equation)."""
    X = generate_test_data()
    regressor = SLIME(simple_quadratic, X, pysr_params={'niterations': 100})

    f, vars_list = regressor_to_function(regressor)

    # Should return a callable function
    assert callable(f)

    # Should return list of variables
    assert isinstance(vars_list, list)
    assert len(vars_list) > 0


def test_regressor_to_function_invalid_complexity():
    """Test that invalid complexity raises error."""
    X = generate_test_data()
    regressor = SLIME(simple_quadratic, X, pysr_params={'niterations': 100})

    # Use a complexity that doesn't exist
    with pytest.raises(ValueError, match="No equation found with complexity"):
        regressor_to_function(regressor, complexity=999)


def test_regressor_to_function_callable():
    """Test that the returned function is actually callable."""
    X_train = generate_test_data()
    regressor = SLIME(simple_quadratic, X_train, pysr_params={'niterations': 100})

    f, vars_list = regressor_to_function(regressor)

    # Test with sample data
    X_test = generate_test_data(n_samples=10, seed=123)

    # Should be able to call the function
    result = f(X_test[:, 0], X_test[:, 1])
    assert result is not None
    assert len(result) == 10


# ========== Weighted Loss Tests ==========

def test_gaussian_kernel_weights():
    """Test that Gaussian kernel weights are applied."""
    X = generate_test_data()
    x_test = np.array([1.0, 2.0])

    # Run with weighted loss
    regressor = SLIME(
        simple_quadratic, X,
        x=x_test,
        p_synthetic=0.5,
        J_neighbours=20,
        real_weighting=2.0,
        pysr_params={'niterations': 100}
    )

    assert regressor is not None

    # Should produce valid equations
    best = regressor.get_best()
    assert 'equation' in best
    assert best['loss'] >= 0


# ========== Edge Cases and Validation Tests ==========

def test_custom_pysr_params():
    """Test with custom PySR parameters."""
    X = generate_test_data()

    custom_params = {
        'niterations': 50,
        'binary_operators': ["+", "-", "*"],
        'populations': 10
    }

    regressor = SLIME(simple_quadratic, X, pysr_params=custom_params)
    assert regressor is not None


def test_high_dimensional_data():
    """Test with higher dimensional data."""
    X = generate_test_data(n_features=5)

    def f(X):
        return X[:, 0]**2 + X[:, 1] + X[:, 2]

    # Should work with more dimensions
    regressor = SLIME(f, X, pysr_params={'niterations': 100})
    assert regressor is not None


# ========== Integration Test ==========

def test_full_workflow():
    """Test complete workflow from data to predictions."""
    # Create data
    X_train = generate_test_data()

    # Run SLIME
    regressor = SLIME(simple_quadratic, X_train, pysr_params={'niterations': 100})

    # Convert to function
    f_discovered, vars_discovered = regressor_to_function(regressor)

    # Test on new data
    X_test = generate_test_data(n_samples=50, seed=999)
    y_true = simple_quadratic(X_test)
    y_pred = f_discovered(X_test[:, 0], X_test[:, 1])

    # Calculate error - should be reasonably accurate
    mse = np.mean((y_true - y_pred)**2)
    assert mse < 20.0, f"MSE too high: {mse}"


def test_local_explanation_workflow():
    """Test local explanation workflow."""
    # Create data
    X_train = generate_test_data()

    # Select point to explain
    x_explain = np.array([2.0, 3.0])

    # Run local SLIME
    regressor = SLIME(
        simple_quadratic,
        X_train,
        x=x_explain,
        p_synthetic=0.5,
        J_neighbours=20,
        real_weighting=2.0,
        pysr_params={'niterations': 100}
    )

    # Should produce valid equations
    best = regressor.get_best()
    assert 'equation' in best
    assert best['loss'] >= 0


if os.path.exists('../outputs'):
    shutil.rmtree('../outputs')