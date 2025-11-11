import sys
import os
import pytest
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import shutil

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from symtorch import SLIMEModel, regressor_to_function


# Test data generation
def generate_test_data(n_samples=200, n_features=3, seed=42):
    """Generate synthetic test data with known relationships."""
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, size=(n_samples, n_features))
    y = X[:, 0]**2 + 2*np.sin(X[:, 1]) + X[:, 2] + np.random.normal(0, 0.1, n_samples)
    return X, y


def train_simple_model(X_train, y_train):
    """Train a simple sklearn model for testing."""
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model


# ========== SLIMEModel Initialization Tests ==========

def test_slime_model_initialization_defaults():
    """Test SLIMEModel initializes with default parameters."""
    slime = SLIMEModel()

    assert slime.J_neighbours == 10
    assert slime.num_synthetic == 0
    assert slime.real_weighting == 1.0
    assert slime.nn_metric == 'euclidean'
    assert slime.pysr_params == {}
    assert slime.regressor_ is None


def test_slime_model_initialization_custom():
    """Test SLIMEModel initializes with custom parameters."""
    custom_params = {
        'niterations': 500,
        'binary_operators': ['+', '*', '-']
    }

    slime = SLIMEModel(
        J_neighbours=20,
        num_synthetic=1000,
        real_weighting=2.5,
        nn_metric='manhattan',
        pysr_params=custom_params
    )

    assert slime.J_neighbours == 20
    assert slime.num_synthetic == 1000
    assert slime.real_weighting == 2.5
    assert slime.nn_metric == 'manhattan'
    assert slime.pysr_params == custom_params


# ========== SLIMEModel.fit() Tests ==========

def test_fit_without_local_point():
    """Test fitting SLIME on all data without specifying a point."""
    X, y = generate_test_data(n_samples=100)
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 50})
    slime.fit(model.predict, X)

    assert slime.regressor_ is not None
    assert hasattr(slime.regressor_, 'equations_')
    assert slime.x0_ is None
    assert slime.real_inputs_ is not None


def test_fit_with_local_point():
    """Test fitting SLIME around a specific point."""
    X, y = generate_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_simple_model(X_train, y_train)

    x0 = X_test[0]
    slime = SLIMEModel(
        J_neighbours=15,
        num_synthetic=500,
        pysr_params={'niterations': 50}
    )
    slime.fit(model.predict, X_test, x=x0)

    assert slime.regressor_ is not None
    assert slime.x0_ is not None
    assert np.array_equal(slime.x0_, x0)
    assert slime.real_inputs_ is not None
    assert slime.synthetic_samples_ is not None
    assert len(slime.real_inputs_) == 15
    assert len(slime.synthetic_samples_) == 500


def test_fit_requires_num_synthetic_with_x():
    """Test that fit raises error when x is specified without num_synthetic."""
    X, y = generate_test_data()
    model = train_simple_model(X, y)

    x0 = X[0]
    slime = SLIMEModel(num_synthetic=0)

    with pytest.raises(ValueError, match="num_synthetic must be > 0"):
        slime.fit(model.predict, X, x=x0)


def test_fit_validates_j_neighbours():
    """Test that fit validates J_neighbours < len(inputs)."""
    X, y = generate_test_data(n_samples=50)
    model = train_simple_model(X, y)

    x0 = X[0]
    slime = SLIMEModel(J_neighbours=100, num_synthetic=100)

    with pytest.raises(ValueError, match="J_neighbours.*must be <"):
        slime.fit(model.predict, X, x=x0)


# ========== SLIMEModel.predict() Tests ==========

def test_predict_basic():
    """Test basic prediction functionality."""
    X, y = generate_test_data(n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_simple_model(X_train, y_train)

    slime = SLIMEModel(pysr_params={'niterations': 100})
    slime.fit(model.predict, X_train)

    # Test predictions
    y_pred = slime.predict(X_test)

    assert y_pred is not None
    assert len(y_pred) == len(X_test)
    assert not np.isnan(y_pred).any()


def test_predict_single_sample():
    """Test prediction on a single sample."""
    X, y = generate_test_data()
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 50})
    slime.fit(model.predict, X[:100])

    # Predict on single sample
    x_single = X[0].reshape(1, -1)
    y_pred = slime.predict(x_single)

    assert len(y_pred) == 1
    assert not np.isnan(y_pred[0])


def test_predict_before_fit_raises_error():
    """Test that predict raises error if model hasn't been fitted."""
    X, _ = generate_test_data()
    slime = SLIMEModel()

    with pytest.raises(ValueError, match="Model not fitted"):
        slime.predict(X)


# ========== SLIMEModel.get_equation() Tests ==========

def test_get_equation():
    """Test getting the best equation."""
    X, y = generate_test_data(n_samples=100)
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 50})
    slime.fit(model.predict, X)

    equation = slime.get_equation()

    assert equation is not None
    assert isinstance(equation, str)
    assert len(equation) > 0


def test_get_equation_specific_complexity():
    """Test getting equation with specific complexity."""
    X, y = generate_test_data(n_samples=100)
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 100})
    slime.fit(model.predict, X)

    # Get available complexities
    complexities = slime.equations_['complexity'].unique()

    if len(complexities) > 1:
        target_complexity = sorted(complexities)[1]
        equation = slime.get_equation(complexity=target_complexity)

        assert equation is not None
        assert isinstance(equation, str)


def test_get_equation_invalid_complexity():
    """Test that invalid complexity raises error."""
    X, y = generate_test_data(n_samples=100)
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 50})
    slime.fit(model.predict, X)

    with pytest.raises(ValueError, match="No equation with complexity"):
        slime.get_equation(complexity=9999)


def test_get_equation_before_fit_raises_error():
    """Test that get_equation raises error before fitting."""
    slime = SLIMEModel()

    with pytest.raises(ValueError, match="Model not fitted"):
        slime.get_equation()


# ========== SLIMEModel.equations_ Property Tests ==========

def test_equations_property():
    """Test accessing equations DataFrame."""
    X, y = generate_test_data(n_samples=100)
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 50})
    slime.fit(model.predict, X)

    equations = slime.equations_

    assert equations is not None
    assert 'equation' in equations.columns
    assert 'loss' in equations.columns
    assert 'complexity' in equations.columns
    assert len(equations) > 0


def test_equations_before_fit_raises_error():
    """Test that equations_ raises error before fitting."""
    slime = SLIMEModel()

    with pytest.raises(ValueError, match="Model not fitted"):
        _ = slime.equations_


# ========== Attributes After Fit Tests ==========

def test_attributes_set_after_fit_global():
    """Test that attributes are properly set after fitting on all data."""
    X, y = generate_test_data()
    model = train_simple_model(X, y)

    slime = SLIMEModel(pysr_params={'niterations': 50})
    slime.fit(model.predict, X)

    assert slime.x0_ is None
    assert slime.real_inputs_ is not None
    assert slime.synthetic_samples_ is None
    assert slime.weights_ is None
    assert slime.var_ is None
    assert slime.regressor_ is not None


def test_attributes_set_after_fit_local():
    """Test that attributes are properly set after local fitting."""
    X, y = generate_test_data()
    model = train_simple_model(X, y)

    x0 = X[0]
    slime = SLIMEModel(J_neighbours=10, num_synthetic=100, pysr_params={'niterations': 50})
    slime.fit(model.predict, X, x=x0)

    assert slime.x0_ is not None
    assert slime.real_inputs_ is not None
    assert slime.synthetic_samples_ is not None
    assert slime.weights_ is not None
    assert slime.var_ is not None
    assert slime.regressor_ is not None
    assert len(slime.weights_) == 10 + 100  # J_neighbours + num_synthetic


# ========== Real Weighting Tests ==========

def test_real_weighting_applied():
    """Test that real_weighting parameter is applied correctly."""
    X, y = generate_test_data()
    model = train_simple_model(X, y)

    x0 = X[0]
    slime = SLIMEModel(
        J_neighbours=10,
        num_synthetic=50,
        real_weighting=5.0,
        pysr_params={'niterations': 50}
    )
    slime.fit(model.predict, X, x=x0)

    assert slime.weights_ is not None
    # First 10 weights should be 5.0 (real_weighting)
    assert np.all(slime.weights_[:10] == 5.0)
    # Remaining weights should be Gaussian kernel weights (0 to 1)
    assert np.all(slime.weights_[10:] <= 1.0)


# ========== Integration Tests ==========

def test_full_workflow_global():
    """Test complete workflow: fit on all data, predict, get equation."""
    X, y = generate_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_simple_model(X_train, y_train)

    # Fit SLIME
    slime = SLIMEModel(pysr_params={'niterations': 200})
    slime.fit(model.predict, X_train)

    # Get equation
    equation = slime.get_equation()
    assert len(equation) > 0

    # Make predictions
    y_pred = slime.predict(X_test)
    y_true = model.predict(X_test)

    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    assert r2 > 0.5, f"R² too low: {r2}"


def test_full_workflow_local():
    """Test complete workflow: local fit, predict, get equation."""
    X, y = generate_test_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_simple_model(X_train, y_train)

    # Fit local SLIME
    x0 = X_test[0]
    slime = SLIMEModel(
        J_neighbours=20,
        num_synthetic=500,
        real_weighting=2.0,
        pysr_params={'niterations': 200}
    )
    slime.fit(model.predict, X_test, x=x0)

    # Get equation
    equation = slime.get_equation()
    assert len(equation) > 0

    # Make predictions on neighborhood
    X_neighborhood = X_test[:30]  # Small neighborhood
    y_pred = slime.predict(X_neighborhood)
    y_true = model.predict(X_neighborhood)

    # Should have reasonable accuracy in local region
    mse = mean_squared_error(y_true, y_pred)
    assert mse < 10.0, f"MSE too high: {mse}"


# ========== Cleanup ==========

def cleanup_sr_outputs():
    """Clean up SR output files and directories."""
    if os.path.exists('SR_output'):
        shutil.rmtree('SR_output')

    for file in os.listdir('.'):
        if file.startswith('hall_of_fame'):
            try:
                os.remove(file)
            except OSError:
                pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Cleanup fixture to remove output files after tests."""
    yield
    cleanup_sr_outputs()
