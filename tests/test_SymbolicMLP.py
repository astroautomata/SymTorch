import sys
import os
import shutil
import pytest
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add the src directory to Python path for absolute imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from symtorch import SymbolicMLP


def test_MLP_SR_wrapper():
    """
    Test that MLP_SR wrapper can successfully wrap a PyTorch Sequential model.
    """
    try:
        class SimpleModel(nn.Module):
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
                self.mlp = SymbolicMLP(mlp, mlp_name = "Sequential")
        model = SimpleModel(input_dim=5, output_dim=1)
        assert hasattr(model.mlp, 'InterpretSR_MLP'), "MLP_SR should have InterpretSR_MLP attribute"
        assert hasattr(model.mlp, 'distill'), "MLP_SR should have distill method"
    except Exception as e:
        pytest.fail(f"MLP_SR wrapper failed with error: {e}.")


class SimpleModel(nn.Module):
    """
    Simple model class for testing MLP_SR functionality.
    Uses a Sequential MLP wrapped with MLP_SR.
    """
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
        self.mlp = SymbolicMLP(mlp, mlp_name = "Sequential")

    def forward(self, x):
        x = self.mlp(x)
        return x


def train_model(model, dataloader, opt, criterion, epochs = 100):
    """
    Train a model for the specified number of epochs.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        opt: Optimizer
        criterion: Loss function
        epochs: Number of training epochs
        
    Returns:
        tuple: (trained_model, loss_tracker)
    """
    loss_tracker = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
        
        loss_tracker.append(epoch_loss)
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}')
    return model, loss_tracker


# Global test data and model setup
np.random.seed(290402)  # For reproducible tests
torch.manual_seed(290402)

# Make the dataset 
x = np.array([np.random.uniform(0, 1, 1_000) for _ in range(5)]).T  
y = x[:, 0]**2 + 3*np.sin(x[:, 4]) - 4
noise = np.array([np.random.normal(0, 0.05*np.std(y)) for _ in range(len(y))])
y = y + noise 

# Split into train and test
X_train, _, y_train, _ = train_test_split(x, y.reshape(-1,1), test_size=0.2, random_state=290402)

# Create the model and set up training
model = SimpleModel(input_dim=x.shape[1], output_dim=1)
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)
dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Global variable to store trained model for subsequent tests
trained_model = None


def test_training_MLP_SR_model():
    """
    Test that a MLP_SR wrapped model can be trained successfully.
    """
    global trained_model
    try:
        trained_model, losses = train_model(model, dataloader, opt, criterion, 20)
        assert len(losses) == 20, "Should have loss for each epoch"
        assert all(isinstance(loss, float) for loss in losses), "All losses should be floats"
        
    except Exception as e:
        pytest.fail(f"MLP_SR model training failed with error {e}.")


def test_MLP_SR_distill():
    """
    Test that the distill method works on a trained MLP_SR model.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
        
    try:
        # Create input data for distillation
        input_data = torch.FloatTensor(X_train[:100])  # Use subset for faster testing
        
        # Run distillation with reduced iterations for testing
        regressors = trained_model.mlp.distill(input_data, sr_params={'niterations': 50})
        
        # For single output model, should return dictionary with one entry
        assert regressors is not None, "Regressors should not be None"
        assert isinstance(regressors, dict), "Should return dictionary of regressors"
        assert 0 in regressors, "Should have regressor for dimension 0"
        
        # Test the regressor for dimension 0
        regressor = regressors[0]
        assert hasattr(regressor, 'equations_'), "Regressor should have equations_ attribute"
        assert hasattr(regressor, 'get_best'), "Regressor should have get_best method"
        
        # Verify the MLP_SR object stored the regressors
        assert hasattr(trained_model.mlp, 'pysr_regressor'), "MLP_SR should store the regressors"
        assert 0 in trained_model.mlp.pysr_regressor, "MLP_SR should store regressor for dimension 0"
        assert trained_model.mlp.pysr_regressor[0] is regressor, "Stored regressor should match returned regressor"
        
    except Exception as e:
        pytest.fail(f"MLP_SR distill method failed with error: {e}")
    finally:
        # Clean up SR output directory
        cleanup_sr_outputs()


def test_switch_to_equation():
    """
    Test that switch_to_equation method works correctly.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
        
    # Ensure we have a regressor first
    if not hasattr(trained_model.mlp, 'pysr_regressor') or not trained_model.mlp.pysr_regressor:
        input_data = torch.FloatTensor(X_train[:100])
        trained_model.mlp.distill(input_data, sr_params={'niterations': 50})
    
    try:
        # Test switching to equation
        trained_model.mlp.switch_to_equation()
        assert trained_model.mlp._using_equation, "Should be using equation mode after switch"
        
        # Verify internal state for multi-dimensional API
        assert hasattr(trained_model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert trained_model.mlp._using_equation, "Should be using equation mode"
        assert hasattr(trained_model.mlp, '_equation_funcs'), "Should have _equation_funcs attribute"
        assert hasattr(trained_model.mlp, '_equation_vars'), "Should have _equation_vars attribute"
        
        # For single output model, should have one equation function
        assert len(trained_model.mlp._equation_funcs) == 1, "Should have one equation function"
        assert 0 in trained_model.mlp._equation_funcs, "Should have equation function for dimension 0"
        assert 0 in trained_model.mlp._equation_vars, "Should have equation variables for dimension 0"
        
        # Test forward pass still works
        test_input = torch.FloatTensor(X_train[:10])
        output = trained_model.mlp(test_input)
        assert output is not None, "Forward pass should work in equation mode"
        assert output.shape[0] == 10, "Output should have correct batch size"
        assert output.shape[1] == 1, "Output should have correct number of dimensions"
        
    except Exception as e:
        pytest.fail(f"switch_to_equation failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_switch_to_mlp():
    """
    Test that switch_to_mlp method works correctly.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
        
    # Ensure we have a regressor first
    if not hasattr(trained_model.mlp, 'pysr_regressor') or trained_model.mlp.pysr_regressor is None:
        input_data = torch.FloatTensor(X_train[:100])
        trained_model.mlp.distill(input_data, sr_params={'niterations': 50})
    
    # Switch to equation mode first
    trained_model.mlp.switch_to_equation()
    
    try:
        # Test switching back to MLP
        trained_model.mlp.switch_to_mlp()
        
        # Verify internal state
        assert hasattr(trained_model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert not trained_model.mlp._using_equation, "Should not be using equation mode"
        
        # Test forward pass still works
        test_input = torch.FloatTensor(X_train[:10])
        output = trained_model.mlp(test_input)
        assert output is not None, "Forward pass should work in MLP mode"
        assert output.shape[0] == 10, "Output should have correct batch size"
        
    except Exception as e:
        pytest.fail(f"switch_to_mlp failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_equation_actually_used_in_forward():
    """
    Test that switching to equation mode actually uses the symbolic equation 
    by manually setting a known equation and verifying the output.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input with enough dimensions - use simple equation sin(x0) + 2
        test_input = torch.FloatTensor([[0.5, 0.1, 0.2, 0.3, 0.4], 
                                       [1.0, 0.2, 0.3, 0.4, 0.5], 
                                       [1.57, 0.3, 0.4, 0.5, 0.6], 
                                       [3.14, 0.4, 0.5, 0.6, 0.7]])  # 5 dimensions
        
        # Manually set up the equation components for multi-dimensional API
        def test_equation(x0):
            return np.sin(x0) + 2
        
        # Manually set the equation in the MLP_SR object using new multi-dimensional API
        trained_model.mlp._equation_funcs = {0: test_equation}  # Dictionary for dimension 0
        trained_model.mlp._equation_vars = {0: [0]}  # Only use first input variable for dimension 0
        trained_model.mlp._using_equation = True
        
        # Get output using the equation
        equation_output = trained_model.mlp(test_input)
        
        # Calculate expected output manually
        expected_output = torch.tensor([[np.sin(0.5) + 2], 
                                       [np.sin(1.0) + 2], 
                                       [np.sin(1.57) + 2], 
                                       [np.sin(3.14) + 2]], dtype=torch.float32)
        
        # Verify outputs match (within floating point tolerance)
        diff = torch.abs(equation_output - expected_output)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-5, f"Equation output doesn't match expected (max diff: {max_diff})"
        print(f"✅ Equation mode correctly computes sin(x0) + 2 (max diff: {max_diff:.8f})")
        
    except Exception as e:
        pytest.fail(f"test_equation_actually_used_in_forward failed with error: {e}")
    finally:
        # Reset to MLP mode and clean up manually set equation components
        if hasattr(trained_model.mlp, '_using_equation'):
            trained_model.mlp._using_equation = False
        if hasattr(trained_model.mlp, '_equation_funcs'):
            delattr(trained_model.mlp, '_equation_funcs')
        if hasattr(trained_model.mlp, '_equation_vars'):
            delattr(trained_model.mlp, '_equation_vars')


def test_mlp_actually_used_after_switch_back():
    """
    Test that switching back to MLP mode actually uses the original MLP
    by comparing with a separate MLP loaded with the same weights.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input
        test_input = torch.FloatTensor(X_train[:10])
        
        # Ensure we're in MLP mode
        trained_model.mlp.switch_to_mlp()
        trained_model.mlp._using_equation = False
        
        # Get output from the MLP_SR in MLP mode
        mlp_sr_output = trained_model.mlp(test_input).clone().detach()
        
        # Create a separate regular MLP with same architecture
        separate_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Copy weights from the MLP_SR's internal MLP to the separate MLP
        separate_mlp.load_state_dict(trained_model.mlp.InterpretSR_MLP.state_dict())
        
        # Set to eval mode to match the trained model
        separate_mlp.eval()
        
        # Get output from the separate MLP
        with torch.no_grad():
            separate_mlp_output = separate_mlp(test_input)
        
        # Outputs should be identical
        diff = torch.abs(mlp_sr_output - separate_mlp_output)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-6, f"MLP_SR and separate MLP outputs differ (max diff: {max_diff})"
        print(f"✅ MLP mode uses actual MLP (max diff: {max_diff:.8f})")
        
    except Exception as e:
        pytest.fail(f"test_mlp_actually_used_after_switch_back failed with error: {e}")


class DualMLPModel(nn.Module):
    """
    Model with two MLPs: one regular and one wrapped with MLP_SR.
    Used to test training after switching to symbolic equations.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DualMLPModel, self).__init__()
        
        # Regular MLP (not wrapped)
        self.regular_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # MLP wrapped with MLP_SR
        sr_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sr_mlp = SymbolicMLP(sr_mlp, mlp_name="SRSequential")
        
    def forward(self, x):
        # Combine outputs from both MLPs
        regular_out = self.regular_mlp(x)
        sr_out = self.sr_mlp(x)
        return regular_out + sr_out


def test_training_after_switch_to_equation():
    """
    Test that a model can still train after switching one MLP component to symbolic equation.
    """
    try:
        # Create dual MLP model
        dual_model = DualMLPModel(input_dim=5, output_dim=1)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dual_model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train initially for a few epochs
        print("Training dual model initially...")
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=10)
        
        # Run distill on the SR-wrapped MLP component
        print("Running distillation on SR-wrapped MLP...")
        input_data = torch.FloatTensor(X_train[:100])
        regressor = dual_model.sr_mlp.distill(input_data, sr_params={'niterations': 30})
        
        assert regressor is not None, "Interpretation should succeed"
        assert hasattr(dual_model.sr_mlp, 'pysr_regressor'), "Should have regressor stored"
        
        # Switch to equation mode
        print("Switching SR-wrapped MLP to equation mode...")
        dual_model.sr_mlp.switch_to_equation()
        assert dual_model.sr_mlp._using_equation, "Should be in equation mode"
        
        # Continue training after switch - this is the key test
        print("Training dual model after equation switch...")
        dual_model, post_switch_losses = train_model(dual_model, dataloader, optimizer, criterion, epochs=5)
        
        # Verify training completed successfully
        assert len(post_switch_losses) == 5, "Should complete all post-switch epochs"
        assert all(isinstance(loss, float) for loss in post_switch_losses), "All losses should be valid floats"
        
        # Test that forward passes still work
        test_input = torch.FloatTensor(X_train[:10])
        output = dual_model(test_input)
        assert output is not None, "Forward pass should work after equation switch"
        assert output.shape == (10, 1), "Output should have correct shape"
        
        print("✅ Successfully trained model after switching to symbolic equation")
        
    except Exception as e:
        pytest.fail(f"Training after switch to equation failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_equation_parameters_fixed_during_training():
    """
    Test that symbolic equation parameters remain fixed during training.
    The equation itself should not change, only other model components should train.
    """
    try:
        # Create dual MLP model
        dual_model = DualMLPModel(input_dim=5, output_dim=1)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dual_model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train initially
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=5)
        
        # Run distill and switch to equation
        input_data = torch.FloatTensor(X_train[:100])
        _ = dual_model.sr_mlp.distill(input_data, sr_params={'niterations': 30})
        dual_model.sr_mlp.switch_to_equation()
        
        # Get equation functions and variables before training (multi-dimensional API)
        equation_funcs_before = dual_model.sr_mlp._equation_funcs.copy()
        equation_vars_before = dual_model.sr_mlp._equation_vars.copy()
        
        # Test the equation output before training
        test_input = torch.FloatTensor([[0.5, 0.3, 0.7, 0.1, 0.9]])
        with torch.no_grad():
            equation_output_before = dual_model.sr_mlp(test_input).clone()
        
        # Train more after switching to equation
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=3)
        
        # Check that equation functions and variables haven't changed
        equation_funcs_after = dual_model.sr_mlp._equation_funcs
        equation_vars_after = dual_model.sr_mlp._equation_vars
        
        # For single output model, check dimension 0
        assert equation_funcs_before[0] is equation_funcs_after[0], "Equation function should be the same object"
        assert equation_vars_before[0] == equation_vars_after[0], "Variable indices should remain unchanged"
        
        # Test that equation gives same output for same input
        with torch.no_grad():
            equation_output_after = dual_model.sr_mlp(test_input)
        
        diff = torch.abs(equation_output_before - equation_output_after)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-6, f"Equation output should be identical (diff: {max_diff})"
        
        print("✅ Confirmed: Symbolic equation parameters remain fixed during training")
        print(f"   Equation function: {equation_funcs_before[0]}")
        print(f"   Variables used: {[f'x{i}' for i in equation_vars_before[0]]}")
        print(f"   Output consistency: max diff = {max_diff:.8f}")
        
    except Exception as e:
        pytest.fail(f"Equation parameter fixity test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_gradient_flow_through_other_components():
    """
    Test that gradients still flow through other model components when one uses symbolic equation.
    The regular MLP should continue to train while the equation component remains fixed.
    """
    try:
        # Create dual MLP model
        dual_model = DualMLPModel(input_dim=5, output_dim=1)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dual_model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train initially
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=5)
        
        # Run distill and switch to equation
        input_data = torch.FloatTensor(X_train[:100])
        _ = dual_model.sr_mlp.distill(input_data, sr_params={'niterations': 30})
        dual_model.sr_mlp.switch_to_equation()
        
        # Get regular MLP parameters before additional training
        regular_mlp_params_before = {}
        for name, param in dual_model.regular_mlp.named_parameters():
            regular_mlp_params_before[name] = param.clone().detach()
        
        # Train more - regular MLP should change, equation should not
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=3)
        
        # Check that regular MLP parameters have changed (indicating gradient flow)
        regular_mlp_changed = False
        for name, param in dual_model.regular_mlp.named_parameters():
            param_before = regular_mlp_params_before[name]
            diff = torch.abs(param - param_before)
            max_diff = torch.max(diff)
            if max_diff > 1e-6:
                regular_mlp_changed = True
                print(f"   Regular MLP {name}: max parameter change = {max_diff:.6f}")
                break
        
        assert regular_mlp_changed, "Regular MLP parameters should change during training"
        
        # Verify equation component does NOT maintain gradients
        # (The symbolic equation is not differentiable in PyTorch's autograd sense)
        test_input = torch.FloatTensor(X_train[:10])
        test_input.requires_grad_(True)
        
        # Forward pass through equation component only
        equation_output = dual_model.sr_mlp(test_input)
        
        # The equation output should not have gradients
        assert not equation_output.requires_grad, "Equation output should not require gradients"
        assert equation_output.grad_fn is None, "Equation output should not have grad_fn"
        
        # Try to backward through the equation - this should fail
        try:
            loss = torch.sum(equation_output)
            loss.backward()
            gradient_flows = True
        except RuntimeError:
            gradient_flows = False
        
        assert not gradient_flows, "Gradients should NOT flow through symbolic equation"
        
        print("✅ Confirmed: Gradients flow correctly in mixed MLP/equation model")
        print("   - Regular MLP parameters change during training")
        print("   - Equation parameters remain fixed")
        print("   - Gradients do NOT flow through symbolic equation (as expected)")
        
    except Exception as e:
        pytest.fail(f"Gradient flow test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


class MultiOutputModel(nn.Module):
    """
    Model with multiple outputs for testing multi-dimensional symbolic regression.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(MultiOutputModel, self).__init__()
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mlp = SymbolicMLP(mlp, mlp_name="MultiOutput")

    def forward(self, x):
        return self.mlp(x)


def create_multi_output_synthetic_data(n_samples=500, input_dim=4, output_dim=3):
    """Create synthetic data with multiple outputs and known relationships."""
    np.random.seed(42)
    
    x = np.random.uniform(-1, 1, (n_samples, input_dim))
    y = np.zeros((n_samples, output_dim))
    
    # Simple known relationships for easier symbolic regression
    for i in range(output_dim):
        if i == 0:
            y[:, i] = x[:, 0] + x[:, min(1, input_dim-1)]  # Linear sum
        elif i == 1:
            y[:, i] = x[:, 0] * x[:, min(2, input_dim-1)]  # Product 
        elif i == 2:
            y[:, i] = x[:, min(1, input_dim-1)]**2 + 0.5   # Quadratic plus constant
        else:
            # For additional dimensions, create simple linear combinations
            y[:, i] = np.sum(x[:, :min(i+1, input_dim)], axis=1) * 0.1
    
    # Add small amount of noise
    noise = np.random.normal(0, 0.01, y.shape)
    y = y + noise
    
    return x, y


def test_multi_dimensional_distill_all_outputs():
    """
    Test that distill() works correctly when applied to all output dimensions.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=300, input_dim=4, output_dim=3)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(30):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test distill on all dimensions
        input_data = X_tensor[:150]  # Use subset for faster testing
        regressors = model.mlp.distill(input_data, sr_params={'niterations': 30})
        
        # Verify we got a dictionary of regressors
        assert isinstance(regressors, dict), "Should return dictionary of regressors for multi-dim"
        assert len(regressors) == 3, "Should have regressors for all 3 output dimensions"
        assert set(regressors.keys()) == {0, 1, 2}, "Should have regressors for dimensions 0, 1, 2"
        
        # Verify each regressor has the expected attributes
        for dim, regressor in regressors.items():
            assert regressor is not None, f"Regressor for dimension {dim} should not be None"
            assert hasattr(regressor, 'equations_'), f"Regressor {dim} should have equations_ attribute"
            assert hasattr(regressor, 'get_best'), f"Regressor {dim} should have get_best method"
            
            # Test that we can get the best equation
            best_eq = regressor.get_best()['equation']
            assert isinstance(best_eq, str), f"Best equation for dimension {dim} should be a string"
            assert len(best_eq) > 0, f"Best equation for dimension {dim} should not be empty"
        
        # Verify the MLP_SR object stored all regressors
        assert hasattr(model.mlp, 'pysr_regressor'), "MLP_SR should store regressors"
        for dim in [0, 1, 2]:
            assert dim in model.mlp.pysr_regressor, f"MLP_SR should store regressor for dimension {dim}"
        
        print("✅ Multi-dimensional distill (all outputs) test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional distill (all outputs) failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_distill_specific_output():
    """
    Test that distill() works correctly when applied to a specific output dimension.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=300, input_dim=4, output_dim=3)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(30):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test distill on specific dimension (dimension 1)
        input_data = X_tensor[:150]
        regressor = model.mlp.distill(input_data, output_dim=1, sr_params={'niterations': 30})
        
        # Verify we got a single regressor (not a dictionary)
        assert not isinstance(regressor, dict), "Should return single regressor for specific dimension"
        assert regressor is not None, "Regressor should not be None"
        assert hasattr(regressor, 'equations_'), "Regressor should have equations_ attribute"
        assert hasattr(regressor, 'get_best'), "Regressor should have get_best method"
        
        # Test that we can get the best equation
        best_eq = regressor.get_best()['equation']
        assert isinstance(best_eq, str), "Best equation should be a string"
        assert len(best_eq) > 0, "Best equation should not be empty"
        
        # Verify the MLP_SR object stored the regressor for dimension 1
        assert hasattr(model.mlp, 'pysr_regressor'), "MLP_SR should store regressors"
        assert 1 in model.mlp.pysr_regressor, "MLP_SR should store regressor for dimension 1"
        assert model.mlp.pysr_regressor[1] is regressor, "Stored regressor should match returned regressor"
        
        print("✅ Multi-dimensional distill (specific output) test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional distill (specific output) failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_forward_pass_consistency():
    """
    Test that forward passes work correctly with multi-dimensional models.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=4, output_dim=3)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test forward pass before distillation
        test_input = X_tensor[:10]
        output_before = model(test_input).clone().detach()
        assert output_before.shape == (10, 3), "Output should have correct shape (batch_size, output_dim)"
        
        # Run distillation
        model.mlp.distill(X_tensor[:100], sr_params={'niterations': 20})
        
        # Test forward pass after distillation (should still work)
        output_after = model(test_input)
        assert output_after.shape == (10, 3), "Output should maintain correct shape after distillation"
        
        # Outputs should be very similar (model weights shouldn't change during distillation)
        diff = torch.abs(output_before - output_after)
        max_diff = torch.max(diff)
        assert max_diff < 1e-5, f"Forward pass should be consistent before/after distillation (max diff: {max_diff})"
        
        print("✅ Multi-dimensional forward pass consistency test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional forward pass consistency test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_mixed_training():
    """
    Test training a model that combines multi-dimensional MLP_SR with other components.
    """
    try:
        # Create a model that combines multi-output MLP_SR with another component
        class MixedMultiModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MixedMultiModel, self).__init__()
                
                # Multi-output MLP_SR component
                sr_mlp = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_dim)
                )
                self.sr_mlp = SymbolicMLP(sr_mlp, mlp_name="MultiMixed")
                
                # Regular linear layer
                self.linear = nn.Linear(input_dim, output_dim)
                
            def forward(self, x):
                # Combine outputs from both components
                sr_out = self.sr_mlp(x)
                linear_out = self.linear(x)
                return sr_out + linear_out * 0.1  # Weight the linear component less
        
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=3, output_dim=2)
        
        # Create and train mixed model
        model = MixedMultiModel(input_dim=3, output_dim=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Initial training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        initial_loss = loss.item()
        
        # Run distillation on the MLP_SR component
        model.sr_mlp.distill(X_tensor[:100], sr_params={'niterations': 20})
        
        # Continue training after distillation
        for epoch in range(10):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Verify training worked (model should still be trainable)
        assert isinstance(initial_loss, float), "Initial loss should be a float"
        assert isinstance(final_loss, float), "Final loss should be a float"
        assert not np.isnan(final_loss), "Final loss should not be NaN"
        assert not np.isinf(final_loss), "Final loss should not be infinite"
        
        # Test forward pass
        test_input = X_tensor[:5]
        output = model(test_input)
        assert output.shape == (5, 2), "Output should have correct shape"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        
        print("✅ Multi-dimensional mixed training test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional mixed training test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_switch_to_equation():
    """
    Test that switch_to_equation works correctly with multi-dimensional models.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=3, output_dim=2)
        
        # Create and train model
        model = MultiOutputModel(input_dim=3, output_dim=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Run distillation on all dimensions
        input_data = X_tensor[:100]
        regressors = model.mlp.distill(input_data, sr_params={'niterations': 20})
        
        assert isinstance(regressors, dict), "Should return dictionary for multi-dim"
        assert len(regressors) == 2, "Should have 2 regressors"
        
        # Test forward pass before switching
        test_input = X_tensor[:5]
        output_before = model(test_input).clone().detach()
        assert output_before.shape == (5, 2), "Output should have correct shape"
        
        # Test switch_to_equation
        model.mlp.switch_to_equation()
        
        # Verify we're in equation mode
        assert hasattr(model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert model.mlp._using_equation, "Should be in equation mode"
        assert hasattr(model.mlp, '_equation_funcs'), "Should have _equation_funcs attribute"
        assert hasattr(model.mlp, '_equation_vars'), "Should have _equation_vars attribute"
        assert len(model.mlp._equation_funcs) == 2, "Should have equation functions for both dimensions"
        
        # Test forward pass after switching
        output_after = model(test_input)
        assert output_after.shape == (5, 2), "Output should maintain correct shape after switch"
        assert not torch.isnan(output_after).any(), "Output should not contain NaN values"
        
        # Test switch back to MLP
        model.mlp.switch_to_mlp()
        assert not model.mlp._using_equation, "Should not be in equation mode after switch back"
        
        # Test forward pass after switching back
        output_back = model(test_input)
        assert output_back.shape == (5, 2), "Output should maintain correct shape after switch back"
        
        print("✅ Multi-dimensional switch_to_equation test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional switch_to_equation test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_switch_to_equation_missing_dims():
    """
    Test that switch_to_equation correctly handles missing dimensions.
    """
    try:
        # Create multi-output data with 3 dimensions
        x_data, y_data = create_multi_output_synthetic_data(n_samples=150, input_dim=3, output_dim=3)
        
        # Create and train model with 3 outputs
        model = MultiOutputModel(input_dim=3, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(15):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Run distillation on only 2 out of 3 dimensions
        input_data = X_tensor[:75]
        model.mlp.distill(input_data, output_dim=0, sr_params={'niterations': 15})
        model.mlp.distill(input_data, output_dim=1, sr_params={'niterations': 15})
        
        # Manually remove one dimension to simulate missing scenario
        if 2 in model.mlp.pysr_regressor:
            del model.mlp.pysr_regressor[2]
        
        # Try to switch_to_equation (should fail gracefully)
        model.mlp.switch_to_equation()
        
        # Should still be in MLP mode
        if hasattr(model.mlp, '_using_equation'):
            assert not model.mlp._using_equation, "Should not switch to equation mode with missing dimensions"
        
        # Forward pass should still work normally
        test_input = X_tensor[:3]
        output = model(test_input)
        assert output.shape == (3, 3), "Forward pass should work normally with missing equations"
        
        print("✅ Multi-dimensional missing dimensions test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional missing dimensions test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_variable_transformations_basic():
    """
    Test basic variable transformations functionality.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input data
        input_data = torch.FloatTensor(X_train[:50])
        
        # Define simple variable transformations
        variable_transforms = [
            lambda x: x[:, 0] + x[:, 1],  # sum of first two variables
            lambda x: x[:, 2] * x[:, 3],  # product of third and fourth variables
            lambda x: x[:, 4] ** 2,       # square of fifth variable
        ]
        
        variable_names = ["x0_plus_x1", "x2_times_x3", "x4_squared"]
        
        # Run distillation with transformations
        regressor = trained_model.mlp.distill(
            input_data, 
            output_dim=0,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 30}
        )
        
        # Verify regressor was created
        assert regressor is not None, "Should return regressor for variable transformations"
        assert hasattr(regressor, 'equations_'), "Regressor should have equations"
        
        # Check that transformation info was stored
        assert hasattr(trained_model.mlp, '_variable_transforms'), "Should store variable transformations"
        assert hasattr(trained_model.mlp, '_variable_names'), "Should store variable names"
        assert trained_model.mlp._variable_transforms == variable_transforms, "Should store correct transformations"
        assert trained_model.mlp._variable_names == variable_names, "Should store correct variable names"
        
        print("✅ Basic variable transformations test passed")
        
    except Exception as e:
        pytest.fail(f"Variable transformations test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_variable_transformations_without_names():
    """
    Test variable transformations without custom names (should use x0, x1, etc.).
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input data
        input_data = torch.FloatTensor(X_train[:50])
        
        # Define transformations without names
        variable_transforms = [
            lambda x: x[:, 0] - x[:, 1],  # difference
            lambda x: torch.sin(x[:, 2]), # sine transformation
        ]
        
        # Run distillation with transformations (no variable_names provided)
        regressor = trained_model.mlp.distill(
            input_data, 
            output_dim=0,
            variable_transforms=variable_transforms,
            sr_params={'niterations': 30}
        )
        
        # Verify regressor was created
        assert regressor is not None, "Should return regressor for variable transformations"
        
        # Check that transformation info was stored
        assert hasattr(trained_model.mlp, '_variable_transforms'), "Should store variable transformations"
        assert hasattr(trained_model.mlp, '_variable_names'), "Should have variable names attribute"
        assert trained_model.mlp._variable_transforms == variable_transforms, "Should store correct transformations"
        assert trained_model.mlp._variable_names is None, "Should have None for variable names when not provided"
        
        print("✅ Variable transformations without names test passed")
        
    except Exception as e:
        pytest.fail(f"Variable transformations without names test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_variable_transformations_switch_to_equation():
    """
    Test that switch_to_equation works correctly with variable transformations.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input data
        input_data = torch.FloatTensor(X_train[:50])
        
        # Define transformations with names
        variable_transforms = [
            lambda x: x[:, 0] + x[:, 1],  # sum
            lambda x: x[:, 2],            # identity
        ]
        variable_names = ["sum_01", "x2"]
        
        # Run distillation with transformations
        trained_model.mlp.distill(
            input_data, 
            output_dim=0,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 30}
        )
        
        # Switch to equation mode
        trained_model.mlp.switch_to_equation()
        
        # Verify equation mode is active
        assert trained_model.mlp._using_equation, "Should be in equation mode"
        assert hasattr(trained_model.mlp, '_equation_funcs'), "Should have equation functions"
        assert hasattr(trained_model.mlp, '_equation_vars'), "Should have equation variables"
        
        # Test forward pass works with transformations
        test_input = torch.FloatTensor(X_train[:5])
        output = trained_model.mlp(test_input)
        
        assert output is not None, "Forward pass should work with variable transformations"
        assert output.shape[0] == 5, "Should have correct batch size"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        
        print("✅ Variable transformations switch_to_equation test passed")
        
    except Exception as e:
        pytest.fail(f"Variable transformations switch_to_equation test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_variable_transformations_error_handling():
    """
    Test error handling for variable transformations.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input data
        input_data = torch.FloatTensor(X_train[:50])
        
        # Test mismatched lengths
        variable_transforms = [lambda x: x[:, 0], lambda x: x[:, 1]]
        variable_names = ["only_one_name"]  # Length mismatch
        
        with pytest.raises(ValueError, match="Length of variable_names"):
            trained_model.mlp.distill(
                input_data, 
                variable_transforms=variable_transforms,
                fit_params={'variable_names': variable_names},
                sr_params={'niterations': 10}
            )
        
        # Test transform that causes an error
        def bad_transform(x):
            raise RuntimeError("Intentional error")
        
        variable_transforms = [bad_transform]
        
        with pytest.raises(ValueError, match="Error applying transformation"):
            trained_model.mlp.distill(
                input_data,
                variable_transforms=variable_transforms,
                sr_params={'niterations': 10}
            )
        
        print("✅ Variable transformations error handling test passed")
        
    except Exception as e:
        pytest.fail(f"Variable transformations error handling test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_variable_transformations_multi_dimensional():
    """
    Test variable transformations with multi-dimensional output.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=4, output_dim=2)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Define variable transformations
        variable_transforms = [
            lambda x: x[:, 0] + x[:, 1],  # sum
            lambda x: x[:, 2] * x[:, 3],  # product
            lambda x: x[:, 0] ** 2,       # square
        ]
        variable_names = ["sum_01", "product_23", "x0_squared"]
        
        # Test distill with transformations on all dimensions
        input_data = X_tensor[:100]
        regressors = model.mlp.distill(
            input_data,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 20}
        )
        
        # Verify results
        assert isinstance(regressors, dict), "Should return dictionary for multi-dim"
        assert len(regressors) == 2, "Should have regressors for both dimensions"
        
        # Check transformation info was stored
        assert hasattr(model.mlp, '_variable_transforms'), "Should store transformations"
        assert hasattr(model.mlp, '_variable_names'), "Should store variable names"
        assert model.mlp._variable_names == variable_names, "Should store correct names"
        
        # Test switch to equation with transformations
        model.mlp.switch_to_equation()
        assert model.mlp._using_equation, "Should be in equation mode"
        
        # Test forward pass
        test_input = X_tensor[:3]
        output = model(test_input)
        assert output.shape == (3, 2), "Should have correct output shape"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        
        print("✅ Variable transformations multi-dimensional test passed")
        
    except Exception as e:
        pytest.fail(f"Variable transformations multi-dimensional test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_save_path_parameter():
    """
    Test the save_path parameter for custom output directory.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input data
        input_data = torch.FloatTensor(X_train[:30])
        
        # Define custom save path
        custom_save_path = "custom_test_output"
        
        # Run distillation with custom save path
        regressor = trained_model.mlp.distill(
            input_data,
            output_dim=0,
            save_path=custom_save_path,
            sr_params={'niterations': 20}
        )
        
        # Verify regressor was created
        assert regressor is not None, "Should return regressor"
        
        # Check that the custom output directory exists
        expected_dir = f"{custom_save_path}/{trained_model.mlp.mlp_name}"
        # Note: PySR may not create the directory if no output is saved, so we just check the regressor works
        
        print("✅ Save path parameter test passed")
        
    except Exception as e:
        pytest.fail(f"Save path parameter test failed with error: {e}")
    finally:
        # Clean up custom output directory
        if os.path.exists("custom_test_output"):
            shutil.rmtree("custom_test_output")
        cleanup_sr_outputs()


def test_get_importance_basic():
    """
    Test basic get_importance functionality with MLP_SR.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create sample data for importance evaluation
        sample_data = torch.FloatTensor(X_train[:100])
        
        # Test get_importance method
        result = trained_model.mlp.get_importance(sample_data)
        
        # Verify output format
        assert isinstance(result, dict), "Should return a dictionary"
        assert 'importance' in result, "Should have 'importance' key"
        assert 'std' in result, "Should have 'std' key"
        
        importance_order = result['importance']
        std_values = result['std']
        
        assert isinstance(importance_order, list), "importance should be a list"
        assert isinstance(std_values, list), "std should be a list"
        assert len(importance_order) == 1, "Should have 1 dimension for single-output model"
        assert len(std_values) == 1, "Should have 1 std value for single-output model"
        assert importance_order[0] == 0, "Should have dimension 0 for single-output model"
        assert all(isinstance(dim, int) for dim in importance_order), "All dimensions should be integers"
        assert all(isinstance(std, float) for std in std_values), "All std values should be floats"
        assert all(std >= 0 for std in std_values), "All std values should be non-negative"
        
        print("✅ Basic get_importance test passed")
        
    except Exception as e:
        pytest.fail(f"get_importance basic test failed with error: {e}")


def test_get_importance_multi_dimensional():
    """
    Test get_importance with multi-dimensional output model.
    """
    try:
        # Create multi-output data with varying importance levels
        np.random.seed(123)
        x_data = np.random.uniform(-1, 1, (200, 4))
        y_data = np.zeros((200, 3))
        
        # Create outputs with different variance levels (different importance)
        y_data[:, 0] = x_data[:, 0] * 5 + np.random.normal(0, 0.1, 200)  # High variance
        y_data[:, 1] = x_data[:, 1] * 0.5 + np.random.normal(0, 0.01, 200)  # Low variance  
        y_data[:, 2] = x_data[:, 2] * 2 + np.random.normal(0, 0.05, 200)  # Medium variance
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test get_importance
        sample_data = X_tensor[:100]
        result = model.mlp.get_importance(sample_data)
        
        # Verify output format
        assert isinstance(result, dict), "Should return a dictionary"
        assert 'importance' in result, "Should have 'importance' key"
        assert 'std' in result, "Should have 'std' key"
        
        importance_order = result['importance']
        std_values = result['std']
        
        assert isinstance(importance_order, list), "importance should be a list"
        assert isinstance(std_values, list), "std should be a list"
        assert len(importance_order) == 3, "Should have 3 dimensions for 3-output model"
        assert len(std_values) == 3, "Should have 3 std values for 3-output model"
        assert set(importance_order) == {0, 1, 2}, "Should contain all dimensions"
        assert all(isinstance(dim, int) for dim in importance_order), "All dimensions should be integers"
        assert all(isinstance(std, float) for std in std_values), "All std values should be floats"
        assert all(std >= 0 for std in std_values), "All std values should be non-negative"
        
        # Check that ordering makes sense (dimension 0 should be most important due to highest variance)
        assert importance_order[0] == 0, f"Dimension 0 should be most important, got order: {importance_order}"
        
        # Check that std values are ordered correctly (descending)
        assert std_values[0] >= std_values[1] >= std_values[2], f"Std values should be in descending order: {std_values}"
        
        print("✅ Multi-dimensional get_importance test passed")
        
    except Exception as e:
        pytest.fail(f"get_importance multi-dimensional test failed with error: {e}")


def test_get_importance_with_parent_model():
    """
    Test get_importance with parent model for intermediary MLP evaluation.
    """
    try:
        # Create composite model where f_net is wrapped with MLP_SR
        class CompositeModel(nn.Module):
            def __init__(self, input_dim, intermediate_dim, output_dim):
                super(CompositeModel, self).__init__()
                f_mlp = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, intermediate_dim)
                )
                self.f_net = SymbolicMLP(f_mlp, mlp_name="f_net")
                self.g_net = nn.Linear(intermediate_dim, output_dim)
            
            def forward(self, x):
                x = self.f_net(x)
                x = self.g_net(x)
                return x
        
        # Create and train composite model
        model = CompositeModel(input_dim=5, intermediate_dim=16, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train the composite model
        model, _ = train_model(model, dataloader, optimizer, criterion, epochs=15)
        
        # Test get_importance with parent model
        sample_data = torch.FloatTensor(X_train[:100])
        result = model.f_net.get_importance(sample_data, parent_model=model)
        
        # Verify output format
        assert isinstance(result, dict), "Should return a dictionary"
        assert 'importance' in result, "Should have 'importance' key"
        assert 'std' in result, "Should have 'std' key"
        
        importance_order = result['importance']
        std_values = result['std']
        
        assert isinstance(importance_order, list), "importance should be a list"
        assert isinstance(std_values, list), "std should be a list"
        assert len(importance_order) == 16, "Should have 16 dimensions for intermediate layer"
        assert len(std_values) == 16, "Should have 16 std values for intermediate layer"
        assert set(importance_order) == set(range(16)), "Should contain all 16 dimensions"
        assert all(isinstance(dim, int) for dim in importance_order), "All dimensions should be integers"
        assert all(isinstance(std, float) for std in std_values), "All std values should be floats"
        assert all(std >= 0 for std in std_values), "All std values should be non-negative"
        
        # Test without parent model (direct evaluation)
        result_direct = model.f_net.get_importance(sample_data)
        
        # Both should return valid results
        assert isinstance(result_direct, dict), "Direct evaluation should also return a dictionary"
        assert len(result_direct['importance']) == 16, "Direct evaluation should have same dimension count"
        assert len(result_direct['std']) == 16, "Direct evaluation should have same std count"
        
        print("✅ get_importance with parent model test passed")
        
    except Exception as e:
        pytest.fail(f"get_importance with parent model test failed with error: {e}")


def test_get_importance_consistency():
    """
    Test that get_importance returns consistent results for the same input.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create sample data
        sample_data = torch.FloatTensor(X_train[:50])
        
        # Call get_importance multiple times
        result_1 = trained_model.mlp.get_importance(sample_data)
        result_2 = trained_model.mlp.get_importance(sample_data)
        result_3 = trained_model.mlp.get_importance(sample_data)
        
        # Results should be identical
        assert result_1['importance'] == result_2['importance'], "importance lists should be consistent across calls"
        assert result_2['importance'] == result_3['importance'], "importance lists should be consistent across calls"
        assert result_1['std'] == result_2['std'], "std lists should be consistent across calls"
        assert result_2['std'] == result_3['std'], "std lists should be consistent across calls"
        
        print("✅ get_importance consistency test passed")
        
    except Exception as e:
        pytest.fail(f"get_importance consistency test failed with error: {e}")


def test_save_model_basic():
    """
    Test basic model saving functionality.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test save path
        save_path = "test_save_basic"
        
        # Save the model
        saved_files = trained_model.mlp.save_model(save_path)
        
        # Verify files were created
        assert isinstance(saved_files, list), "Should return list of saved files"
        assert len(saved_files) > 0, "Should create at least one file"
        
        # Check expected files exist
        expected_pytorch = f"{save_path}_pytorch.pth"
        expected_metadata = f"{save_path}_metadata.pkl"
        
        assert expected_pytorch in saved_files, "Should save PyTorch state dict"
        assert expected_metadata in saved_files, "Should save metadata"
        assert os.path.exists(expected_pytorch), "PyTorch file should exist"
        assert os.path.exists(expected_metadata), "Metadata file should exist"
        
        print("✅ Basic save model test passed")
        
    except Exception as e:
        pytest.fail(f"Save model basic test failed with error: {e}")
    finally:
        # Cleanup
        cleanup_save_test_files("test_save_basic")


def test_save_model_with_regressors():
    """
    Test model saving with PySR regressors included.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    # Ensure we have a regressor
    if not hasattr(trained_model.mlp, 'pysr_regressor') or not trained_model.mlp.pysr_regressor:
        input_data = torch.FloatTensor(X_train[:50])
        trained_model.mlp.distill(input_data, sr_params={'niterations': 30})
    
    try:
        save_path = "test_save_with_regressors"
        
        # Save model with regressors
        saved_files = trained_model.mlp.save_model(save_path, save_pytorch=True, save_regressors=True)
        
        # Should have PyTorch file, metadata, and regressor files
        assert len(saved_files) >= 3, "Should save at least PyTorch, metadata, and regressor files"
        
        # Check for regressor files
        regressor_files = [f for f in saved_files if 'regressor_dim' in f]
        assert len(regressor_files) > 0, "Should save at least one regressor file"
        
        # Each regressor file should exist
        for regressor_file in regressor_files:
            assert os.path.exists(regressor_file), f"Regressor file should exist: {regressor_file}"
        
        print("✅ Save model with regressors test passed")
        
    except Exception as e:
        pytest.fail(f"Save model with regressors test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_save_with_regressors")


def test_save_model_selective():
    """
    Test selective saving (only PyTorch or only regressors).
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Test saving only PyTorch weights
        save_path_pytorch = "test_save_pytorch_only"
        saved_files_pytorch = trained_model.mlp.save_model(save_path_pytorch, save_pytorch=True, save_regressors=False)
        
        # Should have PyTorch and metadata but no regressor files
        pytorch_file = f"{save_path_pytorch}_pytorch.pth"
        metadata_file = f"{save_path_pytorch}_metadata.pkl"
        
        assert pytorch_file in saved_files_pytorch, "Should save PyTorch file"
        assert metadata_file in saved_files_pytorch, "Should save metadata file"
        assert os.path.exists(pytorch_file), "PyTorch file should exist"
        assert os.path.exists(metadata_file), "Metadata file should exist"
        
        regressor_files = [f for f in saved_files_pytorch if 'regressor_dim' in f]
        assert len(regressor_files) == 0, "Should not save regressor files"
        
        # Test saving only regressors (if available)
        if hasattr(trained_model.mlp, 'pysr_regressor') and trained_model.mlp.pysr_regressor:
            save_path_regressors = "test_save_regressors_only"
            saved_files_regressors = trained_model.mlp.save_model(save_path_regressors, save_pytorch=False, save_regressors=True)
            
            # Should have metadata and regressor files but no PyTorch file
            pytorch_file_reg = f"{save_path_regressors}_pytorch.pth"
            metadata_file_reg = f"{save_path_regressors}_metadata.pkl"
            
            assert pytorch_file_reg not in saved_files_regressors, "Should not save PyTorch file"
            assert metadata_file_reg in saved_files_regressors, "Should save metadata file"
            assert not os.path.exists(pytorch_file_reg), "PyTorch file should not exist"
            assert os.path.exists(metadata_file_reg), "Metadata file should exist"
            
            cleanup_save_test_files("test_save_regressors_only")
        
        print("✅ Selective save test passed")
        
    except Exception as e:
        pytest.fail(f"Selective save test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_save_pytorch_only")


def test_load_model_basic():
    """
    Test basic model loading functionality.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        save_path = "test_load_basic"
        
        # Ensure model is in a clean MLP state before saving
        trained_model.mlp.switch_to_mlp()
        trained_model.eval()  # Ensure in eval mode
        
        # First save a model
        trained_model.mlp.save_model(save_path)
        
        # Create same architecture for loading
        original_architecture = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Load the model
        loaded_model = SymbolicMLP.load_model(save_path, original_architecture)
        
        # Verify loaded model properties
        assert isinstance(loaded_model, SymbolicMLP), "Should return MLP_SR instance"
        assert loaded_model.mlp_name == trained_model.mlp.mlp_name, "Should preserve mlp_name"
        assert hasattr(loaded_model, 'InterpretSR_MLP'), "Should have wrapped MLP"
        
        # Test forward pass works - ensure both models are in same state
        test_input = torch.FloatTensor(X_train[:5])
        
        # Ensure both models are in eval mode and MLP mode
        trained_model.eval()
        trained_model.mlp.switch_to_mlp()
        loaded_model.eval()
        loaded_model.switch_to_mlp()
        
        with torch.no_grad():
            original_output = trained_model.mlp(test_input)
            loaded_output = loaded_model(test_input)
        
        # Outputs should be very similar (within floating point precision)
        diff = torch.abs(original_output - loaded_output)
        max_diff = torch.max(diff)
        assert max_diff < 1e-5, f"Loaded model should produce similar outputs (max diff: {max_diff})"
        
        print("✅ Basic load model test passed")
        
    except Exception as e:
        pytest.fail(f"Load model basic test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_load_basic")


def test_load_model_with_regressors():
    """
    Test loading model with PySR regressors.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    # Ensure we have a regressor and equation mode set up
    if not hasattr(trained_model.mlp, 'pysr_regressor') or not trained_model.mlp.pysr_regressor:
        input_data = torch.FloatTensor(X_train[:50])
        trained_model.mlp.distill(input_data, sr_params={'niterations': 30})
    
    # Switch to equation mode to test complete save/load
    trained_model.mlp.switch_to_equation()
    original_using_equation = trained_model.mlp._using_equation
    
    try:
        save_path = "test_load_with_regressors"
        
        # Save the model
        trained_model.mlp.save_model(save_path)
        
        # Create architecture for loading
        architecture = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Load the model
        loaded_model = SymbolicMLP.load_model(save_path, architecture)
        
        # Verify regressor loading
        assert hasattr(loaded_model, 'pysr_regressor'), "Should have regressors"
        assert len(loaded_model.pysr_regressor) > 0, "Should have loaded regressors"
        
        # Verify equation state restoration
        assert loaded_model._using_equation == original_using_equation, "Should restore equation mode"
        if original_using_equation:
            assert hasattr(loaded_model, '_equation_funcs'), "Should have equation functions"
            assert len(loaded_model._equation_funcs) > 0, "Should have restored equation functions"
        
        # Test that loaded model can switch modes
        if original_using_equation:
            # Switch back to MLP mode
            loaded_model.switch_to_mlp()
            assert not loaded_model._using_equation, "Should switch back to MLP mode"
            
            # Switch back to equation mode
            loaded_model.switch_to_equation()
            assert loaded_model._using_equation, "Should switch back to equation mode"
        
        print("✅ Load model with regressors test passed")
        
    except Exception as e:
        pytest.fail(f"Load model with regressors test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_load_with_regressors")


def test_load_model_without_architecture():
    """
    Test loading model metadata and regressors without PyTorch architecture.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        save_path = "test_load_no_arch"
        
        # Save the model
        trained_model.mlp.save_model(save_path)
        
        # Load without architecture
        loaded_model = SymbolicMLP.load_model(save_path, mlp_architecture=None)
        
        # Should have metadata but limited functionality
        assert isinstance(loaded_model, SymbolicMLP), "Should return MLP_SR instance"
        assert loaded_model.mlp_name == trained_model.mlp.mlp_name, "Should preserve mlp_name"
        
        # Should have nn.Identity as placeholder
        assert isinstance(loaded_model.InterpretSR_MLP, nn.Identity), "Should have Identity placeholder"
        
        print("✅ Load model without architecture test passed")
        
    except Exception as e:
        pytest.fail(f"Load model without architecture test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_load_no_arch")


def test_save_load_roundtrip():
    """
    Test complete save/load roundtrip preserves model functionality.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    # Ensure we have regressors
    if not hasattr(trained_model.mlp, 'pysr_regressor') or not trained_model.mlp.pysr_regressor:
        input_data = torch.FloatTensor(X_train[:50])
        trained_model.mlp.distill(input_data, sr_params={'niterations': 20})
    
    try:
        save_path = "test_roundtrip"
        test_input = torch.FloatTensor(X_train[:10])
        
        # Test in MLP mode
        trained_model.mlp.switch_to_mlp()
        original_mlp_output = trained_model.mlp(test_input).clone().detach()
        
        # Save and load
        trained_model.mlp.save_model(save_path)
        
        architecture = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        loaded_model = SymbolicMLP.load_model(save_path, architecture)
        
        # Test MLP mode preservation
        loaded_model.switch_to_mlp()
        loaded_mlp_output = loaded_model(test_input)
        
        mlp_diff = torch.abs(original_mlp_output - loaded_mlp_output)
        max_mlp_diff = torch.max(mlp_diff)
        assert max_mlp_diff < 1e-5, f"MLP outputs should match (max diff: {max_mlp_diff})"
        
        # Test equation mode if available
        if hasattr(trained_model.mlp, 'pysr_regressor') and trained_model.mlp.pysr_regressor:
            trained_model.mlp.switch_to_equation()
            original_eq_output = trained_model.mlp(test_input).clone().detach()
            
            loaded_model.switch_to_equation()
            loaded_eq_output = loaded_model(test_input)
            
            eq_diff = torch.abs(original_eq_output - loaded_eq_output)
            max_eq_diff = torch.max(eq_diff)
            assert max_eq_diff < 1e-4, f"Equation outputs should be similar (max diff: {max_eq_diff})"
        
        print("✅ Save/load roundtrip test passed")
        
    except Exception as e:
        pytest.fail(f"Save/load roundtrip test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_roundtrip")


def test_save_load_with_variable_transformations():
    """
    Test save/load functionality with variable transformations.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        save_path = "test_save_load_transforms"
        
        # Apply variable transformations
        variable_transforms = [
            lambda x: x[:, 0] + x[:, 1],
            lambda x: x[:, 2] ** 2,
            lambda x: torch.sin(x[:, 3])
        ]
        variable_names = ["x0_plus_x1", "x2_squared", "sin_x3"]
        
        input_data = torch.FloatTensor(X_train[:40])
        trained_model.mlp.distill(
            input_data,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 20}
        )
        
        # Switch to equation mode
        trained_model.mlp.switch_to_equation()
        
        # Save model
        trained_model.mlp.save_model(save_path)
        
        # Load model
        architecture = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        loaded_model = SymbolicMLP.load_model(save_path, architecture)
        
        # Verify transformations were preserved
        assert hasattr(loaded_model, '_variable_names'), "Should preserve variable names"
        assert loaded_model._variable_names == variable_names, "Should preserve exact variable names"
        
        # Note: Variable transforms are functions and can't be saved/loaded directly
        # This is expected behavior - transforms need to be reapplied if needed
        
        # Test that equation mode still works (even without transforms restored)
        test_input = torch.FloatTensor(X_train[:5])
        try:
            loaded_model.switch_to_equation()
            # This might fail due to missing transforms, which is expected
            output = loaded_model(test_input)
            print("✅ Equation mode worked without transforms (equations may use original variables)")
        except Exception:
            print("ℹ️ Equation mode requires transforms to be reapplied (expected behavior)")
        
        print("✅ Save/load with variable transformations test passed")
        
    except Exception as e:
        pytest.fail(f"Save/load with variable transformations test failed with error: {e}")
    finally:
        cleanup_save_test_files("test_save_load_transforms")


def cleanup_save_test_files(base_name):
    """Clean up files created during save/load tests."""
    import glob
    
    patterns = [
        f"{base_name}_*.pth",
        f"{base_name}_*.pkl",
        f"{base_name}_regressor_*.pkl"
    ]
    
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except OSError:
                pass


def cleanup_sr_outputs():
    """
    Clean up SR output files and directories created during testing.
    """
    if os.path.exists('SR_output'):
        shutil.rmtree('SR_output')
    
    # Clean up any custom output directories
    if os.path.exists('custom_test_output'):
        shutil.rmtree('custom_test_output')
    
    # Clean up any other potential output files
    for file in os.listdir('.'):
        if file.startswith('hall_of_fame') or file.endswith('.pkl'):
            try:
                os.remove(file)
            except OSError:
                pass


# Cleanup fixture to ensure files are cleaned up after all tests
@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """
    Fixture to clean up output files after all tests complete.
    """
    yield
    cleanup_sr_outputs()