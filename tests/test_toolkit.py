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
import math

# Add the src directory to Python path for absolute imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from symtorch.mlp_sr import MLP_SR
from symtorch.toolkit import Pruning_MLP


class SimpleCompositeModel(nn.Module):
    """
    Simple composite model for testing Pruning_MLP functionality.
    Has f_net that outputs intermediate features and g_net that maps to final output.
    """
    def __init__(self, input_dim, output_dim, output_dim_f=32, hidden_dim=64):
        super(SimpleCompositeModel, self).__init__()
        self.f_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim_f)
        )
        self.g_net = nn.Linear(output_dim_f, output_dim)
    
    def forward(self, x):
        x = self.f_net(x)
        x = self.g_net(x)
        return x


def train_model(model, dataloader, opt, criterion, epochs=20):
    """Train a model for the specified number of epochs."""
    loss_tracker = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        loss_tracker.append(epoch_loss)
    return model, loss_tracker


# Global test data setup for Pruning_MLP tests
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data with known relationship
x = np.array([np.random.uniform(-1, 1, 500) for _ in range(5)]).T
y = x[:, 0]**2 + 2*np.sin(x[:, 1]) - x[:, 2] + 0.5
noise = np.array([np.random.normal(0, 0.01*np.std(y)) for _ in range(len(y))])
y = y + noise

X_train, X_test, y_train, y_test = train_test_split(x, y.reshape(-1,1), test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)


def test_pruning_mlp_initialization():
    """Test that Pruning_MLP initializes correctly."""
    mlp = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 32)
    )
    
    pruning_mlp = Pruning_MLP(mlp, initial_dim=32, target_dim=8, mlp_name="test_mlp")
    
    # Check basic attributes
    assert pruning_mlp.initial_dim == 32
    assert pruning_mlp.current_dim == 32
    assert pruning_mlp.target_dim == 8
    assert pruning_mlp.mlp_name == "test_mlp"
    assert pruning_mlp.pruning_mask.sum().item() == 32  # All dimensions initially active
    assert hasattr(pruning_mlp, 'InterpretSR_MLP')  # Inherited from MLP_SR
    assert hasattr(pruning_mlp, 'interpret')  # Inherited from MLP_SR


def test_pruning_schedule_cosine():
    """Test cosine pruning schedule generation."""
    mlp = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 16))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=16, target_dim=4, mlp_name="test_schedule")
    
    pruning_mlp.set_schedule(total_epochs=100, decay_rate='cosine', end_epoch_frac=0.5)
    
    assert pruning_mlp.pruning_schedule is not None
    assert len(pruning_mlp.pruning_schedule) == 100
    
    # Check that dimensions decrease over time during pruning phase
    # Note: cosine decay starts at initial_dim and decreases to target_dim
    prune_end_epoch = int(0.5 * 100)  # epoch 50
    
    # At epoch 0, should have some dimensions pruned due to cosine schedule
    # Check that by the end of pruning phase we reach target
    assert pruning_mlp.pruning_schedule[prune_end_epoch] == 4  # Should reach target_dim at end_epoch
    assert pruning_mlp.pruning_schedule[99] == 4  # Should stay at target_dim
    
    # Check monotonic decrease during pruning phase (cosine schedule)
    early_dims = pruning_mlp.pruning_schedule[5] 
    mid_dims = pruning_mlp.pruning_schedule[25]
    late_dims = pruning_mlp.pruning_schedule[45]
    assert early_dims >= mid_dims >= late_dims  # Should decrease monotonically


def test_pruning_schedule_linear():
    """Test linear pruning schedule generation."""
    mlp = nn.Sequential(nn.Linear(5, 20), nn.ReLU(), nn.Linear(20, 20))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=20, target_dim=5, mlp_name="test_linear")
    
    pruning_mlp.set_schedule(total_epochs=80, decay_rate='linear', end_epoch_frac=0.6)
    
    assert pruning_mlp.pruning_schedule is not None
    prune_end_epoch = int(0.6 * 80)  # epoch 48
    
    # Linear schedule should have monotonic decrease
    dims_to_check = [0, 10, 20, 30, 40, 48, 60, 79]
    for i in range(len(dims_to_check) - 1):
        epoch1, epoch2 = dims_to_check[i], dims_to_check[i + 1]
        if epoch1 < prune_end_epoch and epoch2 < prune_end_epoch:
            assert pruning_mlp.pruning_schedule[epoch2] <= pruning_mlp.pruning_schedule[epoch1]
        elif epoch1 >= prune_end_epoch:
            assert pruning_mlp.pruning_schedule[epoch1] == 5


def test_pruning_schedule_exponential():
    """Test exponential pruning schedule generation."""
    mlp = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 24))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=24, target_dim=6, mlp_name="test_exp")
    
    pruning_mlp.set_schedule(total_epochs=60, decay_rate='exp', end_epoch_frac=0.7)
    
    assert pruning_mlp.pruning_schedule is not None
    prune_end_epoch = int(0.7 * 60)  # epoch 42
    
    # Check that schedule reaches target dimensions
    assert pruning_mlp.pruning_schedule[prune_end_epoch] == 6
    assert pruning_mlp.pruning_schedule[59] == 6  # Should stay at target_dim after pruning phase
    
    # Check that dimensions decrease over time during pruning phase
    # Exponential schedule should show decreasing trend
    dims_at_quarter = pruning_mlp.pruning_schedule[prune_end_epoch // 4]
    dims_at_half = pruning_mlp.pruning_schedule[prune_end_epoch // 2]
    dims_at_three_quarters = pruning_mlp.pruning_schedule[3 * prune_end_epoch // 4]
    
    assert dims_at_quarter >= dims_at_half >= dims_at_three_quarters  # Should decrease monotonically


def test_pruning_with_sample_data():
    """Test that pruning actually reduces dimensions based on importance."""
    # Create model with pruning
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=16)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=16, target_dim=4, mlp_name="f_net")
    model.f_net.set_schedule(total_epochs=50, decay_rate='linear', end_epoch_frac=0.8)
    
    # Train the model first to give some structure to the features
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train for a few epochs to develop feature importance
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=10)
    
    # Test pruning at different epochs
    sample_data = X_train_tensor[:50]  # Use sample for importance evaluation
    
    # Initially should have all 16 dimensions
    assert model.f_net.current_dim == 16
    assert model.f_net.pruning_mask.sum().item() == 16
    
    # Prune at epoch 20 (should be partway through pruning schedule)
    model.f_net.prune(20, sample_data, parent_model=model)
    
    # Should have fewer dimensions now
    assert model.f_net.current_dim < 16
    assert model.f_net.current_dim >= 4
    assert model.f_net.pruning_mask.sum().item() == model.f_net.current_dim
    
    # Prune at epoch 40 (should be at target dimensions)
    model.f_net.prune(40, sample_data, parent_model=model)
    
    # Should be at target dimension
    assert model.f_net.current_dim == 4
    assert model.f_net.pruning_mask.sum().item() == 4


def test_get_active_dimensions():
    """Test that get_active_dimensions returns correct indices."""
    mlp = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 10))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=10, target_dim=3, mlp_name="test_active")
    
    # Initially all dimensions should be active
    active_dims = pruning_mlp.get_active_dimensions()
    assert len(active_dims) == 10
    assert active_dims == list(range(10))
    
    # Manually set a pruning mask
    pruning_mlp.pruning_mask = torch.tensor([True, False, True, False, False, True, False, False, False, False])
    pruning_mlp.current_dim = 3
    
    active_dims = pruning_mlp.get_active_dimensions()
    assert len(active_dims) == 3
    assert active_dims == [0, 2, 5]


def test_pruned_forward_pass():
    """Test that forward pass works correctly with pruning mask applied."""
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=8)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=8, target_dim=3, mlp_name="f_net")
    
    # Set a specific pruning mask
    model.f_net.pruning_mask = torch.tensor([True, False, True, False, False, True, False, False])
    model.f_net.current_dim = 3
    
    # Test forward pass
    test_input = X_train_tensor[:5]
    output = model.f_net(test_input)
    
    # Output should have same batch size and full dimensionality, but inactive dims should be zero
    assert output.shape == (5, 8)
    
    # Check that inactive dimensions are indeed zero
    inactive_dims = torch.where(~model.f_net.pruning_mask)[0]
    for dim in inactive_dims:
        assert torch.allclose(output[:, dim], torch.zeros(5)), f"Dimension {dim} should be zero"
    
    # Check that active dimensions are non-zero (at least some of them)
    active_dims = torch.where(model.f_net.pruning_mask)[0]
    active_outputs = output[:, active_dims]
    assert not torch.allclose(active_outputs, torch.zeros_like(active_outputs)), "Active dimensions should have non-zero outputs"


def test_pruned_interpret():
    """Test that interpret works only on active dimensions."""
    # Create and train a model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=12)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=12, target_dim=4, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=5)
    
    # Set pruning schedule and prune to active dimensions
    model.f_net.set_schedule(total_epochs=20, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:50]
    model.f_net.prune(10, sample_data, parent_model=model)  # Should prune to target_dim
    
    active_dims = model.f_net.get_active_dimensions()
    assert len(active_dims) == 4
    
    # Run interpret - should only work on active dimensions
    input_data = X_train_tensor[:100]
    regressors = model.f_net.interpret(input_data, parent_model=model, sr_params={'niterations': 20})
    
    # Should return dictionary with entries only for active dimensions
    assert isinstance(regressors, dict)
    assert len(regressors) == 4  # Should have regressors for 4 active dimensions
    assert set(regressors.keys()) == set(active_dims)
    
    # Each regressor should be valid
    for dim_idx, regressor in regressors.items():
        assert regressor is not None
        assert hasattr(regressor, 'equations_')
        assert hasattr(regressor, 'get_best')
        assert dim_idx in active_dims


def test_pruned_switch_to_equation():
    """Test that switch_to_equation works correctly with pruned dimensions."""
    # Create and train a simple model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=8)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=8, target_dim=3, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=3)
    
    # Set up pruning and prune
    model.f_net.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:30]
    model.f_net.prune(5, sample_data, parent_model=model)
    
    # Run interpret
    input_data = X_train_tensor[:50]
    regressors = model.f_net.interpret(input_data, parent_model=model, sr_params={'niterations': 15})
    
    active_dims = model.f_net.get_active_dimensions()
    assert len(regressors) == len(active_dims)
    
    # Test switch to equation
    model.f_net.switch_to_equation()
    
    # Should be in equation mode
    assert hasattr(model.f_net, '_using_equation')
    assert model.f_net._using_equation
    assert hasattr(model.f_net, '_equation_funcs')
    assert hasattr(model.f_net, '_equation_vars')
    
    # Should have equations only for active dimensions
    assert len(model.f_net._equation_funcs) == len(active_dims)
    assert set(model.f_net._equation_funcs.keys()) == set(active_dims)
    
    # Test forward pass in equation mode
    test_input = X_train_tensor[:3]
    output = model.f_net(test_input)
    
    # Should have correct shape with inactive dimensions as zeros
    assert output.shape == (3, 8)
    
    # Inactive dimensions should be zero
    inactive_mask = ~model.f_net.pruning_mask
    inactive_outputs = output[:, inactive_mask]
    assert torch.allclose(inactive_outputs, torch.zeros_like(inactive_outputs))
    
    # Active dimensions should have non-zero values (equations evaluated)
    active_mask = model.f_net.pruning_mask
    active_outputs = output[:, active_mask]
    assert not torch.allclose(active_outputs, torch.zeros_like(active_outputs))


def test_pruned_forward_equation_vs_mlp_consistency():
    """Test that switching between equation and MLP modes maintains active/inactive dimension structure."""
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=6)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=6, target_dim=2, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=3)
    
    # Prune to 2 dimensions
    model.f_net.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=0.5)
    model.f_net.prune(5, X_train_tensor[:30], parent_model=model)
    
    # Get forward pass in MLP mode
    test_input = X_train_tensor[:5]
    mlp_output = model.f_net(test_input).clone().detach()
    
    # Check that inactive dimensions are zero in MLP mode
    inactive_mask = ~model.f_net.pruning_mask
    assert torch.allclose(mlp_output[:, inactive_mask], torch.zeros(5, inactive_mask.sum()))
    
    # Run interpret and switch to equation
    model.f_net.interpret(X_train_tensor[:40], parent_model=model, sr_params={'niterations': 10})
    model.f_net.switch_to_equation()
    
    # Get forward pass in equation mode
    equation_output = model.f_net(test_input)
    
    # Inactive dimensions should still be zero in equation mode
    assert torch.allclose(equation_output[:, inactive_mask], torch.zeros(5, inactive_mask.sum()))
    
    # Switch back to MLP
    model.f_net.switch_to_mlp()
    mlp_output_2 = model.f_net(test_input)
    
    # Should match original MLP output
    assert torch.allclose(mlp_output, mlp_output_2, atol=1e-6)


def test_pruning_with_composite_model_training():
    """Test that pruning works during full training of a composite model."""
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=20)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=20, target_dim=5, mlp_name="f_net")
    model.f_net.set_schedule(total_epochs=30, decay_rate='cosine', end_epoch_frac=0.6)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop with pruning
    initial_dims = model.f_net.current_dim
    dims_history = []
    
    for epoch in range(30):
        # Training step
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Pruning step
        sample_data = X_train_tensor[:50]
        model.f_net.prune(epoch, sample_data, parent_model=model)
        dims_history.append(model.f_net.current_dim)
        
        if epoch % 10 == 9:
            print(f"Epoch {epoch+1}, Dims: {model.f_net.current_dim}, Loss: {epoch_loss:.4f}")
    
    # Check that pruning happened progressively
    assert dims_history[0] == initial_dims  # Should start at initial dimensions
    assert dims_history[-1] == 5  # Should end at target dimensions
    
    # Check that dimensions decreased over time (at least at some point)
    assert min(dims_history) < max(dims_history)
    
    # Final forward pass should work
    test_input = X_test_tensor[:10]
    output = model(test_input)
    assert output.shape == (10, 1)
    assert not torch.isnan(output).any()


def test_no_active_dimensions_edge_case():
    """Test edge case where no dimensions are active."""
    mlp = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 8))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=8, target_dim=2, mlp_name="test_empty")
    
    # Set mask to all False (no active dimensions)
    pruning_mlp.pruning_mask = torch.zeros(8, dtype=torch.bool)
    pruning_mlp.current_dim = 0
    
    # get_active_dimensions should return empty list
    active_dims = pruning_mlp.get_active_dimensions()
    assert active_dims == []
    
    # interpret should return empty dict
    input_data = X_train_tensor[:20]
    regressors = pruning_mlp.interpret(input_data, sr_params={'niterations': 5})
    assert regressors == {}
    
    # switch_to_equation should handle gracefully
    pruning_mlp.switch_to_equation()
    assert not hasattr(pruning_mlp, '_using_equation') or not pruning_mlp._using_equation
    
    # Forward pass should return all zeros
    test_input = X_train_tensor[:3]
    output = pruning_mlp(test_input)
    assert output.shape == (3, 8)
    assert torch.allclose(output, torch.zeros(3, 8))


def test_pruning_epoch_not_in_schedule():
    """Test that pruning does nothing when epoch is not in schedule."""
    mlp = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 10))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=10, target_dim=4, mlp_name="test_no_prune")
    pruning_mlp.set_schedule(total_epochs=20, decay_rate='linear', end_epoch_frac=0.5)
    
    # Current state
    initial_mask = pruning_mlp.pruning_mask.clone()
    initial_dim = pruning_mlp.current_dim
    
    # Try to prune at epoch 100 (not in schedule)
    sample_data = X_train_tensor[:20]
    pruning_mlp.prune(100, sample_data)
    
    # Should be unchanged
    assert torch.equal(pruning_mlp.pruning_mask, initial_mask)
    assert pruning_mlp.current_dim == initial_dim


def test_pruning_schedule_validation():
    """Test edge cases in pruning schedule generation."""
    mlp = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 16))
    pruning_mlp = Pruning_MLP(mlp, initial_dim=16, target_dim=16, mlp_name="test_validation")
    
    # Case: target_dim equals initial_dim (no pruning needed)
    pruning_mlp.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=0.5)
    assert all(dim == 16 for dim in pruning_mlp.pruning_schedule.values())
    
    # Case: end_epoch_frac = 1.0 (prune until the very end)
    # Note: Linear schedule goes from initial_dim to target_dim over prune_epochs
    pruning_mlp.target_dim = 4
    pruning_mlp.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=1.0)
    # Linear schedule with end_epoch_frac=1.0 means pruning happens over all 10 epochs
    # At epoch 9 (last pruning epoch), should be close to target_dim
    # Since linear progress at epoch 9: progress = 9/10 = 0.9
    # dims_pruned = ceil(12 * 0.9) = ceil(10.8) = 11
    # target_dims = max(16 - 11, 4) = max(5, 4) = 5
    assert pruning_mlp.pruning_schedule[9] == 5  # Should be 5 due to discretization


class CompositeModelWithMiddleMLP(nn.Module):
    """
    Composite model where MLP is in the middle - between encoder and decoder.
    This tests pruning functionality for MLPs that are not at the beginning.
    """
    def __init__(self, input_dim, output_dim, encoder_dim=16, middle_dim=24, decoder_dim=12):
        super(CompositeModelWithMiddleMLP, self).__init__()
        # Encoder: input -> encoder features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )
        
        # Middle MLP: encoder features -> middle features (this will be wrapped with Pruning_MLP)
        self.middle_mlp = nn.Sequential(
            nn.Linear(encoder_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, middle_dim)
        )
        
        # Decoder: middle features -> output
        self.decoder = nn.Sequential(
            nn.Linear(middle_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle_mlp(x)
        x = self.decoder(x)
        return x


def test_pruning_mlp_in_middle_of_model():
    """Test pruning functionality when MLP is in the middle of a composite model."""
    # Create model with MLP in the middle
    model = CompositeModelWithMiddleMLP(input_dim=5, output_dim=1, 
                                       encoder_dim=16, middle_dim=20, decoder_dim=12)
    
    # Wrap the middle MLP with Pruning_MLP
    model.middle_mlp = Pruning_MLP(model.middle_mlp, initial_dim=20, target_dim=6, 
                                  mlp_name="middle_mlp")
    
    # Set up pruning schedule
    model.middle_mlp.set_schedule(total_epochs=40, decay_rate='cosine', end_epoch_frac=0.6)
    
    # Train the model briefly to develop feature importance
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train for a few epochs
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=8)
    
    # Initially should have all 20 dimensions
    assert model.middle_mlp.current_dim == 20
    assert model.middle_mlp.pruning_mask.sum().item() == 20
    
    # Test pruning at various epochs
    sample_data = X_train_tensor[:50]
    
    # Prune at epoch 10 (should be in pruning phase)
    model.middle_mlp.prune(10, sample_data, parent_model=model)
    
    # Should have fewer dimensions now
    assert model.middle_mlp.current_dim < 20
    assert model.middle_mlp.current_dim >= 6
    assert model.middle_mlp.pruning_mask.sum().item() == model.middle_mlp.current_dim
    
    # Prune at epoch 24 (should be at target dimensions)
    model.middle_mlp.prune(24, sample_data, parent_model=model)
    
    # Should be at target dimension
    assert model.middle_mlp.current_dim == 6
    assert model.middle_mlp.pruning_mask.sum().item() == 6
    
    # Test that full model still works after pruning
    test_input = X_test_tensor[:10]
    output = model(test_input)
    assert output.shape == (10, 1)
    assert not torch.isnan(output).any()
    
    # Test that the middle MLP forward pass respects pruning
    middle_output = model.middle_mlp(model.encoder(test_input))
    assert middle_output.shape == (10, 20)  # Still full dimensionality
    
    # Check that inactive dimensions are zero
    inactive_mask = ~model.middle_mlp.pruning_mask
    assert torch.allclose(middle_output[:, inactive_mask], 
                         torch.zeros(10, inactive_mask.sum()))
    
    # Check that active dimensions are non-zero
    active_mask = model.middle_mlp.pruning_mask
    active_outputs = middle_output[:, active_mask]
    assert not torch.allclose(active_outputs, torch.zeros_like(active_outputs))


def test_middle_mlp_interpret_and_switch():
    """Test interpret and equation switching for middle MLP in composite model."""
    # Create model with middle MLP
    model = CompositeModelWithMiddleMLP(input_dim=5, output_dim=1, 
                                       encoder_dim=10, middle_dim=16, decoder_dim=8)
    
    model.middle_mlp = Pruning_MLP(model.middle_mlp, initial_dim=16, target_dim=4, 
                                  mlp_name="middle_mlp")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=5)
    
    # Prune to target dimensions
    model.middle_mlp.set_schedule(total_epochs=20, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:40]
    model.middle_mlp.prune(10, sample_data, parent_model=model)
    
    # Should have 4 active dimensions
    active_dims = model.middle_mlp.get_active_dimensions()
    assert len(active_dims) == 4
    
    # Run interpret on the middle MLP - should work with parent_model
    input_data = X_train_tensor[:60]
    regressors = model.middle_mlp.interpret(input_data, parent_model=model, sr_params={'niterations': 15})
    
    # Should have regressors for active dimensions only
    assert isinstance(regressors, dict)
    assert len(regressors) == 4
    assert set(regressors.keys()) == set(active_dims)
    
    # Each regressor should be valid
    for dim_idx, regressor in regressors.items():
        assert regressor is not None
        assert hasattr(regressor, 'equations_')
        assert dim_idx in active_dims
    
    # Test switch to equation mode
    model.middle_mlp.switch_to_equation()
    
    # Should be in equation mode
    assert hasattr(model.middle_mlp, '_using_equation')
    assert model.middle_mlp._using_equation
    assert len(model.middle_mlp._equation_funcs) == 4
    
    # Test forward pass in equation mode
    test_input = X_train_tensor[:5]
    
    # Get original MLP output for comparison
    model.middle_mlp.switch_to_mlp()
    mlp_output = model(test_input).clone().detach()
    
    # Switch back to equation and test
    model.middle_mlp.switch_to_equation()
    equation_output = model(test_input)
    
    # Both should have same shape and be finite
    assert mlp_output.shape == equation_output.shape == (5, 1)
    assert torch.isfinite(mlp_output).all()
    assert torch.isfinite(equation_output).all()
    
    # The middle MLP outputs should respect pruning in both modes
    model.middle_mlp.switch_to_mlp()
    middle_mlp_output = model.middle_mlp(model.encoder(test_input))
    
    model.middle_mlp.switch_to_equation()  
    middle_eq_output = model.middle_mlp(model.encoder(test_input))
    
    # Both should have inactive dimensions as zeros
    inactive_mask = ~model.middle_mlp.pruning_mask
    assert torch.allclose(middle_mlp_output[:, inactive_mask], 
                         torch.zeros(5, inactive_mask.sum()))
    assert torch.allclose(middle_eq_output[:, inactive_mask], 
                         torch.zeros(5, inactive_mask.sum()))


def test_pruning_variable_transformations_basic():
    """Test basic variable transformations functionality with Pruning_MLP."""
    # Create and train a model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=12)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=12, target_dim=4, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=5)
    
    # Prune to active dimensions
    model.f_net.set_schedule(total_epochs=20, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:50]
    model.f_net.prune(10, sample_data, parent_model=model)
    
    active_dims = model.f_net.get_active_dimensions()
    assert len(active_dims) == 4
    
    try:
        # Define variable transformations
        variable_transforms = [
            lambda x: x[:, 0] + x[:, 1],  # sum of first two variables
            lambda x: x[:, 2] * x[:, 3],  # product of third and fourth variables
            lambda x: x[:, 4] ** 2,       # square of fifth variable
        ]
        variable_names = ["x0_plus_x1", "x2_times_x3", "x4_squared"]
        
        # Run interpretation with transformations
        input_data = X_train_tensor[:80]
        regressors = model.f_net.interpret(
            input_data,
            parent_model=model,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 20}
        )
        
        # Should return dictionary with entries only for active dimensions
        assert isinstance(regressors, dict)
        assert len(regressors) == 4
        assert set(regressors.keys()) == set(active_dims)
        
        # Check that transformation info was stored
        assert hasattr(model.f_net, '_variable_transforms')
        assert hasattr(model.f_net, '_variable_names')
        assert model.f_net._variable_transforms == variable_transforms
        assert model.f_net._variable_names == variable_names
        
        # Each regressor should be valid
        for dim_idx, regressor in regressors.items():
            assert regressor is not None
            assert hasattr(regressor, 'equations_')
            assert hasattr(regressor, 'get_best')
            assert dim_idx in active_dims
        
        print("✅ Pruning variable transformations basic test passed")
        
    except Exception as e:
        pytest.fail(f"Pruning variable transformations test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_pruning_variable_transformations_switch_to_equation():
    """Test that switch_to_equation works with variable transformations in Pruning_MLP."""
    # Create and train a model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=8)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=8, target_dim=3, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=3)
    
    # Prune to active dimensions
    model.f_net.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:30]
    model.f_net.prune(5, sample_data, parent_model=model)
    
    active_dims = model.f_net.get_active_dimensions()
    assert len(active_dims) == 3
    
    try:
        # Define transformations with names
        variable_transforms = [
            lambda x: x[:, 0] + x[:, 1],  # sum
            lambda x: x[:, 2],            # identity
        ]
        variable_names = ["sum_01", "x2"]
        
        # Run interpretation with transformations
        input_data = X_train_tensor[:50]
        regressors = model.f_net.interpret(
            input_data,
            parent_model=model,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 15}
        )
        
        assert len(regressors) == 3
        
        # Switch to equation mode
        model.f_net.switch_to_equation()
        
        # Verify equation mode is active
        assert model.f_net._using_equation
        assert hasattr(model.f_net, '_equation_funcs')
        assert hasattr(model.f_net, '_equation_vars')
        
        # Should have equations only for active dimensions
        assert len(model.f_net._equation_funcs) == 3
        assert set(model.f_net._equation_funcs.keys()) == set(active_dims)
        
        # Test forward pass works with transformations
        test_input = X_train_tensor[:5]
        output = model.f_net(test_input)
        
        # Should have correct shape with inactive dimensions as zeros
        assert output.shape == (5, 8)
        
        # Inactive dimensions should be zero
        inactive_mask = ~model.f_net.pruning_mask
        assert torch.allclose(output[:, inactive_mask], torch.zeros(5, inactive_mask.sum()))
        
        # Active dimensions should have non-zero values
        active_mask = model.f_net.pruning_mask
        active_outputs = output[:, active_mask]
        assert not torch.allclose(active_outputs, torch.zeros_like(active_outputs))
        
        print("✅ Pruning variable transformations switch_to_equation test passed")
        
    except Exception as e:
        pytest.fail(f"Pruning variable transformations switch_to_equation test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_pruning_variable_transformations_specific_dimension():
    """Test variable transformations with specific output dimension in Pruning_MLP."""
    # Create and train a model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=10)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=10, target_dim=3, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=4)
    
    # Prune to active dimensions
    model.f_net.set_schedule(total_epochs=15, decay_rate='linear', end_epoch_frac=0.6)
    sample_data = X_train_tensor[:40]
    model.f_net.prune(9, sample_data, parent_model=model)
    
    active_dims = model.f_net.get_active_dimensions()
    assert len(active_dims) == 3
    
    try:
        # Define transformations
        variable_transforms = [
            lambda x: x[:, 0] - x[:, 1],  # difference
            lambda x: torch.sin(x[:, 2]), # sine transformation
            lambda x: x[:, 3] * x[:, 4],  # product
        ]
        variable_names = ["x0_minus_x1", "sin_x2", "x3_times_x4"]
        
        # Run interpretation with transformations on specific active dimension
        input_data = X_train_tensor[:60]
        target_dim = active_dims[1]  # Pick one active dimension
        
        regressor = model.f_net.interpret(
            input_data,
            parent_model=model,
            output_dim=target_dim,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 15}
        )
        
        # Should return single regressor (not a dictionary)
        assert not isinstance(regressor, dict)
        assert regressor is not None
        assert hasattr(regressor, 'equations_')
        assert hasattr(regressor, 'get_best')
        
        # Check that transformation info was stored
        assert hasattr(model.f_net, '_variable_transforms')
        assert hasattr(model.f_net, '_variable_names')
        assert model.f_net._variable_names == variable_names
        
        # Verify the regressor is stored correctly
        assert hasattr(model.f_net, 'pysr_regressor')
        assert target_dim in model.f_net.pysr_regressor
        assert model.f_net.pysr_regressor[target_dim] is regressor
        
        print("✅ Pruning variable transformations specific dimension test passed")
        
    except Exception as e:
        pytest.fail(f"Pruning variable transformations specific dimension test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_pruning_variable_transformations_error_handling():
    """Test error handling for variable transformations with Pruning_MLP."""
    # Create a simple model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=6)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=6, target_dim=2, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=2)
    
    try:
        # Create test input data
        input_data = X_train_tensor[:30]
        
        # Test mismatched lengths
        variable_transforms = [lambda x: x[:, 0], lambda x: x[:, 1]]
        variable_names = ["only_one_name"]  # Length mismatch
        
        with pytest.raises(ValueError, match="Length of variable_names"):
            model.f_net.interpret(
                input_data,
                parent_model=model,
                variable_transforms=variable_transforms,
                fit_params={'variable_names': variable_names},
                sr_params={'niterations': 10}
            )
        
        # Test transform that causes an error
        def bad_transform(x):
            raise RuntimeError("Intentional error")
        
        variable_transforms = [bad_transform]
        
        with pytest.raises(ValueError, match="Error applying transformation"):
            model.f_net.interpret(
                input_data,
                parent_model=model,
                variable_transforms=variable_transforms,
                sr_params={'niterations': 10}
            )
        
        print("✅ Pruning variable transformations error handling test passed")
        
    except Exception as e:
        pytest.fail(f"Pruning variable transformations error handling test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_pruning_save_path_parameter():
    """Test the save_path parameter for custom output directory with Pruning_MLP."""
    # Create and train a model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=8)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=8, target_dim=3, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=3)
    
    # Prune to active dimensions
    model.f_net.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:30]
    model.f_net.prune(5, sample_data, parent_model=model)
    
    try:
        # Define custom save path
        custom_save_path = "custom_pruning_output"
        
        # Run interpretation with custom save path
        input_data = X_train_tensor[:50]
        regressors = model.f_net.interpret(
            input_data,
            parent_model=model,
            save_path=custom_save_path,
            sr_params={'niterations': 15}
        )
        
        # Verify regressors were created
        assert isinstance(regressors, dict)
        assert len(regressors) > 0  # Should have regressors for active dimensions
        
        for regressor in regressors.values():
            assert regressor is not None
            assert hasattr(regressor, 'equations_')
        
        print("✅ Pruning save path parameter test passed")
        
    except Exception as e:
        pytest.fail(f"Pruning save path parameter test failed with error: {e}")
    finally:
        # Clean up custom output directory
        if os.path.exists("custom_pruning_output"):
            shutil.rmtree("custom_pruning_output")
        cleanup_sr_outputs()


def test_pruning_variable_transformations_inactive_dimension_request():
    """Test requesting symbolic regression on inactive dimension with transformations."""
    # Create and train a model
    model = SimpleCompositeModel(input_dim=5, output_dim=1, output_dim_f=8)
    model.f_net = Pruning_MLP(model.f_net, initial_dim=8, target_dim=3, mlp_name="f_net")
    
    # Train briefly
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model, _ = train_model(model, dataloader, optimizer, criterion, epochs=3)
    
    # Prune to active dimensions
    model.f_net.set_schedule(total_epochs=10, decay_rate='linear', end_epoch_frac=0.5)
    sample_data = X_train_tensor[:30]
    model.f_net.prune(5, sample_data, parent_model=model)
    
    active_dims = model.f_net.get_active_dimensions()
    assert len(active_dims) == 3
    
    # Find an inactive dimension
    all_dims = set(range(8))
    inactive_dims = all_dims - set(active_dims)
    inactive_dim = list(inactive_dims)[0]
    
    try:
        # Define transformations
        variable_transforms = [lambda x: x[:, 0], lambda x: x[:, 1]]
        variable_names = ["x0", "x1"]
        
        # Try to run interpretation on inactive dimension
        input_data = X_train_tensor[:40]
        result = model.f_net.interpret(
            input_data,
            parent_model=model,
            output_dim=inactive_dim,
            variable_transforms=variable_transforms,
            fit_params={'variable_names': variable_names},
            sr_params={'niterations': 10}
        )
        
        # Should return empty dict for inactive dimension
        assert result == {}
        
        print("✅ Pruning inactive dimension request test passed")
        
    except Exception as e:
        pytest.fail(f"Pruning inactive dimension request test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def cleanup_sr_outputs():
    """Clean up SR output files and directories created during testing."""
    if os.path.exists('SR_output'):
        shutil.rmtree('SR_output')
    
    # Clean up any custom output directories
    if os.path.exists('custom_pruning_output'):
        shutil.rmtree('custom_pruning_output')
    
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
    """Fixture to clean up output files after all tests complete."""
    yield
    cleanup_sr_outputs()