import torch
import numpy as np
import time
import warnings

# Suppress torch.compile warnings about tensor construction
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
warnings.filterwarnings("ignore", message="torch.compile for Metal is an early prototype")

from symtorch import MLP_SR
from pinn_script import RegularNN, PINN

# Load the models with symbolic regression using SymTorch's load_model
print("Loading models with symbolic regression...")

# Create architectures and load with SymTorch
pinn = PINN()
pinn.net = MLP_SR.load_model('pinn_with_sr', pinn.net, device='mps')
pinn = pinn.to(torch.device('mps'))

regular_NN = RegularNN()
regular_NN.net = MLP_SR.load_model('regular_NN_with_sr', regular_NN.net, device='mps')
regular_NN = regular_NN.to(torch.device('mps'))

print("✅ Models loaded successfully with SymTorch functionality")

# Prepare test data
num_data = 5000
# Use torch.from_numpy to avoid tensor construction warnings
np_data = np.stack([np.linspace(0,1,num_data), np.linspace(0,1,num_data)], axis=1).astype(np.float32)
sample_data = torch.from_numpy(np_data)
test_data = sample_data.to(torch.device('mps'))

print("\n=== Forward Pass Timing Tests ===")
print(f"Testing with {len(test_data)} samples")

num_runs = 100  # Increase for more accurate timing

def time_forward_pass(model, test_data, num_runs, description):
    """Time a model's forward pass"""
    model.eval()
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_data)
    
    # Time the forward passes
    torch.mps.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(test_data)
    
    torch.mps.synchronize()
    elapsed_time = (time.time() - start_time) / num_runs
    
    print(f"{description}: {elapsed_time*1000:.2f} ms per forward pass")
    return elapsed_time

# Test PINN in regular neural network mode
pinn.net.switch_to_mlp()  # Make sure we're in regular mode
regular_time = time_forward_pass(pinn.net, test_data, num_runs, "PINN regular mode")

# Test PINN in equation mode
pinn.net.switch_to_equation()
equation_time = time_forward_pass(pinn.net, test_data, num_runs, "PINN equation mode")

# Test torch.compile optimization for equation mode
print("\n--- Testing torch.compile optimization ---")
print("⚠️ Note: torch.compile with MPS is experimental and may show warnings")

try:
    # Make sure we're in equation mode
    pinn.net.switch_to_equation()
    
    # Create compiled version with reduced mode for MPS compatibility
    print("Attempting torch.compile with 'reduce-overhead' mode for MPS...")
    compiled_net = torch.compile(pinn.net, mode='reduce-overhead')
    
    compiled_time = time_forward_pass(compiled_net, test_data, num_runs, "PINN equation mode (torch.compile)")
    
    # Calculate and display speedups
    print(f"\n=== Speedup Analysis ===")
    print(f"Equation vs Regular: {regular_time/equation_time:.2f}x speedup")
    print(f"Compiled vs Regular: {regular_time/compiled_time:.2f}x speedup") 
    print(f"Compiled vs Equation: {equation_time/compiled_time:.2f}x speedup")
    
except Exception as e:
    print(f"❌ torch.compile failed: {e}")
    print("💡 This is expected on MPS - torch.compile support for Metal is experimental")
    print(f"\n=== Speedup Analysis (without torch.compile) ===")
    print(f"Equation vs Regular: {regular_time/equation_time:.2f}x speedup")

print(f"\nRegular NN timing for comparison:")
regular_nn_time = time_forward_pass(regular_NN.net, test_data, num_runs, "Regular NN (equation mode)")

print(f"\n=== Analysis ===")
print(f"PINN equation complexity: {pinn.net.pysr_regressor[0].get_best()['complexity']}")
print(f"PINN equation: {pinn.net.pysr_regressor[0].get_best()['equation']}")
print(f"\nNote: Equation mode may be slower for complex equations with exp/sin operations")
print(f"compared to optimized matrix operations in the neural network.")