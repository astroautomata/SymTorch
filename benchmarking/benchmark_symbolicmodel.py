#!/usr/bin/env python
"""
Estimates the speedup gained by replacing layers of a simple torch model with
symbolic equations.
"""
import argparse
import logging
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from symtorch import SymbolicModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class SimpleModel(nn.Module):
    """
    Simple model class.
    """
    def __init__(self, input_dim, output_dim, hidden_dim = 4):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.n_hidden_layers = 3
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ) for _ in range(self.n_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in (self.input_layer, self.hidden_layers, self.output_layer):
            x = layer(x)
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


def generate_data():
    # Make the dataset 
    x = np.array([np.random.uniform(0, 1, 10_000) for _ in range(5)]).T  
    y = x[:, 0]**2 + 3*np.sin(x[:, 4]) - 4
    noise = np.array([np.random.normal(0, 0.05*np.std(y)) for _ in range(len(y))])
    y = y + noise 
    X_train, _, y_train, _ = train_test_split(
        x, y.reshape(-1, 1),
        test_size=0.2,
        random_state=290402
    )
    return x, y, X_train, y_train


def load_or_train_model(
    model_path, x, y, X_train, y_train, epochs=20, sr_params=None, retrain=False
):
    """
    Load existing model if available, otherwise train a new one and fit
    symbolic regression.

    Args:
        model_path: Path to load/save the model
        x, y: Full dataset
        X_train, y_train: Training data
        epochs: Number of epochs to train if creating new model
        sr_params: Symbolic regression parameters (if None, uses defaults)
        retrain: If True, force retraining even if model exists

    Returns:
        Trained model with symbolic regression fitted
    """
    if os.path.exists(f'{model_path}.pt') and not retrain:
        print(f"Loading existing model from {model_path}")
        model = torch.load(f"{model_path}.pt", weights_only=False)
        model.eval()
    else:
        print(f"Training new model (saving to {model_path})")

        # Train the model
        model = SimpleModel(input_dim=x.shape[1], output_dim=1)
        criterion = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        model, losses = train_model(model, dataloader, opt, criterion, epochs)

        # Wrap the MLP
        model.hidden_layers = SymbolicModel(
            model.hidden_layers, block_name='hidden_layers'
        )

        # Configure the SR
        if sr_params is None:
            sr_params = {
                'complexity_of_operators': {"sin": 3, "exp": 3},
                'complexity_of_constants': 2,
                'parsimony': 0.1,
                'verbosity': 0,
                'niterations': 1
            }

        # Distill the model
        model.hidden_layers.distill(
            torch.FloatTensor(X_train),
            sr_params=sr_params,
            parent_model=model
        )

        # Save the model
        if os.path.dirname(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, f"{model_path}.pt")

    return model


def benchmark_model(model, X_data, y_data, device='cpu', num_runs=100):
    """
    Benchmark model performance comparing MLP vs symbolic equation modes.

    Args:
        model: Model with SymbolicModel hidden layers
        X_data: Input data for testing
        y_data: Ground truth output data
        device: Device to run on ('cpu' or 'cuda')
        num_runs: Number of runs for speed benchmarking

    Returns:
        dict: Benchmark results including timing and accuracy metrics
    """
    import time

    # Move model and data to device
    model = model.to(device)
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)

    results = {}

    # Benchmark MLP mode
    model.hidden_layers.switch_to_block()
    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(X_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Time multiple runs
        start = time.time()
        for _ in range(num_runs):
            mlp_output = model(X_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
        mlp_time = (time.time() - start) / num_runs

        # Calculate accuracy
        mlp_mse = nn.MSELoss()(mlp_output, y_tensor).item()
        mlp_mae = torch.mean(torch.abs(mlp_output - y_tensor)).item()

    results['mlp'] = {
        'time': mlp_time,
        'mse': mlp_mse,
        'mae': mlp_mae
    }

    # Benchmark equation mode
    model.hidden_layers.switch_to_symbolic()
    model.eval()
    with torch.no_grad():
        # Warmup
        _ = model(X_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Time multiple runs
        start = time.time()
        for _ in range(num_runs):
            eq_output = model(X_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
        eq_time = (time.time() - start) / num_runs

        # Calculate accuracy
        eq_mse = nn.MSELoss()(eq_output, y_tensor).item()
        eq_mae = torch.mean(torch.abs(eq_output - y_tensor)).item()

    results['equation'] = {
        'time': eq_time,
        'mse': eq_mse,
        'mae': eq_mae
    }

    # Calculate speedup
    results['speedup'] = mlp_time / eq_time

    return results


def print_benchmark_report(results, device='cpu'):
    """
    Print a formatted benchmark report.

    Args:
        results: Dictionary of benchmark results from benchmark_model()
        device: Device used for benchmarking
    """
    import pandas as pd

    print("\n" + "="*60)
    print(f"BENCHMARK REPORT: MLP vs Symbolic Equations ({device.upper()})")
    print("="*60)

    # Create comparison dataframe
    data = {
        'MLP': [
            results['mlp']['time'] * 1000,
            results['mlp']['mse'],
            results['mlp']['mae']
        ],
        'Equation': [
            results['equation']['time'] * 1000,
            results['equation']['mse'],
            results['equation']['mae']
        ]
    }
    df = pd.DataFrame(data, index=['Time (ms)', 'MSE', 'MAE'])
    df['Change (%)'] = ((df['Equation'] - df['MLP']) / df['MLP'] * 100).round(2)

    print("\n" + df.to_string())

    speedup = results['speedup']
    if speedup > 1:
        print(
            f"\nSPEEDUP: {speedup:.2f}x "
            f"(Equations are {speedup:.2f}x faster than MLP)"
        )
    else:
        slowdown = 1 / speedup
        print(
            f"\nSLOWDOWN: {slowdown:.2f}x "
            f"(Equations are {slowdown:.2f}x slower than MLP)"
        )

    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-o', '--outdir',
        type=str,
        default='./benchmark_output',
        help='Directory for saving models and results (default: ./benchmark_output)'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retraining even if saved model exists'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.path.join(args.outdir, 'model')

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Generate data
    x, y, X_train, y_train = generate_data()

    # Load or train model with symbolic regression
    model = load_or_train_model(
        model_path, x, y, X_train, y_train, epochs=25, retrain=args.retrain
    )

    # Benchmark on CPU
    print("\nRunning CPU benchmarks...")
    cpu_results = benchmark_model(model, X_train, y_train, device='cpu', num_runs=100)
    print_benchmark_report(cpu_results, device='cpu')

    # Benchmark on GPU if available
    if torch.cuda.is_available():
        print("\nRunning GPU benchmarks...")
        gpu_results = benchmark_model(model, X_train, y_train, device='cuda', num_runs=100)
        print_benchmark_report(gpu_results, device='cuda')
    else:
        print("\nGPU not available, skipping GPU benchmarks.")


if __name__ == "__main__":
    main()
