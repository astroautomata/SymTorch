import torch
import torch.nn as nn
import numpy as np

from symtorch import MLP_SR, Pruning_MLP

import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple MLP.
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
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

    def forward(self, x):
        return self.mlp(x)

class SimpleModel(nn.Module):
    """
    Model with MLP f_net and linear g_net.
    """
    def __init__(self, input_dim, output_dim, output_dim_f=32, hidden_dim=128):
        super(SimpleModel, self).__init__()

        self.f_net = MLP(input_dim, output_dim_f, hidden_dim)
        # g is linear - only learns to combine the 2 pruned outputs from f
        self.g_net = nn.Linear(output_dim_f, output_dim)  # Will use first 2 dims of f after pruning

    def forward(self, x):
        x = self.f_net(x)
        x = self.g_net(x)
        return x


# Make the dataset 
x = np.array([np.random.uniform(0, 1, 10_000) for _ in range(5)]).T

def f_func(x):
    f0 = x[:, 0]**2 
    f1 = np.sin(x[:, 4])  
    return np.stack([f0, f1], axis=1)

def g_func(f_output):
    a, b = 2.5, -1.3  
    return a * f_output[:, 0] + b * f_output[:, 1]

# Generate ground truth data
f_true = f_func(x)
y = g_func(f_true)

noise = np.array([np.random.normal(0, 0.05*np.std(y)) for _ in range(len(y))])
y = y + noise 

# Create model with pruning for f, linear g_net
model = SimpleModel(input_dim=x.shape[1], output_dim=1, output_dim_f=32)
model_base = SimpleModel(input_dim=x.shape[1], output_dim=1, output_dim_f=32)
model.f_net = Pruning_MLP(model.f_net,
                      initial_dim=32, # Initial dimensionality of the MLP
                      target_dim=2, # Target dimensionality - final output dim after pruning
                      mlp_name="f_net")


# Set up the pruning schedule
epochs = 100
model.f_net.set_schedule(total_epochs=epochs, 
                     end_epoch_frac=0.7 # End pruning after 70% of epochs
                     )

# Set up training

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def train_model(model, dataloader, X_val, opt, criterion, epochs=100, pruning = True):
    """
    Train model with MLP f (with pruning) and linear g_net.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        X_val, y_val: Validation data for pruning
        opt: Optimizer
        criterion: Loss function
        epochs: Number of training epochs
        
    Returns:
        tuple: (trained_model, loss_tracker, active_dims_tracker)
    """
    loss_tracker = []
    active_dims_tracker = []
    
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
        

        if pruning == True:
            active_dims_tracker.append(model.f_net.pruning_mask.sum().item())
            model.f_net.prune(epoch, sample_data = X_val, # Pass in the validation set (or a subset of) to the model
                            parent_model = model) # Pass in the parent model to get the correct inputs to the layer

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            if pruning == True:
                active_dims = model.f_net.pruning_mask.sum().item()
                print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}, Active dims: {active_dims}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}')
    
    if pruning == True:
        return model, loss_tracker, active_dims_tracker
    else:
        return model, loss_tracker

# Set up training
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)
# Split data
X_train, X_val, y_train, y_val = train_test_split(
    x, y.reshape(-1,1), test_size=0.1, random_state=290402)

# Set up dataset - only x as input now
dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model and save the weights
print("Starting training on pruning model...")
model, losses, active_dims = train_model(model, dataloader, torch.FloatTensor(X_val), opt, criterion, epochs=epochs)
print("Training completed!")

print("Starting training on base model...")
model_base, losses_base = train_model(model_base, dataloader, torch.FloatTensor(X_val), opt, criterion, epochs=epochs, pruning=False)
print("Training completed!")
# torch.save(model.state_dict(), 'model_weights.pth')

print("\nRunning symbolic regression on pruned f...")

sr_params = {'complexity_of_operators':  {"sin":3, "exp":3},
             'complexity_of_constants': 2, 
             'constraints': {"sin": 3, "exp":3},
             'parsimony': 0.05,
            #  'verbosity': 0, 
             'niterations': 500}

model.f_net.distill(torch.FloatTensor(X_train), 
                       sr_params=sr_params)


# Now try to distill the model with no pruning 

model_base.f_net = MLP_SR(model_base.f_net)
importance_info= model_base.f_net.get_importance(torch.FloatTensor(X_val))
dims = importance_info['importance']
model_base.f_net.distill(torch.FloatTensor(X_train), 
                       sr_params=sr_params,
                       output_dim = dims[0])

model_base.f_net.distill(torch.FloatTensor(X_train), 
                       sr_params=sr_params,
                       output_dim = dims[1])