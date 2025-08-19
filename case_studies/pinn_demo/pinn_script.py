import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 

from symtorch import MLP_SR

alpha = 0.2
def temp(x,t):
    return np.exp(-np.pi**2 * alpha * t) * np.sin(np.pi * x)


# Create 10 random points
np.random.seed(290402)
N = 10
x = np.random.rand(N)
t = np.random.rand(N)


# Make the data points for training
xt = torch.tensor(np.stack([x, t], axis=1), dtype=torch.float32) 
u = torch.tensor(temp(x,t).reshape(-1,1), dtype=torch.float32)

class RegularNN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=32):
        super(RegularNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        self.eval()
        return self.net(x)

class PINN(RegularNN):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=32):
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)

        self.type = 'pinn'
        self.alpha = nn.Parameter(data=torch.tensor([0.]))

    def pde_residual(self, xt):
        # xt: columns are [x, t]
        xt = xt.requires_grad_(True)              # make inputs differentiable
        u  = self.forward(xt)                     # (N,1)

        grads = torch.autograd.grad(
            u, xt, torch.ones_like(u), create_graph=True
        )[0]                                      # (N,2)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xt, torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]

        res = u_t - self.alpha * u_xx
        return res
    
    def bc_residual(self, xt):
            """
            Dirichlet BC: u(0,t)=0 and u(1,t)=0.
            Uses the t values from xt and evaluates the net at x=0 and x=1.
            Returns a single residual stack of shape (2N,1): [u(0,t); u(1,t)].
            """
            t  = xt[:, 1:2].detach()                    # (N,1)
            x0 = torch.zeros_like(t)
            x1 = torch.ones_like(t)

            u0 = self.forward(torch.cat([x0, t], dim=1))  # should be 0
            u1 = self.forward(torch.cat([x1, t], dim=1))  # should be 0

            return torch.cat([u0, u1], dim=0)           # (2N,1)

    def ic_residual(self, xt):
        """
        Initial condition at t=0: u(x,0) = sin(pi x).
        Uses the x values from xt and evaluates the net at t=0.
        Returns residual u(x,0) - sin(pi x) of shape (N,1).
        """
        x  = xt[:, 0:1].detach()                    # (N,1)
        t0 = torch.zeros_like(x)

        u_init = self.forward(torch.cat([x, t0], dim=1))
        target = torch.sin(torch.pi * x)

        return u_init - target                      # (N,1)

    

import torch.optim as optim

def train(model, xt, u, epochs=3000, lr=1e-3, weight_decay=0.0, device="mps", verbose=False):
    reg_pde = 1
    reg_ic = 5
    reg_bc = 5
    """
    Supervised-only training on a tiny labeled set (e.g., 10 points).
    Works for both RegularNN and PINN since both implement forward(x).

    Args:
        model: nn.Module with forward([x,t]) -> u
        xyt:  (N,2) float tensor of inputs
        u:    (N,1) float tensor of targets
    """
    model = model.to(device)
    xt = xt.to(device)
    u   = u.to(device)

    model.train()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    loss_hist = []
    for ep in range(1, epochs+1):
        opt.zero_grad()
        pred = model(xt)
        loss = loss_fn(pred, u)

        if model.type == 'pinn':
            pde_res = model.pde_residual(xt)
            bc_res = model.bc_residual(xt)
            ic_res = model.ic_residual(xt)
            loss += reg_pde * (pde_res**2).mean() + reg_ic * (ic_res**2).mean() + reg_bc * (bc_res**2).mean()

        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        if verbose and ep % 500 == 0:
            print(f"[sup10] ep={ep} loss={loss.item():.3e}")

    return model, loss_hist

def make_2d_hists(nn_model, pinn_model, alpha=1.0, Nx=100, Nt=100):
    import json
    
    # space–time grid
    x = np.linspace(0.0, 1.0, Nx)
    t = np.linspace(0.0, 1.0, Nt)
    X, T = np.meshgrid(x, t, indexing="ij")  # X: (Nx,Nt), T: (Nx,Nt)

    # True solution
    U_exact = np.exp(-np.pi**2 * alpha * T) * np.sin(np.pi * X)
    std = np.std(U_exact)

    def predict_grid(model):
        model.eval()
        XT = np.stack([X.ravel(), T.ravel()], axis=1)  # (Nx*Nt, 2) with [x,t]
        xt = torch.tensor(XT, dtype=torch.float32, device='cpu')
        with torch.no_grad():
            U = model(xt).reshape(Nx, Nt).detach().cpu().numpy()
        return U

    U_nn   = predict_grid(nn_model)
    U_pinn = predict_grid(pinn_model)

    # Calculate MSE losses
    mse_nn = np.mean((U_nn - U_exact)**2)
    mse_pinn = np.mean((U_pinn - U_exact)**2)
    
    # Save MSE losses to JSON
    mse_results = {
        "mse_regular_nn": float(mse_nn),
        "mse_pinn": float(mse_pinn),
        "variance": float(std**2),
        "grid_size": {"Nx": Nx, "Nt": Nt},
        "pred_alpha": float(pinn_model.alpha)
    }
    
    with open('mse_losses.json', 'w') as f:
        json.dump(mse_results, f, indent=2)

    # plot side by side (1x3) with exact in the middle
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    data_plots = [("Regular NN", U_nn), ("True", U_exact), ("PINN", U_pinn)]

    # Find global min/max for consistent colorbar scale
    all_data = [U_nn, U_exact, U_pinn]
    vmin = min(U.min() for U in all_data)
    vmax = max(U.max() for U in all_data)

    for i, (ax, (title, U)) in enumerate(zip(axes, data_plots)):
        im = ax.imshow(U, origin="lower", extent=[0, 1, 0, 1], aspect="auto", 
                      vmin=vmin, vmax=vmax)
        
        # All plots get x-axis (t), only leftmost gets y-axis (x)
        ax.set_xlabel("t", fontsize=16)
        if i == 0:
            ax.set_ylabel("x", fontsize=16)
        else:
            ax.tick_params(left=False, labelleft=False)
        
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Use tight_layout and add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, label="$u(x,t)$")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("$u(x,t)$", fontsize=16)
    plt.savefig('comparison.png', dpi = 300)

# # Train the pINN
# pinn = PINN()
# pinn, loss = train(pinn, xt, u)

# # Save the mode weights
# torch.save(pinn.state_dict(), 'pinn.pth')

# #Train the regular NN
# regular_NN = RegularNN()
# regular_NN, _ = train(regular_NN, xt, u)

# # Save the mode weights
# torch.save(regular_NN.state_dict(), 'regular_NN.pth')


pinn = PINN()
pinn.load_state_dict(torch.load('pinn.pth'))


regular_NN = RegularNN()
regular_NN.load_state_dict(torch.load('regular_NN.pth'))

make_2d_hists(regular_NN, pinn, alpha = alpha)
