import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

# Define the PDE residual: u_xx + pi^2 * sin(pi x) = 0
def pde_residual(model, x):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = - (np.pi**2) * torch.sin(np.pi * x)
    return u_xx - f

# Training data
n_domain = 100
n_boundary_conditions = 2
x_domain = torch.linspace(0, 1, n_domain).reshape(-1, 1).to(device)
x_boundary_conditions = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(device)
u_boundary_conditions = torch.zeros((n_boundary_conditions, 1)).to(device)

# Model setup
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Loss logs
loss_total_hist = []
loss_boundary_conditions_hist = []
loss_pde_hist = []

# Training loop
n_epochs = 5000
for epoch in tqdm(range(n_epochs)):

    # remove old gradients
    optimizer.zero_grad()

    # PDE loss
    res = pde_residual(model, x_domain)
    loss_pde = torch.mean(res**2)

    # Boundary loss
    u_pred_boundary_conditions = model(x_boundary_conditions)
    loss_boundary_conditions = loss_fn(u_pred_boundary_conditions, u_boundary_conditions)

    # optimize
    loss = loss_pde + loss_boundary_conditions
    loss.backward()
    optimizer.step()

    # log losses
    loss_total_hist.append(loss.item())
    loss_boundary_conditions_hist.append(loss_boundary_conditions.item())
    loss_pde_hist.append(loss_pde.item())

    if epoch % 500 == 0:
        tqdm.write(f"Epoch {epoch}, PDE Loss: {loss_pde.item():.5e}, boundary_conditions Loss: {loss_boundary_conditions.item():.5e}")

# Evaluation
x_test = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
u_pred = model(x_test).detach().cpu().numpy()
u_true = np.sin(np.pi * x_test.cpu().numpy())

# Plotting
fig = plt.figure()
plt.plot(x_test.cpu(), u_true, ".", color="black", label="Exact")
plt.plot(x_test.cpu(), u_pred, "-", color="red", label="PINN Prediction")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Poisson Equation Solution")
plt.grid(True)
fig_dir = "experiments/toy_problem/figures/"
os.makedirs(fig_dir, exist_ok=True)
fig.savefig(fig_dir + "pinn_poisson.png")

# Plot loss curves
fig_loss, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
axes[0].plot(loss_total_hist, label="Total Loss")
axes[0].set_ylabel("Total Loss")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(loss_boundary_conditions_hist, label="BC Loss")
axes[1].set_ylabel("BC Loss")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(loss_pde_hist, label="PDE Loss")
axes[2].set_ylabel("PDE Loss")
axes[2].set_xlabel("Epoch")
axes[2].grid(True)
axes[2].legend()

fig_loss.tight_layout()
fig_loss.savefig(fig_dir + "loss_curves.png")
