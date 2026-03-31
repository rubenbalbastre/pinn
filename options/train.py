import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from dataset import OptionsDataset
from loss import Loss


# For PyTorch random
torch.manual_seed(45)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create Dataset
Smax = 120.0
T = 10.0
K = 60.0
r = 0.03
sigma = 0.2
dataset = OptionsDataset(
    n_samples=1,
    nS=20,
    nt=20,
    Smax=Smax,
    T=T,
    K=K,
    r=r,
    sigma=sigma,
)

# Define PINNs

class OptionPriceNetwork(nn.Module):

    def __init__(self, hidden_dim=16):
        super(OptionPriceNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=1),
            nn.Softplus()
        )

    def forward(self, St):
        return self.net(St)
 
model = OptionPriceNetwork(hidden_dim=64).to(device)

# training
model.train()

lr = 3e-3
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
criterion = Loss(T=T, K=K, r=r, sigma=sigma, Smax=Smax, data_coeff=0.01, pde_coeff=10.0)

losses = []
loss_data_hist = []
loss_pde_hist = []
loss_bc_hist = []
num_epochs = 4000
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0
    epoch_data = 0.0
    epoch_pde = 0.0
    epoch_bc = 0.0
    for data_batch in dataset:

        # get data
        St = data_batch["St_tau"].to(device)
        V_St = data_batch["V_St"].to(device)
        St.requires_grad_(True)

        # predict V(S,t)
        V_pred_flat = model(St)
        V_pred = V_pred_flat.reshape(data_batch["nS"], data_batch["nt"])

        # remove previous gradients
        optimizer.zero_grad()

        # compute loss components
        loss_data = criterion.data_loss(V_pred, V_St)
        loss_pde = criterion.pde_loss(V_pred_flat, St)
        loss_bc = criterion.boundary_condition_loss(
            V_pred,
            St,
            nS=data_batch["nS"],
            nt=data_batch["nt"]
        )
        loss = (
            criterion.data_coeff * loss_data
            + criterion.pde_coeff * loss_pde
            + criterion.bc_coeff * loss_bc
        )
        epoch_loss += loss
        epoch_data += loss_data
        epoch_pde += loss_pde
        epoch_bc += loss_bc

        # backpropagation & optimization step
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(dataset)
    epoch_data = epoch_data / len(dataset)
    epoch_pde = epoch_pde / len(dataset)
    epoch_bc = epoch_bc / len(dataset)
    losses.append(epoch_loss.item())
    loss_data_hist.append(epoch_data.item())
    loss_pde_hist.append(epoch_pde.item())
    loss_bc_hist.append(epoch_bc.item())

    if epoch % 200 == 0:
        tqdm.write(f"Epoch {epoch}: Loss = {epoch_loss.item():.4e}")

# Evaluation and plots
model.eval()
with torch.no_grad():
    data_batch = dataset[0]
    St = data_batch["St_tau"].to(device)
    V_true = data_batch["V_St"].cpu().numpy()
    V_pred_flat = model(St)
    V_pred = V_pred_flat.reshape(data_batch["nS"], data_batch["nt"]).cpu().numpy()
    S = data_batch["S"].cpu().numpy().reshape(-1)
    tau = data_batch["tau"].cpu().numpy().reshape(-1)


fig_dir = "options/figures/"
os.makedirs(fig_dir, exist_ok=True)

# Plot option price contours (2D)
tau_plot = tau[::-1]
V_true_plot = V_true[:, ::-1]
V_pred_plot = V_pred[:, ::-1]
T_grid, S_grid = np.meshgrid(tau_plot, S, indexing="xy")

fig_price, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
levels = 20

cs1 = axes[0].contourf(T_grid, S_grid, V_true_plot, levels=levels, cmap="viridis")
axes[0].contour(T_grid, S_grid, V_true_plot, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
axes[0].set_title("Exact Contours")
axes[0].set_ylabel("S")
fig_price.colorbar(cs1, ax=axes[0], fraction=0.046, pad=0.04)

axes[1].contour(T_grid, S_grid, V_pred_plot, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
axes[1].set_title("PINN Prediction Contours")
axes[1].set_ylabel("S")
fig_price.colorbar(cs1, ax=axes[1], fraction=0.046, pad=0.04)

residuals = V_pred_plot - V_true_plot
cs3 = axes[2].contourf(T_grid, S_grid, residuals, levels=levels, cmap="coolwarm")
axes[2].contour(T_grid, S_grid, residuals, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
axes[2].set_title("Residuals (Pred - Exact)")
axes[2].set_xlabel("tau")
axes[2].set_ylabel("S")
fig_price.colorbar(cs3, ax=axes[2], fraction=0.046, pad=0.04)

fig_price.tight_layout()
fig_price.savefig(fig_dir + "option_price_contours.png")

# Plot loss curves
fig_loss, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
axes[0].plot(losses, label="Total Loss")
axes[0].set_ylabel("Total Loss")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(loss_data_hist, label="Data Loss")
axes[1].set_ylabel("Data Loss")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(loss_pde_hist, label="PDE Loss")
axes[2].set_ylabel("PDE Loss")
axes[2].grid(True)
axes[2].legend()

axes[3].plot(loss_bc_hist, label="BC Loss")
axes[3].set_ylabel("BC Loss")
axes[3].set_xlabel("Epoch")
axes[3].grid(True)
axes[3].legend()

fig_loss.tight_layout()
fig_loss.savefig(fig_dir + "loss_curves.png")
