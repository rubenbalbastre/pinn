import torch
from torch.utils.data import Dataset
import numpy as np

from src.data_generator.mesh_grid import generate_mesh_grid


def solve_heat_equation(alpha_x, x, t, dt, dx, u0, nx):

    u = u0.clone()
    u_sol = [u.unsqueeze(0)]

    for _ in range(1, t.shape[0]):
        u_new = u.clone()
        for i in range(1, nx - 1):
            u_new[i] = u[i] + dt * alpha_x[i] * (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new
        u_sol.append(u.unsqueeze(0))

    return torch.cat(u_sol, dim=0)


# Random diffusivity profiles
def generate_alpha_profile(nx, kind="smooth"):
    x = torch.linspace(0, 1, nx)
    if kind == "smooth":
        return 1.0 + 0.5 * torch.sin(2 * np.pi * x)
    elif kind == "piecewise":
        alpha = torch.ones_like(x)
        alpha[x > 0.5] = 0.2
        return alpha
    elif kind == "random":
        return torch.rand_like(x) * 0.8 + 0.2
    else:
        raise ValueError("Unknown kind")


# Torch Dataset
class HeatEquationDataset(Dataset):

    def __init__(self, n_samples=100, nx=100, nt=100, L=1.0, T=0.1):

        self.data = []

        for _ in range(n_samples):

            # alpha
            alpha_kind = np.random.choice(["smooth", "piecewise", "random"])
            alpha_x = generate_alpha_profile(nx, kind=alpha_kind)
            # mesh
            mesh = generate_mesh_grid(alpha_x=alpha_x, nx=nx, nt=nt, L=L, T=T)
            # boundaries
            u0 = torch.sin(np.pi * mesh["x"])  # initial condition
            # solve PDE
            u_xt = solve_heat_equation(
                alpha_x=alpha_x, 
                x=mesh["x"],
                t=mesh["t"],
                dt=mesh["dt"],
                dx=mesh["dx"],
                u0=u0,
                nx=mesh["nx"]
            )

            item_information = mesh | {
                "data_type": "heat",
                "kind": alpha_kind,
                "alpha": alpha_x,     # shape: (nx,)
                "u_xt": u_xt,         # shape: (nt+1, nx)
                "u0": u0,
            }

            item_information["x"] = item_information["x"].reshape(-1,1)

            self.data.append(item_information)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
