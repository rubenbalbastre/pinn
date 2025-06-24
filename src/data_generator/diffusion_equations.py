import torch
import math
import random
from torch.utils.data import Dataset

from src.data_generator.mesh_grid import generate_mesh_grid
from src.data_generator.utils import encode_u_type


def solve_diffusion_equation(kappa_x, x, t, dt, dx, u0):
    """
    Solves the 1D diffusion equation:
        u_t = d/dx (kappa(x) * du/dx)
    using finite difference approximation.
    """
    nx = x.shape[0]
    nt = t.shape[0]
    u = u0.clone()
    u_sol = [u.unsqueeze(0)]

    for _ in range(1, nt):
        u_new = u.clone()
        for i in range(1, nx - 1):
            kappa_avg_right = 0.5 * (kappa_x[i] + kappa_x[i+1])
            kappa_avg_left = 0.5 * (kappa_x[i] + kappa_x[i-1])
            flux_right = kappa_avg_right * (u[i+1] - u[i]) / dx
            flux_left = kappa_avg_left * (u[i] - u[i-1]) / dx
            u_new[i] = u[i] + dt * (flux_right - flux_left) / dx
        u_new[0] = u_new[-1] = 0.0  # Dirichlet BCs
        u = u_new
        u_sol.append(u.unsqueeze(0))

    return torch.cat(u_sol, dim=0)  # [nt, nx]


def generate_kappa_profile(nx, kind="smooth"):
    x = torch.linspace(0, 1, nx)
    if kind == "smooth":
        return 0.5 + 0.3 * torch.sin(2 * math.pi * x)
    elif kind == "piecewise":
        kappa = torch.ones_like(x) * 0.5
        kappa[x > 0.5] = 0.2
        return kappa
    elif kind == "random":
        return torch.rand_like(x) * 0.6 + 0.2
    else:
        raise ValueError("Unknown kind")
    

class DiffusionEquationDataset(Dataset):

    def __init__(self, n_samples=100, nx=100, nt=100, L=1.0, T=0.1):

        self.data = []
        for _ in range(n_samples):
            # kappa
            kind = random.choice(["smooth", "piecewise", "random"])
            kappa_x = generate_kappa_profile(nx, kind)
            # mesh
            mesh = generate_mesh_grid(kappa_x, nx, nt, L, T)
            # boundaries
            u0 = torch.sin(math.pi * mesh["x"])
            u_xt = solve_diffusion_equation(
                kappa_x, 
                x=mesh["x"], 
                t=mesh["x"], 
                dt=mesh["dt"],
                dx=mesh["dx"], 
                u0=u0
            )
            item_information = mesh | {
                "u_type": encode_u_type("diffusion"),
                "u_type_txt": "diffusion",
                "kind": kind,
                "alpha": kappa_x,     # shape: (nx,)
                "u_xt": u_xt,         # shape: (nt+1, nx)
                "u0": u0,
            }

            item_information["x"] = item_information["x"].reshape(-1,1)

            self.data.append(item_information)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
