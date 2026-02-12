import torch
import math
import random
from torch.utils.data import Dataset

from src.data_generator.mesh_grid import generate_mesh_grid
from src.data_generator.utils import encode_u_type, concat_encoder_input


def solve_diffusion_equation(alpha_x, x, t, dt, dx, u0):
    """
    Solves the 1D diffusion equation:
        u_t = d/dx (alpha(x) * du/dx)
    using finite difference approximation.
    """
    nx = x.shape[0]
    nt = t.shape[0]
    u = u0.clone()
    u_sol = [u.unsqueeze(0)]

    for _ in range(1, nt):
        u_new = u.clone()
        for i in range(1, nx - 1):
            alpha_avg_right = 0.5 * (alpha_x[i] + alpha_x[i+1])
            alpha_avg_left = 0.5 * (alpha_x[i] + alpha_x[i-1])
            flux_right = alpha_avg_right * (u[i+1] - u[i]) / dx
            flux_left = alpha_avg_left * (u[i] - u[i-1]) / dx
            u_new[i] = u[i] + dt * (flux_right - flux_left) / dx
        u_new[0] = u_new[-1] = 0.0  # Dirichlet BCs
        u = u_new
        u_sol.append(u.unsqueeze(0))

    return torch.cat(u_sol, dim=0)  # [nt, nx]


def generate_alpha_profile(nx, kind="smooth"):
    x = torch.linspace(0, 1, nx)
    if kind == "smooth":
        return 0.5 + 0.3 * torch.sin(2 * math.pi * x)
    elif kind == "piecewise":
        alpha = torch.ones_like(x) * 0.5
        alpha[x > 0.5] = 0.2
        return alpha
    else:
        raise ValueError("Unknown kind")
    

class DiffusionEquationDataset(Dataset):

    def __init__(self, n_samples=100, nx=100, nt=100, L=1.0, T=0.1):

        self.data = []
        for _ in range(n_samples):
            # alpha
            kind = random.choice(["smooth", "piecewise", "random"])
            alpha_x = generate_alpha_profile(nx, kind="smooth")
            # mesh
            mesh = generate_mesh_grid(alpha_x, nx, nt, L, T)
            # boundaries
            u0 = torch.sin(math.pi * mesh["x"])
            u_xt = solve_diffusion_equation(
                alpha_x, 
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
                "alpha": alpha_x,     # shape: (nx,)
                "u_xt": u_xt,         # shape: (nt+1, nx)
                "u0": u0,
            }
            for v in ["x", "xt", "u_xt", "u_type"]:
                item_information[v] = item_information[v].requires_grad_(True)
            
            item_information["encoder_input"] = concat_encoder_input(
                u_xt=item_information["u_xt"],
                xt=item_information["xt"],
                u_type=item_information["u_type"]
            ).requires_grad_(True)

            item_information["x"] = item_information["x"].reshape(-1,1)

            self.data.append(item_information)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
