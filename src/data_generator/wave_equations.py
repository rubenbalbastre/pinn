import torch
import math
import random
from torch.utils.data import Dataset

from src.data_generator.mesh_grid import generate_mesh_grid
from src.data_generator.utils import encode_u_type


def solve_wave_equation(c_x, x, t, dt, dx, u0, v0):
    """
    Solves the 1D wave equation:
        u_tt = c(x)^2 * u_xx
    where:
        - c_x: wave speed profile (tensor of shape [nx])
        - u0: initial displacement
        - v0: initial velocity
    """
    nx = x.shape[0]
    nt = t.shape[0]
    u_prev = u0.clone()
    u = u0 + dt * v0 + 0.5 * dt**2 * (c_x**2) * (torch.roll(u0, -1) - 2*u0 + torch.roll(u0, 1)) / dx**2
    u[0] = u[-1] = 0.0  # Dirichlet BCs

    u_sol = [u_prev.unsqueeze(0), u.unsqueeze(0)]

    for _ in range(2, nt):
        u_next = 2*u - u_prev + (dt**2) * (c_x**2) * (torch.roll(u, -1) - 2*u + torch.roll(u, 1)) / dx**2
        u_next[0] = u_next[-1] = 0.0  # Dirichlet BCs
        u_prev, u = u, u_next
        u_sol.append(u.unsqueeze(0))

    return torch.cat(u_sol, dim=0)  # [nt, nx]


def generate_wave_speed_profile(nx, kind="smooth"):
    x = torch.linspace(0, 1, nx)
    if kind == "smooth":
        return 1.0 + 0.5 * torch.sin(2 * math.pi * x)
    elif kind == "piecewise":
        c = torch.ones_like(x)
        c[x > 0.5] = 0.3
        return c
    elif kind == "random":
        return torch.rand_like(x) * 0.8 + 0.2
    else:
        raise ValueError("Unknown kind")


class WaveEquationDataset(Dataset):
    def __init__(self, n_samples=100, nx=100, nt=100, L=1.0, T=0.1):
        self.data = []
        for _ in range(n_samples):

            # c
            kind = random.choice(["smooth", "piecewise", "random"])
            c_x = generate_wave_speed_profile(nx, kind)

            # mesh
            mesh = generate_mesh_grid(c_x, nx, nt, L, T)

            # boundaries
            u0 = torch.sin(math.pi * mesh["x"])
            x = torch.linspace(0, L, nx)
            t = torch.linspace(0, T, nt)
            dx = L / (nx - 1)
            dt = T / nt
            u0 = torch.sin(math.pi * x)
            v0 = torch.zeros_like(x)

            u_xt = solve_wave_equation(c_x, x, t, dt, dx, u0, v0)

            item_information = mesh | {
                "u_type": encode_u_type("wave"),
                "u_type_txt": "wave",
                "kind": kind,
                "alpha": c_x,     # shape: (nx,)
                "u_xt": u_xt,         # shape: (nt+1, nx)
                "u0": u0,
            }

            item_information["x"] = item_information["x"].reshape(-1,1)

            self.data.append(item_information)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
