import torch


def generate_mesh_grid(alpha_x, nx=100, nt=100, L=1.0, T=0.1):

    dx = L / (nx - 1)
    
    # Stability check for explicit scheme
    max_alpha = alpha_x.max().item()
    dt_stable = 0.5 * dx**2 / max_alpha
    dt = min(T / nt, dt_stable * 0.9)  # reduce a bit below the limit

    x = torch.linspace(0, L, nx)
    t = torch.linspace(0, T, nt)

    # create input for u_network
    X_mesh, T_mesh = torch.meshgrid(x, t, indexing='ij') # shape [nx, nt]
    xt = torch.stack([X_mesh.flatten(), T_mesh.flatten()], dim=1) # shape [nx*nt, 2]

    return {"xt": xt, "x": x, "t": t, "nx": nx, "nt": nt, "dt": dt, "dx": dx, "L": L, "T": T}
