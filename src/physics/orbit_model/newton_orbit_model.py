import torch

def NewtonianOrbitModel_EMR(t, u, model_params):
    """
    Defines ODEs for Newtonian motion of a point-like particle.

    Args:
        t: time (not used explicitly, required by ODE solver)
        u: torch.tensor([χ, ϕ])
        model_params: tuple (p, M, e, a)

    Returns:
        torch.tensor([χ̇, ϕ̇])
    """
    χ, ϕ = u
    p, M, e, a = model_params

    numer = (1 + e * torch.cos(χ)) ** 2
    denom = M * (p ** 1.5)

    χ_dot = numer / denom
    ϕ_dot = numer / denom

    return torch.stack([χ_dot, ϕ_dot])
