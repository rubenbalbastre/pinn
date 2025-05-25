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


def NNOrbitModel_Newton_EMR(t, u, model_params, NN=None, NN_params=None):
    """
    Neural Newtonian ODE model with learned corrections.

    Args:
        t: time
        u: torch.tensor([χ, ϕ])
        model_params: tuple (p, M, e, a)
        NN: neural network model taking input [χ, ϕ, a, p, M, e]
        NN_params: optional parameters (not needed if using standard PyTorch models)

    Returns:
        torch.tensor([χ̇, ϕ̇])
    """
    χ, ϕ = u
    p, M, e, a = model_params

    input_tensor = torch.tensor([χ, ϕ, a, p, M, e], dtype=torch.float32)

    if NN is None:
        nn = torch.tensor([1.0, 1.0], dtype=torch.float32)
    else:
        # Output should be a 2-element tensor (e.g., [Δχ̇, Δϕ̇])
        nn_output = NN(input_tensor)
        nn = 1.0 + nn_output  # Additive correction to baseline dynamics

    numer = (1 + e * torch.cos(χ)) ** 2
    denom = M * (p ** 1.5)

    χ_dot = (numer / denom) * nn[0]
    ϕ_dot = (numer / denom) * nn[1]

    return torch.stack([χ_dot, ϕ_dot])
