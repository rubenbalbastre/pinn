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


def NNOrbitModel_Newton_EMR(t, u, model_params):
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


class NNOrbitModel_Newton_EMR(nn.Module):
    
    def __init__(self, p, M , e, network):
        super().__init__()
        self.p = p
        self.M = M
        self.e = e
        self.network = network

    def forward(self, t, u):
        
        χ, ϕ = torch.unbind(u, dim=1)
        χ = χ.unsqueeze(1)  # (B, 1)
        ϕ = ϕ.unsqueeze(1)  # (B, 1

        # Output should be a (B,2) - element tensor (e.g., [Δχ̇, Δϕ̇])
        out = 1 + self.network(u)

        numer = (1 + self.e * torch.cos(χ)) ** 2
        denom = self.M * (self.p ** 1.5)
        χ_dot = (numer / denom) * out[:, 0].unsqueeze(1)
        ϕ_dot = (numer / denom) * out[:, 1].unsqueeze(1)
        
        return torch.cat([χ_dot, ϕ_dot], dim=1)