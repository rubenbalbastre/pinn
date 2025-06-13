import torch

def RelativisticOrbitModel_Schwarzschild_EMR(t, u, model_params):
    """
    Schwarzschild EMR orbit ODE using PyTorch.

    Args:
        t: time (unused)
        u: tensor([chi, phi])
        model_params: tensor([p, M, e, a]) (a unused here)

    Returns:
        tensor([chi_dot, phi_dot])
    """
    chi, phi = u
    p, M, e, a = model_params

    numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
    denom = torch.sqrt((p - 2)**2 - 4 * e**2)

    phi_dot = numer / (M * p**(1.5) * denom)
    chi_dot = numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)

    return torch.stack([chi_dot, phi_dot])


def NNOrbitModel_Schwarzschild_EMR(t, u, model_params, NN=None):
    """
    Schwarzschild EMR orbit model with NN correction in PyTorch.

    Args:
        t: time (unused)
        u: tensor([chi, phi])
        model_params: tensor([p, M, e, a])
        NN: neural network model

    Returns:
        tensor([chi_dot, phi_dot]) with NN correction
    """
    chi, phi = u
    p, M, e, a = model_params

    nn_input = torch.stack([chi, phi, a, p, M, e])

    if NN is None:
        nn = torch.ones(2, dtype=u.dtype, device=u.device)
    else:
        with torch.no_grad():
            nn_output = NN(nn_input)
        nn = 1.0 + nn_output  # correction factors

    numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
    denom = torch.sqrt((p - 2)**2 - 4 * e**2)

    phi_dot = (numer / (M * p**(1.5) * denom)) * nn[0]
    chi_dot = (numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)) * nn[1]

    return torch.stack([chi_dot, phi_dot])
