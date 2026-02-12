import torch


class RelativisticOrbitModelSchwarzschildODE(torch.nn.Module):
    def __init__(self, p, M, e, a=0.0):
        super().__init__()
        self.p = p
        self.M = M
        self.e = e
        self.a = a

    def forward(self, u):
        chi, phi = u
        p = self.p
        M = self.M
        e = self.e

        numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
        denom = torch.sqrt((p - 2)**2 - 4 * e**2)

        phi_dot = numer / (M * p**1.5 * denom)
        chi_dot = numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)

        return torch.stack([chi_dot, phi_dot])


# def RelativisticOrbitModel_Schwarzschild_EMR(u):
#     """
#     Schwarzschild EMR orbit ODE using PyTorch.

#     Args:
#         t: time (unused)
#         u: tensor([chi, phi])
#         model_params: tensor([p, M, e, a]) (a unused here)

#     Returns:
#         tensor([chi_dot, phi_dot])
#     """
#     chi, phi, p, e, M, a  = u

#     numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
#     denom = torch.sqrt((p - 2)**2 - 4 * e**2)

#     phi_dot = numer / (M * p**(1.5) * denom)
#     chi_dot = numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)

#     return torch.stack([chi_dot, phi_dot])
    
class RelativisticOrbitModel_Schwarzschild_EMR(torch.nn.Module):

    def __init__(self, p, M, e, a):

        super().__init__()

        self.p = torch.tensor(p)
        self.M = torch.tensor(M)
        self.e = torch.tensor(e)
        self.a = torch.tensor(a)

    def forward(self, t, u):

        chi, phi = u
        p = self.p
        M = self.M
        e = self.e
        a = self.a

        numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
        denom = torch.sqrt((p - 2)**2 - 4 * e**2)

        phi_dot = numer / (M * p**(1.5) * denom)
        chi_dot = numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)

        du = torch.stack([chi_dot, phi_dot])

        return du


def NNOrbitModel_Schwarzschild_EMR(model, u):
    """
    Schwarzschild EMR orbit model with NN correction in PyTorch.

    Args:
        model: neural network model

    Returns:
        tensor([chi_dot, phi_dot]) with NN correction
    """

    chi, phi, p, e, M, a  = u

    nn_output = model(u)
    nn = 1.0 + nn_output

    numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
    denom = torch.sqrt((p - 2)**2 - 4 * e**2)

    phi_dot = (numer / (M * p**(1.5) * denom)) * nn[0]
    chi_dot = (numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)) * nn[1]

    return torch.stack([chi_dot, phi_dot])
