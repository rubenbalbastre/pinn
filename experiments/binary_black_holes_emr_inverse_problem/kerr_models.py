import torch
import torch.nn as nn


class RelativisticOrbitModelODE(nn.Module):
    def __init__(self, p, M, e, a):
        super().__init__()
        self.p = p
        self.M = M
        self.e = e
        self.a = a

    def forward(self, t, u):
        chi, phi = torch.unbind(u, dim=1)
        p = self.p
        M = self.M
        e = self.e
        a = self.a

        L = L_kerr(p, e, M, a)
        E = E_kerr(p, e, M, a)

        r = p * M / (1 + e * torch.cos(chi))
        drdchi = p * M * e * torch.sin(chi) / (1 + e * torch.cos(chi))**2
        Delta = r**2 - 2 * M * r + a**2

        dphidtau = ((1 - 2*M/r) * L + 2 * M * a * E / r) / Delta
        dtdtau = ((r**2 + a**2 + 2 * M * a**2 / r) * E - 2 * M * a * L / r) / Delta
        drdtau = compute_drdtau(chi, a, Delta, L, E, r, M)
        
        phi_dot = dphidtau / dtdtau
        chi_dot = drdtau / (dtdtau * drdchi + 1e-10)

        return torch.cat([chi_dot, phi_dot], dim=1)
    

# Helper function: to create a lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)


class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class BasicNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            LambdaLayer(lambda x: torch.stack([torch.cos(x[:,0]), x[:,0]], dim=1)),
            nn.Linear(2, 64),
            Cosine(),
            nn.Linear(64, 2)
        )
        for l in self.model:
            if hasattr(l, 'weight'):
                torch.nn.init.uniform_(l.weight, -0.005,0.005)
                # torch.nn.init.zeros_(l.weight)

    def forward(self, x):
        return self.model(x)


class NNOrbitModel_Kerr_EMR(nn.Module):
    
    def __init__(self, p, M , e, network):
        super().__init__()
        self.p = p
        self.M = M
        self.e = e
        self.network = network

    def update_params(self, p=None, M=None, e=None):
        if p is not None: 
            self.p = p 
        if M is not None: 
            self.M = M 
        if e is not None: 
            self.e = e

    def forward(self, t, u):
        
        χ, ϕ = torch.unbind(u, dim=1)
        χ = χ.unsqueeze(1)  # (B, 1)
        ϕ = ϕ.unsqueeze(1)  # (B, 1)

        p = self.p
        e = self.e
        a = self.a 
        M = self.M

        # network prediction
        out = self.network(u)

        # Kerr dynamics
        L = L_kerr(p, e, M, a)
        E = E_kerr(p, e, M, a)
        r = p * M / (1 + e * torch.cos(χ))
        drdchi = p * M * e * torch.sin(χ) / (1 + e * torch.cos(χ))**2
        Delta = r**2 - 2 * M * r + a**2

        dphidtau = ((1 - 2*M/r) * L + 2 * M * a * E / r) / Delta
        dtdtau = ((r**2 + a**2 + 2 * M * a**2 / r) * E - 2 * M * a * L / r) / Delta
        drdtau = compute_drdtau(χ, a, Delta, L, E, r, M)

        ϕ_dot = (dphidtau / dtdtau) * out[0]
        χ_dot = (drdtau / (dtdtau * drdchi)) * out[1]

        return torch.cat([χ_dot, ϕ_dot], dim=1)


def compute_drdtau(chi, a, Delta, L, E, r, M):
    """
    Avoid numerical issues computing dr/dτ in PyTorch
    """
    # term = (r**2 * E**2 + 2 * M * (a * E - L)**2 / r + (a**2 * E**2 - L**2) - Delta) / r**2
    # dr_dtau = torch.sqrt(term)  # add epsilon for stability
    # dr_dtau = torch.where(torch.sin(chi) < 0, -dr_dtau, dr_dtau)

    term = (r**2 * E**2 + 2 * M * (a * E - L)**2 / r + (a**2 * E**2 - L**2) - Delta) / r**2
    # Clamp to avoid invalid sqrt from small negative values due to numerics.
    term = torch.clamp(term, min=1e-12)
    dr_dtau = torch.sqrt(term)

    return dr_dtau


def E_kerr(p, e, M, a):
    """
    Energy of Kerr time-like geodesic
    """
    inner = a**2*(-1 + e**2)**4*M**2*p**3*(a**4*(-1 + e**2)**2 +
        M**4*(-4*e**2 + (-2 + p)**2)*p**2 +
        2*a**2*M**2*p*(-2 + p + e**2*(2 + p)))
    inner = torch.clamp(inner, min=1e-12)
    numerator = (M**4*p**3*(-2 - 2*e + p)*(-2 + 2*e + p)*(-3 - e**2 + p) -
    a**2*(-1 + e**2)**2*M**2*p**2*(-5 + e**2 + 3*p) -
    2*torch.sqrt(inner))
    denom = (M**2*p**3*(-4*a**2*(-1 + e**2)**2 + M**2*(3 + e**2 - p)**2*p))
    res = torch.sqrt(torch.clamp(numerator / denom, min=1e-12))
    
    return res


def L_kerr(p, e, M, a):    
    """
    Angular momentum of Kerr time-like geodesic
    """
    eps = torch.tensor(1e-8, dtype=a.dtype, device=a.device)
    a_safe = torch.where(torch.abs(a) < eps, eps, a)

    inner = a_safe**2*(-1 + e**2)**4*M**2*p**3*(a_safe**4*(-1 + e**2)**2 +
        M**4*(-4*e**2 + (-2 + p)**2)*p**2 +
        2*a_safe**2*M**2*p*(-2 + p + e**2*(2 + p)))
    inner = torch.clamp(inner, min=1e-12)
    numerator = (M**4*p**3*(-2 - 2*e + p)*(-2 + 2*e + p)*(-3 - e**2 + p) -
            a_safe**2*(-1 + e**2)**2*M**2*p**2*(-5 + e**2 + 3*p) -
            2*torch.sqrt(inner))
    denom = (M**2*p**3*(-4*a**2*(-1 + e**2)**2 + M**2*(3 + e**2 - p)**2*p))
    prefactor = torch.sqrt(torch.clamp(numerator / denom, min=1e-12))
    numerator2 = (a_safe**4*(-1 + e**2)**4 +
        a_safe**2*(-1 + e**2)**2*M**2*p*(-4 + 3*p + e**2*(4 + p)) - torch.sqrt(inner))
    denom2 = (a_safe**3*(-1 + e**2)**4 -
        a_safe*(-1 + e**2)**2*M**2*(-4*e**2 + (-2 + p)**2)*p)
    res = prefactor * (numerator2 / denom2)
    return res
