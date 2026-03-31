import torch
import torch.nn as nn


class RelativisticOrbitModelSchwarzschildODE(nn.Module):
    def __init__(self, p, M, e):
        super().__init__()
        self.p = p
        self.M = M
        self.e = e

    def forward(self, t, u):
        chi, phi = torch.unbind(u, dim=1)
        p = self.p
        M = self.M
        e = self.e

        numer = (p - 2 - 2 * e * torch.cos(chi)) * (1 + e * torch.cos(chi))**2
        denom = torch.sqrt((p - 2)**2 - 4 * e**2)

        phi_dot = numer / (M * p**1.5 * denom)
        chi_dot = numer * torch.sqrt(p - 6 - 2 * e * torch.cos(chi)) / (M * p**2 * denom)
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


class NNOrbitModel_Schwarzcshild_EMR(nn.Module):
    
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
        ϕ = ϕ.unsqueeze(1)  # (B, 1

        # Output should be a (B,2) - element tensor (e.g., [Δχ̇, Δϕ̇])
        out = 1 + self.network(u)

        numer = (self.p-2-2*self.e*torch.cos(χ)) * (1+self.e*torch.cos(χ))**2
        denom = torch.sqrt( (self.p-2)**2-4*self.e**2 )

        χ_dot = (numer / (self.M*(self.p**(3/2))*denom)) * out[:, 0].unsqueeze(1)
        ϕ_dot = (numer * torch.sqrt( self.p-6-2*self.e*torch.cos(χ) )/( self.M*(self.p**2)*denom )) * out[:, 1].unsqueeze(1)
        
        return torch.cat([χ_dot, ϕ_dot], dim=1)
