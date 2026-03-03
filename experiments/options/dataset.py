from torch.utils.data import Dataset
import torch
import math


def generate_mesh_grid(nS: int = 100, nt: int = 100, Smax: float = 100.0, T: float = 100.0):

    dS = Smax / (nS - 1)
    dt = T / (nt - 1)
    
    S = torch.linspace(0, Smax, nS)
    t = torch.linspace(0, T, nt)

    S_mesh, T_mesh = torch.meshgrid(S, t, indexing='ij') # shape [nS, nt]
    St = torch.stack([S_mesh.flatten(), T_mesh.flatten()], dim=1) # shape [nS*nt, 2]

    return {"St": St, "S": S, "t": t, "nS": nS, "nt": nt, "dt": dt, "dS": dS, "Smax": Smax, "T": T}


def solve_black_scholes_equation(S, t, K, T, r, sigma):
    """
    Analytical Black–Scholes price for a European call (no dividends) in PyTorch.

    Parameters
    ----------
    S : torch.Tensor, shape (N,)
        Spot prices.
    t : torch.Tensor, shape (N,)
        Current times in [0, T].
    K : float
        Strike price.
    T : float
        Maturity time.
    r : float
        Risk-free interest rate (annualized, cont. comp.).
    sigma : float
        Volatility (annualized).

    Returns
    -------
    torch.Tensor, shape (N,)
        Call option prices for each (S, t) pair.
    """
    # Ensure tensors and float64 for precision
    S = torch.as_tensor(S, dtype=torch.float64)
    t = torch.as_tensor(t, dtype=torch.float64)

    # Time to maturity
    tau = torch.clamp(T - t, min=0.0)
    eps = 1e-12
    sqrt_tau = torch.sqrt(torch.clamp(tau, min=eps))
    S_safe = torch.clamp(S, min=eps)

    # Payoff at expiry
    payoff = torch.clamp(S - K, min=0.0)

    # For tau=0, return payoff
    mask_expired = tau == 0.0
    price = torch.empty_like(S, dtype=torch.float64)

    # Compute d1, d2 where tau > 0
    tau_pos = tau[~mask_expired]
    S_pos = S_safe[~mask_expired]
    sqrt_tau_pos = sqrt_tau[~mask_expired]

    d1 = (torch.log(S_pos / K) + (r + 0.5 * sigma**2) * tau_pos) / (sigma * sqrt_tau_pos + eps)
    d2 = d1 - sigma * sqrt_tau_pos

    Phi = lambda x: 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    price_pos = S_pos * Phi(d1) - K * torch.exp(-r * tau_pos) * Phi(d2)

    # Assign values
    price[mask_expired] = payoff[mask_expired]
    price[~mask_expired] = price_pos

    return price


class OptionsDataset(Dataset):
    """
    Solves the Black-Scholes equation
    using finite difference approximation.
    """

    def __init__(self, n_samples: int = 1, Smax: float = 120.0, T: float = 100.0, nS: int = 100, nt: int = 100):

        self.data = []

        # mesh grid
        mesh = generate_mesh_grid(
            nS=nS,
            nt=nt,
            Smax=Smax,
            T=T
        )

        # PDE parameters
        K: float = 120.0
        r: float = 0.03
        sigma: float = 1.2
        
        for _ in range(n_samples):

            # boundaries
            # V0 = torch.sin(math.pi * mesh["x"])
            V_St = solve_black_scholes_equation(
                K=K,
                S=mesh["S"],
                t=mesh["t"],
                T=T,
                r=r,
                sigma=sigma
            )
            item_information = mesh | {
                "V_St": V_St,         # shape: (nt+1, nx)
                # "V0": V0,
            }
            for v in ["S", "St", "V_St"]:
                item_information[v] = item_information[v].requires_grad_(True)
            
            item_information["S"] = item_information["S"].reshape(-1,1)

            self.data.append(item_information)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
