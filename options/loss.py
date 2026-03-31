import torch
import torch.autograd as autograd


class BoundaryConditionLoss(torch.nn.Module):

    def __init__(self, T, K, r, Smax):
        super().__init__()
        self.T = T # time to maturity
        self.K = K # strike price
        self.r = r
        self.Smax = Smax
    
    def forward(self, V_St, St, nS: int, nt: int):
        """
        V(S,t): price option
        K: strike price
        """

        # auxiliary
        S = St[:, 0].reshape(nS, nt)
        tau = St[:, 1].reshape(nS, nt)

        # call
        payoff = torch.clamp(S[:, -1] - self.K, min=0.0)
        loss_1 = torch.mean((V_St[:, -1] - payoff) ** 2)
        loss_2 = torch.mean((V_St[0, :]) ** 2)
        upper_bc = self.Smax - self.K * torch.exp(-self.r * tau[-1, :])
        loss_3 = torch.mean((V_St[-1, :] - upper_bc) ** 2)
        loss = loss_1 + loss_2 + loss_3

        return loss


class DataLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, V_St, true_V_St):
        return torch.mean((V_St - true_V_St)**2)
    

class PDELoss(torch.nn.Module):

    def __init__(self, r: float, sigma: float):
        super().__init__()
        self.r = r
        self.sigma = sigma

    def forward(self, V, St):

        # Compute gradients of u_pred w.r.t input xt (x and t)
        dV_dSdt = autograd.grad(
            outputs=V,
            inputs=St,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True
        )[0]  # shape [N, 2]

        V_S = dV_dSdt[:, 0].unsqueeze(-1)  # ∂V/∂S
        V_t = dV_dSdt[:, 1].unsqueeze(-1)  # ∂V/∂tau

        # ∂/∂S (α ∂u/∂S)
        dV_S_dSdt = autograd.grad(
            outputs=V_S,
            inputs=St,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True,
            retain_graph=True
        )[0]

        V_SS = dV_S_dSdt[:, 0].unsqueeze(-1)  # ∂^2V/∂S^2
        S = St[:, 0].unsqueeze(-1)  # S
    
        residual = (
            -V_t + 1/2 * self.sigma**2 * S**2 * V_SS 
            + self.r * S * V_S 
            - self.r * V.reshape(-1))
        loss = torch.mean(residual**2)

        return loss


class Loss(torch.nn.Module):

    def __init__(
        self,
        T: float,
        K: float,
        r: float,
        sigma: float,
        Smax: float,
        data_coeff: float = 0.01,
        pde_coeff: float = 10.0,
        bc_coeff: float = 1.0,
    ):
        super().__init__()
        self.r = r
        self.sigma = sigma
        self.data_coeff = data_coeff
        self.pde_coeff = pde_coeff
        self.bc_coeff = bc_coeff

        self.data_loss = DataLoss()
        self.pde_loss = PDELoss(r=r, sigma=sigma)
        self.boundary_condition_loss = BoundaryConditionLoss(T=T, K=K, r=r, Smax=Smax)

    def forward(self, V_pred_flat, V_pred, V_obs, St, nS: int, nt: int):
        loss = (
            self.data_coeff * self.data_loss(V_pred, V_obs)
            + self.pde_coeff * self.pde_loss(V_pred_flat, St)
            + self.bc_coeff * self.boundary_condition_loss(V_pred, St, nS=nS, nt=nt)
        )
        return loss
