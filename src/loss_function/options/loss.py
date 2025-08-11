import torch
import torch.autograd as autograd


class BoundaryConditionLoss(torch.nn.Module):

    def __init__(self, T, K):
        super().__init__()
        self.T = T
        self.K = K
    
    def forward(self, V_St, St):
        """
        V(S,t): price option
        K: strike price
        """

        # auxiliary
        S = St[:, 0].unsqueeze(-1)

        # call
        loss_1 = torch.mean(V_St[:, self.T] - max([S-self.K, 0]))
        loss_2 = torch.mean(V_St[0,:])
        loss = loss_1 + loss_2

        return loss


class DataLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, V_St, true_V_St):
        return torch.mean((V_St - true_V_St)**2)
    

class PDELoss(torch.nn.Module):

    def __init__(self, risk_free_interest_rate: float, volatility: float):
        super().__init__()
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility

    def forward(self, V, St):

        S = St[:, 0].unsqueeze(-1)  # S
        
        # Compute gradients of u_pred w.r.t input xt (x and t)
        dV_dSdt = autograd.grad(
            outputs=V,
            inputs=St,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True
        )[0]  # shape [N, 2]

        V_S = dV_dSdt[:, 0].unsqueeze(-1)  # ∂V/∂S
        V_t = dV_dSdt[:, 1].unsqueeze(-1)  # ∂V/∂t

        # ∂/∂S (α ∂u/∂x)
        dV_S_dSdt = autograd.grad(
            outputs=V_S,
            inputs=St,
            grad_outputs=torch.ones_like(V_S),
            create_graph=True,
            retain_graph=True
        )[0]

        V_SS = dV_S_dSdt[:, 0].unsqueeze(-1)  # ∂^2V/∂S^2

        residual = V_t + 1/2 * self.volatility**2 * S**2 * V_SS + self.risk_free_interest_rate * S * V_S - self.risk_free_interest_rate * V.reshape(-1)
        loss = torch.mean(residual**2)

        return loss


class Loss(torch.nn.Module):

    def __init__(self, T: float, K: float, risk_free_interest_rate: float, volatility: float, pde_coeff: float = 1.0):
        super().__init__()
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility
        self.pde_coeff = pde_coeff

        self.data_loss = DataLoss()
        self.pde_loss = PDELoss(risk_free_interest_rate=risk_free_interest_rate, volatility=volatility)
        self.boundary_condition_loss = BoundaryConditionLoss(T=T, K=K)

    def forward(self, V_pred, V_obs, St):
        loss = (
            self.data_loss(V_pred, V_obs)
            + self.pde_coeff * self.pde_loss(V_pred, St)
            # + self.boundary_condition_loss(V_pred, St)
        )
        return loss
