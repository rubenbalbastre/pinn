import torch
import torch.nn as nn
import torch.autograd as autograd


class DataLoss(nn.Module):

    def forward(self, u, u_pred):
        loss = torch.sum((u-u_pred)**2)

        return loss
    

class PDEResidualLoss(nn.Module):

    def forward(self, xt, u_pred, alpha_pred, nt):

        # Compute gradients of u_pred w.r.t input xt (x and t)
        du_dxt = autograd.grad(
            outputs=u_pred,
            inputs=xt,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0]  # shape [N, 2]

        u_x = du_dxt[:, 0].unsqueeze(-1)  # ∂u/∂x
        u_t = du_dxt[:, 1].unsqueeze(-1)  # ∂u/∂t

        # Compute ∂²u/∂x² by differentiating u_x w.r.t xt again, select derivative w.r.t x
        u_xx = autograd.grad(
            outputs=u_x,
            inputs=xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0].unsqueeze(-1)

        # residual
        alpha_expanded = alpha_pred.repeat_interleave(nt, dim=0)  # 20 because nx=20; shape: [nx * nt, 1]
        residual = u_t - alpha_expanded * u_xx

        loss = torch.sum(residual**2)

        return loss


class AlphaRegularizationLoss(nn.Module):

    def forward(self, alpha_pred, x):
        alpha_x = autograd.grad(alpha_pred, x, torch.ones_like(alpha_pred), create_graph=True)[0]
        loss = torch.mean(alpha_x**2)
        return loss

class Loss(nn.Module):

    def __init__(self, pde_coefficient: float = 1.0, alpha_reg_coefficient: float = 1.0):
        
        super().__init__()
        self.data_loss = DataLoss()
        self.pde_loss = PDEResidualLoss()
        self.alpha_regularization = AlphaRegularizationLoss()

        self.pde_coefficient = pde_coefficient
        self.alpha_reg_coefficient = alpha_reg_coefficient

    def forward(self, x, xt, u_pred, u_obs, alpha_pred, nt):

        loss = (
            self.data_loss(u=u_obs, u_pred=u_pred) 
            + self.pde_coefficient * self.pde_loss(xt=xt, u_pred=u_pred, alpha_pred=alpha_pred, nt=nt) 
            + self.alpha_reg_coefficient * self.alpha_regularization(alpha_pred=alpha_pred, x=x)
        )

        return loss