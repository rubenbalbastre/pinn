import torch
import torch.nn as nn
import torch.autograd as autograd


class HeatPDEResidualLoss(nn.Module):

    def forward(self, xt, u_pred, phys_coeff_pred, nt):

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
        alpha_expanded = phys_coeff_pred.repeat_interleave(nt, dim=0)
        residual = u_t - alpha_expanded * u_xx

        loss = torch.mean(residual**2)

        return loss


class AlphaRegularizationLoss(nn.Module):

    def forward(self, phys_coeff_pred, x):
        alpha_x = autograd.grad(phys_coeff_pred, x, torch.ones_like(phys_coeff_pred), create_graph=True)[0]
        loss = torch.mean(alpha_x**2)
        return loss