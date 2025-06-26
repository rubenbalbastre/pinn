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

        # α(x) * ∂u/∂x
        alpha_expanded = phys_coeff_pred.repeat_interleave(nt, dim=0)
        flux = alpha_expanded * u_x

        # ∂/∂x (α ∂u/∂x)
        dflux_dxdt = autograd.grad(
            outputs=flux,
            inputs=xt,
            grad_outputs=torch.ones_like(flux),
            create_graph=True,
            retain_graph=True
        )[0]

        dflux_dx = dflux_dxdt[:, 0].unsqueeze(-1)  # ∂flux/∂x

        residual = u_t - dflux_dx
        loss = torch.mean(residual**2)

        return loss


class AlphaRegularizationLoss(nn.Module):

    def forward(self, phys_coeff_pred, x):
        alpha_x = autograd.grad(phys_coeff_pred, x, torch.ones_like(phys_coeff_pred), create_graph=True)[0]
        loss = torch.mean(alpha_x**2)
        return loss