import torch
import torch.nn as nn
import torch.autograd as autograd


class DiffusionPDEResidualLoss(nn.Module):

    def forward(self, xt, u_pred, phys_coeff_pred, nt):
        du_dxt = autograd.grad(
            outputs=u_pred,
            inputs=xt,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = du_dxt[:, 0].unsqueeze(-1)
        u_t = du_dxt[:, 1].unsqueeze(-1)

        kappa_expanded = phys_coeff_pred.repeat_interleave(nt, dim=0)

        grad_kappa_u_x = autograd.grad(
            outputs=kappa_expanded * u_x,
            inputs=xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0].unsqueeze(-1)

        residual = u_t - grad_kappa_u_x

        return torch.mean(residual**2)


class KappaRegularizationLoss(nn.Module):

    def forward(self, phys_coeff_pred, x):
        kappa_x = autograd.grad(phys_coeff_pred, x, torch.ones_like(phys_coeff_pred), create_graph=True)[0]
        return torch.mean(kappa_x**2)
