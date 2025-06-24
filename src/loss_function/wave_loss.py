import torch
import torch.nn as nn
import torch.autograd as autograd


class WavePDEResidualLoss(nn.Module):

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

        u_tt = autograd.grad(
            outputs=u_t,
            inputs=xt,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True
        )[0][:, 1].unsqueeze(-1)

        u_xx = autograd.grad(
            outputs=u_x,
            inputs=xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0].unsqueeze(-1)

        c_expanded = phys_coeff_pred.repeat_interleave(nt, dim=0)
        residual = u_tt - c_expanded**2 * u_xx

        return torch.sum(residual**2)


class CRegularizationLoss(nn.Module):

    def forward(self, phys_coeff_pred, x):
        c_x = autograd.grad(phys_coeff_pred, x, torch.ones_like(phys_coeff_pred), create_graph=True)[0]
        return torch.mean(c_x**2)
