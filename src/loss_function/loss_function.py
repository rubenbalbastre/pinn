import torch
import torch.nn as nn

from src.loss_function.heat_loss import HeatPDEResidualLoss, AlphaRegularizationLoss
from src.loss_function.wave_loss import WavePDEResidualLoss, CRegularizationLoss
from src.loss_function.diffusion_loss import DiffusionPDEResidualLoss, KappaRegularizationLoss


class DataLoss(nn.Module):

    def forward(self, u, u_pred):
        loss = torch.sum((u-u_pred)**2)

        return loss
    

class Loss(nn.Module):

    def __init__(self, pde_coefficient: float = 1.0, alpha_reg_coefficient: float = 1.0):
        
        super().__init__()
        self.data_loss = DataLoss()
        self.pde_loss = {
            "diffusion": DiffusionPDEResidualLoss(),
            "wave": WavePDEResidualLoss(),
            "heat": HeatPDEResidualLoss()
        }
        self.phys_coeff_regularization = {
            "diffusion": KappaRegularizationLoss(),
            "wave": CRegularizationLoss(),
            "heat": AlphaRegularizationLoss()
        }

        self.pde_coefficient = pde_coefficient
        self.alpha_reg_coefficient = alpha_reg_coefficient

    def forward(self, x, xt, u_pred, u_obs, phys_coeff_pred, nt, data_type):

        loss = (
            self.data_loss(u=u_obs, u_pred=u_pred) 
            + self.pde_coefficient * self.pde_loss[data_type](xt=xt, u_pred=u_pred, phys_coeff_pred=phys_coeff_pred, nt=nt) 
            + self.alpha_reg_coefficient * self.phys_coeff_regularization[data_type](phys_coeff_pred=phys_coeff_pred, x=x)
        )

        return loss