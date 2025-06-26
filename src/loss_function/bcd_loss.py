import torch.nn as nn
import torch

from src.loss_function.wave_loss import WavePDEResidualLoss, CRegularizationLoss
from src.loss_function.diffusion_loss import DiffusionPDEResidualLoss, KappaRegularizationLoss
from src.loss_function.loss_function import DataLoss, PhysicalPropertyResidualLoss


class ULoss(nn.Module):

    def __init__(self, pde_coefficient: float = 10.0):
        
        super().__init__()
        self.data_loss = DataLoss()
        self.property_loss = PhysicalPropertyResidualLoss()
        self.pde_loss = {
            "diffusion": DiffusionPDEResidualLoss(),
            "wave": WavePDEResidualLoss()
        }

        self.pde_coefficient = pde_coefficient

    def forward(self, x, xt, u_pred, u_obs, phys_coeff, phys_coeff_pred, nt, u_type):

        loss = (
            self.data_loss(u=u_obs, u_pred=u_pred)
            + self.pde_coefficient * self.pde_loss[u_type](xt=xt, u_pred=u_pred, phys_coeff_pred=phys_coeff_pred, nt=nt) 
        )

        return loss
    

class AlphaLoss(nn.Module):

    def __init__(self, pde_coefficient: float = 10.0, alpha_reg_coefficient: float = 1.0, phys_property_coefficient: float = 0.0):
        
        super().__init__()
        self.data_loss = DataLoss()
        self.pde_loss = {
            "diffusion": DiffusionPDEResidualLoss(),
            "wave": WavePDEResidualLoss()
        }
        self.phys_coeff_regularization = {
            "diffusion": KappaRegularizationLoss(),
            "wave": CRegularizationLoss()
        }

        self.pde_coefficient = pde_coefficient
        self.alpha_reg_coefficient = alpha_reg_coefficient

    def forward(self, x, xt, u_pred, u_obs, phys_coeff, phys_coeff_pred, nt, u_type):

        loss = (
            + self.pde_coefficient * self.pde_loss[u_type](xt=xt, u_pred=u_pred, phys_coeff_pred=phys_coeff_pred, nt=nt) 
            + self.alpha_reg_coefficient * self.phys_coeff_regularization[u_type](phys_coeff_pred=phys_coeff_pred, x=x)
        )

        return loss