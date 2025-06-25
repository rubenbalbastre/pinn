import torch
import torch.nn as nn

from src.loss_function.heat_loss import HeatPDEResidualLoss, AlphaRegularizationLoss
from src.loss_function.wave_loss import WavePDEResidualLoss, CRegularizationLoss
from src.loss_function.diffusion_loss import DiffusionPDEResidualLoss, KappaRegularizationLoss


class DataLoss(nn.Module):

    def forward(self, u, u_pred):
        loss = torch.mean((u-u_pred)**2)
        return loss
    

class PhysicalPropertyResidualLoss(nn.Module):

    def forward(self, phys_coeff, phys_coeff_pred):
        res = torch.mean((phys_coeff - phys_coeff_pred)**2)
        return res
    

class Loss(nn.Module):

    def __init__(self, pde_coefficient: float = 10.0, alpha_reg_coefficient: float = 1.0, phys_property_coefficient: float = 0.0):
        
        super().__init__()
        self.data_loss = DataLoss()
        self.property_loss = PhysicalPropertyResidualLoss()
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

        self.phys_property_coefficient = phys_property_coefficient
        self.pde_coefficient = pde_coefficient
        self.alpha_reg_coefficient = alpha_reg_coefficient

    def forward(self, x, xt, u_pred, u_obs, phys_coeff, phys_coeff_pred, nt, u_type):

        loss = (
            self.data_loss(u=u_obs, u_pred=u_pred)
            + self.phys_property_coefficient * self.property_loss(phys_coeff=phys_coeff, phys_coeff_pred=phys_coeff_pred)
            + self.pde_coefficient * self.pde_loss[u_type](xt=xt, u_pred=u_pred, phys_coeff_pred=phys_coeff_pred, nt=nt) 
            + self.alpha_reg_coefficient * self.phys_coeff_regularization[u_type](phys_coeff_pred=phys_coeff_pred, x=x)
        )

        # print("Data Loss: ", self.data_loss(u=u_obs, u_pred=u_pred))
        # print("PDE Loss: ", self.pde_loss[u_type](xt=xt, u_pred=u_pred, phys_coeff_pred=phys_coeff_pred, nt=nt) )
        # print("alpha Loss: ", self.phys_coeff_regularization[u_type](phys_coeff_pred=phys_coeff_pred, x=x))

        return loss