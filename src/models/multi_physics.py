import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encodes (x, y, z, t, u, u_type) into a latent material representation z_mat.
    u_type is expected to be a one-hot or learned embedding.
    """
    def __init__(self, input_dim=6, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, xytu_type):
        return self.net(xytu_type)  # [N, latent_dim]


class SharedDecoder(nn.Module):
    """
    Decodes z_mat and spatial coords into a hidden representation shared across all property heads.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim + 3, 64),  # x, y, z + z_mat
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, z_mat, xyz):
        z_repeated = z_mat.expand(xyz.size(0), -1)  # [N, latent_dim]
        inp = torch.cat([xyz, z_repeated], dim=-1)  # [N, latent_dim+3]
        return self.shared_net(inp)  # [N, 64]


class MultiMeasurementHeads(nn.Module):

    def __init__(self, latent_dim: int = 32):

        self.net = nn.Sequential(

        )

    def forward():
        return self.heads()


class MultiPropertyHeads(nn.Module):
    """
    Outputs multiple physical properties from the shared decoder representation.
    Each head corresponds to one property (e.g., alpha, wave_speed).
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.heads = nn.ModuleDict({
            "alpha": nn.Linear(hidden_dim + 3, 1),         # thermal diffusivity
            "wave_speed": nn.Linear(hidden_dim + 3, 1),    # wave speed
            "permeability": nn.Linear(hidden_dim + 3, 1)   # porous flow
            # Add more as needed
        })

    def forward(self, hidden, prop_name):
        return self.heads[prop_name](hidden)  # [N, 1]
