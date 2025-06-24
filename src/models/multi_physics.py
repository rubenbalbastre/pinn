import torch.nn as nn
import torch


class Encoder(nn.Module):
    """
    Encodes (x, t, u, u_type) into a latent material representation z_mat
    """
    def __init__(self, input_dim=6, latent_dim=32, num_heads=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))  # shape: [B, 1, latent_dim]

    def forward(self, xytu_type):
        z_points = self.net(xytu_type).unsqueeze(0)  # [1, N, latent_dim] (batch-first)
        Q = self.query # [1, 1, latent_dim]
        z_global, attn_weights = self.attn(Q, z_points, z_points)  # output: [1, 1, latent_dim]
        return z_global.squeeze(0)       # [1, latent_dim], [1, N]


class SharedDecoder(nn.Module):
    """
    Decodes z_mat and spatial coords into a hidden representation shared across all property heads.
    """
    def __init__(self, latent_dim=32, output_dim=16):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim + 2, 32),  # x + z_mat
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, xtz_mat):
        return self.shared_net(xtz_mat)  # [N, output_dim]


class MultiMeasurementHeads(nn.Module):
    """
    Generates physical properties from spacetime coordinates and z_mat
    """

    def __init__(self, spacetime_dim=2):
        super().__init__()
        self.spacetime_dim = spacetime_dim
        self.heads = nn.ModuleDict({
            "heat": nn.Sequential(
            nn.Linear(in_features=self.spacetime_dim + 1, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1),
            nn.ReLU()
        ),
            "diffusion": nn.Sequential(
            nn.Linear(in_features=self.spacetime_dim + 1, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1),
            nn.ReLU()
        ),
            "wave": nn.Sequential(
            nn.Linear(in_features=self.spacetime_dim + 1, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1),
            nn.ReLU()
        )
        })

    def forward(self, xtz_mat, physical_measurement):
        return self.heads[physical_measurement](xtz_mat)


class MultiPropertyHeads(nn.Module):
    """
    Outputs multiple physical properties from the shared decoder representation.
    Each head corresponds to one property (e.g., alpha, wave_speed).
    """
    def __init__(self, latent_dim=32, spacetime_dim=1):
        super().__init__()
        self.heads = nn.ModuleDict({
            "heat": nn.Sequential(
            nn.Linear(in_features=latent_dim + spacetime_dim, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1),
            nn.ReLU()
        ), # thermal diffusivity
            "wave": nn.Sequential(
            nn.Linear(in_features=latent_dim + spacetime_dim, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1),
            nn.ReLU()
        ),  # wave speed
            "diffusion": nn.Sequential(
            nn.Linear(in_features=latent_dim + spacetime_dim, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1),
            nn.ReLU()
        )  # porous flow
            # Add more as needed
        })

    def forward(self, xz_mat, prop_name):
        return self.heads[prop_name](xz_mat)  # [N, 1]
