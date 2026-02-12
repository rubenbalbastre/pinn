import torch.nn as nn


class OptionPriceNetwork(nn.Module):

    def __init__(self, hidden_dim=16):
        super(OptionPriceNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim,out_features=1),
            nn.Sigmoid()
        )

    def forward(self, St):
        return self.net(St)