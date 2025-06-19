import torch
import torch.nn as nn


class alpha_network(nn.Module):

    def __init__(self):
        super(alpha_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1)
        )

    def forward(self, x):
        return self.net(x)
    

class u_network(nn.Module):

    def __init__(self):
        super(u_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12,out_features=1)
        )

    def forward(self, xt):
        return self.net(xt)



