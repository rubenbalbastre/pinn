import torch.nn as nn


class alpha_network(nn.Module):

    def __init__(self, hidden_dim=16):
        super(alpha_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim,out_features=1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)
    

class u_network(nn.Module):

    def __init__(self, hidden_dim=16):
        super(u_network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim,out_features=1),
            nn.Softplus()
        )

    def forward(self, xt):
        return self.net(xt)


# class u_network(nn.Module):

#     def __init__(self, hidden_dim=16):
#         super(u_network, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_features=3, out_features=hidden_dim),
#             nn.Tanh(),
#             nn.Linear(in_features=hidden_dim,out_features=1),
#             nn.Tanh()
#         )

#     def forward(self, xtalpha):
#         return self.net(xtalpha)