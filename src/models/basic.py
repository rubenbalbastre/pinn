import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper function: to create a lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)
    

class BasicNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            LambdaLayer(lambda x: torch.stack([x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,2]**2, x[:,2]**3], dim=1).float()),
            nn.Linear(5, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.model(x)

