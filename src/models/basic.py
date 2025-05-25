import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def process_chain(chain, initialize_as_zero=True):
    """
    Extract parameters from chain.
    If initialize_as_zero is True, reduce all weights by factor 100.
    """
    if initialize_as_zero:
        print("Reducing weights two orders of magnitude")
        with torch.no_grad():
            for param in chain.parameters():
                param.div_(100)
    return chain

# Helper function: to create a lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

def nn_model_emr_kerr_from_schwarzschild(number_of_neurons_in_hidden_layer, activation_function, type_="standard"):
    """
    Define architecture similar to the Julia version
    """
    if type_ == "standard":
        chain = nn.Sequential(
            LambdaLayer(lambda x: torch.stack([x[:,0], x[:,0]*x[:,2], x[:,2], x[:,2]**2, x[:,2]**3], dim=1).float()),
            nn.Linear(5, number_of_neurons_in_hidden_layer // 2),
            activation_function,
            nn.Linear(number_of_neurons_in_hidden_layer // 2, number_of_neurons_in_hidden_layer),
            activation_function,
            nn.Linear(number_of_neurons_in_hidden_layer, number_of_neurons_in_hidden_layer),
            activation_function,
            nn.Linear(number_of_neurons_in_hidden_layer, 2)
        )
    else:
        raise ValueError(f"Unknown type {type_}")
    return process_chain(chain)

def nn_model_emr_kerr_from_newton(number_of_neurons_in_hidden_layer, activation_function, type_="standard"):
    """
    Similar to nn_model_emr_kerr_from_schwarzschild
    """
    if type_ == "standard":
        chain = nn.Sequential(
            LambdaLayer(lambda x: torch.stack([x[:,0], x[:,0]*x[:,2], x[:,2], x[:,2]**2, x[:,2]**3], dim=1).float()),
            nn.Linear(5, number_of_neurons_in_hidden_layer // 2),
            activation_function,
            nn.Linear(number_of_neurons_in_hidden_layer // 2, number_of_neurons_in_hidden_layer),
            activation_function,
            nn.Linear(number_of_neurons_in_hidden_layer, 2)
        )

    else:
        raise ValueError(f"Unknown type {type_}")
    return process_chain(chain)
