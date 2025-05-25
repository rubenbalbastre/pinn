import numpy as np
import torch

def RelativisticOrbitModel_Schwarzschild_EMR(t, u, model_params):
    """
    Motion of a point-like particle in Schwarzschild background.

    Parameters:
    - u: [chi, phi]
    - model_params: [p, M, e, a] (a is unused)
    - t: time (unused)

    Returns:
    - [dchi/dt, dphi/dt]
    """
    chi, phi = u
    p, M, e, a = model_params

    numer = (p - 2 - 2 * e * np.cos(chi)) * (1 + e * np.cos(chi))**2
    denom = np.sqrt((p - 2)**2 - 4 * e**2)

    phi_dot = numer / (M * p**(3/2) * denom)
    chi_dot = numer * np.sqrt(p - 6 - 2 * e * np.cos(chi)) / (M * p**2 * denom)

    return [chi_dot, phi_dot]


def NNOrbitModel_Schwarzschild_EMR(t, u, model_params, NN=None, NN_params=None):
    """
    Schwarzschild EMR orbit model with neural network perturbation.

    Parameters:
    - u: [chi, phi]
    - model_params: [p, M, e, a]
    - t: time (unused)
    - NN: neural network model
    - NN_params: parameters for the neural network

    Returns:
    - [dchi/dt, dphi/dt] with NN correction
    """
    chi, phi = u
    p, M, e, a = model_params

    nn_input = np.array([chi, phi, a, p, M, e], dtype=np.float32)

    if NN is None:
        nn = np.array([1.0, 1.0])
    else:
        input_tensor = torch.tensor(nn_input, dtype=torch.float32)
        with torch.no_grad():
            nn_output = NN(input_tensor, NN_params).numpy()
        nn = 1.0 + nn_output

    numer = (p - 2 - 2 * e * np.cos(chi)) * (1 + e * np.cos(chi))**2
    denom = np.sqrt((p - 2)**2 - 4 * e**2)

    phi_dot = (numer / (M * p**(3/2) * denom)) * nn[0]
    chi_dot = (numer * np.sqrt(p - 6 - 2 * e * np.cos(chi)) / (M * p**2 * denom)) * nn[1]

    return [chi_dot, phi_dot]
