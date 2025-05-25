import numpy as np
import torch


def E_kerr(p, e, M, a):
    """
    Energy of Kerr time-like geodesic
    """
    term1 = M**4 * p**3 * (-2 - 2*e + p) * (-2 + 2*e + p) * (-3 - e**2 + p)
    term2 = a**2 * (1 - e**2)**2 * M**2 * p**2 * (-5 + e**2 + 3*p)
    inner_sqrt = a**2 * (1 - e**2)**4 * M**2 * p**3 * (
        a**4 * (1 - e**2)**2 +
        M**4 * (-4*e**2 + (-2 + p)**2) * p**2 +
        2 * a**2 * M**2 * p * (-2 + p + e**2 * (2 + p))
    )
    numerator = np.sqrt(term1 - term2 - 2 * np.sqrt(inner_sqrt))
    denominator = M**2 * p**3 * (-4 * a**2 * (1 - e**2)**2 + M**2 * (3 + e**2 - p)**2 * p)
    return numerator / denominator


def L_kerr(p, e, M, a):
    """
    Angular momentum of Kerr time-like geodesic
    """
    term1 = M**4 * p**3 * (-2 - 2*e + p) * (-2 + 2*e + p) * (-3 - e**2 + p)
    term2 = a**2 * (1 - e**2)**2 * M**2 * p**2 * (-5 + e**2 + 3*p)
    inner_sqrt = a**2 * (1 - e**2)**4 * M**2 * p**3 * (
        a**4 * (1 - e**2)**2 +
        M**4 * (-4*e**2 + (-2 + p)**2) * p**2 +
        2 * a**2 * M**2 * p * (-2 + p + e**2 * (2 + p))
    )
    num1 = np.sqrt(term1 - term2 - 2 * np.sqrt(inner_sqrt))
    den1 = M**2 * p**3 * (-4 * a**2 * (1 - e**2)**2 + M**2 * (3 + e**2 - p)**2 * p)
    main_term = num1 / den1

    aux1 = a**4 * (1 - e**2)**4
    aux2 = a**2 * (1 - e**2)**2 * M**2 * p * (-4 + 3*p + e**2*(4 + p))
    aux3 = np.sqrt(inner_sqrt)
    num2 = aux1 + aux2 - aux3
    den2 = a**3 * (1 - e**2)**4 - a * (1 - e**2)**2 * M**2 * (-4 * e**2 + (-2 + p)**2) * p

    return main_term * (num2 / den2)


def compute_drdtau(chi, a, Delta, L, E, r, M):
    """
    Avoid numerical issues computing dr/dτ
    """
    term = (r**2 * E**2 + 2*M*(a*E - L)**2 / r + (a**2 * E**2 - L**2) - Delta) / r**2
    dr_dtau = np.sqrt(term)
    if np.sin(chi) < 0:
        dr_dtau = -dr_dtau
    return dr_dtau


def RelativisticOrbitModel_Kerr_EMR(t, u, model_params):

    chi, phi = u
    p, M, e, a = model_params

    L = L_kerr(p, e, M, a)
    E = E_kerr(p, e, M, a)

    r = p * M / (1 + e * np.cos(chi))
    drdchi = p * M * e * np.sin(chi) / (1 + e * np.cos(chi))**2
    Delta = r**2 - 2 * M * r + a**2

    dphidtau = ((1 - 2*M/r) * L + 2 * M * a * E / r) / Delta
    dtdtau = ((r**2 + a**2 + 2 * M * a**2 / r) * E - 2 * M * a * L / r) / Delta
    drdtau = compute_drdtau(chi, a, Delta, L, E, r, M)

    phi_dot = dphidtau / dtdtau
    chi_dot = drdtau / (dtdtau * drdchi)

    return np.array([chi_dot, phi_dot])


def NNOrbitModel_Kerr_EMR(t, u, model_params, NN=None, NN_params=None):
    chi, phi = u
    p, M, e, a = model_params

    input_vector = np.array([chi, phi, a, p, M, e], dtype=np.float32)

    if NN is None:
        nn = np.array([1.0, 1.0])
    else:
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        with torch.no_grad():
            nn_output = NN(input_tensor, NN_params).numpy()
        nn = 1.0 + nn_output

    L = L_kerr(p, e, M, a)
    E = E_kerr(p, e, M, a)

    r = p * M / (1 + e * np.cos(chi))
    drdchi = p * M * e * np.sin(chi) / (1 + e * np.cos(chi))**2
    Delta = r**2 - 2 * M * r + a**2

    dphidtau = ((1 - 2*M/r) * L + 2 * M * a * E / r) / Delta
    dtdtau = ((r**2 + a**2 + 2 * M * a**2 / r) * E - 2 * M * a * L / r) / Delta
    drdtau = compute_drdtau(chi, a, Delta, L, E, r, M)

    phi_dot = (dphidtau / dtdtau) * nn[0]
    chi_dot = (drdtau / (dtdtau * drdchi)) * nn[1]

    return [chi_dot, phi_dot]
