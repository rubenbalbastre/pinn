import numpy as np

from src.physics.orbit_model.kerr_orbit_model import NNOrbitModel_Kerr_EMR
from src.physics.orbit_model.schwarzschild_orbit_model import NNOrbitModel_Schwarzschild_EMR
from src.physics.orbit_model.newton_orbit_model import NNOrbitModel_Newton_EMR


def get_pinn_EMR_newton(chi0, phi0, p, M, e, a, tspan, datasize, dt, factor=1, NN=None, NN_params=None):
    """
    Get ODE NN problem in Newtonian orbit modified.
    """

    u0 = np.array([chi0, phi0], dtype=np.float64)
    t_start, t_end = tspan
    t_end *= factor

    tsteps = np.linspace(t_start, t_end, datasize * factor)
    model_params = [p, M, e, a]
    dt_data = tsteps[1] - tsteps[0]

    def ODE_model(t, u):
        du = NNOrbitModel_Newton_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
        return du

    return {
        "nn_problem": ODE_model,
        "tsteps": tsteps,
        "model_params": model_params,
        "u0": u0,
        "dt_data": dt_data,
        "tspan": (t_start, t_end),
        "q": 0.0,
        "p": p,
        "e": e,
        "a": a,
        "M": M,
        "dt": dt
    }


def get_pinn_EMR_schwarzschild(chi0, phi0, p, M, e, a, tspan, datasize, dt, factor=1, NN=None, NN_params=None):
    """
    Get ODE NN problem in Schwarzschild metric modified.
    """

    u0 = np.array([chi0, phi0], dtype=np.float64)
    t_start, t_end = tspan
    t_end *= factor

    tsteps = np.linspace(t_start, t_end, datasize * factor)
    model_params = [p, M, e, a]
    dt_data = tsteps[1] - tsteps[0]

    def ODE_model(t, u):
        du = NNOrbitModel_Schwarzschild_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
        return du

    return {
        "nn_problem": ODE_model,
        "tsteps": tsteps,
        "model_params": model_params,
        "u0": u0,
        "dt_data": dt_data,
        "tspan": (t_start, t_end),
        "q": 0.0,
        "p": p,
        "e": e,
        "a": a,
        "M": M,
        "dt": dt
    }


def get_pinn_EMR_kerr(chi0, phi0, p, M, e, a, tspan, datasize, dt, factor=1, NN=None, NN_params=None):
    """
    Get ODE NN problem in Kerr metric.
    """

    u0 = np.array([chi0, phi0], dtype=np.float64)
    t_start, t_end = tspan
    t_end *= factor

    tsteps = np.linspace(t_start, t_end, datasize * factor)
    model_params = [p, M, e, a]
    dt_data = tsteps[1] - tsteps[0]

    def ODE_model(t, u):
        du = NNOrbitModel_Kerr_EMR(u, model_params, t, NN=NN, NN_params=NN_params)
        return du

    return {
        "nn_problem": ODE_model,
        "tsteps": tsteps,
        "model_params": model_params,
        "u0": u0,
        "dt_data": dt_data,
        "tspan": (t_start, t_end),
        "q": 0.0,
        "p": p,
        "e": e,
        "a": a,
        "M": M,
        "dt": dt
    }


def process_datasets(datasets):
    """
    Create set of datasets. Usually train and test
    """
    processed_data = {}

    for set_name in datasets.keys():
        print(f"Creating {set_name} dataset")
        processed_data[set_name] = []

        for ind, data in enumerate(datasets[set_name], start=1):
            data_dictionary_to_add = data.copy()
            data_dictionary_to_add["index"] = ind
            processed_data[set_name].append(data_dictionary_to_add)

    return processed_data


def get_batch(dataset, batch_size=None):
    """
    Get random subset of data
    """
    import random

    if batch_size is not None and batch_size < len(dataset):
        subset = random.sample(dataset, batch_size)
    else:
        subset = dataset

    return subset
