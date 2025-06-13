import torch

from src.physics.orbit_model.kerr_orbit_model import NNOrbitModel_Kerr_EMR
from src.physics.orbit_model.schwarzschild_orbit_model import NNOrbitModel_Schwarzschild_EMR
from src.physics.orbit_model.newton_orbit_model import NNOrbitModel_Newton_EMR


def _build_pinn_problem(metric_model, chi0, phi0, p, M, e, a, tspan, datasize, dt, factor=1):
    """
    Generalized constructor for PyTorch-compatible PINN problems.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u0 = torch.tensor([chi0, phi0], dtype=torch.float64, device=device)
    t_start, t_end = tspan
    t_end *= factor

    tsteps = torch.linspace(t_start, t_end, datasize * factor, dtype=torch.float64, device=device)
    model_params = torch.tensor([p, M, e, a])
    dt_data = tsteps[1] - tsteps[0]

    class TorchOrbitModel(torch.nn.Module):
        def forward(self, t, u):
            du = metric_model(t, u, model_params)
            return torch.tensor(du, dtype=torch.float64, device=device)

    return {
        "nn_problem": TorchOrbitModel(),
        "tsteps": tsteps,
        "model_params": model_params,
        "u0": u0,
        "dt_data": dt_data,
        "tspan": tspan,
        "q": 0.0,
        "p": p,
        "e": e,
        "a": a,
        "M": M,
        "dt": dt
    }


def get_pinn_EMR_newton(*args, **kwargs):
    return _build_pinn_problem(NNOrbitModel_Newton_EMR, *args, **kwargs)


def get_pinn_EMR_schwarzschild(*args, **kwargs):
    return _build_pinn_problem(NNOrbitModel_Schwarzschild_EMR, *args, **kwargs)


def get_pinn_EMR_kerr(*args, **kwargs):
    return _build_pinn_problem(NNOrbitModel_Kerr_EMR, *args, **kwargs)


def process_datasets(datasets):
    """
    Create set of datasets. Usually train and test
    """
    processed_data = {}

    for set_name in datasets.keys():
        print(f"Creating {set_name} dataset")
        processed_data[set_name] = []

        for ind, data in enumerate(datasets[set_name], start=1):
            data_dict = data.copy()
            data_dict["index"] = ind
            processed_data[set_name].append(data_dict)

    return processed_data


def get_batch(dataset, batch_size=None):
    """
    Get random subset of data
    """
    import random
    if batch_size is not None and batch_size < len(dataset):
        return random.sample(dataset, batch_size)
    return dataset
