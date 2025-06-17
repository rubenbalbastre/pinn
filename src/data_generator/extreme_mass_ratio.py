import torch
import random


def build_pinn_problem(nn_model, chi0, phi0, p, M, e, a, tspan, datasize, dt, factor=1):
    """
    Generalized constructor for PyTorch-compatible PINN problems.
    """

    u0 = torch.tensor([chi0, phi0], dtype=torch.float64)
    t_start, t_end = tspan
    t_end *= factor

    tsteps = torch.linspace(t_start, t_end, datasize * factor, dtype=torch.float64)
    model_params = torch.tensor([p, M, e, a])
    dt_data = tsteps[1] - tsteps[0]

    return {
        "nn_problem": nn_model,
        "tsteps": tsteps,
        "model_params": model_params,
        "u0": u0,
        "dt_data": dt_data,
        "tspan": tspan,
        "q": 0.0,
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
            data_dict = data.copy()
            data_dict["index"] = ind
            processed_data[set_name].append(data_dict)

    return processed_data


def get_batch(dataset, batch_size=None):
    """
    Get random subset of data
    """

    if batch_size is not None and batch_size < len(dataset):
        return random.sample(dataset, batch_size)
    return dataset
