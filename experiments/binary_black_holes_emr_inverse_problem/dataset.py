import math
import torch
from torch.utils.data import Dataset

from numerical_solvers import solve_ode_rk2
from orbital_mechanics import compute_waveform, compute_orbit
from schwarzschild_models import RelativisticOrbitModelSchwarzschildODE
from kerr_models import RelativisticOrbitModelODE


def generate_mesh(tspan, dt, factor=1):
    t_start, t_end = tspan
    t_end *= factor
    datasize = math.ceil((t_end - t_start) / dt)
    tsteps = torch.linspace(t_start, t_end, datasize * factor, dtype=torch.float64)
    return {"tsteps": tsteps, "dt_data": torch.tensor([dt])}


class ScharzschildDataset(Dataset):

    def __init__(self, chi0: float, phi0: float, p_space: list, M: float, e: float, a: float, train_mesh):
        self.data_list = []
        self._generate_data(chi0, phi0, p_space, M, e, a, train_mesh)


    def _generate_data(self, chi0, phi0, p_space, M, e, a, train_mesh):

        for p in p_space:

            system_params = {
                # mesh
                "tsteps": train_mesh["tsteps"].unsqueeze(0),
                "dt_data": train_mesh["dt_data"].unsqueeze(0),
                # system
                "u0": torch.tensor([chi0, phi0], dtype=torch.float64).unsqueeze(0),
                "p": torch.tensor([p], dtype=torch.float64).unsqueeze(0),
                "M": torch.tensor([M], dtype=torch.float64).unsqueeze(0),
                "e": torch.tensor([e], dtype=torch.float64).unsqueeze(0),
                "a": torch.tensor([a], dtype=torch.float64).unsqueeze(0)
            }
            ode_problem = RelativisticOrbitModelSchwarzschildODE(
                p=system_params['p'],
                M=system_params['M'],
                e=system_params['e'],
            )
            system_solution = solve_ode_rk2(
                ode_problem=ode_problem,
                system_params=system_params
            )
            system_orbit = compute_orbit(system_solution, system_params)
            system_waveform = compute_waveform(system_orbit, system_params=system_params)

            item = system_params | {'solution': system_solution, 'orbit': system_orbit, 'waveform': system_waveform}

            item = {k: v.squeeze(0) for k, v in item.items()}  # remove batch dim for storage
            self.data_list.append(item)
            
            print(f"Generated training data for system: p: {system_params['p'].item()}, a: {system_params['a'].item()}, e: {system_params['e'].item()}, M: {system_params['M'].item()}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item


class KerrDataset(Dataset):

    def __init__(self, chi0: float, phi0: float, p_space: list, M: float, e: float, a: float, train_mesh):
        self.data_list = []
        self._generate_data(chi0, phi0, p_space, M, e, a, train_mesh)


    def _generate_data(self, chi0, phi0, p_space, M, e, a, train_mesh):

        for p in p_space:

            system_params = {
                # mesh
                "tsteps": train_mesh["tsteps"].unsqueeze(0),
                "dt_data": train_mesh["dt_data"].unsqueeze(0),
                # system
                "u0": torch.tensor([chi0, phi0], dtype=torch.float64).unsqueeze(0),
                "p": torch.tensor([p], dtype=torch.float64).unsqueeze(0),
                "M": torch.tensor([M], dtype=torch.float64).unsqueeze(0),
                "e": torch.tensor([e], dtype=torch.float64).unsqueeze(0),
                "a": torch.tensor([a], dtype=torch.float64).unsqueeze(0)
            }
            ode_problem = RelativisticOrbitModelODE(
                p=system_params['p'],
                M=system_params['M'],
                e=system_params['e'],
                a=system_params['a']
            )
            system_solution = solve_ode_rk2(
                ode_problem=ode_problem,
                system_params=system_params
            )
            system_orbit = compute_orbit(system_solution, system_params)
            system_waveform = compute_waveform(system_orbit, system_params=system_params)

            item = system_params | {'solution': system_solution, 'orbit': system_orbit, 'waveform': system_waveform}

            item = {k: v.squeeze(0) for k, v in item.items()}  # remove batch dim for storage
            self.data_list.append(item)
            
            print(f"Generated training data for system: p: {system_params['p'].item()}, a: {system_params['a'].item()}, e: {system_params['e'].item()}, M: {system_params['M'].item()}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item