import torch
from torchdiffeq import odeint

from src.physics.orbit_model.schwarzschild_orbit_model import RelativisticOrbitModel_Schwarzschild_EMR
from src.physics.orbit_model.kerr_orbit_model import RelativisticOrbitModel_Kerr_EMR
from src.physics.orbital_mechanics import compute_waveform


def get_true_solution_EMR_schwarzschild(u0, model_params, total_mass, tspan, tsteps, dt_data, dt):
    """
    Computes true solution of a Schwarzschild system using torchdiffeq.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mass_ratio = 0.0

    u0 = torch.tensor(u0, dtype=torch.float64, device=device)
    #tsteps = torch.tensor(tsteps, dtype=torch.float64, device=device)

    class SchwarzschildODE(torch.nn.Module):
        def forward(self, t, u):
            return torch.tensor(
                RelativisticOrbitModel_Schwarzschild_EMR(t, u, model_params),
                dtype=torch.float64,
                device=device
            )

    solver = SchwarzschildODE()
    true_solution = odeint(solver, u0, tsteps, method='dopri5')  # RK45 equivalent

    # shape: (time, state_dim) -> transpose to match original
    true_solution = true_solution.T

    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return {"true_solution": true_solution, "true_waveform": true_waveform}


def get_true_solution_EMR_kerr(u0, model_params, total_mass, tspan, tsteps, dt_data, dt):
    """
    Computes true solution of a Kerr system using torchdiffeq.
    """
    mass_ratio = 0.0

    class KerrODE(torch.nn.Module):
        def forward(self, t, u):
            return RelativisticOrbitModel_Kerr_EMR(t, u, model_params)

    solver = KerrODE()
    true_solution = odeint(solver, u0, tsteps, method='dopri5')

    true_solution = true_solution.T
    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return {"true_solution": true_solution, "true_waveform": true_waveform}
