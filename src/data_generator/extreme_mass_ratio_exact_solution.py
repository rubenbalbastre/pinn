import numpy as np
from scipy.integrate import solve_ivp

from src.physics.orbit_model.schwarzschild_orbit_model import RelativisticOrbitModel_Schwarzschild_EMR
from src.physics.orbit_model.kerr_orbit_model import RelativisticOrbitModel_Kerr_EMR
from src.physics.orbital_mechanics import compute_waveform


def get_true_solution_EMR_schwarzschild(u0, model_params, total_mass, tspan, tsteps, dt_data, dt):
    """
    Computes true solution of a Schwarzschild system in Kerr metric.
    """

    mass_ratio = 0.0

    def ode_system(t, u):
        return RelativisticOrbitModel_Schwarzschild_EMR(t, u, model_params)

    # Using fixed-step RK4 integration
    # Scipy's solve_ivp can be forced to RK45 or RK23 but not RK4 directly,
    # so use RK45 and set max_step=dt for fixed step approx.
    sol = solve_ivp(
        ode_system,
        tspan,
        u0,
        t_eval=tsteps,
        method='RK45',
        max_step=dt
    )

    true_solution = sol.y  # shape: (state_dim, time_steps)

    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return {"true_solution": true_solution, "true_waveform": true_waveform}


def get_true_solution_EMR_kerr(u0, model_params, total_mass, tspan, tsteps, dt_data, dt):
    """
    Computes true solution of an EMR system in Kerr metric.
    """

    mass_ratio = 0.0

    def ode_system(t, u):
        return RelativisticOrbitModel_Kerr_EMR(t, u, model_params)

    # Using fixed-step RK4 approximation with max_step=dt
    sol = solve_ivp(
        fun=ode_system,
        t_span=tspan,
        y0=u0,
        t_eval=tsteps,
        method='RK45',
        max_step=dt
    )

    true_solution = sol.y

    true_waveform, _ = compute_waveform(dt_data, true_solution, mass_ratio, total_mass, model_params)

    return {"true_solution": true_solution, "true_waveform": true_waveform}
