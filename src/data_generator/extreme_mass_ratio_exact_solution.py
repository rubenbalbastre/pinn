from torchdiffeq import odeint

from src.physics.orbital_mechanics import compute_waveform


def get_true_solution(ode_problem, u0, model_params, tsteps, dt):
    """
    Computes true solution of a Kerr system using torchdiffeq.
    """

    true_solution = odeint(ode_problem, u0, tsteps)

    u = true_solution.T
    true_waveform, _ = compute_waveform(dt=dt, u=u, model_params=model_params)

    return {"true_solution": u, "true_waveform": true_waveform}
