import torch


def solve_ode_rk4(ode_problem, system_params):
    """
    Computes true solution of a Kerr system using an explicit RK4 integrator.
    This keeps the solution fully differentiable with torch autograd.
    """

    u = system_params['u0']
    us = [u]
    tsteps = system_params['tsteps'][0, :]

    for i in range(1, tsteps.numel()):
        t_prev = tsteps[i - 1]
        dt_i = tsteps[i] - t_prev

        k1 = ode_problem(t_prev, u)
        k2 = ode_problem(t_prev + 0.5 * dt_i, u + 0.5 * dt_i * k1)
        k3 = ode_problem(t_prev + 0.5 * dt_i, u + 0.5 * dt_i * k2)
        k4 = ode_problem(t_prev + dt_i, u + dt_i * k3)

        u = u + (dt_i / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        us.append(u)

    u = torch.stack(us, dim=-1)
    return u


def solve_ode_rk2(ode_problem, system_params):
    """
    Midpoint (RK2) integrator.
    Fully differentiable with torch autograd.
    """
    u = system_params['u0']
    us = [u]
    tsteps = system_params['tsteps'][0, :]

    for i in range(1, tsteps.numel()):
        t_prev = tsteps[i - 1]
        dt_i = tsteps[i] - t_prev

        k1 = ode_problem(t_prev, u)
        u_mid = u + 0.5 * dt_i * k1
        k2 = ode_problem(t_prev + 0.5 * dt_i, u_mid)

        u = u + dt_i * k2
        us.append(u)

    return torch.stack(us, dim=-1)
