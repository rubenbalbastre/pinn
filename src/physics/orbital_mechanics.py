import numpy as np

def soln2orbit(sol, model_params):
    """
    Converts (chi(t), phi(t)) into (x(t), y(t)) in Cartesian coordinates.
    """
    chi, phi = sol[0], sol[1]
    p, e = sol[2, 0], sol[3, 0]
    M = model_params.get("M", 1.0)
    r = p * M / (1 + e * np.cos(chi))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y], axis=0)

def d_dt(y, dt):
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dt)
    dy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dt)
    return dy

def d2_dt2(y, dt):
    d2y = np.zeros_like(y)
    d2y[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dt**2
    d2y[0] = (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / dt**2
    d2y[-1] = (2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / dt**2
    return d2y

def compute_quadrupole_tensor(x, y):
    I_xx = x * x
    I_yy = y * y
    I_xy = x * y
    I_xx_tt = d2_dt2(I_xx, 1)
    I_yy_tt = d2_dt2(I_yy, 1)
    I_xy_tt = d2_dt2(I_xy, 1)
    return I_xx_tt, I_yy_tt, I_xy_tt

def compute_waveform(dt, sol, q, M, model_params):
    orbit = soln2orbit(sol, model_params)
    if q == 0:
        I_xx_tt, I_yy_tt, I_xy_tt = compute_quadrupole_tensor(orbit[0], orbit[1])
    else:
        mass1 = model_params["mass1"]
        mass2 = model_params["mass2"]
        orbit1, orbit2 = one2two(orbit, mass1, mass2)
        I_xx_tt_1, I_yy_tt_1, I_xy_tt_1 = compute_quadrupole_tensor(orbit1[0], orbit1[1])
        I_xx_tt_2, I_yy_tt_2, I_xy_tt_2 = compute_quadrupole_tensor(orbit2[0], orbit2[1])
        I_xx_tt = I_xx_tt_1 + I_xx_tt_2
        I_yy_tt = I_yy_tt_1 + I_yy_tt_2
        I_xy_tt = I_xy_tt_1 + I_xy_tt_2

    h_plus = (I_xx_tt - I_yy_tt) * np.sqrt(np.pi / 5.0)
    h_cross = 2 * I_xy_tt * np.sqrt(np.pi / 5.0)
    waveform = h_plus + 1j * h_cross
    return waveform, orbit

def one2two(rel_orbit, mass1, mass2):
    m_total = mass1 + mass2
    m1_ratio = mass2 / m_total
    m2_ratio = mass1 / m_total
    orbit1 = m1_ratio * rel_orbit
    orbit2 = -m2_ratio * rel_orbit
    return orbit1, orbit2
