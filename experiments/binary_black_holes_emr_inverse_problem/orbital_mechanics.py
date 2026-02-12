import torch


def _soln2orbit_batch(chi, phi, p, M, e, a):
    """
    Batched version of soln2orbit.
    chi, phi: (B, T)
    p, M, e, a: (B, 4)
    Returns orbit: (B, 2, T)
    """
    r = p * M / (1 + e * torch.cos(chi))
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return torch.stack((x, y), dim=1)


def _orbit2tensor_batch(orbit, component, mass=1.0):
    """
    Batched version of orbit2tensor.
    orbit: (B, 2, T) -> returns (B, T)
    """
    x = orbit[:, 0, :]
    y = orbit[:, 1, :]

    Ixx = x ** 2
    Iyy = y ** 2
    Ixy = x * y
    trace = Ixx + Iyy

    if component == (1, 1):
        tmp = Ixx - (1.0 / 3.0) * trace
    elif component == (2, 2):
        tmp = Iyy - (1.0 / 3.0) * trace
    else:
        tmp = Ixy

    return mass * tmp


def d2_dt2(v, dt):
    """
    Numerical second derivative using second-order one-sided difference stencils at the endpoints.
    """
    if isinstance(dt, torch.Tensor):
        dt = dt.item()
    dv_dt = torch.gradient(v, spacing=(dt,), dim=-1)[0]
    return torch.gradient(dv_dt, spacing=(dt,), dim=-1)[0]


def _smooth_1d_batch(v, window=5):
    """
    v: (B, T)
    """
    if window <= 1:
        return v
    pad = window // 2
    kernel = torch.ones(window, device=v.device, dtype=v.dtype) / window
    v_pad = torch.nn.functional.pad(v[:, None, :], (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(v_pad, kernel[None, None, :]).squeeze(1)


def _h_22_quadrupole_components_batch(dt, orbit, component, mass=1.0, smooth_window=21):
    """
    orbit: (B, 2, T)
    """
    mtensor = _orbit2tensor_batch(orbit, component, mass)
    if smooth_window and smooth_window > 1:
        mtensor = _smooth_1d_batch(mtensor, window=smooth_window)
    mtensor_ddot = d2_dt2(mtensor, dt)
    return 2 * mtensor_ddot


def _h_22_quadrupole_batch(dt, orbit, mass=1.0, smooth_window=21):
    h11 = _h_22_quadrupole_components_batch(dt, orbit, (1, 1), mass, smooth_window=smooth_window)
    h22 = _h_22_quadrupole_components_batch(dt, orbit, (2, 2), mass, smooth_window=smooth_window)
    h12 = _h_22_quadrupole_components_batch(dt, orbit, (1, 2), mass, smooth_window=smooth_window)
    return h11, h12, h22


def _h_22_strain_one_body_batch(dt, orbit, smooth_window=21):
    h11, h12, h22 = _h_22_quadrupole_batch(dt, orbit, smooth_window=smooth_window)
    h_plus = h11 - h22
    h_cross = 2.0 * h12
    scaling_const = torch.sqrt(torch.tensor(torch.pi / 5, device=orbit.device, dtype=orbit.dtype))
    return scaling_const * h_plus, -scaling_const * h_cross


def compute_waveform(
    u,
    system_params,
    smooth_window: int = 21,
):
    """
    Batched waveform computation.

    Inputs:
    - u: Tensor (B, 2, T)
    - model_params: [p, M, e, a] (B, 4)
    Returns:
    - waveform: (h_plus, h_cross) each (B, 2, T)
    """
    chi = u[:, 0, :]
    phi = u[:, 1, :]

    p = system_params['p']
    M = system_params['M']
    e = system_params['e']
    a = system_params['a']
    dt = system_params['dt_data'][0]

    orbit = _soln2orbit_batch(chi, phi, p, M, e, a)

    waveform, _ = _h_22_strain_one_body_batch(dt, orbit, smooth_window=smooth_window)

    return waveform
