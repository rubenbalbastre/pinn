import torch

def soln2orbit(chi, phi, p, M, e, a):
    """
    Performs change of variables:
    (χ(t), ϕ(t)) ↦ (x(t), y(t))
    """

    r = p * M / (1 + e * torch.cos(chi))
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)

    orbit = torch.vstack((x, y))

    return orbit


def orbit2tensor(orbit, component, mass=1.0):
    """
    Construct trace-free moment tensor Ι(t) for orbit from BH orbit (x(t), y(t)).

    component defines the Cartesian indices in x,y. For example,
    I_{22} is the yy component of the moment tensor.
    """
    x = orbit[0, :]
    y = orbit[1, :]

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


def d_dt(v, dt):
    """
    Numerical derivative using second-order one-sided difference stencils at the endpoints.
    """
    a = (-1.5 * v[0] + 2 * v[1] - 0.5 * v[2])
    b = (v[2:] - v[:-2]) / 2
    c = (1.5 * v[-1] - 2 * v[-2] + 0.5 * v[-3])
    return torch.cat((torch.tensor([a], device=v.device), b, torch.tensor([c], device=v.device))) / dt


def d2_dt2(v, dt):
    """
    Numerical second derivative using second-order one-sided difference stencils at the endpoints.
    """
    if isinstance(dt, torch.Tensor):
        dt = dt.item()
    dv_dt = torch.gradient(v, spacing=(dt,))[0]
    return torch.gradient(dv_dt, spacing=(dt,))[0]


def _smooth_1d(v, window=5):
    if window <= 1:
        return v
    pad = window // 2
    kernel = torch.ones(window, device=v.device, dtype=v.dtype) / window
    v_pad = torch.nn.functional.pad(v[None, None, :], (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(v_pad, kernel[None, None, :])[0, 0]


def h_22_quadrupole_components(dt, orbit, component, mass=1.0, smooth_window=21):
    """
    x(t) and y(t) inputs are the trajectory of the orbiting BH.
    WARNING: assuming x and y are on a uniform grid of spacing dt
    """
    mtensor = orbit2tensor(orbit, component, mass)
    if smooth_window and smooth_window > 1:
        mtensor = _smooth_1d(mtensor, window=smooth_window)
    mtensor_ddot = d2_dt2(mtensor, dt)
    return 2 * mtensor_ddot


def h_22_quadrupole(dt, orbit, mass=1.0, smooth_window=21):
    """
    Returns h_22 quadrupole components from orbit
    """
    h11 = h_22_quadrupole_components(dt, orbit, (1, 1), mass, smooth_window=smooth_window)
    h22 = h_22_quadrupole_components(dt, orbit, (2, 2), mass, smooth_window=smooth_window)
    h12 = h_22_quadrupole_components(dt, orbit, (1, 2), mass, smooth_window=smooth_window)
    return h11, h12, h22


def h_22_strain_one_body(dt, orbit, smooth_window=21):

    h11, h12, h22 = h_22_quadrupole(dt, orbit, smooth_window=smooth_window)

    h_plus = h11 - h22
    h_cross = 2.0 * h12

    scaling_const = torch.sqrt(torch.tensor(torch.pi / 5))
    
    return scaling_const * h_plus, -scaling_const * h_cross


def h_22_quadrupole_two_body(dt, orbit1, mass1, orbit2, mass2):
    """
    Returns h_22 quadrupole components from orbit for 2 body problem
    """
    h11_1, h12_1, h22_1 = h_22_quadrupole(dt, orbit1, mass1)
    h11_2, h12_2, h22_2 = h_22_quadrupole(dt, orbit2, mass2)
    h11 = h11_1 + h11_2
    h12 = h12_1 + h12_2
    h22 = h22_1 + h22_2
    return h11, h12, h22


def h_22_strain_two_body(dt, orbit1, mass1, orbit2, mass2):
    """
    compute (2,2) mode strain from orbits of BH 1 of mass1 and BH2 of mass 2
    """
    h11, h12, h22 = h_22_quadrupole_two_body(dt, orbit1, mass1, orbit2, mass2)
    h_plus = h11 - h22
    h_cross = 2.0 * h12
    scaling_const = torch.sqrt(torch.tensor(torch.pi / 5))
    return scaling_const * h_plus, -scaling_const * h_cross


def one2two(path, m1, m2):
    """
    We need a very crude 2-body path.
    Assume the 1-body motion is a Newtonian 2-body position vector r = r1 - r2
    and use Newtonian formulas to get r1, r2
    """
    M = m1 + m2
    r1 = m2 / M * path
    r2 = -m1 / M * path
    return r1, r2


def _split_u_batch(u: torch.Tensor):
    """
    Accepts u as (B, 2, T) or (2, B, T) and returns (chi, phi) as (B, T).
    """
    if u.ndim != 3:
        raise ValueError(f"Expected u to have 3 dims, got shape {tuple(u.shape)}")
    if u.shape[1] == 2:
        chi = u[:, 0, :]
        phi = u[:, 1, :]
    elif u.shape[0] == 2:
        chi = u[0, :, :]
        phi = u[1, :, :]
    else:
        raise ValueError(f"Expected u shape (B,2,T) or (2,B,T), got {tuple(u.shape)}")
    return chi, phi


def _soln2orbit_batch(chi, phi, p, M, e, a):
    """
    Batched version of soln2orbit.
    chi, phi: (B, T)
    p, M, e, a: scalars or (B,) or (B, 1)
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


def _d_dt_batch(v, dt):
    """
    Batched numerical derivative along last dim using second-order one-sided stencils.
    v: (B, T) -> returns (B, T)
    """
    a = (-1.5 * v[:, 0] + 2 * v[:, 1] - 0.5 * v[:, 2])
    b = (v[:, 2:] - v[:, :-2]) / 2
    c = (1.5 * v[:, -1] - 2 * v[:, -2] + 0.5 * v[:, -3])
    return torch.cat((a[:, None], b, c[:, None]), dim=1) / dt


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


def _h_22_quadrupole_two_body_batch(dt, orbit1, mass1, orbit2, mass2):
    h11_1, h12_1, h22_1 = _h_22_quadrupole_batch(dt, orbit1, mass1)
    h11_2, h12_2, h22_2 = _h_22_quadrupole_batch(dt, orbit2, mass2)
    h11 = h11_1 + h11_2
    h12 = h12_1 + h12_2
    h22 = h22_1 + h22_2
    return h11, h12, h22


def _h_22_strain_two_body_batch(dt, orbit1, mass1, orbit2, mass2):
    h11, h12, h22 = _h_22_quadrupole_two_body_batch(dt, orbit1, mass1, orbit2, mass2)
    h_plus = h11 - h22
    h_cross = 2.0 * h12
    scaling_const = torch.sqrt(torch.tensor(torch.pi / 5, device=orbit1.device, dtype=orbit1.dtype))
    return scaling_const * h_plus, -scaling_const * h_cross


def compute_waveform(
    dt,
    u,
    model_params,
    mass_ratio: float = 0.0,
    smooth_window: int = 21,
):
    """
    Batched waveform computation.

    Inputs:
    - u: Tensor (B, 2, T) or (2, B, T), or tuple/list (chi, phi) where each is (B, T)
    - model_params: [p, M, e, a] with each scalar or (B,) or (B, 1)
    Returns:
    - waveform: (h_plus, h_cross) each (B, T)
    """
    if isinstance(u, (tuple, list)):
        if len(u) != 2:
            raise ValueError("u tuple/list must be (chi, phi)")
        chi, phi = u
    else:
        chi, phi = _split_u_batch(u)

    p, M, e, a = model_params

    orbit = _soln2orbit_batch(chi, phi, p, M, e, a)

    if mass_ratio > 0:
        mass1 = M * mass_ratio / (1.0 + mass_ratio)
        mass2 = M / (1.0 + mass_ratio)
        orbit1, orbit2 = one2two(orbit, mass1, mass2)
        waveform = _h_22_strain_two_body_batch(dt, orbit1, mass1, orbit2, mass2)
    else:
        waveform = _h_22_strain_one_body_batch(dt, orbit, smooth_window=smooth_window)

    return waveform


def interpolate_time_series(tsteps, tdata, fdata):
    """
    Interpolate time series to adapt the waveform length.

    Assumes tsteps, tdata, fdata are 1D torch tensors.
    """
    assert tdata.numel() == fdata.numel(), "lengths of tdata and fdata must match"

    interp_fdata = torch.zeros(len(tsteps), device=tdata.device)

    for j, tj in enumerate(tsteps):
        # Find interval in tdata that contains tj
        for i in range(len(tdata) - 1):
            if tdata[i] <= tj < tdata[i + 1]:
                weight = (tj - tdata[i]) / (tdata[i + 1] - tdata[i])
                interp_fdata[j] = (1 - weight) * fdata[i] + weight * fdata[i + 1]
                break
    return interp_fdata


# Note: file reading functions would typically use python file I/O and 
# convert data to torch tensors after reading.

def file2waveform(tsteps, filename="waveform.txt"):
    """
    Reads waveform data file and interpolates to tsteps.
    """
    data = torch.tensor(torch.loadtxt(filename))  # Replace with appropriate PyTorch file reading if needed
    tdata = data[:, 0]
    wdata = data[:, 1]

    waveform = interpolate_time_series(tsteps, tdata, wdata)
    return waveform


def file2trajectory(tsteps, filename="trajectoryA.txt"):
    """
    Reads trajectory data file and interpolates to tsteps.
    """
    data = torch.tensor(torch.loadtxt(filename))  # Replace with appropriate PyTorch file reading if needed
    tdata = data[:, 0]
    xdata = data[:, 1]
    ydata = data[:, 2]

    x = interpolate_time_series(tsteps, tdata, xdata)
    y = interpolate_time_series(tsteps, tdata, ydata)

    return x, y
