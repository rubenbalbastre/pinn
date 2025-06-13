import torch

def soln2orbit(soln, M, model_params=None):
    """
    Performs change of variables:
    (χ(t), ϕ(t)) ↦ (x(t), y(t))
    """
    size_soln = soln.shape[0]

    if size_soln == 2:  # EMR case
        p, M_, e, a = model_params
        χ = soln[0, :]
        ϕ = soln[1, :]
    elif size_soln == 4:  # non-EMR case
        χ = soln[0, :]
        ϕ = soln[1, :]
        p = soln[2, :]
        e = soln[3, :]
    else:
        raise ValueError("soln.shape[0] must be either 2 or 4")

    r = p * M / (1 + e * torch.cos(χ))
    x = r * torch.cos(ϕ)
    y = r * torch.sin(ϕ)

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
    a = 2 * v[0] - 5 * v[1] + 4 * v[2] - v[3]
    b = v[:-2] - 2 * v[1:-1] + v[2:]
    c = 2 * v[-1] - 5 * v[-2] + 4 * v[-3] - v[-4]
    return torch.cat((torch.tensor([a], device=v.device), b, torch.tensor([c], device=v.device))) / (dt ** 2)


def h_22_quadrupole_components(dt, orbit, component, mass=1.0):
    """
    x(t) and y(t) inputs are the trajectory of the orbiting BH.
    WARNING: assuming x and y are on a uniform grid of spacing dt
    """
    mtensor = orbit2tensor(orbit, component, mass)
    mtensor_ddot = d2_dt2(mtensor, dt)
    return 2 * mtensor_ddot


def h_22_quadrupole(dt, orbit, mass=1.0):
    """
    Returns h_22 quadrupole components from orbit
    """
    h11 = h_22_quadrupole_components(dt, orbit, (1, 1), mass)
    h22 = h_22_quadrupole_components(dt, orbit, (2, 2), mass)
    h12 = h_22_quadrupole_components(dt, orbit, (1, 2), mass)
    return h11, h12, h22


def h_22_strain_one_body(dt, orbit):
    h11, h12, h22 = h_22_quadrupole(dt, orbit)
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


def compute_waveform(dt, soln, mass_ratio, total_mass, model_params):
    """
    Calculate Waveform from solution
    """
    assert mass_ratio >= 0.0, "mass_ratio must be non-negative"

    orbit = soln2orbit(soln, total_mass, model_params)
    if mass_ratio > 0:
        mass1 = total_mass * mass_ratio / (1.0 + mass_ratio)
        mass2 = total_mass / (1.0 + mass_ratio)
        orbit1, orbit2 = one2two(orbit, mass1, mass2)
        waveform = h_22_strain_two_body(dt, orbit1, mass1, orbit2, mass2)
    else:
        waveform = h_22_strain_one_body(dt, orbit)

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
