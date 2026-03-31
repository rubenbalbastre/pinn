# PINNs Portfolio: Inverse PDE Learning

Repository of Physics-Informed Neural Network (PINN) experiments focused on inverse problems, parameter discovery, and physics-constrained learning.

## Why This Repo

- End-to-end implementations (dataset generation, model, loss design, training, plotting).
- Mix of canonical PDEs and domain problems (quant finance and relativistic dynamics).
- Explicit attention to practical issues in inverse PINNs (identifiability, loss balancing, regularization).

## Tech Stack

- Python, PyTorch
- Autograd-based PDE residuals
- Numerical ODE/PDE solvers
- Matplotlib, Jupyter notebooks

## Experiments

### 1) Toy Problem: 1D Poisson

- Path: `toy_problem/`
- Task: recover `u(x)` under PDE + boundary constraints.
- Value: minimal PINN baseline to validate residual/boundary loss design.
- Entry point: `python3 toy_problem/train.py`
- Details: [toy_problem/README.md](toy_problem/README.md)

### 2) Diffusion Equation (Inverse Coefficient)

- Path: `diffusion_equation/`
- Task: recover `u(x,t)` and infer spatial diffusion coefficient `alpha(x)`.
- Value: demonstrates a real inverse challenge where field fit can succeed while coefficient recovery remains hard.
- Entry point: `python3 diffusion_equation/train.py`
- Details: [diffusion_equation/README.md](diffusion_equation/README.md)

### 3) Options Pricing (Black-Scholes PINN)

- Path: `options/`
- Task: solve Black-Scholes with physics constraints and compare with analytic solution.
- Value: finance-oriented PDE modeling with data + PDE + boundary losses.
- Entry point: `python3 options/train.py`
- Details: [options/README.md](options/README.md)

### 4) BBH EMR (Neural ODE for Orbital Dynamics)

- Path: `bbh_emr/`
- Task: learn corrections to relativistic orbital dynamics and fit waveform signals.
- Value: continuous-time modeling with fixed-step solvers and curriculum-style training.
- Entry points:
  - `bbh_emr/run_schwarzschild.ipynb`
  - `bbh_emr/run_kerr.ipynb`
- Details: [bbh_emr/README.md](bbh_emr/README.md)

## Quick Start

Recommended environment: Python 3.12+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Developer Notes

- Scope: research-style prototypes with clear separation between physics, models, and optimization.
- Summary:
  - translating PDE/ODE physics into differentiable training objectives
  - building custom inverse-learning pipelines
  - diagnosing failure modes (for example, non-identifiable coefficients)
- Code intent: readable and modular, favoring experiment iteration over framework complexity.
