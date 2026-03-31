# Physics-Informed Neural Networks (PINNs)

This repo collects PINN experiments for inverse PDE problems.

## Problem Types
- All problems are solved as inverse problems.
- All problems except **BBH EMR** are **discrete inverse problems**.
- **BBH EMR** is solved in **both discrete and continuous** inverse forms.

## Experiments
- `toy_problem/` 1D Poisson Equation
- `bbh_emr/` Extreme Mass Ratio (EMR) Binary Black Holes (BBH)
- `diffusion_equation/` 1D Diffusion Equation
- `options/` Black-Scholes PDE
- `multi_physics/` [pending]

## Setup
Use Python 3.12.3 and install dependencies:

```bash
pip install -r requirements.txt
```

