# Physics-Informed Neural Networks (PINNs)

This repo collects PINN experiments for inverse PDE problems. All problems are formulated and solved in inverse form.

## Problem Types
- All problems are solved as inverse problems.
- All problems except **BBH EMR** are **discrete inverse problems**.
- **BBH EMR** is solved in **both discrete and continuous** inverse forms.

## Experiments
The main notebooks live under `experiments/`, organized by problem:
- `experiments/toy_problem/`
- `experiments/diffusion_equation/`
- `experiments/options/`
- `experiments/multi_physics/`
- `experiments/binary_black_holes_emr_inverse_problem/` (BBH EMR)

## Setup
Use Python 3.12.3 and install dependencies:

```bash
pip install -r requirements.txt
```

