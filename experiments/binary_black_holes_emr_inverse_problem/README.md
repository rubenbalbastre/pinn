# Binary Black Holes EMR Inverse Problem Experiments

This folder contains experiments for learning corrections to relativistic orbital dynamics and fitting gravitational-wave strain using a neural ODE.

## Files
- `run_schwarzschild.ipynb`: End-to-end pipeline for dataset generation, model definition, training, and evaluation for the Schwarzschild case.

## Workflow Summary
1. Generate training/validation data by integrating the Schwarzschild equations of motion with a fixed-step solver and computing the quadrupole waveform.
2. Define a small neural network that outputs multiplicative corrections to the analytic RHS.
3. Train a neural ODE by minimizing L1 loss between predicted and true waveforms.
4. Evaluate the fitted model by comparing predicted vs. true waveform and by visualizing phase space (phi vs. chi).

## Training Notes
- Training uses a fixed-step ODE solver inside the forward pass. This can be slow on CPU for long sequences.
- The notebook uses RK4 for training to match evaluation behavior.
- Curriculum learning is used: start with a shorter time window and progressively extend to the full window to stabilize LBFGS and improve early convergence.
- Smoothing of the waveform is disabled during training to keep gradients sharp.

## Plots
- `plot_waveform(...)` shows the waveform over time.
- `plot_phase_space(...)` shows the space trajectory.

