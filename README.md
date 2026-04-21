# Learnable FIR Filter for Signal Denoising

Model-based deep learning approach to signal denoising using a 
trainable FIR filter implemented as a PyTorch Conv1d layer.

## Overview

**Phase 1 — Convex FIR Design (CVXPY)**  
A 41-tap linear-phase FIR filter is designed via convex optimization,
minimizing passband ripple and stopband attenuation simultaneously.

**Phase 2 — Learnable FIR (PyTorch)**  
The convex solution initializes a trainable `nn.Conv1d` layer, which 
is then fine-tuned end-to-end using Adam optimizer on synthetic 
low-frequency signals (5–12 Hz) corrupted by high-frequency and 
Gaussian noise (fs = 500 Hz).

## Results

- Learned filter achieves lower MSE than convex-only baseline
- Demonstrates generalization on unseen noisy test signals

## Tools
Python · PyTorch · CVXPY · NumPy · Matplotlib

## How to Run
```bash
pip install numpy matplotlib cvxpy torch
jupyter notebook Filter_Optimization.ipynb
```
