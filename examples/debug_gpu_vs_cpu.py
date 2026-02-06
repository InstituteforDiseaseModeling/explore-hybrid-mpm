"""
Debug script to compare GPU vs CPU intermediate values.
"""

from dataclasses import asdict

import numpy as np

from SEIR_pde_metapop import ModelConfig, run_simulation
from SEIR_pde_metapop_taichi import TaichiModelConfig, initialize_taichi, run_simulation_taichi

# Small configuration for debugging
config = ModelConfig(
    n_nodes=2,
    n_age=2,
    n_bins=5,
    gravity_k=0.01,
    duration_days=2,  # Need at least 2 days to get multiple time points
    output_freq_days=1.0,
)

print("="*70)
print("Debug: GPU vs CPU Intermediate Values")
print("="*70)

# Run CPU baseline
print("\nRunning CPU baseline...")
results_cpu = run_simulation(config, spatial_seed=42, epi_seed=123)

# Run GPU version
print("\nRunning GPU version...")
config_gpu = TaichiModelConfig(**asdict(config), backend='cpu', use_float64=True)
initialize_taichi(config_gpu)
results_gpu = run_simulation_taichi(config_gpu, spatial_seed=42, epi_seed=123)

# Compare initial conditions
print("\n" + "="*70)
print("Initial Conditions Comparison")
print("="*70)

# Extract initial state from solve_ivp
y0_cpu = results_cpu['sol'].y[:, 0]
y0_gpu = results_gpu['sol'].y[:, 0]

n_SEI = config.n_nodes * config.n_age * config.n_bins
n_R = config.n_nodes * config.n_age

S_cpu = y0_cpu[:n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
E_cpu = y0_cpu[n_SEI:2*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
I_cpu = y0_cpu[2*n_SEI:3*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
R_cpu = y0_cpu[3*n_SEI:].reshape((config.n_nodes, config.n_age))

S_gpu = y0_gpu[:n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
E_gpu = y0_gpu[n_SEI:2*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
I_gpu = y0_gpu[2*n_SEI:3*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
R_gpu = y0_gpu[3*n_SEI:].reshape((config.n_nodes, config.n_age))

print("\nInitial state comparison:")
print(f"  S max diff: {np.abs(S_cpu - S_gpu).max():.2e}")
print(f"  E max diff: {np.abs(E_cpu - E_gpu).max():.2e}")
print(f"  I max diff: {np.abs(I_cpu - I_gpu).max():.2e}")
print(f"  R max diff: {np.abs(R_cpu - R_gpu).max():.2e}")

print("\nI compartment (seed node, age 0):")
print(f"  CPU: {I_cpu[0, 0, :]}")
print(f"  GPU: {I_gpu[0, 0, :]}")

# Compare derivatives at t=0
print("\n" + "="*70)
print("Derivatives at t=0")
print("="*70)

# Get derivatives from both versions
dydt_cpu = results_cpu['sol'].y[:, 1] - results_cpu['sol'].y[:, 0]  # Approximate
dydt_gpu = results_gpu['sol'].y[:, 1] - results_gpu['sol'].y[:, 0]

dSdt_cpu = dydt_cpu[:n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
dEdt_cpu = dydt_cpu[n_SEI:2*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
dIdt_cpu = dydt_cpu[2*n_SEI:3*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
dRdt_cpu = dydt_cpu[3*n_SEI:].reshape((config.n_nodes, config.n_age))

dSdt_gpu = dydt_gpu[:n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
dEdt_gpu = dydt_gpu[n_SEI:2*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
dIdt_gpu = dydt_gpu[2*n_SEI:3*n_SEI].reshape((config.n_nodes, config.n_age, config.n_bins))
dRdt_gpu = dydt_gpu[3*n_SEI:].reshape((config.n_nodes, config.n_age))

print("\nDerivatives comparison:")
print(f"  dSdt max diff: {np.abs(dSdt_cpu - dSdt_gpu).max():.2e}")
print(f"  dEdt max diff: {np.abs(dEdt_cpu - dEdt_gpu).max():.2e}")
print(f"  dIdt max diff: {np.abs(dIdt_cpu - dIdt_gpu).max():.2e}")
print(f"  dRdt max diff: {np.abs(dRdt_cpu - dRdt_gpu).max():.2e}")

# Show where the differences are largest
print("\ndSdt (node 0, age 0):")
print(f"  CPU: {dSdt_cpu[0, 0, :]}")
print(f"  GPU: {dSdt_gpu[0, 0, :]}")
print(f"  Diff: {dSdt_cpu[0, 0, :] - dSdt_gpu[0, 0, :]}")

print("\ndEdt (node 0, age 0):")
print(f"  CPU: {dEdt_cpu[0, 0, :]}")
print(f"  GPU: {dEdt_gpu[0, 0, :]}")
print(f"  Diff: {dEdt_cpu[0, 0, :] - dEdt_gpu[0, 0, :]}")

# Check if P matrix is the issue
print("\n" + "="*70)
print("P Matrix Check")
print("="*70)
P_cpu = results_cpu['aux']['P']
P_gpu = results_gpu['aux']['P']

print(f"P matrix shape: {P_cpu.shape}")
print(f"P max diff: {np.abs(P_cpu - P_gpu).max():.2e}")
print("\nP matrix (CPU):")
print(P_cpu)

# Manually compute P.T @ new_infections to verify
theta_vals = results_cpu['aux']['theta_vals']
foi_manual = 1e-6  # Small value for testing
new_inf_manual = theta_vals * foi_manual * S_cpu[0, 0, :]
new_E_baseline = P_cpu.T @ new_inf_manual
new_E_gpu_style = np.zeros(config.n_bins)
for theta_bin in range(config.n_bins):
    for phi_bin in range(config.n_bins):
        new_E_gpu_style[phi_bin] += P_cpu[theta_bin, phi_bin] * new_inf_manual[theta_bin]

print("\nManual P.T @ new_infections comparison:")
print(f"  Baseline (P.T @ x): {new_E_baseline}")
print(f"  GPU style (sum P[i,j]*x[i]): {new_E_gpu_style}")
print(f"  Difference: {new_E_baseline - new_E_gpu_style}")
print(f"  Max diff: {np.abs(new_E_baseline - new_E_gpu_style).max():.2e}")
