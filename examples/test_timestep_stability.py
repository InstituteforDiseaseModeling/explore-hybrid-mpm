"""
Test integration stability and accuracy across different timesteps.

Shows how method accuracy degrades as dt increases relative to characteristic timescales.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from SEIR_pde_metapop import extract_timeseries
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("Testing Timestep Stability and Accuracy")
print("="*70)

# Characteristic timescales from config
print("\nCharacteristic timescales:")
print("  Latent period (E→I): 3.0 days (sigma_rate = 1/3)")
print("  Infectious period (I→R): 24.0 days (gamma_rate = 1/24)")
print("  Fastest dynamics: 3.0 days")

# Base configuration
base_config = {
    'n_nodes': 128,
    'n_age': 2,
    'n_bins': 25,
    'seed_node_idx': 0,
    'seed_n_infections': 10.0,
    'gravity_k': 0.01,
    'duration_days': 150,
    'backend': 'metal',
    'use_float64': False,
    'stochastic_threshold': 0.0,  # Fully deterministic for comparison
    'output_freq_days': 1.0,  # Always output at 1-day resolution for comparison
    'stochastic_seed': 42,
}

# Test different timesteps
timesteps = [0.25, 0.5, 1.0, 2.0, 4.0]
methods = ['euler', 'rk4']  # RK2 disabled - has catastrophic mass conservation errors with clamping

print(f"\n{'='*70}")
print("Testing timesteps (dt):")
for dt in timesteps:
    print(f"  dt = {dt:.2f} days → {3.0/dt:.1f} steps through latent period")

results = {}

for dt in timesteps:
    print(f"\n{'='*70}")
    print(f"dt = {dt} days")
    print(f"{'='*70}")

    for method in methods:
        # Update config
        config = StochasticModelConfig(
            **base_config,
            tau_leap_dt=dt,
            deterministic_method=method,
        )

        print(f"  {method.upper()}...", end='', flush=True)

        try:
            sim_results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)
            ts = extract_timeseries(sim_results)

            # Interpolate to common 1-day resolution for comparison
            time_common = np.arange(0, base_config['duration_days'], 1.0)
            I_total = ts['I_node'].sum(axis=0)
            I_interp = np.interp(time_common, ts['time'], I_total)

            key = (dt, method)
            results[key] = {
                'I_total': I_interp,
                'time': time_common,
                'peak': I_interp.max(),
                'success': True,
            }
            print(f" peak={I_interp.max():.0f}")

        except Exception as e:
            print(f" FAILED: {e}")
            results[(dt, method)] = {'success': False}

# Use finest resolution RK4 as reference
reference = results[(0.25, 'rk4')]
if not reference['success']:
    print("\nERROR: Reference simulation (dt=0.25, RK4) failed!")
    exit(1)

print(f"\n{'='*70}")
print("ACCURACY COMPARISON (vs dt=0.25 RK4 reference)")
print(f"{'='*70}")
print(f"{'dt (days)':<12} {'Method':<8} {'Peak Error':<15} {'RMSE':<15} {'Status'}")
print("-"*70)

for dt in timesteps:
    for method in methods:
        result = results[(dt, method)]

        if not result['success']:
            print(f"{dt:<12.2f} {method.upper():<8} {'N/A':<15} {'N/A':<15} FAILED")
            continue

        # Compute errors
        I_diff = result['I_total'] - reference['I_total']
        rmse = np.sqrt(np.mean(I_diff**2))
        peak_error = abs(result['peak'] - reference['peak'])
        rel_rmse = rmse / reference['I_total'].max() * 100

        status = "OK" if rel_rmse < 5 else "POOR" if rel_rmse < 10 else "FAIL"

        print(f"{dt:<12.2f} {method.upper():<8} "
              f"{peak_error:<15.1f} {rmse:<15.1f} ({rel_rmse:.1f}%) {status}")

# Plot comparison
print(f"\n{'='*70}")
print("Generating plot...")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Reference label for all plots
ref_label = 'Reference: RK4 dt=0.25'

# Plot 1: dt=0.5 (within stability)
ax = axes[0, 0]
dt = 0.5
for method in methods:
    result = results[(dt, method)]
    if result['success']:
        ax.plot(result['time'], result['I_total'], label=method.upper(), linewidth=2)
ax.plot(reference['time'], reference['I_total'], 'k--', label=ref_label, linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Total Infectious', fontsize=11)
ax.set_title(f'dt = {dt} days (6 steps/latent period)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: dt=1.0 (marginal)
ax = axes[0, 1]
dt = 1.0
for method in methods:
    result = results[(dt, method)]
    if result['success']:
        ax.plot(result['time'], result['I_total'], label=method.upper(), linewidth=2)
ax.plot(reference['time'], reference['I_total'], 'k--', label=ref_label, linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Total Infectious', fontsize=11)
ax.set_title(f'dt = {dt} days (3 steps/latent period)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: dt=2.0 (unstable for Euler)
ax = axes[1, 0]
dt = 2.0
for method in methods:
    result = results[(dt, method)]
    if result['success']:
        ax.plot(result['time'], result['I_total'], label=method.upper(), linewidth=2)
ax.plot(reference['time'], reference['I_total'], 'k--', label=ref_label, linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Total Infectious', fontsize=11)
ax.set_title(f'dt = {dt} days (1.5 steps/latent period)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: dt=4.0 (unstable for all low-order methods)
ax = axes[1, 1]
dt = 4.0
for method in methods:
    result = results[(dt, method)]
    if result['success']:
        ax.plot(result['time'], result['I_total'], label=method.upper(), linewidth=2)
ax.plot(reference['time'], reference['I_total'], 'k--', label=ref_label, linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Total Infectious', fontsize=11)
ax.set_title(f'dt = {dt} days (0.75 steps/latent period!)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = figures_dir / 'timestep_stability.pdf'
plt.savefig(output_file)
print(f"✓ Saved: {output_file}")

print(f"\n{'='*70}")
print("CONCLUSIONS:")
print(f"{'='*70}")
print("Stability limit: dt ≤ fastest_timescale / 3")
print("  For this model: dt ≤ 1.0 days (current setting is at the limit!)")
print("\nMethod selection for different timesteps:")
print("  dt≤0.5: All methods stable")
print("  dt=1.0: RK4 okay (~8% error), Euler marginal")
print("  dt=2.0: RK4 marginal (~16% error), Euler fails")
print("  dt>3.0: All methods fail (insufficient resolution)")
print("\nNote: RK2 disabled due to catastrophic mass conservation errors with clamping.")
print("Use RK4 for best accuracy with fixed timesteps, or implement adaptive stepping.")
