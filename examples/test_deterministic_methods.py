"""
Test different deterministic integration methods (Euler, RK2, RK4).

Compares speed and accuracy of the three methods for fully deterministic
simulations (threshold=0).
"""

import time
import numpy as np

from SEIR_pde_metapop import extract_timeseries
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

print("="*70)
print("Testing Deterministic Integration Methods")
print("="*70)

# Base configuration (fully deterministic with threshold=0)
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
    'stochastic_threshold': 0.0,  # Fully deterministic
    'tau_leap_dt': 1.0,
    'output_freq_days': 1.0,
    'stochastic_seed': 42,
}

methods = ['euler', 'rk2', 'rk4']
results_dict = {}

for method in methods:
    print(f"\n{'='*70}")
    print(f"Testing: {method.upper()}")
    print(f"{'='*70}")

    config = StochasticModelConfig(**base_config, deterministic_method=method)

    t0 = time.time()
    results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)
    runtime = time.time() - t0

    ts = extract_timeseries(results)
    I_total = ts['I_node'].sum(axis=0)

    results_dict[method] = {
        'runtime': runtime,
        'I_total': I_total,
        'peak': I_total.max(),
        'time': ts['time'],
    }

    print(f"\nResults:")
    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Peak infections: {I_total.max():.0f}")
    print(f"  Final infections: {I_total[-1]:.0f}")

# Summary comparison
print(f"\n{'='*70}")
print("SUMMARY COMPARISON")
print(f"{'='*70}")
print(f"{'Method':<10} {'Runtime (s)':<15} {'Peak I':<15} {'Speedup vs RK4'}")
print("-"*70)

rk4_time = results_dict['rk4']['runtime']
for method in methods:
    speedup = rk4_time / results_dict[method]['runtime']
    print(f"{method.upper():<10} {results_dict[method]['runtime']:<15.2f} "
          f"{results_dict[method]['peak']:<15.0f} {speedup:.2f}×")

# Compare accuracy: compute difference from RK4 (most accurate)
print(f"\n{'='*70}")
print("ACCURACY COMPARISON (vs RK4)")
print(f"{'='*70}")

rk4_I = results_dict['rk4']['I_total']
for method in ['euler', 'rk2']:
    I_diff = results_dict[method]['I_total'] - rk4_I
    rmse = np.sqrt(np.mean(I_diff**2))
    max_abs_error = np.abs(I_diff).max()
    rel_error = rmse / rk4_I.max() * 100

    print(f"\n{method.upper()}:")
    print(f"  RMSE: {rmse:.2f} infections")
    print(f"  Max absolute error: {max_abs_error:.2f} infections")
    print(f"  Relative error: {rel_error:.3f}%")

print(f"\n{'='*70}")
print("CONCLUSIONS:")
print(f"{'='*70}")
print(f"- Euler: {results_dict['euler']['runtime']/rk4_time:.1f}× faster, but with accuracy tradeoffs")
print(f"- RK2: {results_dict['rk2']['runtime']/rk4_time:.1f}× faster, good accuracy/speed balance")
print(f"- RK4: Most accurate, but {rk4_time/results_dict['euler']['runtime']:.1f}× slower than Euler")
print("\nRecommendation: Use RK2 for dt=1.0 days (good balance)")
