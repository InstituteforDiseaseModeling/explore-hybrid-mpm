"""
Test with the user's exact configuration (774 nodes).
"""

import numpy as np

from SEIR_pde_metapop import extract_timeseries
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

print("="*70)
print("Testing User's Configuration (774 nodes)")
print("="*70)

config = StochasticModelConfig(
    n_nodes=774,
    n_age=16,
    n_bins=50,
    seed_node_idx=0,
    seed_n_infections=50.0,
    gravity_k=0.01,
    duration_days=50,  # Shorter for testing
    backend='cpu',
    use_float64=True,
    stochastic_threshold=100.0,
    tau_leap_dt=1.0,
    output_freq_days=1.0,
    stochastic_seed=45,
)

print("\nRunning simulation...")
results = run_simulation_stochastic(config, spatial_seed=40, epi_seed=1234)

# Extract timeseries
ts = extract_timeseries(results)

# Check for NaN
print("\n" + "="*70)
print("Results Check")
print("="*70)

I_total = ts['I_node'].sum(axis=0)
S_total = ts['S_node'].sum(axis=0)

print(f"Initial I: {I_total[0]:.2f}")
print(f"Peak I: {I_total.max():.2f}")
print(f"Final I: {I_total[-1]:.2f}")

if np.any(np.isnan(I_total)):
    print("\n❌ ERROR: NaN detected in I!")
    print(f"First NaN at time index: {np.where(np.isnan(I_total))[0][0]}")
else:
    print("\n✓ No NaN values detected")

if np.any(np.isnan(S_total)):
    print("❌ ERROR: NaN detected in S!")
else:
    print("✓ S is finite")

# Check if epidemic took off
if I_total.max() > 100:
    print(f"✓ Epidemic took off (peak = {I_total.max():.0f})")
else:
    print(f"⚠ Epidemic did not take off (peak = {I_total.max():.2f})")
