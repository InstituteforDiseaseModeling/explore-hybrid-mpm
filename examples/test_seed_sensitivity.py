"""
Test sensitivity to seed_n_infections.
"""

from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic
from SEIR_pde_metapop import extract_timeseries
import numpy as np

print("="*70)
print("Testing Seed Infection Sensitivity")
print("="*70)

# Test different seed levels
seed_levels = [10.0, 50.0, 100.0, 200.0]

for seed_n in seed_levels:
    print(f"\n{'='*70}")
    print(f"Testing seed_n_infections = {seed_n}")
    print(f"{'='*70}")

    config = StochasticModelConfig(
        n_nodes=20,  # Smaller for faster testing
        n_age=4,
        n_bins=10,
        seed_node_idx=0,
        seed_n_infections=seed_n,
        gravity_k=0.01,
        duration_days=100,
        backend='cpu',
        use_float64=True,
        stochastic_threshold=100.0,
        tau_leap_dt=1.0,
        output_freq_days=1.0,
        stochastic_seed=42,
    )

    results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)
    ts = extract_timeseries(results)

    I_total = ts['I_node'].sum(axis=0)

    # Check regime at start
    regime = "Stochastic" if seed_n < config.stochastic_threshold else "Deterministic"

    print(f"\nResults:")
    print(f"  Initial regime: {regime}")
    print(f"  Initial I: {I_total[0]:.2f}")
    print(f"  Peak I: {I_total.max():.2f}")
    print(f"  Final I: {I_total[-1]:.2f}")

    if np.any(np.isnan(I_total)):
        print(f"  ❌ NaN detected!")
    elif I_total.max() > seed_n * 2:
        print(f"  ✓ Epidemic took off!")
    else:
        print(f"  ⚠ Epidemic faded out or stayed small")
