"""
Validation tests for hybrid stochastic SEIR model.

Tests:
1. Pure deterministic mode (threshold=∞) should approximate baseline
2. Stochastic mode shows higher variance than deterministic
3. Conservation: S+E+I+R = N_total throughout
4. Fade-out behavior in small nodes
"""

from dataclasses import replace

import numpy as np

from SEIR_pde_metapop import ModelConfig, extract_timeseries, run_simulation
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

print("="*70)
print("Validation Tests for Hybrid Stochastic SEIR")
print("="*70)

# ============================================================================
# Test 1: Pure Deterministic Mode (threshold=∞)
# ============================================================================

print("\n" + "="*70)
print("Test 1: Pure Deterministic Mode (threshold=∞)")
print("="*70)
print("Testing if stochastic model with threshold=∞ approximates baseline...")

config_baseline = ModelConfig(
    n_nodes=5,
    n_age=4,
    n_bins=10,
    gravity_k=0.01,
    duration_days=100,
    output_freq_days=1.0,
)

config_stoch = StochasticModelConfig(
    n_nodes=5,
    n_age=4,
    n_bins=10,
    gravity_k=0.01,
    duration_days=100,
    output_freq_days=1.0,
    stochastic_threshold=1e10,  # Infinite threshold → always deterministic
    tau_leap_dt=0.5,
    backend='cpu',  # Use CPU for better reproducibility
    use_float64=True,
)

print("\nRunning baseline deterministic simulation...")
results_baseline = run_simulation(config_baseline, spatial_seed=42, epi_seed=123)

print("\nRunning stochastic simulation in deterministic mode...")
results_stoch = run_simulation_stochastic(config_stoch, spatial_seed=42, epi_seed=123)

# Extract timeseries
ts_baseline = extract_timeseries(results_baseline)
ts_stoch = extract_timeseries(results_stoch)

# Compare total infectious counts
I_baseline_total = ts_baseline['I_node'].sum(axis=0)  # Sum over nodes
I_stoch_total = ts_stoch['I_node'].sum(axis=0)

# Compute relative difference
rel_diff = np.abs(I_baseline_total - I_stoch_total) / (I_baseline_total + 1e-10)
max_rel_diff = rel_diff.max()
mean_rel_diff = rel_diff.mean()

print("\nComparison of total infectious counts:")
print(f"  Max relative difference: {max_rel_diff:.2e}")
print(f"  Mean relative difference: {mean_rel_diff:.2e}")

if max_rel_diff < 0.05:  # 5% tolerance (due to dt differences)
    print("  ✓ PASS: Deterministic mode matches baseline within 5%")
else:
    print(f"  ✗ FAIL: Deterministic mode differs by {max_rel_diff:.1%}")

# ============================================================================
# Test 2: Conservation (S+E+I+R = N_total)
# ============================================================================

print("\n" + "="*70)
print("Test 2: Conservation of Total Population")
print("="*70)

# Run a small stochastic simulation
config_conserv = StochasticModelConfig(
    n_nodes=5,
    n_age=4,
    n_bins=10,
    gravity_k=0.01,
    duration_days=50,
    output_freq_days=1.0,
    stochastic_threshold=10.0,  # Mix of stochastic and deterministic
    tau_leap_dt=0.5,
    backend='cpu',
)

results_conserv = run_simulation_stochastic(config_conserv, spatial_seed=42, epi_seed=123)
ts_conserv = extract_timeseries(results_conserv)

# Check conservation at each time point
S_total = ts_conserv['S_node'].sum(axis=0)  # Sum over nodes
E_total = ts_conserv['E_node'].sum(axis=0)
I_total = ts_conserv['I_node'].sum(axis=0)
R_total = ts_conserv['R_node'].sum(axis=0)

pop_total = S_total + E_total + I_total + R_total
N_expected = config_conserv.N_total

conservation_error = np.abs(pop_total - N_expected) / N_expected
max_error = conservation_error.max()
mean_error = conservation_error.mean()

print(f"\nConservation check (S+E+I+R = {N_expected:,.0f}):")
print(f"  Max relative error: {max_error:.2e}")
print(f"  Mean relative error: {mean_error:.2e}")

if max_error < 1e-3:  # 0.1% tolerance
    print("  ✓ PASS: Population conserved within 0.1%")
else:
    print(f"  ✗ FAIL: Population not conserved (error {max_error:.1%})")

# ============================================================================
# Test 3: Stochastic Variability
# ============================================================================

print("\n" + "="*70)
print("Test 3: Stochastic Variability (Multiple Realizations)")
print("="*70)
print("Running 5 stochastic realizations with different seeds...")

config_var = StochasticModelConfig(
    n_nodes=5,
    n_age=4,
    n_bins=10,
    gravity_k=0.01,
    duration_days=100,
    output_freq_days=1.0,
    stochastic_threshold=50.0,  # More stochastic nodes
    tau_leap_dt=0.5,
    backend='cpu',
)

I_trajectories = []
for seed in range(5):
    config_run = replace(config_var, stochastic_seed=seed)
    results = run_simulation_stochastic(config_run, spatial_seed=42, epi_seed=123)
    ts = extract_timeseries(results)
    I_total = ts['I_node'].sum(axis=0)  # Total infectious over time
    I_trajectories.append(I_total)

I_trajectories = np.array(I_trajectories)  # Shape: (n_realizations, n_timepoints)

# Compute coefficient of variation at peak
peak_time_idx = I_trajectories.mean(axis=0).argmax()
I_at_peak = I_trajectories[:, peak_time_idx]
cv_at_peak = I_at_peak.std() / I_at_peak.mean()

print("\nVariability across realizations:")
print(f"  Mean peak infectious: {I_at_peak.mean():.0f}")
print(f"  Std dev at peak: {I_at_peak.std():.0f}")
print(f"  Coefficient of variation: {cv_at_peak:.2%}")

if cv_at_peak > 0.01:  # Should have >1% variation
    print(f"  ✓ PASS: Stochastic model shows variability (CV={cv_at_peak:.1%})")
else:
    print(f"  ✗ WARNING: Low variability (CV={cv_at_peak:.1%}), may be too deterministic")

# ============================================================================
# Test 4: Fade-out Behavior
# ============================================================================

print("\n" + "="*70)
print("Test 4: Fade-out in Small Nodes")
print("="*70)

config_fadeout = StochasticModelConfig(
    n_nodes=10,
    n_age=4,
    n_bins=10,
    gravity_k=0.001,  # Weak coupling for more independent dynamics
    duration_days=200,
    output_freq_days=1.0,
    stochastic_threshold=20.0,
    tau_leap_dt=0.5,
    backend='cpu',
)

results_fadeout = run_simulation_stochastic(config_fadeout, spatial_seed=42, epi_seed=123)
ts_fadeout = extract_timeseries(results_fadeout)

# Check for nodes that hit I=0 then reintroduce
I_by_node = ts_fadeout['I_node']  # Shape: (n_nodes, n_timepoints)
n_fadeouts = 0
n_reintroductions = 0

for node in range(config_fadeout.n_nodes):
    I_node = I_by_node[node, :]

    # Find periods where I=0
    zero_periods = I_node < 0.1

    # Count transitions 0 → nonzero (fadeout then reintroduction)
    for t in range(1, len(I_node)):
        if I_node[t-1] > 1.0 and I_node[t] < 0.1:
            n_fadeouts += 1
        if I_node[t-1] < 0.1 and I_node[t] > 1.0:
            n_reintroductions += 1

print("\nFade-out dynamics:")
print(f"  Number of fade-outs (I → 0): {n_fadeouts}")
print(f"  Number of reintroductions (0 → I): {n_reintroductions}")

if n_fadeouts > 0:
    print("  ✓ PASS: Stochastic fade-out observed")
else:
    print("  ✗ WARNING: No fade-outs observed (may need longer simulation or more nodes)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ All core tests completed")
print("\nNext steps:")
print("  - Test dt sensitivity (0.1, 0.25, 0.5, 1.0 days)")
print("  - Run large-scale simulation (774 nodes)")
print("  - Implement discrete P matrix mode (optional)")
print("  - Add overdispersion for importation (optional)")
