"""
Bin Resolution and Age Resolution Performance Experiment

Tests how discretization parameters (n_bins, n_age) affect accuracy and performance
using the deterministic CPU-based ODE solver (SEIR_pde_metapop.py).

Experiments:
1. Heterogeneity resolution: n_bins = [2, 5, 10, 25, 50, 75, 100] with fixed n_age=16
2. Age resolution: n_age = [1, 4, 8, 16, 32] with fixed n_bins=50

For each configuration:
- Single run (deterministic, no stochasticity)
- Record system size (3 × n_nodes × n_age × n_bins + n_nodes × n_age)
- Record I(t) time series and peak prevalence
- Calculate L2 error relative to baseline
- CPU runtime (scipy.integrate.solve_ivp with RK45)

Outputs: figures/resolution_analysis.pdf (2 rows × 3 columns)
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src directory to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from SEIR_pde_metapop import ModelConfig, run_simulation, extract_timeseries, MetaPopState

# Ensure figures directory exists
figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*80)
print("RESOLUTION ANALYSIS EXPERIMENT")
print("="*80)
print()
print("Deterministic CPU-based ODE solver (scipy RK45, single node)")
print("Testing how discretization parameters affect accuracy and performance:")
print("1. Heterogeneity resolution (n_bins): 2, 5, 10, 25, 50, 75, 100")
print("2. Age resolution (n_age): 1, 4, 8, 16, 32")
print()

# ============================================================================
# Base Configuration (Single Node)
# ============================================================================

base_config = {
    'n_nodes': 1,
    'seed_node_idx': 0,
    'seed_n_infections': 10.0,
    'pop_dist_type': 'uniform',
    'N_total': 186_763,
    'sigma_rate': 1/3,
    'gamma_rate': 1/24,
    'beta_dist_type': 'uniform',
    'beta_mean': 1.5,  # Reduced to spread epidemic peak to ~day 100
    'beta_variance': 0.0,  # Uniform (no variance)
    'gravity_k': 0.0,  # Single node, no spatial coupling
    'duration_days': 365,
    'output_freq_days': 1.0,
}

# ============================================================================
# Experiment 1: Heterogeneity Resolution (n_bins)
# ============================================================================

print("="*80)
print("EXPERIMENT 1: Heterogeneity Resolution (n_bins)")
print("="*80)
print()

n_bins_values = [2, 5, 10, 25, 50, 75, 100]
baseline_n_bins = 50
fixed_n_age = 16
n_runs = 1  # Single run (deterministic ODE solver, no stochasticity)

exp1_results = {}

for n_bins in n_bins_values:
    print(f"Testing n_bins={n_bins} (n_age={fixed_n_age})...")

    config = ModelConfig(**base_config, n_bins=n_bins, n_age=fixed_n_age)
    system_size = MetaPopState.state_size(config.n_nodes, config.n_age, config.n_bins)

    print(f"  System size: {system_size:,} ODEs")

    print(f"    Running...", end='', flush=True)

    t_start = time.time()
    results = run_simulation(config, spatial_seed=42, epi_seed=123)
    runtime = time.time() - t_start

    # Extract I(t) time series
    ts = extract_timeseries(results)
    I_total = ts['I_node'].sum(axis=0)  # Sum over nodes (only 1 node)
    peak_I = I_total.max()

    print(f" {runtime:.2f}s, peak I={peak_I:.0f}")

    I_mean = I_total
    mean_runtime = runtime
    std_runtime = 0.0  # No variation in deterministic simulation

    exp1_results[n_bins] = {
        'system_size': system_size,
        'mean_runtime': mean_runtime,
        'std_runtime': std_runtime,
        'I_mean': I_mean,
        'peak_I': peak_I,
        'time': ts['time'],
    }

    print(f"  Mean runtime: {mean_runtime:.2f} ± {std_runtime:.2f}s")
    print(f"  Peak I: {peak_I:.0f}")
    print()

# Compute L2 errors relative to n_bins=50 baseline
baseline_I = exp1_results[baseline_n_bins]['I_mean']
for n_bins in n_bins_values:
    I_test = exp1_results[n_bins]['I_mean']
    l2_error = np.sqrt(np.mean((I_test - baseline_I)**2))
    rel_error = l2_error / baseline_I.max()
    exp1_results[n_bins]['l2_error'] = l2_error
    exp1_results[n_bins]['rel_error'] = rel_error

# ============================================================================
# Experiment 2: Age Resolution (n_age)
# ============================================================================

print("="*80)
print("EXPERIMENT 2: Age Resolution (n_age)")
print("="*80)
print()

n_age_values = [1, 4, 8, 16, 32]
baseline_n_age = 16
fixed_n_bins = 50
# n_runs = 1 (inherited from Experiment 1)

exp2_results = {}

for n_age in n_age_values:
    print(f"Testing n_age={n_age} (n_bins={fixed_n_bins})...")

    config = ModelConfig(**base_config, n_bins=fixed_n_bins, n_age=n_age)
    system_size = MetaPopState.state_size(config.n_nodes, config.n_age, config.n_bins)

    print(f"  System size: {system_size:,} ODEs")

    print(f"    Running...", end='', flush=True)

    t_start = time.time()
    results = run_simulation(config, spatial_seed=42, epi_seed=123)
    runtime = time.time() - t_start

    # Extract I(t) time series
    ts = extract_timeseries(results)
    I_total = ts['I_node'].sum(axis=0)
    peak_I = I_total.max()

    print(f" {runtime:.2f}s, peak I={peak_I:.0f}")

    I_mean = I_total
    mean_runtime = runtime
    std_runtime = 0.0  # No variation in deterministic simulation

    exp2_results[n_age] = {
        'system_size': system_size,
        'mean_runtime': mean_runtime,
        'std_runtime': std_runtime,
        'I_mean': I_mean,
        'peak_I': peak_I,
        'time': ts['time'],
    }

    print(f"  Mean runtime: {mean_runtime:.2f} ± {std_runtime:.2f}s")
    print(f"  Peak I: {peak_I:.0f}")
    print()

# Compute L2 errors relative to n_age=16 baseline
baseline_I = exp2_results[baseline_n_age]['I_mean']
for n_age in n_age_values:
    I_test = exp2_results[n_age]['I_mean']
    l2_error = np.sqrt(np.mean((I_test - baseline_I)**2))
    rel_error = l2_error / baseline_I.max()
    exp2_results[n_age]['l2_error'] = l2_error
    exp2_results[n_age]['rel_error'] = rel_error

# ============================================================================
# Plotting
# ============================================================================

print("="*80)
print("GENERATING PLOTS")
print("="*80)

fig = plt.figure(figsize=(18, 10))

# Colors for each configuration
colors_nbins = {2: '#8c564b', 5: '#e377c2', 10: '#e41a1c', 25: '#377eb8',
                50: '#4daf4a', 75: '#984ea3', 100: '#ff7f00'}
colors_nage = {1: '#e41a1c', 4: '#377eb8', 8: '#4daf4a',
               16: '#984ea3', 32: '#ff7f00'}

# --------------------------------------------------------------------------
# Row 1: Heterogeneity Resolution (n_bins)
# --------------------------------------------------------------------------

# Panel A1: Epidemic Curves
ax1 = plt.subplot(2, 3, 1)
for n_bins in n_bins_values:
    data = exp1_results[n_bins]
    label = f'n_bins={n_bins}'
    if n_bins == baseline_n_bins:
        label += ' (baseline)'
    ax1.plot(data['time'], data['I_mean'], color=colors_nbins[n_bins],
             linewidth=2, label=label)

ax1.set_xlabel('Time (days)', fontsize=11)
ax1.set_ylabel('Total Infectious I(t)', fontsize=11)
ax1.set_title('A1. Heterogeneity Resolution: Epidemic Curves',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Panel B1: Runtime vs Resolution
ax2 = plt.subplot(2, 3, 2)
n_bins_array = np.array(n_bins_values)
runtimes = [exp1_results[nb]['mean_runtime'] for nb in n_bins_values]
runtime_stds = [exp1_results[nb]['std_runtime'] for nb in n_bins_values]

ax2.plot(n_bins_array, runtimes, 'o-', markersize=8, linewidth=2, color='steelblue')

# Fit linear model
from scipy.optimize import curve_fit
def linear(x, a, b):
    return a * x + b

popt_lin, _ = curve_fit(linear, n_bins_array, runtimes)

x_fit = np.linspace(n_bins_array.min(), n_bins_array.max(), 100)
y_fit_lin = linear(x_fit, *popt_lin)

ax2.plot(x_fit, y_fit_lin, '--', color='red', alpha=0.7, linewidth=2,
         label=f'Linear fit: {popt_lin[0]:.4f}n + {popt_lin[1]:.2f}')

ax2.set_xlabel('n_bins', fontsize=11)
ax2.set_ylabel('Runtime (seconds)', fontsize=11)
ax2.set_title('B1. Runtime vs Heterogeneity Resolution',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)

# Panel C1: Accuracy vs Cost
ax3 = plt.subplot(2, 3, 3)
runtimes_plot = [exp1_results[nb]['mean_runtime'] for nb in n_bins_values]
rel_errors = [exp1_results[nb]['rel_error'] * 100 for nb in n_bins_values]  # Convert to %

for i, n_bins in enumerate(n_bins_values):
    marker = 's' if n_bins == baseline_n_bins else 'o'
    size = 150 if n_bins == baseline_n_bins else 100
    ax3.scatter(runtimes_plot[i], rel_errors[i],
                s=size, color=colors_nbins[n_bins],
                marker=marker, edgecolors='black', linewidth=1.5,
                label=f'n_bins={n_bins}', zorder=10)

ax3.set_xlabel('Runtime (seconds)', fontsize=11)
ax3.set_ylabel(f'L2 Error vs n_bins={baseline_n_bins} (%)', fontsize=11)
ax3.set_title('C1. Accuracy vs Cost (Heterogeneity)',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)

# --------------------------------------------------------------------------
# Row 2: Age Resolution (n_age)
# --------------------------------------------------------------------------

# Panel A2: Epidemic Curves
ax4 = plt.subplot(2, 3, 4)
for n_age in n_age_values:
    data = exp2_results[n_age]
    label = f'n_age={n_age}'
    if n_age == baseline_n_age:
        label += ' (baseline)'
    ax4.plot(data['time'], data['I_mean'], color=colors_nage[n_age],
             linewidth=2, label=label)

ax4.set_xlabel('Time (days)', fontsize=11)
ax4.set_ylabel('Total Infectious I(t)', fontsize=11)
ax4.set_title('A2. Age Resolution: Epidemic Curves',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)

# Panel B2: Runtime vs Resolution
ax5 = plt.subplot(2, 3, 5)
n_age_array = np.array(n_age_values)
runtimes_age = [exp2_results[na]['mean_runtime'] for na in n_age_values]
runtime_stds_age = [exp2_results[na]['std_runtime'] for na in n_age_values]

ax5.plot(n_age_array, runtimes_age, 'o-', markersize=8, linewidth=2, color='steelblue')

# Fit O(n_age) linear curve
def linear(x, a, b):
    return a * x + b
popt_age, _ = curve_fit(linear, n_age_array, runtimes_age)
x_fit_age = np.linspace(n_age_array.min(), n_age_array.max(), 100)
y_fit_age = linear(x_fit_age, *popt_age)
ax5.plot(x_fit_age, y_fit_age, '--', color='red', alpha=0.6,
         label=f'O(n) fit: {popt_age[0]:.2f}n + {popt_age[1]:.2f}')

ax5.set_xlabel('n_age', fontsize=11)
ax5.set_ylabel('Runtime (seconds)', fontsize=11)
ax5.set_title('B2. Runtime vs Age Resolution',
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Panel C2: Accuracy vs Cost
ax6 = plt.subplot(2, 3, 6)
runtimes_age_plot = [exp2_results[na]['mean_runtime'] for na in n_age_values]
rel_errors_age = [exp2_results[na]['rel_error'] * 100 for na in n_age_values]

for i, n_age in enumerate(n_age_values):
    marker = 's' if n_age == baseline_n_age else 'o'
    size = 150 if n_age == baseline_n_age else 100
    ax6.scatter(runtimes_age_plot[i], rel_errors_age[i],
                s=size, color=colors_nage[n_age],
                marker=marker, edgecolors='black', linewidth=1.5,
                label=f'n_age={n_age}', zorder=10)

ax6.set_xlabel('Runtime (seconds)', fontsize=11)
ax6.set_ylabel(f'L2 Error vs n_age={baseline_n_age} (%)', fontsize=11)
ax6.set_title('C2. Accuracy vs Cost (Age)',
              fontsize=12, fontweight='bold')
ax6.legend(fontsize=9, loc='best')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
output_file = figures_dir / 'resolution_analysis.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# ============================================================================
# Summary Statistics
# ============================================================================

print()
print("="*80)
print("EXPERIMENT 1 SUMMARY: Heterogeneity Resolution (n_bins)")
print("="*80)
print(f"{'n_bins':<10} {'System Size':<15} {'Runtime (s)':<20} {'Peak I':<12} {'Rel Error':<12}")
print("-"*80)

for n_bins in n_bins_values:
    data = exp1_results[n_bins]
    size_str = f"{data['system_size']:,}"
    runtime_str = f"{data['mean_runtime']:.2f} ± {data['std_runtime']:.2f}"
    peak_str = f"{data['peak_I']:.0f}"
    error_str = f"{data['rel_error']*100:.2f}%"

    marker = " *" if n_bins == baseline_n_bins else ""
    print(f"{n_bins:<10} {size_str:<15} {runtime_str:<20} {peak_str:<12} {error_str:<12}{marker}")

print()
print("="*80)
print("EXPERIMENT 2 SUMMARY: Age Resolution (n_age)")
print("="*80)
print(f"{'n_age':<10} {'System Size':<15} {'Runtime (s)':<20} {'Peak I':<12} {'Rel Error':<12}")
print("-"*80)

for n_age in n_age_values:
    data = exp2_results[n_age]
    size_str = f"{data['system_size']:,}"
    runtime_str = f"{data['mean_runtime']:.2f} ± {data['std_runtime']:.2f}"
    peak_str = f"{data['peak_I']:.0f}"
    error_str = f"{data['rel_error']*100:.2f}%"

    marker = " *" if n_age == baseline_n_age else ""
    print(f"{n_age:<10} {size_str:<15} {runtime_str:<20} {peak_str:<12} {error_str:<12}{marker}")

print()
print("="*80)
print("KEY FINDINGS")
print("="*80)

# Experiment 1 findings
print("\nHeterogeneity Resolution (n_bins):")
print(f"  - Runtime scales approximately linearly: {popt_lin[0]:.4f}n + {popt_lin[1]:.2f}")
print(f"  - Baseline (n_bins=50): {exp1_results[50]['mean_runtime']:.2f}s")
print(f"  - Very coarse (n_bins=2): {exp1_results[2]['mean_runtime']:.2f}s "
      f"({exp1_results[2]['rel_error']*100:.2f}% error)")
print(f"  - Coarse (n_bins=10): {exp1_results[10]['mean_runtime']:.2f}s "
      f"({exp1_results[10]['rel_error']*100:.2f}% error)")
print(f"  - Fine (n_bins=100): {exp1_results[100]['mean_runtime']:.2f}s "
      f"({exp1_results[100]['rel_error']*100:.2f}% error)")

# Experiment 2 findings
print("\nAge Resolution (n_age):")
print(f"  - Runtime scales as O(n_age): fit coefficient = {popt_age[0]:.2f}")
print(f"  - Baseline (n_age=16): {exp2_results[16]['mean_runtime']:.2f}s")
print(f"  - Reducing to n_age=1: {exp2_results[1]['mean_runtime']:.2f}s "
      f"({exp2_results[1]['rel_error']*100:.2f}% error, "
      f"{exp2_results[16]['mean_runtime']/exp2_results[1]['mean_runtime']:.1f}× speedup)")
print(f"  - Reducing to n_age=4: {exp2_results[4]['mean_runtime']:.2f}s "
      f"({exp2_results[4]['rel_error']*100:.2f}% error, "
      f"{exp2_results[16]['mean_runtime']/exp2_results[4]['mean_runtime']:.1f}× speedup)")

# Check if age structure matters
if exp2_results[1]['rel_error'] < 0.01:  # Less than 1% error
    print("\n⚠️  IMPORTANT: Age structure has minimal impact on epidemic dynamics!")
    print("    With no age-dependent mixing, n_age could be reduced to 1 or 4")
    print("    for substantial computational savings without loss of accuracy.")

print()
print("="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
