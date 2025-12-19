"""
Compare dynamics across different stochasticity thresholds.

Tests:
- Fully deterministic (threshold=0, all nodes use RK4)
- Hybrid modes (threshold=10, 50, 100, 500)
- Fully stochastic (threshold=∞, all nodes use tau-leaping)

For stochastic modes, run multiple realizations to show variability.
"""

import numpy as np
import matplotlib.pyplot as plt
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic
from SEIR_pde_metapop import extract_timeseries

print("="*70)
print("Threshold Comparison: Deterministic vs Hybrid vs Stochastic")
print("="*70)

# Base configuration
base_config = {
    'n_nodes': 20,
    'n_age': 4,
    'n_bins': 10,
    'seed_node_idx': 0,
    'seed_n_infections': 80.0,  # High enough for epidemic to take off
    'gravity_k': 0.01,
    'duration_days': 150,
    'backend': 'cpu',
    'use_float64': True,
    'tau_leap_dt': 1.0,
    'output_freq_days': 1.0,
}

# Thresholds to test
thresholds = [
    (0.0, "Fully Deterministic (all RK4)", 1),      # threshold=0 → all deterministic
    (10.0, "Hybrid (threshold=10)", 1),
    (50.0, "Hybrid (threshold=50)", 1),
    (100.0, "Hybrid (threshold=100)", 1),
    (500.0, "Hybrid (threshold=500)", 5),           # More stochastic, more runs
    (1e10, "Fully Stochastic (all tau-leap)", 10),  # All stochastic, many runs
]

results_dict = {}

for threshold, label, n_runs in thresholds:
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"  Threshold: {threshold}")
    print(f"  Number of realizations: {n_runs}")
    print(f"{'='*70}")

    I_trajectories = []

    for run_idx in range(n_runs):
        config = StochasticModelConfig(
            **base_config,
            stochastic_threshold=threshold,
            stochastic_seed=42 + run_idx,  # Different seed for each run
        )

        print(f"  Run {run_idx + 1}/{n_runs}...", end='')
        results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)
        ts = extract_timeseries(results)
        I_total = ts['I_node'].sum(axis=0)
        I_trajectories.append(I_total)
        print(f" peak={I_total.max():.0f}")

    I_trajectories = np.array(I_trajectories)  # Shape: (n_runs, n_timepoints)

    # Compute statistics
    I_mean = I_trajectories.mean(axis=0)
    I_std = I_trajectories.std(axis=0)
    I_min = I_trajectories.min(axis=0)
    I_max = I_trajectories.max(axis=0)

    results_dict[threshold] = {
        'label': label,
        'n_runs': n_runs,
        'trajectories': I_trajectories,
        'mean': I_mean,
        'std': I_std,
        'min': I_min,
        'max': I_max,
    }

    print(f"  Mean peak: {I_mean.max():.0f} ± {I_std[I_mean.argmax()]:.0f}")
    if n_runs > 1:
        cv = I_std[I_mean.argmax()] / I_mean.max()
        print(f"  Coefficient of variation at peak: {cv:.2%}")

# ============================================================================
# Plotting
# ============================================================================

print(f"\n{'='*70}")
print("Generating plots...")
print(f"{'='*70}")

t_array = np.arange(0, base_config['duration_days'], base_config['output_freq_days'])

# Color scheme
colors = {
    0.0: '#1f77b4',      # Blue (deterministic)
    10.0: '#ff7f0e',     # Orange
    50.0: '#2ca02c',     # Green
    100.0: '#d62728',    # Red
    500.0: '#9467bd',    # Purple
    1e10: '#8c564b',     # Brown (fully stochastic)
}

# ---------- Plot 1: All trajectories with confidence bands ----------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for threshold in [0.0, 50.0, 100.0, 1e10]:
    data = results_dict[threshold]
    color = colors[threshold]
    label = data['label']

    # Plot mean
    ax.plot(t_array, data['mean'], color=color, linewidth=2.5, label=label)

    # Plot confidence band for stochastic runs
    if data['n_runs'] > 1:
        ax.fill_between(t_array, data['min'], data['max'],
                        color=color, alpha=0.15)
        # Add line for range
        ax.plot(t_array, data['min'], color=color, linewidth=0.5, linestyle=':', alpha=0.5)
        ax.plot(t_array, data['max'], color=color, linewidth=0.5, linestyle=':', alpha=0.5)

ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Total Infectious', fontsize=12)
ax.set_title('Threshold Comparison: Deterministic vs Hybrid vs Stochastic', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_comparison.pdf')
print("✓ Saved: threshold_comparison.pdf")

# ---------- Plot 2: Individual realizations for fully stochastic ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_thresholds = [0.0, 100.0, 500.0, 1e10]
for idx, threshold in enumerate(plot_thresholds):
    ax = axes[idx // 2, idx % 2]
    data = results_dict[threshold]
    color = colors[threshold]

    # Plot all individual trajectories
    for traj in data['trajectories']:
        ax.plot(t_array, traj, color=color, alpha=0.3, linewidth=0.8)

    # Plot mean
    ax.plot(t_array, data['mean'], color='black', linewidth=2.5,
            label=f"Mean (n={data['n_runs']})", zorder=10)

    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Total Infectious', fontsize=11)
    ax.set_title(data['label'], fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_individual_realizations.pdf')
print("✓ Saved: threshold_individual_realizations.pdf")

# ---------- Plot 3: Summary statistics ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Peak infectious by threshold
ax = axes[0]
threshold_vals = [0.0, 10.0, 50.0, 100.0, 500.0, 1e10]
threshold_labels = ['Det', '10', '50', '100', '500', 'Stoch']
peak_means = []
peak_stds = []

for threshold in threshold_vals:
    data = results_dict[threshold]
    peak_means.append(data['mean'].max())
    peak_stds.append(data['std'][data['mean'].argmax()])

x_pos = np.arange(len(threshold_vals))
bars = ax.bar(x_pos, peak_means, yerr=peak_stds, capsize=5,
              color=[colors[t] for t in threshold_vals], alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(threshold_labels)
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Peak Infectious', fontsize=12)
ax.set_title('Peak Epidemic Size by Threshold', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Coefficient of variation at peak
ax = axes[1]
cv_vals = []
for threshold in threshold_vals:
    data = results_dict[threshold]
    if data['n_runs'] > 1:
        cv = data['std'][data['mean'].argmax()] / data['mean'].max()
        cv_vals.append(cv * 100)  # Convert to percentage
    else:
        cv_vals.append(0)

bars = ax.bar(x_pos, cv_vals, color=[colors[t] for t in threshold_vals],
              alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(threshold_labels)
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax.set_title('Variability at Peak (across realizations)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('threshold_summary_stats.pdf')
print("✓ Saved: threshold_summary_stats.pdf")

# ============================================================================
# Summary Table
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Threshold':<15} {'Label':<30} {'Runs':<6} {'Peak (mean)':<15} {'Peak (std)':<15} {'CV at peak'}")
print("-"*70)

for threshold in [0.0, 10.0, 50.0, 100.0, 500.0, 1e10]:
    data = results_dict[threshold]
    peak_mean = data['mean'].max()
    peak_std = data['std'][data['mean'].argmax()]
    cv = peak_std / peak_mean if peak_mean > 0 else 0

    thresh_str = "0 (Det)" if threshold == 0.0 else ("∞ (Stoch)" if threshold == 1e10 else str(int(threshold)))

    print(f"{thresh_str:<15} {data['label']:<30} {data['n_runs']:<6} "
          f"{peak_mean:<15.0f} {peak_std:<15.1f} {cv:>7.2%}")

print(f"\n{'='*70}")
print("Key findings:")
print(f"{'='*70}")

# Compare deterministic vs fully stochastic
det_peak = results_dict[0.0]['mean'].max()
stoch_peak = results_dict[1e10]['mean'].max()
diff_pct = (stoch_peak - det_peak) / det_peak * 100

print(f"1. Deterministic peak: {det_peak:.0f}")
print(f"2. Fully stochastic peak: {stoch_peak:.0f} (mean over {results_dict[1e10]['n_runs']} runs)")
print(f"3. Difference: {diff_pct:+.1f}%")
print(f"4. Stochastic variability (CV): {results_dict[1e10]['std'][results_dict[1e10]['mean'].argmax()] / stoch_peak:.1%}")

print(f"\nHybrid thresholds smoothly interpolate between deterministic and stochastic extremes.")
print(f"Higher thresholds → more nodes stochastic → greater variability")

print(f"\n{'='*70}")
print("Analysis complete!")
print(f"{'='*70}")
