"""
Compare dynamics across different stochasticity thresholds.

Tests:
- Fully deterministic (threshold=0, all nodes use RK4)
- Hybrid modes (threshold=10, 50, 100, 500)
- Fully stochastic (threshold=∞, all nodes use tau-leaping)

For all models with stochastic elements (threshold > 0), run multiple realizations to show variability.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from SEIR_pde_metapop import extract_timeseries
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

# Ensure figures directory exists
figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("Threshold Comparison: Deterministic vs Hybrid vs Stochastic")
print("="*70)

# Base configuration
base_config = {
    'n_nodes': 4,
    'n_age': 16,
    'n_bins': 50,
    'seed_node_idx': 0,
    'seed_n_infections': 10.0,
    'gravity_k': 0.01,
    'duration_days': 150,
    'backend': 'metal',  # Use GPU acceleration (Metal on macOS, or 'cuda' on NVIDIA)
    'use_float64': False,  # float32 is sufficient for stochastic models
    'tau_leap_dt': 0.5,  # Reduced from 1.0 to avoid numerical instability in deterministic mode
    'output_freq_days': 1.0,
}

# Thresholds to test
# Note: All thresholds > 0 have stochastic elements, so run multiple replicates
reps = 10
thresholds = [
    (0.0, "Fully Deterministic (all RK4)", 1),      # threshold=0 → all deterministic, no stochasticity
    (10.0, "Hybrid (threshold=10)", reps),            # Hybrid: stochastic at low I
    (50.0, "Hybrid (threshold=50)", reps),            # Hybrid: stochastic at low I
    (100.0, "Hybrid (threshold=100)", reps),          # Hybrid: stochastic at low I
    (500.0, "Hybrid (threshold=500)", reps),          # More stochastic
    (1e10, "Fully Stochastic (all tau-leap)", reps),  # All stochastic
]

results_dict = {}

for threshold, label, n_runs in thresholds:
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"  Threshold: {threshold}")
    print(f"  Number of realizations: {n_runs}")
    print(f"{'='*70}")

    I_trajectories = []
    run_times = []

    for run_idx in range(n_runs):
        config = StochasticModelConfig(
            **base_config,
            stochastic_threshold=threshold,
            stochastic_seed=42 + run_idx,  # Different seed for each run
        )

        print(f"  Run {run_idx + 1}/{n_runs}...", end='')
        t0 = time.time()
        results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)
        run_time = time.time() - t0
        run_times.append(run_time)

        ts = extract_timeseries(results)
        I_total = ts['I_node'].sum(axis=0)
        I_trajectories.append(I_total)
        print(f" peak={I_total.max():.0f}, time={run_time:.2f}s")

    I_trajectories = np.array(I_trajectories)  # Shape: (n_runs, n_timepoints)
    run_times = np.array(run_times)

    # Compute statistics
    I_mean = I_trajectories.mean(axis=0)
    I_std = I_trajectories.std(axis=0)
    I_min = I_trajectories.min(axis=0)
    I_max = I_trajectories.max(axis=0)

    mean_time = run_times.mean()
    std_time = run_times.std()

    results_dict[threshold] = {
        'label': label,
        'n_runs': n_runs,
        'trajectories': I_trajectories,
        'mean': I_mean,
        'std': I_std,
        'min': I_min,
        'max': I_max,
        'mean_time': mean_time,
        'std_time': std_time,
    }

    print(f"  Mean peak: {I_mean.max():.0f} ± {I_std[I_mean.argmax()]:.0f}")
    print(f"  Mean run time: {mean_time:.2f} ± {std_time:.2f}s")
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

for threshold in [0.0, 10.0, 50.0, 1e10]:
    data = results_dict[threshold]
    color = colors[threshold]

    # Create label with timing info
    if data['n_runs'] > 1:
        label = f"{data['label']} ({data['mean_time']:.1f}±{data['std_time']:.1f}s)"
    else:
        label = f"{data['label']} ({data['mean_time']:.1f}s)"

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
plt.savefig(figures_dir / 'threshold_comparison.pdf')
print(f"✓ Saved: {figures_dir / 'threshold_comparison.pdf'}")

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

    # Plot mean with timing info
    if data['n_runs'] > 1:
        mean_label = f"Mean (n={data['n_runs']}, {data['mean_time']:.1f}±{data['std_time']:.1f}s)"
    else:
        mean_label = f"Mean ({data['mean_time']:.1f}s)"
    ax.plot(t_array, data['mean'], color='black', linewidth=2.5,
            label=mean_label, zorder=10)

    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Total Infectious', fontsize=11)
    ax.set_title(data['label'], fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'threshold_individual_realizations.pdf')
print(f"✓ Saved: {figures_dir / 'threshold_individual_realizations.pdf'}")

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
plt.savefig(figures_dir / 'threshold_summary_stats.pdf')
print(f"✓ Saved: {figures_dir / 'threshold_summary_stats.pdf'}")

# ============================================================================
# Summary Table
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Threshold':<15} {'Label':<30} {'Runs':<6} {'Peak (mean)':<15} {'Peak (std)':<12} {'CV':<8} {'Time (s)'}")
print("-"*70)

for threshold in [0.0, 10.0, 50.0, 100.0, 500.0, 1e10]:
    data = results_dict[threshold]
    peak_mean = data['mean'].max()
    peak_std = data['std'][data['mean'].argmax()]
    cv = peak_std / peak_mean if peak_mean > 0 else 0

    thresh_str = "0 (Det)" if threshold == 0.0 else ("∞ (Stoch)" if threshold == 1e10 else str(int(threshold)))

    if data['n_runs'] > 1:
        time_str = f"{data['mean_time']:.1f}±{data['std_time']:.1f}"
    else:
        time_str = f"{data['mean_time']:.1f}"

    print(f"{thresh_str:<15} {data['label']:<30} {data['n_runs']:<6} "
          f"{peak_mean:<15.0f} {peak_std:<12.1f} {cv:<8.2%} {time_str}")

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

print("\nHybrid thresholds smoothly interpolate between deterministic and stochastic extremes.")
print("Higher thresholds → more nodes stochastic → greater variability")

print(f"\n{'='*70}")
print("Analysis complete!")
print(f"{'='*70}")
