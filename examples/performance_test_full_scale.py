"""
Performance test with full-scale configuration:
- 774 nodes (full network)
- 2 age bins
- 25 heterogeneity bins
- 365 days simulation
- Metal backend (GPU acceleration)

Records timing and generates diagnostic visualizations.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from SEIR_pde_metapop import extract_timeseries
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

print("=" * 70)
print("FULL-SCALE PERFORMANCE TEST")
print("=" * 70)
print("\nConfiguration:")
print("  n_nodes: 774 (full network)")
print("  n_age: 2")
print("  n_bins: 25")
print(f"  Total ODEs: {774 * 2 * 25 * 3 + 774 * 2:,}")
print("  duration_days: 365")
print("  backend: metal (GPU)")
print("  deterministic_method: rk4")
print("  tau_leap_dt: 1.0 days")
print()

# Configuration
config = StochasticModelConfig(
    n_nodes=774,
    n_age=2,
    n_bins=25,
    seed_node_idx=0,
    seed_n_infections=50.0,
    gravity_k=0.01,
    duration_days=365,
    backend='metal',  # GPU acceleration
    use_float64=False,  # float32 only option on Metal (float64 not supported)
    stochastic_threshold=100.0,  # Hybrid: stochastic when I < 100
    tau_leap_dt=1.0,
    deterministic_method='rk4',  # Use RK4 for accuracy
    output_freq_days=1.0,
    stochastic_seed=45,
)

print("=" * 70)
print("RUNNING SIMULATION")
print("=" * 70)

# Time the simulation
t_start = time.time()
results = run_simulation_stochastic(config, spatial_seed=40, epi_seed=1234)
t_total = time.time() - t_start

# Extract reported times from results if available
if 'timing' in results:
    t_setup = results['timing'].get('setup', 0)
    t_sim = results['timing'].get('simulation', 0)
else:
    t_setup = 0
    t_sim = t_total

print(f"\n{'=' * 70}")
print("TIMING RESULTS")
print(f"{'=' * 70}")
print(f"  Setup time:      {t_setup:.2f}s")
print(f"  Simulation time: {t_sim:.2f}s")
print(f"  Total time:      {t_total:.2f}s")
print(f"  Time per step:   {t_sim / 365 * 1000:.2f}ms")
print(f"  Steps per second: {365 / t_sim:.2f}")

# Extract timeseries
print(f"\n{'=' * 70}")
print("EXTRACTING TIMESERIES")
print(f"{'=' * 70}")
ts = extract_timeseries(results)

# Compute summary statistics
I_node = ts['I_node']  # Shape: (n_nodes, n_time)
I_total = I_node.sum(axis=0)
S_total = ts['S_node'].sum(axis=0)
E_total = ts['E_node'].sum(axis=0)
R_total = ts['R_node'].sum(axis=0)
total_pop = S_total + E_total + I_total + R_total

print(f"\nSimulation Summary:")
print(f"  Initial I: {I_total[0]:.0f}")
print(f"  Peak I: {I_total.max():.0f} at day {ts['time'][I_total.argmax()]:.0f}")
print(f"  Final I: {I_total[-1]:.0f}")
print(f"  Attack rate: {R_total[-1] / total_pop[0] * 100:.1f}%")

# Check for issues
print(f"\n{'=' * 70}")
print("VALIDATION")
print(f"{'=' * 70}")

if np.any(np.isnan(I_total)):
    print("❌ ERROR: NaN detected in I!")
    print(f"   First NaN at time index: {np.where(np.isnan(I_total))[0][0]}")
else:
    print("✓ No NaN values detected")

mass_change = (total_pop[-1] - total_pop[0]) / total_pop[0] * 100
if abs(mass_change) < 5:
    print(f"✓ Mass conservation OK ({mass_change:.2f}% change)")
else:
    print(f"⚠️  Mass conservation warning ({mass_change:.2f}% change)")

if I_total.max() > 1000:
    print(f"✓ Epidemic took off (peak = {I_total.max():.0f})")
else:
    print(f"⚠️  Weak epidemic (peak = {I_total.max():.0f})")

# =============================================================================
# GENERATE DIAGNOSTIC FIGURES
# =============================================================================

print(f"\n{'=' * 70}")
print("GENERATING DIAGNOSTIC FIGURES")
print(f"{'=' * 70}")

# Figure 1: Time series of compartments
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(ts['time'], S_total, label='S', linewidth=2)
ax.plot(ts['time'], E_total, label='E', linewidth=2)
ax.plot(ts['time'], I_total, label='I', linewidth=2)
ax.plot(ts['time'], R_total, label='R', linewidth=2)
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Population', fontsize=11)
ax.set_title('SEIR Dynamics (Total)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Figure 2: Infections by node over time (heatmap)
ax = axes[0, 1]
# Normalize by node population for better visualization
node_pops = results['node_pops']
I_prevalence = I_node / node_pops[:, np.newaxis]
im = ax.imshow(I_prevalence, aspect='auto', cmap='YlOrRd', interpolation='nearest',
               extent=[0, 365, 774, 0])
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Node ID', fontsize=11)
ax.set_title('Prevalence Heatmap (I/N per node)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Prevalence')

# Figure 3: Number of nodes with active infections
ax = axes[1, 0]
n_active_nodes = (I_node > 1).sum(axis=0)
ax.plot(ts['time'], n_active_nodes, linewidth=2, color='C3')
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Number of nodes', fontsize=11)
ax.set_title('Spatial Spread (nodes with I>1)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 774)

# Figure 4: Peak infection timing by node
ax = axes[1, 1]
peak_day_per_node = ts['time'][I_node.argmax(axis=1)]
peak_I_per_node = I_node.max(axis=1)
# Only plot nodes that had >10 infections at peak
mask = peak_I_per_node > 10
scatter = ax.scatter(peak_day_per_node[mask], np.where(mask)[0],
                     c=peak_I_per_node[mask], cmap='plasma',
                     s=20, alpha=0.6, edgecolors='none')
ax.set_xlabel('Day of peak infection', fontsize=11)
ax.set_ylabel('Node ID', fontsize=11)
ax.set_title('Epidemic Wave (peak timing by node)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Peak infections')
ax.set_xlim(0, 365)

plt.tight_layout()
output_file = figures_dir / 'performance_test_diagnostics.pdf'
plt.savefig(output_file)
print(f"✓ Saved: {output_file}")

# Figure 5: Detailed heatmap (larger, separate figure)
fig, ax = plt.subplots(figsize=(16, 10))
im = ax.imshow(I_node, aspect='auto', cmap='YlOrRd', interpolation='nearest',
               extent=[0, 365, 774, 0], norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=I_node.max()))
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Node ID', fontsize=12)
ax.set_title('Infections by Node Over Time (log scale)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='Infections')
cbar.set_label('Infections (log scale)', fontsize=12)

# Add timing annotation
textstr = f'Setup: {t_setup:.2f}s\nSimulation: {t_sim:.2f}s\nTotal: {t_total:.2f}s\n' \
          f'Time/step: {t_sim / 365 * 1000:.2f}ms'
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
output_file = figures_dir / 'performance_test_heatmap.pdf'
plt.savefig(output_file)
print(f"✓ Saved: {output_file}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'=' * 70}")
print("PERFORMANCE SUMMARY")
print(f"{'=' * 70}")
print(f"Configuration: 774 nodes, 2 age, 25 bins, 365 days")
print(f"Total ODEs: {774 * 2 * 25 * 3 + 774 * 2:,}")
print(f"Backend: metal (GPU)")
print(f"Method: RK4")
print(f"\nTiming:")
print(f"  Total time:      {t_total:.2f}s ({t_total / 60:.1f} min)")
print(f"  Simulation time: {t_sim:.2f}s")
print(f"  Time per day:    {t_sim / 365 * 1000:.1f}ms")
print(f"  Throughput:      {365 / t_sim:.1f} days/second")
print(f"\nResults:")
print(f"  Peak infections: {I_total.max():.0f}")
print(f"  Attack rate:     {R_total[-1] / total_pop[0] * 100:.1f}%")
print(f"  Nodes affected:  {(I_node.max(axis=1) > 1).sum()}/{774}")
print(f"\nFigures saved to: {figures_dir}/")
print("  - performance_test_diagnostics.pdf (4-panel overview)")
print("  - performance_test_heatmap.pdf (detailed infection heatmap)")
print(f"\n{'=' * 70}")
