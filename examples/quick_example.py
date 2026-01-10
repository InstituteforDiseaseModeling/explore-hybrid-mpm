"""
Quick example demonstrating hybrid stochastic/deterministic SEIR model.

This runs a small simulation (50 nodes, 30 days) that completes in ~5-10 seconds
and produces a basic visualization showing the epidemic dynamics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from SEIR_pde_metapop import extract_timeseries
from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic

# Output directory
figures_dir = Path(__file__).parent.parent / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("Quick Example: Hybrid Stochastic/Deterministic SEIR Model")
print("="*70)

# Small, fast configuration
config = StochasticModelConfig(
    n_nodes=25,              # Small network for quick demo
    n_age=16,
    n_bins=50,
    seed_node_idx=12,
    seed_n_infections=50.0,  # More initial infections for faster takeoff
    beta_mean=15.0,          # Higher transmission rate for faster spread
    gravity_k=1.0,           # Strong connectivity for small network
    duration_days=120,        # Full epidemic curve including decline
    backend='metal',         # Use GPU for speed
    use_float64=False,       # float32 is sufficient
    stochastic_threshold=50.0,  # Hybrid: stochastic when I < 50
    tau_leap_dt=1.0,
    output_freq_days=1.0,
    stochastic_seed=42,
)

print("\nRunning simulation...")
print(f"  Nodes: {config.n_nodes}")
print(f"  Duration: {config.duration_days} days")
print(f"  Stochastic threshold: {config.stochastic_threshold}")

results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)

# Extract and plot results
ts = extract_timeseries(results)
I_total = ts['I_node'].sum(axis=0)
S_total = ts['S_node'].sum(axis=0)
E_total = ts['E_node'].sum(axis=0)
R_total = ts['R_node'].sum(axis=0)
time = ts['time']

print("\n" + "="*70)
print("Results Summary")
print("="*70)
print(f"Initial infections: {I_total[0]:.1f}")
print(f"Peak infections: {I_total.max():.1f} at day {time[I_total.argmax()]:.0f}")
print(f"Final infections: {I_total[-1]:.1f}")
print(f"Total recovered: {R_total[-1]:.0f} ({R_total[-1]/(config.N_total)*100:.1f}% of population)")

# Node-level analysis
I_node = ts['I_node']  # Shape: (n_nodes, n_timepoints)
I_node_max = I_node.max(axis=1)  # Max infections per node over time
nodes_infected = (I_node_max > 1.0).sum()  # Nodes with more than 1 infection
node_pops = results['node_pops']

print("\nSpatial spread:")
print(f"  Nodes infected: {nodes_infected}/{config.n_nodes} ({nodes_infected/config.n_nodes*100:.1f}%)")
print(f"  Node populations: min={node_pops.min():.0f}, max={node_pops.max():.0f}, median={np.median(node_pops):.0f}")
print(f"  Seed node (idx {config.seed_node_idx}): pop={node_pops[config.seed_node_idx]:.0f}, rank={(node_pops < node_pops[config.seed_node_idx]).sum()+1}/{config.n_nodes} by size")
print(f"  Small nodes (< median): {(node_pops < np.median(node_pops)).sum()}")
print(f"  Small nodes infected: {((node_pops < np.median(node_pops)) & (I_node_max > 1.0)).sum()}")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Panel 1: SEIR compartments over time
ax = axes[0]
ax.plot(time, S_total, label='Susceptible', linewidth=2, color='#1f77b4')
ax.plot(time, E_total, label='Exposed', linewidth=2, color='#ff7f0e')
ax.plot(time, I_total, label='Infectious', linewidth=2, color='#d62728')
ax.plot(time, R_total, label='Recovered', linewidth=2, color='#2ca02c')
ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Number of individuals', fontsize=11)
ax.set_title('SEIR Epidemic Dynamics', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Prevalence per node (heatmap)
ax = axes[1]
I_node = ts['I_node']  # Shape: (n_nodes, n_timepoints)
# Compute prevalence (proportion infected)
prevalence = I_node / node_pops[:, np.newaxis]  # Broadcast node_pops across time
# Sort nodes by population size for better visualization
sorted_idx = np.argsort(node_pops)
prevalence_sorted = prevalence[sorted_idx, :]

# Find seed node position in sorted array
seed_node_sorted_pos = np.where(sorted_idx == config.seed_node_idx)[0][0]

im = ax.imshow(prevalence_sorted, aspect='auto', cmap='YlOrRd', origin='lower',
               extent=[time[0], time[-1], 0, config.n_nodes], vmin=0, vmax=0.2)

# Mark the seed node with lines above and below
ax.axhline(y=seed_node_sorted_pos, color='cyan', linewidth=1.5, linestyle='-', alpha=0.9)
ax.axhline(y=seed_node_sorted_pos + 1, color='cyan', linewidth=1.5, linestyle='-',
           label=f'Seed node (idx {config.seed_node_idx})', alpha=0.9)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

ax.set_xlabel('Time (days)', fontsize=11)
ax.set_ylabel('Node (sorted by population)', fontsize=11)
ax.set_title('Prevalence by Node (small nodes at bottom)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='Prevalence (I/N)')
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

plt.tight_layout()
output_file = figures_dir / 'quick_example.pdf'
plt.savefig(output_file)
print(f"\n✓ Saved figure: {output_file}")

print("\n" + "="*70)
print("Example complete!")
print("="*70)
