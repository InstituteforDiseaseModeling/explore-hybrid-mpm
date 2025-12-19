"""
Test fully stochastic simulation - all nodes use tau-leaping.
"""

from SEIR_pde_metapop_stochastic_taichi import StochasticModelConfig, run_simulation_stochastic
from SEIR_pde_metapop import extract_timeseries
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("Fully Stochastic Simulation Test")
print("="*70)

# Small network for fully stochastic test
config = StochasticModelConfig(
    n_nodes=10,
    n_age=4,
    n_bins=10,
    seed_node_idx=0,
    seed_n_infections=50.0,  # Moderate seed
    gravity_k=0.01,
    duration_days=200,
    backend='cpu',
    use_float64=True,
    stochastic_threshold=1e10,  # Very high → all nodes stochastic
    tau_leap_dt=1.0,
    output_freq_days=1.0,
    stochastic_seed=42,
)

print(f"\nConfiguration:")
print(f"  n_nodes: {config.n_nodes}")
print(f"  seed_n_infections: {config.seed_n_infections}")
print(f"  stochastic_threshold: {config.stochastic_threshold}")
print(f"  → ALL nodes will use stochastic tau-leaping")

# Run simulation
results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)
ts = extract_timeseries(results)

# Extract timeseries
I_total = ts['I_node'].sum(axis=0)
I_by_node = ts['I_node']  # Shape: (n_nodes, n_timepoints)

print(f"\n" + "="*70)
print("Results")
print("="*70)
print(f"Initial I: {I_total[0]:.2f}")
print(f"Peak I: {I_total.max():.2f}")
print(f"Peak day: {I_total.argmax()}")
print(f"Final I: {I_total[-1]:.2f}")

if np.any(np.isnan(I_total)):
    print(f"❌ NaN detected at day {np.where(np.isnan(I_total))[0][0]}")
else:
    print(f"✓ No NaN values")

# Count fadeouts
n_fadeouts = 0
n_reintroductions = 0

for node in range(config.n_nodes):
    I_node = I_by_node[node, :]

    for t in range(1, len(I_node)):
        if I_node[t-1] > 1.0 and I_node[t] < 0.1:
            n_fadeouts += 1
        if I_node[t-1] < 0.1 and I_node[t] > 1.0:
            n_reintroductions += 1

print(f"\nStochastic dynamics:")
print(f"  Fadeouts (I → 0): {n_fadeouts}")
print(f"  Reintroductions (0 → I): {n_reintroductions}")

if n_fadeouts > 0:
    print(f"  ✓ Stochastic fadeout observed!")
else:
    print(f"  ⚠ No fadeouts (may need longer simulation)")

# Plot timeseries by node
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Get time array
t_array = np.arange(0, config.duration_days, config.output_freq_days)

# Total infectious
axes[0].plot(t_array, I_total, 'b-', linewidth=2, label='Total I')
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Total Infectious')
axes[0].set_title('Fully Stochastic Simulation: Total Infectious')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Individual nodes
for node in range(config.n_nodes):
    axes[1].plot(t_array, I_by_node[node, :], alpha=0.6, label=f'Node {node}')

axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Infectious')
axes[1].set_title('Individual Node Dynamics (Stochastic)')
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig('fully_stochastic_test.pdf')
print(f"\n✓ Saved plot: fully_stochastic_test.pdf")

print("\n" + "="*70)
print("Fully stochastic test complete!")
print("="*70)
