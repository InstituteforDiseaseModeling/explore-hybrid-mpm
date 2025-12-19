"""
Debug script to track down NaN values in stochastic simulation.
"""

import numpy as np
from dataclasses import asdict
from SEIR_pde_metapop_stochastic_taichi import (
    StochasticModelConfig, run_simulation_stochastic,
    initialize_taichi_stochastic, TaichiSEIRState,
    compute_I_totals, stochastic_step_kernel, deterministic_step_rk4_kernel
)
from SEIR_pde_metapop import (
    generate_node_positions, compute_distance_matrix,
    generate_node_populations, generate_node_betas,
    build_gravity_network, setup_age_structure, build_P
)
from SEIR_pde_metapop_taichi import (
    compute_infectious_pressure_kernel, compute_foi_kernel
)
from scipy.stats import lognorm, gamma
import taichi as ti

print("="*70)
print("Debug: NaN in Stochastic Simulation")
print("="*70)

# Small configuration for debugging
config = StochasticModelConfig(
    n_nodes=3,
    n_age=2,
    n_bins=5,
    seed_node_idx=0,
    seed_n_infections=10.0,
    gravity_k=0.01,
    duration_days=5,
    backend='cpu',
    use_float64=True,
    stochastic_threshold=50.0,
    tau_leap_dt=1.0,
    output_freq_days=1.0,
    stochastic_seed=42,
)

print(f"\nConfiguration:")
print(f"  n_nodes: {config.n_nodes}")
print(f"  n_age: {config.n_age}")
print(f"  n_bins: {config.n_bins}")
print(f"  seed_n_infections: {config.seed_n_infections}")

# Initialize Taichi
initialize_taichi_stochastic(config)

# Setup spatial structure
print("\n" + "="*70)
print("Setup Phase")
print("="*70)

positions = generate_node_positions(config.n_nodes, seed=42)
distances = compute_distance_matrix(positions)
node_pops = generate_node_populations(
    config.n_nodes, config.N_total,
    config.pop_dist_type, config.pop_dist_params,
    seed=42
)
betas = generate_node_betas(
    config.n_nodes, config.beta_mean, config.beta_variance,
    config.beta_dist_type, seed=42
)
network = build_gravity_network(
    node_pops, distances,
    a=config.gravity_a, b=config.gravity_b,
    c=config.gravity_c, k=config.gravity_k
)

print(f"Node populations: {node_pops}")
print(f"Beta values: {betas}")
print(f"Network:\n{network}")

# Check for zeros in node_pops (could cause division by zero)
if np.any(node_pops == 0):
    print("WARNING: Zero population detected!")
else:
    print("✓ All node populations > 0")

# Initialize state
age_labels, age_counts, aging_rates = setup_age_structure(config.n_age, config.N_total)

# Build heterogeneity distributions
sigma_ln = np.sqrt(np.log(1 + config.theta_variance / config.theta_mean**2))
mu_ln = np.log(config.theta_mean) - 0.5 * sigma_ln**2
s_dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))
theta_vals = np.linspace(s_dist.ppf(0.01), s_dist.ppf(0.99), config.n_bins)
dtheta = theta_vals[1] - theta_vals[0]

phi_dist = gamma(a=config.phi_shape, scale=config.phi_scale)
phi_vals = np.linspace(phi_dist.ppf(0.01), phi_dist.ppf(0.99), config.n_bins)
dphi = phi_vals[1] - phi_vals[0]

P = build_P(config.n_bins, width=config.P_width)

aux = {
    'theta_vals': theta_vals,
    'phi_vals': phi_vals,
    'dtheta': dtheta,
    'dphi': dphi,
    'P': P,
    'aging_rates': aging_rates,
}

print(f"\ntheta_vals: {theta_vals}")
print(f"phi_vals: {phi_vals}")
print(f"P matrix shape: {P.shape}, sum of rows: {P.sum(axis=1)}")

# Check for NaN/inf in distributions
if np.any(np.isnan(theta_vals)) or np.any(np.isinf(theta_vals)):
    print("ERROR: NaN/inf in theta_vals!")
if np.any(np.isnan(phi_vals)) or np.any(np.isinf(phi_vals)):
    print("ERROR: NaN/inf in phi_vals!")
if np.any(np.isnan(P)) or np.any(np.isinf(P)):
    print("ERROR: NaN/inf in P matrix!")

# Initialize compartments
S_np = np.zeros((config.n_nodes, config.n_age, config.n_bins))
E_np = np.zeros((config.n_nodes, config.n_age, config.n_bins))
I_np = np.zeros((config.n_nodes, config.n_age, config.n_bins))
R_np = np.zeros((config.n_nodes, config.n_age))

np.random.seed(123)
for node in range(config.n_nodes):
    age_counts_node = age_counts * (node_pops[node] / config.N_total)

    for age_idx in range(config.n_age):
        S_age_theta = s_dist.pdf(theta_vals)
        S_age_theta = S_age_theta / S_age_theta.sum() * age_counts_node[age_idx]

        if node == config.seed_node_idx and age_idx == 0:
            S_age_theta = S_age_theta / S_age_theta.sum() * (age_counts_node[age_idx] - config.seed_n_infections)

        S_np[node, age_idx, :] = S_age_theta

    if node == config.seed_node_idx:
        I_age_phi = phi_dist.pdf(phi_vals)
        I_age_phi = I_age_phi / I_age_phi.sum() * config.seed_n_infections
        I_np[node, 0, :] = I_age_phi

print(f"\nInitial S total: {S_np.sum():.2f}")
print(f"Initial E total: {E_np.sum():.2f}")
print(f"Initial I total: {I_np.sum():.2f}")
print(f"Initial R total: {R_np.sum():.2f}")
print(f"Total population: {S_np.sum() + E_np.sum() + I_np.sum() + R_np.sum():.2f}")

# Check for NaN/inf in initial state
if np.any(np.isnan(S_np)) or np.any(np.isinf(S_np)):
    print("ERROR: NaN/inf in initial S!")
if np.any(np.isnan(I_np)) or np.any(np.isinf(I_np)):
    print("ERROR: NaN/inf in initial I!")

# Create GPU state
gpu_state = TaichiSEIRState(config.n_nodes, config.n_age, config.n_bins, config.use_float64)
gpu_state.load_from_numpy(S_np, E_np, I_np, R_np, aux)
gpu_state.load_network(network, betas, node_pops)

# Additional fields
dtype_ti = ti.f64 if config.use_float64 else ti.f32
I_total_field = ti.field(dtype=dtype_ti, shape=config.n_nodes)

# RK4 temporary storage
k1_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k1_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k1_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k1_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

k2_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k2_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k2_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k2_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

k3_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k3_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k3_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k3_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

k4_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k4_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k4_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
k4_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

S_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
E_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
I_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
R_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

# Manual time-stepping with diagnostics
print("\n" + "="*70)
print("Time-Stepping Phase")
print("="*70)

t = 0.0
for step in range(3):  # Just do 3 steps
    print(f"\n--- Step {step}, t={t:.2f} ---")

    # Extract current state
    S_curr = gpu_state.S.to_numpy()
    E_curr = gpu_state.E.to_numpy()
    I_curr = gpu_state.I.to_numpy()
    R_curr = gpu_state.R.to_numpy()

    print(f"Before step: S={S_curr.sum():.2f}, E={E_curr.sum():.2f}, I={I_curr.sum():.2f}, R={R_curr.sum():.2f}")

    if np.any(np.isnan(S_curr)):
        print(f"ERROR: NaN in S before step {step}!")
        break
    if np.any(np.isnan(I_curr)):
        print(f"ERROR: NaN in I before step {step}!")
        break

    # Compute FOI
    compute_infectious_pressure_kernel(gpu_state.I, gpu_state.phi_vals, dphi, gpu_state.infectious_pressure)
    compute_foi_kernel(gpu_state.network, gpu_state.betas, gpu_state.node_pops,
                      gpu_state.infectious_pressure, gpu_state.foi)

    # Check FOI
    foi_vals = gpu_state.foi.to_numpy()
    print(f"FOI: {foi_vals}")

    if np.any(np.isnan(foi_vals)):
        print("ERROR: NaN in FOI!")
        break
    if np.any(np.isinf(foi_vals)):
        print("ERROR: Inf in FOI!")
        break

    # Classify regimes
    compute_I_totals(gpu_state.I, I_total_field)
    I_totals = I_total_field.to_numpy()
    print(f"I_total by node: {I_totals}")

    # Advance stochastic nodes
    stochastic_step_kernel(
        gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
        gpu_state.foi, config.tau_leap_dt, config.stochastic_threshold, I_total_field,
        gpu_state.theta_vals, gpu_state.P,
        config.sigma_rate, config.gamma_rate, gpu_state.aging_rates
    )

    # Check after stochastic step
    S_after_stoch = gpu_state.S.to_numpy()
    I_after_stoch = gpu_state.I.to_numpy()

    if np.any(np.isnan(S_after_stoch)):
        print("ERROR: NaN in S after stochastic step!")
        break
    if np.any(np.isnan(I_after_stoch)):
        print("ERROR: NaN in I after stochastic step!")
        break

    print(f"After stochastic: S={S_after_stoch.sum():.2f}, I={I_after_stoch.sum():.2f}")

    # Advance deterministic nodes with RK4
    deterministic_step_rk4_kernel(
        gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
        gpu_state.foi, config.tau_leap_dt, config.stochastic_threshold, I_total_field,
        gpu_state.theta_vals, gpu_state.P,
        config.sigma_rate, config.gamma_rate, gpu_state.aging_rates,
        k1_S, k1_E, k1_I, k1_R,
        k2_S, k2_E, k2_I, k2_R,
        k3_S, k3_E, k3_I, k3_R,
        k4_S, k4_E, k4_I, k4_R,
        S_temp, E_temp, I_temp, R_temp
    )

    # Check after RK4 step
    S_after_rk4 = gpu_state.S.to_numpy()
    E_after_rk4 = gpu_state.E.to_numpy()
    I_after_rk4 = gpu_state.I.to_numpy()
    R_after_rk4 = gpu_state.R.to_numpy()

    print(f"After RK4: S={S_after_rk4.sum():.2f}, E={E_after_rk4.sum():.2f}, I={I_after_rk4.sum():.2f}, R={R_after_rk4.sum():.2f}")

    if np.any(np.isnan(S_after_rk4)):
        print("ERROR: NaN in S after RK4 step!")
        print(f"S values:\n{S_after_rk4}")
        break
    if np.any(np.isnan(I_after_rk4)):
        print("ERROR: NaN in I after RK4 step!")
        print(f"I values:\n{I_after_rk4}")
        break

    t += config.tau_leap_dt

print("\n" + "="*70)
print("Debug complete")
print("="*70)
