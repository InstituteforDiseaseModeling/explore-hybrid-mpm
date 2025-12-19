"""
Debug RK4 NaN issue - trace where NaN first appears.
"""

import numpy as np
from dataclasses import asdict
from SEIR_pde_metapop_stochastic_taichi import (
    StochasticModelConfig, initialize_taichi_stochastic, TaichiSEIRState,
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
print("Debug: RK4 NaN Issue")
print("="*70)

# Configuration that triggers NaN (deterministic mode)
config = StochasticModelConfig(
    n_nodes=2,
    n_age=2,
    n_bins=3,
    seed_node_idx=0,
    seed_n_infections=150.0,  # Above threshold=100 → deterministic
    gravity_k=0.01,
    duration_days=10,
    backend='cpu',
    use_float64=True,
    stochastic_threshold=100.0,
    tau_leap_dt=1.0,
    output_freq_days=1.0,
    stochastic_seed=42,
)

print(f"\nConfiguration:")
print(f"  n_nodes: {config.n_nodes}")
print(f"  seed_n_infections: {config.seed_n_infections}")
print(f"  stochastic_threshold: {config.stochastic_threshold}")
print(f"  → Seed node will be in DETERMINISTIC mode (RK4)")

# Initialize Taichi
initialize_taichi_stochastic(config)

# Setup
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

age_labels, age_counts, aging_rates = setup_age_structure(config.n_age, config.N_total)

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

print(f"\nInitial state:")
print(f"  S total: {S_np.sum():.2f}")
print(f"  I total: {I_np.sum():.2f}")
print(f"  I in seed node: {I_np[0].sum():.2f}")

# Create GPU state
gpu_state = TaichiSEIRState(config.n_nodes, config.n_age, config.n_bins, config.use_float64)
gpu_state.load_from_numpy(S_np, E_np, I_np, R_np, aux)
gpu_state.load_network(network, betas, node_pops)

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
print("Time-Stepping with RK4")
print("="*70)

t = 0.0
for step in range(5):
    print(f"\n--- Step {step}, t={t:.2f} ---")

    S_before = gpu_state.S.to_numpy()
    E_before = gpu_state.E.to_numpy()
    I_before = gpu_state.I.to_numpy()

    print(f"Before: S={S_before.sum():.2f}, E={E_before.sum():.2f}, I={I_before.sum():.2f}")

    if np.any(np.isnan(S_before)) or np.any(np.isnan(I_before)):
        print("ERROR: NaN detected BEFORE step!")
        break

    # Compute FOI
    compute_infectious_pressure_kernel(gpu_state.I, gpu_state.phi_vals, dphi, gpu_state.infectious_pressure)
    compute_foi_kernel(gpu_state.network, gpu_state.betas, gpu_state.node_pops,
                      gpu_state.infectious_pressure, gpu_state.foi)

    foi_vals = gpu_state.foi.to_numpy()
    print(f"FOI: {foi_vals}")

    if np.any(np.isnan(foi_vals)) or np.any(np.isinf(foi_vals)):
        print("ERROR: NaN/Inf in FOI!")
        break

    # Classify regimes
    compute_I_totals(gpu_state.I, I_total_field)
    I_totals = I_total_field.to_numpy()
    print(f"I_total: {I_totals} (threshold={config.stochastic_threshold})")

    # Only RK4 (no stochastic)
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

    S_after = gpu_state.S.to_numpy()
    E_after = gpu_state.E.to_numpy()
    I_after = gpu_state.I.to_numpy()
    R_after = gpu_state.R.to_numpy()

    print(f"After: S={S_after.sum():.2f}, E={E_after.sum():.2f}, I={I_after.sum():.2f}, R={R_after.sum():.2f}")

    if np.any(np.isnan(S_after)):
        print("\n❌ ERROR: NaN in S after RK4!")
        print(f"S min: {np.nanmin(S_after)}, max: {np.nanmax(S_after)}")
        print(f"First NaN at: {np.where(np.isnan(S_after))}")
        # Check intermediate stages
        print(f"\nChecking k1_S:")
        k1_S_np = k1_S.to_numpy()
        if np.any(np.isnan(k1_S_np)):
            print(f"  NaN in k1_S!")
        print(f"\nChecking S_temp (used for k2):")
        S_temp_np = S_temp.to_numpy()
        if np.any(np.isnan(S_temp_np)):
            print(f"  NaN in S_temp!")
        if np.any(S_temp_np < 0):
            print(f"  S_temp has negative values! Min: {S_temp_np.min()}")
        break

    if np.any(np.isnan(I_after)):
        print("\n❌ ERROR: NaN in I after RK4!")
        break

    # Check for negative values
    if np.any(S_after < -1e-6):
        print(f"⚠ WARNING: Negative S values! Min: {S_after.min()}")
    if np.any(I_after < -1e-6):
        print(f"⚠ WARNING: Negative I values! Min: {I_after.min()}")

    t += config.tau_leap_dt

print("\n" + "="*70)
print("Debug complete")
print("="*70)
