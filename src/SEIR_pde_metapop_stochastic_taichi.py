"""
Hybrid Stochastic/Deterministic Multi-Node SEIR Model using Taichi GPU acceleration.

This extends SEIR_pde_metapop_taichi.py with hybrid stochastic/deterministic dynamics:
- Nodes with I_total >= threshold: deterministic ODE stepping
- Nodes with I_total < threshold: stochastic tau-leaping with Poisson sampling

Key features:
- FOI treated as constant over each time step dt (decouples network from within-node dynamics)
- Each node simulated independently within time step (GPU parallelizable)
- Automatic importation/fade-out via network coupling
- Optional overdispersion (Negative Binomial) for importation events
- Both continuous (P.T@) and discrete (multinomial) P matrix modes

Scientific behaviors:
- Realistic fade-out in small/peripheral nodes
- Stochastic reintroduction creating "stuttering chains"
- Higher variance in epidemic trajectories
- Long stochastic tail at low prevalence

Performance targets:
- n_nodes=100: ~1-2× slower than deterministic (small dt overhead)
- n_nodes=774: ~2-5× slower than deterministic
- Much faster than pure NumPy loops via GPU parallelization
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import taichi as ti
from scipy.stats import gamma, lognorm

# Import baseline components we'll reuse
from SEIR_pde_metapop import (
    build_gravity_network,
    build_P,
    compute_distance_matrix,
    generate_node_betas,
    generate_node_populations,
    generate_node_positions,
    plot_heatmap,
    plot_network,
    plot_node_timeseries,
    setup_age_structure,
)

# Import deterministic Taichi components
from SEIR_pde_metapop_taichi import (
    TaichiModelConfig,
    TaichiSEIRState,
    compute_foi_kernel,
    compute_infectious_pressure_kernel,
)

# ============================================================================
# Stochastic Configuration
# ============================================================================

@dataclass
class StochasticModelConfig(TaichiModelConfig):
    """Extends TaichiModelConfig with stochastic-specific parameters."""

    # Stochastic regime switching
    stochastic_threshold: float = 10.0  # I_total < this → stochastic
    # Alternative thresholds (TODO):
    # - Scaled: max(10, 0.001 * N_node) for population-dependent switching
    # - Prevalence: I_total / N_node < 0.0001 for prevalence-based switching

    # Deterministic integration method (for nodes above threshold)
    deterministic_method: str = 'rk4'  # Options: 'euler', 'rk2', 'rk4'
    # - 'euler': 1 stage, fastest, least accurate (1× tau-leap cost)
    # - 'rk2': 2 stages, good balance (2× tau-leap cost)
    # - 'rk4': 4 stages, most accurate (4× tau-leap cost)

    # Time-stepping
    tau_leap_dt: float = 0.5  # Internal time step (days)
    output_freq_days: float = 1.0  # Save output frequency

    # P matrix handling (theta→phi mapping)
    use_discrete_P: bool = False  # False=continuous (P.T@), True=multinomial

    # Overdispersion for importation (I=0 nodes)
    importation_overdispersion: Optional[float] = None  # If set, use NegBin instead of Poisson
    # Note: Only applied when I_total=0 for patchier importation dynamics

    # Random seed for reproducibility
    stochastic_seed: int = 42


def initialize_taichi_stochastic(config: StochasticModelConfig):
    """
    Initialize Taichi backend with random seed for stochastic simulation.

    IMPORTANT: Taichi does not allow changing the random seed after initialization.
    The seed can ONLY be set during ti.init(). This means we MUST reinitialize
    for each replicate with a different seed, which includes kernel recompilation overhead.
    """
    if config.backend == 'metal':
        ti.init(arch=ti.metal, debug=config.taichi_debug,
                default_fp=ti.f64 if config.use_float64 else ti.f32,
                random_seed=config.stochastic_seed)
    elif config.backend == 'cuda':
        ti.init(arch=ti.cuda, debug=config.taichi_debug,
                default_fp=ti.f64 if config.use_float64 else ti.f32,
                random_seed=config.stochastic_seed)
    else:
        ti.init(arch=ti.cpu, debug=config.taichi_debug,
                default_fp=ti.f64 if config.use_float64 else ti.f32,
                random_seed=config.stochastic_seed)

    print(f"  Taichi backend: {config.backend}")
    print(f"  Precision: {'float64' if config.use_float64 else 'float32'}")
    print(f"  Random seed: {config.stochastic_seed}")


# ============================================================================
# Manual RNG Implementation (PCG32)
# ============================================================================
# Taichi's Metal backend does not properly respect the random_seed parameter.
# We implement our own PRNG using PCG32 (Permuted Congruential Generator).
# This ensures reproducible stochastic simulations across all backends.

@ti.func
def pcg32_init(seed: ti.u32, sequence: ti.u32) -> ti.u64:
    """
    Initialize PCG32 state from seed and sequence.
    Returns 64-bit state value.
    """
    state = ti.u64(0)
    state = state * ti.u64(6364136223846793005) + (ti.u64(sequence) << 1 | ti.u64(1))
    state = state * ti.u64(6364136223846793005) + ti.u64(seed)
    return state


@ti.func
def pcg32_random(state: ti.template()) -> ti.u32:
    """
    Generate next random uint32 using PCG32 algorithm.
    Updates state in-place.
    Returns random uint32.
    """
    oldstate = state[None]
    # LCG step
    state[None] = oldstate * ti.u64(6364136223846793005) + ti.u64(1)
    # XSH-RR output function
    xorshifted = ti.u32(((oldstate >> 18) ^ oldstate) >> 27)
    rot = ti.u32(oldstate >> 59)
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1) & 31))


@ti.func
def pcg32_random_float(state: ti.template()) -> ti.f32:
    """
    Generate random float in [0, 1) using PCG32.
    """
    u = pcg32_random(state)
    return ti.cast(u, ti.f32) / ti.cast(4294967296.0, ti.f32)  # u / 2^32


# ============================================================================
# Stochastic GPU Kernels
# ============================================================================

@ti.func
def pcg32_random_float_inplace(rng_states: ti.template(), node_idx: ti.i32) -> ti.f32:
    """
    Generate random float using PCG32 and update state in-place in global array.

    Args:
        rng_states: Global array of RNG states (one per node)
        node_idx: Index of the node (selects which RNG state to use/update)

    Returns:
        Random float in [0, 1)
    """
    oldstate = rng_states[node_idx]
    # LCG step
    rng_states[node_idx] = oldstate * ti.u64(6364136223846793005) + ti.u64(1)
    # XSH-RR output function
    xorshifted = ti.u32(((oldstate >> 18) ^ oldstate) >> 27)
    rot = ti.u32(oldstate >> 59)
    u = (xorshifted >> rot) | (xorshifted << ((~rot + 1) & 31))
    return ti.cast(u, ti.f32) / ti.cast(4294967296.0, ti.f32)


@ti.func
def sample_poisson(lam: ti.f32, rng_states: ti.template(), node_idx: ti.i32) -> ti.f32:
    """
    Sample from Poisson distribution using Knuth's algorithm for small lambda,
    or Normal approximation for large lambda.

    Args:
        lam: Poisson rate parameter (lambda)
        rng_states: Global RNG states array
        node_idx: Node index for RNG state

    Returns:
        Sample from Poisson(lam)
    """
    result = 0.0

    if lam < 10.0:
        # Knuth's algorithm for small lambda
        L = ti.exp(-lam)
        k = 0.0
        p = 1.0

        # Loop until p drops below L
        for _ in range(1000):  # Max iterations to avoid infinite loop
            k += 1.0
            p *= pcg32_random_float_inplace(rng_states, node_idx)
            if p <= L:
                break

        result = k - 1.0

    elif lam < 100.0:
        # Normal approximation for moderate lambda: N(lambda, sqrt(lambda))
        # Use Box-Muller transform to generate normal
        u1 = pcg32_random_float_inplace(rng_states, node_idx)
        u2 = pcg32_random_float_inplace(rng_states, node_idx)
        z = ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265 * u2)
        result = ti.max(0.0, ti.round(lam + ti.sqrt(lam) * z))

    else:
        # For very large lambda, use deterministic approximation
        result = lam

    return result


@ti.kernel
def init_rng_states(rng_states: ti.template(), seed: ti.i32):
    """Initialize RNG state for each node using PCG32."""
    n_nodes = rng_states.shape[0]
    for node in range(n_nodes):
        rng_states[node] = pcg32_init(ti.u32(seed), ti.u32(node))


@ti.kernel
def compute_I_totals(I: ti.template(), I_total: ti.template()):
    """
    Compute total infectious count per node for regime classification.
    I_total[node] = sum over all ages and phi bins.
    """
    n_nodes, n_age, n_bins = I.shape

    for node in range(n_nodes):
        total = 0.0
        for age in range(n_age):
            for phi_bin in range(n_bins):
                total += I[node, age, phi_bin]
        I_total[node] = total


@ti.kernel
def stochastic_step_kernel(S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                          foi: ti.template(), dt: ti.f32, threshold: ti.f32, I_total: ti.template(),
                          theta_vals: ti.template(), P: ti.template(),
                          sigma_rate: ti.f32, gamma_rate: ti.f32, aging_rates: ti.template(),
                          rng_states: ti.template()):
    """
    Tau-leaping step for stochastic nodes (I_total < threshold).

    For each node below threshold:
    1. Sample S→E transitions from Poisson(theta * foi * S * dt)
    2. Apply P matrix to map theta bins → phi bins
    3. Sample E→I transitions from Poisson(sigma * E * dt)
    4. Sample I→R transitions from Poisson(gamma * I * dt)
    5. Apply aging fluxes deterministically (rates are tiny)
    6. Clamp to prevent negative values

    Parallelized over nodes.
    """
    n_nodes, n_age, n_bins = S.shape

    for node in range(n_nodes):
        # Only process stochastic nodes
        if I_total[node] < threshold:
            # === S→E transitions (new infections) ===
            for age in range(n_age):
                for theta_bin in range(n_bins):
                    rate = theta_vals[theta_bin] * foi[node] * S[node, age, theta_bin]

                    # Sample from Poisson using our RNG (modifies rng_states[node] in-place)
                    new_inf = sample_poisson(rate * dt, rng_states, node)
                    new_inf = ti.min(new_inf, S[node, age, theta_bin])  # Can't exceed S

                    S[node, age, theta_bin] -= new_inf

                    # Apply P matrix: theta → phi
                    # Continuous mode: fractional assignment
                    for phi_bin in range(n_bins):
                        E[node, age, phi_bin] += P[theta_bin, phi_bin] * new_inf

            # === E→I transitions (progression) ===
            for age in range(n_age):
                for phi_bin in range(n_bins):
                    rate = sigma_rate * E[node, age, phi_bin]
                    progressions = sample_poisson(rate * dt, rng_states, node)
                    progressions = ti.min(progressions, E[node, age, phi_bin])

                    E[node, age, phi_bin] -= progressions
                    I[node, age, phi_bin] += progressions

            # === I→R transitions (recovery) ===
            for age in range(n_age):
                for phi_bin in range(n_bins):
                    rate = gamma_rate * I[node, age, phi_bin]
                    recoveries = sample_poisson(rate * dt, rng_states, node)
                    recoveries = ti.min(recoveries, I[node, age, phi_bin])

                    I[node, age, phi_bin] -= recoveries
                    R[node, age] += recoveries

            # === Aging fluxes (deterministic, rates ~1/1825 days) ===
            # S, E, I aging (3D arrays)
            for age in range(n_age - 1):
                aging_rate = aging_rates[age]
                for bin_idx in range(n_bins):
                    # Compute flux
                    S_flux = aging_rate * S[node, age, bin_idx] * dt
                    E_flux = aging_rate * E[node, age, bin_idx] * dt
                    I_flux = aging_rate * I[node, age, bin_idx] * dt

                    # Apply flux
                    S[node, age, bin_idx] -= S_flux
                    S[node, age + 1, bin_idx] += S_flux

                    E[node, age, bin_idx] -= E_flux
                    E[node, age + 1, bin_idx] += E_flux

                    I[node, age, bin_idx] -= I_flux
                    I[node, age + 1, bin_idx] += I_flux

            # R aging (2D array)
            for age in range(n_age - 1):
                aging_rate = aging_rates[age]
                R_flux = aging_rate * R[node, age] * dt
                R[node, age] -= R_flux
                R[node, age + 1] += R_flux

            # === Clamp negative values (can occur due to Poisson variance) ===
            for age in range(n_age):
                for bin_idx in range(n_bins):
                    S[node, age, bin_idx] = ti.max(S[node, age, bin_idx], 0.0)
                    E[node, age, bin_idx] = ti.max(E[node, age, bin_idx], 0.0)
                    I[node, age, bin_idx] = ti.max(I[node, age, bin_idx], 0.0)
                R[node, age] = ti.max(R[node, age], 0.0)
            # Note: RNG state is updated in-place by sample_poisson


@ti.func
def compute_node_derivatives(node_idx: ti.i32, n_age: ti.i32, n_bins: ti.i32,
                             S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                             foi_val: ti.f32, theta_vals: ti.template(), P: ti.template(),
                             sigma_rate: ti.f32, gamma_rate: ti.f32, aging_rates: ti.template(),
                             dS_out: ti.template(), dE_out: ti.template(), dI_out: ti.template(), dR_out: ti.template()):
    """
    Compute derivatives for a single node (for RK4 substeps).
    This is a ti.func so it can be called from within kernels.
    """
    # Zero derivatives
    for age, bin_idx in ti.ndrange(n_age, n_bins):
        dS_out[node_idx, age, bin_idx] = 0.0
        dE_out[node_idx, age, bin_idx] = 0.0
        dI_out[node_idx, age, bin_idx] = 0.0
    for age in range(n_age):
        dR_out[node_idx, age] = 0.0

    # S→E transitions
    for age in range(n_age):
        for theta_bin in range(n_bins):
            new_inf_rate = theta_vals[theta_bin] * foi_val * S[node_idx, age, theta_bin]
            dS_out[node_idx, age, theta_bin] = -new_inf_rate

            # Apply P matrix
            for phi_bin in range(n_bins):
                dE_out[node_idx, age, phi_bin] += P[theta_bin, phi_bin] * new_inf_rate

    # E→I and I→R transitions
    for age in range(n_age):
        for phi_bin in range(n_bins):
            dE_out[node_idx, age, phi_bin] -= sigma_rate * E[node_idx, age, phi_bin]
            dI_out[node_idx, age, phi_bin] = sigma_rate * E[node_idx, age, phi_bin] - gamma_rate * I[node_idx, age, phi_bin]
            dR_out[node_idx, age] += gamma_rate * I[node_idx, age, phi_bin]

    # Aging fluxes
    for age in range(n_age - 1):
        aging_rate = aging_rates[age]
        for bin_idx in range(n_bins):
            dS_out[node_idx, age, bin_idx] -= aging_rate * S[node_idx, age, bin_idx]
            dS_out[node_idx, age + 1, bin_idx] += aging_rate * S[node_idx, age, bin_idx]

            dE_out[node_idx, age, bin_idx] -= aging_rate * E[node_idx, age, bin_idx]
            dE_out[node_idx, age + 1, bin_idx] += aging_rate * E[node_idx, age, bin_idx]

            dI_out[node_idx, age, bin_idx] -= aging_rate * I[node_idx, age, bin_idx]
            dI_out[node_idx, age + 1, bin_idx] += aging_rate * I[node_idx, age, bin_idx]

        dR_out[node_idx, age] -= aging_rate * R[node_idx, age]
        dR_out[node_idx, age + 1] += aging_rate * R[node_idx, age]


@ti.kernel
def deterministic_step_rk4_kernel(S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                                  foi: ti.template(), dt: ti.f32, threshold: ti.f32, I_total: ti.template(),
                                  theta_vals: ti.template(), P: ti.template(),
                                  sigma_rate: ti.f32, gamma_rate: ti.f32, aging_rates: ti.template(),
                                  # Temporary storage for RK4
                                  k1_S: ti.template(), k1_E: ti.template(), k1_I: ti.template(), k1_R: ti.template(),
                                  k2_S: ti.template(), k2_E: ti.template(), k2_I: ti.template(), k2_R: ti.template(),
                                  k3_S: ti.template(), k3_E: ti.template(), k3_I: ti.template(), k3_R: ti.template(),
                                  k4_S: ti.template(), k4_E: ti.template(), k4_I: ti.template(), k4_R: ti.template(),
                                  S_temp: ti.template(), E_temp: ti.template(), I_temp: ti.template(), R_temp: ti.template()):
    """
    RK4 step for deterministic nodes (I_total >= threshold).
    Much more accurate than Euler, should match baseline RK45 closely.
    """
    n_nodes, n_age, n_bins = S.shape

    for node in range(n_nodes):
        # Only process deterministic nodes
        if I_total[node] >= threshold:
            # === RK4 Stage 1: k1 = f(y_n) ===
            compute_node_derivatives(node, n_age, n_bins, S, E, I, R,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   k1_S, k1_E, k1_I, k1_R)

            # Compute y + dt/2 * k1 (with clamping to prevent negative values)
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S_temp[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + 0.5 * dt * k1_S[node, age, bin_idx])
                E_temp[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + 0.5 * dt * k1_E[node, age, bin_idx])
                I_temp[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + 0.5 * dt * k1_I[node, age, bin_idx])
            for age in range(n_age):
                R_temp[node, age] = ti.max(0.0, R[node, age] + 0.5 * dt * k1_R[node, age])

            # === RK4 Stage 2: k2 = f(y_n + dt/2 * k1) ===
            compute_node_derivatives(node, n_age, n_bins, S_temp, E_temp, I_temp, R_temp,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   k2_S, k2_E, k2_I, k2_R)

            # Compute y + dt/2 * k2 (with clamping to prevent negative values)
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S_temp[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + 0.5 * dt * k2_S[node, age, bin_idx])
                E_temp[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + 0.5 * dt * k2_E[node, age, bin_idx])
                I_temp[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + 0.5 * dt * k2_I[node, age, bin_idx])
            for age in range(n_age):
                R_temp[node, age] = ti.max(0.0, R[node, age] + 0.5 * dt * k2_R[node, age])

            # === RK4 Stage 3: k3 = f(y_n + dt/2 * k2) ===
            compute_node_derivatives(node, n_age, n_bins, S_temp, E_temp, I_temp, R_temp,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   k3_S, k3_E, k3_I, k3_R)

            # Compute y + dt * k3 (with clamping to prevent negative values)
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S_temp[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + dt * k3_S[node, age, bin_idx])
                E_temp[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + dt * k3_E[node, age, bin_idx])
                I_temp[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + dt * k3_I[node, age, bin_idx])
            for age in range(n_age):
                R_temp[node, age] = ti.max(0.0, R[node, age] + dt * k3_R[node, age])

            # === RK4 Stage 4: k4 = f(y_n + dt * k3) ===
            compute_node_derivatives(node, n_age, n_bins, S_temp, E_temp, I_temp, R_temp,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   k4_S, k4_E, k4_I, k4_R)

            # === Final update: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4) ===
            # Apply update and clamp to prevent negative values
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + dt / 6.0 * (k1_S[node, age, bin_idx] + 2.0 * k2_S[node, age, bin_idx] +
                                                      2.0 * k3_S[node, age, bin_idx] + k4_S[node, age, bin_idx]))
                E[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + dt / 6.0 * (k1_E[node, age, bin_idx] + 2.0 * k2_E[node, age, bin_idx] +
                                                      2.0 * k3_E[node, age, bin_idx] + k4_E[node, age, bin_idx]))
                I[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + dt / 6.0 * (k1_I[node, age, bin_idx] + 2.0 * k2_I[node, age, bin_idx] +
                                                      2.0 * k3_I[node, age, bin_idx] + k4_I[node, age, bin_idx]))
            for age in range(n_age):
                R[node, age] = ti.max(0.0, R[node, age] + dt / 6.0 * (k1_R[node, age] + 2.0 * k2_R[node, age] +
                                            2.0 * k3_R[node, age] + k4_R[node, age]))


@ti.kernel
def deterministic_step_rk2_kernel(S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                                  foi: ti.template(), dt: ti.f32, threshold: ti.f32, I_total: ti.template(),
                                  theta_vals: ti.template(), P: ti.template(),
                                  sigma_rate: ti.f32, gamma_rate: ti.f32, aging_rates: ti.template(),
                                  # Temporary storage for RK2
                                  k1_S: ti.template(), k1_E: ti.template(), k1_I: ti.template(), k1_R: ti.template(),
                                  k2_S: ti.template(), k2_E: ti.template(), k2_I: ti.template(), k2_R: ti.template(),
                                  S_temp: ti.template(), E_temp: ti.template(), I_temp: ti.template(), R_temp: ti.template()):
    """
    RK2 (Midpoint) step for deterministic nodes (I_total >= threshold).
    2 stages instead of 4 → ~2× faster than RK4, still 2nd-order accurate.
    """
    n_nodes, n_age, n_bins = S.shape

    for node in range(n_nodes):
        # Only process deterministic nodes
        if I_total[node] >= threshold:
            # === RK2 Stage 1: k1 = f(y_n) ===
            compute_node_derivatives(node, n_age, n_bins, S, E, I, R,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   k1_S, k1_E, k1_I, k1_R)

            # Compute y + dt/2 * k1 (with clamping to prevent negative values)
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S_temp[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + 0.5 * dt * k1_S[node, age, bin_idx])
                E_temp[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + 0.5 * dt * k1_E[node, age, bin_idx])
                I_temp[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + 0.5 * dt * k1_I[node, age, bin_idx])
            for age in range(n_age):
                R_temp[node, age] = ti.max(0.0, R[node, age] + 0.5 * dt * k1_R[node, age])

            # === RK2 Stage 2: k2 = f(y_n + dt/2 * k1) ===
            compute_node_derivatives(node, n_age, n_bins, S_temp, E_temp, I_temp, R_temp,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   k2_S, k2_E, k2_I, k2_R)

            # === Final update: y_{n+1} = y_n + dt * k2 ===
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + dt * k2_S[node, age, bin_idx])
                E[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + dt * k2_E[node, age, bin_idx])
                I[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + dt * k2_I[node, age, bin_idx])
            for age in range(n_age):
                R[node, age] = ti.max(0.0, R[node, age] + dt * k2_R[node, age])


@ti.kernel
def deterministic_step_euler_kernel(S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                                    foi: ti.template(), dt: ti.f32, threshold: ti.f32, I_total: ti.template(),
                                    theta_vals: ti.template(), P: ti.template(),
                                    sigma_rate: ti.f32, gamma_rate: ti.f32, aging_rates: ti.template(),
                                    # Temporary storage for derivatives
                                    dS: ti.template(), dE: ti.template(), dI: ti.template(), dR: ti.template()):
    """
    Forward Euler step for deterministic nodes (I_total >= threshold).
    1 stage → fastest, but least accurate (only 1st-order).
    """
    n_nodes, n_age, n_bins = S.shape

    for node in range(n_nodes):
        # Only process deterministic nodes
        if I_total[node] >= threshold:
            # Compute derivatives
            compute_node_derivatives(node, n_age, n_bins, S, E, I, R,
                                   foi[node], theta_vals, P, sigma_rate, gamma_rate, aging_rates,
                                   dS, dE, dI, dR)

            # Apply Euler update: y_{n+1} = y_n + dt * f(y_n)
            for age, bin_idx in ti.ndrange(n_age, n_bins):
                S[node, age, bin_idx] = ti.max(0.0, S[node, age, bin_idx] + dt * dS[node, age, bin_idx])
                E[node, age, bin_idx] = ti.max(0.0, E[node, age, bin_idx] + dt * dE[node, age, bin_idx])
                I[node, age, bin_idx] = ti.max(0.0, I[node, age, bin_idx] + dt * dI[node, age, bin_idx])
            for age in range(n_age):
                R[node, age] = ti.max(0.0, R[node, age] + dt * dR[node, age])


# ============================================================================
# Hybrid Integration Loop
# ============================================================================

def run_simulation_stochastic(config: StochasticModelConfig, spatial_seed: int = 42,
                              epi_seed: int = 123) -> dict:
    """
    Run hybrid stochastic/deterministic meta-population SEIR simulation.

    Algorithm:
    1. Setup spatial network and initial conditions (reuse from deterministic)
    2. Manual time-stepping loop with dt = tau_leap_dt
    3. At each time step:
       - Compute FOI globally (treat as constant over dt)
       - Classify nodes into stochastic vs deterministic regimes
       - Advance stochastic nodes via tau-leaping (Poisson sampling)
       - Advance deterministic nodes via Euler step
    4. Save output at requested frequency

    Returns results in same format as baseline for comparison.
    """
    print("="*70)
    print("Hybrid Stochastic/Deterministic SEIR Simulation (Taichi GPU)")
    print("="*70)
    print("Configuration:")
    print(f"  n_nodes: {config.n_nodes}")
    print(f"  n_age: {config.n_age}")
    print(f"  n_bins: {config.n_bins}")
    print(f"  gravity_k: {config.gravity_k}")
    print(f"  stochastic_threshold: {config.stochastic_threshold}")
    print(f"  deterministic_method: {config.deterministic_method}")
    print(f"  tau_leap_dt: {config.tau_leap_dt} days")
    print(f"  output_freq: {config.output_freq_days} days")

    n_odes = 3 * config.n_nodes * config.n_age * config.n_bins + config.n_nodes * config.n_age
    print(f"  Total ODEs: {n_odes:,}")
    print()

    # Initialize Taichi with random seed
    initialize_taichi_stochastic(config)
    print()

    # Setup spatial structure (reuse from deterministic)
    print("Setting up spatial structure...")
    t_start_setup = time.time()

    positions = generate_node_positions(config.n_nodes, seed=spatial_seed)
    distances = compute_distance_matrix(positions)
    node_pops = generate_node_populations(
        config.n_nodes, config.N_total,
        config.pop_dist_type, config.pop_dist_params,
        seed=spatial_seed
    )
    betas = generate_node_betas(
        config.n_nodes, config.beta_mean, config.beta_variance,
        config.beta_dist_type, seed=spatial_seed
    )
    network = build_gravity_network(
        node_pops, distances,
        a=config.gravity_a, b=config.gravity_b,
        c=config.gravity_c, k=config.gravity_k
    )

    print(f"  Node populations: min={node_pops.min():.0f}, max={node_pops.max():.0f}, mean={node_pops.mean():.0f}")
    print(f"  Beta values: min={betas.min():.2f}, max={betas.max():.2f}, mean={betas.mean():.2f}")
    print(f"  Network density: {(network > 0).sum() / network.size:.2%}")

    # Initialize state
    print("Initializing state...")
    age_labels, age_counts, aging_rates = setup_age_structure(config.n_age, config.N_total)

    # Build heterogeneity distributions (MUST match baseline exactly!)
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

    np.random.seed(epi_seed)
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

    # Create GPU state
    gpu_state = TaichiSEIRState(config.n_nodes, config.n_age, config.n_bins, config.use_float64)
    gpu_state.load_from_numpy(S_np, E_np, I_np, R_np, aux)
    gpu_state.load_network(network, betas, node_pops)

    # Additional fields for I_total tracking and RK4 temporary storage
    dtype_ti = ti.f64 if config.use_float64 else ti.f32
    I_total_field = ti.field(dtype=dtype_ti, shape=config.n_nodes)

    # RNG states for stochastic simulation (one per node)
    rng_states = ti.field(dtype=ti.u64, shape=config.n_nodes)
    # Initialize RNG state for each node using PCG32 (must be done in a kernel)
    init_rng_states(rng_states, config.stochastic_seed)

    # Allocate temporary storage based on deterministic method
    # Validate method selection
    if config.deterministic_method not in ['euler', 'rk2', 'rk4']:
        raise ValueError(f"Invalid deterministic_method: {config.deterministic_method}. Choose from: 'euler', 'rk2', 'rk4'")

    # All methods need at least k1 (derivative storage)
    k1_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
    k1_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
    k1_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
    k1_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

    # RK2 and RK4 need k2
    if config.deterministic_method in ['rk2', 'rk4']:
        k2_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k2_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k2_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k2_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

        S_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        E_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        I_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        R_temp = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))
    else:
        # Euler doesn't need these, but create dummy refs for consistency
        k2_S = k2_E = k2_I = k2_R = None
        S_temp = E_temp = I_temp = R_temp = None

    # RK4 needs k3 and k4
    if config.deterministic_method == 'rk4':
        k3_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k3_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k3_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k3_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))

        k4_S = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k4_E = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k4_I = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age, config.n_bins))
        k4_R = ti.field(dtype=dtype_ti, shape=(config.n_nodes, config.n_age))
    else:
        # RK2 and Euler don't need these
        k3_S = k3_E = k3_I = k3_R = None
        k4_S = k4_E = k4_I = k4_R = None

    t_setup = time.time() - t_start_setup

    # Manual time-stepping loop
    print("Running hybrid simulation...")
    print(f"  Total steps: {int(config.duration_days / config.tau_leap_dt):,}")

    t_start_sim = time.time()

    t_values = []
    y_values = []

    t = 0.0
    step = 0

    # Determine output times (match baseline: np.arange(0, duration, freq))
    output_times = list(np.arange(0, config.duration_days, config.output_freq_days))
    next_output_idx = 0

    # Save initial condition
    if next_output_idx < len(output_times) and output_times[next_output_idx] == 0:
        t_values.append(0.0)
        y_values.append(gpu_state.state_to_flat())
        next_output_idx += 1

    while t < config.duration_days:
        # Compute FOI globally (constant over this time step)
        compute_infectious_pressure_kernel(gpu_state.I, gpu_state.phi_vals, dphi, gpu_state.infectious_pressure)
        compute_foi_kernel(gpu_state.network, gpu_state.betas, gpu_state.node_pops,
                          gpu_state.infectious_pressure, gpu_state.foi)

        # Classify regimes
        compute_I_totals(gpu_state.I, I_total_field)

        # Advance stochastic nodes
        stochastic_step_kernel(
            gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
            gpu_state.foi, config.tau_leap_dt, config.stochastic_threshold, I_total_field,
            gpu_state.theta_vals, gpu_state.P,
            config.sigma_rate, config.gamma_rate, gpu_state.aging_rates,
            rng_states
        )

        # Advance deterministic nodes with selected method
        if config.deterministic_method == 'euler':
            deterministic_step_euler_kernel(
                gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
                gpu_state.foi, config.tau_leap_dt, config.stochastic_threshold, I_total_field,
                gpu_state.theta_vals, gpu_state.P,
                config.sigma_rate, config.gamma_rate, gpu_state.aging_rates,
                k1_S, k1_E, k1_I, k1_R  # Euler uses k1 for derivative storage
            )
        elif config.deterministic_method == 'rk2':
            deterministic_step_rk2_kernel(
                gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
                gpu_state.foi, config.tau_leap_dt, config.stochastic_threshold, I_total_field,
                gpu_state.theta_vals, gpu_state.P,
                config.sigma_rate, config.gamma_rate, gpu_state.aging_rates,
                k1_S, k1_E, k1_I, k1_R,
                k2_S, k2_E, k2_I, k2_R,
                S_temp, E_temp, I_temp, R_temp
            )
        else:  # rk4
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

        t += config.tau_leap_dt
        step += 1

        # Save output when we cross an output time
        while next_output_idx < len(output_times) and t >= output_times[next_output_idx]:
            t_values.append(output_times[next_output_idx])
            y_values.append(gpu_state.state_to_flat())
            next_output_idx += 1

        # Progress indicator
        if step % 100 == 0:
            print(f"  Step {step:,}, t={t:.1f} days", end='\r')

    t_sim = time.time() - t_start_sim
    print()  # Clear progress line

    print(f"  Total steps: {step:,}")
    print(f"  Setup time: {t_setup:.2f}s")
    print(f"  Simulation time: {t_sim:.2f}s")
    print(f"  Total time: {t_setup + t_sim:.2f}s")
    print(f"  Time per step: {t_sim / step * 1000:.2f}ms")

    # Package results (compatible with baseline format)
    # Create a mock sol object to match baseline interface
    class MockSol:
        def __init__(self, t, y):
            self.t = np.array(t)
            self.y = np.array(y).T  # Transpose to match solve_ivp format
            self.message = "Hybrid stochastic simulation completed"
            self.nfev = step

    sol = MockSol(t_values, y_values)

    results = {
        'sol': sol,
        'config': config,
        'aux': aux,
        'positions': positions,
        'distances': distances,
        'node_pops': node_pops,
        'betas': betas,
        'network': network,
        'gpu_state': gpu_state,
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Run hybrid stochastic simulation with default configuration."""
    from pathlib import Path

    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    config = StochasticModelConfig(
        n_nodes=774,
        n_age=16,
        n_bins=50,
        seed_node_idx=0,
        seed_n_infections=150.0,  # High enough to start in deterministic mode
        gravity_k=0.01,
        duration_days=365,
        backend='metal',
        stochastic_threshold=100.0,
        tau_leap_dt=1.0,
        output_freq_days=1.0,
        stochastic_seed=42,
    )

    results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)

    # Visualize
    print("\nGenerating visualizations...")
    plot_heatmap(results, str(figures_dir / 'metapop_heatmap_stochastic.pdf'))
    if config.n_nodes <= 20:
        plot_node_timeseries(results, str(figures_dir / 'metapop_timeseries_stochastic.pdf'))
    plot_network(results, str(figures_dir / 'metapop_network_stochastic.pdf'))

    print("\nDone!")


if __name__ == '__main__':
    main()
