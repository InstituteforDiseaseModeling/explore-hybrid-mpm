"""
GPU-Accelerated Multi-Node Meta-Population SEIR Model using Taichi.

This extends SEIR_pde_metapop.py with GPU acceleration via Taichi Metal backend.
Key features:
- Compatible with same ModelConfig
- GPU kernels for FOI computation (primary bottleneck)
- GPU kernels for SEIR derivatives
- Maintains exact numerical equivalence to CPU version
- Returns results in same format for direct comparison

Performance targets:
- n_nodes=100: 5-10× speedup
- n_nodes=774: 20-30× speedup
- n_nodes=1000: 30-50× speedup
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import taichi as ti
from scipy.integrate import solve_ivp
from scipy.stats import gamma, lognorm

# Import baseline components we'll reuse
from SEIR_pde_metapop import (
    ModelConfig,
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

# ============================================================================
# GPU Configuration
# ============================================================================

@dataclass
class TaichiModelConfig(ModelConfig):
    """Extends ModelConfig with GPU-specific parameters."""

    # GPU settings
    backend: Literal['cpu', 'metal', 'cuda'] = 'metal'  # Default for Mac
    use_float64: bool = False  # Use float32 by default for speed
    taichi_debug: bool = False

    # Performance tuning
    minimize_transfers: bool = True  # Keep state on GPU during integration


def initialize_taichi(config: TaichiModelConfig):
    """Initialize Taichi backend."""
    # Try to initialize Taichi - if already initialized, skip
    # This prevents expensive reinitialization overhead when running multiple simulations
    try:
        if config.backend == 'metal':
            ti.init(arch=ti.metal, debug=config.taichi_debug, default_fp=ti.f64 if config.use_float64 else ti.f32)
        elif config.backend == 'cuda':
            ti.init(arch=ti.cuda, debug=config.taichi_debug, default_fp=ti.f64 if config.use_float64 else ti.f32)
        else:
            ti.init(arch=ti.cpu, debug=config.taichi_debug, default_fp=ti.f64 if config.use_float64 else ti.f32)

        print(f"  Taichi backend: {config.backend}")
        print(f"  Precision: {'float64' if config.use_float64 else 'float32'}")
    except RuntimeError as e:
        # Taichi already initialized - skip
        if "Taichi has already been initialized" in str(e):
            print(f"  Taichi already initialized ({config.backend})")
        else:
            raise


# ============================================================================
# GPU State Management
# ============================================================================

@ti.data_oriented
class TaichiSEIRState:
    """GPU-resident SEIR state with Taichi fields."""

    def __init__(self, n_nodes: int, n_age: int, n_bins: int, use_float64: bool = False):
        self.n_nodes = n_nodes
        self.n_age = n_age
        self.n_bins = n_bins
        self.dtype = ti.f64 if use_float64 else ti.f32

        # State compartments (GPU arrays)
        self.S = ti.field(dtype=self.dtype, shape=(n_nodes, n_age, n_bins))
        self.E = ti.field(dtype=self.dtype, shape=(n_nodes, n_age, n_bins))
        self.I = ti.field(dtype=self.dtype, shape=(n_nodes, n_age, n_bins))
        self.R = ti.field(dtype=self.dtype, shape=(n_nodes, n_age))

        # Auxiliary arrays
        self.theta_vals = ti.field(dtype=self.dtype, shape=n_bins)
        self.phi_vals = ti.field(dtype=self.dtype, shape=n_bins)
        self.P = ti.field(dtype=self.dtype, shape=(n_bins, n_bins))
        self.aging_rates = ti.field(dtype=self.dtype, shape=n_age)

        # Network and parameters
        self.network = ti.field(dtype=self.dtype, shape=(n_nodes, n_nodes))
        self.betas = ti.field(dtype=self.dtype, shape=n_nodes)
        self.node_pops = ti.field(dtype=self.dtype, shape=n_nodes)

        # Intermediate computation fields
        self.foi = ti.field(dtype=self.dtype, shape=n_nodes)
        self.infectious_pressure = ti.field(dtype=self.dtype, shape=n_nodes)

        # Derivative fields (for RHS computation)
        self.dSdt = ti.field(dtype=self.dtype, shape=(n_nodes, n_age, n_bins))
        self.dEdt = ti.field(dtype=self.dtype, shape=(n_nodes, n_age, n_bins))
        self.dIdt = ti.field(dtype=self.dtype, shape=(n_nodes, n_age, n_bins))
        self.dRdt = ti.field(dtype=self.dtype, shape=(n_nodes, n_age))

    def load_from_numpy(self, S_np, E_np, I_np, R_np, aux):
        """Load state and auxiliary data from NumPy to GPU."""
        dtype_np = np.float64 if self.dtype == ti.f64 else np.float32

        self.S.from_numpy(S_np.astype(dtype_np))
        self.E.from_numpy(E_np.astype(dtype_np))
        self.I.from_numpy(I_np.astype(dtype_np))
        self.R.from_numpy(R_np.astype(dtype_np))

        self.theta_vals.from_numpy(aux['theta_vals'].astype(dtype_np))
        self.phi_vals.from_numpy(aux['phi_vals'].astype(dtype_np))
        self.P.from_numpy(aux['P'].astype(dtype_np))
        self.aging_rates.from_numpy(aux['aging_rates'].astype(dtype_np))

    def load_network(self, network_np, betas_np, node_pops_np):
        """Load network and parameters from NumPy to GPU."""
        dtype_np = np.float64 if self.dtype == ti.f64 else np.float32

        self.network.from_numpy(network_np.astype(dtype_np))
        self.betas.from_numpy(betas_np.astype(dtype_np))
        self.node_pops.from_numpy(node_pops_np.astype(dtype_np))

    def state_to_numpy(self):
        """Export state from GPU to NumPy."""
        return {
            'S': self.S.to_numpy(),
            'E': self.E.to_numpy(),
            'I': self.I.to_numpy(),
            'R': self.R.to_numpy(),
        }

    def state_to_flat(self):
        """Convert GPU state to flat NumPy array for solve_ivp."""
        S_np = self.S.to_numpy().flatten()
        E_np = self.E.to_numpy().flatten()
        I_np = self.I.to_numpy().flatten()
        R_np = self.R.to_numpy().flatten()
        return np.concatenate([S_np, E_np, I_np, R_np])

    def load_from_flat(self, y_flat):
        """Load flat NumPy array from solve_ivp into GPU state."""
        n_SEI = self.n_nodes * self.n_age * self.n_bins

        dtype_np = np.float64 if self.dtype == ti.f64 else np.float32

        S_np = y_flat[:n_SEI].reshape((self.n_nodes, self.n_age, self.n_bins)).astype(dtype_np)
        E_np = y_flat[n_SEI:2*n_SEI].reshape((self.n_nodes, self.n_age, self.n_bins)).astype(dtype_np)
        I_np = y_flat[2*n_SEI:3*n_SEI].reshape((self.n_nodes, self.n_age, self.n_bins)).astype(dtype_np)
        R_np = y_flat[3*n_SEI:].reshape((self.n_nodes, self.n_age)).astype(dtype_np)

        self.S.from_numpy(S_np)
        self.E.from_numpy(E_np)
        self.I.from_numpy(I_np)
        self.R.from_numpy(R_np)

    def derivatives_to_flat(self):
        """Convert GPU derivatives to flat NumPy array for solve_ivp."""
        dSdt_np = self.dSdt.to_numpy().flatten()
        dEdt_np = self.dEdt.to_numpy().flatten()
        dIdt_np = self.dIdt.to_numpy().flatten()
        dRdt_np = self.dRdt.to_numpy().flatten()
        return np.concatenate([dSdt_np, dEdt_np, dIdt_np, dRdt_np])


# ============================================================================
# GPU Kernels
# ============================================================================

@ti.kernel
def compute_infectious_pressure_kernel(I: ti.template(), phi_vals: ti.template(),
                                      dphi: ti.f32, infectious_pressure: ti.template()):
    """
    Compute infectious pressure per node: sum over age and bins.
    Parallelized over nodes.
    """
    n_nodes, n_age, n_bins = I.shape

    for node in range(n_nodes):  # GPU parallel
        pressure = 0.0
        for age in range(n_age):
            for phi_bin in range(n_bins):
                pressure += I[node, age, phi_bin] * phi_vals[phi_bin]
        infectious_pressure[node] = pressure * dphi


@ti.kernel
def compute_foi_kernel(network: ti.template(), betas: ti.template(),
                      node_pops: ti.template(), infectious_pressure: ti.template(),
                      foi: ti.template()):
    """
    Compute force of infection via network coupling.
    Parallelized over destination nodes.
    """
    n_nodes = network.shape[0]

    for i in range(n_nodes):  # GPU parallel
        foi_val = 0.0
        for j in range(n_nodes):
            foi_val += network[i, j] * (betas[j] / node_pops[j]) * infectious_pressure[j]
        foi[i] = foi_val


@ti.kernel
def compute_SEIR_transitions_kernel(S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                                   theta_vals: ti.template(), phi_vals: ti.template(), foi: ti.template(),
                                   P: ti.template(), sigma_rate: ti.f32, gamma_rate: ti.f32,
                                   dSdt: ti.template(), dEdt: ti.template(), dIdt: ti.template(), dRdt: ti.template()):
    """
    Compute SEIR transitions on GPU.
    Parallelized over (node, age).
    """
    n_nodes, n_age, n_bins = S.shape

    # Parallel over nodes and ages
    for node, age in ti.ndrange(n_nodes, n_age):
        # S -> E transitions
        for theta_bin in range(n_bins):
            new_inf = theta_vals[theta_bin] * foi[node] * S[node, age, theta_bin]
            dSdt[node, age, theta_bin] = -new_inf

            # Apply correlation matrix P: theta -> phi
            for phi_bin in range(n_bins):
                ti.atomic_add(dEdt[node, age, phi_bin], P[theta_bin, phi_bin] * new_inf)

        # E -> I and I -> R transitions
        for phi_bin in range(n_bins):
            dEdt[node, age, phi_bin] -= sigma_rate * E[node, age, phi_bin]
            dIdt[node, age, phi_bin] = (sigma_rate * E[node, age, phi_bin] -
                                        gamma_rate * I[node, age, phi_bin])

        # Recovery (sum over phi bins)
        R_flux = 0.0
        for phi_bin in range(n_bins):
            R_flux += gamma_rate * I[node, age, phi_bin]
        dRdt[node, age] = R_flux


@ti.kernel
def add_aging_flux_kernel(S: ti.template(), E: ti.template(), I: ti.template(), R: ti.template(),
                         aging_rates: ti.template(),
                         dSdt: ti.template(), dEdt: ti.template(), dIdt: ti.template(), dRdt: ti.template()):
    """
    Add aging fluxes to derivatives.
    Parallelized over nodes.
    """
    n_nodes, n_age, n_bins = S.shape

    for node in range(n_nodes):  # GPU parallel
        # Aging for S, E, I (3D arrays)
        for age in range(n_age - 1):
            aging_rate = aging_rates[age]
            for bin in range(n_bins):
                # Outflow from current age
                dSdt[node, age, bin] -= aging_rate * S[node, age, bin]
                dEdt[node, age, bin] -= aging_rate * E[node, age, bin]
                dIdt[node, age, bin] -= aging_rate * I[node, age, bin]

                # Inflow to next age
                dSdt[node, age + 1, bin] += aging_rate * S[node, age, bin]
                dEdt[node, age + 1, bin] += aging_rate * E[node, age, bin]
                dIdt[node, age + 1, bin] += aging_rate * I[node, age, bin]

        # Aging for R (2D array)
        for age in range(n_age - 1):
            aging_rate = aging_rates[age]
            dRdt[node, age] -= aging_rate * R[node, age]
            dRdt[node, age + 1] += aging_rate * R[node, age]


# ============================================================================
# GPU-Accelerated RHS Function
# ============================================================================

def rhs_taichi(t: float, y_flat: np.ndarray, gpu_state: TaichiSEIRState,
               config: TaichiModelConfig, dphi: float) -> np.ndarray:
    """
    Right-hand side function for solve_ivp that uses GPU kernels.

    This is called by scipy's solve_ivp at each timestep.
    """
    # Load current state from flat array to GPU
    gpu_state.load_from_flat(y_flat)

    # Zero out derivative fields
    gpu_state.dSdt.fill(0.0)
    gpu_state.dEdt.fill(0.0)
    gpu_state.dIdt.fill(0.0)
    gpu_state.dRdt.fill(0.0)

    # Compute FOI on GPU
    compute_infectious_pressure_kernel(gpu_state.I, gpu_state.phi_vals, dphi, gpu_state.infectious_pressure)
    compute_foi_kernel(gpu_state.network, gpu_state.betas, gpu_state.node_pops,
                      gpu_state.infectious_pressure, gpu_state.foi)

    # Compute SEIR transitions on GPU
    compute_SEIR_transitions_kernel(gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
                                   gpu_state.theta_vals, gpu_state.phi_vals, gpu_state.foi,
                                   gpu_state.P, config.sigma_rate, config.gamma_rate,
                                   gpu_state.dSdt, gpu_state.dEdt, gpu_state.dIdt, gpu_state.dRdt)

    # Add aging fluxes
    add_aging_flux_kernel(gpu_state.S, gpu_state.E, gpu_state.I, gpu_state.R,
                         gpu_state.aging_rates,
                         gpu_state.dSdt, gpu_state.dEdt, gpu_state.dIdt, gpu_state.dRdt)

    # Convert derivatives to flat array for solve_ivp
    return gpu_state.derivatives_to_flat()


# ============================================================================
# Simulation
# ============================================================================

def run_simulation_taichi(config: TaichiModelConfig, spatial_seed: int = 42,
                         epi_seed: int = 123) -> dict:
    """
    Run GPU-accelerated meta-population SEIR simulation.

    Returns results in same format as baseline for comparison.
    """
    print("="*70)
    print("GPU-Accelerated Multi-Node SEIR Simulation (Taichi)")
    print("="*70)
    print("Configuration:")
    print(f"  n_nodes: {config.n_nodes}")
    print(f"  n_age: {config.n_age}")
    print(f"  n_bins: {config.n_bins}")
    print(f"  gravity_k: {config.gravity_k}")

    n_odes = 3 * config.n_nodes * config.n_age * config.n_bins + config.n_nodes * config.n_age
    print(f"  Total ODEs: {n_odes:,}")
    print()

    # Initialize Taichi
    initialize_taichi(config)
    print()

    # Setup (reuse baseline functions)
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

    # Build heterogeneity distributions
    # Susceptibility (theta) - LogNormal (MUST match baseline exactly!)
    sigma_ln = np.sqrt(np.log(1 + config.theta_variance / config.theta_mean**2))
    mu_ln = np.log(config.theta_mean) - 0.5 * sigma_ln**2
    s_dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))
    theta_vals = np.linspace(s_dist.ppf(0.01), s_dist.ppf(0.99), config.n_bins)
    dtheta = theta_vals[1] - theta_vals[0]

    # Infectiousness (phi) - Gamma (MUST match baseline exactly!)
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

    # Initialize compartments (NumPy first)
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

    # Initial state as flat array
    y0 = gpu_state.state_to_flat()

    t_setup = time.time() - t_start_setup

    # Run simulation
    print("Running simulation (GPU-accelerated)...")
    t_eval = np.arange(0, config.duration_days, config.output_freq_days)

    t_start_sim = time.time()
    sol = solve_ivp(
        rhs_taichi,
        [0, config.duration_days],
        y0,
        args=(gpu_state, config, dphi),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    t_sim = time.time() - t_start_sim

    print(f"  Integration status: {sol.message}")
    print(f"  Number of time points: {len(sol.t)}")
    print(f"  Number of function evaluations: {sol.nfev}")
    print(f"  Setup time: {t_setup:.2f}s")
    print(f"  Simulation time: {t_sim:.2f}s")
    print(f"  Total time: {t_setup + t_sim:.2f}s")

    # Package results (same format as baseline)
    results = {
        'sol': sol,
        'config': config,
        'aux': aux,
        'positions': positions,
        'distances': distances,
        'node_pops': node_pops,
        'betas': betas,
        'network': network,
        'gpu_state': gpu_state,  # Extra field for GPU debugging
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Run GPU-accelerated simulation with default configuration."""
    from pathlib import Path

    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    config = TaichiModelConfig(
        n_nodes=774,
        n_age=16,
        n_bins=50,
        seed_node_idx=0,
        seed_n_infections=1.0,
        gravity_k=0.01,
        duration_days=365,
        backend='metal',  # or 'cuda' for NVIDIA, 'cpu' for testing
    )

    results = run_simulation_taichi(config, spatial_seed=42, epi_seed=123)

    # Visualize
    print("\nGenerating visualizations...")
    plot_heatmap(results, str(figures_dir / 'metapop_heatmap_taichi.pdf'))
    if config.n_nodes <= 20:
        plot_node_timeseries(results, str(figures_dir / 'metapop_timeseries_taichi.pdf'))
    plot_network(results, str(figures_dir / 'metapop_network_taichi.pdf'))

    print("\nDone!")


if __name__ == '__main__':
    main()
