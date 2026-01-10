"""
Multi-Node Meta-Population SEIR PDE Model

Extends SEIR_pde.py to multiple spatially-coupled nodes with network-based
force of infection. Each node maintains full age-stratified heterogeneity
structure (theta, phi distributions with correlation).

Author: Dan Klein
Date: 2025-12-17
"""

from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gamma, lognorm, powerlaw

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for multi-node SEIR meta-population model."""

    # Spatial parameters
    n_nodes: int = 5
    seed_node_idx: int = 0
    seed_n_infections: float = 1000.0

    # Population distribution
    pop_dist_type: Literal['lognormal', 'powerlaw', 'uniform'] = 'lognormal'
    pop_dist_params: dict | None = None  # e.g., {'mu': 10, 'sigma': 1.5} for lognormal
    N_total: int = 186_763

    # Heterogeneity discretization
    n_age: int = 16
    n_bins: int = 50

    # Epidemiological parameters
    sigma_rate: float = 1/3  # E -> I rate (3 day incubation)
    gamma_rate: float = 1/24  # I -> R rate (24 day infectious period)

    # Beta distribution (per node)
    beta_dist_type: Literal['lognormal', 'uniform'] = 'lognormal'
    beta_mean: float = 10.0
    beta_variance: float = 4.0

    # Gravity model parameters
    gravity_a: float = 1.0  # Exponent on origin population
    gravity_b: float = 1.0  # Exponent on destination population
    gravity_c: float = 2.0  # Exponent on distance (higher = faster decay)
    gravity_k: float = 1.0  # Spatial coupling strength (0=local only, 0.01-0.25=calibrated range)

    # Susceptibility/Infectiousness distributions
    theta_mean: float = 1.0
    theta_variance: float = 4.0
    phi_shape: float = 1.0
    phi_scale: Optional[float] = None  # Defaults to beta_mean / 24

    # Correlation structure
    P_width: float = 9.9242  # Achieves ~0.8 correlation for n_bins=50

    # Simulation parameters
    duration_days: int = 365 * 3
    output_freq_days: float = 365 / 52  # Weekly

    def __post_init__(self):
        if self.pop_dist_params is None:
            if self.pop_dist_type == 'lognormal':
                self.pop_dist_params = {'mu': 10, 'sigma': 1.5}
            elif self.pop_dist_type == 'powerlaw':
                self.pop_dist_params = {'a': 2.0}
            else:
                self.pop_dist_params = {}

        if self.phi_scale is None:
            self.phi_scale = self.beta_mean / 24


# ============================================================================
# State Management
# ============================================================================

@dataclass
class MetaPopState:
    """
    State vector wrapper for multi-node SEIR model.

    Provides named access to compartments while maintaining flat numpy array
    for ODE solver compatibility.
    """
    y: np.ndarray
    n_nodes: int
    n_age: int
    n_bins: int

    @property
    def S(self) -> np.ndarray:
        """Susceptible by (node, age, theta). Shape: (n_nodes, n_age, n_bins)"""
        end = self.n_nodes * self.n_age * self.n_bins
        return self.y[:end].reshape((self.n_nodes, self.n_age, self.n_bins))

    @property
    def E(self) -> np.ndarray:
        """Exposed by (node, age, phi). Shape: (n_nodes, n_age, n_bins)"""
        start = self.n_nodes * self.n_age * self.n_bins
        end = 2 * self.n_nodes * self.n_age * self.n_bins
        return self.y[start:end].reshape((self.n_nodes, self.n_age, self.n_bins))

    @property
    def I(self) -> np.ndarray:
        """Infectious by (node, age, phi). Shape: (n_nodes, n_age, n_bins)"""
        start = 2 * self.n_nodes * self.n_age * self.n_bins
        end = 3 * self.n_nodes * self.n_age * self.n_bins
        return self.y[start:end].reshape((self.n_nodes, self.n_age, self.n_bins))

    @property
    def R(self) -> np.ndarray:
        """Recovered by (node, age). Shape: (n_nodes, n_age)"""
        start = 3 * self.n_nodes * self.n_age * self.n_bins
        end = start + self.n_nodes * self.n_age
        return self.y[start:end].reshape((self.n_nodes, self.n_age))

    @classmethod
    def state_size(cls, n_nodes: int, n_age: int, n_bins: int) -> int:
        """Calculate total state vector size."""
        return 3 * n_nodes * n_age * n_bins + n_nodes * n_age


# ============================================================================
# Correlation Structure
# ============================================================================

def build_P(n_bins: int, width: float = 9.9242) -> np.ndarray:
    """
    Build conditional probability matrix P(phi | theta).

    Uses Gaussian kernel: P(phi_j | theta_i) ∝ exp(-(i-j)^2 / (2*width^2))

    Args:
        n_bins: Number of discretization bins
        width: Kernel width (9.9242 achieves ~0.8 correlation for n_bins=50)

    Returns:
        P: (n_bins, n_bins) stochastic matrix
    """
    P = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            P[i, j] = np.exp(-((i - j)**2) / (2 * width**2))
        P[i, :] /= P[i, :].sum()
    return P


# ============================================================================
# Spatial Setup
# ============================================================================

def generate_node_positions(n_nodes: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random 2D positions for nodes.

    Args:
        n_nodes: Number of nodes
        seed: Random seed for reproducibility

    Returns:
        positions: (n_nodes, 2) array of (x, y) coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 100, size=(n_nodes, 2))


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Args:
        positions: (n_nodes, 2) array of coordinates

    Returns:
        distances: (n_nodes, n_nodes) symmetric matrix
    """
    n_nodes = positions.shape[0]
    distances = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                distances[i, j] = np.linalg.norm(positions[i] - positions[j])
            else:
                distances[i, j] = 1.0  # Avoid division by zero
    return distances


def generate_node_populations(n_nodes: int, N_total: int, dist_type: str,
                              dist_params: dict | None, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate node populations from specified distribution.

    Args:
        n_nodes: Number of nodes
        N_total: Total population across all nodes
        dist_type: 'lognormal', 'powerlaw', or 'uniform'
        dist_params: Distribution parameters (or None for defaults)
        seed: Random seed

    Returns:
        populations: (n_nodes,) array, sums to N_total
    """
    if seed is not None:
        np.random.seed(seed)

    if dist_params is None:
        dist_params = {}

    if dist_type == 'uniform':
        pops = np.ones(n_nodes)
    elif dist_type == 'lognormal':
        mu = dist_params.get('mu', 10)
        sigma = dist_params.get('sigma', 1.5)
        pops = np.random.lognormal(mu, sigma, n_nodes)
    elif dist_type == 'powerlaw':
        a = dist_params.get('a', 2.0)
        pops = powerlaw.rvs(a, size=n_nodes)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    # Normalize to sum to N_total
    pops = pops / pops.sum() * N_total
    return pops


def generate_node_betas(n_nodes: int, beta_mean: float, beta_variance: float,
                       dist_type: str = 'lognormal', seed: Optional[int] = None) -> np.ndarray:
    """
    Generate transmission rates for each node.

    Args:
        n_nodes: Number of nodes
        beta_mean: Mean transmission rate
        beta_variance: Variance of transmission rate
        dist_type: Distribution type
        seed: Random seed

    Returns:
        betas: (n_nodes,) array of transmission rates
    """
    if seed is not None:
        np.random.seed(seed)

    if dist_type == 'lognormal':
        sigma_ln = np.sqrt(np.log(1 + beta_variance / beta_mean**2))
        mu_ln = np.log(beta_mean) - 0.5 * sigma_ln**2
        betas = np.random.lognormal(mu_ln, sigma_ln, n_nodes)
    elif dist_type == 'uniform':
        betas = np.ones(n_nodes) * beta_mean
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    return betas


def gravity(populations: np.ndarray, distances: np.ndarray,
            k: float = 1.0, a: float = 1.0, b: float = 1.0, c: float = 2.0) -> np.ndarray:
    """
    Compute gravity-based connectivity matrix.

    Formula: M_ij = k * (N_i^a * N_j^b) / d_ij^c for i != j
             M_ii = 0 (diagonal is zero)

    Args:
        populations: (n_nodes,) population per node
        distances: (n_nodes, n_nodes) distance matrix
        k: Scaling constant
        a: Origin population exponent
        b: Destination population exponent
        c: Distance decay exponent

    Returns:
        gravity_matrix: (n_nodes, n_nodes) off-diagonal connectivity
    """
    n = len(populations)
    gravity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j and distances[i, j] > 0:
                gravity_matrix[i, j] = k * (populations[i]**a * populations[j]**b) / (distances[i, j]**c)

    return gravity_matrix


def build_gravity_network(populations: np.ndarray, distances: np.ndarray,
                         a: float = 1.0, b: float = 1.0, c: float = 2.0,
                         k: float = 1.0) -> np.ndarray:
    """
    Build gravity-based migration network.

    Uses gravity model (compatible with laser-core implementation),
    then adds diagonal for local transmission and row-normalizes.

    Network construction:
    1. gravity_offdiag = gravity(pop, dist, k, a, b, c)
       - Returns off-diagonal only (diagonal = 0 by design)
       - k parameter scales the entire gravity matrix
    2. network = I + gravity_offdiag
       - Add identity matrix for local transmission baseline
    3. network /= network.sum(axis=1)
       - Row-normalize so each row sums to 1

    Interpretation of k parameter:
    - k=0: M_ii=1, M_ij=0 → purely local (independent nodes)
    - k=0.01: ~99% local, ~1% spatial (slow spread, similar to laser-polio calibrated values)
    - k=0.1: ~90% local, ~10% spatial
    - k=1.0: ~50/50 local/spatial (balanced, depends on topology)
    - k→large: mostly spatial (synchronized epidemics)

    Args:
        populations: (n_nodes,) population per node
        distances: (n_nodes, n_nodes) distance matrix
        a: Origin population exponent (default 1.0)
        b: Destination population exponent (default 1.0)
        c: Distance decay exponent (default 2.0)
        k: Spatial coupling strength (default 1.0)
           Directly passed to laser_core.gravity()

    Returns:
        network: (n_nodes, n_nodes) row-normalized mixing matrix

    Notes:
        - Exactly matches laser-core pattern (no extra normalization)
        - Compatible with laser-polio ABM for comparison studies
        - The k=0.01-0.25 range matches laser-polio calibration bounds
    """
    n_nodes = len(populations)

    # Special case: single node
    if n_nodes == 1:
        return np.array([[1.0]])

    # Use gravity function to get raw connectivity structure
    # Pass k=1.0 to get unscaled gravity weights (diagonal = 0 by design)
    # Convert distances to float to avoid integer power issues
    gravity_raw = gravity(populations, distances.astype(float), k=1.0, a=a, b=b, c=c)

    # Normalize gravity matrix by its max row sum to get relative connectivity
    # This makes the gravity weights comparable to the diagonal (1.0)
    max_row_sum = gravity_raw.sum(axis=1).max()
    if max_row_sum > 0:
        gravity_normalized = gravity_raw / max_row_sum
    else:
        gravity_normalized = gravity_raw

    # Build network: diagonal (1.0) + k × normalized_gravity
    # When k=0: network = I (purely local)
    # When k=1: balanced between local and spatial
    # When k>1: spatial dominates
    network = np.eye(n_nodes) + k * gravity_normalized

    # Row-normalize: each row sums to 1
    # This makes the network represent mixing probabilities
    row_sums = network.sum(axis=1, keepdims=True)
    network = network / row_sums

    return network


# ============================================================================
# Age Structure
# ============================================================================

def setup_age_structure(n_age: int, N_total: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Setup age bins and population distribution.

    Args:
        n_age: Number of age groups
        N_total: Total population

    Returns:
        age_labels: List of age group labels
        age_counts: Population counts per age group
        aging_rates: Aging rates per age group
    """
    age_labels = [f"{a}-{a+1}" for a in range(n_age-1)] + [f"{n_age-1}+"]
    age_counts = np.ones(n_age)
    age_counts[-1] *= 5  # Make terminal bin larger
    age_counts = age_counts / age_counts.sum() * N_total

    aging_rates = np.ones(n_age) / 365.0  # 1/day
    aging_rates[-1] = 0.0  # No aging out of terminal bin

    return age_labels, age_counts, aging_rates


def age_flux(X: np.ndarray, aging_rates: np.ndarray) -> np.ndarray:
    """
    Compute aging flux for compartment X.

    Args:
        X: Compartment array, shape (n_nodes, n_age, ...) or (n_nodes, n_age)
        aging_rates: (n_age,) aging rates

    Returns:
        flux: Same shape as X
    """
    flux = np.zeros_like(X)

    if X.ndim == 3:  # (n_nodes, n_age, n_bins)
        # Outflux from each age bin except last
        flux[:, :-1, :] -= aging_rates[None, :-1, None] * X[:, :-1, :]
        # Influx to next age bin
        flux[:, 1:, :] += aging_rates[None, :-1, None] * X[:, :-1, :]
    elif X.ndim == 2:  # (n_nodes, n_age) for R compartment
        flux[:, :-1] -= aging_rates[None, :-1] * X[:, :-1]
        flux[:, 1:] += aging_rates[None, :-1] * X[:, :-1]

    return flux


# ============================================================================
# Initialization
# ============================================================================

def initialize_state(config: ModelConfig, node_pops: np.ndarray,
                    seed: Optional[int] = None) -> tuple[MetaPopState, dict]:
    """
    Initialize model state and auxiliary variables.

    Args:
        config: Model configuration
        node_pops: (n_nodes,) population per node
        seed: Random seed

    Returns:
        state: Initial MetaPopState
        aux: Dictionary of auxiliary variables (theta_vals, phi_vals, P, etc.)
    """
    if seed is not None:
        np.random.seed(seed)

    # Age structure
    age_labels, age_counts_template, aging_rates = setup_age_structure(
        config.n_age, config.N_total
    )

    # Heterogeneity distributions
    # Susceptibility (theta) - LogNormal
    sigma_ln = np.sqrt(np.log(1 + config.theta_variance / config.theta_mean**2))
    mu_ln = np.log(config.theta_mean) - 0.5 * sigma_ln**2
    s_dist = lognorm(s=sigma_ln, scale=np.exp(mu_ln))
    theta_vals = np.linspace(s_dist.ppf(0.01), s_dist.ppf(0.99), config.n_bins)
    dtheta = theta_vals[1] - theta_vals[0]

    # Infectiousness (phi) - Gamma
    phi_dist = gamma(a=config.phi_shape, scale=config.phi_scale)
    phi_vals = np.linspace(phi_dist.ppf(0.01), phi_dist.ppf(0.99), config.n_bins)
    dphi = phi_vals[1] - phi_vals[0]

    # Correlation matrix
    P = build_P(config.n_bins, config.P_width)

    # Initialize compartments
    state_size = MetaPopState.state_size(config.n_nodes, config.n_age, config.n_bins)
    y0 = np.zeros(state_size)

    # For each node, initialize S, E, I, R
    for node in range(config.n_nodes):
        # Scale age distribution to this node's population
        age_counts = age_counts_template * (node_pops[node] / config.N_total)

        # Initialize susceptibles with heterogeneity
        for age_idx in range(config.n_age):
            S_age_theta = s_dist.pdf(theta_vals)
            S_age_theta = S_age_theta / S_age_theta.sum() * age_counts[age_idx]

            # If this is the seed node and first age group, remove seed_n_infections individuals
            if node == config.seed_node_idx and age_idx == 0:
                S_age_theta = S_age_theta / S_age_theta.sum() * (age_counts[age_idx] - config.seed_n_infections)

            # Place in state vector
            start_idx = node * config.n_age * config.n_bins + age_idx * config.n_bins
            y0[start_idx:start_idx + config.n_bins] = S_age_theta

        # Initialize infectious (only in seed node)
        if node == config.seed_node_idx:
            I_age_phi = phi_dist.pdf(phi_vals)
            I_age_phi = I_age_phi / I_age_phi.sum() * config.seed_n_infections

            # Place in I compartment (first age group)
            I_start = 2 * config.n_nodes * config.n_age * config.n_bins
            I_node_start = I_start + node * config.n_age * config.n_bins
            y0[I_node_start:I_node_start + config.n_bins] = I_age_phi

    state = MetaPopState(y=y0, n_nodes=config.n_nodes,
                        n_age=config.n_age, n_bins=config.n_bins)

    aux = {
        'theta_vals': theta_vals,
        'phi_vals': phi_vals,
        'dtheta': dtheta,
        'dphi': dphi,
        'P': P,
        'age_labels': age_labels,
        'age_counts': age_counts_template,
        'aging_rates': aging_rates,
    }

    return state, aux


# ============================================================================
# ODE System
# ============================================================================

def compute_foi(I: np.ndarray, phi_vals: np.ndarray, dphi: float,
               betas: np.ndarray, node_pops: np.ndarray,
               network: np.ndarray) -> np.ndarray:
    """
    Compute force of infection for each node.

    Args:
        I: Infectious compartment, shape (n_nodes, n_age, n_bins)
        phi_vals: Infectiousness values
        dphi: Phi bin width
        betas: Transmission rates per node
        node_pops: Population per node
        network: (n_nodes, n_nodes) migration matrix

    Returns:
        foi: (n_nodes,) force of infection per node
    """
    n_nodes = I.shape[0]
    foi = np.zeros(n_nodes)

    # Total infectious pressure from each node (weighted by phi, summed over age and bins)
    infectious_pressure = np.sum(I * phi_vals[None, None, :], axis=(1, 2)) * dphi

    # FOI at node i = sum over nodes j of: M_ij * (beta_j / N_j) * infectious_j
    for i in range(n_nodes):
        for j in range(n_nodes):
            foi[i] += network[i, j] * (betas[j] / node_pops[j]) * infectious_pressure[j]

    return foi


def rhs_metapop(t: float, y: np.ndarray, config: ModelConfig, aux: dict,
               betas: np.ndarray, node_pops: np.ndarray, network: np.ndarray) -> np.ndarray:
    """
    Right-hand side of multi-node SEIR ODE system.

    Args:
        t: Time
        y: State vector
        config: Model configuration
        aux: Auxiliary variables
        betas: Transmission rates per node
        node_pops: Population per node
        network: Migration matrix

    Returns:
        dydt: Derivative of state vector
    """
    state = MetaPopState(y=y, n_nodes=config.n_nodes,
                        n_age=config.n_age, n_bins=config.n_bins)

    S = state.S
    E = state.E
    I = state.I
    R = state.R

    theta_vals = aux['theta_vals']
    phi_vals = aux['phi_vals']
    dphi = aux['dphi']
    P = aux['P']
    aging_rates = aux['aging_rates']

    # Compute force of infection for each node
    foi = compute_foi(I, phi_vals, dphi, betas, node_pops, network)

    # Initialize derivatives
    dSdt = np.zeros_like(S)
    dEdt = np.zeros_like(E)
    dIdt = np.zeros_like(I)
    dRdt = np.zeros_like(R)

    # Dynamics for each node
    for node in range(config.n_nodes):
        for age_idx in range(config.n_age):
            # New infections (S -> E transition)
            new_infections_theta = theta_vals * foi[node] * S[node, age_idx, :]
            dSdt[node, age_idx, :] = -new_infections_theta

            # New exposures (with correlation structure)
            new_E_phi = P.T @ new_infections_theta

            # E -> I transition
            dEdt[node, age_idx, :] = new_E_phi - config.sigma_rate * E[node, age_idx, :]

            # I -> R transition
            dIdt[node, age_idx, :] = config.sigma_rate * E[node, age_idx, :] - config.gamma_rate * I[node, age_idx, :]

            # Recovery
            dRdt[node, age_idx] = config.gamma_rate * np.sum(I[node, age_idx, :])

    # Add aging fluxes
    dSdt += age_flux(S, aging_rates)
    dEdt += age_flux(E, aging_rates)
    dIdt += age_flux(I, aging_rates)
    dRdt += age_flux(R, aging_rates)

    # Flatten and return
    return np.concatenate([dSdt.flatten(), dEdt.flatten(), dIdt.flatten(), dRdt.flatten()])


# ============================================================================
# Simulation
# ============================================================================

def run_simulation(config: ModelConfig, spatial_seed: int = 42,
                  epi_seed: int = 123) -> dict:
    """
    Run multi-node SEIR simulation.

    Args:
        config: Model configuration
        spatial_seed: Random seed for spatial setup
        epi_seed: Random seed for epidemic initialization

    Returns:
        results: Dictionary containing solution and metadata
    """
    import time

    print("Setting up spatial structure...")
    t_start_setup = time.time()

    # Generate spatial structure
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

    print(f"  Node populations: min={node_pops.min():.0f}, max={node_pops.max():.0f}, mean={node_pops.mean():.0f}")
    print(f"  Beta values: min={betas.min():.2f}, max={betas.max():.2f}, mean={betas.mean():.2f}")

    # Build network
    network = build_gravity_network(
        node_pops, distances,
        a=config.gravity_a, b=config.gravity_b,
        c=config.gravity_c, k=config.gravity_k
    )

    print(f"  Network density: {(network > 0).sum() / network.size:.2%}")

    # Initialize state
    print("Initializing state...")
    state, aux = initialize_state(config, node_pops, seed=epi_seed)

    t_setup = time.time() - t_start_setup

    # Run simulation
    print("Running simulation...")
    t_eval = np.arange(0, config.duration_days, config.output_freq_days)

    t_start_sim = time.time()
    sol = solve_ivp(
        rhs_metapop,
        [0, config.duration_days],
        state.y,
        args=(config, aux, betas, node_pops, network),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    t_sim = time.time() - t_start_sim

    print(f"  Integration status: {sol.message}")
    print(f"  Number of time points: {len(sol.t)}")
    print(f"  Setup time: {t_setup:.2f}s")
    print(f"  Simulation time: {t_sim:.2f}s")

    # Package results
    results = {
        'sol': sol,
        'config': config,
        'aux': aux,
        'positions': positions,
        'distances': distances,
        'node_pops': node_pops,
        'betas': betas,
        'network': network,
    }

    return results


# ============================================================================
# Visualization
# ============================================================================

def extract_timeseries(results: dict) -> dict:
    """
    Extract time series from solution.

    Args:
        results: Results dictionary from run_simulation

    Returns:
        ts: Dictionary of time series arrays
    """
    sol = results['sol']
    config = results['config']

    n_nodes = config.n_nodes
    n_age = config.n_age
    n_bins = config.n_bins
    n_time = len(sol.t)

    # Extract compartments over time
    S_all = sol.y[:n_nodes * n_age * n_bins, :].reshape((n_nodes, n_age, n_bins, n_time))
    E_all = sol.y[n_nodes * n_age * n_bins:2 * n_nodes * n_age * n_bins, :].reshape((n_nodes, n_age, n_bins, n_time))
    I_all = sol.y[2 * n_nodes * n_age * n_bins:3 * n_nodes * n_age * n_bins, :].reshape((n_nodes, n_age, n_bins, n_time))
    R_all = sol.y[3 * n_nodes * n_age * n_bins:3 * n_nodes * n_age * n_bins + n_nodes * n_age, :].reshape((n_nodes, n_age, n_time))

    # Sum over heterogeneity bins to get by node and age
    S_node_age = S_all.sum(axis=2)  # (n_nodes, n_age, n_time)
    E_node_age = E_all.sum(axis=2)
    I_node_age = I_all.sum(axis=2)
    R_node_age = R_all  # Already (n_nodes, n_age, n_time)

    # Sum over age to get by node only
    S_node = S_node_age.sum(axis=1)  # (n_nodes, n_time)
    E_node = E_node_age.sum(axis=1)
    I_node = I_node_age.sum(axis=1)
    R_node = R_node_age.sum(axis=1)

    return {
        'time': sol.t,
        'S_node_age': S_node_age,
        'E_node_age': E_node_age,
        'I_node_age': I_node_age,
        'R_node_age': R_node_age,
        'S_node': S_node,
        'E_node': E_node,
        'I_node': I_node,
        'R_node': R_node,
    }


def plot_heatmap(results: dict, output_file: str = 'metapop_heatmap.pdf'):
    """
    Create heatmap of infectious prevalence (nodes × time).

    Args:
        results: Results dictionary from run_simulation
        output_file: Output filename
    """
    ts = extract_timeseries(results)
    I_node = ts['I_node']
    time = ts['time']
    node_pops = results['node_pops']

    # Compute prevalence (I / N)
    prevalence = I_node / node_pops[:, None]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(prevalence, aspect='auto', cmap='YlOrRd',
                   extent=(time[0], time[-1], results['config'].n_nodes, 0),
                   interpolation='nearest')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Node')
    ax.set_title('Infectious Prevalence (I/N) by Node Over Time')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Prevalence')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved heatmap to {output_file}")
    plt.close()


def plot_node_timeseries(results: dict, output_file: str = 'metapop_timeseries.pdf'):
    """
    Create time series plots for each node (all compartments).

    Only practical for small n_nodes (≤ 20).

    Args:
        results: Results dictionary from run_simulation
        output_file: Output filename
    """
    config = results['config']

    if config.n_nodes > 20:
        print(f"  Skipping node time series (n_nodes={config.n_nodes} > 20)")
        return

    ts = extract_timeseries(results)
    time = ts['time']

    n_cols = 4  # S, E, I, R
    n_rows = config.n_nodes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)

    if n_rows == 1:
        axes = axes[None, :]  # Make 2D

    for node in range(config.n_nodes):
        # S
        axes[node, 0].plot(time, ts['S_node'][node, :], 'b-')
        axes[node, 0].set_ylabel(f'Node {node}')
        if node == 0:
            axes[node, 0].set_title('Susceptible')
        axes[node, 0].grid(True, alpha=0.3)

        # E
        axes[node, 1].plot(time, ts['E_node'][node, :], 'orange')
        if node == 0:
            axes[node, 1].set_title('Exposed')
        axes[node, 1].grid(True, alpha=0.3)

        # I
        axes[node, 2].plot(time, ts['I_node'][node, :], 'r-')
        if node == 0:
            axes[node, 2].set_title('Infectious')
        axes[node, 2].grid(True, alpha=0.3)

        # R
        axes[node, 3].plot(time, ts['R_node'][node, :], 'g-')
        if node == 0:
            axes[node, 3].set_title('Recovered')
        axes[node, 3].grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel('Time (days)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved time series to {output_file}")
    plt.close()


def plot_network(results: dict, output_file: str = 'metapop_network.pdf'):
    """
    Plot spatial network with node positions and connections.

    Args:
        results: Results dictionary from run_simulation
        output_file: Output filename
    """
    positions = results['positions']
    network = results['network']
    node_pops = results['node_pops']
    config = results['config']

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot edges (only strong connections for clarity)
    threshold = np.percentile(network[network > 0], 90) if (network > 0).any() else 0
    for i in range(config.n_nodes):
        for j in range(config.n_nodes):
            if network[i, j] > threshold:
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       'k-', alpha=0.2, linewidth=0.5)

    # Plot nodes (size proportional to population)
    sizes = 100 * node_pops / node_pops.max()
    ax.scatter(positions[:, 0], positions[:, 1], s=sizes, c='steelblue',
              alpha=0.7, edgecolors='black', linewidths=1)

    # Label nodes
    for i in range(config.n_nodes):
        ax.text(positions[i, 0], positions[i, 1], str(i),
               ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Spatial Network ({config.n_nodes} nodes)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved network plot to {output_file}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run multi-node SEIR simulation with default configuration."""
    from pathlib import Path

    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # Configure model
    config = ModelConfig(
        n_nodes=774,
        n_age=16,
        n_bins=25,
        seed_node_idx=0,
        seed_n_infections=1.0,
        pop_dist_type='lognormal',
        pop_dist_params={'mu': 10, 'sigma': 1.5},
        beta_dist_type='lognormal',
        beta_mean=10.0,
        beta_variance=4.0,
        gravity_a=1.0,
        gravity_b=1.0,
        gravity_c=2.0,
        gravity_k=0.01,  # 0=local only, 0.01-0.25=spatial coupling (laser-polio calibrated range)
        duration_days=365 * 1,
    )

    print("="*70)
    print("Multi-Node Meta-Population SEIR Model")
    print("="*70)
    print("Configuration:")
    print(f"  n_nodes: {config.n_nodes}")
    print(f"  n_age: {config.n_age}")
    print(f"  n_bins: {config.n_bins}")
    print(f"  gravity_k: {config.gravity_k} (0=local only, >0=spatial coupling)")
    print(f"  Total ODEs: {MetaPopState.state_size(config.n_nodes, config.n_age, config.n_bins)}")
    print()

    # Run simulation
    results = run_simulation(config)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_heatmap(results, str(figures_dir / 'metapop_heatmap.pdf'))
    plot_node_timeseries(results, str(figures_dir / 'metapop_timeseries.pdf'))
    plot_network(results, str(figures_dir / 'metapop_network.pdf'))

    print("\nDone!")
    print("="*70)

    return results


if __name__ == '__main__':
    results = main()
