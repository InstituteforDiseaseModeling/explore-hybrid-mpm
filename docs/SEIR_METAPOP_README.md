# Multi-Node Meta-Population SEIR Model

**Implementation Date:** December 17, 2025
**Phase:** Phase 1 - NumPy Baseline (Complete ✓)

## Overview

`SEIR_pde_metapop.py` extends the single-population SEIR PDE model ([SEIR_pde.py](SEIR_pde.py)) to multiple spatially-coupled nodes connected via a gravity-based network. Each node maintains the full complexity of the original model:

- **16 age groups** (0-1, 1-2, ..., 15+) with aging dynamics
- **50 bins** for susceptibility (θ) and infectiousness (φ) heterogeneity
- **Correlated heterogeneity** (P(φ|θ) with correlation ~0.8)
- **Network-based force of infection** (FOI couples nodes spatially)

## Key Features

### Configuration System

All parameters are user-configurable via the `ModelConfig` dataclass:

```python
config = ModelConfig(
    n_nodes=5,                    # Number of spatial nodes
    n_age=16,                     # Age groups (configurable)
    n_bins=50,                    # Heterogeneity bins (configurable)

    # Seeding
    seed_node_idx=0,              # Which node to seed
    seed_K=1.0,                   # Number of initial infections

    # Population distribution
    pop_dist_type='lognormal',    # 'lognormal', 'powerlaw', 'uniform'
    pop_dist_params={'mu': 10, 'sigma': 1.5},
    N_total=186_763,

    # Beta (transmission rate) per node
    beta_dist_type='lognormal',   # 'lognormal' or 'uniform'
    beta_mean=10.0,
    beta_variance=4.0,

    # Gravity model: M_ij ∝ (N_i^a × N_j^b) / d_ij^c
    gravity_a=1.0,                # Origin population exponent
    gravity_b=1.0,                # Destination population exponent
    gravity_c=2.0,                # Distance decay exponent
    gravity_k=1.0,                # Scaling factor

    # Standard SEIR parameters
    sigma_rate=1/3,               # E → I rate
    gamma_rate=1/24,              # I → R rate

    # Simulation
    duration_days=365 * 3,        # 3 years
    output_freq_days=365/52,      # Weekly outputs
)
```

### State Management

Uses **Option B** (dataclass with view properties) for clean code:

```python
@dataclass
class MetaPopState:
    y: np.ndarray        # Flat array for solve_ivp
    n_nodes: int
    n_age: int
    n_bins: int

    @property
    def S(self) -> np.ndarray:
        """Returns view of S compartment: (n_nodes, n_age, n_bins)"""
        ...

    @property
    def I(self) -> np.ndarray:
        """Returns view of I compartment: (n_nodes, n_age, n_bins)"""
        ...
```

Properties return **views** (not copies) into the flat state vector, so zero overhead.

### Spatial Structure

- **Node positions:** Random 2D coordinates (uniform in [0, 100] × [0, 100])
- **Distances:** Euclidean distance matrix
- **Node populations:** Drawn from configurable distribution (log-normal, power-law, or uniform)
- **Node betas:** Drawn from log-normal distribution (allows spatial heterogeneity in transmission)
- **Gravity network:** Row-normalized (each row sums to 1)

### Force of Infection

Multi-node FOI couples nodes via the network:

```
λ_i(t) = Σ_j M_ij × (β_j / N_j) × Σ_{age,φ} φ × I_j(age, φ, t)
```

- **M_ij:** Migration/connectivity weight from node i to node j
- **β_j / N_j:** Transmission rate per contact in node j
- **Infectious pressure:** Weighted sum over age and infectiousness bins

For single-node case: `M[0,0] = 1.0` (local transmission only)

### ODE System

State vector size: **3 × n_nodes × n_age × n_bins + n_nodes × n_age**

For default parameters (n_nodes=5, n_age=16, n_bins=50): **12,080 ODEs**

Dynamics per node:
- S → E transition with heterogeneity in susceptibility (θ)
- Correlation structure maps θ bins to φ bins via P matrix
- E → I → R progression
- Aging flux between age groups

## Validation

All validation tests pass (see [test_metapop_validation.py](test_metapop_validation.py)):

### ✓ Test 1: Single-Node Limit
- Population conservation: error < 1e-15
- Non-negativity: all compartments ≥ -1e-5 (numerical tolerance)
- Epidemic occurs: 62% attack rate in 3 years

### ✓ Test 2: Multi-Node Conservation
- Tested at n_nodes = 2, 5, 10
- Conservation holds per node (error < 1e-5)

### ✓ Test 3: Scaling Performance
| n_nodes | ODEs  | Runtime (1 year) |
|---------|-------|------------------|
| 2       | 4,832 | 0.27s           |
| 5       | 12,080| 0.62s           |
| 10      | 24,160| 1.19s           |
| 20      | 48,320| 2.21s           |

Roughly **O(n_nodes)** scaling for moderate node counts.

### ✓ Test 4: Spatial Spread
- Infection seeded in node 0 spreads to all 5 nodes
- Final recovered: 16k, 6k, 21k, 74k, 5k per node
- Network connectivity enables realistic spatial transmission

## Visualization

Three visualizations generated automatically:

1. **Heatmap** (`metapop_heatmap.pdf`):
   - Y-axis: Nodes
   - X-axis: Time (days)
   - Color: Infectious prevalence (I/N)
   - **Priority #1 visualization**

2. **Time series** (`metapop_timeseries.pdf`):
   - Per-node trajectories for S, E, I, R
   - Only generated for n_nodes ≤ 20 (too many panels otherwise)

3. **Network plot** (`metapop_network.pdf`):
   - Node positions in 2D space
   - Node size ∝ population
   - Edges show strong connections (>90th percentile)

## Usage

### Basic Usage

```bash
uv run SEIR_pde_metapop.py
```

Runs default configuration (5 nodes, 3 years) and saves plots.

### Custom Configuration

```python
from SEIR_pde_metapop import ModelConfig, run_simulation

config = ModelConfig(
    n_nodes=10,
    n_age=8,              # Fewer age groups for speed
    n_bins=20,            # Fewer bins for speed
    duration_days=365,    # 1 year
    gravity_c=1.5,        # Less distance decay
)

results = run_simulation(config, spatial_seed=42, epi_seed=123)
```

### Extract Time Series

```python
from SEIR_pde_metapop import extract_timeseries

ts = extract_timeseries(results)

# Access data
time = ts['time']              # (n_time,)
I_node = ts['I_node']          # (n_nodes, n_time)
I_node_age = ts['I_node_age']  # (n_nodes, n_age, n_time)
```

## Performance Considerations

### Current Implementation (NumPy + SciPy)

- **Good for:** n_nodes ≤ 50, exploratory analysis
- **Bottleneck:** Force of infection computation (O(n_nodes² × n_age × n_bins))
- **Memory:** ~500 MB for n_nodes=50

### Potential Optimizations (Not Yet Implemented)

1. **Numba JIT compilation** for hot loops:
   ```python
   from numba import jit

   @jit(nopython=True)
   def compute_foi(...):
       ...
   ```

2. **Sparse networks:** For large n_nodes, store only strong connections

3. **GPU acceleration** (separate file: `SEIR_pde_metapop_taichi.py`):
   - Taichi kernels for parallel FOI computation
   - Mac Metal / CUDA backend
   - Expected speedup: 10-100× for n_nodes > 100

## File Structure

```
spatial-features/
├── SEIR_pde.py                    # Original single-node model
├── SEIR_pde_metapop.py            # ✓ Multi-node extension (Phase 1)
├── test_metapop_validation.py     # ✓ Validation test suite
├── SEIR_METAPOP_README.md         # ✓ This file
│
├── metapop_heatmap.pdf            # ✓ Output: prevalence heatmap
├── metapop_timeseries.pdf         # ✓ Output: per-node SEIR traces
├── metapop_network.pdf            # ✓ Output: spatial network
└── test_spatial_spread.pdf        # ✓ Output: validation plot
```

## Future Directions

### Direction 1: Hybrid Stochastic/Deterministic (SEIR_pde_metapop_stochastic.py)

**Goal:** Low-prevalence nodes use stochastic dynamics (tau-leaping), high-prevalence nodes use ODEs.

**Components:**
- Threshold detection per node (I < 5 or I/N < 0.001)
- Tau-leaping solver for discrete events
- Stochastic between-node transmission (Poisson jumps)
- Adaptive switching between regimes

**Use case:** Simulate extinction dynamics, importation events, fade-out/reemergence

### Direction 2: GPU Acceleration (SEIR_pde_metapop_taichi.py)

**Goal:** Scale to n_nodes = 1000+ using GPU parallelism.

**Implementation:**
- Taichi Lang with Metal backend (Mac) or CUDA (Linux)
- Parallelize FOI computation across nodes/age/bins
- Expected: 10-100× speedup for large n_nodes

**Benchmark targets:**
- n_nodes = 100: < 5s per simulation
- n_nodes = 1000: < 30s per simulation

### Direction 3: Streamlit Dashboard (Optional)

Interactive exploration with:
- Parameter sliders
- Animated network visualization (time scrubbing)
- Node selection for detailed traces
- Real-time simulation

## Technical Notes

### Correlation Matrix

Hardcoded width = 9.9242 (achieves ρ ≈ 0.8 for n_bins=50). This skips the expensive optimization in SEIR_pde.py. To retune for different n_bins:

```python
P = build_P(n_bins, width=desired_width)
# Or use optimization from SEIR_pde.py
```

### Aging Dynamics

Aging flux uses upwind scheme:
```
flux[age] = r[age-1] × X[age-1] - r[age] × X[age]
```

Terminal age bin (15+) has r=0 (absorbing boundary).

### Network Construction

Gravity model with special cases:
- **Single node:** M[0,0] = 1 (local transmission)
- **Multi-node:** M[i,j] = 0 for i=j (no self-loops), row-normalized

To allow self-loops (mixing within node):
```python
# In build_gravity_network, remove "if i != j" check
# and add diagonal boost term
```

### Known Limitations

1. **Numerical precision:** Small negative values (< 1e-5) can occur near zero due to finite precision
2. **Fixed age structure:** Births/deaths not yet implemented (easy to add)
3. **Homogeneous mixing:** Within each node, individuals mix randomly (no further spatial structure)
4. **Deterministic:** Stochastic effects at low prevalence ignored (Phase 2 addresses this)

## Comparison with ABM Approach

The existing `laser-polio` ABM (in [src/model.py](src/model.py)) also implements multi-node dynamics, but with a different philosophy:

| Feature | SEIR_pde_metapop.py (PDE) | laser-polio ABM |
|---------|---------------------------|-----------------|
| Approach | Integro-PDE with discretization | Agent-based simulation |
| Heterogeneity | Continuous (binned) | Individual-level |
| Stochasticity | Deterministic (Phase 1) | Fully stochastic |
| Nodes | User-configurable | Admin regions (Nigeria) |
| Geography | Abstract 2D | Real shapefiles + lat/lon |
| Calibration | Direct | Optuna-based framework |
| Speed | Fast (< 1s for 5 nodes) | Slower (ABM overhead) |
| Use case | Theoretical exploration | Applied forecasting |

Both approaches have merits:
- **PDE:** Better for understanding dynamics, parameter sweeps, theory
- **ABM:** Better for realism, rare events, individual-level interventions

## References

- Original model documentation: [SEIR_heterogeneity_models.md](SEIR_heterogeneity_models.md)
- Gravity model: Implemented via `laser-core.migration` (see [scripts/plot_networks.py](scripts/plot_networks.py))
- Correlation method: Gaussian kernel approximation (same as SEIR_pde.py)

## Contact

For questions or issues, see repository maintainer.

---

**Status:** Phase 1 complete and validated ✓
**Next steps:** Implement Direction 1 (stochastic) or Direction 2 (GPU) based on research priorities
