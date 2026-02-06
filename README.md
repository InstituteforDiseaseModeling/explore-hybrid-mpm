# Hybrid Stochastic/Deterministic SEIR Metapopulation Model

GPU-accelerated epidemic modeling with automatic regime switching between stochastic tau-leaping and deterministic RK4 integration.

## Overview

This repository implements a hybrid stochastic/deterministic SEIR (Susceptible-Exposed-Infectious-Recovered) metapopulation model with:

- **Hybrid dynamics**: Automatic switching between stochastic tau-leaping (low prevalence) and deterministic RK4 (high prevalence)
- **GPU acceleration**: Powered by [Taichi](https://www.taichi-lang.org/) for Metal (macOS) and CUDA backends
- **Age stratification**: 16 age groups with aging dynamics
- **Individual heterogeneity**: Heterogeneous susceptibility (θ) and infectiousness (φ) with correlation
- **Spatial network**: Gravity-based coupling between nodes
- **Stochastic fadeout**: Natural extinction dynamics in low-prevalence nodes

## Key Features

### 1. Automatic Regime Switching

Nodes automatically switch between integration methods based on infectious count:
- **I < threshold**: Stochastic tau-leaping with Poisson-distributed transitions
- **I ≥ threshold**: Deterministic RK4 integration (4th-order accurate)

### 2. Performance

- **774 nodes, 365 days**: ~15-20 minutes on M1/M2 Mac (Metal backend)
- **20 nodes, 150 days**: ~1 second per realization
- **Fully stochastic mode**: 3-12 ms per time step depending on problem size

### 3. Biological Realism

- Stochastic fadeout in small transmission chains (~15% reduction in peak epidemic size)
- Natural importation dynamics via network coupling
- Conservation of total population (S+E+I+R = N_total)

## Installation

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)

```bash
git clone https://github.com/institutefordiseasemodeling/explore-hybrid-mpm.git
cd explore-hybrid-mpm
uv sync
```

### Using pip

```bash
pip install -e .
```

### Dependencies

- `taichi>=1.7.0` - GPU-accelerated computing
- `numpy` - Numerical arrays
- `scipy` - Statistical distributions
- `matplotlib` - Visualization

## Quick Start

![Example output showing epidemic dynamics](figures/quick_example.pdf)

### Basic Usage

```python
from src.SEIR_pde_metapop_stochastic_taichi import (
    StochasticModelConfig,
    run_simulation_stochastic
)
from src.SEIR_pde_metapop import extract_timeseries

# Configure simulation
config = StochasticModelConfig(
    n_nodes=20,                    # Number of spatial nodes
    n_age=4,                       # Number of age groups
    n_bins=10,                     # Heterogeneity resolution
    seed_n_infections=150.0,       # Initial infections in seed node
    stochastic_threshold=100.0,    # Switching threshold
    tau_leap_dt=1.0,               # Time step (days)
    duration_days=150,             # Simulation duration
    backend='metal',               # 'metal', 'cuda', or 'cpu'
)

# Run simulation
results = run_simulation_stochastic(config, spatial_seed=42, epi_seed=123)

# Extract timeseries
ts = extract_timeseries(results)
I_total = ts['I_node'].sum(axis=0)  # Total infectious over time
```

### Fully Stochastic Mode

```python
config = StochasticModelConfig(
    n_nodes=10,
    stochastic_threshold=1e10,  # All nodes use tau-leaping
    duration_days=200,
)
```

### Fully Deterministic Mode

```python
config = StochasticModelConfig(
    n_nodes=20,
    stochastic_threshold=0.0,  # All nodes use RK4
    duration_days=150,
)
```

## Repository Structure

```
explore-hybrid-mpm/
├── src/
│   ├── SEIR_pde_metapop_stochastic_taichi.py  # Main implementation
│   ├── SEIR_pde_metapop_taichi.py             # GPU baseline (deterministic)
│   └── SEIR_pde_metapop.py                    # CPU baseline (deterministic)
├── tests/
│   ├── test_stochastic_validation.py          # Full validation suite
│   ├── test_quick_validation.py               # Quick correctness checks
│   └── test_taichi_validation.py              # GPU vs CPU validation
├── examples/
│   ├── compare_stochastic_thresholds.py       # Threshold comparison analysis
│   ├── test_fully_stochastic.py               # Fully stochastic example
│   ├── test_user_config.py                    # Custom configuration example
│   └── debug_*.py                             # Debugging utilities
├── figures/
│   ├── threshold_comparison.pdf               # Main results figure
│   ├── threshold_individual_realizations.pdf  # Stochastic trajectories
│   └── threshold_summary_stats.pdf            # Summary statistics
├── docs/
├── README.md
├── pyproject.toml
└── uv.lock
```

## Examples

### Threshold Comparison

Compare dynamics across different stochasticity levels:

```bash
uv run python examples/compare_stochastic_thresholds.py
```

**Output**: Three publication-ready figures showing:
1. Epidemic curves with confidence bands
2. Individual realizations demonstrating stochastic divergence
3. Summary statistics (peak size, coefficient of variation)

**Key findings**:
- Fully deterministic: Peak ~47K infections
- Fully stochastic: Peak ~40K infections (mean of 10 runs, CV=2.9%)
- Hybrid thresholds smoothly interpolate between extremes

### Validation Tests

Run comprehensive validation suite:

```bash
uv run python tests/test_stochastic_validation.py
```

**Tests**:
1. Pure deterministic mode (threshold=∞) approximates baseline
2. Conservation of population (S+E+I+R = N_total)
3. Stochastic variability across multiple realizations
4. Fade-out and reintroduction dynamics

### Quick Validation

Fast correctness check:

```bash
uv run python tests/test_quick_validation.py
```

## Algorithm Details

### Hybrid Integration Loop

At each time step dt:

1. **Compute FOI globally**: Force of infection from neighboring nodes (held constant over dt)
2. **Classify regimes**: Compute I_total per node and compare to threshold
3. **Stochastic nodes** (I < threshold):
   - Sample S→E, E→I, I→R transitions from Poisson distributions
   - Apply P matrix for θ → φ mapping (continuous mode)
   - Clamp to ensure non-negative values
4. **Deterministic nodes** (I ≥ threshold):
   - Advance with 4th-order Runge-Kutta (RK4)
   - Clamp intermediate states to prevent numerical instability
5. **Save output** at requested frequency

### Custom Poisson Sampler

Since Taichi doesn't provide `ti.random.poisson()`, we implement a custom sampler:

- **λ < 10**: Knuth's multiplicative algorithm
- **10 ≤ λ < 100**: Normal approximation via Box-Muller transform
- **λ ≥ 100**: Deterministic approximation (return λ)

### Numerical Stability

Critical fixes for RK4 stability:
- Clamp all intermediate states (`S_temp`, `E_temp`, `I_temp`) to be non-negative
- Prevents exponential explosion when derivatives are computed on negative values
- Ensures realistic epidemic dynamics even with large dt (e.g., 1.0 day)

## Configuration Options

### StochasticModelConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_nodes` | int | 10 | Number of spatial nodes |
| `n_age` | int | 16 | Number of age groups |
| `n_bins` | int | 50 | Heterogeneity resolution (θ, φ) |
| `seed_node_idx` | int | 0 | Index of initially infected node |
| `seed_n_infections` | float | 1.0 | Initial infections in seed node |
| `stochastic_threshold` | float | 10.0 | Regime switching threshold (I_total) |
| `tau_leap_dt` | float | 0.5 | Time step for integration (days) |
| `duration_days` | float | 365.0 | Simulation duration |
| `output_freq_days` | float | 1.0 | Output sampling frequency |
| `backend` | str | 'cpu' | Taichi backend ('metal', 'cuda', 'cpu') |
| `use_float64` | bool | False | Use double precision |
| `stochastic_seed` | int | 42 | RNG seed for reproducibility |
| `gravity_k` | float | 1.0 | Gravity network coupling strength |

### Epidemiological Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_rate` | 1/3 days⁻¹ | Latent period (E→I) |
| `gamma_rate` | 1/5 days⁻¹ | Infectious period (I→R) |
| `beta_mean` | 10.0 | Mean transmission rate |
| `theta_mean` | 1.0 | Mean susceptibility |
| `phi_shape` | 2.0 | Infectiousness distribution shape |

## Performance Benchmarks

| Configuration | Backend | Time | Notes |
|--------------|---------|------|-------|
| 774 nodes, 365 days, dt=1.0 | Metal (M1) | ~18 min | Production scale |
| 20 nodes, 150 days, dt=1.0 | CPU | ~1 sec | Quick testing |
| 10 nodes, 200 days, dt=1.0, fully stochastic | CPU | ~0.8 sec | Single realization |

**Speedup factors**:
- Taichi GPU vs baseline CPU: ~100× for large problems (774 nodes)
- dt=1.0 vs dt=0.5: 2× speedup with acceptable accuracy

## Known Issues and Limitations

### Precision Warnings

You may see Taichi warnings about `f32 <- f64` precision loss in `sample_poisson()`. This is expected when using `use_float64=True` but is generally harmless (the Poisson sampler uses f32 internally for RNG).

### Stochastic Fadeout with Low Seed Infections

When `seed_n_infections < stochastic_threshold`, the epidemic may fade out immediately due to stochastic extinction. This is correct biological behavior but may not be desired for testing epidemic spread. Use higher seed infections (e.g., 150) to ensure takeoff.

### RK4 Clamping vs Adaptive Step Size

The fixed dt with RK4 and clamping is less accurate than adaptive RK45 but much faster for GPU parallelization. For high-accuracy deterministic simulations, use the baseline `SEIR_pde_metapop.py` with `solve_ivp`.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{explore_hybrid_mpm,
  title = {Hybrid Stochastic/Deterministic SEIR Metapopulation Model},
  author = {Institute for Disease Modeling},
  year = {2026},
  url = {https://github.com/InstituteforDiseaseModeling/explore-hybrid-mpm}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on the [GitHub repository](https://github.com/InstituteforDiseaseModeling/explore-hybrid-mpm).

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/InstituteforDiseaseModeling/explore-hybrid-mpm/issues)
- Contact the maintainer: Daniel J. Klein (daniel.klein@gatesfoundation.org)

## Acknowledgments

- Built with [Taichi](https://www.taichi-lang.org/) for GPU acceleration
- Inspired by hybrid tau-leaping methods in computational biology
- Developed for spatial epidemic modeling research
