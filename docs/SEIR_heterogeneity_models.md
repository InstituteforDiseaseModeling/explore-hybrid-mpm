# Age-Stratified SEIR Model with Correlated Heterogeneity

Dan Klein

June 16, 2025

## Overview

This model (SEIR_pde.py) implements SEIR dynamics with correlated individual-level heterogeneity in susceptibility ($\theta$) and infectiousness ($\phi$). The correlation structure captures the biological reality that individuals with higher susceptibility tend to exhibit higher infectiousness upon infection. The implementation solves a system of integro-differential equations using finite-dimensional discretization.

## Model Features

**Population structure:**
- Age-stratified model with 16 age groups (0-1, 1-2, ..., 14-15, 15+)
- Aging dynamics with rate 1/365 per day between consecutive age bins
- $N = 186{,}763$, discretized into 50 bins for $\theta$ and $\phi$
- 3-year simulation horizon with weekly output

**Correlation method:**
Uses Gaussian kernel approximation:

$$P(\phi_j|\theta_i) \propto \exp\left(-\frac{(i-j)^2}{2w^2}\right)$$

The kernel width $w$ is optimized via scalar minimization to achieve target correlation $\rho = 0.8$. This method:
- Avoids expensive sampling procedures
- Guarantees stochastic matrix properties (non-negative, rows sum to 1)
- Scales efficiently to large n_bins
- Produces smooth conditional distributions

**Age structure:**
The model tracks $(S, E, I, R)$ compartments for each age group with aging flux terms that transfer individuals between adjacent age bins at rate $r_a = 1/365$ day$^{-1}$ (no outflux from terminal bin).

---

## Mathematical Formulation

### Continuous PDE System

The discretized ODE system approximates the following integro-PDE:

**Susceptible dynamics:**

$$\frac{\partial S(t,\theta,a)}{\partial t} + \frac{\partial[r(a)S(t,\theta,a)]}{\partial a} = -\theta \lambda(t) S(t,\theta,a)$$

**Exposed dynamics:**

$$\frac{\partial E(t,\phi,a)}{\partial t} + \frac{\partial[r(a)E(t,\phi,a)]}{\partial a} = \int_0^\infty P(\phi|\theta) \theta \lambda(t) S(t,\theta,a) d\theta - \sigma E(t,\phi,a)$$

**Infectious dynamics:**

$$\frac{\partial I(t,\phi,a)}{\partial t} + \frac{\partial[r(a)I(t,\phi,a)]}{\partial a} = \sigma E(t,\phi,a) - \gamma I(t,\phi,a)$$

**Recovered dynamics:**

$$\frac{\partial R(t,a)}{\partial t} + \frac{\partial[r(a)R(t,a)]}{\partial a} = \gamma \int_0^\infty I(t,\phi,a) d\phi$$

**Force of infection:**

$$\lambda(t) = \frac{\beta}{N} \int_0^\infty \int_A \phi I(t,\phi,a) \, d\phi \, da$$

where:
- $\theta \in [0,\infty)$: relative susceptibility
- $\phi \in [0,\infty)$: relative infectiousness
- $a$: age
- $r(a)$: aging rate (individuals move from age $a$ to $a+da$)
- $P(\phi|\theta)$: conditional probability of infectiousness given susceptibility
- $\beta$: base transmission rate
- $\sigma$: progression rate $(E \to I)$
- $\gamma$: recovery rate $(I \to R)$
- $N$: total population size

### Discretization Scheme

**Spatial discretization:**
- $\theta$ discretized into n_bins bins: $\theta_i$, $i = 1,\ldots,n_{\text{bins}}$ (evaluated at bin centers)
- $\phi$ discretized into n_bins bins: $\phi_j$, $j = 1,\ldots,n_{\text{bins}}$
- Age discretized into n_age bins: $a_k$, $k = 1,\ldots,n_{\text{age}}$

**State variables:**
- $S(t,\theta_i,a_k) \to S_{i,k}(t)$
- $E(t,\phi_j,a_k) \to E_{j,k}(t)$
- $I(t,\phi_j,a_k) \to I_{j,k}(t)$
- $R(t,a_k) \to R_k(t)$

**Finite-dimensional ODE system:**

$$\frac{dS_{i,k}}{dt} = -\theta_i \lambda(t) S_{i,k} + [\text{aging flux}]_k$$

$$\frac{dE_{j,k}}{dt} = \sum_i P_{ji} \theta_i \lambda(t) S_{i,k} - \sigma E_{j,k} + [\text{aging flux}]_k$$

$$\frac{dI_{j,k}}{dt} = \sigma E_{j,k} - \gamma I_{j,k} + [\text{aging flux}]_k$$

$$\frac{dR_k}{dt} = \gamma \sum_j I_{j,k} + [\text{aging flux}]_k$$

$$\lambda(t) = \frac{\beta}{N} \sum_k \sum_j \phi_j I_{j,k} \Delta\phi$$

**Aging flux (upwind scheme):**

$$[\text{aging flux}]_k = r_{k-1} X_{k-1} - r_k X_k$$

where $X \in \{S, E, I, R\}$ and $r_{n_{\text{age}}} = 0$ (absorbing boundary).

**Integration scheme:**
The integral $\int I(t,\phi) d\phi$ is approximated by Riemann sum: $\sum_j I_j \Delta\phi$, where $\Delta\phi$ is the bin width.

### Conservation Properties

The model preserves:
1. **Infection conservation:** Total new infections in $S(\theta)$ space equals total new exposures in $E(\phi)$ space (enforced by stochastic $P$ matrix)
2. **Population conservation:** $\frac{d}{dt}(S + E + I + R) = 0$ (up to numerical error)
3. **Non-negativity:** All compartments remain $\geq 0$ (guaranteed by non-negative initial conditions and model structure)

---

## Implementation Notes

### Correlation Matrix Construction

The Gaussian kernel ansatz assumes $P(\phi|\theta)$ has support concentrated near $\phi \approx \theta$ (in bin-index space). The optimization procedure:

1. Generates $n_{\text{samples}}$ random $\theta$ indices
2. For candidate width $w$, samples $\phi$ indices according to $P(\phi_j|\theta_i)$
3. Computes empirical correlation between $\theta_{\text{samples}}$ and $\phi_{\text{samples}}$
4. Minimizes $|\rho_{\text{empirical}} - \rho_{\text{target}}|$ using bounded scalar minimization

This produces a correlation structure that is approximately correct for the chosen marginal distributions while maintaining computational tractability.

### Parameter Choices

**Marginal distributions:**
- Susceptibility: $\text{LogNormal}(\mu=1.0, \sigma^2=4.0)$
- Infectiousness: $\text{Gamma}(\text{shape}=1.0, \text{scale}=\beta/24)$

**Epidemiological parameters:**
- $\beta = 10.0$ (transmission rate)
- $\sigma = 1/3$ day$^{-1}$ (incubation period: 3 days)
- $\gamma = 1/24$ day$^{-1}$ (infectious period: 24 days)

These values produce baseline $R_0 \approx \beta/\gamma = 240$ in a homogeneous population, though heterogeneity reduces effective $R_0$.

### Computational Complexity

| Operation | Complexity |
|-----------|------------|
| $P$ matrix construction | $O(n_{\text{bins}}^2 \times n_{\text{opt\_iter}}) \approx 10^4$ |
| ODE evaluation | $O(n_{\text{age}} \times n_{\text{bins}}^2)$ |
| Memory | $O(n_{\text{age}} \times n_{\text{bins}}^2)$ |

The model scales efficiently to $n_{\text{bins}} = 50\text{-}100$ and $n_{\text{age}} = 16$.

---

## Model Validation

The implementation includes an assertion checking infection conservation:
```python
assert np.isclose(new_E_phi.sum(), new_infections_theta.sum())
```
This ensures the correlation structure does not introduce numerical artifacts during the $S \to E$ transition.
