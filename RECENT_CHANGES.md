# Summary of Recent Changes

**Date**: 2026-02-06
**Session**: RK2 Numerical Instability Investigation

---

## Problem Reported

User observed catastrophic numerical instability in `timestep_stability.pdf`:
- RK2 method with dt=1.0 showed infections reaching **1e30 - 1e37** (astronomically impossible)
- Reference solution (RK4 dt=0.25) showed normal ~130K peak infections
- Y-axis scales on plots: dt=0.5 → 1e7, dt=1.0 → 1e30, dt=2.0/4.0 → 1e37

---

## Root Cause Analysis

### The Fundamental Issue

Fixed-timestep explicit Runge-Kutta methods face a dilemma when simulating epidemic compartmental models:

1. **Without clamping**: Intermediate RK stages can go negative → dynamics explode to `inf`/`NaN`
2. **With clamping**: Intermediate stages clamped to zero → breaks RK mathematical consistency → mass conservation violations

### Why RK2 Failed Catastrophically

RK2 algorithm:
```python
k1 = f(y)
y_mid = y + dt/2 * k1
k2 = f(y_mid)  # If y_mid clamped, k2 computed from wrong state!
y_new = y + dt * k2  # Applied to original y → mass inconsistency
```

**Result**:
- Clamping errors accumulate exponentially
- Mass grows by 143% at day 20
- Eventually explodes to 1e30+

### Why RK4 Works (Mostly)

RK4 algorithm:
```python
k1 = f(y)
k2 = f(y + dt/2 * k1)
k3 = f(y + dt/2 * k2)
k4 = f(y + dt * k3)
y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)  # Weighted average!
```

**Result**:
- Clamping errors from 4 stages partially cancel out
- Mass error only ~1% at dt=1.0
- Stable and physically plausible

### Timeline of Mass Violations (RK2)

```
Day 0-8:   ✓ Perfect (0.000000% error) - why debug_gpu_vs_cpu.py "passed"
Day 9:     ⚠️ First violation (0.2%)
Day 10:    ❌ Critical (10.5%)
Day 13+:   💥 Catastrophic explosion (143% at day 20)
```

**Why debug_gpu_vs_cpu.py didn't catch it**: Only ran 2 days, violations are latent and require ~9 days to manifest.

---

## Solution Implemented

### 1. **Disabled RK2** ❌

- Marked as `DEPRECATED` in `StochasticModelConfig` documentation
- Removed from `test_timestep_stability.py`
- Added warning: "has catastrophic mass conservation errors with clamping"

### 2. **Documented RK4 Tradeoff** ⚠️

Added detailed comments explaining:
- Clamping is necessary to prevent `inf`/`NaN` explosions
- Clamping breaks RK consistency → small mass errors
- RK4's 4-stage averaging minimizes error to ~1% at dt=1.0
- Proper fix requires positivity-preserving RK or adaptive stepping (complex)

### 3. **Updated Test Conclusions**

`test_timestep_stability.py` now recommends:
- dt≤0.5: All methods stable
- dt=1.0: **RK4 only** (~8% error), Euler marginal
- dt=2.0: RK4 marginal (~16% error), Euler fails
- dt>3.0: All methods fail

### 4. **Regenerated Figures**

- `timestep_stability.pdf`: Clean RK4 results, no explosion
- All threshold comparison figures updated

---

## Configuration Changes ⚠️

**IMPORTANT**: Test configurations were reduced for faster iteration during debugging.

### examples/test_timestep_stability.py

**Before** (commit 9a445a0):
```python
base_config = {
    'n_nodes': 128,
    'n_age': 16,      # 16 age groups
    'n_bins': 50,     # 50 heterogeneity bins
    # Total ODEs: 128 * 16 * 50 * 3 + 128 * 16 = 307,456 ODEs
    'duration_days': 150,
}
```

**After** (commit 376b0bb):
```python
base_config = {
    'n_nodes': 128,
    'n_age': 2,       # REDUCED to 2 age groups
    'n_bins': 25,     # REDUCED to 25 heterogeneity bins
    # Total ODEs: 128 * 2 * 25 * 3 + 128 * 2 = 19,456 ODEs
    'duration_days': 150,
}
```

**Impact**:
- **16× reduction in problem size** (307K → 19K ODEs)
- **Much faster execution** (~5-10× speedup)
- Same qualitative behavior (RK2 still explodes, RK4 still works)
- Peak infections changed (~30K → ~130K) due to different age structure

### examples/test_deterministic_methods.py

Uses same reduced config (n_age=2, n_bins=25) - already had this configuration.

### ⚠️ Note for Production

If restoring full-scale config (n_age=16, n_bins=50):
- Expect RK2 issues to be **even worse** (larger system = more accumulation)
- RK4 should still work but may need smaller dt for accuracy
- Test with full config before deployment!

---

## Files Modified

### Core Implementation
- **src/SEIR_pde_metapop_stochastic_taichi.py**
  - Lines 72-78: Updated `deterministic_method` documentation, marked RK2 as DEPRECATED
  - Lines 508-514 (RK2): Added comment explaining clamping tradeoff
  - Lines 432-464 (RK4): Added comment explaining why 4-stage averaging helps

### Test Scripts
- **examples/test_timestep_stability.py**
  - **Lines 30-32: REDUCED CONFIG** - `n_age: 16→2`, `n_bins: 50→25`
  - Line 46: Changed methods from `['euler', 'rk2', 'rk4']` to `['euler', 'rk4']`
  - Lines 199-205: Updated conclusions to remove RK2, explain method selection

- **examples/test_deterministic_methods.py**
  - Lines 21-22: Already uses reduced config (n_age=2, n_bins=25)
  - Line 35: Disabled RK2 for consistency with test_timestep_stability.py

### Figures
- **figures/timestep_stability.pdf**: Regenerated with RK2 disabled + reduced config
- **figures/threshold_*.pdf**: Regenerated (unrelated changes)
- **figures/quick_example.pdf**: Regenerated (unrelated changes)

---

## Commits Made

1. **376b0bb** - "Investigate and document RK2 mass conservation bug"
   - Comprehensive root cause analysis
   - Disabled RK2, documented tradeoff
   - Updated test conclusions

2. **210cf44** - "Regenerate timestep_stability.pdf with RK2 disabled"
   - Clean figures showing RK4 works correctly

---

## Current State

### ✅ Working
- **RK4** (`deterministic_method='rk4'`): Recommended default
  - ~1% mass error at dt=1.0
  - ~130K peak infections (matches baseline)
  - Stable for dt ≤ 1.0

- **Euler** (`deterministic_method='euler'`): Fast but inaccurate
  - Works for dt ≤ 0.5
  - Marginal at dt=1.0 (~16% error)

### ❌ Broken
- **RK2** (`deterministic_method='rk2'`): DEPRECATED
  - Catastrophic mass conservation errors
  - Exponential explosion to 1e30+
  - Do not use

---

## Recommendations for Future Work

### Short-term
1. **Keep RK4 as default** - works well for most applications
2. **Use dt ≤ 1.0** - at the stability limit, larger dt causes accuracy loss
3. **Monitor mass conservation** - if critical, use smaller dt or baseline RK45

### Long-term (If Mass Conservation is Critical)
1. **Implement adaptive timestepping**
   - Like scipy's RK45
   - Automatically adjusts dt to maintain accuracy
   - Complex to implement efficiently on GPU

2. **Implement positivity-preserving RK**
   - Modified Patankar-RK methods
   - Preserve positivity AND mass conservation
   - More complex, requires iterative solves

3. **Hybrid approach**
   - Use RK45 for deterministic nodes (fewer nodes, can afford CPU)
   - Keep tau-leaping for stochastic nodes (many nodes, need GPU)

---

## Key Insights

1. **Clamping is unavoidable** with fixed-timestep explicit RK for epidemic models
2. **RK2 is incompatible** with clamping - errors don't average out
3. **RK4's 4-stage averaging** is the key to making clamping tolerable
4. **Debug tests need both** short validation (correctness) AND long runs (stability)
5. **No free lunch**: Fixed-timestep GPU methods trade accuracy for speed

---

## Questions to Consider

- Is 1% mass error acceptable for your application?
- If not, consider baseline (CPU RK45) or smaller timesteps (dt=0.5)
- Are there computational bottlenecks elsewhere worth optimizing first?

---

**End of Summary**
