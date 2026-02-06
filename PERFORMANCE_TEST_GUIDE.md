# Performance Test Guide

## Overview

**Script**: `examples/performance_test_full_scale.py`

A comprehensive performance benchmark for the full-scale hybrid stochastic/deterministic SEIR model with GPU acceleration.

---

## Purpose

- Benchmark the production-scale configuration (774 nodes, 365 days)
- Generate timing metrics for performance tracking
- Produce diagnostic visualizations to validate simulation behavior
- Verify GPU acceleration is working correctly

---

## Configuration

**System Parameters**:
- **774 nodes** (full spatial network)
- **2 age groups**
- **25 heterogeneity bins**
- **117,648 total ODEs**
- **365 days** simulation duration

**Computational Settings**:
- **Backend**: Metal (Apple Silicon GPU acceleration)
- **Precision**: float32 (only option on Metal backend)
- **Method**: RK4 (most accurate fixed-timestep method)
- **Timestep**: dt = 1.0 days
- **Hybrid threshold**: 100 infections (stochastic below, deterministic above)

---

## How to Run

```bash
uv run python examples/performance_test_full_scale.py
```

**Runtime**: ~5 seconds on M-series Mac
**Output**: Console timing + 2 diagnostic PDFs + timing results file

---

## Output Files

### 1. Timing Results (for tracking)
- **Location**: `results/performance_test_timing.txt`
- **Format**: Machine-readable key-value pairs
- **Contents**:
  ```
  total_time_seconds: 4.24
  simulation_time_seconds: 2.79
  time_per_day_ms: 11.6
  throughput_days_per_second: 86.1
  n_nodes: 774
  n_age: 2
  n_bins: 25
  total_odes: 117648
  duration_days: 365
  backend: metal
  precision: float32
  method: rk4
  timestamp: 2026-02-06T22:30:00
  ```

### 2. Diagnostic Figures (for validation)
- **Location**: `figures/performance_test_diagnostics.pdf` (57KB)
  - 4-panel overview: SEIR dynamics, prevalence heatmap, spatial spread, wave timing
- **Location**: `figures/performance_test_heatmap.pdf` (94KB)
  - Detailed log-scale infection heatmap with timing annotations

**Note**: Figures are .gitignored but regenerate automatically on each run.

---

## Expected Performance

### Current Baseline (M-series Mac, Feb 2026)

```
Total time:      4.24s (0.1 min)
Time per day:    11.6ms
Throughput:      86.1 days/second
```

**Interpretation**: Simulates 86 epidemic days per real-time second

### Performance Indicators

✅ **Good performance**:
- Total time: 4-5 seconds
- Time per day: 10-12ms
- Throughput: 75-90 days/second

⚠️ **Degraded performance** (investigate if you see):
- Total time: >10 seconds
- Time per day: >25ms
- Throughput: <40 days/second

**Common causes of slowdown**:
- GPU not being used (check backend in output)
- Thermal throttling (run fewer concurrent processes)
- Background processes consuming GPU resources

---

## Validation Checks

The script automatically validates:

1. **No NaN values**: Checks for numerical instability
2. **Mass conservation**: Should be <5% error (typically ~1.3% with RK4)
3. **Epidemic dynamics**: Peak should be in reasonable range (10K-100K)

**Expected Results**:
```
Peak infections: ~67,000 at day 59
Attack rate:     ~89% (final recovered/total)
Nodes affected:  ~331/774 (43%)
Mass conservation: ~1.3% error
```

---

## Interpreting Timing Results

### Setup Time
- **Expected**: 1-2 seconds
- **Includes**: Spatial network generation, initial state allocation, Taichi compilation
- **Note**: First run after code changes may be slower due to JIT compilation

### Simulation Time
- **Expected**: 2-3 seconds for 365 days
- **This is the core performance metric**
- **Scales**: Roughly linear with duration_days

### Time per Day
- **Expected**: 8-12ms
- **Most useful metric for extrapolating to longer simulations**
- **Calculate**: `time_per_day_ms * desired_days / 1000` for time estimate

---

## Troubleshooting

### Script fails with "Type f64 not supported"
- **Cause**: Trying to use `use_float64=True` on Metal backend
- **Fix**: Keep `use_float64=False` (float64 not supported on Apple GPU)

### Script fails with "No module named 'taichi'"
- **Cause**: Taichi not installed
- **Fix**: `uv pip install taichi`

### Much slower than expected
- **Check 1**: Verify Metal backend is being used (look for "Taichi backend: metal" in output)
- **Check 2**: Try `backend='cpu'` for comparison - should be ~3-5× slower
- **Check 3**: Reduce problem size to isolate issue (try n_nodes=128)

### NaN values appear
- **Cause**: Numerical instability (rare with RK4 at dt=1.0)
- **Fix**: Reduce timestep to dt=0.5 or use smaller system for debugging

---

## Comparing Performance Over Time

To track performance regression/improvement:

```bash
# Run test and save results
uv run python examples/performance_test_full_scale.py

# Check timing
cat results/performance_test_timing.txt

# Compare with previous runs
# (timing file includes timestamp for tracking)
```

**Baseline to maintain**:
- M-series Mac: ~11ms/day (86 days/sec)
- Should not degrade significantly without major code changes

---

## Scaling Expectations

Based on empirical testing:

| Configuration | ODEs | Time/day | Speedup vs 774 nodes |
|--------------|------|----------|---------------------|
| 128 nodes, 2 age, 25 bins | 19,456 | 7.7ms | 1.5× |
| 774 nodes, 2 age, 25 bins | 117,648 | 11.6ms | 1.0× (baseline) |

**Key insight**: GPU parallelization overhead is sublinear
- 6× larger problem → only 1.5× slower
- Good parallelization efficiency

---

## Integration with CI/CD

To use in automated testing:

```bash
# Run test
uv run python examples/performance_test_full_scale.py

# Parse timing results
TOTAL_TIME=$(grep "total_time_seconds:" results/performance_test_timing.txt | cut -d' ' -f2)

# Check against threshold (fail if >10s)
if (( $(echo "$TOTAL_TIME > 10" | bc -l) )); then
    echo "Performance regression detected!"
    exit 1
fi
```

---

## Notes for Other Agent

### Key Points

1. **This is a production benchmark** - uses full-scale 774-node network
2. **Timing results are saved** - check `results/performance_test_timing.txt`
3. **Figures are for validation** - ensure epidemic dynamics look reasonable
4. **float32 is required** - Metal backend doesn't support float64
5. **RK4 is recommended** - RK2 has catastrophic mass conservation errors (see RECENT_CHANGES.md)

### Expected Output Structure

```
results/
  performance_test_timing.txt      # Machine-readable timing data
figures/
  performance_test_diagnostics.pdf # Visual validation (gitignored)
  performance_test_heatmap.pdf     # Detailed heatmap (gitignored)
```

### When to Re-run

- After any changes to GPU kernels
- After updating Taichi version
- When comparing different hardware
- Weekly as performance regression test

---

**Last Updated**: 2026-02-06
**Baseline System**: M-series Mac (Metal backend)
**Expected Runtime**: ~4-5 seconds
