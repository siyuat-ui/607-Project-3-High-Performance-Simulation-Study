# Optimization Documentation

## Summary

Two optimization strategies implemented:
1. **Parallelization** across CPU cores (5.34× speedup)
2. **Algorithmic improvement** in loss computation (~5% additional gain)

**Combined result**: 8.5 min → 1.59 min (5.34× speedup, 81.3% time reduction)

---

## Optimization 1: Parallelization

### Problem Identified
Training loop dominates 98.9% of runtime. All 150 experiments run sequentially but are independent.

### Solution Implemented
Use `multiprocessing.Pool` to distribute experiments across CPU cores.

### Code Comparison

**Before**:
```python
for generator in generators:
    for size in sample_sizes:
        for rep in range(n_replications):
            result = run_single_experiment(...)
```

**After**:
```python
experiment_args = [(config, size, rep, seed) for ...]
with Pool(processes=cpu_count()) as pool:
    results = pool.map(worker_function, experiment_args)
```

### Performance Impact

| Metric | Sequential | Parallel | Improvement |
|--------|-----------|----------|-------------|
| Runtime | 510s (8.5 min) | 95s (1.59 min) | **5.34×** |
| Per experiment | 3.4s | 0.63s | 5.4× |
| Efficiency | - | 53.4% | (10 cores) |

### Trade-offs
- Massive speedup, no accuracy loss
- 5× memory usage (5GB total)
- +300 lines code complexity

---

## Optimization 2: Algorithmic Improvement

### Problem Identified
`engression_loss()` uses inefficient tensor operations (manual broadcasting, mask creation).

### Solution Implemented
1. Single forward pass for all epsilons
2. `torch.cdist()` for GPU-optimized distances
3. Diagonal extraction instead of mask
4. Reduced memory allocations

### Code Comparison

**Before**:
```python
epsilons = torch.randn(batch_size, m, input_dim)
g_eps = g(epsilons.view(-1, input_dim)).view(batch_size, m, output_dim)
X_expanded = X_batch.unsqueeze(1).expand(-1, m, -1)
term1 = torch.norm(X_expanded - g_eps, dim=2).mean(dim=1)
```

**After**:
```python
epsilons = torch.randn(batch_size * m, input_dim)
g_eps = g(epsilons).view(batch_size, m, output_dim)
term1 = torch.cdist(X_batch.unsqueeze(1), g_eps, p=2).squeeze(1).mean(dim=1)
```

### Performance Impact
- Additional ~5% speedup
- Loss time: 25.5% → 20% of epoch time
- Memory: -10% (no mask allocation)

### Trade-offs
- GPU-optimized, cleaner code
- Same accuracy
- Weaker readability (I suppose so becasue I am not that familiar with GPU optimization)

---

## Profiling Evidence

**Component timing** (from baseline profiling):
```
Training:          98.9% of time
  - Loss computation: 25.5% per epoch
  - Backprop:         30.2%
  - Optimizer:        14.9%
Metrics (MMD):      0.9%
Data generation:   <0.01%
```

**Optimization targets correctly identified**: Training (98.9%) and loss computation (25.5%).

**Results of `make parallel`**:

```bash
======================================================================
SPEEDUP ANALYSIS
======================================================================
Baseline (sequential): ~8.5 minutes
Parallel (this run):   1.59 minutes
Speedup:               5.34x
Time saved:            6.91 minutes (81.3%)
======================================================================
```

---

## Lessons Learned

### Which optimizations provided the best return on investment?
1. **Parallelization** - 5.34× with minimal code changes
2. **Library functions** - `torch.cdist()` beats manual operations
3. **Profile first** - Confirmed training was bottleneck before optimizing

### What surprised me about where time was actually spent?
- I didn't know that `Backprop` could cost ~30% time. This part is almost impossible to optimize.

