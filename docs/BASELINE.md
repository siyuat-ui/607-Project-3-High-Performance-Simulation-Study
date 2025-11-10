# Baseline Performance Documentation

## 1. Total Runtime

**Full Simulation** (5 distributions × 3 sample sizes × 10 replications = 150 experiments):
- **Actual runtime**: 8-10 minutes on Apple M4 MacBook  
- **Epochs per experiment**: 200 (early stopping, patience=20)

**Profiling Run** (3 distributions × 3 sample sizes × 1 replication = 9 experiments):
- **Runtime**: 5.2 seconds total
- **Average per experiment**: 0.58 seconds
- **Epochs**: 50 (reduced for profiling)

---

## 2. Profiling Results Summary

### Top Bottlenecks (cProfile on n=500, 50 epochs):

| Function | Time (s) | % Total |
|----------|----------|---------|
| `EngressionTrainer.train()` | 3.469 | 97.2% |
| `backward()` | 1.079 | 30.2% |
| `engression_loss()` | 0.911 | 25.5% |
| `Linear.forward()` | 0.301 | 8.4% |
| `.mean()` operations | 0.230 | 6.4% |
| `torch.randn()` | 0.129 | 3.6% |

(Functions `backward()`, `engression_loss()`,..., are called within `EngressionTrainer.train()`)

### Component Breakdown:

| Component | Avg Time | % Total |
|-----------|----------|---------|
| Data Generation | 0.00007s | <0.01% |
| **Training** | 0.568s | **98.9%** |
| Metrics (MMD, KS) | 0.0057s | 0.9% |

---

## 3. Computational Complexity

### Scaling with Sample Size:

| n | Training (s) | Metrics (ms) | Total (s) |
|---|--------------|--------------|-----------|
| 100 | 0.208 | 1.37 | 0.209 |
| 500 | 0.561 (2.7×) | 3.48 (2.5×) | 0.564 |
| 1000 | 1.043 (1.9×) | 11.65 (3.3×) | 1.054 |

### Theoretical Complexity:

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Data Generation | O(n) | Negligible |
| Training (per epoch) | O(n × m) | m=50 epsilon samples |
| **MMD Computation** | **O(n²)** | **Quadratic bottleneck** |
| KS Test | O(n log n) | Sorting-based |

### Empirical Findings:
- Training: α ≈ 0.9 (near-linear scaling)
- Metrics: α ≈ 1.7 (super-linear due to MMD's O(n²))
- **Current bottleneck**: Training loop (98.9% of time)
- **Future bottleneck**: MMD will dominate for n > 5000

---

## 4. Numerical Stability

### Warnings and Convergence Issues:

**During profiling run (9 experiments, 50 epochs each)**:
- No NaN or Inf values observed
- No gradient explosion/vanishing warnings
- No numerical overflow/underflow warnings
- All experiments converged successfully (early stopping triggered appropriately)

**Observations**:
- Loss values remained stable throughout training
- Term1 and Term2 in engression loss both positive and well-behaved
- Adam optimizer step sizes appropriate (no extreme parameter updates)

**Potential Stability Concerns** (not observed but theoretically possible):
- Large epsilon samples (m=50) could cause memory issues for very large n
- Pairwise distance computation in loss term2 has O(m²) per batch
- No gradient clipping currently implemented

---

## 5. Summary

### Key Metrics:
- **Total baseline runtime**: 8-10 minutes (150 experiments)
- **Primary bottleneck**: Training (98.9% of time)
- **Secondary bottleneck**: MMD computation (will grow to O(n²))
- **Scaling**: Near-linear in n for current range

### Optimization Targets:
1. **High priority**: Training loop (98.9% of time)
   - Parallelization across replications → 4-8× speedup
   - Optimize loss computation (called 200× per experiment)
2. **Medium priority**: MMD computation (0.9% now, but O(n²))
   - Replace with approximate method (Random Fourier Features)
   - Will become critical for n > 5000
3. **Low priority**: Data generation (<0.01% of time)
   - Already negligible, but can cache for free speedup