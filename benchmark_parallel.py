"""Benchmark: Original Baseline vs Final Optimized Version

Compares:
- Baseline: Sequential execution with original methods.py
- Optimized: Parallel execution with optimized loss function

Uses reduced configuration for faster benchmarking.
"""

import os
import sys
from pathlib import Path
import time
import pandas as pd

# Set thread limits BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dgps import NormalGenerator, ExponentialGenerator, UniformGenerator, LognormalGenerator, ChiSquareGenerator

# Baseline uses original methods
from simulation import SimulationExperiment

# Optimized uses parallel + optimized methods
from parallel_simulation import ParallelSimulationExperiment


def run_benchmark():
    """Run benchmark comparing baseline vs optimized."""
    
    print("="*70)
    print("BASELINE vs OPTIMIZED BENCHMARK")
    print("="*70)
    print("\nVersions:")
    print("  BASELINE:  Sequential + original loss (Project 2)")
    print("  OPTIMIZED: Parallel + optimized loss (current)")
    print("\nConfiguration (reduced for benchmarking):")
    print("  - 5 distributions (Normal, Exponential, Uniform, Lognormal, Chi-Square)")
    print("  - 5 sample sizes (100, 300, 500, 1000, 2000)")
    print("  - 10 replications")
    print("  - Total: 250 experiments")
    print("  - Epochs: 200 (early stopping with patience=20)")
    print("="*70)
    
    # Setup
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2),
        LognormalGenerator(mean=0, sigma=1),
        ChiSquareGenerator(df=5),
    ]
    
    sample_sizes = [100, 300, 500, 1000, 2000]
    n_replications = 10
    
    training_params = {
        'num_epochs': 200,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'm': 50,
        'patience': 20,
        'input_dim': 128,
    }
    
    # ========== BASELINE (Sequential + Original) ==========
    print("\n" + "="*70)
    print("RUNNING BASELINE (Sequential + Original Loss)...")
    print("="*70)
    print("Expected time: ~8-10 minutes")
    
    baseline_start = time.time()
    
    sim_baseline = SimulationExperiment(
        generators=generators,
        sample_sizes=sample_sizes,
        n_replications=n_replications,
        training_params=training_params,
        save_results=False,
        verbose=False
    )
    
    results_baseline = sim_baseline.run_all_experiments()
    baseline_time = time.time() - baseline_start
    
    print(f"\n✅ Baseline complete: {baseline_time:.1f}s ({baseline_time/60:.2f} min)")
    
    # ========== OPTIMIZED (Parallel + Optimized Loss) ==========
    print("\n" + "="*70)
    print("RUNNING OPTIMIZED (Parallel + Optimized Loss)...")
    print("="*70)
    print("Expected time: ~60-120 seconds")
    
    optimized_start = time.time()
    
    sim_optimized = ParallelSimulationExperiment(
        generators=generators,
        sample_sizes=sample_sizes,
        n_replications=n_replications,
        training_params=training_params,
        n_jobs=-1,  # Use all cores
        save_results=False,
        verbose=False
    )
    
    results_optimized = sim_optimized.run_all_experiments()
    optimized_time = time.time() - optimized_start
    
    print(f"\n✅ Optimized complete: {optimized_time:.1f}s ({optimized_time/60:.2f} min)")
    
    # ========== COMPARISON ==========
    speedup = baseline_time / optimized_time
    n_cores = sim_optimized.n_jobs
    efficiency = speedup / n_cores * 100
    
    print("\n" + "="*70)
    print("BENCHMARK RESULTS (150 experiments)")
    print("="*70)
    print(f"Baseline (sequential + original):  {baseline_time:6.1f}s ({baseline_time/60:.2f} min)")
    print(f"Optimized (parallel + fast loss):  {optimized_time:6.1f}s ({optimized_time/60:.2f} min)")
    print(f"")
    print(f"Speedup:           {speedup:.2f}×")
    print(f"Cores used:        {n_cores}")
    print(f"Efficiency:        {efficiency:.1f}%")
    print(f"Time saved:        {baseline_time - optimized_time:.1f}s ({(1-optimized_time/baseline_time)*100:.1f}%)")
    print("="*70)
    
    # ========== VALIDATION ==========
    print("\n" + "="*70)
    print("RESULTS VALIDATION")
    print("="*70)
    
    baseline_mmd = results_baseline['mmd'].mean()
    optimized_mmd = results_optimized['mmd'].mean()
    mmd_diff_pct = abs(baseline_mmd - optimized_mmd) / baseline_mmd * 100
    
    print(f"Baseline mean MMD:  {baseline_mmd:.6f}")
    print(f"Optimized mean MMD: {optimized_mmd:.6f}")
    print(f"Difference:         {mmd_diff_pct:.2f}%")
    
    if mmd_diff_pct < 10:
        print("✅ Results consistent (< 10% difference)")
    else:
        print("⚠️  Warning: Results differ by > 10%")
    
    print("="*70)
    
    # ========== OPTIMIZATION BREAKDOWN ==========
    print("\n" + "="*70)
    print("OPTIMIZATION BREAKDOWN")
    print("="*70)
    print("Total speedup comes from TWO optimizations:")
    print(f"  1. Parallelization:    ~{n_cores*0.9:.1f}× (theoretical, 90% efficiency)")
    print(f"  2. Optimized loss:     ~1.05-1.10× (5-10% improvement)")
    print(f"  Combined measured:     {speedup:.2f}×")
    print("")
    print("Note: Actual speedup includes both optimizations together.")
    print("="*70)
    
    print("\n✅ Benchmark complete!")
    print("\nFor documentation:")
    print(f"  - Baseline: {baseline_time/60:.1f} min (sequential, original loss)")
    print(f"  - Optimized: {optimized_time/60:.1f} min (parallel, optimized loss)")
    print(f"  - Speedup: {speedup:.2f}×")


if __name__ == "__main__":
    run_benchmark()
