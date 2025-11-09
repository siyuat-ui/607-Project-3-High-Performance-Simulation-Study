"""Run full parallel simulation with all 150 experiments.

This script runs the complete simulation study (5 generators × 3 sample sizes × 10 replications)
using parallelization across all available CPU cores.
"""

import os
import sys
from pathlib import Path
import time

# CRITICAL: Set thread limits BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parallel_simulation import run_parallel_simulation


def main():
    """Run full parallel simulation."""
    
    print("="*70)
    print("FULL PARALLEL SIMULATION")
    print("="*70)
    print("\nConfiguration:")
    print("  - 5 distributions (Normal, Exponential, Uniform, Lognormal, Chi-Square)")
    print("  - 3 sample sizes (100, 500, 1000)")
    print("  - 10 replications per configuration")
    print("  - Total: 150 experiments")
    print("  - Epochs: 200 (with early stopping)")
    print("  - Parallelization: ALL available CPU cores")
    print("="*70)
    
    # Run full parallel simulation
    print("\nStarting parallel simulation...")
    print("This will take several minutes (estimated: 1.5-3 minutes on 8-10 cores)")
    print()
    
    start_time = time.time()
    
    results_df, summary_df = run_parallel_simulation(
        generators=None,              # Use all 5 default generators
        n_replications=10,            # Full replications
        sample_sizes=[100, 500, 1000], # All sample sizes
        n_jobs=-1,                    # Use all CPU cores
        verbose=True,
        save_results=True
    )
    
    total_time = time.time() - start_time
    
    # Display results summary
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time:.1f}s = {total_time/60:.2f} minutes")
    print(f"Total experiments: {len(results_df)}")
    print()
    
    print("Summary Statistics:")
    print("-"*70)
    print(f"Average MMD:           {results_df['mmd'].mean():.6f} ± {results_df['mmd'].std():.6f}")
    print(f"Average Mean Distance: {results_df['mean_distance'].mean():.6f}")
    print(f"Average Training Time: {results_df['training_time'].mean():.2f}s per experiment")
    print(f"Average Epochs:        {results_df['n_epochs'].mean():.1f}")
    print()
    
    print("Performance by Distribution:")
    print("-"*70)
    perf_summary = results_df.groupby('generator').agg({
        'mmd': 'mean',
        'training_time': 'mean',
        'n_epochs': 'mean'
    }).round(4)
    print(perf_summary.to_string())
    print()
    
    print("Results saved to:")
    print(f"  - results/raw/parallel_simulation_results_*.csv")
    print(f"  - results/raw/parallel_simulation_summary_*.csv")
    print("="*70)
    
    # Compare to baseline
    baseline_time = 8.5 * 60  # 8.5 minutes (middle of 8-10 min range)
    speedup = baseline_time / total_time
    
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)
    print(f"Baseline (sequential): ~8.5 minutes")
    print(f"Parallel (this run):   {total_time/60:.2f} minutes")
    print(f"Speedup:               {speedup:.2f}x")
    print(f"Time saved:            {(baseline_time - total_time)/60:.2f} minutes ({(1-total_time/baseline_time)*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
