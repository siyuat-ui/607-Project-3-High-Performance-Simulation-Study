"""Compare sequential vs parallel for FULL simulation (150 experiments).

This script runs both versions with the complete configuration to measure
actual speedup on the full workload.
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

from dgps import (
    NormalGenerator,
    ExponentialGenerator,
    UniformGenerator,
    LognormalGenerator,
    ChiSquareGenerator
)
from simulation import SimulationExperiment
from parallel_simulation import ParallelSimulationExperiment


def main():
    """Run comparison of sequential vs parallel."""
    
    print("\n" + "="*70)
    print("FULL SIMULATION COMPARISON: SEQUENTIAL vs PARALLEL")
    print("="*70)
    print("\nConfiguration:")
    print("  - 5 distributions")
    print("  - 3 sample sizes (100, 500, 1000)")
    print("  - 10 replications")
    print("  - 150 total experiments")
    print("  - 200 epochs (early stopping)")
    print("="*70)
    
    # Setup
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2),
        LognormalGenerator(mean=0, sigma=1),
        ChiSquareGenerator(df=5),
    ]
    
    sample_sizes = [100, 500, 1000]
    n_replications = 10
    
    print("\n" + "="*70)
    print("OPTION 1: Run Sequential First (then Parallel)")
    print("="*70)
    print("Estimated time:")
    print("  Sequential: ~8-10 minutes")
    print("  Parallel:   ~1.5-3 minutes")
    print("  Total:      ~10-13 minutes")
    
    print("\n" + "="*70)
    print("OPTION 2: Run Parallel Only (Recommended)")
    print("="*70)
    print("Estimated time:")
    print("  Parallel:   ~1.5-3 minutes")
    print("\nYou already know baseline is 8-10 minutes from Project 2")
    
    print("\n" + "="*70)
    choice = input("Choose option (1 or 2, or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Exiting.")
        return
    
    if choice == '1':
        # Run sequential
        print("\n" + "="*70)
        print("RUNNING SEQUENTIAL SIMULATION...")
        print("="*70)
        print("This will take 8-10 minutes...")
        
        seq_start = time.time()
        
        sim_seq = SimulationExperiment(
            generators=generators,
            sample_sizes=sample_sizes,
            n_replications=n_replications,
            save_results=False,
            verbose=True
        )
        
        results_seq = sim_seq.run_all_experiments()
        seq_time = time.time() - seq_start
        
        print(f"\nSequential completed in: {seq_time:.1f}s = {seq_time/60:.2f} minutes")
    
    elif choice == '2':
        # Use known baseline
        seq_time = 8.5 * 60  # 8.5 minutes (middle of range)
        results_seq = None
        print(f"\nUsing known baseline: {seq_time/60:.1f} minutes")
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Run parallel
    print("\n" + "="*70)
    print("RUNNING PARALLEL SIMULATION...")
    print("="*70)
    print("This will take 1.5-3 minutes...")
    
    par_start = time.time()
    
    sim_par = ParallelSimulationExperiment(
        generators=generators,
        sample_sizes=sample_sizes,
        n_replications=n_replications,
        n_jobs=-1,
        save_results=True,
        verbose=True
    )
    
    results_par = sim_par.run_all_experiments()
    par_time = time.time() - par_start
    
    print(f"\nParallel completed in: {par_time:.1f}s = {par_time/60:.2f} minutes")
    
    # Compare
    speedup = seq_time / par_time
    efficiency = speedup / sim_par.n_jobs * 100
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Sequential time:  {seq_time/60:.2f} minutes")
    print(f"Parallel time:    {par_time/60:.2f} minutes")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Efficiency:       {efficiency:.1f}% (of {sim_par.n_jobs} cores)")
    print(f"Time saved:       {(seq_time - par_time)/60:.2f} minutes ({(1-par_time/seq_time)*100:.1f}%)")
    print("="*70)
    
    # Validate results if we ran sequential
    if results_seq is not None:
        print("\n" + "="*70)
        print("RESULTS VALIDATION")
        print("="*70)
        
        seq_mmd_mean = results_seq['mmd'].mean()
        par_mmd_mean = results_par['mmd'].mean()
        mmd_diff = abs(seq_mmd_mean - par_mmd_mean)
        
        print(f"Sequential mean MMD: {seq_mmd_mean:.6f}")
        print(f"Parallel mean MMD:   {par_mmd_mean:.6f}")
        print(f"Difference:          {mmd_diff:.6f} ({mmd_diff/seq_mmd_mean*100:.2f}%)")
        
        if mmd_diff / seq_mmd_mean < 0.10:
            print("✅ Results are consistent (< 10% difference)")
        else:
            print("⚠️  Warning: Results differ by > 10%")
        print("="*70)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("Results saved to: results/raw/")


if __name__ == "__main__":
    main()
