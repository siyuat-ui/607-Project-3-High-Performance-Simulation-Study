"""Profiling script for baseline performance analysis.

This script runs a representative subset of the simulation with profiling
enabled to identify computational bottlenecks.
"""

import cProfile
import pstats
import io
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dgps import NormalGenerator, ExponentialGenerator, UniformGenerator
from train_and_inference import train_and_generate
from metrics import compute_all_metrics
import torch


def profile_single_experiment(generator, sample_size):
    """Profile a single experiment run.
    
    Parameters
    ----------
    generator : DataGenerator
        Data generator to use
    sample_size : int
        Number of samples
        
    Returns
    -------
    dict
        Timing breakdown by component
    """
    print(f"\n{'='*70}")
    print(f"Profiling: {generator.name}, n={sample_size}")
    print(f"{'='*70}")
    
    timings = {}
    
    # 1. Profile data generation
    start = time.time()
    X_original_np = generator.generate(sample_size)
    if X_original_np.ndim == 1:
        X_original_np = X_original_np.reshape(-1, 1)
    X_original = torch.from_numpy(X_original_np).float()
    timings['data_generation'] = time.time() - start
    print(f"Data generation: {timings['data_generation']:.4f}s")
    
    # 2. Profile training
    start = time.time()
    model, history, X_generated = train_and_generate(
        X_original,
        num_samples=sample_size,
        num_epochs=50,  # Reduced for profiling
        batch_size=128,
        learning_rate=1e-4,
        m=50,
        patience=10,  # Reduced patience
        input_dim=128,
        verbose=False
    )
    timings['training'] = time.time() - start
    print(f"Training: {timings['training']:.4f}s")
    
    # 3. Profile metrics computation
    start = time.time()
    metrics = compute_all_metrics(
        X_original, X_generated,
        verbose=False
    )
    timings['metrics'] = time.time() - start
    print(f"Metrics: {timings['metrics']:.4f}s")
    
    timings['total'] = sum(timings.values())
    timings['generator'] = generator.name
    timings['sample_size'] = sample_size
    timings['n_epochs'] = len(history['loss'])
    
    return timings


def run_detailed_profiling():
    """Run detailed cProfile on a single experiment."""
    print("\n" + "#"*70)
    print("DETAILED PROFILING WITH cProfile")
    print("#"*70)
    
    # Use a single representative experiment
    generator = NormalGenerator(loc=0, scale=1)
    sample_size = 500
    
    print(f"\nProfiling: {generator.name}, n={sample_size}")
    print("This will take a few minutes...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the experiment
    profiler.enable()
    
    X_original_np = generator.generate(sample_size)
    if X_original_np.ndim == 1:
        X_original_np = X_original_np.reshape(-1, 1)
    X_original = torch.from_numpy(X_original_np).float()
    
    model, history, X_generated = train_and_generate(
        X_original,
        num_samples=sample_size,
        num_epochs=50,
        batch_size=128,
        learning_rate=1e-4,
        m=50,
        patience=10,
        input_dim=128,
        verbose=False
    )
    
    metrics = compute_all_metrics(
        X_original, X_generated,
        verbose=False
    )
    
    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*70)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("="*70)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    print("\n" + "="*70)
    print("TOP 20 FUNCTIONS BY TOTAL TIME")
    print("="*70)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(20)
    print(s.getvalue())
    
    # Save detailed stats to file
    output_dir = Path('results/profiling')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    profile_file = output_dir / 'baseline_profile.txt'
    with open(profile_file, 'w') as f:
        ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
        ps.print_stats()
    
    print(f"\nDetailed profile saved to: {profile_file}")
    
    return profiler


def run_component_timing():
    """Run timing analysis across different configurations."""
    print("\n" + "#"*70)
    print("COMPONENT TIMING ANALYSIS")
    print("#"*70)
    
    # Test configurations
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2)
    ]
    
    sample_sizes = [100, 500, 1000]
    
    results = []
    
    for generator in generators:
        for sample_size in sample_sizes:
            timings = profile_single_experiment(generator, sample_size)
            results.append(timings)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    # Calculate percentages
    print("\n" + "="*70)
    print("TIME BREAKDOWN (% of total)")
    print("="*70)
    
    summary = df.groupby('sample_size').agg({
        'data_generation': 'mean',
        'training': 'mean',
        'metrics': 'mean',
        'total': 'mean'
    })
    
    for col in ['data_generation', 'training', 'metrics']:
        summary[f'{col}_pct'] = 100 * summary[col] / summary['total']
    
    print(summary[['data_generation_pct', 'training_pct', 'metrics_pct']].to_string())
    
    # Save results
    output_dir = Path('results/profiling')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'component_timings.csv', index=False)
    summary.to_csv(output_dir / 'timing_summary.csv')
    
    print(f"\nResults saved to: {output_dir}/")
    
    return df, summary


def estimate_full_simulation_time(component_timings):
    """Estimate total time for full simulation based on component timings.
    
    Parameters
    ----------
    component_timings : pd.DataFrame
        DataFrame with timing data
        
    Returns
    -------
    dict
        Estimated times for full simulation
    """
    print("\n" + "="*70)
    print("FULL SIMULATION TIME ESTIMATE")
    print("="*70)
    
    # Full simulation: 5 generators × 3 sample sizes × 10 replications = 150 experiments
    n_generators = 5
    n_sizes = 3
    n_replications = 10
    total_experiments = n_generators * n_sizes * n_replications
    
    # Average time per experiment
    avg_time_per_experiment = component_timings['total'].mean()
    
    # Estimate for full simulation
    estimated_total = avg_time_per_experiment * total_experiments
    
    print(f"Average time per experiment: {avg_time_per_experiment:.2f}s")
    print(f"Total experiments: {total_experiments}")
    print(f"Estimated total time: {estimated_total:.2f}s = {estimated_total/60:.2f} minutes")
    
    # Breakdown by component
    print(f"\nEstimated time breakdown:")
    for component in ['data_generation', 'training', 'metrics']:
        avg_component_time = component_timings[component].mean()
        total_component_time = avg_component_time * total_experiments
        percentage = 100 * avg_component_time / avg_time_per_experiment
        print(f"  {component:20s}: {total_component_time:6.1f}s ({percentage:5.1f}%)")
    
    return {
        'avg_time_per_experiment': avg_time_per_experiment,
        'total_experiments': total_experiments,
        'estimated_total_seconds': estimated_total,
        'estimated_total_minutes': estimated_total / 60
    }


def main():
    """Main profiling routine."""
    print("="*70)
    print("BASELINE PERFORMANCE PROFILING")
    print("="*70)
    print("\nThis script will:")
    print("1. Run detailed cProfile on a representative experiment")
    print("2. Time individual components across multiple configurations")
    print("3. Estimate full simulation runtime")
    print("\nNote: Using reduced epochs (50 instead of 200) for faster profiling")
    print("="*70)
    
    # Run detailed profiling
    profiler = run_detailed_profiling()
    
    # Run component timing
    component_timings, summary = run_component_timing()
    
    # Estimate full simulation time
    estimates = estimate_full_simulation_time(component_timings)
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE!")
    print("="*70)
    print("\nKey files created:")
    print("  - results/profiling/baseline_profile.txt")
    print("  - results/profiling/component_timings.csv")
    print("  - results/profiling/timing_summary.csv")
    print("\nNext steps:")
    print("1. Review the profiling output to identify bottlenecks")
    print("2. Use this data to create docs/BASELINE.md")
    print("3. Begin optimization implementation")
    print("="*70)


if __name__ == "__main__":
    main()
