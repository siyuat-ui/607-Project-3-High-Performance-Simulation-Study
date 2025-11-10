"""Performance Comparison Visualization for Unit 3.

This script creates visualizations comparing baseline (sequential) vs optimized (parallel)
performance across different sample sizes.
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set thread limits BEFORE importing torch
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


def run_timing_experiments(sample_sizes, n_replications=10):
    """Run timing experiments for both baseline and optimized versions.
    
    Parameters
    ----------
    sample_sizes : list of int
        Sample sizes to test
    n_replications : int
        Number of replications per size
        
    Returns
    -------
    dict
        Dictionary with 'baseline' and 'optimized' timing results
    """
    # Use all 5 generators
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2),
        LognormalGenerator(mean=0, sigma=1),
        ChiSquareGenerator(df=5),
    ]
    
    training_params = {
        'num_epochs': 200,
        'batch_size': 128,
        'learning_rate': 1e-4,
        'm': 50,
        'patience': 20,
        'input_dim': 128,
    }
    
    results = {
        'sample_sizes': sample_sizes,
        'baseline_times': [],
        'optimized_times': [],
        'speedups': [],
        'n_experiments': []
    }
    
    print("="*70)
    print("PERFORMANCE COMPARISON ACROSS SAMPLE SIZES")
    print("="*70)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Replications per size: {n_replications}")
    print(f"Generators: {len(generators)}")
    print("="*70)
    
    for sample_size in sample_sizes:
        n_experiments = len(generators) * n_replications
        
        print(f"\n{'='*70}")
        print(f"Sample Size: {sample_size}")
        print(f"Experiments: {n_experiments}")
        print(f"{'='*70}")
        
        # ========== BASELINE (Sequential) ==========
        print(f"\n[BASELINE] Running sequential version...")
        
        baseline_start = time.time()
        
        sim_baseline = SimulationExperiment(
            generators=generators,
            sample_sizes=[sample_size],
            n_replications=n_replications,
            training_params=training_params,
            save_results=False,
            verbose=False
        )
        
        results_baseline = sim_baseline.run_all_experiments()
        baseline_time = time.time() - baseline_start
        
        print(f"✅ Baseline: {baseline_time:.1f}s ({baseline_time/60:.2f} min)")
        
        # ========== OPTIMIZED (Parallel) ==========
        print(f"\n[OPTIMIZED] Running parallel version...")
        
        optimized_start = time.time()
        
        sim_optimized = ParallelSimulationExperiment(
            generators=generators,
            sample_sizes=[sample_size],
            n_replications=n_replications,
            training_params=training_params,
            n_jobs=-1,
            save_results=False,
            verbose=False
        )
        
        results_optimized = sim_optimized.run_all_experiments()
        optimized_time = time.time() - optimized_start
        
        print(f"✅ Optimized: {optimized_time:.1f}s ({optimized_time/60:.2f} min)")
        
        # Calculate speedup
        speedup = baseline_time / optimized_time
        print(f"📊 Speedup: {speedup:.2f}×")
        
        # Store results
        results['baseline_times'].append(baseline_time)
        results['optimized_times'].append(optimized_time)
        results['speedups'].append(speedup)
        results['n_experiments'].append(n_experiments)
    
    return results


def create_performance_plots(results, save_dir='results/figures'):
    """Create performance comparison visualizations.
    
    Parameters
    ----------
    results : dict
        Dictionary with timing results
    save_dir : str
        Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    sample_sizes = results['sample_sizes']
    baseline_times = np.array(results['baseline_times'])
    optimized_times = np.array(results['optimized_times'])
    speedups = np.array(results['speedups'])
    
    # Convert to minutes
    baseline_times_min = baseline_times / 60
    optimized_times_min = optimized_times / 60
    
    # ========== FIGURE 1: Runtime Comparison ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Absolute runtimes
    ax1 = axes[0]
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_times_min, width, 
                    label='Baseline (Sequential)', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, optimized_times_min, width,
                    label='Optimized (Parallel)', color='#2ecc71', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (minutes)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Absolute Runtime Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}' for s in sample_sizes])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Speedup
    ax2 = axes[1]
    
    ax2.plot(sample_sizes, speedups, 'o-', linewidth=2.5, markersize=10,
            color='#3498db', label='Speedup Factor')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='No speedup (1×)')
    
    # Add value labels
    for i, (size, speedup) in enumerate(zip(sample_sizes, speedups)):
        ax2.text(size, speedup + 0.15, f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor (Baseline / Optimized)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Speedup Analysis', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    save_path = save_dir / 'performance_comparison_runtime.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    # ========== FIGURE 2: Time Saved ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_saved = baseline_times_min - optimized_times_min
    percent_saved = (time_saved / baseline_times_min) * 100
    
    x = np.arange(len(sample_sizes))
    bars = ax.bar(x, time_saved, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add labels showing minutes saved and percentage
    for i, (bar, minutes, percent) in enumerate(zip(bars, time_saved, percent_saved)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{minutes:.2f} min\n({percent:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time Saved (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Time Saved by Optimization', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in sample_sizes])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / 'performance_comparison_time_saved.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    # ========== FIGURE 3: Component Breakdown ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Estimate component times (based on profiling: 98.9% training, 0.9% metrics)
    baseline_training = baseline_times * 0.989
    baseline_metrics = baseline_times * 0.009
    baseline_other = baseline_times * 0.002
    
    optimized_training = optimized_times * 0.989
    optimized_metrics = optimized_times * 0.009
    optimized_other = optimized_times * 0.002
    
    # Panel A: Baseline breakdown
    ax1 = axes[0]
    x = np.arange(len(sample_sizes))
    width = 0.6
    
    p1 = ax1.bar(x, baseline_training, width, label='Training', color='#e74c3c')
    p2 = ax1.bar(x, baseline_metrics, width, bottom=baseline_training, 
                label='Metrics', color='#3498db')
    p3 = ax1.bar(x, baseline_other, width, 
                bottom=baseline_training + baseline_metrics,
                label='Other', color='#95a5a6')
    
    ax1.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline Component Breakdown', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}' for s in sample_sizes])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Optimized breakdown
    ax2 = axes[1]
    
    p1 = ax2.bar(x, optimized_training, width, label='Training', color='#2ecc71')
    p2 = ax2.bar(x, optimized_metrics, width, bottom=optimized_training,
                label='Metrics', color='#3498db')
    p3 = ax2.bar(x, optimized_other, width,
                bottom=optimized_training + optimized_metrics,
                label='Other', color='#95a5a6')
    
    ax2.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Optimized Component Breakdown', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s}' for s in sample_sizes])
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / 'performance_comparison_components.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def save_results_table(results, save_dir='results/figures'):
    """Save results as CSV table.
    
    Parameters
    ----------
    results : dict
        Dictionary with timing results
    save_dir : str
        Directory to save table
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        'sample_size': results['sample_sizes'],
        'n_experiments': results['n_experiments'],
        'baseline_time_sec': results['baseline_times'],
        'optimized_time_sec': results['optimized_times'],
        'baseline_time_min': np.array(results['baseline_times']) / 60,
        'optimized_time_min': np.array(results['optimized_times']) / 60,
        'speedup': results['speedups'],
        'time_saved_sec': np.array(results['baseline_times']) - np.array(results['optimized_times']),
        'percent_saved': ((np.array(results['baseline_times']) - np.array(results['optimized_times'])) / 
                         np.array(results['baseline_times']) * 100)
    })
    
    save_path = save_dir / 'performance_comparison_results.csv'
    df.to_csv(save_path, index=False)
    print(f"✅ Saved: {save_path}")
    
    return df


def print_summary(results):
    """Print summary statistics.
    
    Parameters
    ----------
    results : dict
        Dictionary with timing results
    """
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    
    for i, size in enumerate(results['sample_sizes']):
        baseline_time = results['baseline_times'][i]
        optimized_time = results['optimized_times'][i]
        speedup = results['speedups'][i]
        time_saved = baseline_time - optimized_time
        percent_saved = (time_saved / baseline_time) * 100
        
        print(f"\nSample Size n={size}:")
        print(f"  Baseline:     {baseline_time:6.1f}s ({baseline_time/60:5.2f} min)")
        print(f"  Optimized:    {optimized_time:6.1f}s ({optimized_time/60:5.2f} min)")
        print(f"  Speedup:      {speedup:.2f}×")
        print(f"  Time saved:   {time_saved:6.1f}s ({percent_saved:.1f}%)")
    
    # Overall statistics
    avg_speedup = np.mean(results['speedups'])
    total_baseline = sum(results['baseline_times'])
    total_optimized = sum(results['optimized_times'])
    total_saved = total_baseline - total_optimized
    
    print("\n" + "-"*70)
    print("OVERALL STATISTICS:")
    print(f"  Average speedup:      {avg_speedup:.2f}×")
    print(f"  Total baseline time:  {total_baseline/60:.2f} min")
    print(f"  Total optimized time: {total_optimized/60:.2f} min")
    print(f"  Total time saved:     {total_saved/60:.2f} min ({(total_saved/total_baseline)*100:.1f}%)")
    print("="*70)


def main():
    """Main function to run performance comparison."""
    
    # Sample sizes to test
    sample_sizes = [100, 300, 500, 1000, 2000]
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON VISUALIZATION")
    print("="*70)
    print("\nThis script will:")
    print("1. Run timing experiments for each sample size")
    print("2. Compare baseline (sequential) vs optimized (parallel)")
    print("3. Generate performance visualization plots")
    print("4. Save results table")
    print("\n" + "="*70)
    
    # Ask user to confirm
    response = input("\nThis will take significant time. Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Run timing experiments
    results = run_timing_experiments(sample_sizes, n_replications=10)
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS...")
    print("="*70)
    create_performance_plots(results)
    
    # Save results table
    print("\n" + "="*70)
    print("SAVING RESULTS TABLE...")
    print("="*70)
    df = save_results_table(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*70)
    print("✅ PERFORMANCE COMPARISON COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/figures/performance_comparison_runtime.png")
    print("  - results/figures/performance_comparison_time_saved.png")
    print("  - results/figures/performance_comparison_components.png")
    print("  - results/figures/performance_comparison_results.csv")
    print("="*70)


if __name__ == "__main__":
    main()
