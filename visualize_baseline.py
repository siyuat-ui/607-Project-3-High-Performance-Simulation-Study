"""Create visualizations for baseline performance analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_complexity_plots():
    """Create computational complexity visualization plots."""
    
    # Load timing data
    df = pd.read_csv('results/profiling/component_timings.csv')
    
    # Create output directory
    output_dir = Path('results/profiling')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data by sample size
    sizes = sorted(df['sample_size'].unique())
    
    # Average across generators for each size
    train_times = df.groupby('sample_size')['training'].mean().values
    metrics_times = df.groupby('sample_size')['metrics'].mean().values * 1000  # Convert to ms
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ====== Plot 1: Log-log scaling plot ======
    ax1 = axes[0]
    
    # Plot training time
    ax1.loglog(sizes, train_times, 'o-', markersize=10, linewidth=2, 
               label='Training', color='#2ecc71')
    
    # Plot metrics time (on secondary axis for scale)
    ax1_metrics = ax1.twinx()
    ax1_metrics.loglog(sizes, metrics_times, 's-', markersize=10, linewidth=2,
                       label='Metrics (MMD, KS)', color='#e74c3c')
    
    # Fit power laws
    log_sizes = np.log(sizes)
    log_train = np.log(train_times)
    train_coef = np.polyfit(log_sizes, log_train, 1)
    train_alpha = train_coef[0]
    
    log_metrics = np.log(metrics_times)
    metrics_coef = np.polyfit(log_sizes, log_metrics, 1)
    metrics_alpha = metrics_coef[0]
    
    # Add fitted lines
    sizes_fit = np.logspace(np.log10(100), np.log10(1000), 100)
    train_fit = np.exp(train_coef[1]) * sizes_fit**train_alpha
    metrics_fit = np.exp(metrics_coef[1]) * sizes_fit**metrics_alpha
    
    ax1.loglog(sizes_fit, train_fit, '--', color='#27ae60', alpha=0.5,
               label=f'Training fit: O(n^{train_alpha:.2f})')
    ax1_metrics.loglog(sizes_fit, metrics_fit, '--', color='#c0392b', alpha=0.5,
                       label=f'Metrics fit: O(n^{metrics_alpha:.2f})')
    
    # Labels and formatting
    ax1.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold', color='#2ecc71')
    ax1_metrics.set_ylabel('Metrics Time (milliseconds)', fontsize=12, fontweight='bold', color='#e74c3c')
    ax1.set_title('Computational Complexity Scaling', fontsize=14, fontweight='bold')
    
    # Set tick colors
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1_metrics.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_metrics.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    ax1.grid(True, alpha=0.3, which='both')
    
    # ====== Plot 2: Component breakdown bar chart ======
    ax2 = axes[1]
    
    # Data for stacked bar chart
    x = np.arange(len(sizes))
    width = 0.6
    
    # Convert everything to milliseconds for uniform scale
    train_ms = train_times * 1000
    
    # Create stacked bars
    p1 = ax2.bar(x, train_ms, width, label='Training', color='#2ecc71')
    p2 = ax2.bar(x, metrics_times, width, bottom=train_ms, 
                 label='Metrics', color='#e74c3c')
    
    # Add percentage labels
    totals = train_ms + metrics_times
    for i, (t, m) in enumerate(zip(train_ms, metrics_times)):
        total = t + m
        train_pct = 100 * t / total
        metrics_pct = 100 * m / total
        
        # Training percentage (middle of bar)
        ax2.text(i, t/2, f'{train_pct:.1f}%', ha='center', va='center',
                fontweight='bold', fontsize=10, color='white')
        
        # Metrics percentage (top of bar)
        if metrics_pct > 1:  # Only show if visible
            ax2.text(i, t + m/2, f'{metrics_pct:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=9, color='white')
    
    ax2.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Component Breakdown by Sample Size', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'n={s}' for s in sizes])
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_complexity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'baseline_complexity_analysis.png'}")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Training time scaling: O(n^{train_alpha:.2f}) - approximately linear")
    print(f"Metrics time scaling: O(n^{metrics_alpha:.2f}) - super-linear (MMD is O(n²))")
    print("\nComponent contribution:")
    for i, size in enumerate(sizes):
        total = train_ms[i] + metrics_times[i]
        print(f"  n={size:4d}: Training {100*train_ms[i]/total:5.1f}%, Metrics {100*metrics_times[i]/total:4.1f}%")
    print("="*70)


def create_profiling_summary_table():
    """Create a summary table from profiling results."""
    
    df = pd.read_csv('results/profiling/component_timings.csv')
    
    print("\n" + "="*70)
    print("BASELINE PROFILING SUMMARY")
    print("="*70)
    print("\nPer-experiment timing (averaged across generators):")
    print()
    
    summary = df.groupby('sample_size').agg({
        'data_generation': 'mean',
        'training': 'mean',
        'metrics': 'mean',
        'total': 'mean',
        'n_epochs': 'mean'
    })
    
    summary['data_gen_ms'] = summary['data_generation'] * 1000
    summary['training_s'] = summary['training']
    summary['metrics_ms'] = summary['metrics'] * 1000
    summary['total_s'] = summary['total']
    
    # Calculate percentages
    summary['train_pct'] = 100 * summary['training'] / summary['total']
    summary['metrics_pct'] = 100 * summary['metrics'] / summary['total']
    
    print(summary[['data_gen_ms', 'training_s', 'metrics_ms', 'total_s', 
                   'train_pct', 'metrics_pct', 'n_epochs']].to_string())
    
    print("\n" + "="*70)
    print("FULL SIMULATION ESTIMATES")
    print("="*70)
    
    # Estimate full simulation
    avg_time = df['total'].mean()
    n_experiments = 150  # 5 dist × 3 sizes × 10 reps
    
    # Scale up epochs (profiling used 50, full uses 200)
    epoch_scale = 200 / 50
    estimated_time = avg_time * n_experiments * epoch_scale
    
    print(f"\nAverage per experiment (50 epochs): {avg_time:.2f}s")
    print(f"Scaled for 200 epochs: {avg_time * epoch_scale:.2f}s")
    print(f"Total experiments: {n_experiments}")
    print(f"Estimated total runtime: {estimated_time:.1f}s = {estimated_time/60:.1f} minutes")
    print("="*70)


if __name__ == "__main__":
    print("Creating baseline performance visualizations...")
    create_complexity_plots()
    create_profiling_summary_table()
    print("\nDone! Check results/profiling/ for outputs.")
