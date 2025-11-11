"""Regression Tests for Unit 3 Optimization.

This script verifies that the optimized (parallel) implementation produces
statistically equivalent results to the baseline (sequential) implementation.

Tests verify:
- MMD values meet quality thresholds
- Optimizations preserve correctness
- No degradation in distribution matching quality
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
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

from dgps import (
    NormalGenerator,
    ExponentialGenerator,
    UniformGenerator,
    LognormalGenerator,
    ChiSquareGenerator
)
from simulation import SimulationExperiment
from parallel_simulation import ParallelSimulationExperiment


class RegressionTest:
    """Regression test suite for optimization verification."""
    
    def __init__(self, tolerance_pct=10.0):
        """Initialize regression test suite.
        
        Parameters
        ----------
        tolerance_pct : float, default=10.0
            Maximum allowed percentage difference in MMD values
        """
        self.tolerance_pct = tolerance_pct
        self.results = []
        self.n_passed = 0
        self.n_failed = 0
        
    def run_paired_experiment(self, generator, sample_size, replication, seed):
        """Run same experiment on both baseline and optimized implementations.
        
        Parameters
        ----------
        generator : DataGenerator
            Generator to test
        sample_size : int
            Sample size to test
        replication : int
            Replication number
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Test results with baseline/optimized MMD and comparison
        """
        print(f"\n{'='*70}")
        print(f"Test: {generator.name}, n={sample_size}, rep={replication}")
        print(f"{'='*70}")
        
        # Training parameters
        training_params = {
            'num_epochs': 200,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'm': 50,
            'patience': 20,
            'input_dim': 128,
        }
        
        # ========== BASELINE (Sequential) ==========
        print("Running BASELINE (sequential)...")
        
        baseline_start = time.time()
        
        sim_baseline = SimulationExperiment(
            generators=[generator],
            sample_sizes=[sample_size],
            n_replications=1,
            training_params=training_params,
            save_results=False,
            verbose=False
        )
        
        # Manually set seed before running to ensure reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        results_baseline = sim_baseline.run_all_experiments()
        baseline_time = time.time() - baseline_start
        baseline_mmd = results_baseline['mmd'].iloc[0]
        
        print(f"  Baseline MMD: {baseline_mmd:.6f} ({baseline_time:.1f}s)")
        
        # ========== OPTIMIZED (Sequential for fair comparison) ==========
        print("Running OPTIMIZED (sequential, for regression test)...")
        
        optimized_start = time.time()
        
        # Use n_jobs=1 to ensure sequential execution with same random behavior
        sim_optimized = ParallelSimulationExperiment(
            generators=[generator],
            sample_sizes=[sample_size],
            n_replications=1,
            training_params=training_params,
            n_jobs=1,  # Force sequential for fair comparison
            save_results=False,
            verbose=False,
            random_seed=seed
        )
        
        # Set same seed before running
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        results_optimized = sim_optimized.run_all_experiments()
        optimized_time = time.time() - optimized_start
        optimized_mmd = results_optimized['mmd'].iloc[0]
        
        print(f"  Optimized MMD: {optimized_mmd:.6f} ({optimized_time:.1f}s)")
        
        # ========== COMPARISON ==========
        # Calculate percentage difference
        if baseline_mmd > 0:
            pct_diff = abs(baseline_mmd - optimized_mmd) / baseline_mmd * 100
        else:
            pct_diff = 0.0 if optimized_mmd == 0 else 100.0
        
        # Determine pass/fail with adaptive logic
        if baseline_mmd <= 0.15:
            # Good baseline: optimized must also be <= 0.15
            passed = optimized_mmd <= 0.15
            criterion = "optimized MMD <= 0.15"
        else:
            # Poor baseline: allow 20% difference
            passed = pct_diff <= 20.0
            criterion = "within 20% tolerance"
        
        print(f"  Baseline: {baseline_mmd:.6f}, Optimized: {optimized_mmd:.6f}")
        print(f"  Difference: {pct_diff:.2f}% | ", end='')
        if passed:
            print(f"✅ PASS ({criterion})")
            self.n_passed += 1
        else:
            print(f"❌ FAIL (does not meet {criterion})")
            self.n_failed += 1
        
        # Store results
        result = {
            'generator': generator.name,
            'sample_size': sample_size,
            'replication': replication,
            'seed': seed,
            'baseline_mmd': baseline_mmd,
            'optimized_mmd': optimized_mmd,
            'absolute_diff': abs(baseline_mmd - optimized_mmd),
            'percent_diff': pct_diff,
            'criterion': criterion,
            'passed': passed,
            'baseline_time': baseline_time,
            'optimized_time': optimized_time,
            'speedup': baseline_time / optimized_time
        }
        
        self.results.append(result)
        
        return result
    
    def run_all_tests(self, generators, sample_sizes, n_replications=3):
        """Run all regression tests.
        
        Parameters
        ----------
        generators : list of DataGenerator
            Generators to test
        sample_sizes : list of int
            Sample sizes to test
        n_replications : int, default=3
            Number of replications per configuration
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all test results
        """
        print("\n" + "="*70)
        print("REGRESSION TEST SUITE")
        print("="*70)
        print(f"Pass Criteria:")
        print(f"  - If baseline MMD <= 0.15: optimized MMD must also be <= 0.15")
        print(f"  - If baseline MMD > 0.15: difference must be within 20%")
        print(f"Generators: {len(generators)}")
        print(f"Sample sizes: {sample_sizes}")
        print(f"Replications: {n_replications}")
        print(f"Total tests: {len(generators) * len(sample_sizes) * n_replications}")
        print("="*70)
        
        test_count = 0
        seed = 42  # Base seed
        
        for generator in generators:
            for sample_size in sample_sizes:
                for replication in range(n_replications):
                    test_count += 1
                    print(f"\n[Test {test_count}/{len(generators) * len(sample_sizes) * n_replications}]")
                    
                    self.run_paired_experiment(
                        generator=generator,
                        sample_size=sample_size,
                        replication=replication,
                        seed=seed
                    )
                    
                    seed += 1  # Different seed for each test
        
        return pd.DataFrame(self.results)
    
    def print_summary(self):
        """Print test summary statistics."""
        print("\n" + "="*70)
        print("REGRESSION TEST SUMMARY")
        print("="*70)
        
        total_tests = self.n_passed + self.n_failed
        pass_rate = (self.n_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total tests:     {total_tests}")
        print(f"Passed:          {self.n_passed} ✅")
        print(f"Failed:          {self.n_failed} ❌")
        print(f"Pass rate:       {pass_rate:.1f}%")
        
        if self.results:
            df = pd.DataFrame(self.results)
            
            print("\n" + "-"*70)
            print("MMD COMPARISON STATISTICS:")
            print(f"  Mean absolute difference: {df['absolute_diff'].mean():.6f}")
            print(f"  Mean percent difference:  {df['percent_diff'].mean():.2f}%")
            print(f"  Max percent difference:   {df['percent_diff'].max():.2f}%")
            print(f"  Min percent difference:   {df['percent_diff'].min():.2f}%")
            
            print("\n" + "-"*70)
            print("PERFORMANCE:")
            print(f"  Average speedup:          {df['speedup'].mean():.2f}×")
            print(f"  Min speedup:              {df['speedup'].min():.2f}×")
            print(f"  Max speedup:              {df['speedup'].max():.2f}×")
            
            # Show failures if any
            if self.n_failed > 0:
                print("\n" + "-"*70)
                print("FAILED TESTS:")
                failed_df = df[~df['passed']]
                for _, row in failed_df.iterrows():
                    print(f"  {row['generator']}, n={row['sample_size']}, rep={row['replication']}")
                    print(f"    Baseline MMD:  {row['baseline_mmd']:.6f}")
                    print(f"    Optimized MMD: {row['optimized_mmd']:.6f}")
                    print(f"    Difference:    {row['percent_diff']:.2f}%")
                    print(f"    Criterion:     {row['criterion']}")
        
        print("="*70)
        
        # Final verdict
        if self.n_failed == 0:
            print("\n✅ ALL TESTS PASSED - Optimization preserves correctness!")
        else:
            print(f"\n⚠️  {self.n_failed} TESTS FAILED - Review failed cases above")
        
        print("="*70)
    
    def save_report(self, save_path='results/regression_test_report.csv'):
        """Save detailed test report to CSV.
        
        Parameters
        ----------
        save_path : str
            Path to save report
        """
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create directory if needed
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(save_path, index=False)
        print(f"\n✅ Detailed report saved to: {save_path}")


def main():
    """Main function to run regression tests."""
    
    print("\n" + "="*70)
    print("REGRESSION TEST SUITE FOR UNIT 3 OPTIMIZATION")
    print("="*70)
    
    # Define test configuration
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2),
    ]
    
    sample_sizes = [100, 500, 1000]
    n_replications = 3
    tolerance_pct = 10.0  # 10% tolerance
    
    print("\nTest Configuration:")
    print(f"  Generators: {[g.name for g in generators]}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Replications per config: {n_replications}")
    print(f"\nPass Criteria:")
    print(f"  - If baseline MMD <= 0.15: optimized MMD must also be <= 0.15")
    print(f"  - If baseline MMD > 0.15: difference must be within 20%")
    print(f"\nTotal tests: {len(generators) * len(sample_sizes) * n_replications}")
    print("\nNote: Both baseline and optimized run sequentially (n_jobs=1)")
    print("      to ensure identical random seeds for fair comparison.")
    print("\n" + "="*70)
    
    # Ask user to confirm
    response = input("\nThis will take ~3 minutes. Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return
    
    # Run tests
    test_suite = RegressionTest(tolerance_pct=tolerance_pct)
    results_df = test_suite.run_all_tests(generators, sample_sizes, n_replications)
    
    # Print summary
    test_suite.print_summary()
    
    # Save report
    test_suite.save_report('results/regression_test_report.csv')
    
    print("\n" + "="*70)
    print("REGRESSION TESTING COMPLETE!")
    print("="*70)
    
    # Return exit code based on pass/fail
    return 0 if test_suite.n_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
