"""Parallelized simulation orchestration for engression experiments.

This module provides a parallelized version of SimulationExperiment that runs
multiple replications simultaneously across CPU cores.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

from dgps import (
    NormalGenerator,
    ExponentialGenerator,
    UniformGenerator,
    LognormalGenerator,
    ChiSquareGenerator
)
from train_and_inference_optimized import train_and_generate
from metrics import compute_all_metrics


def run_single_experiment_worker(args):
    """Worker function to run a single experiment (for multiprocessing).
    
    This function is designed to be called by multiprocessing.Pool. It takes
    all necessary arguments and returns results as a dictionary.
    
    Parameters
    ----------
    args : tuple
        (generator_config, sample_size, replication, training_params, seed)
        
    Returns
    -------
    dict
        Dictionary containing all experiment results
    """
    generator_config, sample_size, replication, training_params, seed = args
    
    # Reconstruct generator from config
    generator_type = generator_config['type']
    generator_params = generator_config['params']
    
    generator_map = {
        'NormalGenerator': NormalGenerator,
        'ExponentialGenerator': ExponentialGenerator,
        'UniformGenerator': UniformGenerator,
        'LognormalGenerator': LognormalGenerator,
        'ChiSquareGenerator': ChiSquareGenerator,
    }
    
    generator_class = generator_map[generator_type]
    generator = generator_class(**generator_params)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate original data
    X_original_np = generator.generate(sample_size)
    if X_original_np.ndim == 1:
        X_original_np = X_original_np.reshape(-1, 1)
    X_original = torch.from_numpy(X_original_np).float()
    
    # Train model and generate samples
    start_time = time.time()
    
    model, history, X_generated = train_and_generate(
        X_original,
        num_samples=sample_size,
        verbose=False,  # Disable verbose for parallel runs
        **training_params
    )
    
    training_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_all_metrics(
        X_original, X_generated,
        verbose=False
    )
    
    # Compile results
    result = {
        'generator': generator.name,
        'sample_size': sample_size,
        'replication': replication,
        'training_time': training_time,
        'final_loss': history['loss'][-1],
        'final_term1': history['term1'][-1],
        'final_term2': history['term2'][-1],
        'n_epochs': len(history['loss']),
        'mmd': metrics['mmd'],
        'two_sample_min_p': metrics['two_sample_test']['min_p_value'],
        'two_sample_mean_p': metrics['two_sample_test']['mean_p_value'],
        'two_sample_rejected': metrics['two_sample_test']['num_rejected'],
        'mean_distance': metrics['mean_distance'],
        'cov_frobenius': metrics['cov_frobenius'],
        'cov_trace': metrics['cov_trace'],
    }
    
    return result


class ParallelSimulationExperiment:
    """Parallelized version of SimulationExperiment.
    
    Uses multiprocessing to run experiments in parallel across CPU cores.
    
    Parameters
    ----------
    generators : list of DataGenerator
        List of data generators to test
    sample_sizes : list of int
        List of sample sizes to test
    n_replications : int, default=10
        Number of replications per configuration
    training_params : dict, optional
        Parameters for training (num_epochs, batch_size, etc.)
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    save_results : bool, default=True
        Whether to save results to files
    results_dir : str or Path, default='results'
        Directory to save results
    verbose : bool, default=True
        Whether to print progress
    random_seed : int, default=42
        Base random seed for reproducibility
    """
    
    def __init__(self, generators, sample_sizes, n_replications=10,
                 training_params=None, n_jobs=-1, save_results=True,
                 results_dir='results', verbose=True, random_seed=42):
        self.generators = generators
        self.sample_sizes = sample_sizes
        self.n_replications = n_replications
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Determine number of parallel jobs
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = min(n_jobs, cpu_count())
        
        # Default training parameters
        self.training_params = {
            'num_epochs': 200,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'm': 50,
            'patience': 20,
            'input_dim': 128,
        }
        if training_params is not None:
            self.training_params.update(training_params)
        
        # Create results directories
        if self.save_results:
            self.raw_dir = self.results_dir / 'raw'
            self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def _generator_to_config(self, generator):
        """Convert generator object to serializable config.
        
        Parameters
        ----------
        generator : DataGenerator
            Generator object
            
        Returns
        -------
        dict
            Serializable configuration
        """
        generator_type = type(generator).__name__
        
        # Extract parameters based on generator type
        if generator_type == 'NormalGenerator':
            params = {'loc': generator.loc, 'scale': generator.scale}
        elif generator_type == 'ExponentialGenerator':
            params = {'scale': generator.scale}
        elif generator_type == 'UniformGenerator':
            params = {'low': generator.low, 'high': generator.high}
        elif generator_type == 'LognormalGenerator':
            params = {'mean': generator.mean, 'sigma': generator.sigma}
        elif generator_type == 'ChiSquareGenerator':
            params = {'df': generator.df}
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        return {'type': generator_type, 'params': params}
    
    def run_all_experiments(self):
        """Run all experiment configurations in parallel.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all results
        """
        total_experiments = (len(self.generators) * 
                           len(self.sample_sizes) * 
                           self.n_replications)
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"PARALLEL SIMULATION")
            print(f"{'#'*70}")
            print(f"Total experiments: {total_experiments}")
            print(f"Generators: {len(self.generators)}")
            print(f"Sample sizes: {self.sample_sizes}")
            print(f"Replications: {self.n_replications}")
            print(f"Parallel jobs: {self.n_jobs}")
            print(f"{'#'*70}\n")
        
        # Prepare all experiment configurations
        experiment_args = []
        seed = self.random_seed
        
        for generator in self.generators:
            generator_config = self._generator_to_config(generator)
            
            for sample_size in self.sample_sizes:
                for replication in range(self.n_replications):
                    args = (
                        generator_config,
                        sample_size,
                        replication,
                        self.training_params,
                        seed
                    )
                    experiment_args.append(args)
                    seed += 1  # Different seed for each experiment
        
        # Run experiments in parallel
        start_time = time.time()
        
        if self.verbose:
            print(f"Running {total_experiments} experiments on {self.n_jobs} cores...")
            print("This may take several minutes...\n")
        
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(run_single_experiment_worker, experiment_args)
        
        total_time = time.time() - start_time
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"PARALLEL SIMULATION COMPLETE!")
            print(f"{'#'*70}")
            print(f"Total time: {total_time/60:.2f} minutes")
            print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")
            print(f"Speedup vs sequential: ~{self.n_jobs:.1f}x (theoretical)")
            print(f"{'#'*70}\n")
        
        # Save results
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.raw_dir / f"parallel_simulation_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"Results saved to: {results_file}")
        
        return results_df
    
    def summarize_results(self, results_df=None):
        """Create summary statistics from results.
        
        Parameters
        ----------
        results_df : pd.DataFrame, optional
            Results dataframe. If None, uses stored results.
            
        Returns
        -------
        pd.DataFrame
            Summary statistics grouped by generator and sample size
        """
        if results_df is None:
            raise ValueError("No results available. Run experiments first.")
        
        # Group by generator and sample size
        summary = results_df.groupby(['generator', 'sample_size']).agg({
            'training_time': ['mean', 'std'],
            'final_loss': ['mean', 'std'],
            'n_epochs': ['mean', 'std'],
            'mmd': ['mean', 'std'],
            'two_sample_mean_p': ['mean', 'std'],
            'two_sample_rejected': ['mean', 'std'],
            'mean_distance': ['mean', 'std'],
            'cov_frobenius': ['mean', 'std'],
        }).round(6)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.raw_dir / f"parallel_simulation_summary_{timestamp}.csv"
            summary.to_csv(summary_file, index=False)
            print(f"Summary saved to: {summary_file}")
        
        return summary


def run_parallel_simulation(generators=None, n_replications=10, 
                            sample_sizes=[100, 300, 500, 1000, 2000],
                            n_jobs=-1, verbose=True, save_results=True):
    """Convenience function to run parallel simulation with default settings.
    
    Parameters
    ----------
    generators : list of DataGenerator, optional
        Generators to test. If None, uses all 5 default generators.
    n_replications : int, default=10
        Number of replications per configuration
    sample_sizes : list of int, default=[100, 300, 500, 1000m 2000]
        Sample sizes to test
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    verbose : bool, default=True
        Whether to print progress
    save_results : bool, default=True
        Whether to save results
        
    Returns
    -------
    tuple
        (results_df, summary_df)
    """
    # Use all generators if none specified
    if generators is None:
        generators = [
            NormalGenerator(loc=0, scale=1),
            ExponentialGenerator(scale=1),
            UniformGenerator(low=0, high=2),
            LognormalGenerator(mean=0, sigma=1),
            ChiSquareGenerator(df=5),
        ]
    
    # Run parallel simulation
    sim = ParallelSimulationExperiment(
        generators=generators,
        sample_sizes=sample_sizes,
        n_replications=n_replications,
        n_jobs=n_jobs,
        save_results=save_results,
        verbose=verbose
    )
    
    results_df = sim.run_all_experiments()
    summary_df = sim.summarize_results(results_df)
    
    return results_df, summary_df
