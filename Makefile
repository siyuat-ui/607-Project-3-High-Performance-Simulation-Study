# Makefile for Engression Simulation Study

.PHONY: all simulate analyze figures clean test profile complexity benchmark parallel stability-check

# Run complete simulation pipeline and generate all outputs
all: simulate analyze figures
	@echo "=========================================="
	@echo "Complete pipeline finished!"
	@echo "Results in: results/raw/"
	@echo "Figures in: results/figures/"
	@echo "=========================================="

# Run simulations and save raw results
simulate:
	@echo "=========================================="
	@echo "Running simulation..."
	@echo "=========================================="
	python src/main.py --mode full --replications 10 --sizes 100 500 1000
	@echo "Simulation complete!"

# Process raw results and generate summary statistics
analyze:
	@echo "=========================================="
	@echo "Analyzing results..."
	@echo "=========================================="
	@if [ -d results/raw ]; then \
		python -c "import pandas as pd; import glob; files = glob.glob('results/raw/simulation_results_*.csv'); df = pd.read_csv(max(files)) if files else None; print('\n=== SUMMARY STATISTICS ===\n' + str(df.groupby('generator')[['mmd', 'two_sample_mean_p', 'mean_distance', 'cov_frobenius', 'training_time']].mean()) + '\n\n=== SAMPLE COUNTS ===\n' + str(df.groupby(['generator', 'sample_size']).size())) if df is not None else print('No results found. Run make simulate first.')"; \
	else \
		echo "No results directory found. Run make simulate first."; \
	fi

# Create all visualizations
figures:
	@echo "=========================================="
	@echo "Generating analysis figures..."
	@echo "=========================================="
	python src/analyze_results.py

# Remove generated files
clean:
	@echo "=========================================="
	@echo "Cleaning generated files..."
	@echo "=========================================="
	rm -rf results/raw/*.csv
	rm -rf results/figures/*.png
	rm -rf results/profiling/*.png
	rm -rf results/profiling/*.txt
	rm -rf results/profiling/*.csv
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Run test suite
test:
	@echo "=========================================="
	@echo "Running test suite..."
	@echo "=========================================="
	pytest tests/ -v

# ============================================================
# NEW TARGETS FOR UNIT 3 - PERFORMANCE OPTIMIZATION
# ============================================================

# Run profiling on representative simulation
profile:
	@echo "=========================================="
	@echo "Running baseline profiling..."
	@echo "=========================================="
	python profile_baseline.py
	@echo ""
	@echo "Profiling complete! Results in results/profiling/"
	@echo "  - baseline_profile.txt (detailed function profiling)"
	@echo "  - component_timings.csv (component breakdown)"
	@echo "  - timing_summary.csv (summary statistics)"

# Run computational complexity analysis
complexity:
	@echo "=========================================="
	@echo "Running complexity analysis..."
	@echo "=========================================="
	python visualize_baseline.py
	@echo ""
	@echo "Complexity analysis complete! Check results/profiling/"
	@echo "  - baseline_complexity_analysis.png"

# Run benchmark: baseline vs optimized
benchmark:
	@echo "=========================================="
	@echo "Benchmarking sequential vs parallel..."
	@echo "=========================================="
	python benchmark_parallel.py
	@echo ""
	@echo "Benchmark complete! Check results/profiling/"
	@echo "  - parallel_speedup.png"

# Run optimized version with parallelization
parallel:
	@echo "=========================================="
	@echo "Running PARALLEL simulation (optimized)..."
	@echo "=========================================="
	@echo "Full configuration: 150 experiments"
	@echo "This will take 1.5-3 minutes..."
	@echo ""
	python run_full_parallel.py

# Compare sequential vs parallel (full simulation)
compare-full:
	@echo "=========================================="
	@echo "Comparing sequential vs parallel..."
	@echo "=========================================="
	python compare_full_simulation.py

# Check for numerical warnings/convergence issues
stability-check:
	@echo "=========================================="
	@echo "Running numerical stability check..."
	@echo "=========================================="
	@echo "Checking for:"
	@echo "  - NaN/Inf values"
	@echo "  - Gradient issues"
	@echo "  - Convergence failures"
	@echo ""
	python -c "import sys; sys.path.insert(0, 'src'); from simulation import run_quick_simulation; import warnings; warnings.simplefilter('always'); results, summary = run_quick_simulation(n_replications=3, sample_sizes=[100, 500], verbose=False); print('Stability check complete!'); print(f'Experiments run: {len(results)}'); print(f'All converged: {(results[\"n_epochs\"] > 0).all()}'); print(f'No extreme losses: {(results[\"final_loss\"].abs() < 1e6).all()}'); print(f'Valid MMD values: {(results[\"mmd\"] >= 0).all() and (results[\"mmd\"] < 1.0).all()}')"

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Main pipeline:"
	@echo "  make all              - Run complete pipeline"
	@echo "  make simulate         - Run simulation only (sequential)"
	@echo "  make analyze          - Analyze results"
	@echo "  make figures          - Generate visualizations"
	@echo ""
	@echo "Development:"
	@echo "  make test             - Run test suite"
	@echo "  make clean            - Remove generated files"
	@echo ""
	@echo "Performance (Unit 3):"
	@echo "  make profile          - Profile baseline performance"
	@echo "  make complexity       - Analyze computational complexity"
	@echo "  make parallel         - Run full simulation with parallelization"
	@echo "  make compare-full     - Compare sequential vs parallel (full)"
	@echo "  make stability-check  - Check numerical stability"
