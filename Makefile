# Makefile for Engression Simulation Study

.PHONY: all simulate analyze figures clean test profile complexity benchmark parallel stability-check help

# Run complete simulation pipeline
all: simulate analyze figures
	@echo "=========================================="
	@echo "Complete pipeline finished!"
	@echo "Results in: results/raw/"
	@echo "Figures in: results/figures/"
	@echo "=========================================="

# Run simulations (sequential baseline)
simulate:
	@echo "=========================================="
	@echo "Running simulation (sequential baseline)..."
	@echo "=========================================="
	python src/main.py --mode full --replications 10 --sizes 100 500 1000

# Analyze results
analyze:
	@echo "=========================================="
	@echo "Analyzing results..."
	@echo "=========================================="
	@if [ -d results/raw ]; then \
		python -c "import pandas as pd; import glob; files = glob.glob('results/raw/simulation_results_*.csv'); df = pd.read_csv(max(files)) if files else None; print('\n=== SUMMARY STATISTICS ===\n' + str(df.groupby('generator')[['mmd', 'two_sample_mean_p', 'mean_distance', 'cov_frobenius', 'training_time']].mean()) + '\n\n=== SAMPLE COUNTS ===\n' + str(df.groupby(['generator', 'sample_size']).size())) if df is not None else print('No results found.')"; \
	else \
		echo "No results found. Run make simulate first."; \
	fi

# Generate visualizations
figures:
	@echo "=========================================="
	@echo "Generating figures..."
	@echo "=========================================="
	python src/analyze_results.py

# Clean generated files
clean:
	@echo "=========================================="
	@echo "Cleaning..."
	@echo "=========================================="
	rm -rf results/raw/*.csv
	rm -rf results/figures/*.png
	rm -rf results/profiling/*.png
	rm -rf results/profiling/*.txt
	rm -rf results/profiling/*.csv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Run test suite
test:
	@echo "=========================================="
	@echo "Running tests..."
	@echo "=========================================="
	pytest tests/ -v

# ============================================================
# UNIT 3 TARGETS - PERFORMANCE OPTIMIZATION
# ============================================================

# Run profiling on representative simulation
profile:
	@echo "=========================================="
	@echo "Profiling baseline performance..."
	@echo "=========================================="
	python profile_baseline.py

# Run computational complexity analysis (timing vs n)
complexity:
	@echo "=========================================="
	@echo "Complexity analysis..."
	@echo "=========================================="
	python visualize_baseline.py

# Run timing comparison: baseline vs optimized
benchmark:
	@echo "=========================================="
	@echo "Benchmark: baseline vs optimized..."
	@echo "=========================================="
	python benchmark_parallel.py

# Run optimized version with parallelization
parallel:
	@echo "=========================================="
	@echo "Running OPTIMIZED simulation..."
	@echo "=========================================="
	python run_full_parallel.py

# Check for warnings/convergence issues
stability-check:
	@echo "=========================================="
	@echo "Stability check..."
	@echo "=========================================="
	@python -c "import sys; sys.path.insert(0, 'src'); from simulation import run_quick_simulation; import warnings; warnings.simplefilter('always'); print('Running stability check...'); results, _ = run_quick_simulation(n_replications=3, sample_sizes=[100, 500], verbose=False); print(f'Experiments: {len(results)}'); print(f'All converged: {(results[\"n_epochs\"] > 0).all()}'); print(f'Valid losses: {(results[\"final_loss\"].abs() < 1e6).all()}'); print(f'Valid MMD: {(results[\"mmd\"] >= 0).all() and (results[\"mmd\"] < 1.0).all()}'); print('✅ No issues' if (results['final_loss'].abs() < 1e6).all() else '⚠️ Issues detected')"

# Help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Main:"
	@echo "  make all              - Run complete pipeline"
	@echo "  make simulate         - Sequential baseline"
	@echo "  make analyze          - Analyze results"
	@echo "  make figures          - Generate plots"
	@echo ""
	@echo "Unit 3:"
	@echo "  make profile          - Profile baseline"
	@echo "  make complexity       - Complexity analysis"
	@echo "  make benchmark        - Compare baseline vs optimized"
	@echo "  make parallel         - Run optimized version"
	@echo "  make stability-check  - Check numerical stability"
	@echo ""
	@echo "Other:"
	@echo "  make test             - Run tests"
	@echo "  make clean            - Remove generated files"
