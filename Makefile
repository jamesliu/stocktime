# Makefile for StockTime Trading System
# Provides convenient commands for development and deployment

.PHONY: help install install-full test clean lint format docs health-check validate-config download-data run-example run-evaluation run-live setup-dev

# Default target
help:
	@echo "🚀 StockTime Trading System - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install package with minimal dependencies"
	@echo "  install-full     Install package with all dependencies"
	@echo "  setup-dev        Setup development environment"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  health-check     Run system health check"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run code linting (flake8)"
	@echo "  format           Format code (black)"
	@echo "  type-check       Run type checking (mypy)"
	@echo ""
	@echo "Configuration Commands:"
	@echo "  validate-config  Validate configuration file"
	@echo "  download-data    Download market data"
	@echo ""
	@echo "Trading Commands:"
	@echo "  run-example      Run quick start example"
	@echo "  run-evaluation   Run walk-forward evaluation only"
	@echo "  run-live         Run live simulation only"
	@echo "  run-full         Run complete trading system"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean            Clean up temporary files"
	@echo "  docs             Generate documentation"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make health-check"
	@echo "  make run-example"
	@echo "  make validate-config"

# Installation and setup
install:
	@echo "📦 Installing StockTime with minimal dependencies..."
	pip install -r requirements-minimal.txt
	pip install -e .
	@echo "✅ Installation completed!"

install-full:
	@echo "📦 Installing StockTime with all dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Full installation completed!"

setup-dev:
	@echo "🛠️ Setting up development environment..."
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "✅ Development environment ready!"

# Testing
test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed!"

test-coverage:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ --cov=stocktime --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/"

health-check:
	@echo "🏥 Running system health check..."
	python scripts/system_health_check.py
	@echo "✅ Health check completed!"

health-check-quick:
	@echo "⚡ Running quick health check..."
	python scripts/system_health_check.py --quick

# Code quality
lint:
	@echo "🔍 Running code linting..."
	flake8 stocktime/ --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code..."
	black stocktime/ examples/ scripts/ tests/
	@echo "✅ Code formatting completed!"

type-check:
	@echo "🔍 Running type checking..."
	mypy stocktime/ --ignore-missing-imports
	@echo "✅ Type checking completed!"

# Configuration and data
validate-config:
	@echo "🔍 Validating configuration..."
	python scripts/validate_config.py --config config/config.yaml
	@echo "✅ Configuration validation completed!"

validate-config-strict:
	@echo "🔍 Validating configuration (strict mode)..."
	python scripts/validate_config.py --config config/config.yaml --strict

download-data:
	@echo "📊 Downloading market data..."
	python scripts/download_market_data.py --symbols AAPL MSFT GOOGL TSLA AMZN --validate
	@echo "✅ Market data downloaded!"

download-sp500:
	@echo "📊 Downloading S&P 500 data..."
	python scripts/download_market_data.py --sp500 --validate
	@echo "✅ S&P 500 data downloaded!"

# Trading system commands
run-example:
	@echo "🚀 Running quick start example..."
	python examples/quick_start.py
	@echo "✅ Example completed!"

run-component-example:
	@echo "🧩 Running component example..."
	python examples/component_example.py
	@echo "✅ Component example completed!"

run-evaluation:
	@echo "📈 Running walk-forward evaluation..."
	python -m stocktime.main.trading_system_runner --mode evaluate --output results/evaluation
	@echo "✅ Evaluation completed!"

run-live:
	@echo "📊 Running live simulation..."
	python -m stocktime.main.trading_system_runner --mode live --output results/live
	@echo "✅ Live simulation completed!"

run-full:
	@echo "🎯 Running complete trading system..."
	python -m stocktime.main.trading_system_runner --mode both --output results/complete
	@echo "✅ Complete system run finished!"

run-custom-config:
	@echo "⚙️ Running with custom configuration..."
	python -m stocktime.main.trading_system_runner --config config/config.yaml --mode both --output results/custom
	@echo "✅ Custom configuration run completed!"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html/
	@echo "📖 Documentation available in docs/_build/html/"

# Cleanup
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "✅ Cleanup completed!"

# Development workflow
dev-setup: setup-dev validate-config health-check
	@echo "🎉 Development environment fully configured!"

# Pre-commit checks
pre-commit: format lint type-check test
	@echo "✅ All pre-commit checks passed!"

# Complete validation
validate-all: validate-config health-check test lint
	@echo "✅ Complete system validation passed!"

# Quick start workflow
quick-start: install validate-config health-check run-example
	@echo "🎉 Quick start completed successfully!"

# Performance testing
performance-test:
	@echo "⚡ Running performance tests..."
	@python -c "import time; from examples.component_example import main; start = time.time(); main(); print(f'Performance test completed in {time.time() - start:.2f}s')"

# System information
system-info:
	@echo "💻 System Information:"
	@echo "Python Version: $$(python --version)"
	@echo "Pip Version: $$(pip --version)"
	@echo "Current Directory: $$(pwd)"
	@echo "Available Memory: $$(python -c 'import psutil; print(f\"{psutil.virtual_memory().available / (1024**3):.1f} GB\")' 2>/dev/null || echo 'N/A')"
	@echo "GPU Available: $$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"

# Installation verification
verify-install:
	@echo "🔍 Verifying installation..."
	python -c "import stocktime; print(f'StockTime version: {stocktime.__version__}')"
	python -c "from stocktime import StockTimePredictor, StockTimeStrategy, PortfolioManager; print('✅ All components imported successfully')"
	@echo "✅ Installation verified!"

# Create directory structure
create-dirs:
	@echo "📁 Creating directory structure..."
	mkdir -p data/market_data
	mkdir -p results/evaluation
	mkdir -p results/live
	mkdir -p results/complete
	mkdir -p logs
	@echo "✅ Directory structure created!"

# Full setup from scratch
setup: create-dirs install validate-config health-check
	@echo "🎉 StockTime setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run: make download-data"
	@echo "  2. Run: make run-example"
	@echo "  3. Customize config/config.yaml as needed"
	@echo "  4. Run: make run-full"

# Help for specific topics
help-trading:
	@echo "📈 Trading Commands Help:"
	@echo "  run-example      - Quick demonstration with synthetic data"
	@echo "  run-evaluation   - Rigorous walk-forward backtesting"
	@echo "  run-live         - Live trading simulation"
	@echo "  run-full         - Complete evaluation + live simulation"
	@echo ""
	@echo "All trading commands validate that strategy outperforms buy-and-hold!"

help-development:
	@echo "🛠️ Development Commands Help:"
	@echo "  setup-dev        - Install development dependencies"
	@echo "  test             - Run unit tests"
	@echo "  lint             - Check code style"
	@echo "  format           - Auto-format code"
	@echo "  pre-commit       - Run all quality checks"

help-config:
	@echo "⚙️ Configuration Help:"
	@echo "  validate-config  - Check configuration file"
	@echo "  download-data    - Get real market data"
	@echo "  health-check     - Verify system health"
	@echo ""
	@echo "Configuration file: config/config.yaml"
	@echo "Modify symbols, capital, risk settings, etc."
