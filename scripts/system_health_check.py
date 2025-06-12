#!/usr/bin/env python3
# scripts/system_health_check.py

"""
System Health Check Utility

Performs comprehensive health checks on the StockTime trading system.
Verifies all components are working correctly before deployment.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stocktime.core import StockTimePredictor
from stocktime.strategies import StockTimeStrategy
from stocktime.execution import PortfolioManager, SimulatedExecutionEngine
from stocktime.evaluation import WalkForwardEvaluator

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_system_requirements():
    """Check system requirements and dependencies"""
    print("🔧 Checking System Requirements")
    print("-" * 40)
    
    issues = []
    
    # Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available Memory: {memory.available / (1024**3):.1f} GB")
        if memory.available < 2 * (1024**3):  # 2GB
            issues.append("Insufficient memory. Recommend at least 2GB")
    except ImportError:
        print("Memory check skipped (psutil not installed)")
    
    # PyTorch check
    try:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
    except Exception as e:
        issues.append(f"PyTorch issue: {e}")
    
    # Disk space check
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"Free Disk Space: {free_gb:.1f} GB")
        if free_gb < 1.0:
            issues.append("Low disk space. Recommend at least 1GB free")
    except Exception as e:
        print(f"Disk check failed: {e}")
    
    return issues

def test_predictor_component():
    """Test StockTime predictor component"""
    print("\n🧠 Testing StockTime Predictor")
    print("-" * 40)
    
    issues = []
    
    try:
        # Initialize predictor with small model
        start_time = time.time()
        predictor = StockTimePredictor(
            llm_model_name="microsoft/DialoGPT-small",
            lookback_window=16,  # Smaller for testing
            patch_length=4
        )
        init_time = time.time() - start_time
        print(f"✅ Predictor initialized in {init_time:.2f}s")
        
        # Test prediction
        np.random.seed(42)
        test_prices = np.cumsum(np.random.randn(32) * 0.01) + 100
        test_timestamps = [f"2024-01-{i+1:02d}" for i in range(16)]
        
        start_time = time.time()
        predictions = predictor.predict_next_prices(
            test_prices[:16], test_timestamps, prediction_steps=3
        )
        pred_time = time.time() - start_time
        
        print(f"✅ Prediction completed in {pred_time:.2f}s")
        print(f"   Predictions: {[f'{p:.2f}' for p in predictions]}")
        
        # Validate prediction output
        if len(predictions) != 3:
            issues.append("Incorrect prediction output length")
        
        if not all(isinstance(p, (int, float, np.number)) for p in predictions):
            issues.append("Invalid prediction data types")
        
        if any(np.isnan(p) or np.isinf(p) for p in predictions):
            issues.append("NaN or infinite values in predictions")
        
    except Exception as e:
        issues.append(f"Predictor test failed: {e}")
        logging.exception("Predictor test error")
    
    return issues

def test_strategy_component():
    """Test trading strategy component"""
    print("\n📡 Testing Trading Strategy")
    print("-" * 40)
    
    issues = []
    
    try:
        # Create test market data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        symbols = ['AAPL', 'MSFT']
        market_data = {}
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.02))
            market_data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 50)
            }, index=dates)
        
        # Initialize strategy
        strategy = StockTimeStrategy(
            symbols=symbols,
            lookback_window=16,
            prediction_horizon=3
        )
        
        print(f"✅ Strategy initialized with {len(symbols)} symbols")
        
        # Generate signals
        start_time = time.time()
        signals = strategy.generate_signals(market_data)
        signal_time = time.time() - start_time
        
        print(f"✅ Generated {len(signals)} signals in {signal_time:.2f}s")
        
        # Validate signals
        for signal in signals:
            if not hasattr(signal, 'symbol') or signal.symbol not in symbols:
                issues.append("Invalid signal symbol")
            
            if not hasattr(signal, 'action') or signal.action not in ['BUY', 'SELL']:
                issues.append("Invalid signal action")
            
            if not (0 <= signal.confidence <= 1):
                issues.append("Invalid signal confidence")
        
    except Exception as e:
        issues.append(f"Strategy test failed: {e}")
        logging.exception("Strategy test error")
    
    return issues

def test_portfolio_component():
    """Test portfolio management component"""
    print("\n💼 Testing Portfolio Manager")
    print("-" * 40)
    
    issues = []
    
    try:
        # Create test data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        market_data = {
            'AAPL': pd.DataFrame({
                'close': 150 + np.random.randn(30) * 5,
                'volume': np.random.randint(1000000, 5000000, 30)
            }, index=dates)
        }
        
        # Initialize components
        execution_engine = SimulatedExecutionEngine(market_data=market_data)
        portfolio_manager = PortfolioManager(
            initial_capital=100000,
            execution_engine=execution_engine
        )
        
        print(f"✅ Portfolio manager initialized with ${100000:,}")
        
        # Test portfolio state update
        test_date = dates[20]
        portfolio_state = portfolio_manager.update_portfolio_state(test_date)
        
        print(f"✅ Portfolio state updated for {test_date.date()}")
        print(f"   Total Value: ${portfolio_state.total_value:,.2f}")
        print(f"   Cash: ${portfolio_state.cash:,.2f}")
        
        # Validate portfolio state
        if portfolio_state.total_value <= 0:
            issues.append("Invalid portfolio value")
        
        if portfolio_state.cash < 0:
            issues.append("Negative cash balance")
        
        # Test performance summary
        performance = portfolio_manager.get_performance_summary()
        
        if not isinstance(performance, dict):
            issues.append("Invalid performance summary format")
        
        required_metrics = ['total_return', 'sharpe_ratio', 'total_trades']
        for metric in required_metrics:
            if metric not in performance:
                issues.append(f"Missing performance metric: {metric}")
        
    except Exception as e:
        issues.append(f"Portfolio test failed: {e}")
        logging.exception("Portfolio test error")
    
    return issues

def test_evaluation_component():
    """Test walk-forward evaluation component"""
    print("\n📈 Testing Walk-Forward Evaluator")
    print("-" * 40)
    
    issues = []
    
    try:
        # Create minimal test data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        symbols = ['AAPL']
        market_data = {}
        
        np.random.seed(42)
        returns = np.random.randn(len(dates)) * 0.015 + 0.0003
        prices = 100 * np.exp(np.cumsum(returns))
        
        market_data['AAPL'] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Initialize evaluator with minimal settings
        evaluator = WalkForwardEvaluator(
            training_window=60,  # Smaller for testing
            retraining_frequency=30,
            out_of_sample_window=10
        )
        
        print(f"✅ Evaluator initialized")
        
        # Test metric calculation
        test_returns = pd.Series(np.random.randn(50) * 0.01)
        metrics = evaluator.calculate_performance_metrics(test_returns)
        
        print(f"✅ Performance metrics calculated")
        print(f"   Sample Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"   Sample Max Drawdown: {metrics.max_drawdown:.2%}")
        
        # Validate metrics
        if np.isnan(metrics.sharpe_ratio) or np.isinf(metrics.sharpe_ratio):
            issues.append("Invalid Sharpe ratio calculation")
        
        if metrics.max_drawdown > 0:
            issues.append("Max drawdown should be negative")
        
        if not (0 <= metrics.win_rate <= 1):
            issues.append("Invalid win rate")
        
    except Exception as e:
        issues.append(f"Evaluation test failed: {e}")
        logging.exception("Evaluation test error")
    
    return issues

def test_integration():
    """Test system integration"""
    print("\n🔗 Testing System Integration")
    print("-" * 40)
    
    issues = []
    
    try:
        from stocktime.main import StockTimeTradingSystem
        
        # Initialize system with minimal config
        system = StockTimeTradingSystem()
        
        # Override config for testing
        system.config.update({
            'symbols': ['AAPL', 'MSFT'],
            'initial_capital': 50000,
            'lookback_window': 16,
            'max_positions': 2
        })
        
        print(f"✅ Trading system initialized")
        
        # Generate minimal test data
        system.market_data = {}
        dates = pd.date_range('2023-06-01', '2024-01-01', freq='D')
        
        for symbol in system.config['symbols']:
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.randn(len(dates)) * 0.015 + 0.0003
            prices = 100 * np.exp(np.cumsum(returns))
            
            system.market_data[symbol] = pd.DataFrame({
                'close': prices,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
        
        # Initialize components
        system.initialize_components()
        print(f"✅ All components initialized successfully")
        
        # Test component interaction
        sample_data = {k: v.tail(50) for k, v in system.market_data.items()}
        signals = system.strategy.generate_signals(sample_data)
        
        print(f"✅ Generated {len(signals)} signals in integration test")
        
    except Exception as e:
        issues.append(f"Integration test failed: {e}")
        logging.exception("Integration test error")
    
    return issues

def main():
    """Main health check function"""
    parser = argparse.ArgumentParser(description='StockTime System Health Check')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick health check only')
    parser.add_argument('--component', choices=['predictor', 'strategy', 'portfolio', 'evaluation'],
                       help='Test specific component only')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🏥 StockTime System Health Check")
    print("=" * 50)
    
    all_issues = []
    
    # System requirements check
    all_issues.extend(check_system_requirements())
    
    if args.quick:
        print(f"\n⚡ Quick check completed")
    else:
        # Component tests
        if not args.component or args.component == 'predictor':
            all_issues.extend(test_predictor_component())
        
        if not args.component or args.component == 'strategy':
            all_issues.extend(test_strategy_component())
        
        if not args.component or args.component == 'portfolio':
            all_issues.extend(test_portfolio_component())
        
        if not args.component or args.component == 'evaluation':
            all_issues.extend(test_evaluation_component())
        
        if not args.component:
            all_issues.extend(test_integration())
    
    # Results summary
    print(f"\n🎯 Health Check Results")
    print("=" * 30)
    
    if all_issues:
        print(f"❌ Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n🔧 Recommended Actions:")
        print(f"   • Check system requirements")
        print(f"   • Update dependencies: pip install -r requirements.txt")
        print(f"   • Verify configuration: python scripts/validate_config.py")
        print(f"   • Check log files for detailed errors")
        
        return 1
    else:
        print("✅ All health checks passed!")
        print(f"\n🚀 System Status: HEALTHY")
        print(f"   • All components functional")
        print(f"   • Dependencies satisfied")
        print(f"   • Ready for trading operations")
        
        return 0

if __name__ == "__main__":
    exit(main())
