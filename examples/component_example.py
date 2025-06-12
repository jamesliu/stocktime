# examples/component_example.py

"""
Component-Level Example for StockTime Trading System

This example demonstrates how to use individual components:
1. StockTime Predictor for price prediction
2. Trading Strategy for signal generation
3. Portfolio Manager for execution
4. Walk-Forward Evaluator for validation

This gives you more control over the individual components.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add the parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stocktime.core import StockTimePredictor
from stocktime.strategies import StockTimeStrategy
from stocktime.execution import PortfolioManager, SimulatedExecutionEngine
from stocktime.evaluation import WalkForwardEvaluator

def generate_sample_data():
    """Generate realistic sample market data"""
    print("📊 Generating sample market data...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    market_data = {}
    
    # Generate correlated market data
    np.random.seed(42)
    market_returns = np.random.randn(len(dates)) * 0.008 + 0.0003
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        
        # Symbol-specific parameters
        stock_vol = 0.015 + np.random.rand() * 0.010
        market_beta = 0.5 + np.random.rand() * 1.0
        
        # Generate returns with momentum and mean reversion
        idiosyncratic_returns = np.random.randn(len(dates)) * stock_vol
        stock_returns = market_beta * market_returns + idiosyncratic_returns
        
        # Add momentum
        momentum = pd.Series(stock_returns).rolling(10).mean().fillna(0) * 0.1
        stock_returns += momentum
        
        # Generate price series
        initial_price = 50 + np.random.rand() * 100
        prices = initial_price * np.exp(np.cumsum(stock_returns))
        
        market_data[symbol] = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    print(f"✅ Generated data for {len(symbols)} symbols over {len(dates)} days")
    return market_data, symbols

def example_predictor_usage():
    """Demonstrate StockTime Predictor usage"""
    print("\n1️⃣ StockTime Predictor Example")
    print("-" * 40)
    
    # Initialize predictor
    predictor = StockTimePredictor(
        llm_model_name="microsoft/DialoGPT-small",  # Smaller model for demo
        lookback_window=32,
        patch_length=4
    )
    
    # Generate sample price data
    np.random.seed(42)
    price_history = np.cumsum(np.random.randn(64) * 0.01) + 100
    timestamps = [f"2024-01-{i+1:02d}" for i in range(32)]
    
    # Make predictions
    print("🔮 Making price predictions...")
    predictions = predictor.predict_next_prices(
        price_history[-32:], timestamps, prediction_steps=5
    )
    
    current_price = price_history[-1]
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Predicted Prices (next 5 steps): {[f'${p:.2f}' for p in predictions]}")
    print(f"   Predicted Return: {(predictions[0] - current_price) / current_price:.2%}")
    
    return predictor

def example_strategy_usage(market_data, symbols):
    """Demonstrate Trading Strategy usage"""
    print("\n2️⃣ Trading Strategy Example")
    print("-" * 40)
    
    # Initialize strategy
    strategy = StockTimeStrategy(
        symbols=symbols,
        lookback_window=32,
        prediction_horizon=5
    )
    
    # Generate trading signals
    print("📡 Generating trading signals...")
    signals = strategy.generate_signals(market_data)
    
    print(f"   Generated {len(signals)} trading signals:")
    for signal in signals[:3]:  # Show first 3 signals
        print(f"   • {signal.symbol}: {signal.action} at ${signal.current_price:.2f}")
        print(f"     Confidence: {signal.confidence:.2f}, Target: ${signal.target_price:.2f}")
    
    if len(signals) > 3:
        print(f"   ... and {len(signals) - 3} more signals")
    
    return strategy, signals

def example_portfolio_usage(market_data, signals):
    """Demonstrate Portfolio Manager usage"""
    print("\n3️⃣ Portfolio Manager Example")
    print("-" * 40)
    
    # Initialize execution engine and portfolio manager
    execution_engine = SimulatedExecutionEngine(market_data=market_data)
    portfolio_manager = PortfolioManager(
        initial_capital=100000,
        execution_engine=execution_engine,
        max_positions=5
    )
    
    # Process signals
    print("💼 Processing trading signals...")
    
    # Get a recent date for simulation
    recent_date = list(market_data.values())[0].index[-50]
    portfolio_state = portfolio_manager.update_portfolio_state(recent_date)
    
    executed_trades = portfolio_manager.process_signals(signals)
    
    print(f"   Executed {len(executed_trades)} trades")
    print(f"   Portfolio Value: ${portfolio_state.total_value:,.2f}")
    print(f"   Cash Remaining: ${portfolio_state.cash:,.2f}")
    print(f"   Active Positions: {len(portfolio_state.positions)}")
    
    # Show executed trades
    for trade in executed_trades[:2]:  # Show first 2 trades
        print(f"   • {trade.action} {trade.quantity:.1f} shares of {trade.symbol}")
        print(f"     Price: ${trade.price:.2f}, Commission: ${trade.commission:.2f}")
    
    return portfolio_manager

def example_evaluation_usage(market_data, symbols):
    """Demonstrate Walk-Forward Evaluator usage"""
    print("\n4️⃣ Walk-Forward Evaluation Example")
    print("-" * 40)
    
    # Initialize evaluator
    evaluator = WalkForwardEvaluator(
        training_window=252,  # 1 year
        retraining_frequency=63,  # Quarterly
        out_of_sample_window=21  # 1 month
    )
    
    print("📈 Running walk-forward analysis...")
    print("   This may take a few minutes...")
    
    # Run evaluation
    results = evaluator.run_walk_forward_analysis(market_data, symbols)
    
    # Display results
    strategy_passed = (results.strategy_metrics.total_return > 
                      results.benchmark_metrics.total_return)
    
    print(f"\n   VALIDATION RESULT: {'PASSED ✅' if strategy_passed else 'FAILED ❌'}")
    print(f"   Strategy Return: {results.strategy_metrics.total_return:.2%}")
    print(f"   Buy-Hold Return: {results.benchmark_metrics.total_return:.2%}")
    print(f"   Sharpe Ratio: {results.strategy_metrics.sharpe_ratio:.3f}")
    print(f"   Max Drawdown: {results.strategy_metrics.max_drawdown:.2%}")
    print(f"   Signals Generated: {results.signals_generated:,}")
    print(f"   Trades Executed: {results.trades_executed:,}")
    
    return results

def main():
    """Main function demonstrating all components"""
    print("🧩 StockTime Component-Level Example")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Generate sample data
    market_data, symbols = generate_sample_data()
    
    # Demonstrate each component
    predictor = example_predictor_usage()
    strategy, signals = example_strategy_usage(market_data, symbols)
    portfolio_manager = example_portfolio_usage(market_data, signals)
    evaluation_results = example_evaluation_usage(market_data, symbols)
    
    print("\n🎯 Component Example Summary")
    print("-" * 30)
    print("✅ StockTime Predictor: Demonstrated price prediction")
    print("✅ Trading Strategy: Generated trading signals")
    print("✅ Portfolio Manager: Executed trades and managed risk")
    print("✅ Walk-Forward Evaluator: Validated strategy performance")
    
    # Final validation check
    strategy_passed = (evaluation_results.strategy_metrics.total_return > 
                      evaluation_results.benchmark_metrics.total_return)
    
    if strategy_passed:
        print(f"\n🎉 Overall Result: Strategy PASSED validation!")
        print(f"   Excess Return: {(evaluation_results.strategy_metrics.total_return - evaluation_results.benchmark_metrics.total_return):.2%}")
    else:
        print(f"\n⚠️  Overall Result: Strategy FAILED validation")
        print(f"   Underperformance: {(evaluation_results.strategy_metrics.total_return - evaluation_results.benchmark_metrics.total_return):.2%}")
    
    print("\n💡 Next Steps:")
    print("   • Modify component parameters for better performance")
    print("   • Use real market data instead of synthetic data")
    print("   • Implement custom specialist roles")
    print("   • Add additional risk management features")
    
    return strategy_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
