#!/usr/bin/env python3
# run_quick_simulation.py

"""
Quick Live Simulation - Fast testing version
Runs a shorter simulation with more verbose output to see LLM decisions
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add stocktime package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stocktime'))

from stocktime.main.trading_system_runner import StockTimeTradingSystem

def run_quick_simulation(days=10, symbols=None):
    """
    Run a quick simulation for testing LLM decisions
    
    Args:
        days: Number of days to simulate (default: 10)
        symbols: List of symbols to trade (default: ['AAPL', 'MSFT'])
    """
    
    print("⚡ StockTime Quick Live Simulation")
    print("=" * 50)
    print(f"🎯 Simulating {days} trading days")
    print(f"🧠 Focus: Observing LLM trading decisions")
    print()
    
    # Initialize system with quick config
    system = StockTimeTradingSystem()
    
    # Override config for quick testing
    if symbols is None:
        symbols = ['AAPL', 'MSFT']  # Just 2 symbols for speed
    
    system.config.update({
        'symbols': symbols,
        'initial_capital': 50000,
        'lookback_window': 16,  # Shorter for speed
        'max_positions': 2,
        'model_params': {
            'llm_model_name': 'microsoft/DialoGPT-small',  # Fast model
            'patch_length': 4,
            'hidden_dim': 128,  # Smaller for speed
            'num_lstm_layers': 1
        }
    })
    
    print(f"📊 Trading Symbols: {symbols}")
    print(f"💰 Initial Capital: ${system.config['initial_capital']:,}")
    print(f"🧠 LLM Model: {system.config['model_params']['llm_model_name']}")
    print()
    
    # Generate minimal test data
    system.market_data = {}
    import pandas as pd
    import numpy as np
    
    # Generate 60 days of data (50 for history + 10 for simulation)
    dates = pd.date_range('2024-11-01', periods=60, freq='D')
    
    print("📈 Generating synthetic market data...")
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.randn(len(dates)) * 0.02 + 0.001  # 2% daily vol, slight upward bias
        prices = 100 * np.exp(np.cumsum(returns))
        
        system.market_data[symbol] = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
    
    # Initialize components
    print("⚙️ Initializing system components...")
    system.initialize_components()
    print("✅ System ready!")
    print()
    
    # Run quick simulation (last N days)
    start_date = dates[-days-5]  # Start a bit before to have data
    end_date = dates[-1]
    
    print(f"🚀 Starting quick simulation: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # Custom quick simulation with detailed LLM tracking
    simulation_dates = pd.date_range(start_date, end_date, freq='D')
    
    for i, date in enumerate(simulation_dates[:days]):  # Limit to requested days
        print(f"\n📅 DAY {i+1}/{days}: {date.strftime('%Y-%m-%d')} ({'Weekend' if date.weekday() >= 5 else 'Weekday'})")
        print("-" * 40)
        
        # Update portfolio
        portfolio_state = system.portfolio_manager.update_portfolio_state(date)
        print(f"💼 Portfolio Value: ${portfolio_state.total_value:,.2f} | Cash: ${portfolio_state.cash:,.2f}")
        
        # Prepare data
        current_data = {}
        for symbol in symbols:
            symbol_data = system.market_data[symbol]
            mask = symbol_data.index <= date
            if mask.sum() >= system.strategy.lookback_window:
                current_data[symbol] = symbol_data[mask]
                current_price = symbol_data[mask]['close'].iloc[-1]
                print(f"📊 {symbol}: ${current_price:.2f}")
        
        if current_data:
            print("🧠 LLM ANALYSIS:")
            print("   Loading StockTime predictor...")
            
            # Generate signals with detailed tracking
            try:
                import time
                start_time = time.time()
                
                # This is where the LLM makes decisions
                signals = system.strategy.generate_signals(current_data)
                
                prediction_time = time.time() - start_time
                print(f"   ✅ LLM processing completed in {prediction_time:.2f} seconds")
                
                if signals:
                    print(f"   🎯 LLM Generated {len(signals)} trading signals:")
                    for signal in signals:
                        confidence_bar = "█" * int(signal.confidence * 10)
                        print(f"      {signal.action:>4} {signal.symbol}: ${signal.current_price:.2f} → ${signal.target_price:.2f}")
                        print(f"           Confidence: {confidence_bar} {signal.confidence:.2%}")
                        print(f"           Expected Return: {signal.predicted_return:.2%}")
                else:
                    print("   💤 LLM decided: No trading opportunities found")
                
                # Execute trades
                executed_trades = system.portfolio_manager.process_signals(signals)
                
                if executed_trades:
                    print(f"   💰 TRADES EXECUTED:")
                    for trade in executed_trades:
                        print(f"      ✓ {trade.action} {trade.quantity:.1f} shares of {trade.symbol}")
                        print(f"        Price: ${trade.price:.2f} | Confidence: {trade.signal_confidence:.2%}")
                elif signals:
                    print("   ⏸️  No trades executed (risk management blocked)")
                else:
                    print("   😴 No signals = No trades")
                    
            except Exception as e:
                print(f"   ❌ LLM Error: {e}")
        
        # Show positions
        if system.portfolio_manager.positions:
            print("📋 Current Positions:")
            for symbol, position in system.portfolio_manager.positions.items():
                current_price = system.execution_engine.get_current_price(symbol)
                if current_price:
                    pnl = (current_price - position.entry_price) * position.quantity
                    pnl_pct = (current_price - position.entry_price) / position.entry_price
                    print(f"   📈 {symbol}: {position.quantity:.1f} shares @ ${position.entry_price:.2f}")
                    print(f"      Current: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2%})")
        else:
            print("📋 No current positions")
    
    # Final summary
    final_performance = system.portfolio_manager.get_performance_summary()
    
    print(f"\n🎯 QUICK SIMULATION SUMMARY")
    print("=" * 30)
    print(f"📊 Total Return: {final_performance.get('total_return', 0):.2%}")
    print(f"💰 Final Value: ${final_performance.get('total_value', 0):,.2f}")
    print(f"📈 Total Trades: {final_performance.get('total_trades', 0)}")
    print(f"🎲 Win Rate: {final_performance.get('avg_confidence', 0):.2%}")
    
    print(f"\n✅ Quick simulation completed!")
    print("💡 The LLM analyzed market data and made trading decisions each day")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Quick StockTime Live Simulation')
    parser.add_argument('--days', type=int, default=10, help='Number of days to simulate')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT'], 
                       help='Symbols to trade')
    
    args = parser.parse_args()
    
    # Reduce logging noise
    logging.basicConfig(level=logging.WARNING)
    
    try:
        run_quick_simulation(days=args.days, symbols=args.symbols)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
