# examples/quick_start.py

"""
Quick Start Example for StockTime Trading System

This example demonstrates how to:
1. Initialize the StockTime trading system
2. Run walk-forward evaluation 
3. Validate that strategy outperforms buy-and-hold
4. Run live simulation
5. Generate specialist insights

CRITICAL: The strategy must outperform buy-and-hold to be considered viable.
"""

import sys
import os
import logging

# Add the parent directory to path to import stocktime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stocktime import StockTimeTradingSystem

def main():
    """
    Quick start demonstration of StockTime trading system
    """
    print("🚀 StockTime Trading System - Quick Start Example")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the trading system
    print("\n1. Initializing StockTime Trading System...")
    system = StockTimeTradingSystem()
    
    # Load market data (using synthetic data for this example)
    print("\n2. Loading market data...")
    system.load_market_data(start_date='2020-01-01', end_date='2024-01-01')
    print(f"✅ Loaded data for {len(system.config['symbols'])} symbols")
    
    # Initialize all components
    print("\n3. Initializing system components...")
    system.initialize_components()
    print("✅ All components initialized successfully")
    
    # Run walk-forward evaluation
    print("\n4. Running walk-forward evaluation...")
    print("⚠️  CRITICAL: Strategy must outperform buy-and-hold to pass validation")
    
    evaluation_results = system.run_walk_forward_evaluation()
    
    # Check validation result
    strategy_passed = (
        evaluation_results.strategy_metrics.total_return > 
        evaluation_results.benchmark_metrics.total_return
    )
    
    if strategy_passed:
        print("\n✅ STRATEGY VALIDATION PASSED!")
        print(f"   Strategy Return: {evaluation_results.strategy_metrics.total_return:.2%}")
        print(f"   Buy-Hold Return: {evaluation_results.benchmark_metrics.total_return:.2%}")
        print(f"   Excess Return: {(evaluation_results.strategy_metrics.total_return - evaluation_results.benchmark_metrics.total_return):.2%}")
        
        # Run live simulation only if validation passed
        print("\n5. Running live trading simulation...")
        system.run_live_simulation()
        
        # Generate specialist insights
        print("\n6. Generating specialist insights...")
        insights = system.generate_specialist_insights()
        
        print("\n📊 Key Performance Metrics:")
        print(f"   Sharpe Ratio: {evaluation_results.strategy_metrics.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {evaluation_results.strategy_metrics.max_drawdown:.2%}")
        print(f"   Win Rate: {evaluation_results.strategy_metrics.win_rate:.2%}")
        print(f"   Information Ratio: {evaluation_results.strategy_metrics.information_ratio:.3f}")
        
        print("\n🧠 Specialist Insights Summary:")
        for specialist, data in insights.items():
            print(f"   {specialist.replace('_', ' ').title()}:")
            if isinstance(data, dict) and 'recommendations' in data:
                for rec in data['recommendations'][:2]:  # Show first 2 recommendations
                    print(f"     • {rec}")
        
        # Save results
        print("\n7. Saving results...")
        system.save_results(output_dir='results/quick_start')
        print("✅ Results saved to results/quick_start/")
        
        print("\n🎯 QUICK START COMPLETED SUCCESSFULLY!")
        print("   The strategy has been validated and is ready for further analysis.")
        
    else:
        print("\n❌ STRATEGY VALIDATION FAILED!")
        print(f"   Strategy Return: {evaluation_results.strategy_metrics.total_return:.2%}")
        print(f"   Buy-Hold Return: {evaluation_results.benchmark_metrics.total_return:.2%}")
        print(f"   Underperformance: {(evaluation_results.strategy_metrics.total_return - evaluation_results.benchmark_metrics.total_return):.2%}")
        print("\n⚠️  This strategy should NOT be deployed as it underperforms buy-and-hold.")
        print("   Consider:")
        print("   • Adjusting model parameters")
        print("   • Using different symbols")
        print("   • Modifying risk management settings")
        print("   • Extending the training window")
        
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Tip: Try modifying the configuration in config/config.yaml")
        print("   and run the example again.")
    
    sys.exit(0 if success else 1)
