#!/usr/bin/env python3
"""
StockTime Model Persistence Example

This example demonstrates how to:
1. Train models with automatic saving during walk-forward evaluation
2. Load saved models to verify results
3. Examine training metadata and model summaries
"""

import os
import sys
import pandas as pd
import logging

# Add StockTime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stocktime.evaluation.walk_forward_evaluator import WalkForwardEvaluator
from stocktime.core.stocktime_predictor import StockTimePredictor
from stocktime.strategies.stocktime_strategy import StockTimeStrategy
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_model_persistence():
    """Demonstrate model saving and loading capabilities"""
    
    print("🔬 StockTime Model Persistence Demo")
    print("=" * 50)
    
    # 1. Create evaluator with model saving enabled
    evaluator = WalkForwardEvaluator(
        training_window=252,
        retraining_frequency=63,
        out_of_sample_window=21,
        save_models=True,
        model_save_dir="models/demo_walk_forward"
    )
    
    print(f"✅ Created evaluator with model saving enabled")
    print(f"   Models will be saved to: {evaluator.model_save_dir}")
    
    # 2. Generate some sample data (normally you'd use real market data)
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range('2023-01-01', '2024-06-01', freq='D')
    
    market_data = {}
    for symbol in symbols:
        # Generate realistic price data
        import numpy as np
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.randn(len(dates)) * 0.015 + 0.0003
        prices = 100 * np.exp(np.cumsum(returns))
        
        market_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 50000000, len(dates)),
            'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            'open': prices + np.random.randn(len(dates)) * 0.5
        }, index=dates)
    
    print(f"📊 Generated sample data for {len(symbols)} symbols over {len(dates)} days")
    
    # 3. Run walk-forward evaluation (this will save models automatically)
    print("\n🚀 Running walk-forward evaluation with automatic model saving...")
    result = evaluator.run_walk_forward_analysis(market_data, symbols)
    
    print(f"✅ Evaluation completed!")
    print(f"   Strategy return: {result.strategy_metrics.total_return:.2%}")
    print(f"   Benchmark return: {result.benchmark_metrics.total_return:.2%}")
    
    # 4. Examine saved models
    print(f"\n📁 Examining saved models...")
    models_summary = evaluator.get_saved_models_summary()
    
    if not models_summary.empty:
        print(f"Found {len(models_summary)} saved models:")
        print(models_summary[['model_type', 'period', 'filename', 'file_size_mb', 'created_time']].to_string(index=False))
        
        # 5. Load and verify a saved model
        best_models = models_summary[models_summary['model_type'] == 'best_models']
        if not best_models.empty:
            model_path = best_models.iloc[0]['file_path']
            print(f"\n🔄 Loading best model: {model_path}")
            
            # Create a fresh predictor instance
            predictor = StockTimePredictor(
                llm_model_name="microsoft/DialoGPT-small",
                lookback_window=32
            )
            
            # Load saved state
            loaded_predictor = evaluator.load_saved_model(model_path, predictor)
            print(f"✅ Model loaded successfully!")
            
            # Test the loaded model
            sample_prices = market_data['AAPL']['close'].values[-50:]  # Last 50 prices
            sample_timestamps = [d.strftime('%Y-%m-%d') for d in dates[-32:]]  # Last 32 timestamps
            
            predictions = loaded_predictor.predict_next_prices(sample_prices[-32:], sample_timestamps)
            print(f"📈 Sample predictions from loaded model: {predictions}")
    
    # 6. Examine training metadata
    metadata_dir = f"{evaluator.model_save_dir}/metadata"
    latest_metadata_file = f"{metadata_dir}/latest_training_metadata.json"
    
    if os.path.exists(latest_metadata_file):
        print(f"\n📊 Examining training metadata...")
        with open(latest_metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Evaluation summary:")
        eval_summary = metadata['evaluation_summary']
        print(f"   Timestamp: {eval_summary['timestamp']}")
        print(f"   Total periods: {eval_summary['total_periods_evaluated']}")
        print(f"   Training window: {eval_summary['training_window']}")
        print(f"   OOS window: {eval_summary['out_of_sample_window']}")
        
        print(f"\nOverall performance:")
        perf = metadata['overall_performance']
        print(f"   Strategy return: {perf['strategy_metrics']['total_return']:.2%}")
        print(f"   Benchmark return: {perf['benchmark_metrics']['total_return']:.2%}")
        print(f"   Excess return: {perf['excess_return']:.2%}")
        print(f"   Validation: {'PASSED ✅' if perf['strategy_outperformed'] else 'FAILED ❌'}")
        
        print(f"\nPeriod details available for {len(metadata['period_details'])} training periods")
    
    print(f"\n🎯 Model Persistence Demo Complete!")
    print(f"   All models and metadata saved to: {evaluator.model_save_dir}")
    print(f"   You can now reload any saved model to verify results")

def load_existing_models_example():
    """Example of loading existing models without retraining"""
    
    models_dir = "models/demo_walk_forward"
    if not os.path.exists(models_dir):
        print(f"❌ No existing models found at {models_dir}")
        print(f"   Run the main demo first to create saved models")
        return
    
    print(f"\n🔍 Loading Existing Models Example")
    print("=" * 40)
    
    # Create evaluator to use helper functions
    evaluator = WalkForwardEvaluator(model_save_dir=models_dir)
    
    # Get summary of available models
    models_summary = evaluator.get_saved_models_summary()
    print(f"Available models:")
    print(models_summary.to_string(index=False))
    
    # Load the latest training metadata
    metadata_file = f"{models_dir}/metadata/latest_training_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 Previous Training Results:")
        perf = metadata['overall_performance']
        print(f"   Strategy return: {perf['strategy_metrics']['total_return']:.2%}")
        print(f"   Sharpe ratio: {perf['strategy_metrics']['sharpe_ratio']:.3f}")
        print(f"   Max drawdown: {perf['strategy_metrics']['max_drawdown']:.2%}")
        print(f"   Validation: {'PASSED ✅' if perf['strategy_outperformed'] else 'FAILED ❌'}")

if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_model_persistence()
    
    # Show how to load existing models
    load_existing_models_example()