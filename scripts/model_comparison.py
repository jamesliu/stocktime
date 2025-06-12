# scripts/model_comparison.py

"""
Compare different LLM models for StockTime
"""

import torch
import time
import numpy as np
from stocktime.core.stocktime_predictor import StockTimePredictor

def compare_models():
    """Compare model performance and resource usage"""
    
    models_to_test = [
        ("microsoft/DialoGPT-small", "DialoGPT Small"),
        # Add Llama only if you have enough VRAM and authentication
        # ("meta-llama/Llama-2-7b-hf", "Llama-2-7B"),
    ]
    
    # Test data
    np.random.seed(42)
    price_data = np.cumsum(np.random.randn(32) * 0.01) + 100
    timestamps = [f"2024-01-{i+1:02d}" for i in range(16)]
    
    results = []
    
    for model_name, display_name in models_to_test:
        print(f"\n🧪 Testing {display_name}...")
        
        try:
            # Initialize predictor
            start_time = time.time()
            predictor = StockTimePredictor(
                llm_model_name=model_name,
                lookback_window=16  # Smaller for testing
            )
            init_time = time.time() - start_time
            
            # Check memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            
            # Make prediction
            pred_start = time.time()
            predictions = predictor.predict_next_prices(
                price_data[:16], timestamps, prediction_steps=3
            )
            pred_time = time.time() - pred_start
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024**3  # GB
            else:
                memory_used = "N/A (CPU)"
            
            results.append({
                'model': display_name,
                'init_time': init_time,
                'pred_time': pred_time,
                'memory_gb': memory_used,
                'predictions': predictions,
                'success': True
            })
            
            print(f"✅ {display_name} - Init: {init_time:.2f}s, Pred: {pred_time:.2f}s")
            print(f"   Memory: {memory_used}, Predictions: {[f'{p:.2f}' for p in predictions]}")
            
        except Exception as e:
            print(f"❌ {display_name} failed: {e}")
            results.append({
                'model': display_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    for result in results:
        if result['success']:
            print(f"✅ {result['model']}")
            print(f"   Init Time: {result['init_time']:.2f}s")
            print(f"   Prediction Time: {result['pred_time']:.2f}s") 
            print(f"   Memory Usage: {result['memory_gb']}")
        else:
            print(f"❌ {result['model']}: {result['error']}")

if __name__ == "__main__":
    compare_models()
