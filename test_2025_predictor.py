#!/usr/bin/env python3
# test_2025_predictor.py

"""
Quick test script for StockTime 2025 LLM integration
Tests the new predictor with optimal small LLM selection
"""

import sys
import os
import logging
import numpy as np

# Add stocktime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stocktime'))

def test_2025_predictor():
    """Test the new StockTime 2025 predictor"""
    
    print("🧪 Testing StockTime 2025 LLM Integration")
    print("=" * 60)
    
    try:
        from stocktime.core.stocktime_predictor_2025 import StockTimePredictor2025
        print("✅ Successfully imported StockTimePredictor2025")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test different deployment modes
    deployment_modes = ["optimal", "efficient", "ultra_light"]
    results = {}
    
    for mode in deployment_modes:
        print(f"\n🔧 Testing {mode.upper()} deployment mode:")
        
        try:
            # Initialize predictor
            predictor = StockTimePredictor2025(
                lookback_window=16,  # Smaller for testing
                patch_length=4,
                deployment_mode=mode
            )
            
            # Get model info
            info = predictor.get_model_info()
            print(f"   📊 Model: {info['model_name']}")
            print(f"   🧠 Parameters: {info['trainable_parameters']:,} trainable")
            print(f"   📖 Context: {info['context_length']:,} tokens")
            
            # Test prediction
            np.random.seed(42)
            sample_prices = np.cumsum(np.random.randn(32) * 0.01) + 100
            sample_timestamps = [f"2024-01-{i+1:02d}" for i in range(16)]
            
            predictions = predictor.predict_next_prices(
                sample_prices[:16], sample_timestamps, prediction_steps=3
            )
            
            print(f"   🔮 Predictions: {[f'${p:.2f}' for p in predictions]}")
            print(f"   ✅ {mode.title()} mode: SUCCESS")
            
            results[mode] = {
                'status': 'SUCCESS',
                'model': info['model_name'],
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"   ❌ {mode.title()} mode: FAILED - {str(e)[:50]}...")
            results[mode] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("-" * 30)
    successful_modes = [mode for mode, result in results.items() if result['status'] == 'SUCCESS']
    failed_modes = [mode for mode, result in results.items() if result['status'] == 'FAILED']
    
    print(f"✅ Successful modes: {len(successful_modes)}/{len(deployment_modes)}")
    if successful_modes:
        print(f"   Working: {', '.join(successful_modes)}")
    
    if failed_modes:
        print(f"❌ Failed modes: {', '.join(failed_modes)}")
        print("💡 Note: Some models may not be available in your environment")
    
    # Recommendation
    if successful_modes:
        recommended = successful_modes[0]
        print(f"\n🎯 RECOMMENDATION")
        print(f"   Use '{recommended}' mode for optimal performance")
        print(f"   Model: {results[recommended]['model']}")
        return True
    else:
        print(f"\n⚠️  No deployment modes working")
        print("   Check your transformers installation and model availability")
        return False

def main():
    """Main test function"""
    # Setup logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    print("🚀 StockTime 2025 LLM Selection Test")
    print("Testing optimal small LLM integration...")
    print()
    
    success = test_2025_predictor()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 StockTime 2025 LLM integration test PASSED!")
        print("🎯 Ready to use optimal small LLMs for trading")
        print()
        print("💡 Next steps:")
        print("   1. Update your config to use 2025 models")
        print("   2. Run: make run-example")
        print("   3. Compare performance with original predictor")
    else:
        print("⚠️  StockTime 2025 LLM integration test had issues")
        print("💡 You can still use the original predictor")
        print("   Check model availability and transformers version")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
