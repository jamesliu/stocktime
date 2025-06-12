# tests/test_stocktime_predictor.py

import unittest
import numpy as np
import torch
import pandas as pd
from stocktime.core.stocktime_predictor import StockTimePredictor

class TestStockTimePredictor(unittest.TestCase):
    """Test cases for StockTime predictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = StockTimePredictor(
            llm_model_name="microsoft/DialoGPT-small",
            lookback_window=32,
            patch_length=4
        )
        
        # Generate sample data
        np.random.seed(42)
        self.sample_prices = np.cumsum(np.random.randn(128) * 0.01) + 100
        self.sample_timestamps = [f"2024-01-{i+1:02d}" for i in range(32)]
    
    def test_initialization(self):
        """Test predictor initialization"""
        self.assertEqual(self.predictor.lookback_window, 32)
        self.assertEqual(self.predictor.patch_length, 4)
        self.assertIsNotNone(self.predictor.llm)
        self.assertIsNotNone(self.predictor.tokenizer)
    
    def test_textual_template_creation(self):
        """Test textual template generation"""
        patches = torch.randn(5, 4)  # 5 patches of length 4
        templates = self.predictor.create_textual_template(patches, self.sample_timestamps[:5])
        
        self.assertEqual(len(templates), 5)
        for template in templates:
            self.assertIn("###Context", template)
            self.assertIn("###Trend analysis", template)
            self.assertIn("###Time stamp", template)
    
    def test_prediction_shape(self):
        """Test prediction output shape"""
        predictions = self.predictor.predict_next_prices(
            self.sample_prices[:32], self.sample_timestamps, prediction_steps=3
        )
        
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_forward_pass(self):
        """Test forward pass through model"""
        price_tensor = torch.FloatTensor(self.sample_prices[:32]).unsqueeze(0)
        
        with torch.no_grad():
            output = self.predictor.forward(price_tensor, self.sample_timestamps)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(len(output.shape), 3)  # batch, patches, patch_length

if __name__ == '__main__':
    unittest.main()
