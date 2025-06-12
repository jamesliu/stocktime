# tests/test_trading_strategy.py

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stocktime.strategies.stocktime_strategy import StockTimeStrategy, TradingSignal

class TestStockTimeStrategy(unittest.TestCase):
    """Test cases for StockTime trading strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = StockTimeStrategy(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            lookback_window=32,
            prediction_horizon=5
        )
        
        # Generate sample market data
        self.dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.market_data = {}
        
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
            self.market_data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=self.dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.symbols, ['AAPL', 'MSFT', 'GOOGL'])
        self.assertEqual(self.strategy.lookback_window, 32)
        self.assertIsNotNone(self.strategy.predictor)
        self.assertIsNotNone(self.strategy.question_architect)
        self.assertIsNotNone(self.strategy.llm_specialist)
        self.assertIsNotNone(self.strategy.risk_specialist)
        self.assertIsNotNone(self.strategy.goal_specialist)
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        signals = self.strategy.generate_signals(self.market_data)
        
        self.assertIsInstance(signals, list)
        
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertIn(signal.symbol, self.strategy.symbols)
            self.assertIn(signal.action, ['BUY', 'SELL'])
            self.assertGreaterEqual(signal.confidence, 0)
            self.assertLessEqual(signal.confidence, 1)
            self.assertIsInstance(signal.timestamp, datetime)
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Test with empty trade history
        sharpe = self.strategy._calculate_current_sharpe()
        self.assertEqual(sharpe, 0.0)
        
        return_val = self.strategy._calculate_current_return()
        self.assertEqual(return_val, 0.0)
        
        drawdown = self.strategy._calculate_current_drawdown()
        self.assertEqual(drawdown, 0.0)

if __name__ == '__main__':
    unittest.main()
