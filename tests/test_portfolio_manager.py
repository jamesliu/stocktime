# tests/test_portfolio_manager.py

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stocktime.execution.portfolio_manager import (
    PortfolioManager, SimulatedExecutionEngine, Trade, PortfolioState
)
from stocktime.strategies.stocktime_strategy import TradingSignal

class TestPortfolioManager(unittest.TestCase):
    """Test cases for portfolio management system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample market data
        self.dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.market_data = {}
        
        for symbol in ['AAPL', 'MSFT']:
            np.random.seed(hash(symbol) % 2**32)
            prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
            self.market_data[symbol] = pd.DataFrame({
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=self.dates)
        
        # Initialize components
        self.execution_engine = SimulatedExecutionEngine(market_data=self.market_data)
        self.portfolio_manager = PortfolioManager(
            initial_capital=100000,
            execution_engine=self.execution_engine
        )
    
    def test_portfolio_initialization(self):
        """Test portfolio manager initialization"""
        self.assertEqual(self.portfolio_manager.initial_capital, 100000)
        self.assertEqual(self.portfolio_manager.cash, 100000)
        self.assertEqual(len(self.portfolio_manager.positions), 0)
        self.assertTrue(self.portfolio_manager.trading_enabled)
    
    def test_execution_engine(self):
        """Test execution engine functionality"""
        # Set current date
        current_date = self.dates[50]
        self.execution_engine.set_current_date(current_date)
        
        # Test price retrieval
        price = self.execution_engine.get_current_price('AAPL')
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)
        
        # Test trade execution
        signal = TradingSignal(
            symbol='AAPL',
            action='BUY',
            confidence=0.8,
            predicted_return=0.05,
            timestamp=current_date,
            current_price=price,
            target_price=price * 1.05
        )
        
        trade = self.execution_engine.execute_trade(signal, 100000)
        self.assertIsNotNone(trade)
        self.assertEqual(trade.symbol, 'AAPL')
        self.assertEqual(trade.action, 'BUY')
        self.assertGreater(trade.quantity, 0)
    
    def test_signal_processing(self):
        """Test signal processing and trade execution"""
        # Create test signals
        signals = [
            TradingSignal(
                symbol='AAPL',
                action='BUY',
                confidence=0.8,
                predicted_return=0.05,
                timestamp=self.dates[50],
                current_price=150.0,
                target_price=157.5
            ),
            TradingSignal(
                symbol='MSFT',
                action='BUY',
                confidence=0.7,
                predicted_return=0.03,
                timestamp=self.dates[50],
                current_price=300.0,
                target_price=309.0
            )
        ]
        
        # Update portfolio state
        self.portfolio_manager.update_portfolio_state(self.dates[50])
        
        # Process signals
        executed_trades = self.portfolio_manager.process_signals(signals)
        
        self.assertIsInstance(executed_trades, list)
        # Check that some trades were executed (depends on signal quality)
        
    def test_portfolio_state_update(self):
        """Test portfolio state updates"""
        current_date = self.dates[50]
        portfolio_state = self.portfolio_manager.update_portfolio_state(current_date)
        
        self.assertIsInstance(portfolio_state, PortfolioState)
        self.assertEqual(portfolio_state.cash, self.portfolio_manager.cash)
        self.assertGreaterEqual(portfolio_state.total_value, 0)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Update portfolio state to generate some history
        for i in range(10):
            self.portfolio_manager.update_portfolio_state(self.dates[i + 50])
        
        summary = self.portfolio_manager.get_performance_summary()
        
        self.assertIsInstance(summary, dict)
        expected_keys = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'total_trades', 'current_positions', 'cash_remaining'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)

if __name__ == '__main__':
    unittest.main()
