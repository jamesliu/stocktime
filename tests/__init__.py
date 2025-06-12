"""
Test suite for StockTime trading system
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import stocktime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_stocktime_predictor import TestStockTimePredictor
from test_trading_strategy import TestStockTimeStrategy
from test_portfolio_manager import TestPortfolioManager

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestStockTimePredictor))
    suite.addTest(unittest.makeSuite(TestStockTimeStrategy))
    suite.addTest(unittest.makeSuite(TestPortfolioManager))
    
    return suite

def run_tests():
    """Run all tests"""
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
