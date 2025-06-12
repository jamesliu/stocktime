"""
StockTime: A Time Series Specialized Large Language Model Trading System

Based on the research paper: "StockTime: A Time Series Specialized Large Language Model 
Architecture for Stock Price Prediction" by Wang et al.

This package implements a complete trading system using:
- StockTime LLM-based stock price prediction
- Specialist role-based trading strategy
- Walk-forward evaluation system
- Portfolio management and execution
"""

__version__ = "1.0.0"
__author__ = "StockTime Team"

from .core.stocktime_predictor import StockTimePredictor
from .strategies.stocktime_strategy import StockTimeStrategy
from .execution.portfolio_manager import PortfolioManager
from .evaluation.walk_forward_evaluator import WalkForwardEvaluator
from .main.trading_system_runner import StockTimeTradingSystem

__all__ = [
    'StockTimePredictor',
    'StockTimeStrategy', 
    'PortfolioManager',
    'WalkForwardEvaluator',
    'StockTimeTradingSystem'
]
