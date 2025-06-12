# stocktime/strategies/stocktime_strategy.py

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from stocktime.core.stocktime_predictor import StockTimePredictor

@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    predicted_return: float
    timestamp: datetime
    current_price: float
    target_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Position:
    """Trading position representation"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    position_type: str  # 'LONG', 'SHORT'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class SpecialistRole(ABC):
    """Base class for specialist roles in trading system"""
    
    @abstractmethod
    def process(self, data: Dict) -> Dict:
        pass

class QuestionArchitect(SpecialistRole):
    """
    Identify and formulate fundamental questions about market behavior
    """
    
    def __init__(self):
        self.current_questions = [
            "What time horizon shows strongest predictive signal?",
            "Which market regimes affect model performance?",
            "How do correlations between stocks impact predictions?",
            "What risk-reward thresholds optimize performance?"
        ]
    
    def process(self, data: Dict) -> Dict:
        """Generate research questions based on market conditions"""
        market_volatility = data.get('volatility', 0)
        
        if market_volatility > 0.02:  # High volatility regime
            priority_questions = [
                "How should position sizing adapt to volatility?",
                "What correlation patterns emerge during stress?",
                "Should prediction horizons shorten in volatile markets?"
            ]
        else:
            priority_questions = [
                "Can we identify momentum continuation patterns?",
                "What statistical trends persist in calm markets?",
                "How do longer lookback windows perform?"
            ]
        
        return {
            'priority_questions': priority_questions,
            'research_focus': 'volatility_adaptation' if market_volatility > 0.02 else 'trend_following'
        }

class LLMReasoningSpecialist(SpecialistRole):
    """
    Leverage LLM capabilities for market analysis
    """
    
    def __init__(self, predictor: StockTimePredictor):
        self.predictor = predictor
        
    def process(self, data: Dict) -> Dict:
        """Analyze market data using LLM reasoning"""
        price_data = data['price_data']
        timestamps = data['timestamps']
        
        # Generate predictions using StockTime
        predictions = self.predictor.predict_next_prices(
            price_data, timestamps, prediction_steps=5
        )
        
        # Calculate confidence based on prediction consistency
        pred_volatility = np.std(predictions)
        confidence = max(0.1, 1.0 - pred_volatility / np.mean(np.abs(predictions)))
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'predicted_direction': 'UP' if predictions[0] > price_data[-1] else 'DOWN',
            'prediction_strength': abs(predictions[0] - price_data[-1]) / price_data[-1]
        }

class RiskManagementSpecialist(SpecialistRole):
    """
    Assess and manage trading risk
    """
    
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.02):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        
    def process(self, data: Dict) -> Dict:
        """Calculate position sizing and risk parameters"""
        predicted_return = data.get('predicted_return', 0)
        confidence = data.get('confidence', 0.5)
        portfolio_value = data.get('portfolio_value', 100000)
        volatility = data.get('volatility', 0.01)
        
        # Kelly criterion with confidence adjustment
        win_rate = confidence
        avg_win = abs(predicted_return)
        avg_loss = avg_win * 0.5  # Conservative assumption
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
        
        # Volatility-adjusted position sizing
        vol_adjustment = min(1.0, 0.01 / volatility)  # Reduce size in high vol
        position_size = kelly_fraction * vol_adjustment * confidence
        
        # Calculate stop loss and take profit
        stop_loss_pct = min(0.03, volatility * 2)  # 2x volatility or 3% max
        take_profit_pct = max(abs(predicted_return), volatility * 3)
        
        return {
            'position_size': position_size,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'max_risk_per_trade': self.max_drawdown / 10  # Max 10 concurrent trades
        }

class TradingGoalSpecialist(SpecialistRole):
    """
    Define and adjust trading goals dynamically
    """
    
    def __init__(self):
        self.target_sharpe = 2.0
        self.target_annual_return = 0.15
        self.max_drawdown_threshold = 0.05
        
    def process(self, data: Dict) -> Dict:
        """Adjust trading goals based on performance"""
        current_sharpe = data.get('current_sharpe', 0)
        current_return = data.get('current_return', 0)
        current_drawdown = data.get('current_drawdown', 0)
        
        # Dynamic goal adjustment
        if current_sharpe > self.target_sharpe:
            # Performing well, can take more risk
            risk_multiplier = 1.2
        elif current_sharpe < 1.0:
            # Underperforming, reduce risk
            risk_multiplier = 0.8
        else:
            risk_multiplier = 1.0
            
        return {
            'adjusted_risk_multiplier': risk_multiplier,
            'should_trade': current_drawdown < self.max_drawdown_threshold,
            'performance_status': 'GOOD' if current_sharpe > 1.5 else 'POOR'
        }

class StockTimeStrategy:
    """
    Main trading strategy using StockTime predictions with specialist roles
    """
    
    def __init__(self, 
                 symbols: List[str],
                 lookback_window: int = 32,
                 prediction_horizon: int = 5):
        
        self.symbols = symbols
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Initialize StockTime predictor
        self.predictor = StockTimePredictor(
            lookback_window=lookback_window,
            patch_length=4
        )
        
        # Initialize specialists
        self.question_architect = QuestionArchitect()
        self.llm_specialist = LLMReasoningSpecialist(self.predictor)
        self.risk_specialist = RiskManagementSpecialist()
        self.goal_specialist = TradingGoalSpecialist()
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 100000  # Starting capital
        self.trade_history = []
        
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """
        Generate trading signals using specialist collaboration
        """
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
                
            df = market_data[symbol]
            
            if len(df) < self.lookback_window:
                continue
                
            # Prepare data for specialists
            price_data = df['close'].values[-self.lookback_window:]
            timestamps = df.index[-self.lookback_window:].strftime('%Y-%m-%d').tolist()
            volatility = df['close'].pct_change().std()
            
            specialist_data = {
                'symbol': symbol,
                'price_data': price_data,
                'timestamps': timestamps,
                'volatility': volatility,
                'portfolio_value': self.portfolio_value,
                'current_price': price_data[-1]
            }
            
            # Question Architect: Set research focus
            qa_output = self.question_architect.process({'volatility': volatility})
            
            # LLM Specialist: Generate predictions
            llm_output = self.llm_specialist.process(specialist_data)
            
            # Calculate predicted return
            predicted_return = (llm_output['predictions'][0] - price_data[-1]) / price_data[-1]
            
            specialist_data.update({
                'predicted_return': predicted_return,
                'confidence': llm_output['confidence']
            })
            
            # Risk Management: Calculate position sizing
            risk_output = self.risk_specialist.process(specialist_data)
            
            # Trading Goals: Check if we should trade
            goal_output = self.goal_specialist.process({
                'current_sharpe': self._calculate_current_sharpe(),
                'current_return': self._calculate_current_return(),
                'current_drawdown': self._calculate_current_drawdown()
            })
            
            # Generate signal if conditions are met
            if (goal_output['should_trade'] and 
                llm_output['confidence'] > 0.6 and
                abs(predicted_return) > volatility):  # Signal must exceed volatility
                
                action = 'BUY' if predicted_return > 0 else 'SELL'
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=llm_output['confidence'],
                    predicted_return=predicted_return,
                    timestamp=datetime.now(),
                    current_price=price_data[-1],
                    target_price=llm_output['predictions'][0],
                    stop_loss=price_data[-1] * (1 - risk_output['stop_loss_pct']) if action == 'BUY' 
                             else price_data[-1] * (1 + risk_output['stop_loss_pct']),
                    take_profit=price_data[-1] * (1 + risk_output['take_profit_pct']) if action == 'BUY'
                               else price_data[-1] * (1 - risk_output['take_profit_pct'])
                )
                
                signals.append(signal)
        
        return signals
    
    def _calculate_current_sharpe(self) -> float:
        """Calculate current Sharpe ratio"""
        if len(self.trade_history) < 10:
            return 0.0
        
        returns = [trade['return'] for trade in self.trade_history[-20:]]
        if len(returns) == 0:
            return 0.0
        
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_current_return(self) -> float:
        """Calculate current portfolio return"""
        if not self.trade_history:
            return 0.0
        return sum(trade['return'] for trade in self.trade_history)
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.trade_history) < 2:
            return 0.0
        
        cumulative_returns = np.cumsum([trade['return'] for trade in self.trade_history])
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        return abs(np.min(drawdown))

def main():
    """
    Test the StockTime strategy
    """
    # Initialize strategy
    strategy = StockTimeStrategy(symbols=['AAPL', 'MSFT', 'GOOGL'])
    
    # Generate sample market data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = {}
    
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        np.random.seed(hash(symbol) % 2**32)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        sample_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    # Generate signals
    signals = strategy.generate_signals(sample_data)
    
    print(f"Generated {len(signals)} trading signals:")
    for signal in signals:
        print(f"{signal.symbol}: {signal.action} at {signal.current_price:.2f}, "
              f"confidence: {signal.confidence:.2f}, target: {signal.target_price:.2f}")
    
    logging.info(f"StockTime strategy generated {len(signals)} signals")

if __name__ == "__main__":
    main()
