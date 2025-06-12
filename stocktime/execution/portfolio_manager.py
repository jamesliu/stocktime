# stocktime/execution/portfolio_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import json

from stocktime.strategies.stocktime_strategy import StockTimeStrategy, TradingSignal, Position

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    signal_confidence: float = 0.0
    trade_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))

@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_return: float
    cumulative_return: float

class ExecutionEngine(ABC):
    """Abstract execution engine interface"""
    
    @abstractmethod
    def execute_trade(self, signal: TradingSignal, portfolio_value: float) -> Optional[Trade]:
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        pass

class SimulatedExecutionEngine(ExecutionEngine):
    """
    Simulated execution engine for backtesting
    """
    
    def __init__(self, 
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_rate: float = 0.0005,   # 0.05% slippage
                 market_data: Dict[str, pd.DataFrame] = None):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_data = market_data or {}
        self.current_date = None
        
    def set_current_date(self, date: datetime):
        """Set current simulation date"""
        self.current_date = date
    
    def execute_trade(self, signal: TradingSignal, portfolio_value: float) -> Optional[Trade]:
        """
        Execute trade with realistic market impact simulation
        """
        if signal.symbol not in self.market_data:
            logging.warning(f"No market data available for {signal.symbol}")
            return None
        
        # Get current price
        current_price = self.get_current_price(signal.symbol)
        if current_price is None:
            return None
        
        # Calculate position size based on portfolio value and signal
        max_position_value = portfolio_value * 0.1  # Max 10% per position
        quantity = max_position_value / current_price
        
        # Apply slippage
        if signal.action == 'BUY':
            execution_price = current_price * (1 + self.slippage_rate)
        else:
            execution_price = current_price * (1 - self.slippage_rate)
        
        # Calculate commission
        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate
        
        return Trade(
            symbol=signal.symbol,
            action=signal.action,
            quantity=quantity,
            price=execution_price,
            timestamp=self.current_date or datetime.now(),
            commission=commission,
            slippage=abs(execution_price - current_price) / current_price,
            signal_confidence=signal.confidence
        )
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        if symbol not in self.market_data or self.current_date is None:
            return None
        
        symbol_data = self.market_data[symbol]
        
        # Find the most recent price at or before current date
        available_dates = symbol_data.index[symbol_data.index <= self.current_date]
        if len(available_dates) == 0:
            return None
        
        latest_date = available_dates[-1]
        return symbol_data.loc[latest_date, 'close']

class PortfolioManager:
    """
    Comprehensive portfolio management system
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 execution_engine: ExecutionEngine = None,
                 max_positions: int = 10,
                 position_size_limit: float = 0.15):  # Max 15% per position
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.execution_engine = execution_engine or SimulatedExecutionEngine()
        self.max_positions = max_positions
        self.position_size_limit = position_size_limit
        
        # Performance tracking
        self.trade_history: List[Trade] = []
        self.portfolio_history: List[PortfolioState] = []
        self.daily_returns: List[float] = []
        
        # Risk management
        self.daily_loss_limit = 0.02  # 2% daily loss limit
        self.total_loss_limit = 0.20  # 20% total loss limit
        
        # State tracking
        self.current_date = None
        self.trading_enabled = True
        
    def update_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """Update market data for execution engine"""
        if isinstance(self.execution_engine, SimulatedExecutionEngine):
            self.execution_engine.market_data = market_data
    
    def process_signals(self, signals: List[TradingSignal]) -> List[Trade]:
        """
        Process trading signals and execute trades
        """
        if not self.trading_enabled:
            logging.warning("Trading is disabled due to risk limits")
            return []
        
        executed_trades = []
        
        # Sort signals by confidence (highest first)
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
        
        for signal in sorted_signals:
            # Check if we should execute this signal
            if self._should_execute_signal(signal):
                trade = self.execution_engine.execute_trade(
                    signal, self.get_total_portfolio_value()
                )
                
                if trade:
                    success = self._execute_portfolio_trade(trade)
                    if success:
                        executed_trades.append(trade)
                        self.trade_history.append(trade)
                        logging.info(f"Executed trade: {trade.action} {trade.quantity:.2f} "
                                   f"{trade.symbol} at {trade.price:.2f}")
        
        return executed_trades
    
    def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """
        Determine if signal should be executed based on portfolio rules
        """
        # Check confidence threshold
        if signal.confidence < 0.6:
            return False
        
        # Check position limits
        if signal.action == 'BUY':
            if len(self.positions) >= self.max_positions:
                return False
            
            if signal.symbol in self.positions:
                return False  # Already have position
            
            # Check position size limit
            portfolio_value = self.get_total_portfolio_value()
            max_position_value = portfolio_value * self.position_size_limit
            
            current_price = self.execution_engine.get_current_price(signal.symbol)
            if current_price is None:
                return False
            
            required_cash = max_position_value
            if required_cash > self.cash:
                return False
        
        elif signal.action == 'SELL':
            if signal.symbol not in self.positions:
                return False  # No position to sell
        
        return True
    
    def _execute_portfolio_trade(self, trade: Trade) -> bool:
        """
        Execute trade at portfolio level
        """
        try:
            if trade.action == 'BUY':
                # Check if we have enough cash
                total_cost = trade.quantity * trade.price + trade.commission
                if total_cost > self.cash:
                    logging.warning(f"Insufficient cash for trade: {total_cost:.2f} > {self.cash:.2f}")
                    return False
                
                # Create new position
                self.positions[trade.symbol] = Position(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    entry_time=trade.timestamp,
                    position_type='LONG'
                )
                
                # Update cash
                self.cash -= total_cost
                
            elif trade.action == 'SELL':
                if trade.symbol not in self.positions:
                    logging.warning(f"Attempting to sell position that doesn't exist: {trade.symbol}")
                    return False
                
                position = self.positions[trade.symbol]
                
                # Calculate proceeds
                gross_proceeds = trade.quantity * trade.price
                net_proceeds = gross_proceeds - trade.commission
                
                # Update cash
                self.cash += net_proceeds
                
                # Remove position
                del self.positions[trade.symbol]
            
            return True
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            return False
    
    def update_portfolio_state(self, current_date: datetime):
        """
        Update portfolio state with current market prices
        """
        self.current_date = current_date
        
        if isinstance(self.execution_engine, SimulatedExecutionEngine):
            self.execution_engine.set_current_date(current_date)
        
        # Calculate current portfolio value
        total_value = self.cash
        unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            current_price = self.execution_engine.get_current_price(symbol)
            if current_price:
                position_value = position.quantity * current_price
                total_value += position_value
                
                # Calculate unrealized P&L
                position_pnl = (current_price - position.entry_price) * position.quantity
                unrealized_pnl += position_pnl
        
        # Calculate daily return
        if len(self.portfolio_history) > 0:
            previous_value = self.portfolio_history[-1].total_value
            daily_return = (total_value - previous_value) / previous_value
        else:
            daily_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate cumulative return
        cumulative_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate realized P&L from completed trades
        realized_pnl = sum(
            (trade.price - prev_trade.price) * trade.quantity 
            for i, trade in enumerate(self.trade_history)
            for prev_trade in self.trade_history[:i]
            if (trade.symbol == prev_trade.symbol and 
                trade.action == 'SELL' and prev_trade.action == 'BUY')
        )
        
        # Create portfolio state
        portfolio_state = PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            daily_return=daily_return,
            cumulative_return=cumulative_return
        )
        
        self.portfolio_history.append(portfolio_state)
        self.daily_returns.append(daily_return)
        
        # Check risk limits
        self._check_risk_limits(portfolio_state)
        
        return portfolio_state
    
    def _check_risk_limits(self, portfolio_state: PortfolioState):
        """
        Check and enforce risk management limits
        """
        # Daily loss limit
        if portfolio_state.daily_return < -self.daily_loss_limit:
            logging.warning(f"Daily loss limit breached: {portfolio_state.daily_return:.2%}")
            self.trading_enabled = False
        
        # Total loss limit
        if portfolio_state.cumulative_return < -self.total_loss_limit:
            logging.error(f"Total loss limit breached: {portfolio_state.cumulative_return:.2%}")
            self.trading_enabled = False
            # Force liquidation of all positions
            self._emergency_liquidation()
    
    def _emergency_liquidation(self):
        """
        Emergency liquidation of all positions
        """
        liquidation_signals = []
        for symbol in self.positions.keys():
            current_price = self.execution_engine.get_current_price(symbol)
            if current_price:
                signal = TradingSignal(
                    symbol=symbol,
                    action='SELL',
                    confidence=1.0,  # Emergency liquidation
                    predicted_return=0.0,
                    timestamp=self.current_date,
                    current_price=current_price,
                    target_price=current_price
                )
                liquidation_signals.append(signal)
        
        # Force execution regardless of normal rules
        original_enabled = self.trading_enabled
        self.trading_enabled = True
        self.process_signals(liquidation_signals)
        self.trading_enabled = original_enabled
        
        logging.error("Emergency liquidation completed")
    
    def get_total_portfolio_value(self) -> float:
        """Get current total portfolio value"""
        if self.portfolio_history:
            return self.portfolio_history[-1].total_value
        return self.initial_capital
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        """
        if not self.portfolio_history:
            return {}
        
        returns = pd.Series(self.daily_returns)
        
        total_return = self.portfolio_history[-1].cumulative_return
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = len(self.trade_history)
        buy_trades = [t for t in self.trade_history if t.action == 'BUY']
        sell_trades = [t for t in self.trade_history if t.action == 'SELL']
        
        avg_trade_confidence = np.mean([t.signal_confidence for t in self.trade_history]) if self.trade_history else 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'avg_confidence': avg_trade_confidence,
            'current_positions': len(self.positions),
            'cash_remaining': self.cash,
            'total_value': self.get_total_portfolio_value()
        }
    
    def save_state(self, filepath: str):
        """Save portfolio state to file"""
        state = {
            'cash': self.cash,
            'positions': {k: {
                'symbol': v.symbol,
                'quantity': v.quantity,
                'entry_price': v.entry_price,
                'entry_time': v.entry_time.isoformat(),
                'position_type': v.position_type
            } for k, v in self.positions.items()},
            'trade_history': [{
                'symbol': t.symbol,
                'action': t.action,
                'quantity': t.quantity,
                'price': t.price,
                'timestamp': t.timestamp.isoformat(),
                'commission': t.commission,
                'slippage': t.slippage,
                'signal_confidence': t.signal_confidence,
                'trade_id': t.trade_id
            } for t in self.trade_history],
            'performance_summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

def main():
    """
    Test the portfolio management system
    """
    # Create sample market data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    market_data = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
        market_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    # Initialize execution engine and portfolio manager
    execution_engine = SimulatedExecutionEngine(market_data=market_data)
    portfolio_manager = PortfolioManager(
        initial_capital=100000,
        execution_engine=execution_engine
    )
    
    # Initialize strategy
    strategy = StockTimeStrategy(symbols=symbols)
    
    # Simulate trading over time
    for i, date in enumerate(dates[30:]):  # Skip first 30 days for lookback
        # Update portfolio state
        portfolio_state = portfolio_manager.update_portfolio_state(date)
        
        # Generate signals (every 5 days to avoid overtrading)
        if i % 5 == 0:
            current_data = {}
            for symbol in symbols:
                # Get data up to current date
                symbol_data = market_data[symbol]
                mask = symbol_data.index <= date
                current_data[symbol] = symbol_data[mask]
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            
            # Execute trades
            executed_trades = portfolio_manager.process_signals(signals)
            
            if executed_trades:
                logging.info(f"Date: {date.date()}, Executed {len(executed_trades)} trades, "
                           f"Portfolio value: {portfolio_state.total_value:.2f}")
    
    # Final performance summary
    performance = portfolio_manager.get_performance_summary()
    print("\nPortfolio Performance Summary:")
    print("=" * 40)
    for key, value in performance.items():
        if isinstance(value, float):
            if 'return' in key.lower() or 'drawdown' in key.lower():
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save portfolio state
    portfolio_manager.save_state('portfolio_state.json')
    logging.info("Portfolio management system test completed successfully")

if __name__ == "__main__":
    main()
