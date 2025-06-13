# stocktime/execution/live_trading_engine.py

import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import yfinance as yf  # For live data
from abc import ABC, abstractmethod

from stocktime.strategies.stocktime_strategy import StockTimeStrategy, TradingSignal
from stocktime.execution.portfolio_manager import PortfolioManager

class LiveDataProvider(ABC):
    """Abstract interface for live market data"""
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, periods: int, interval: str) -> pd.DataFrame:
        pass

class YahooFinanceProvider(LiveDataProvider):
    """Yahoo Finance live data provider"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # Cache for 1 minute
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.history(period="1d", interval="1m")
            if not info.empty:
                return info['Close'].iloc[-1]
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol: str, periods: int, interval: str = "30m") -> pd.DataFrame:
        """Get historical data for StockTime analysis"""
        try:
            # Map intervals
            interval_map = {"30m": "30m", "1h": "1h", "1d": "1d"}
            yf_interval = interval_map.get(interval, "30m")
            
            # Calculate period string for yfinance
            if interval == "30m":
                period_str = f"{periods * 30}m" if periods * 30 < 1440 else "5d"
            elif interval == "1h":
                period_str = f"{periods}h" if periods < 24 else "10d"
            else:
                period_str = f"{periods}d"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period_str, interval=yf_interval)
            
            if not data.empty:
                # Rename columns to match StockTime format
                data = data.rename(columns={
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                return data[['open', 'high', 'low', 'close', 'volume']].tail(periods)
            
        except Exception as e:
            logging.error(f"Error getting historical data for {symbol}: {e}")
        
        return pd.DataFrame()

class BrokerInterface(ABC):
    """Abstract interface for broker integration"""
    
    @abstractmethod
    def place_order(self, symbol: str, action: str, quantity: float) -> bool:
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict:
        pass

class PaperTradingBroker(BrokerInterface):
    """Paper trading implementation for testing"""
    
    def __init__(self, initial_capital: float = 100000):
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
    
    def place_order(self, symbol: str, action: str, quantity: float) -> bool:
        """Simulate order execution"""
        try:
            # Get current price (in real implementation, use broker's price)
            data_provider = YahooFinanceProvider()
            current_price = data_provider.get_current_price(symbol)
            
            if current_price is None:
                return False
            
            if action == 'BUY':
                cost = quantity * current_price
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                    self.trades.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': datetime.now()
                    })
                    logging.info(f"✅ BUY {quantity:.2f} {symbol} at ${current_price:.2f}")
                    return True
                else:
                    logging.warning(f"❌ Insufficient funds for {symbol}")
                    return False
            
            elif action == 'SELL':
                if symbol in self.positions and self.positions[symbol] >= quantity:
                    proceeds = quantity * current_price
                    self.cash += proceeds
                    self.positions[symbol] -= quantity
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    
                    self.trades.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': datetime.now()
                    })
                    logging.info(f"✅ SELL {quantity:.2f} {symbol} at ${current_price:.2f}")
                    return True
                else:
                    logging.warning(f"❌ Insufficient shares to sell {symbol}")
                    return False
            
        except Exception as e:
            logging.error(f"Order execution failed: {e}")
            return False
        
        return False
    
    def get_account_info(self) -> Dict:
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            data_provider = YahooFinanceProvider()
            price = data_provider.get_current_price(symbol)
            if price:
                total_value += quantity * price
        
        return {
            'cash': self.cash,
            'total_value': total_value,
            'positions': self.positions.copy()
        }
    
    def get_positions(self) -> Dict:
        return self.positions.copy()

class LiveTradingBot:
    """
    StockTime Live Trading Bot
    
    Runs continuously, analyzes market data every 30 minutes,
    and executes trades based on StockTime predictions.
    """
    
    def __init__(self, 
                 config_path: str = None,
                 data_provider: LiveDataProvider = None,
                 broker: BrokerInterface = None):
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_provider = data_provider or YahooFinanceProvider()
        self.broker = broker or PaperTradingBroker(self.config.get('initial_capital', 100000))
        
        # Initialize StockTime strategy
        self.strategy = StockTimeStrategy(
            symbols=self.config['symbols'],
            lookback_window=self.config['lookback_window'],
            prediction_horizon=self.config['prediction_horizon']
        )
        
        # Control flags
        self.running = False
        self.trading_enabled = True
        
        # Performance tracking
        self.performance_log = []
        
        logging.info("🤖 StockTime Live Trading Bot initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load trading configuration"""
        import yaml
        
        default_config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'initial_capital': 100000,
            'lookback_window': 32,
            'prediction_horizon': 5,
            'trading_interval': 1800,  # 30 minutes in seconds
            'position_size_limit': 0.1,  # 10% max per position
            'confidence_threshold': 0.6,
            'max_positions': 5,
            'market_hours': {
                'start': '09:30',
                'end': '16:00',
                'timezone': 'US/Eastern'
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        import pytz
        
        # Get current time in market timezone
        tz = pytz.timezone(self.config['market_hours']['timezone'])
        now = datetime.now(tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within market hours
        market_start = now.replace(
            hour=int(self.config['market_hours']['start'].split(':')[0]),
            minute=int(self.config['market_hours']['start'].split(':')[1]),
            second=0, microsecond=0
        )
        market_end = now.replace(
            hour=int(self.config['market_hours']['end'].split(':')[0]),
            minute=int(self.config['market_hours']['end'].split(':')[1]),
            second=0, microsecond=0
        )
        
        return market_start <= now <= market_end
    
    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get current market data for all symbols"""
        market_data = {}
        
        for symbol in self.config['symbols']:
            data = self.data_provider.get_historical_data(
                symbol, 
                self.config['lookback_window'], 
                "30m"
            )
            
            if not data.empty and len(data) >= self.config['lookback_window']:
                market_data[symbol] = data
            else:
                logging.warning(f"Insufficient data for {symbol}")
        
        return market_data
    
    def execute_signals(self, signals: List[TradingSignal]) -> List[bool]:
        """Execute trading signals through broker"""
        results = []
        account_info = self.broker.get_account_info()
        
        for signal in signals:
            # Check if we should execute this signal
            if signal.confidence < self.config['confidence_threshold']:
                logging.info(f"⏭️  Skipping {signal.symbol}: confidence {signal.confidence:.2f} < {self.config['confidence_threshold']}")
                results.append(False)
                continue
            
            # Calculate position size
            max_position_value = account_info['total_value'] * self.config['position_size_limit']
            current_price = self.data_provider.get_current_price(signal.symbol)
            
            if current_price is None:
                results.append(False)
                continue
            
            quantity = max_position_value / current_price
            
            # Execute trade
            success = self.broker.place_order(signal.symbol, signal.action, quantity)
            results.append(success)
            
            if success:
                logging.info(f"📈 Executed {signal.action} {quantity:.2f} {signal.symbol} "
                           f"at ${current_price:.2f} (confidence: {signal.confidence:.2f})")
        
        return results
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        try:
            if not self.is_market_open():
                logging.info("🕒 Market is closed, skipping trading cycle")
                return
            
            if not self.trading_enabled:
                logging.info("⏸️  Trading is disabled")
                return
            
            logging.info("🔄 Starting trading cycle...")
            
            # Get current market data
            market_data = self.get_market_data()
            
            if not market_data:
                logging.warning("❌ No market data available")
                return
            
            # Generate trading signals
            signals = self.strategy.generate_signals(market_data)
            
            if signals:
                logging.info(f"📊 Generated {len(signals)} signals")
                for signal in signals:
                    logging.info(f"   {signal.symbol}: {signal.action} "
                               f"(confidence: {signal.confidence:.2f}, "
                               f"predicted return: {signal.predicted_return:.2%})")
                
                # Execute signals
                results = self.execute_signals(signals)
                executed_count = sum(results)
                logging.info(f"✅ Executed {executed_count}/{len(signals)} trades")
            else:
                logging.info("📊 No trading signals generated")
            
            # Log performance
            account_info = self.broker.get_account_info()
            self.performance_log.append({
                'timestamp': datetime.now(),
                'total_value': account_info['total_value'],
                'cash': account_info['cash'],
                'positions': len(account_info['positions']),
                'signals_generated': len(signals),
                'trades_executed': sum(results) if signals else 0
            })
            
            logging.info(f"💰 Portfolio value: ${account_info['total_value']:.2f}, "
                        f"Cash: ${account_info['cash']:.2f}, "
                        f"Positions: {len(account_info['positions'])}")
            
        except Exception as e:
            logging.error(f"Error in trading cycle: {e}")
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        logging.info("🚀 Starting StockTime Live Trading Bot")
        
        while self.running:
            try:
                self.run_trading_cycle()
                
                # Wait for next cycle
                time.sleep(self.config['trading_interval'])
                
            except KeyboardInterrupt:
                logging.info("⏹️  Received stop signal")
                self.stop()
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logging.info("🛑 StockTime Live Trading Bot stopped")
        
        # Final performance report
        if self.performance_log:
            initial_value = self.performance_log[0]['total_value']
            final_value = self.performance_log[-1]['total_value']
            total_return = (final_value - initial_value) / initial_value
            
            logging.info(f"📈 Final Performance:")
            logging.info(f"   Initial Value: ${initial_value:.2f}")
            logging.info(f"   Final Value: ${final_value:.2f}")
            logging.info(f"   Total Return: {total_return:.2%}")
    
    def enable_trading(self):
        """Enable trading"""
        self.trading_enabled = True
        logging.info("✅ Trading enabled")
    
    def disable_trading(self):
        """Disable trading (monitoring only)"""
        self.trading_enabled = False
        logging.info("⏸️  Trading disabled (monitoring mode)")

def main():
    """Run the live trading bot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockTime Live Trading Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--paper-trading', action='store_true', help='Use paper trading (default)')
    parser.add_argument('--monitoring-only', action='store_true', help='Monitor only, no trading')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stocktime_live_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize bot
    bot = LiveTradingBot(config_path=args.config)
    
    if args.monitoring_only:
        bot.disable_trading()
    
    # Start trading
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()

if __name__ == "__main__":
    main()