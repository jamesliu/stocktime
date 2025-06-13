# StockTime Live Trading Bot Documentation

## ⚠️ Important Safety Notice

**TRADING WITH REAL MONEY INVOLVES SIGNIFICANT RISK OF LOSS. ALWAYS:**
- Start with paper trading
- Test thoroughly with historical data
- Use small amounts initially  
- Understand the risks involved
- Never invest more than you can afford to lose

## Overview

The StockTime Live Trading Bot runs the StockTime prediction system continuously during market hours, making automated trading decisions every 30 minutes based on AI-powered price predictions.

## Architecture

```
Live Trading Bot Components:
├── LiveTradingBot (Main orchestrator)
├── LiveDataProvider (Market data feeds)
│   ├── YahooFinanceProvider (Free real-time data)
│   └── [Your broker's data feed]
├── BrokerInterface (Trade execution)
│   ├── PaperTradingBroker (Simulated trading)
│   └── [Real broker integration]
└── StockTimeStrategy (AI predictions)
```

## Quick Start

### 1. Paper Trading (Recommended First)

```bash
# Start with paper trading
python run_trading_bot.py --paper-trading

# Monitor only (no trades executed)
python run_trading_bot.py --monitoring-only

# Custom configuration
python run_trading_bot.py --config config/live_trading_config.yaml
```

### 2. Live Trading (After Validation)

```bash
# ⚠️ REAL MONEY - Use only after thorough testing
python run_trading_bot.py --config config/live_trading_config.yaml
```

## Configuration

### Live Trading Configuration (`config/live_trading_config.yaml`)

```yaml
# Trading universe
symbols:
  - 'AAPL'
  - 'MSFT' 
  - 'GOOGL'

# Capital management
initial_capital: 10000        # Start small for live trading
position_size_limit: 0.15     # Max 15% per position
max_positions: 3              # Conservative limit

# StockTime parameters  
lookback_window: 32           # 32 × 30-min bars = 16 hours
prediction_horizon: 5
confidence_threshold: 0.65    # Higher threshold for live trading

# Trading schedule
trading_interval: 1800        # 30 minutes
market_hours:
  start: '09:30'
  end: '16:00'
  timezone: 'US/Eastern'

# Risk management (stricter for live trading)
risk_management:
  daily_loss_limit: 0.015     # 1.5% daily loss limit
  total_loss_limit: 0.10      # 10% total loss limit
```

## How It Works

### Trading Cycle (Every 30 Minutes)

1. **Market Hours Check**
   - Only trades Monday-Friday, 9:30 AM - 4:00 PM ET
   - Automatically pauses outside market hours

2. **Data Collection**
   - Fetches latest 30-minute OHLCV data
   - Collects 32 periods (16 hours) of historical data
   - Validates data quality and completeness

3. **StockTime Analysis**
   ```
   For each symbol:
   ├── LSTM processes 32 × 30-minute price patches
   ├── LLM analyzes textual market patterns
   ├── Predicts next 5 × 30-minute prices
   ├── Calculates confidence score (0-1)
   └── Determines predicted return
   ```

4. **Signal Generation**
   - **BUY**: Predicted return > 0, confidence > 65%, return > volatility
   - **SELL**: Predicted return < 0, confidence > 65%, |return| > volatility  
   - **HOLD**: Low confidence or insufficient signal strength

5. **Risk Validation**
   - Position size limits (max 15% per position)
   - Maximum positions (3 concurrent)
   - Daily loss limits (1.5%)
   - Available capital checks

6. **Trade Execution**
   - Calculates optimal position size
   - Submits orders through broker interface
   - Logs all decisions and executions
   - Updates portfolio state

### Example Trading Scenario

```
📊 StockTime Analysis - AAPL
Time: 2024-01-15 14:30:00 ET
Current Price: $185.50

Historical Data: 32 × 30-minute bars (Jan 14 15:30 → Jan 15 14:30)
LLM Analysis: "minimum $184.20, maximum $186.45, moving average $185.33, change rate +0.68%"

LSTM Predictions (next 5 × 30-min periods):
├── 15:00: $186.20 (+0.38%)
├── 15:30: $186.85 (+0.75%) 
├── 16:00: $186.40 (+0.49%)
├── Next day 09:30: $187.10 (+0.86%)
└── Next day 10:00: $187.50 (+1.08%)

Analysis Results:
├── Predicted Return: +0.38%
├── Confidence: 0.73
├── 30-min Volatility: 0.12%
└── Decision: BUY ✅

Trade Execution:
├── Position Size: $1,500 (15% of $10,000 portfolio)
├── Quantity: 8.1 shares
├── Stop Loss: $180.14 (-2.9%)
└── Take Profit: $191.07 (+3.0%)
```

## Data Providers

### Yahoo Finance (Default - Free)

```python
from stocktime.execution.live_trading_engine import YahooFinanceProvider

provider = YahooFinanceProvider()
```

**Pros:**
- Free and reliable
- Real-time data with ~15 second delay
- No API key required

**Cons:**
- Rate limits for high-frequency requests
- 15-second delay
- Limited to major exchanges

### Custom Data Provider Integration

```python
class MyBrokerDataProvider(LiveDataProvider):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        # Implement your broker's real-time price API
        pass
    
    def get_historical_data(self, symbol: str, periods: int, interval: str) -> pd.DataFrame:
        # Implement your broker's historical data API
        pass
```

## Broker Integration

### Paper Trading (Default - Safe Testing)

```python
from stocktime.execution.live_trading_engine import PaperTradingBroker

broker = PaperTradingBroker(initial_capital=10000)
```

**Features:**
- Simulated order execution
- Realistic slippage and commission modeling
- Portfolio tracking and performance metrics
- Risk-free testing environment

### Real Broker Integration Examples

#### Alpaca Integration
```python
import alpaca_trade_api as tradeapi

class AlpacaBroker(BrokerInterface):
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url)
    
    def place_order(self, symbol: str, action: str, quantity: float) -> bool:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=action.lower(),
                type='market',
                time_in_force='day'
            )
            return order.status == 'filled'
        except Exception as e:
            logging.error(f"Alpaca order failed: {e}")
            return False
```

#### Interactive Brokers Integration
```python
from ib_insync import IB, Stock, MarketOrder

class IBKRBroker(BrokerInterface):
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
    
    def place_order(self, symbol: str, action: str, quantity: float) -> bool:
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        return trade.orderStatus.status == 'Filled'
```

## Monitoring and Logging

### Log Files

The bot generates detailed logs in multiple files:

```
stocktime_bot_20241215.log     # Daily log with all activities
stocktime_live_trading.log     # Persistent trading log
```

### Log Entries

```
2024-01-15 14:30:15 - INFO - 🔄 Starting trading cycle...
2024-01-15 14:30:16 - INFO - 📊 Generated 2 signals
2024-01-15 14:30:16 - INFO -    AAPL: BUY (confidence: 0.73, predicted return: 0.38%)
2024-01-15 14:30:16 - INFO -    MSFT: SELL (confidence: 0.68, predicted return: -0.45%)
2024-01-15 14:30:17 - INFO - ✅ BUY 8.10 AAPL at $185.50
2024-01-15 14:30:17 - INFO - ✅ SELL 5.20 MSFT at $380.20
2024-01-15 14:30:17 - INFO - ✅ Executed 2/2 trades
2024-01-15 14:30:17 - INFO - 💰 Portfolio value: $10,247.50, Cash: $3,180.25, Positions: 2
```

### Performance Tracking

```python
# Access performance data
performance_log = bot.performance_log

# Recent performance
latest = performance_log[-1]
print(f"Portfolio Value: ${latest['total_value']:.2f}")
print(f"Positions: {latest['positions']}")
print(f"Trades Today: {latest['trades_executed']}")
```

## Risk Management

### Automatic Risk Controls

1. **Position Limits**
   - Maximum 15% per position
   - Maximum 3 concurrent positions
   - Prevents over-concentration

2. **Loss Limits**  
   - Daily loss limit: 1.5%
   - Total loss limit: 10%
   - Automatic trading halt when exceeded

3. **Market Hours**
   - Only trades during market hours
   - Automatic pause on weekends/holidays

4. **Confidence Thresholds**
   - Minimum 65% confidence for live trading
   - Higher threshold than backtesting (60%)

### Manual Risk Controls

```python
# Disable trading (monitoring only)
bot.disable_trading()

# Enable trading
bot.enable_trading()

# Emergency stop
bot.stop()
```

## Troubleshooting

### Common Issues

#### 1. "No market data available"
```bash
# Check internet connection and data provider
# Verify symbols are valid
# Check market hours
```

#### 2. "Insufficient funds for trade"
```bash
# Check available cash balance
# Reduce position sizes in config
# Verify commission calculations
```

#### 3. "Trading is disabled due to risk limits"
```bash
# Check daily/total loss limits
# Review recent performance
# Manually re-enable if appropriate
```

#### 4. "Model prediction failed"
```bash
# Check CUDA/GPU availability
# Verify model files are accessible
# Check log for specific errors
```

### Debug Mode

```bash
# Run with debug logging
python run_trading_bot.py --log-level DEBUG
```

### Health Checks

```python
# Check bot status
print(f"Bot running: {bot.running}")
print(f"Trading enabled: {bot.trading_enabled}")
print(f"Market open: {bot.is_market_open()}")

# Check data connectivity  
data = bot.get_market_data()
print(f"Data available for: {list(data.keys())}")

# Check broker connectivity
account = bot.broker.get_account_info()
print(f"Account value: ${account['total_value']:.2f}")
```

## Performance Optimization

### Hardware Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores, GPU (for faster predictions)
- **Storage**: 1GB for logs and model cache

### Configuration Tuning

```yaml
# For faster execution
trading_interval: 900     # 15 minutes instead of 30

# For more conservative trading  
confidence_threshold: 0.75  # Higher confidence required
position_size_limit: 0.10   # Smaller positions

# For more aggressive trading
max_positions: 5           # More concurrent positions
confidence_threshold: 0.60 # Lower confidence required
```

## Security Considerations

### API Key Management

```bash
# Store API keys in environment variables
export BROKER_API_KEY="your_api_key"
export BROKER_SECRET_KEY="your_secret_key"

# Never commit API keys to version control
echo "*.env" >> .gitignore
```

### Network Security

- Use VPN for trading connections
- Enable two-factor authentication on broker accounts
- Monitor for unusual trading activity
- Keep software updated

## Compliance and Legal

- **Algorithmic Trading Registration**: Check if your jurisdiction requires registration
- **Market Maker Rules**: Understand pattern day trader rules
- **Tax Implications**: Keep detailed records for tax reporting
- **Broker Terms**: Ensure algorithmic trading is permitted by your broker

## Support and Community

### Getting Help

1. **Check Logs**: Most issues are documented in log files
2. **GitHub Issues**: Report bugs and request features
3. **Documentation**: Refer to this guide and system documentation
4. **Community**: Join discussions about algorithmic trading

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting issues
- Submitting improvements
- Adding new broker integrations
- Testing and validation

---

**Disclaimer**: This software is for educational and research purposes. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through use of this software.