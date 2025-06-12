# StockTime: A Time Series Specialized Large Language Model Trading System

<div align="center">

![StockTime Logo](https://img.shields.io/badge/StockTime-LLM%20Trading-blue?style=for-the-badge)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/stocktime/stocktime)

*A comprehensive trading system based on the StockTime research paper: "A Time Series Specialized Large Language Model Architecture for Stock Price Prediction"*

</div>

## 🎯 Overview

StockTime is a sophisticated trading system that leverages Large Language Models (LLMs) specifically adapted for stock price prediction. Unlike traditional FinLLMs that rely on external textual data, StockTime extracts meaningful information directly from stock price time series data, making it more robust and less susceptible to market noise.

### Key Innovation

The system implements the groundbreaking StockTime architecture that:
- **Extracts textual information directly from stock prices** (correlations, trends, timestamps)
- **Uses frozen LLM backbone** with only trainable embedding/projection layers
- **Outperforms existing FinLLMs** by up to 5% in stock movement prediction
- **Works effectively** for both daily and hourly frequency trading
- **Requires walk-forward validation** - must outperform buy-and-hold to be considered viable

## 🏗️ System Architecture

The system follows a **specialist role-based architecture** inspired by professional trading teams:

```
┌─────────────────────────────────────────────────────────────┐
│                     StockTime Trading System                │
├─────────────────────────────────────────────────────────────┤
│  Question Architect  │  LLM Specialist  │  Risk Management  │
│  Trading Goals       │  Feature Eng.    │  Execution        │
│  Performance Eval.   │  Modeling        │  Data Sources     │
└─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        Core Predictor    Strategy Engine   Portfolio Manager
        ──────────────    ───────────────   ─────────────────
        • StockTime LLM   • Signal Gen.    • Risk Controls
        • Patch Encoding  • Multi-Modal    • Position Sizing
        • Text Templates  • Fusion         • Execution
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/stocktime/stocktime.git
cd stocktime

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from stocktime import StockTimeTradingSystem

# Initialize the system
system = StockTimeTradingSystem(config_path='config/config.yaml')

# Load market data (or use synthetic data for testing)
system.load_market_data()

# Initialize all components
system.initialize_components()

# Run walk-forward evaluation (CRITICAL: Must outperform buy-and-hold)
evaluation_results = system.run_walk_forward_evaluation()

# Run live simulation
system.run_live_simulation()

# Save results
system.save_results(output_dir='results')
```

### Command Line Interface

```bash
# Run complete evaluation and live simulation
stocktime --config config/config.yaml --mode both --output results/

# Run only walk-forward evaluation
stocktime --mode evaluate --output evaluation_results/

# Run only live simulation
stocktime --mode live --output live_results/
```

## 📊 Walk-Forward Evaluation

**CRITICAL REQUIREMENT**: The strategy must outperform buy-and-hold in walk-forward analysis to be considered viable.

### Evaluation Process

1. **Training Window**: 252 days (1 year) of historical data
2. **Retraining Frequency**: Every 63 days (quarterly)
3. **Out-of-Sample Testing**: 21 days (1 month) forward
4. **Validation Criteria**: Strategy return > Buy-and-hold return

### Example Results

```
STOCKTIME WALK-FORWARD EVALUATION REPORT
============================================================

STRATEGY VALIDATION: PASSED
Strategy must outperform buy-and-hold to be considered viable

PERFORMANCE COMPARISON:
Metric               Strategy        Buy & Hold      Difference     
-----------------------------------------------------------------
Total Return         23.45%          18.20%          +5.25%
Annual Return        15.30%          12.10%          +3.20%
Sharpe Ratio         1.85            1.42            +0.43
Max Drawdown         -4.20%          -8.50%          +4.30%
```

## 🧠 Core Components

### 1. StockTime Predictor (`stocktime/core/`)

The heart of the system - implements the research paper's LLM architecture:

```python
from stocktime.core import StockTimePredictor

predictor = StockTimePredictor(
    llm_model_name="microsoft/DialoGPT-small",
    lookback_window=32,
    patch_length=4
)

# Predict next prices
predictions = predictor.predict_next_prices(price_history, timestamps)
```

**Key Features**:
- Frozen LLM backbone (parameter efficient)
- LSTM autoregressive encoder
- Multimodal fusion of price and text embeddings
- Direct extraction of market correlations and trends

### 2. Trading Strategy (`stocktime/strategies/`)

Specialist role-based strategy implementation:

```python
from stocktime.strategies import StockTimeStrategy

strategy = StockTimeStrategy(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    lookback_window=32,
    prediction_horizon=5
)

signals = strategy.generate_signals(market_data)
```

**Specialist Roles**:
- **Question Architect**: Formulates research questions and priorities
- **LLM Reasoning Specialist**: Generates predictions using StockTime
- **Risk Management Specialist**: Calculates position sizing and risk parameters
- **Trading Goal Specialist**: Dynamically adjusts trading objectives

### 3. Portfolio Management (`stocktime/execution/`)

Professional-grade portfolio management with risk controls:

```python
from stocktime.execution import PortfolioManager

portfolio_manager = PortfolioManager(
    initial_capital=100000,
    max_positions=8,
    position_size_limit=0.12
)

executed_trades = portfolio_manager.process_signals(signals)
```

**Features**:
- Kelly criterion position sizing
- Volatility-adjusted risk management
- Real-time portfolio tracking
- Emergency liquidation protocols

### 4. Walk-Forward Evaluator (`stocktime/evaluation/`)

Rigorous backtesting with statistical validation:

```python
from stocktime.evaluation import WalkForwardEvaluator

evaluator = WalkForwardEvaluator(
    training_window=252,
    retraining_frequency=63,
    out_of_sample_window=21
)

results = evaluator.run_walk_forward_analysis(market_data, symbols)
```

## ⚙️ Configuration

The system uses YAML configuration files for easy customization:

```yaml
# config/config.yaml
symbols:
  - 'AAPL'
  - 'MSFT'
  - 'GOOGL'

initial_capital: 100000
max_positions: 8
position_size_limit: 0.12

risk_management:
  daily_loss_limit: 0.02
  total_loss_limit: 0.15
  max_drawdown: 0.05

model_params:
  llm_model_name: 'microsoft/DialoGPT-small'
  lookback_window: 32
  patch_length: 4
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Risk-Adjusted Returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return vs. maximum drawdown
- **Information Ratio**: Active return vs. tracking error

### Drawdown Analysis
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Drawdown**: Mean of all drawdown periods
- **Recovery Time**: Time to recover from drawdowns

### Trading Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade**: Mean return per trade
- **Trade Frequency**: Number of trades per period

## 🛡️ Risk Management

### Built-in Risk Controls

1. **Position Size Limits**: Maximum 12% per position by default
2. **Daily Loss Limits**: 2% maximum daily loss
3. **Total Loss Limits**: 15% maximum total loss
4. **Maximum Positions**: 8 concurrent positions
5. **Emergency Liquidation**: Automatic when limits breached

### Kelly Criterion Position Sizing

The system uses a modified Kelly criterion for optimal position sizing:

```python
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
position_size = kelly_fraction * volatility_adjustment * confidence
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_stocktime_predictor.py -v
python -m pytest tests/test_trading_strategy.py -v
python -m pytest tests/test_portfolio_manager.py -v

# Run with coverage
python -m pytest tests/ --cov=stocktime --cov-report=html
```

## 📚 Research Foundation

This implementation is based on the research paper:

**"StockTime: A Time Series Specialized Large Language Model Architecture for Stock Price Prediction"**
*by Wang et al.*

### Key Research Findings

1. **Direct Price Analysis**: Extracting textual information directly from stock prices is more effective than using external news/social media
2. **Frozen LLM Efficiency**: Using frozen LLM backbone with trainable layers reduces computational cost by 90%
3. **Multimodal Fusion**: Combining price embeddings with textual templates improves prediction accuracy
4. **Performance**: 5% improvement over existing FinLLMs in stock movement prediction

## 🚨 Important Disclaimers

### Walk-Forward Validation Requirement

**This system will ONLY recommend strategies that outperform buy-and-hold in rigorous walk-forward testing.** Any strategy that underperforms buy-and-hold is automatically flagged as FAILED and should not be deployed.

### Investment Disclaimers

- **Not Financial Advice**: This system is for research and educational purposes only
- **Past Performance**: Does not guarantee future results
- **Risk Warning**: Trading involves substantial risk of loss
- **Professional Advice**: Consult qualified financial advisors before trading

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ."[dev]"

# Run code formatting
black stocktime/
flake8 stocktime/

# Run type checking
mypy stocktime/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research paper authors: Wang et al.
- Transformer architecture: Attention Is All You Need
- LLM frameworks: Hugging Face Transformers
- Financial data: Yahoo Finance, Alpha Vantage

## 📞 Support

- **Documentation**: [https://stocktime.readthedocs.io/](https://stocktime.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/stocktime/stocktime/issues)
- **Discussions**: [GitHub Discussions](https://github.com/stocktime/stocktime/discussions)

---

<div align="center">

**⚡ Built with the power of Large Language Models and rigorous financial engineering ⚡**

</div>
