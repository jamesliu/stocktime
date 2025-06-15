# 🎯 StockTime Training Guide

This guide shows you how to train StockTime from scratch with different data sources and configurations.

## 🚀 Quick Training Start

### Method 1: Simple Training with Synthetic Data (Fastest)

```bash
# Basic training with synthetic data
python run_stocktime.py --mode evaluate --output results/training_test

# With custom config
python run_stocktime.py --config config/config.yaml --mode evaluate
```

### Method 2: Real Data Training (Recommended)

```bash
# Train with real 30m data (if you have Schwab data)
python stocktime/main/real_data_trading_system.py \
    --data-dir ~/data/finance/schwab/ohlcv/30m/ \
    --symbols AAPL MSFT GOOGL TSLA \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Train with real daily data
python stocktime/main/real_data_trading_system.py \
    --data-dir ~/data/finance/daily/ \
    --symbols AAPL MSFT GOOGL \
    --start-date 2023-01-01 \
    --end-date 2024-06-01
```

### Method 3: Advanced Training with Model Persistence

```bash
# Train with automatic model saving
python examples/model_persistence_example.py
```

## 📊 Training Methods Explained

### 1. Synthetic Data Training

**Best for**: Learning the system, testing configurations, development

```python
from stocktime.main.trading_system_runner import StockTimeTradingSystem

# Initialize system
system = StockTimeTradingSystem()

# Load synthetic data (automatically generated)
system.load_market_data(
    start_date='2023-01-01',
    end_date='2024-06-01'
)

# Initialize components
system.initialize_components()

# Run walk-forward evaluation (this trains the models)
results = system.run_walk_forward_evaluation()
```

**What happens during training:**
- Generates realistic synthetic stock data
- Runs walk-forward analysis with automatic retraining
- Models train for 10 epochs per period
- Must outperform buy-and-hold to pass validation

### 2. Real Data Training

**Best for**: Actual trading preparation, real-world validation

```python
from stocktime.main.real_data_trading_system import RealDataStockTimeTradingSystem

# Initialize system with real data support
system = RealDataStockTimeTradingSystem()

# Setup data loader
system.setup_real_data_loader('~/data/finance/schwab/ohlcv/30m/')

# Load real market data
system.load_real_market_data(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Initialize and train
system.initialize_components()
results = system.run_real_data_evaluation()
```

**Training with different timeframes:**
- **30m data**: 390 periods training (30 days), 65 periods test (1 week)
- **1h data**: 195 periods training (30 days), 32 periods test (1 week)  
- **1d data**: 252 periods training (1 year), 21 periods test (1 month)

### 3. Model Persistence Training

**Best for**: Serious development, result verification, model analysis

```python
from stocktime.evaluation.walk_forward_evaluator import WalkForwardEvaluator

# Enable model saving during training
evaluator = WalkForwardEvaluator(
    training_window=390,  # For 30m data
    out_of_sample_window=65,
    save_models=True,
    model_save_dir="models/my_training"
)

# Training automatically saves:
# - Best models (when loss improves by 5%+)
# - Checkpoints (epochs 0, 5, 9)
# - Final models (end of each period)
# - Complete metadata
```

## 🏗️ Training Architecture

### Walk-Forward Training Process

```
Period 1: Train[Jan-Mar] → Test[Apr] → Retrain
Period 2: Train[Feb-Apr] → Test[May] → Retrain
Period 3: Train[Mar-May] → Test[Jun] → Retrain
...and so on
```

### What Happens During Each Training Period

1. **Data Preparation**
   - Load historical price data for training window
   - Create price patches (default: 4-period patches)
   - Generate timestamps and correlations

2. **Model Training** (10 epochs)
   - LSTM autoregressive encoder processes price patches
   - LLM generates textual templates from price statistics
   - Multimodal fusion combines price + text embeddings
   - Backpropagation only updates LSTM/projection layers (LLM frozen)

3. **Model Saving** (if enabled)
   - Save when loss improves by 5%+
   - Checkpoint at epochs 0, 5, 9
   - Final model at end

4. **Out-of-Sample Testing**
   - Generate trading signals using trained model
   - Execute simulated trades
   - Compare against buy-and-hold benchmark

5. **Validation**
   - Strategy MUST outperform buy-and-hold to pass
   - Calculate comprehensive performance metrics

## 📈 Example Training Session

Let's run a complete training session:

```bash
# Step 1: Train with 30m real data
python stocktime/main/real_data_trading_system.py \
    --data-dir ~/data/finance/schwab/ohlcv/30m/ \
    --symbols AAPL MSFT GOOGL TSLA AMZN \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --output results/training_30m
```

**Expected Output:**
```
📊 StockTime Real Data Trading System
============================================================

📊 Real Market Data Summary
==================================================
Symbol  Records  Start       End         Avg Price  Volatility  Return
AAPL    8640     2024-01-01  2024-12-31  $185.50    0.025      23.4%
MSFT    8640     2024-01-01  2024-12-31  $415.20    0.023      18.7%
...

🔬 Starting real data walk-forward evaluation...
   Data source: Real market data (real)
   Symbols: 5
   Training window: 390
   OOS window: 65

📈 Walk-forward period 1/15: Training...
💾 Saved improved model: 0.002451 -> models/walk_forward/best_models/period_001_epoch_03_loss_0.00.pth
📁 Checkpoint saved: epoch 0 -> models/walk_forward/checkpoints/period_001_epoch_00.pth
📁 Checkpoint saved: epoch 5 -> models/walk_forward/checkpoints/period_001_epoch_05.pth
📁 Checkpoint saved: epoch 9 -> models/walk_forward/checkpoints/period_001_epoch_09.pth
🎯 Final model saved: models/walk_forward/final/period_001_final.pth

Testing period: 2024-04-15 to 2024-05-10
Period strategy return: 0.0234, benchmark return: 0.0156

...continues for all periods...

============================================================
STOCKTIME WALK-FORWARD EVALUATION REPORT
============================================================

STRATEGY VALIDATION: PASSED ✅
Strategy must outperform buy-and-hold to be considered viable

PERFORMANCE COMPARISON:
Metric               Strategy        Buy & Hold      Difference     
-----------------------------------------------------------------
Total Return         45.08%          9.76%           +35.32%
Annual Return        38.24%          8.92%           +29.32%
Sharpe Ratio         2.15            0.87            +1.28
Max Drawdown         -3.20%          -12.50%         +9.30%
```

## 🛠️ Training Configuration

### Basic Configuration (config/config.yaml)

```yaml
# Symbols to trade
symbols:
  - 'AAPL'
  - 'MSFT' 
  - 'GOOGL'
  - 'TSLA'
  - 'AMZN'

# Training parameters
lookback_window: 32        # Input sequence length
prediction_horizon: 5      # Forecast horizon
retraining_frequency: 63   # Retrain every 63 periods

# Model parameters
model_params:
  llm_model_name: 'microsoft/DialoGPT-small'
  patch_length: 4          # Price patch size
  hidden_dim: 256          # LSTM hidden size
  num_lstm_layers: 2       # LSTM depth

# Risk management
risk_management:
  daily_loss_limit: 0.02   # 2% daily loss limit
  total_loss_limit: 0.15   # 15% total loss limit
  max_drawdown: 0.05       # 5% max drawdown
```

### Advanced Model Configuration

```yaml
# Enhanced 2025 model settings
model_params:
  llm_model_name: 'Qwen/Qwen2.5-0.5B-Instruct'        # Primary model
  llm_model_fallback: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'  # Fallback
  llm_model_ultra_light: 'HuggingFaceTB/SmolLM2-360M-Instruct'  # Ultra-light
  deployment_mode: 'optimal'    # adaptive, lightweight, optimal
  max_context_length: 8192      # Extended context
```

### Timeframe-Specific Training

**30-minute data:**
```yaml
timeframe: '30m'
training_window: 390    # 30 days × 13 periods/day
out_of_sample_window: 65  # 5 days × 13 periods/day
```

**Hourly data:**
```yaml
timeframe: '1h'
training_window: 195    # 30 days × 6.5 periods/day
out_of_sample_window: 32  # 5 days × 6.5 periods/day
```

**Daily data:**
```yaml
timeframe: '1d'
training_window: 252    # ~1 year
out_of_sample_window: 21  # ~1 month
```

## 🔄 Loading Trained Models

After training, you can reload models to verify results:

```python
from stocktime.evaluation.walk_forward_evaluator import WalkForwardEvaluator
from stocktime.core.stocktime_predictor import StockTimePredictor

# Load evaluator
evaluator = WalkForwardEvaluator(model_save_dir="models/walk_forward")

# See available models
models_summary = evaluator.get_saved_models_summary()
print(models_summary)

# Load a specific model
predictor = StockTimePredictor()
best_model_path = "models/walk_forward/best_models/period_001_epoch_03_loss_0.00.pth"
loaded_predictor = evaluator.load_saved_model(best_model_path, predictor)

# Use loaded model for predictions
sample_prices = [100, 101, 102, 103, ...]  # Your price data
predictions = loaded_predictor.predict_next_prices(sample_prices[-32:], timestamps)
```

## 📊 Training Progress Monitoring

### Log Files

Training creates detailed logs:
- `stocktime_trading.log` - General system logs
- `stocktime_real_data.log` - Real data training logs

### Key Log Messages to Watch

```bash
# Model saving
💾 Saved improved model: 0.002451 -> models/walk_forward/best_models/...
📁 Checkpoint saved: epoch 5 -> models/walk_forward/checkpoints/...

# Training progress  
Training epoch 3, loss: 0.002451
Training epoch 6, loss: 0.001892
Training epoch 9, loss: 0.001654

# Validation results
✅ STRATEGY VALIDATION PASSED - Outperformed buy-and-hold
❌ STRATEGY VALIDATION FAILED - Underperformed buy-and-hold
```

## ⚠️ Critical Training Requirements

### 1. Validation Requirement
**The strategy MUST outperform buy-and-hold in walk-forward testing.**
- Failed strategies should NOT be deployed
- Only PASSED strategies are viable for live trading

### 2. Minimum Data Requirements
- **30m data**: Minimum 5,000 data points (≈1 month)
- **1h data**: Minimum 2,500 data points (≈2 months) 
- **1d data**: Minimum 500 data points (≈2 years)

### 3. Hardware Requirements
- **RAM**: 8GB+ recommended for large datasets
- **Storage**: 1GB+ for model checkpoints and metadata
- **GPU**: Optional but speeds up training significantly

## 🎯 Next Steps After Training

1. **Verify Results**: Load saved models and double-check performance
2. **Paper Trading**: Test with `run_trading_bot.py --paper-trading`
3. **Live Trading**: Only after thorough validation
4. **Model Analysis**: Examine training metadata and period performance

## 🆘 Troubleshooting Training Issues

### Common Issues

**1. CUDA/GPU Issues**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python run_stocktime.py --mode evaluate
```

**2. Memory Issues**
```python
# Reduce batch size or training window
training_window = 200  # Instead of 390
```

**3. Data Issues**
```bash
# Validate data first
python scripts/validate_config.py --data-dir ~/data/finance/
```

**4. Model Loading Issues**
```python
# Load with CPU fallback
state_dict = torch.load(model_path, map_location='cpu')
```

Ready to start training? Pick your method and run the commands above! 🚀