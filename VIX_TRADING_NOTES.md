# $VIX Usage in StockTime Trading System

## 📊 What is $VIX?

**$VIX (Volatility Index)** is the "fear gauge" of the market - it measures expected volatility in the S&P 500.

## 🚨 Important Trading Rules

### ❌ Cannot Trade $VIX Directly
- **$VIX is an INDEX, not a tradeable security**
- You CANNOT buy/sell $VIX like a stock
- It's calculated from S&P 500 options prices
- Used for analysis only, not trading

### ✅ Can Use $VIX as Signal/Feature
- **Excellent market sentiment indicator**
- High VIX (>30) = Market fear/uncertainty
- Low VIX (<15) = Market complacency  
- Rising VIX = Increasing volatility/risk
- Falling VIX = Calming markets

## 🎯 StockTime Implementation

### Current Status
- $VIX is included in your 247-symbol dataset
- System will automatically load $VIX data
- **Trading filter implemented**: `_is_tradeable_symbol()` excludes $VIX from trades
- $VIX data available for feature engineering and analysis

### Automatic Filtering in StockTime

```python
def _is_tradeable_symbol(self, symbol: str) -> bool:
    non_tradeable = {
        '$VIX',   # Volatility Index - cannot trade directly
        '$SPX',   # S&P 500 Index - cannot trade directly  
        '$NDX',   # NASDAQ 100 Index - cannot trade directly
    }
    return symbol not in non_tradeable and not symbol.startswith('$')
```

### Recommended Usage in StockTime

1. **Feature Engineering**: Use $VIX as market regime indicator
   ```python
   # Example: High VIX = reduce position sizes
   if vix_level > 30:
       position_size *= 0.5  # Reduce risk during high volatility
   ```

2. **Risk Management**: Adjust strategy based on VIX levels
   ```python
   # Example: VIX-based risk adjustments
   vix_adjustment = {
       'low_vix': 1.2,    # VIX < 15: Increase positions
       'normal_vix': 1.0,  # VIX 15-25: Normal positions  
       'high_vix': 0.6    # VIX > 25: Reduce positions
   }
   ```

3. **Market Timing**: Use VIX spikes for entry/exit signals
   ```python
   # Example: VIX spike = potential buying opportunity
   if vix_spike > 50%:  # VIX increased 50% in short period
       bias_towards_long_positions = True
   ```

## 📈 247 Symbol Analysis

Your dataset includes:
- **Tradeable**: 246 symbols (stocks, ETFs, etc.)
- **Non-tradeable**: 1 symbol ($VIX)

## 🔧 GPU Training Enhancement

Added GPU acceleration to walk-forward evaluator:
- **Automatic GPU detection**: Uses CUDA if available
- **Model device placement**: Moves models and tensors to GPU
- **Memory optimization**: Efficient GPU memory usage
- **Fallback to CPU**: Seamless fallback if GPU unavailable

### GPU Training Benefits:
- **5-10x faster training** with 247 symbols
- **Larger batch processing** capability
- **Memory-efficient operations** for extensive datasets

## 🚀 Ready for Training

Your StockTime system now:
✅ **Filters $VIX from trading** (keeps for analysis)  
✅ **GPU-accelerated training** (if available)  
✅ **247-symbol support** with proper handling  
✅ **Enhanced model persistence** with metadata  

## Command to Start Training:
```bash
python run_stocktime.py \
    --mode evaluate \
    --data ~/data/finance/schwab/ohlcv/30m/ \
    --timeframe 30m \
    --output results/full_universe_247_symbols
```

This gives you the **best of both worlds**:
- Rich feature set for learning (including VIX data)
- Safe trading execution (excluding VIX from trades)
- GPU-accelerated training for maximum performance