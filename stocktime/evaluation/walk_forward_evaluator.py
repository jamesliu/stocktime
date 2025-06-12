# stocktime/evaluation/walk_forward_evaluator.py

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from stocktime.strategies.stocktime_strategy import StockTimeStrategy, TradingSignal
from stocktime.core.stocktime_predictor import StockTimePredictor

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    sortino_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    var_95: float
    cvar_95: float

@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis"""
    strategy_metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    signals_generated: int
    trades_executed: int
    out_of_sample_periods: int
    training_periods: int

class WalkForwardEvaluator:
    """
    Walk-forward analysis evaluator for StockTime strategy
    Critical requirement: Must outperform buy-and-hold
    """
    
    def __init__(self, 
                 training_window: int = 252,  # 1 year
                 retraining_frequency: int = 63,  # Quarterly
                 out_of_sample_window: int = 21,  # 1 month
                 min_training_samples: int = 100):
        
        self.training_window = training_window
        self.retraining_frequency = retraining_frequency
        self.out_of_sample_window = out_of_sample_window
        self.min_training_samples = min_training_samples
        
        # Performance tracking
        self.strategy_returns = []
        self.benchmark_returns = []
        self.signals_history = []
        self.trades_history = []
        
    def calculate_performance_metrics(self, 
                                    returns: pd.Series, 
                                    benchmark_returns: pd.Series = None,
                                    risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        """
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        periods_per_year = 252  # Trading days
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods_per_year)
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = excess_returns.mean() / (downside_deviation + 1e-8) * np.sqrt(periods_per_year)
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading metrics
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        winning_trades = returns[returns > 0].sum()
        losing_trades = abs(returns[returns < 0].sum())
        profit_factor = winning_trades / (losing_trades + 1e-8)
        
        # Market relative metrics
        alpha, beta, information_ratio = 0, 1, 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Calculate beta and alpha
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / (benchmark_variance + 1e-8)
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_returns.mean() * periods_per_year - risk_free_rate))
            
            # Information ratio
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(periods_per_year)
            information_ratio = active_returns.mean() * np.sqrt(periods_per_year) / (tracking_error + 1e-8)
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sortino_ratio=sortino_ratio,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def run_walk_forward_analysis(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 symbols: List[str]) -> WalkForwardResult:
        """
        Run comprehensive walk-forward analysis
        """
        # Combine all data to get common date range
        all_dates = sorted(set().union(*[df.index for df in market_data.values()]))
        all_dates = pd.DatetimeIndex(all_dates)
        
        # Initialize containers
        strategy_returns = []
        benchmark_returns = []
        signals_count = 0
        trades_count = 0
        out_of_sample_periods = 0
        training_periods = 0
        
        # Walk-forward loop
        start_idx = self.training_window
        current_idx = start_idx
        
        while current_idx + self.out_of_sample_window < len(all_dates):
            # Define training and testing periods
            train_start = max(0, current_idx - self.training_window)
            train_end = current_idx
            test_start = current_idx
            test_end = min(current_idx + self.out_of_sample_window, len(all_dates))
            
            train_dates = all_dates[train_start:train_end]
            test_dates = all_dates[test_start:test_end]
            
            if len(train_dates) < self.min_training_samples:
                current_idx += self.retraining_frequency
                continue
            
            # Prepare training data
            train_data = {}
            for symbol in symbols:
                if symbol in market_data:
                    symbol_data = market_data[symbol]
                    # Get overlapping dates with training period
                    overlapping_train = symbol_data.index.intersection(train_dates)
                    if len(overlapping_train) > self.min_training_samples:
                        train_data[symbol] = symbol_data.loc[overlapping_train]
            
            if not train_data:
                current_idx += self.retraining_frequency
                continue
            
            # Train strategy
            strategy = StockTimeStrategy(symbols=list(train_data.keys()))
            
            # Train the predictor on historical data
            self._train_predictor(strategy.predictor, train_data)
            
            # Prepare test data
            test_data = {}
            for symbol in symbols:
                if symbol in market_data:
                    symbol_data = market_data[symbol]
                    overlapping_test = symbol_data.index.intersection(test_dates)
                    if len(overlapping_test) > 0:
                        # Include lookback window for prediction
                        extended_test_start = max(0, test_start - strategy.lookback_window)
                        extended_test_dates = all_dates[extended_test_start:test_end]
                        overlapping_extended = symbol_data.index.intersection(extended_test_dates)
                        if len(overlapping_extended) >= strategy.lookback_window:
                            test_data[symbol] = symbol_data.loc[overlapping_extended]
            
            if not test_data:
                current_idx += self.retraining_frequency
                continue
            
            # Generate signals and simulate trading
            period_returns, period_signals, period_trades = self._simulate_trading_period(
                strategy, test_data, test_dates
            )
            
            # Calculate benchmark returns (buy-and-hold)
            period_benchmark_returns = self._calculate_benchmark_returns(
                test_data, test_dates, symbols
            )
            
            # Accumulate results
            strategy_returns.extend(period_returns)
            benchmark_returns.extend(period_benchmark_returns)
            signals_count += period_signals
            trades_count += period_trades
            out_of_sample_periods += len(test_dates)
            training_periods += len(train_dates)
            
            logging.info(f"Completed walk-forward period: {test_dates[0]} to {test_dates[-1]}")
            logging.info(f"Period strategy return: {sum(period_returns):.4f}, "
                        f"benchmark return: {sum(period_benchmark_returns):.4f}")
            
            # Move to next period
            current_idx += self.retraining_frequency
        
        # Convert to pandas Series for analysis
        strategy_returns_series = pd.Series(strategy_returns)
        benchmark_returns_series = pd.Series(benchmark_returns)
        
        # Calculate performance metrics
        strategy_metrics = self.calculate_performance_metrics(
            strategy_returns_series, benchmark_returns_series
        )
        benchmark_metrics = self.calculate_performance_metrics(
            benchmark_returns_series
        )
        
        return WalkForwardResult(
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            strategy_returns=strategy_returns_series,
            benchmark_returns=benchmark_returns_series,
            signals_generated=signals_count,
            trades_executed=trades_count,
            out_of_sample_periods=out_of_sample_periods,
            training_periods=training_periods
        )
    
    def _train_predictor(self, predictor: StockTimePredictor, train_data: Dict[str, pd.DataFrame]):
        """
        Train the StockTime predictor on historical data
        """
        predictor.train()
        optimizer = torch.optim.Adam(
            [p for p in predictor.parameters() if p.requires_grad], 
            lr=1e-3
        )
        
        # Prepare training dataset
        training_samples = []
        for symbol, data in train_data.items():
            prices = data['close'].values
            if len(prices) < predictor.lookback_window + 5:
                continue
                
            # Create overlapping sequences
            for i in range(len(prices) - predictor.lookback_window - 5):
                input_seq = prices[i:i + predictor.lookback_window]
                target_seq = prices[i + predictor.lookback_window:i + predictor.lookback_window + 5]
                timestamps = data.index[i:i + predictor.lookback_window].strftime('%Y-%m-%d').tolist()
                training_samples.append((input_seq, target_seq, timestamps))
        
        if not training_samples:
            return
        
        # Training loop
        for epoch in range(10):  # Limited epochs for efficiency
            total_loss = 0
            for input_seq, target_seq, timestamps in training_samples:
                optimizer.zero_grad()
                
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)
                predictions = predictor(input_tensor, timestamps)
                
                # Calculate loss on overlapping patches
                target_patches = torch.FloatTensor(target_seq[:predictions.shape[2]]).unsqueeze(0).unsqueeze(0)
                loss = torch.nn.MSELoss()(predictions[:, -1:, :target_patches.shape[2]], target_patches)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 3 == 0:
                logging.info(f"Training epoch {epoch}, loss: {total_loss/len(training_samples):.6f}")
        
        predictor.eval()
    
    def _simulate_trading_period(self, 
                                strategy: StockTimeStrategy,
                                test_data: Dict[str, pd.DataFrame],
                                test_dates: pd.DatetimeIndex) -> Tuple[List[float], int, int]:
        """
        Simulate trading for a specific period
        """
        period_returns = []
        signals_count = 0
        trades_count = 0
        
        positions = {}
        
        for date in test_dates:
            # Prepare data up to current date
            current_data = {}
            for symbol, data in test_data.items():
                mask = data.index <= date
                if mask.sum() >= strategy.lookback_window:
                    current_data[symbol] = data[mask]
            
            if not current_data:
                period_returns.append(0.0)
                continue
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            signals_count += len(signals)
            
            # Execute trades
            daily_return = 0.0
            for signal in signals:
                if signal.symbol in current_data:
                    current_price = current_data[signal.symbol]['close'].iloc[-1]
                    
                    # Simple execution: assume we can trade at current price
                    if signal.action == 'BUY' and signal.symbol not in positions:
                        positions[signal.symbol] = {
                            'entry_price': current_price,
                            'quantity': 1.0,  # Normalized position size
                            'entry_date': date
                        }
                        trades_count += 1
                    
                    elif signal.action == 'SELL' and signal.symbol in positions:
                        entry_price = positions[signal.symbol]['entry_price']
                        trade_return = (current_price - entry_price) / entry_price
                        daily_return += trade_return * positions[signal.symbol]['quantity']
                        del positions[signal.symbol]
                        trades_count += 1
            
            # Mark-to-market existing positions
            for symbol, position in positions.items():
                if symbol in current_data:
                    current_price = current_data[symbol]['close'].iloc[-1]
                    unrealized_return = (current_price - position['entry_price']) / position['entry_price']
                    # Small fraction for mark-to-market impact
                    daily_return += unrealized_return * position['quantity'] * 0.01
            
            period_returns.append(daily_return)
        
        return period_returns, signals_count, trades_count
    
    def _calculate_benchmark_returns(self, 
                                   test_data: Dict[str, pd.DataFrame],
                                   test_dates: pd.DatetimeIndex,
                                   symbols: List[str]) -> List[float]:
        """
        Calculate buy-and-hold benchmark returns
        """
        benchmark_returns = []
        
        for i, date in enumerate(test_dates):
            if i == 0:
                benchmark_returns.append(0.0)
                continue
            
            prev_date = test_dates[i-1]
            daily_return = 0.0
            valid_symbols = 0
            
            for symbol in symbols:
                if symbol in test_data:
                    data = test_data[symbol]
                    # Get prices for both dates
                    prev_price_mask = data.index <= prev_date
                    curr_price_mask = data.index <= date
                    
                    if prev_price_mask.sum() > 0 and curr_price_mask.sum() > 0:
                        prev_price = data[prev_price_mask]['close'].iloc[-1]
                        curr_price = data[curr_price_mask]['close'].iloc[-1]
                        
                        if prev_price > 0:
                            symbol_return = (curr_price - prev_price) / prev_price
                            daily_return += symbol_return
                            valid_symbols += 1
            
            # Equal-weighted portfolio
            if valid_symbols > 0:
                daily_return /= valid_symbols
            
            benchmark_returns.append(daily_return)
        
        return benchmark_returns
    
    def generate_report(self, result: WalkForwardResult) -> str:
        """
        Generate comprehensive evaluation report
        """
        report = []
        report.append("="*60)
        report.append("STOCKTIME WALK-FORWARD EVALUATION REPORT")
        report.append("="*60)
        
        # Strategy vs Benchmark comparison
        strategy_outperformed = (result.strategy_metrics.total_return > 
                               result.benchmark_metrics.total_return)
        
        report.append(f"\nSTRATEGY VALIDATION: {'PASSED' if strategy_outperformed else 'FAILED'}")
        report.append(f"Strategy must outperform buy-and-hold to be considered viable")
        
        report.append("\nPERFORMANCE COMPARISON:")
        report.append(f"{'Metric':<20} {'Strategy':<15} {'Buy & Hold':<15} {'Difference':<15}")
        report.append("-" * 65)
        
        metrics_comparison = [
            ('Total Return', 'total_return', '%'),
            ('Annual Return', 'annualized_return', '%'),
            ('Volatility', 'volatility', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Calmar Ratio', 'calmar_ratio', ''),
            ('Win Rate', 'win_rate', '%'),
            ('Sortino Ratio', 'sortino_ratio', ''),
        ]
        
        for name, attr, unit in metrics_comparison:
            strategy_val = getattr(result.strategy_metrics, attr)
            benchmark_val = getattr(result.benchmark_metrics, attr)
            diff = strategy_val - benchmark_val
            
            if unit == '%':
                strategy_str = f"{strategy_val*100:.2f}%"
                benchmark_str = f"{benchmark_val*100:.2f}%"
                diff_str = f"{diff*100:+.2f}%"
            else:
                strategy_str = f"{strategy_val:.3f}"
                benchmark_str = f"{benchmark_val:.3f}"
                diff_str = f"{diff:+.3f}"
            
            report.append(f"{name:<20} {strategy_str:<15} {benchmark_str:<15} {diff_str:<15}")
        
        report.append(f"\nTRADING STATISTICS:")
        report.append(f"Signals Generated: {result.signals_generated:,}")
        report.append(f"Trades Executed: {result.trades_executed:,}")
        report.append(f"Out-of-Sample Periods: {result.out_of_sample_periods}")
        report.append(f"Training Periods: {result.training_periods}")
        
        # Statistical significance test
        if len(result.strategy_returns) > 30:
            excess_returns = result.strategy_returns - result.benchmark_returns
            t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
            report.append(f"\nSTATISTICAL SIGNIFICANCE:")
            report.append(f"T-statistic: {t_stat:.3f}")
            report.append(f"P-value: {p_value:.6f}")
            report.append(f"Significance: {'YES' if p_value < 0.05 else 'NO'} (p < 0.05)")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def plot_performance(self, result: WalkForwardResult):
        """
        Create performance visualization plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        strategy_cumulative = (1 + result.strategy_returns).cumprod()
        benchmark_cumulative = (1 + result.benchmark_returns).cumprod()
        
        axes[0,0].plot(strategy_cumulative.index, strategy_cumulative.values, 
                      label='StockTime Strategy', linewidth=2)
        axes[0,0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                      label='Buy & Hold', linewidth=2)
        axes[0,0].set_title('Cumulative Returns')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_window = 60
        if len(result.strategy_returns) > rolling_window:
            strategy_rolling_sharpe = result.strategy_returns.rolling(rolling_window).mean() / \
                                    result.strategy_returns.rolling(rolling_window).std() * np.sqrt(252)
            benchmark_rolling_sharpe = result.benchmark_returns.rolling(rolling_window).mean() / \
                                     result.benchmark_returns.rolling(rolling_window).std() * np.sqrt(252)
            
            axes[0,1].plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe.values, 
                          label='StockTime Strategy')
            axes[0,1].plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe.values, 
                          label='Buy & Hold')
            axes[0,1].set_title(f'Rolling Sharpe Ratio ({rolling_window} periods)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Drawdown analysis
        strategy_dd = (strategy_cumulative / strategy_cumulative.expanding().max() - 1)
        benchmark_dd = (benchmark_cumulative / benchmark_cumulative.expanding().max() - 1)
        
        axes[1,0].fill_between(strategy_dd.index, strategy_dd.values, 0, 
                              alpha=0.7, label='StockTime Strategy')
        axes[1,0].fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                              alpha=0.7, label='Buy & Hold')
        axes[1,0].set_title('Drawdown Analysis')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Return distribution
        axes[1,1].hist(result.strategy_returns, bins=50, alpha=0.7, 
                      label='StockTime Strategy', density=True)
        axes[1,1].hist(result.benchmark_returns, bins=50, alpha=0.7, 
                      label='Buy & Hold', density=True)
        axes[1,1].set_title('Return Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stocktime_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Test walk-forward evaluation system
    """
    # Generate sample data for testing
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    market_data = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        # Generate more realistic price series with trend and volatility
        returns = np.random.randn(len(dates)) * 0.015 + 0.0003  # Small positive drift
        prices = 100 * np.exp(np.cumsum(returns))
        
        market_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 50000000, len(dates)),
            'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            'open': prices + np.random.randn(len(dates)) * 0.5
        }, index=dates)
    
    # Run walk-forward evaluation
    evaluator = WalkForwardEvaluator(
        training_window=252,
        retraining_frequency=63,
        out_of_sample_window=21
    )
    
    result = evaluator.run_walk_forward_analysis(market_data, symbols)
    
    # Generate and print report
    report = evaluator.generate_report(result)
    print(report)
    
    # Create performance plots
    evaluator.plot_performance(result)
    
    logging.info("Walk-forward evaluation completed successfully")

if __name__ == "__main__":
    main()
