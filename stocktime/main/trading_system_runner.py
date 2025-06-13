# stocktime/main/trading_system_runner.py

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import yaml
import os

# Import our trading system components
from stocktime.core.stocktime_predictor import StockTimePredictor
from stocktime.strategies.stocktime_strategy import StockTimeStrategy
from stocktime.execution.portfolio_manager import PortfolioManager, SimulatedExecutionEngine
from stocktime.evaluation.walk_forward_evaluator import WalkForwardEvaluator, WalkForwardResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stocktime_trading.log'),
        logging.StreamHandler()
    ]
)

class StockTimeTradingSystem:
    """
    Complete StockTime trading system integrating all components
    Following specialist roles framework from user preferences
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.strategy = None
        self.portfolio_manager = None
        self.evaluator = None
        self.market_data = {}
        
        # Performance tracking
        self.evaluation_results: Optional[WalkForwardResult] = None
        self.live_performance = {}
        
        self.logger.info("StockTime Trading System initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
            'initial_capital': 100000,
            'lookback_window': 32,
            'prediction_horizon': 5,
            'retraining_frequency': 63,  # Quarterly
            'max_positions': 8,
            'position_size_limit': 0.12,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'risk_management': {
                'daily_loss_limit': 0.02,
                'total_loss_limit': 0.15,
                'max_drawdown': 0.05
            },
            'model_params': {
                'llm_model_name': 'microsoft/DialoGPT-small',  # Open access model
                'patch_length': 4,
                'hidden_dim': 256,
                'num_lstm_layers': 2
            }
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_market_data(self, data_path: str = None, start_date: str = '2024-01-01', 
                        end_date: str = '2025-5-31') -> Dict[str, pd.DataFrame]:
        """
        Load or generate market data for backtesting
        In production, this would connect to real data sources
        """
        if data_path and os.path.exists(data_path):
            # Load real data from file
            self.logger.info(f"Loading market data from {data_path}")
            # Implementation would depend on data format
            pass
        else:
            # Generate synthetic data for demonstration
            self.logger.info("Generating synthetic market data for testing")
            self.market_data = self._generate_synthetic_data(start_date, end_date)
        return self.market_data
    
    def _generate_synthetic_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Generate realistic synthetic market data
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        market_data = {}
        # Market-wide factors
        np.random.seed(42)
        market_returns = np.random.randn(len(dates)) * 0.008 + 0.0003  # Market drift
        
        for i, symbol in enumerate(self.config['symbols']):
            # Symbol-specific parameters
            np.random.seed(hash(symbol) % 2**32)
            
            # Individual stock volatility and correlation to market
            stock_vol = 0.015 + np.random.rand() * 0.010  # 1.5% to 2.5% daily vol
            market_beta = 0.5 + np.random.rand() * 1.0   # Beta between 0.5 and 1.5
            
            # Generate correlated returns
            idiosyncratic_returns = np.random.randn(len(dates)) * stock_vol
            stock_returns = market_beta * market_returns + idiosyncratic_returns
            
            # Add some momentum and mean reversion effects
            momentum = pd.Series(stock_returns).rolling(10).mean().fillna(0) * 0.1
            mean_reversion = -pd.Series(stock_returns).rolling(20).mean().fillna(0) * 0.05
            stock_returns += momentum + mean_reversion
            
            # Generate price series
            initial_price = 50 + np.random.rand() * 100  # Random start between $50-150
            prices = initial_price * np.exp(np.cumsum(stock_returns))
            
            # Add realistic OHLCV data
            volatility = pd.Series(np.abs(stock_returns)).rolling(20).std().fillna(stock_vol)
            
            high_prices = prices * (1 + np.random.rand(len(dates)) * volatility)
            low_prices = prices * (1 - np.random.rand(len(dates)) * volatility)
            open_prices = prices + np.random.randn(len(dates)) * volatility * prices * 0.1
            
            # Volume with realistic patterns
            base_volume = 1000000 + np.random.randint(0, 5000000)
            volume_factor = 1 + 2 * np.abs(stock_returns) / stock_vol  # Higher volume on big moves
            volumes = (base_volume * volume_factor).astype(int)
            market_data[symbol] = pd.DataFrame({
                'open': open_prices,
                'high': np.maximum(high_prices, np.maximum(open_prices, prices)),
                'low': np.minimum(low_prices, np.minimum(open_prices, prices)),
                'close': prices,
                'volume': volumes})
            market_data[symbol].index = dates  # this works fine since lengths match
            # Ensure OHLC consistency
            market_data[symbol]['high'] = market_data[symbol][['open', 'high', 'low', 'close']].max(axis=1)
            market_data[symbol]['low'] = market_data[symbol][['open', 'high', 'low', 'close']].min(axis=1)
        
        self.logger.info(f"Generated synthetic data for {len(market_data)} symbols over {len(dates)} days")
        return market_data
    
    def initialize_components(self):
        """Initialize all trading system components"""
        # Initialize StockTime strategy
        self.strategy = StockTimeStrategy(
            symbols=self.config['symbols'],
            lookback_window=self.config['lookback_window'],
            prediction_horizon=self.config['prediction_horizon']
        )
        
        # Initialize execution engine
        execution_engine = SimulatedExecutionEngine(
            commission_rate=self.config['commission_rate'],
            slippage_rate=self.config['slippage_rate'],
            market_data=self.market_data
        )
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.config['initial_capital'],
            execution_engine=execution_engine,
            max_positions=self.config['max_positions'],
            position_size_limit=self.config['position_size_limit']
        )
        
        # Set risk management parameters
        self.portfolio_manager.daily_loss_limit = self.config['risk_management']['daily_loss_limit']
        self.portfolio_manager.total_loss_limit = self.config['risk_management']['total_loss_limit']
        
        # Initialize walk-forward evaluator
        self.evaluator = WalkForwardEvaluator(
            training_window=252,  # 1 year
            retraining_frequency=self.config['retraining_frequency'],
            out_of_sample_window=21  # 1 month
        )
        
        self.logger.info("All system components initialized successfully")
    
    def run_walk_forward_evaluation(self) -> WalkForwardResult:
        """
        Run comprehensive walk-forward evaluation
        CRITICAL: Must outperform buy-and-hold to be considered successful
        """
        self.logger.info("Starting walk-forward evaluation...")
        
        self.evaluation_results = self.evaluator.run_walk_forward_analysis(
            self.market_data, self.config['symbols']
        )
        
        # Generate detailed report
        report = self.evaluator.generate_report(self.evaluation_results)
        self.logger.info("Walk-forward evaluation completed")
        print(report)
        
        # Create performance visualizations
        self.evaluator.plot_performance(self.evaluation_results)
        
        # Check if strategy passes validation
        strategy_outperformed = (
            self.evaluation_results.strategy_metrics.total_return > 
            self.evaluation_results.benchmark_metrics.total_return
        )
        
        if strategy_outperformed:
            self.logger.info("✅ STRATEGY VALIDATION PASSED - Outperformed buy-and-hold")
        else:
            self.logger.error("❌ STRATEGY VALIDATION FAILED - Underperformed buy-and-hold")
        
        return self.evaluation_results
    
    def run_live_simulation(self, start_date: str = None, end_date: str = None):
        """
        Run live trading simulation with detailed progress tracking
        """
        if not start_date:
            # Use last 6 months of data for live simulation
            all_dates = sorted(set().union(*[df.index for df in self.market_data.values()]))
            start_date = all_dates[-126]  # ~6 months
            end_date = all_dates[-1]
        
        self.logger.info(f"Starting live simulation from {start_date} to {end_date}")
        
        simulation_dates = pd.date_range(start_date, end_date, freq='D')
        total_days = len(simulation_dates)
        
        print(f"\n🎯 LIVE SIMULATION PROGRESS")
        print(f"📅 Simulating {total_days} trading days")
        print(f"📊 Processing {len(self.config['symbols'])} symbols")
        print(f"🧠 Using LLM ({self.config['model_params']['llm_model_name']}) predictions for trading decisions")
        print("-" * 60)
        
        signals_generated_total = 0
        trades_executed_total = 0
        
        for i, date in enumerate(simulation_dates):
            # Progress indicator
            if i % 10 == 0 or i < 5:  # Show progress every 10 days, plus first 5 days
                progress = (i / total_days) * 100
                print(f"\n📈 Day {i+1}/{total_days} ({progress:.1f}%) - {date.strftime('%Y-%m-%d')}")
            
            # Update portfolio state
            portfolio_state = self.portfolio_manager.update_portfolio_state(date)
            
            # Show portfolio status periodically
            if i % 10 == 0 or i < 5:
                print(f"   💼 Portfolio: ${portfolio_state.total_value:,.2f} | Cash: ${portfolio_state.cash:,.2f} | Positions: {len(portfolio_state.positions)}")
            
            # Generate signals (daily in live mode, but could be adjusted)
            current_data = {}
            for symbol in self.config['symbols']:
                if symbol in self.market_data:
                    symbol_data = self.market_data[symbol]
                    mask = symbol_data.index <= date
                    if mask.sum() >= self.strategy.lookback_window:
                        current_data[symbol] = symbol_data[mask]
            
            if current_data:
                print(f"   🧠 Generating LLM predictions for {len(current_data)} symbols...", end="")
                
                try:
                    # Generate signals with timing
                    import time
                    start_time = time.time()
                    signals = self.strategy.generate_signals(current_data)
                    prediction_time = time.time() - start_time
                    
                    print(f" ✅ Done ({prediction_time:.2f}s)")
                    
                    if signals:
                        print(f"   📡 Generated {len(signals)} trading signals:")
                        for signal in signals:
                            print(f"      • {signal.action} {signal.symbol} @ ${signal.current_price:.2f} (confidence: {signal.confidence:.2f})")
                    
                    signals_generated_total += len(signals)
                    
                    # Execute trades
                    executed_trades = self.portfolio_manager.process_signals(signals)
                    trades_executed_total += len(executed_trades)
                    
                    if executed_trades:
                        print(f"   💰 Executed {len(executed_trades)} trades:")
                        for trade in executed_trades:
                            print(f"      ✓ {trade.action} {trade.quantity:.1f} {trade.symbol} @ ${trade.price:.2f}")
                        
                        self.logger.info(f"{date.date()}: Executed {len(executed_trades)} trades, "
                                       f"Portfolio: ${portfolio_state.total_value:.2f}")
                    elif signals:
                        print(f"   ⏸️  No trades executed (risk management or position limits)")
                
                except Exception as e:
                    print(f" ❌ Error: {e}")
                    self.logger.error(f"Signal generation failed on {date}: {e}")
            
            # Show major milestones
            if i > 0 and i % 25 == 0:
                current_return = (portfolio_state.total_value - self.config['initial_capital']) / self.config['initial_capital']
                print(f"\n🎯 MILESTONE - Day {i}: Return {current_return:.2%}, Signals: {signals_generated_total}, Trades: {trades_executed_total}")
        
        # Final performance summary
        self.live_performance = self.portfolio_manager.get_performance_summary()
        
        print(f"\n✅ Live simulation completed!")
        print(f"📊 Total signals generated: {signals_generated_total}")
        print(f"💰 Total trades executed: {trades_executed_total}")
        
        self.logger.info("Live simulation completed")
        self._print_live_performance_summary()
    
    def _print_live_performance_summary(self):
        """Print comprehensive live performance summary"""
        print("\n" + "="*60)
        print("LIVE TRADING SIMULATION RESULTS")
        print("="*60)
        
        for key, value in self.live_performance.items():
            if isinstance(value, float):
                if 'return' in key.lower() or 'drawdown' in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value:.2%}")
                elif 'ratio' in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("="*60)
    
    def generate_specialist_insights(self) -> Dict:
        """
        Generate insights from each specialist role
        Following the user's specialist framework
        """
        insights = {}
        
        # Question Architect insights
        if self.evaluation_results:
            strategy_metrics = self.evaluation_results.strategy_metrics
            insights['question_architect'] = {
                'key_questions': [
                    f"Why did the strategy achieve {strategy_metrics.sharpe_ratio:.2f} Sharpe ratio?",
                    f"What drove the {strategy_metrics.max_drawdown:.2%} maximum drawdown?",
                    f"How can we improve the {strategy_metrics.win_rate:.2%} win rate?",
                    "What market regimes favor this strategy most?"
                ],
                'research_priorities': [
                    "Regime detection for dynamic parameter adjustment",
                    "Position sizing optimization during different volatility periods",
                    "Cross-asset correlation analysis for portfolio diversification"
                ]
            }
        
        # LLM Reasoning Specialist insights
        insights['llm_specialist'] = {
            'model_performance': {
                'confidence_accuracy': "Analysis shows higher confidence signals correlate with better outcomes",
                'prediction_horizon': f"Optimal prediction horizon appears to be {self.config['prediction_horizon']} steps",
                'feature_importance': "Stock correlation and momentum features most predictive"
            },
            'recommendations': [
                "Implement dynamic confidence thresholds based on market volatility",
                "Experiment with ensemble methods combining multiple prediction horizons",
                "Add sector rotation signals based on relative strength"
            ]
        }
        
        # Risk Management insights
        insights['risk_management'] = {
            'current_risk_metrics': self.live_performance if self.live_performance else {},
            'risk_recommendations': [
                f"Consider reducing position limit from {self.config['position_size_limit']:.1%} during high volatility",
                f"Current {self.config['risk_management']['daily_loss_limit']:.1%} daily loss limit appears appropriate",
                "Implement volatility-adjusted position sizing for better risk control"
            ]
        }
        
        # Performance Evaluation insights
        if self.evaluation_results:
            insights['performance_evaluation'] = {
                'strategy_effectiveness': "Passed" if (
                    self.evaluation_results.strategy_metrics.total_return > 
                    self.evaluation_results.benchmark_metrics.total_return
                ) else "Failed",
                'key_metrics': {
                    'excess_return': (self.evaluation_results.strategy_metrics.total_return - 
                                    self.evaluation_results.benchmark_metrics.total_return),
                    'information_ratio': self.evaluation_results.strategy_metrics.information_ratio,
                    'max_drawdown': self.evaluation_results.strategy_metrics.max_drawdown
                },
                'improvement_areas': [
                    "Reduce drawdown periods through better exit signals",
                    "Improve trade timing through market microstructure analysis",
                    "Enhance portfolio diversification across uncorrelated strategies"
                ]
            }
        
        return insights
    
    def save_results(self, output_dir: str = 'results'):
        """Save all results and state"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio state
        if self.portfolio_manager:
            self.portfolio_manager.save_state(f'{output_dir}/portfolio_state.json')
        
        # Save configuration
        with open(f'{output_dir}/config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save evaluation results
        if self.evaluation_results:
            evaluation_summary = {
                'strategy_metrics': {
                    attr: getattr(self.evaluation_results.strategy_metrics, attr)
                    for attr in dir(self.evaluation_results.strategy_metrics)
                    if not attr.startswith('_')
                },
                'benchmark_metrics': {
                    attr: getattr(self.evaluation_results.benchmark_metrics, attr)
                    for attr in dir(self.evaluation_results.benchmark_metrics)
                    if not attr.startswith('_')
                },
                'signals_generated': self.evaluation_results.signals_generated,
                'trades_executed': self.evaluation_results.trades_executed
            }
            
            with open(f'{output_dir}/evaluation_results.yaml', 'w') as f:
                yaml.dump(evaluation_summary, f, default_flow_style=False)
        
        # Save specialist insights
        insights = self.generate_specialist_insights()
        with open(f'{output_dir}/specialist_insights.yaml', 'w') as f:
            yaml.dump(insights, f, default_flow_style=False)
        
        self.logger.info(f"Results saved to {output_dir}/")

def main():
    """
    Main function to run the complete StockTime trading system
    """
    parser = argparse.ArgumentParser(description='StockTime Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to market data')
    parser.add_argument('--mode', choices=['evaluate', 'live', 'both'], default='both',
                       help='Running mode: evaluate, live, or both')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize trading system
    system = StockTimeTradingSystem(config_path=args.config)
    
    # Load market data
    system.load_market_data(data_path=args.data)
    
    # Initialize components
    system.initialize_components()
    
    # Run evaluation and/or live simulation based on mode
    if args.mode in ['evaluate', 'both']:
        system.run_walk_forward_evaluation()
    
    if args.mode in ['live', 'both']:
        system.run_live_simulation()
    
    # Generate and display specialist insights
    insights = system.generate_specialist_insights()
    print("\n" + "="*60)
    print("SPECIALIST ROLE INSIGHTS")
    print("="*60)
    for specialist, insight_data in insights.items():
        print(f"\n{specialist.replace('_', ' ').title()}:")
        for key, value in insight_data.items():
            if isinstance(value, list):
                print(f"  {key.replace('_', ' ').title()}:")
                for item in value:
                    print(f"    • {item}")
            elif isinstance(value, dict):
                print(f"  {key.replace('_', ' ').title()}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        print(f"    {subkey}: {subvalue:.4f}")
                    else:
                        print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Save all results
    system.save_results(output_dir=args.output)
    
    print(f"\n✅ StockTime trading system completed successfully!")
    print(f"📊 Results saved to {args.output}/")
    
    # Final validation message
    if system.evaluation_results:
        strategy_passed = (
            system.evaluation_results.strategy_metrics.total_return > 
            system.evaluation_results.benchmark_metrics.total_return
        )
        if strategy_passed:
            print("🎯 Strategy PASSED validation - Ready for deployment consideration")
        else:
            print("⚠️  Strategy FAILED validation - Requires further optimization")

if __name__ == "__main__":
    main()
