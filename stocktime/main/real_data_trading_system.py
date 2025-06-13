# stocktime/main/real_data_trading_system.py

"""
StockTime Trading System with Real Market Data Integration

Enhanced version that can use real market data for walk-forward evaluation
instead of synthetic data.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import yaml
import os

# Import StockTime components
from stocktime.core.stocktime_predictor_2025 import StockTimePredictor2025
from stocktime.strategies.stocktime_strategy import StockTimeStrategy
from stocktime.execution.portfolio_manager import PortfolioManager, SimulatedExecutionEngine
from stocktime.evaluation.walk_forward_evaluator import WalkForwardEvaluator, WalkForwardResult
from stocktime.data.real_data_loader import RealMarketDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stocktime_real_data.log'),
        logging.StreamHandler()
    ]
)

class RealDataStockTimeTradingSystem:
    """
    StockTime Trading System with Real Market Data Support
    
    Enhanced version that can:
    - Load real OHLCV data from CSV files
    - Perform walk-forward evaluation on real market data
    - Handle various timeframes (30m, 1h, 1d)
    - Validate data quality and handle missing data
    - Compare synthetic vs real data performance
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.strategy = None
        self.portfolio_manager = None
        self.evaluator = None
        self.data_loader = None
        
        # Data storage
        self.market_data = {}
        self.data_source = "synthetic"  # "synthetic" or "real"
        
        # Performance tracking
        self.evaluation_results: Optional[WalkForwardResult] = None
        self.live_performance = {}
        
        self.logger.info("📊 RealDataStockTimeTradingSystem initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with real data settings"""
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
                'llm_model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
                'llm_model_fallback': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'llm_model_ultra_light': 'HuggingFaceTB/SmolLM2-360M-Instruct',
                'deployment_mode': 'optimal',
                'patch_length': 4,
                'hidden_dim': 256,
                'num_lstm_layers': 2,
                'max_context_length': 8192
            },
            # Real data configuration
            'real_data': {
                'enabled': False,
                'data_directory': '~/data/finance/schwab/ohlcv/30m/',
                'timeframe': '30m',
                'date_column': 'datetime',
                'start_date': None,  # Auto-detect from data
                'end_date': None,    # Auto-detect from data
                'min_data_points': 100,
                'max_missing_ratio': 0.05,
                'symbols_filter': None  # None = use all available
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_real_data_loader(self, data_directory: str = None) -> RealMarketDataLoader:
        """
        Initialize real data loader
        
        Args:
            data_directory: Override data directory from config
            
        Returns:
            Configured RealMarketDataLoader
        """
        real_data_config = self.config.get('real_data', {})
        
        # Use provided directory or config directory
        if data_directory:
            data_dir = data_directory
        else:
            data_dir = real_data_config.get('data_directory', '~/data/finance/schwab/ohlcv/30m/')
        
        # Expand user path
        data_dir = os.path.expanduser(data_dir)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Real data directory not found: {data_dir}")
        
        # Initialize loader
        self.data_loader = RealMarketDataLoader(
            data_directory=data_dir,
            timeframe=real_data_config.get('timeframe', '30m'),
            date_column=real_data_config.get('date_column', 'datetime')
        )
        
        self.logger.info(f"📊 Real data loader initialized: {data_dir}")
        return self.data_loader
    
    def load_real_market_data(self, 
                             symbols: List[str] = None,
                             start_date: str = None,
                             end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load real market data for backtesting
        
        Args:
            symbols: List of symbols to load (None = auto-discover)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Dictionary of market data DataFrames
        """
        if not self.data_loader:
            raise ValueError("Data loader not initialized. Call setup_real_data_loader() first.")
        
        real_data_config = self.config.get('real_data', {})
        
        # Use provided dates or config dates
        start_date = start_date or real_data_config.get('start_date')
        end_date = end_date or real_data_config.get('end_date')
        
        # Use provided symbols or discover available symbols
        if symbols is None:
            if real_data_config.get('symbols_filter'):
                symbols = real_data_config['symbols_filter']
            else:
                # Discover available symbols and filter by config symbols if specified
                available_symbols = self.data_loader.discover_symbols()
                config_symbols = self.config.get('symbols', [])
                
                if config_symbols:
                    # Use intersection of config symbols and available symbols
                    symbols = [s for s in config_symbols if s in available_symbols]
                    self.logger.info(f"🎯 Using {len(symbols)} symbols from config that are available in data")
                else:
                    # Use all available symbols (limit to reasonable number)
                    symbols = available_symbols[:20]  # Limit to first 20 for performance
                    self.logger.info(f"🎯 Using first {len(symbols)} available symbols")
        
        # Load data
        self.logger.info(f"📈 Loading real market data...")
        self.logger.info(f"   Symbols: {symbols}")
        self.logger.info(f"   Date range: {start_date} to {end_date}")
        
        market_data = self.data_loader.load_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            min_data_points=real_data_config.get('min_data_points', 100)
        )
        
        # Export in StockTime format
        self.market_data = self.data_loader.export_for_stocktime(
            output_symbols=list(market_data.keys())
        )
        
        self.data_source = "real"
        
        # Update config with loaded symbols
        self.config['symbols'] = list(self.market_data.keys())
        
        self.logger.info(f"✅ Loaded real data for {len(self.market_data)} symbols")
        
        # Display data summary
        self._display_data_summary()
        
        return self.market_data
    
    def _display_data_summary(self):
        """Display summary of loaded data"""
        if not self.market_data:
            return
        
        print(f"\n📊 Real Market Data Summary")
        print("=" * 50)
        
        summary_data = []
        for symbol, df in self.market_data.items():
            returns = df['close'].pct_change().dropna()
            summary_data.append({
                'Symbol': symbol,
                'Records': len(df),
                'Start': df.index[0].strftime('%Y-%m-%d'),
                'End': df.index[-1].strftime('%Y-%m-%d'),
                'Avg Price': f"${df['close'].mean():.2f}",
                'Volatility': f"{returns.std():.3f}",
                'Return': f"{((df['close'].iloc[-1] / df['close'].iloc[0]) - 1):.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print()
    
    def initialize_components(self):
        """Initialize all trading system components"""
        # Initialize StockTime strategy with 2025 predictor
        self.strategy = StockTimeStrategy(
            symbols=self.config['symbols'],
            lookback_window=self.config['lookback_window'],
            prediction_horizon=self.config['prediction_horizon']
        )
        
        # Update strategy to use 2025 predictor
        model_params = self.config['model_params']
        self.strategy.predictor = StockTimePredictor2025(
            llm_model_name=model_params['llm_model_name'],
            llm_model_fallback=model_params['llm_model_fallback'],
            llm_model_ultra_light=model_params['llm_model_ultra_light'],
            lookback_window=self.config['lookback_window'],
            patch_length=model_params['patch_length'],
            hidden_dim=model_params['hidden_dim'],
            num_lstm_layers=model_params['num_lstm_layers'],
            deployment_mode=model_params['deployment_mode']
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
        risk_config = self.config['risk_management']
        self.portfolio_manager.daily_loss_limit = risk_config['daily_loss_limit']
        self.portfolio_manager.total_loss_limit = risk_config['total_loss_limit']
        
        # Initialize walk-forward evaluator with adaptive parameters for real data
        self.evaluator = WalkForwardEvaluator(
            training_window=self._calculate_training_window(),
            retraining_frequency=self.config['retraining_frequency'],
            out_of_sample_window=self._calculate_oos_window()
        )
        
        self.logger.info("✅ All system components initialized successfully")
    
    def _calculate_training_window(self) -> int:
        """Calculate optimal training window based on data frequency"""
        timeframe = self.config.get('real_data', {}).get('timeframe', '1d')
        
        if timeframe == '30m':
            # 30-minute data: ~2 weeks of trading days = 336 intervals
            return 336
        elif timeframe == '1h':
            # Hourly data: ~1 month of trading days = 168 intervals  
            return 168
        elif timeframe == '1d':
            # Daily data: ~1 year = 252 trading days
            return 252
        else:
            # Default to 252 for unknown timeframes
            return 252
    
    def _calculate_oos_window(self) -> int:
        """Calculate optimal out-of-sample window based on data frequency"""
        timeframe = self.config.get('real_data', {}).get('timeframe', '1d')
        
        if timeframe == '30m':
            # 30-minute data: ~3 days = 48 intervals
            return 48
        elif timeframe == '1h':
            # Hourly data: ~1 week = 40 intervals
            return 40
        elif timeframe == '1d':
            # Daily data: ~1 month = 21 trading days
            return 21
        else:
            # Default to 21 for unknown timeframes
            return 21
    
    def run_real_data_evaluation(self) -> WalkForwardResult:
        """
        Run walk-forward evaluation on real market data
        
        Returns:
            WalkForwardResult with evaluation metrics
        """
        if self.data_source != "real":
            raise ValueError("Real data not loaded. Call load_real_market_data() first.")
        
        self.logger.info("🔬 Starting real data walk-forward evaluation...")
        self.logger.info(f"   Data source: Real market data ({self.data_source})")
        self.logger.info(f"   Symbols: {len(self.market_data)}")
        self.logger.info(f"   Training window: {self.evaluator.training_window}")
        self.logger.info(f"   OOS window: {self.evaluator.out_of_sample_window}")
        
        # Run evaluation
        self.evaluation_results = self.evaluator.run_walk_forward_analysis(
            self.market_data, self.config['symbols']
        )
        
        # Generate detailed report
        report = self.evaluator.generate_report(self.evaluation_results)
        print(report)
        
        # Create performance visualizations
        try:
            self.evaluator.plot_performance(self.evaluation_results)
        except Exception as e:
            self.logger.warning(f"Could not generate plots: {e}")
        
        # Validation check
        strategy_passed = (
            self.evaluation_results.strategy_metrics.total_return > 
            self.evaluation_results.benchmark_metrics.total_return
        )
        
        if strategy_passed:
            self.logger.info("✅ REAL DATA VALIDATION PASSED - Strategy outperformed buy-and-hold on real data!")
        else:
            self.logger.error("❌ REAL DATA VALIDATION FAILED - Strategy underperformed buy-and-hold on real data")
        
        return self.evaluation_results
    
    def compare_synthetic_vs_real(self) -> Dict:
        """
        Compare performance on synthetic vs real data
        
        Returns:
            Comparison results
        """
        # This would require running both synthetic and real evaluations
        # Implementation would compare the two results
        pass
    
    def save_results(self, output_dir: str = 'results/real_data'):
        """Save results with real data indicators"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio state
        if self.portfolio_manager:
            self.portfolio_manager.save_state(f'{output_dir}/portfolio_state_real_data.json')
        
        # Save data summary
        if self.data_loader:
            summary = self.data_loader.get_data_summary()
            summary.to_csv(f'{output_dir}/data_summary.csv', index=False)
        
        # Save configuration
        config_copy = self.config.copy()
        config_copy['data_source'] = self.data_source
        config_copy['evaluation_timestamp'] = datetime.now().isoformat()
        
        with open(f'{output_dir}/config_real_data.yaml', 'w') as f:
            yaml.dump(config_copy, f, default_flow_style=False)
        
        self.logger.info(f"📁 Real data results saved to {output_dir}/")

def main():
    """Main function for real data trading system"""
    parser = argparse.ArgumentParser(description='StockTime Real Data Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, help='Path to real data directory')
    parser.add_argument('--symbols', nargs='+', help='Symbols to analyze')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='results/real_data', help='Output directory')
    
    args = parser.parse_args()
    
    print("📊 StockTime Real Data Trading System")
    print("=" * 60)
    
    try:
        # Initialize system
        system = RealDataStockTimeTradingSystem(config_path=args.config)
        
        # Setup data loader
        data_directory = args.data_dir or "~/data/finance/schwab/ohlcv/30m/"
        system.setup_real_data_loader(data_directory)
        
        # Load real market data
        system.load_real_market_data(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Initialize components
        system.initialize_components()
        
        # Run evaluation
        results = system.run_real_data_evaluation()
        
        # Save results
        system.save_results(output_dir=args.output)
        
        # Final summary
        print("\n🎯 REAL DATA EVALUATION COMPLETE")
        print("=" * 40)
        strategy_passed = (
            results.strategy_metrics.total_return > 
            results.benchmark_metrics.total_return
        )
        
        print(f"Data Source: Real market data")
        print(f"Symbols Analyzed: {len(system.config['symbols'])}")
        print(f"Strategy Return: {results.strategy_metrics.total_return:.2%}")
        print(f"Benchmark Return: {results.benchmark_metrics.total_return:.2%}")
        print(f"Excess Return: {(results.strategy_metrics.total_return - results.benchmark_metrics.total_return):.2%}")
        print(f"Validation: {'PASSED ✅' if strategy_passed else 'FAILED ❌'}")
        
        return 0 if strategy_passed else 1
        
    except Exception as e:
        logging.exception("Real data evaluation failed")
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
