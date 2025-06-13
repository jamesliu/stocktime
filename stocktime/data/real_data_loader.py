# stocktime/data/real_data_loader.py

"""
Real Market Data Loader for StockTime Trading System

Loads and processes real OHLCV data from CSV files for walk-forward evaluation.
Supports various timeframes and data formats.
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from pathlib import Path

class RealMarketDataLoader:
    """
    Load and process real market data for StockTime trading system
    
    Supports:
    - Multiple CSV file formats
    - Various timeframes (30m, 1h, 1d)
    - Data validation and cleaning
    - Symbol filtering and selection
    - Time range filtering
    """
    
    def __init__(self, 
                 data_directory: str,
                 timeframe: str = "30m",
                 date_column: str = "datetime",
                 required_columns: List[str] = None):
        """
        Initialize the real data loader
        
        Args:
            data_directory: Path to directory containing CSV files
            timeframe: Timeframe of the data (30m, 1h, 1d)
            date_column: Name of the datetime column
            required_columns: List of required columns
        """
        self.data_directory = Path(data_directory)
        self.timeframe = timeframe
        self.date_column = date_column
        self.required_columns = required_columns or ['open', 'high', 'low', 'close', 'volume']
        
        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.data_stats: Dict[str, Dict] = {}
        
        # Validation settings
        self.min_data_points = 100
        self.max_missing_ratio = 0.05  # 5% max missing data
        
        logging.info(f"📊 RealMarketDataLoader initialized")
        logging.info(f"   Data directory: {self.data_directory}")
        logging.info(f"   Timeframe: {self.timeframe}")
        
    def discover_symbols(self) -> List[str]:
        """
        Discover available symbols in the data directory
        
        Returns:
            List of available symbol names
        """
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
        
        # Look for CSV files
        csv_files = list(self.data_directory.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_directory}")
        
        # Extract symbol names from filenames
        symbols = []
        for file_path in csv_files:
            symbol = file_path.stem  # Remove .csv extension
            symbols.append(symbol)
        
        symbols.sort()
        logging.info(f"📈 Discovered {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
        
        return symbols
    
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a specific symbol
        
        Args:
            symbol: Symbol to load
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_directory / f"{symbol}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
            
            # Parse datetime column
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
                df.set_index(self.date_column, inplace=True)
            else:
                raise ValueError(f"DateTime column '{self.date_column}' not found in {symbol}")
            
            # Sort by datetime
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            logging.info(f"✅ Loaded {symbol}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to load {symbol}: {e}")
            raise
    
    def load_multiple_symbols(self, 
                             symbols: List[str] = None,
                             start_date: str = None,
                             end_date: str = None,
                             min_data_points: int = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols with filtering and validation
        
        Args:
            symbols: List of symbols to load (None = all available)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            min_data_points: Minimum number of data points required
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        if symbols is None:
            symbols = self.discover_symbols()
        
        if min_data_points is None:
            min_data_points = self.min_data_points
        
        loaded_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                # Load raw data
                df = self.load_symbol_data(symbol)
                
                # Apply date filters
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # Validate data quality
                if len(df) < min_data_points:
                    logging.warning(f"⚠️ {symbol}: Only {len(df)} data points, minimum {min_data_points} required")
                    failed_symbols.append(symbol)
                    continue
                
                # Check for missing data
                missing_ratio = df[self.required_columns].isnull().sum().sum() / (len(df) * len(self.required_columns))
                if missing_ratio > self.max_missing_ratio:
                    logging.warning(f"⚠️ {symbol}: {missing_ratio:.1%} missing data, maximum {self.max_missing_ratio:.1%} allowed")
                    failed_symbols.append(symbol)
                    continue
                
                # Clean data
                df = self.clean_data(df, symbol)
                
                # Store data and stats
                loaded_data[symbol] = df
                self.data_stats[symbol] = self.calculate_data_stats(df)
                
            except Exception as e:
                logging.error(f"❌ Failed to process {symbol}: {e}")
                failed_symbols.append(symbol)
        
        # Summary
        logging.info(f"📊 Data loading summary:")
        logging.info(f"   Successfully loaded: {len(loaded_data)} symbols")
        logging.info(f"   Failed to load: {len(failed_symbols)} symbols")
        
        if failed_symbols:
            logging.info(f"   Failed symbols: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")
        
        self.processed_data = loaded_data
        return loaded_data
    
    def clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and validate OHLCV data
        
        Args:
            df: Raw DataFrame
            symbol: Symbol name for logging
            
        Returns:
            Cleaned DataFrame
        """
        original_length = len(df)
        
        # Remove rows with invalid prices (negative or zero)
        price_cols = ['open', 'high', 'low', 'close']
        invalid_prices = (df[price_cols] <= 0).any(axis=1)
        df = df[~invalid_prices]
        
        # Check OHLC consistency
        ohlc_invalid = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )
        df = df[~ohlc_invalid]
        
        # Remove extreme outliers (>50% single-period change)
        returns = df['close'].pct_change()
        extreme_moves = abs(returns) > 0.5
        df = df[~extreme_moves]
        
        # Forward fill small gaps (max 3 consecutive missing values)
        df = df.fillna(method='ffill', limit=3)
        
        # Drop remaining NaN values
        df = df.dropna()
        
        cleaned_length = len(df)
        removed_count = original_length - cleaned_length
        
        if removed_count > 0:
            logging.info(f"🧹 {symbol}: Cleaned {removed_count} invalid records ({removed_count/original_length:.1%})")
        
        return df
    
    def calculate_data_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for the loaded data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with data statistics
        """
        returns = df['close'].pct_change().dropna()
        
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'days': (df.index[-1] - df.index[0]).days
            },
            'price_stats': {
                'min_price': df['close'].min(),
                'max_price': df['close'].max(),
                'avg_price': df['close'].mean(),
                'current_price': df['close'].iloc[-1]
            },
            'returns_stats': {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'max_gain': returns.max(),
                'max_loss': returns.min(),
                'sharpe_estimate': returns.mean() / (returns.std() + 1e-8) * np.sqrt(len(returns))
            },
            'volume_stats': {
                'avg_volume': df['volume'].mean(),
                'max_volume': df['volume'].max(),
                'min_volume': df['volume'].min()
            },
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'consecutive_days': len(df),
                'gaps_detected': 0  # Could implement gap detection
            }
        }
        
        return stats
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all loaded symbols
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.data_stats:
            raise ValueError("No data loaded. Call load_multiple_symbols() first.")
        
        summary_data = []
        
        for symbol, stats in self.data_stats.items():
            summary_data.append({
                'symbol': symbol,
                'records': stats['total_records'],
                'start_date': stats['date_range']['start'],
                'end_date': stats['date_range']['end'],
                'days': stats['date_range']['days'],
                'avg_price': stats['price_stats']['avg_price'],
                'volatility': stats['returns_stats']['volatility'],
                'sharpe_estimate': stats['returns_stats']['sharpe_estimate'],
                'avg_volume': stats['volume_stats']['avg_volume']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('sharpe_estimate', ascending=False)
        
        return summary_df
    
    def export_for_stocktime(self, 
                           output_symbols: List[str] = None,
                           stocktime_format: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Export data in StockTime-compatible format
        
        Args:
            output_symbols: Symbols to export (None = all loaded)
            stocktime_format: Whether to format for StockTime compatibility
            
        Returns:
            Dictionary of DataFrames ready for StockTime
        """
        if not self.processed_data:
            raise ValueError("No processed data available. Call load_multiple_symbols() first.")
        
        if output_symbols is None:
            output_symbols = list(self.processed_data.keys())
        
        exported_data = {}
        
        for symbol in output_symbols:
            if symbol not in self.processed_data:
                logging.warning(f"⚠️ Symbol {symbol} not in processed data, skipping")
                continue
            
            df = self.processed_data[symbol].copy()
            
            if stocktime_format:
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                df = df[required_cols]
                
                # Add any additional processing for StockTime compatibility
                df = df.astype(float)
                
                # Ensure no missing values
                df = df.dropna()
            
            exported_data[symbol] = df
        
        logging.info(f"📤 Exported {len(exported_data)} symbols for StockTime")
        return exported_data

def main():
    """
    Test the real data loader
    """
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Testing Real Market Data Loader")
    print("=" * 50)
    
    # Test with sample directory (adjust path as needed)
    sample_data_dir = "~/data/finance/schwab/ohlcv/30m/"
    expanded_path = os.path.expanduser(sample_data_dir)
    
    if not os.path.exists(expanded_path):
        print(f"❌ Sample data directory not found: {expanded_path}")
        print("💡 Update the path to your actual data directory")
        return
    
    try:
        # Initialize loader
        loader = RealMarketDataLoader(
            data_directory=expanded_path,
            timeframe="30m"
        )
        
        # Discover symbols
        symbols = loader.discover_symbols()
        print(f"📈 Found {len(symbols)} symbols")
        
        # Load a subset for testing
        test_symbols = symbols[:5]  # First 5 symbols
        print(f"🧪 Testing with symbols: {test_symbols}")
        
        # Load data
        market_data = loader.load_multiple_symbols(
            symbols=test_symbols,
            start_date="2024-09-01",  # Adjust based on your data
            min_data_points=50
        )
        
        # Get summary
        summary = loader.get_data_summary()
        print(f"\n📊 Data Summary:")
        print(summary.to_string(index=False))
        
        # Export for StockTime
        stocktime_data = loader.export_for_stocktime()
        print(f"\n✅ Successfully prepared {len(stocktime_data)} symbols for StockTime")
        
        # Show sample data
        if stocktime_data:
            sample_symbol = list(stocktime_data.keys())[0]
            sample_data = stocktime_data[sample_symbol]
            print(f"\n📋 Sample data for {sample_symbol}:")
            print(sample_data.head())
            print(f"   Shape: {sample_data.shape}")
            print(f"   Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.exception("Detailed error:")

if __name__ == "__main__":
    main()
