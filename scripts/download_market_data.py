#!/usr/bin/env python3
# scripts/download_market_data.py

"""
Market Data Download Utility

Downloads real market data for use with StockTime trading system.
Supports multiple data sources and formats.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_yahoo_data(symbols: List[str], 
                       start_date: str, 
                       end_date: str,
                       interval: str = '1d') -> Dict[str, pd.DataFrame]:
    """
    Download data from Yahoo Finance
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) 
        interval: Data interval (1d, 1h, etc.)
    
    Returns:
        Dictionary of DataFrames with market data
    """
    logging.info(f"Downloading data for {len(symbols)} symbols from Yahoo Finance")
    
    market_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            logging.info(f"Downloading {symbol}...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data.empty:
                logging.warning(f"No data found for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                logging.warning(f"Missing required columns for {symbol}")
                failed_symbols.append(symbol)
                continue
            
            market_data[symbol] = data[required_cols]
            logging.info(f"Successfully downloaded {len(data)} rows for {symbol}")
            
        except Exception as e:
            logging.error(f"Failed to download {symbol}: {e}")
            failed_symbols.append(symbol)
    
    if failed_symbols:
        logging.warning(f"Failed to download: {failed_symbols}")
    
    logging.info(f"Successfully downloaded data for {len(market_data)} symbols")
    return market_data

def validate_data(market_data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate downloaded market data
    
    Args:
        market_data: Dictionary of market data DataFrames
    
    Returns:
        True if data is valid, False otherwise
    """
    logging.info("Validating market data...")
    
    issues_found = False
    
    for symbol, data in market_data.items():
        # Check for missing data
        if data.isnull().any().any():
            logging.warning(f"{symbol}: Contains missing values")
            issues_found = True
        
        # Check for negative prices
        if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
            logging.warning(f"{symbol}: Contains non-positive prices")
            issues_found = True
        
        # Check OHLC consistency
        invalid_ohlc = (
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            logging.warning(f"{symbol}: OHLC data inconsistencies found")
            issues_found = True
        
        # Check for extreme outliers (>50% daily change)
        daily_returns = data['close'].pct_change()
        extreme_returns = (abs(daily_returns) > 0.5).sum()
        
        if extreme_returns > 0:
            logging.warning(f"{symbol}: {extreme_returns} extreme daily returns (>50%)")
    
    if not issues_found:
        logging.info("✅ Data validation passed")
    else:
        logging.warning("⚠️ Data validation found issues")
    
    return not issues_found

def save_data(market_data: Dict[str, pd.DataFrame], output_dir: str):
    """
    Save market data to files
    
    Args:
        market_data: Dictionary of market data DataFrames
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Saving data to {output_dir}")
    
    # Save individual symbol files
    for symbol, data in market_data.items():
        filepath = os.path.join(output_dir, f"{symbol}.csv")
        data.to_csv(filepath)
        logging.info(f"Saved {symbol} data to {filepath}")
    
    # Save combined data summary
    summary = {
        'symbols': list(market_data.keys()),
        'date_range': {
            'start': min(data.index.min() for data in market_data.values()).strftime('%Y-%m-%d'),
            'end': max(data.index.max() for data in market_data.values()).strftime('%Y-%m-%d')
        },
        'total_records': sum(len(data) for data in market_data.values()),
        'download_timestamp': datetime.now().isoformat()
    }
    
    import json
    with open(os.path.join(output_dir, 'data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"✅ Saved data for {len(market_data)} symbols")

def get_sp500_symbols() -> List[str]:
    """Get S&P 500 symbols from Wikipedia"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', {'id': 'constituents'})
        symbols = []
        
        for row in table.find_all('tr')[1:]:  # Skip header
            symbol = row.find_all('td')[0].text.strip()
            symbols.append(symbol)
        
        logging.info(f"Retrieved {len(symbols)} S&P 500 symbols")
        return symbols
        
    except Exception as e:
        logging.error(f"Failed to retrieve S&P 500 symbols: {e}")
        # Fallback to common symbols
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'BAC', 'DIS'
        ]

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download market data for StockTime')
    
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to download')
    parser.add_argument('--sp500', action='store_true', help='Download all S&P 500 symbols')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'), 
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', default='1d', choices=['1d', '1h', '5m'],
                       help='Data interval')
    parser.add_argument('--output-dir', default='data/market_data', 
                       help='Output directory')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate downloaded data')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Determine symbols to download
    if args.sp500:
        symbols = get_sp500_symbols()
        logging.info("Using S&P 500 symbols")
    elif args.symbols:
        symbols = args.symbols
        logging.info(f"Using provided symbols: {symbols}")
    else:
        # Default symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'BRK-B']
        logging.info(f"Using default symbols: {symbols}")
    
    # Download data
    market_data = download_yahoo_data(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval
    )
    
    if not market_data:
        logging.error("No data was downloaded successfully")
        return 1
    
    # Validate data if requested
    if args.validate:
        validation_passed = validate_data(market_data)
        if not validation_passed:
            logging.warning("Data validation failed, but proceeding with save")
    
    # Save data
    save_data(market_data, args.output_dir)
    
    print(f"\n✅ Market data download completed!")
    print(f"   Symbols: {len(market_data)}")
    print(f"   Date range: {args.start_date} to {args.end_date}")
    print(f"   Output: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
