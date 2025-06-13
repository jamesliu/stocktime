#!/usr/bin/env python3
"""
StockTime Live Trading Bot Runner

This script runs the StockTime trading system as a live trading bot.

IMPORTANT: 
- Start with paper trading to validate performance
- Only use real money after thorough backtesting
- Monitor performance closely and have stop-loss mechanisms

Usage:
    # Paper trading (recommended to start)
    python run_trading_bot.py --paper-trading

    # Monitoring only (no trades)
    python run_trading_bot.py --monitoring-only
    
    # Custom config
    python run_trading_bot.py --config config/live_trading_config.yaml
"""

import sys
import os
import signal
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stocktime.execution.live_trading_engine import LiveTradingBot

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Received shutdown signal. Stopping trading bot...")
    global bot
    if 'bot' in globals():
        bot.stop()
    sys.exit(0)

def main():
    """Main function to run the trading bot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockTime Live Trading Bot')
    parser.add_argument('--config', type=str, 
                       default='config/live_trading_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--paper-trading', action='store_true', default=True,
                       help='Use paper trading (default: True)')
    parser.add_argument('--monitoring-only', action='store_true', 
                       help='Monitor only, no actual trading')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'stocktime_bot_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🤖 StockTime Live Trading Bot")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Config: {args.config}")
    print(f"📊 Mode: {'Paper Trading' if args.paper_trading else 'LIVE TRADING'}")
    print(f"🔍 Monitoring Only: {args.monitoring_only}")
    
    if not args.paper_trading and not args.monitoring_only:
        print("\n⚠️  WARNING: LIVE TRADING MODE")
        print("   This will execute real trades with real money!")
        confirm = input("   Type 'YES' to confirm: ")
        if confirm != 'YES':
            print("   Exiting...")
            return
    
    try:
        # Initialize and start the bot
        global bot
        bot = LiveTradingBot(config_path=args.config)
        
        if args.monitoring_only:
            bot.disable_trading()
            print("🔍 Starting in monitoring mode (no trading)")
        else:
            print("🚀 Starting trading bot...")
        
        # Display initial status
        if hasattr(bot.broker, 'get_account_info'):
            account = bot.broker.get_account_info()
            print(f"💰 Initial capital: ${account.get('total_value', 0):.2f}")
        
        print(f"📈 Symbols: {', '.join(bot.config['symbols'])}")
        print(f"🕐 Trading interval: {bot.config['trading_interval']/60:.0f} minutes")
        print(f"🎯 Confidence threshold: {bot.config['confidence_threshold']}")
        print("\n▶️  Bot is running... Press Ctrl+C to stop")
        
        # Start the bot
        bot.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping bot...")
        if 'bot' in globals():
            bot.stop()
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
        print(f"❌ Bot crashed: {e}")
        if 'bot' in globals():
            bot.stop()
        return 1
    
    print("✅ Bot stopped successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)