#!/usr/bin/env python3
# run_stocktime.py

"""
Main entry point for StockTime Trading System

This script provides a convenient way to run the StockTime trading system
with various options and configurations.
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add stocktime package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stocktime'))

try:
    from stocktime.main.trading_system_runner import main as run_trading_system
except ImportError as e:
    print(f"❌ Error importing StockTime: {e}")
    print("💡 Try running: pip install -e .")
    sys.exit(1)

def setup_logging(level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'stocktime_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

def print_banner():
    """Print StockTime banner"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   ███████╗████████╗ ██████╗  ██████╗██╗  ██╗████████╗██╗███╗   ███╗███████╗
║   ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝╚══██╔══╝██║████╗ ████║██╔════╝
║   ███████╗   ██║   ██║   ██║██║     █████╔╝    ██║   ██║██╔████╔██║█████╗  
║   ╚════██║   ██║   ██║   ██║██║     ██╔═██╗    ██║   ██║██║╚██╔╝██║██╔══╝  
║   ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗   ██║   ██║██║ ╚═╝ ██║███████╗
║   ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝
║                                                                ║
║           A Time Series Specialized LLM Trading System         ║
║                                                                ║
║   🧠 LLM-Powered Predictions  📈 Walk-Forward Validation       ║
║   🛡️ Professional Risk Mgmt   ⚡ Specialist Role Framework    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_ready():
    """Check if system is ready to run"""
    issues = []
    
    # Check required files
    required_files = [
        'config/config.yaml',
        'stocktime/__init__.py',
        'requirements.txt'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            issues.append(f"Missing required file: {file_path}")
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check key dependencies
    try:
        import torch
        import pandas
        import numpy
    except ImportError as e:
        issues.append(f"Missing dependency: {e}")
    
    return issues

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='StockTime Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stocktime.py --quick                    # Quick example run
  python run_stocktime.py --mode evaluate            # Walk-forward evaluation only
  python run_stocktime.py --mode live                # Live simulation only
  python run_stocktime.py --mode both                # Complete system run
  python run_stocktime.py --config custom.yaml       # Custom configuration
  python run_stocktime.py --symbols AAPL MSFT GOOGL  # Custom symbols

For more information: https://github.com/stocktime/stocktime
        """
    )
    
    # Main options
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['evaluate', 'live', 'both'], default='both',
                       help='Running mode (default: both)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    
    # Quick start options
    parser.add_argument('--quick', action='store_true',
                       help='Run quick start example with minimal configuration')
    parser.add_argument('--symbols', nargs='+', 
                       help='Override symbols from config')
    
    # System options
    parser.add_argument('--check', action='store_true',
                       help='Check system health and exit')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate configuration and exit')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    # Advanced options
    parser.add_argument('--no-banner', action='store_true',
                       help='Suppress banner display')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate setup without running')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Display banner unless suppressed
    if not args.no_banner:
        print_banner()
        print("🚀 Starting StockTime Trading System...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Check system readiness
    issues = check_system_ready()
    if issues:
        print("❌ System readiness check failed:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Try running:")
        print("   make install")
        print("   make health-check")
        return 1
    
    # Handle special commands
    if args.check:
        print("🏥 Running system health check...")
        os.system("python scripts/system_health_check.py")
        return 0
    
    if args.validate_config:
        print("🔍 Validating configuration...")
        os.system(f"python scripts/validate_config.py --config {args.config}")
        return 0
    
    if args.dry_run:
        print("✅ Dry run completed - system appears ready")
        print(f"   Config: {args.config}")
        print(f"   Mode: {args.mode}")
        print(f"   Output: {args.output}")
        return 0
    
    # Quick start mode
    if args.quick:
        print("⚡ Running quick start example...")
        try:
            os.system("python examples/quick_start.py")
            return 0
        except Exception as e:
            print(f"❌ Quick start failed: {e}")
            return 1
    
    # Prepare arguments for main trading system
    sys_args = [
        '--config', args.config,
        '--mode', args.mode,
        '--output', args.output
    ]
    
    # Override sys.argv for the trading system
    original_argv = sys.argv.copy()
    sys.argv = ['trading_system_runner.py'] + sys_args
    
    try:
        print("🎯 Launching StockTime Trading System...")
        print(f"   Configuration: {args.config}")
        print(f"   Mode: {args.mode}")
        print(f"   Output Directory: {args.output}")
        
        if args.symbols:
            print(f"   Custom Symbols: {args.symbols}")
        
        print()
        
        # Run the main trading system
        result = run_trading_system()
        
        print("\n" + "="*60)
        if result == 0:
            print("✅ StockTime execution completed successfully!")
            print(f"📊 Results available in: {args.output}/")
        else:
            print("❌ StockTime execution encountered issues")
            print("💡 Check logs for details")
        
        return result
        
    except KeyboardInterrupt:
        print("\n⚠️ StockTime execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ StockTime execution failed: {e}")
        logging.exception("Execution error")
        return 1
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    exit(main())
