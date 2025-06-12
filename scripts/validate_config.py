#!/usr/bin/env python3
# scripts/validate_config.py

"""
Configuration Validation Utility

Validates StockTime configuration files and system setup.
Checks for common configuration errors and provides recommendations.
"""

import yaml
import os
import sys
import argparse
import logging
from typing import Dict, List, Any

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"✅ Successfully loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"❌ Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"❌ Invalid YAML syntax: {e}")
        return {}

def validate_symbols(symbols: List[str]) -> List[str]:
    """Validate stock symbols"""
    issues = []
    
    if not symbols:
        issues.append("No symbols specified")
        return issues
    
    if len(symbols) < 3:
        issues.append(f"Only {len(symbols)} symbols specified. Recommend at least 3 for diversification")
    
    if len(symbols) > 20:
        issues.append(f"{len(symbols)} symbols specified. Consider reducing for better performance")
    
    # Check for invalid symbol formats
    invalid_symbols = []
    for symbol in symbols:
        if not isinstance(symbol, str) or len(symbol) > 10 or not symbol.replace('-', '').replace('.', '').isalnum():
            invalid_symbols.append(symbol)
    
    if invalid_symbols:
        issues.append(f"Invalid symbol formats: {invalid_symbols}")
    
    logging.info(f"✅ Symbol validation: {len(symbols)} symbols")
    return issues

def validate_capital_settings(config: Dict[str, Any]) -> List[str]:
    """Validate capital allocation settings"""
    issues = []
    
    initial_capital = config.get('initial_capital', 0)
    max_positions = config.get('max_positions', 0)
    position_size_limit = config.get('position_size_limit', 0)
    
    if initial_capital <= 0:
        issues.append("Initial capital must be positive")
    elif initial_capital < 10000:
        issues.append(f"Initial capital ${initial_capital:,} is quite low. Consider at least $10,000")
    
    if max_positions <= 0:
        issues.append("Max positions must be positive")
    elif max_positions > len(config.get('symbols', [])):
        issues.append("Max positions exceeds number of symbols")
    
    if position_size_limit <= 0 or position_size_limit > 1:
        issues.append("Position size limit must be between 0 and 1")
    elif position_size_limit > 0.2:
        issues.append(f"Position size limit {position_size_limit:.1%} is quite high. Consider max 20%")
    
    # Check if position sizes are compatible
    if max_positions > 0 and position_size_limit > 0:
        max_allocation = max_positions * position_size_limit
        if max_allocation > 1.0:
            issues.append(f"Max positions × position size limit = {max_allocation:.1%} > 100%")
    
    logging.info(f"✅ Capital settings validation completed")
    return issues

def validate_model_params(config: Dict[str, Any]) -> List[str]:
    """Validate model parameters"""
    issues = []
    
    model_params = config.get('model_params', {})
    
    lookback_window = config.get('lookback_window', 0)
    if lookback_window < 10:
        issues.append("Lookback window too small. Recommend at least 10")
    elif lookback_window > 100:
        issues.append("Lookback window very large. May impact performance")
    
    prediction_horizon = config.get('prediction_horizon', 0)
    if prediction_horizon < 1:
        issues.append("Prediction horizon must be at least 1")
    elif prediction_horizon > 20:
        issues.append("Prediction horizon very large. May reduce accuracy")
    
    patch_length = model_params.get('patch_length', 0)
    if patch_length < 2:
        issues.append("Patch length too small. Recommend at least 2")
    elif lookback_window % patch_length != 0:
        issues.append(f"Lookback window ({lookback_window}) should be divisible by patch length ({patch_length})")
    
    hidden_dim = model_params.get('hidden_dim', 0)
    if hidden_dim < 64:
        issues.append("Hidden dimension quite small. May limit model capacity")
    elif hidden_dim > 1024:
        issues.append("Hidden dimension very large. May cause memory issues")
    
    logging.info(f"✅ Model parameters validation completed")
    return issues

def validate_risk_management(config: Dict[str, Any]) -> List[str]:
    """Validate risk management settings"""
    issues = []
    
    risk_mgmt = config.get('risk_management', {})
    
    daily_loss_limit = risk_mgmt.get('daily_loss_limit', 0)
    if daily_loss_limit <= 0 or daily_loss_limit > 0.1:
        issues.append("Daily loss limit should be between 0 and 10%")
    
    total_loss_limit = risk_mgmt.get('total_loss_limit', 0)
    if total_loss_limit <= 0 or total_loss_limit > 0.5:
        issues.append("Total loss limit should be between 0 and 50%")
    
    max_drawdown = risk_mgmt.get('max_drawdown', 0)
    if max_drawdown <= 0 or max_drawdown > 0.2:
        issues.append("Max drawdown threshold should be between 0 and 20%")
    
    # Check consistency
    if daily_loss_limit > 0 and total_loss_limit > 0:
        if daily_loss_limit > total_loss_limit:
            issues.append("Daily loss limit should not exceed total loss limit")
    
    commission_rate = config.get('commission_rate', 0)
    if commission_rate < 0 or commission_rate > 0.01:
        issues.append("Commission rate seems unrealistic. Should be 0-1%")
    
    slippage_rate = config.get('slippage_rate', 0)
    if slippage_rate < 0 or slippage_rate > 0.01:
        issues.append("Slippage rate seems unrealistic. Should be 0-1%")
    
    logging.info(f"✅ Risk management validation completed")
    return issues

def validate_evaluation_params(config: Dict[str, Any]) -> List[str]:
    """Validate evaluation parameters"""
    issues = []
    
    evaluation = config.get('evaluation', {})
    
    training_window = evaluation.get('training_window', 0)
    if training_window < 50:
        issues.append("Training window too small. Recommend at least 50 days")
    elif training_window > 1000:
        issues.append("Training window very large. May slow down evaluation")
    
    retraining_frequency = config.get('retraining_frequency', 0)
    if retraining_frequency < 10:
        issues.append("Retraining frequency too frequent. May cause overfitting")
    elif retraining_frequency > training_window / 2:
        issues.append("Retraining frequency too infrequent relative to training window")
    
    out_of_sample_window = evaluation.get('out_of_sample_window', 0)
    if out_of_sample_window < 5:
        issues.append("Out-of-sample window too small for reliable testing")
    elif out_of_sample_window > retraining_frequency:
        issues.append("Out-of-sample window larger than retraining frequency")
    
    logging.info(f"✅ Evaluation parameters validation completed")
    return issues

def check_dependencies() -> List[str]:
    """Check if required dependencies are installed"""
    issues = []
    
    required_packages = [
        'torch', 'transformers', 'numpy', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'scipy', 'yaml', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Missing required packages: {missing_packages}")
        issues.append("Run: pip install -r requirements.txt")
    
    # Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            logging.info("ℹ️ CUDA not available. Using CPU (slower training)")
    except ImportError:
        pass
    
    logging.info(f"✅ Dependencies check completed")
    return issues

def generate_recommendations(config: Dict[str, Any], all_issues: List[str]) -> List[str]:
    """Generate configuration recommendations"""
    recommendations = []
    
    # Performance recommendations
    symbols_count = len(config.get('symbols', []))
    if symbols_count < 5:
        recommendations.append("Consider adding more symbols for better diversification")
    
    # Risk recommendations
    position_limit = config.get('position_size_limit', 0)
    if position_limit > 0.15:
        recommendations.append("Consider reducing position size limit to 15% or less")
    
    # Model recommendations
    lookback_window = config.get('lookback_window', 0)
    if lookback_window < 30:
        recommendations.append("Consider increasing lookback window to 30+ for better patterns")
    
    # Capital recommendations
    initial_capital = config.get('initial_capital', 0)
    if initial_capital < 50000:
        recommendations.append("Higher initial capital allows for better diversification")
    
    return recommendations

def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate StockTime configuration')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--strict', action='store_true',
                       help='Exit with error if any issues found')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🔍 StockTime Configuration Validation")
    print("=" * 50)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return 1
    
    # Run all validations
    all_issues = []
    
    print("\n📊 Validating symbols...")
    all_issues.extend(validate_symbols(config.get('symbols', [])))
    
    print("\n💰 Validating capital settings...")
    all_issues.extend(validate_capital_settings(config))
    
    print("\n🧠 Validating model parameters...")
    all_issues.extend(validate_model_params(config))
    
    print("\n🛡️ Validating risk management...")
    all_issues.extend(validate_risk_management(config))
    
    print("\n📈 Validating evaluation parameters...")
    all_issues.extend(validate_evaluation_params(config))
    
    print("\n📦 Checking dependencies...")
    all_issues.extend(check_dependencies())
    
    # Generate recommendations
    recommendations = generate_recommendations(config, all_issues)
    
    # Display results
    print(f"\n🎯 Validation Results")
    print("=" * 30)
    
    if all_issues:
        print(f"❌ Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ No issues found!")
    
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Summary
    print(f"\n📋 Configuration Summary:")
    print(f"   Symbols: {len(config.get('symbols', []))}")
    print(f"   Initial Capital: ${config.get('initial_capital', 0):,}")
    print(f"   Max Positions: {config.get('max_positions', 0)}")
    print(f"   Position Limit: {config.get('position_size_limit', 0):.1%}")
    print(f"   Lookback Window: {config.get('lookback_window', 0)}")
    
    # Exit based on strict mode
    if args.strict and all_issues:
        print(f"\n❌ Exiting due to validation issues (strict mode)")
        return 1
    
    print(f"\n✅ Configuration validation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
