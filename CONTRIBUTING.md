# Contributing to StockTime Trading System

Thank you for your interest in contributing to StockTime! This document provides guidelines and information for contributors.

## 🎯 Project Mission

StockTime aims to advance the field of algorithmic trading through rigorous implementation of Large Language Model architectures specifically designed for financial time series prediction. Our core principles:

- **Research-Based**: Grounded in peer-reviewed research
- **Validation-First**: Mandatory walk-forward testing with buy-and-hold outperformance
- **Risk-Aware**: Professional-grade risk management
- **Open Science**: Transparent, reproducible, and educational

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of financial markets and machine learning
- Familiarity with PyTorch and Transformers

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/stocktime.git
   cd stocktime
   ```

2. **Setup Development Environment**
   ```bash
   make setup-dev
   # or manually:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Verify Installation**
   ```bash
   make health-check
   make test
   ```

4. **Run Example**
   ```bash
   make run-example
   ```

## 🔧 Development Workflow

### Branch Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for new features
- `feature/feature-name`: Individual feature development
- `bugfix/issue-number`: Bug fixes
- `research/experiment-name`: Research experiments

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation

3. **Run Quality Checks**
   ```bash
   make pre-commit  # Runs format, lint, type-check, test
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(predictor): add ensemble prediction capability
fix(portfolio): handle edge case in position sizing
docs(readme): update installation instructions
test(strategy): add integration tests for signal generation
```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 100 characters (instead of 79)
- **String Quotes**: Prefer single quotes `'` over double quotes `"`
- **Import Order**: Use `isort` for automatic import sorting

### Code Formatting

- **Black**: For code formatting
- **Flake8**: For linting
- **MyPy**: For type checking

Run all at once:
```bash
make pre-commit
```

### Documentation Style

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Required for all public functions
- **Comments**: Explain why, not what

Example:
```python
def calculate_position_size(self, 
                          signal_confidence: float, 
                          predicted_return: float,
                          portfolio_value: float) -> float:
    """
    Calculate optimal position size using Kelly criterion.
    
    Args:
        signal_confidence: Model confidence score [0, 1]
        predicted_return: Expected return for the trade
        portfolio_value: Current total portfolio value
        
    Returns:
        Position size as fraction of portfolio value
        
    Raises:
        ValueError: If confidence is not in valid range
    """
    if not 0 <= signal_confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    
    # Kelly criterion with confidence adjustment
    # We reduce position size based on uncertainty
    kelly_fraction = predicted_return / (predicted_return ** 2)
    return kelly_fraction * signal_confidence
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_stocktime_predictor.py     # Core predictor tests
├── test_trading_strategy.py        # Strategy component tests
├── test_portfolio_manager.py       # Portfolio management tests
├── test_walk_forward_evaluator.py  # Evaluation tests
├── integration/                    # Integration tests
├── performance/                    # Performance benchmarks
└── fixtures/                       # Test data and utilities
```

### Test Requirements

1. **Coverage**: Aim for >90% test coverage
2. **Types**: Unit tests, integration tests, performance tests
3. **Data**: Use synthetic data for reproducibility
4. **Assertions**: Clear, descriptive assertion messages

### Writing Tests

```python
import unittest
import numpy as np
from stocktime.core import StockTimePredictor

class TestStockTimePredictor(unittest.TestCase):
    """Test cases for StockTime predictor component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = StockTimePredictor(
            llm_model_name="microsoft/DialoGPT-small",
            lookback_window=16  # Smaller for testing
        )
        
    def test_prediction_output_shape(self):
        """Test that predictions have correct shape."""
        # Arrange
        price_data = np.random.randn(32) + 100
        timestamps = [f"2024-01-{i:02d}" for i in range(16)]
        
        # Act
        predictions = self.predictor.predict_next_prices(
            price_data[:16], timestamps, prediction_steps=3
        )
        
        # Assert
        self.assertEqual(len(predictions), 3, 
                        "Should return 3 predictions")
        self.assertTrue(all(isinstance(p, (int, float, np.number)) 
                           for p in predictions),
                       "All predictions should be numeric")
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
python -m pytest tests/test_stocktime_predictor.py -v

# Run specific test method
python -m pytest tests/test_stocktime_predictor.py::TestStockTimePredictor::test_prediction_output_shape -v
```

## 📊 Research Contributions

### Research Guidelines

StockTime is built on solid research foundations. When contributing research-based features:

1. **Literature Review**: Reference relevant papers
2. **Hypothesis**: Clearly state what you're testing
3. **Methodology**: Describe your approach
4. **Validation**: Include walk-forward testing
5. **Results**: Compare against baselines

### Experiment Structure

```python
# research/experiments/new_feature_experiment.py

"""
Experiment: Enhanced Risk Management with Dynamic Position Sizing

Hypothesis: Volatility-adjusted position sizing improves risk-adjusted returns
compared to fixed position sizing.

Methodology:
1. Implement volatility-adjusted Kelly criterion
2. Compare against fixed position sizing baseline
3. Evaluate using walk-forward analysis
4. Measure Sharpe ratio, max drawdown, and total return

Expected Outcome: 10-15% improvement in Sharpe ratio
"""

class DynamicPositionSizingExperiment:
    def run_experiment(self):
        # Implementation
        pass
    
    def analyze_results(self):
        # Statistical analysis
        pass
```

## 🛡️ Financial Validation Requirements

### Mandatory Validation

**CRITICAL**: All trading strategies must pass walk-forward validation:

1. **Outperformance**: Strategy return > Buy-and-hold return
2. **Statistical Significance**: p-value < 0.05 for excess returns
3. **Risk Metrics**: Sharpe ratio > 1.0, Max drawdown < 10%
4. **Robustness**: Performance across different market regimes

### Validation Checklist

- [ ] Walk-forward evaluation implemented
- [ ] Benchmark comparison (buy-and-hold)
- [ ] Statistical significance testing
- [ ] Risk metrics within acceptable bounds
- [ ] Out-of-sample testing period ≥ 20% of total data
- [ ] Documentation of validation results

Example validation:
```python
def validate_strategy(strategy_returns, benchmark_returns):
    """Validate trading strategy against requirements."""
    # Must outperform benchmark
    if strategy_returns.sum() <= benchmark_returns.sum():
        raise ValidationError("Strategy underperforms buy-and-hold")
    
    # Statistical significance test
    excess_returns = strategy_returns - benchmark_returns
    t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
    if p_value >= 0.05:
        raise ValidationError("Strategy not statistically significant")
    
    # Risk metrics
    sharpe = calculate_sharpe_ratio(strategy_returns)
    if sharpe < 1.0:
        raise ValidationError(f"Sharpe ratio {sharpe:.2f} below minimum 1.0")
```

## 📚 Documentation Guidelines

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step instructions
3. **Research Notes**: Implementation details and references
4. **Examples**: Working code demonstrations

### Writing Documentation

- **Clear and Concise**: Avoid jargon when possible
- **Examples**: Include code examples for complex concepts
- **Links**: Reference related sections and external resources
- **Accuracy**: Keep documentation synchronized with code

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
make docs

# View documentation
open docs/_build/html/index.html
```

## 🐛 Bug Reports

### Before Reporting

1. **Search Existing Issues**: Check if bug already reported
2. **Reproduce**: Confirm bug is reproducible
3. **Test Environment**: Try in clean environment
4. **Latest Version**: Ensure you're using latest version

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or log files.

**Environment:**
 - OS: [e.g. iOS]
 - Python version: [e.g. 3.9.0]
 - StockTime version: [e.g. 1.0.0]
 - Dependencies: [paste pip freeze output]

**Additional Context**
Add any other context about the problem here.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of what you want to happen.

**Use Case**
Describe the specific use case or problem this solves.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Describe alternative solutions you've considered.

**Research Support**
Link to any research papers or references supporting this feature.

**Implementation Notes**
Any technical considerations or constraints.
```

## 🔍 Code Review Process

### For Contributors

1. **Self Review**: Review your own code first
2. **Documentation**: Update relevant documentation
3. **Tests**: Ensure tests pass and coverage maintained
4. **Description**: Write clear PR description

### Review Criteria

- **Functionality**: Does it work as intended?
- **Testing**: Adequate test coverage?
- **Documentation**: Clear and accurate?
- **Performance**: Any performance implications?
- **Security**: Any security considerations?
- **Financial Validation**: Passes required validation?

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Research contribution

## Financial Validation
- [ ] Strategy passes walk-forward validation
- [ ] Outperforms buy-and-hold benchmark
- [ ] Statistical significance confirmed
- [ ] Risk metrics within bounds

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass
```

## 🏆 Recognition

### Contributor Recognition

We recognize contributors through:

- **GitHub Contributors**: Listed in repository
- **Release Notes**: Mentioned in CHANGELOG.md
- **Documentation**: Listed in acknowledgments
- **Research Papers**: Co-authorship for significant research contributions

### Hall of Fame

Outstanding contributors may be invited to join the core maintainer team.

## 📞 Getting Help

### Community Support

- **GitHub Discussions**: General questions and discussions
- **Issues**: Bug reports and feature requests
- **Email**: team@stocktime.ai for sensitive matters

### Development Questions

- **Architecture Questions**: Ask in GitHub Discussions
- **Implementation Help**: Include code examples
- **Research Questions**: Provide literature references

## 📋 Development Checklist

Before submitting a PR:

- [ ] Code follows style guidelines (`make lint` passes)
- [ ] All tests pass (`make test` passes)
- [ ] Documentation updated
- [ ] Changelog updated (for significant changes)
- [ ] Financial validation completed (for trading features)
- [ ] Performance impact assessed
- [ ] Security implications considered

## 🎉 Thank You!

Your contributions help advance the field of algorithmic trading and make sophisticated trading tools accessible to researchers and practitioners worldwide.

Together, we're building the future of intelligent trading systems! 🚀
