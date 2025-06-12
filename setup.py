#!/usr/bin/env python3
# setup.py

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stocktime",
    version="1.0.0",
    author="StockTime Team",
    author_email="team@stocktime.ai",
    description="A Time Series Specialized Large Language Model Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stocktime/stocktime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "stocktime=stocktime.main.trading_system_runner:main",
        ],
    },
    include_package_data=True,
    package_data={
        "stocktime": ["config/*.yaml"],
    },
    keywords="trading, llm, stock prediction, time series, finance, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/stocktime/stocktime/issues",
        "Source": "https://github.com/stocktime/stocktime",
        "Documentation": "https://stocktime.readthedocs.io/",
    },
)
