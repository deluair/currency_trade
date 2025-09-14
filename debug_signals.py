#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from strategies.carry_trade_strategy import CarryTradeStrategy

def debug_signal_generation():
    """Debug the signal generation process"""
    
    # Load the data
    currency_data = pd.read_csv('data/processed/currency_prices.csv', index_col=0, parse_dates=True)
    interest_data = pd.read_csv('data/processed/interest_rates.csv', index_col=0, parse_dates=True)
    
    print("Currency data shape:", currency_data.shape)
    print("Interest data shape:", interest_data.shape)
    print("\nCurrency data columns:", currency_data.columns.tolist())
    print("Interest data columns:", interest_data.columns.tolist())
    
    # Initialize strategy
    strategy = CarryTradeStrategy(
        min_rate_diff=0.5,
        momentum_window=20,
        volatility_window=30,
        max_position_size=0.15,
        transaction_cost=0.0002
    )
    
    # Test individual components
    print("\n=== Testing Rate Differentials ===")
    rate_diffs = strategy.calculate_interest_rate_differential(interest_data)
    print("Rate diffs shape:", rate_diffs.shape)
    print("Rate diffs columns:", rate_diffs.columns.tolist())
    print("Rate diffs sample:")
    print(rate_diffs.head())
    
    print("\n=== Testing Momentum ===")
    momentum = strategy.calculate_momentum(currency_data)
    print("Momentum shape:", momentum.shape)
    print("Momentum columns:", momentum.columns.tolist())
    print("Momentum sample:")
    print(momentum.head())
    
    print("\n=== Testing Volatility ===")
    volatility = strategy.calculate_volatility(currency_data)
    print("Volatility shape:", volatility.shape)
    print("Volatility columns:", volatility.columns.tolist())
    print("Volatility sample:")
    print(volatility.head())
    
    print("\n=== Testing Carry Scores ===")
    carry_scores = strategy.calculate_carry_score(rate_diffs, momentum, volatility)
    print("Carry scores shape:", carry_scores.shape)
    print("Carry scores columns:", carry_scores.columns.tolist())
    print("Carry scores sample:")
    print(carry_scores.head())
    print("Carry scores stats:")
    print(carry_scores.describe())
    
    print("\n=== Testing Full Signal Generation ===")
    signals = strategy.generate_signals(currency_data, interest_data)
    print("Signals shape:", signals.shape)
    print("Signals columns:", signals.columns.tolist())
    print("Signals sample:")
    print(signals.head())
    
    # Check for non-zero signals
    signal_cols = [col for col in signals.columns if col.endswith('_signal')]
    for col in signal_cols:
        non_zero_count = (signals[col] != 0).sum()
        print(f"{col}: {non_zero_count} non-zero signals out of {len(signals)}")

if __name__ == "__main__":
    debug_signal_generation()