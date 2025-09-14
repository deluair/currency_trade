#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from risk_management.risk_manager import RiskManager

def debug_position_calculation():
    """Debug the position calculation process"""
    
    # Load the data
    signals = pd.read_csv('data/processed/trading_signals.csv', index_col=0, parse_dates=True)
    currency_data = pd.read_csv('data/processed/currency_prices.csv', index_col=0, parse_dates=True)
    
    print("Signals shape:", signals.shape)
    print("Currency data shape:", currency_data.shape)
    print("\nSignals columns:", signals.columns.tolist())
    print("Currency data columns:", currency_data.columns.tolist())
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_portfolio_risk=0.15,
        max_individual_weight=0.2,
        var_confidence=0.05,
        transaction_cost_bps=2.0
    )
    
    # Calculate returns
    returns = risk_manager.calculate_returns(currency_data)
    print("\nReturns shape:", returns.shape)
    print("Returns columns:", returns.columns.tolist())
    
    # Check signal values
    signal_cols = [col for col in signals.columns if col.endswith('_signal')]
    print("\n=== Signal Analysis ===")
    for col in signal_cols:
        non_zero = (signals[col] != 0).sum()
        unique_vals = signals[col].unique()
        print(f"{col}: {non_zero} non-zero signals, unique values: {unique_vals}")
    
    # Test position calculation
    print("\n=== Position Calculation ===")
    positions = risk_manager.calculate_position_sizes(
        signals, returns, portfolio_value=1000000, method='risk_parity'
    )
    
    print("Positions shape:", positions.shape)
    print("Positions columns:", positions.columns.tolist())
    
    # Check position values
    position_cols = [col for col in positions.columns if col.endswith('_position_size')]
    print("\n=== Position Analysis ===")
    for col in position_cols:
        non_zero = (positions[col] != 0).sum()
        max_val = positions[col].abs().max()
        print(f"{col}: {non_zero} non-zero positions, max absolute value: {max_val:.2f}")
    
    # Sample data
    print("\n=== Sample Data ===")
    sample_idx = signals.index[50:55]  # Skip early NaN values
    print("Sample signals:")
    print(signals.loc[sample_idx, signal_cols])
    print("\nSample positions:")
    print(positions.loc[sample_idx, [col for col in positions.columns if 'position_size' in col]])
    print("\nSample weights:")
    print(positions.loc[sample_idx, [col for col in positions.columns if 'weight' in col]])

if __name__ == "__main__":
    debug_position_calculation()