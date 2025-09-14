#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from strategies.carry_trade_strategy import CarryTradeStrategy
from risk_management.risk_manager import RiskManager
from backtesting.backtest_engine import BacktestEngine

def optimize_strategy_parameters():
    """Optimize strategy parameters using grid search"""
    
    print("=== STRATEGY PARAMETER OPTIMIZATION ===")
    
    # Load data
    print("Loading data...")
    currency_data = pd.read_csv('data/processed/currency_prices.csv', index_col=0, parse_dates=True)
    interest_rates = pd.read_csv('data/processed/interest_rates.csv', index_col=0, parse_dates=True)
    
    # Parameter grid for optimization
    param_grid = {
        'min_rate_diff': [0.1, 0.25, 0.5, 1.0],  # More sensitive thresholds
        'momentum_window': [10, 20, 30, 50],
        'volatility_window': [20, 30, 60],
        'max_position_size': [0.05, 0.1, 0.15, 0.2],  # Test different position sizes
        'transaction_cost': [0.0001, 0.0002]  # Lower transaction costs
    }
    
    # Generate all combinations (limit to reasonable number)
    param_combinations = list(product(
        param_grid['min_rate_diff'][:2],  # Test top 2 values
        param_grid['momentum_window'][:3],  # Test top 3 values
        param_grid['volatility_window'][:2],  # Test top 2 values
        param_grid['max_position_size'][:3],  # Test top 3 values
        param_grid['transaction_cost'][:1]  # Use best transaction cost
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    results = []
    
    for i, (min_rate_diff, momentum_window, volatility_window, max_position_size, transaction_cost) in enumerate(param_combinations):
        try:
            print(f"\nTesting combination {i+1}/{len(param_combinations)}:")
            print(f"  min_rate_diff: {min_rate_diff}")
            print(f"  momentum_window: {momentum_window}")
            print(f"  volatility_window: {volatility_window}")
            print(f"  max_position_size: {max_position_size}")
            print(f"  transaction_cost: {transaction_cost}")
            
            # Initialize strategy with current parameters
            strategy = CarryTradeStrategy(
                min_rate_diff=min_rate_diff,
                momentum_window=momentum_window,
                volatility_window=volatility_window,
                max_position_size=max_position_size,
                transaction_cost=transaction_cost
            )
            
            # Generate signals
            signals = strategy.generate_signals(currency_data, interest_rates)
            
            # Calculate returns for risk manager
            risk_manager = RiskManager()
            returns = risk_manager.calculate_returns(currency_data)
            
            # Calculate position sizes
            positions = risk_manager.calculate_position_sizes(
                signals, returns, portfolio_value=1000000, method='risk_parity'
            )
            
            # Run backtest
            backtest_engine = BacktestEngine(
                initial_capital=1000000,
                transaction_cost_bps=transaction_cost * 10000,
                rebalance_frequency='daily'
            )
            
            portfolio_df, backtest_results = backtest_engine.run_backtest(
                currency_data, signals, positions, interest_rates
            )
            
            # Store results
            result = {
                'min_rate_diff': min_rate_diff,
                'momentum_window': momentum_window,
                'volatility_window': volatility_window,
                'max_position_size': max_position_size,
                'transaction_cost': transaction_cost,
                'total_return': backtest_results.total_return,
                'annualized_return': backtest_results.annualized_return,
                'volatility': backtest_results.volatility,
                'sharpe_ratio': backtest_results.sharpe_ratio,
                'max_drawdown': backtest_results.max_drawdown,
                'total_trades': backtest_results.total_trades,
                'win_rate': backtest_results.win_rate,
                'profit_factor': backtest_results.profit_factor
            }
            
            results.append(result)
            
            print(f"  Results: Return={backtest_results.total_return:.2%}, Sharpe={backtest_results.sharpe_ratio:.3f}, Trades={backtest_results.total_trades}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    # Convert results to DataFrame and analyze
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print("\n=== OPTIMIZATION RESULTS ===")
        
        # Sort by Sharpe ratio (primary metric)
        results_df_sorted = results_df.sort_values('sharpe_ratio', ascending=False)
        
        print("\nTop 5 parameter combinations by Sharpe ratio:")
        print(results_df_sorted.head().to_string(index=False))
        
        # Best parameters
        best_params = results_df_sorted.iloc[0]
        print(f"\n=== BEST PARAMETERS ===")
        print(f"min_rate_diff: {best_params['min_rate_diff']}")
        print(f"momentum_window: {best_params['momentum_window']}")
        print(f"volatility_window: {best_params['volatility_window']}")
        print(f"max_position_size: {best_params['max_position_size']}")
        print(f"transaction_cost: {best_params['transaction_cost']}")
        print(f"\nBest Performance:")
        print(f"Total Return: {best_params['total_return']:.2%}")
        print(f"Annualized Return: {best_params['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {best_params['max_drawdown']:.2%}")
        print(f"Total Trades: {best_params['total_trades']}")
        print(f"Win Rate: {best_params['win_rate']:.1%}")
        
        # Save results
        results_df_sorted.to_csv('results/optimization_results.csv', index=False)
        print(f"\nOptimization results saved to 'results/optimization_results.csv'")
        
        return best_params
    else:
        print("No successful optimization runs completed.")
        return None

def test_optimized_strategy(best_params):
    """Test the optimized strategy with best parameters"""
    
    if best_params is None:
        print("No optimized parameters available.")
        return
    
    print("\n=== TESTING OPTIMIZED STRATEGY ===")
    
    # Load data
    currency_data = pd.read_csv('data/processed/currency_prices.csv', index_col=0, parse_dates=True)
    interest_rates = pd.read_csv('data/processed/interest_rates.csv', index_col=0, parse_dates=True)
    
    # Initialize optimized strategy
    strategy = CarryTradeStrategy(
        min_rate_diff=best_params['min_rate_diff'],
        momentum_window=int(best_params['momentum_window']),
        volatility_window=int(best_params['volatility_window']),
        max_position_size=best_params['max_position_size'],
        transaction_cost=best_params['transaction_cost']
    )
    
    # Generate signals and run full backtest
    signals = strategy.generate_signals(currency_data, interest_rates)
    
    risk_manager = RiskManager()
    returns = risk_manager.calculate_returns(currency_data)
    positions = risk_manager.calculate_position_sizes(
        signals, returns, portfolio_value=1000000, method='risk_parity'
    )
    
    backtest_engine = BacktestEngine(
        initial_capital=1000000,
        transaction_cost_bps=best_params['transaction_cost'] * 10000,
        rebalance_frequency='daily'
    )
    
    portfolio_df, backtest_results = backtest_engine.run_backtest(
        currency_data, signals, positions, interest_rates
    )
    
    # Generate detailed report
    report = backtest_engine.generate_performance_report(portfolio_df, backtest_results, 'results/optimized_performance_report.txt')
    backtest_engine.plot_performance(portfolio_df, 'results/optimized_performance_plots.png')
    
    print("\n=== OPTIMIZED STRATEGY RESULTS ===")
    print(report)
    
    # Save optimized data
    signals.to_csv('data/processed/optimized_trading_signals.csv')
    positions.to_csv('data/processed/optimized_position_sizes.csv')
    portfolio_df.to_csv('results/optimized_portfolio_history.csv')
    
    print("\nOptimized strategy results saved to 'results/' directory")

if __name__ == "__main__":
    # Run optimization
    best_params = optimize_strategy_parameters()
    
    # Test optimized strategy
    test_optimized_strategy(best_params)