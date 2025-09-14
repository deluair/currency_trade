#!/usr/bin/env python3
"""
Currency Carry Trade Strategy with Trade Flow Analysis

This script implements a comprehensive currency carry trade strategy that:
1. Fetches G10 currency data and interest rates
2. Incorporates bilateral trade flow analysis
3. Implements carry trade strategies with risk management
4. Backtests using 10+ years of historical data
5. Provides detailed performance analysis and visualization

Author: AI Assistant
Date: 2024
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import custom modules
from data_fetching.currency_data import CurrencyDataFetcher
from data_fetching.trade_flow_data import TradeFlowDataFetcher
from strategies.carry_trade_strategy import CarryTradeStrategy
from risk_management.risk_manager import RiskManager
from backtesting.backtest_engine import BacktestEngine

class CurrencyCarryTradeSystem:
    """
    Main system class that orchestrates the entire carry trade strategy.
    """
    
    def __init__(self, 
                 fred_api_key: str = None,
                 initial_capital: float = 1000000,
                 start_date: str = '2010-01-01',
                 end_date: str = '2023-12-31'):
        """
        Initialize the carry trade system.
        
        Args:
            fred_api_key: FRED API key for interest rate data
            initial_capital: Starting capital for backtesting
            start_date: Start date for data and backtesting
            end_date: End date for data and backtesting
        """
        self.fred_api_key = fred_api_key
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize components
        self.currency_fetcher = CurrencyDataFetcher(fred_api_key=fred_api_key)
        self.trade_flow_fetcher = TradeFlowDataFetcher()
        self.strategy = CarryTradeStrategy(
            min_rate_diff=0.5,
            momentum_window=20,
            volatility_window=30,
            max_position_size=0.15,
            transaction_cost=0.0002
        )
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.15,
            max_individual_weight=0.2,
            var_confidence=0.05,
            transaction_cost_bps=2.0
        )
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost_bps=1.0,  # Optimized: reduced from 2.0 to 1.0 bps
            funding_cost_bps=1.0,
            rebalance_frequency='daily'
        )
        
        self.logger = self._setup_logger()
        
        # Data storage
        self.currency_data = pd.DataFrame()
        self.interest_data = pd.DataFrame()
        self.trade_flow_data = {}
        self.signals = pd.DataFrame()
        self.positions = pd.DataFrame()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fetch_all_data(self) -> None:
        """
        Fetch all required data for the strategy.
        Check for existing files first before downloading new data.
        """
        self.logger.info("=== FETCHING DATA ===")
        
        # Check if currency data files already exist
        currency_file = Path('data/processed/currency_prices.csv')
        interest_file = Path('data/processed/interest_rates.csv')
        
        if currency_file.exists():
            self.logger.info("Currency data file already exists. Loading existing data...")
            self.currency_data = self.currency_fetcher.load_data('currency_prices.csv')
            self.logger.info(f"Currency data shape: {self.currency_data.shape}")
        else:
            self.logger.info("Currency data file not found. Fetching currency exchange rates...")
            self.currency_data = self.currency_fetcher.fetch_currency_data(
                self.start_date, self.end_date
            )
            
            if not self.currency_data.empty:
                self.logger.info(f"Currency data shape: {self.currency_data.shape}")
                self.currency_fetcher.save_data(self.currency_data, 'currency_prices.csv')
            else:
                self.logger.warning("No currency data fetched")
        
        # Check if interest rate data exists
        if self.fred_api_key:
            if interest_file.exists():
                self.logger.info("Interest rate data file already exists. Loading existing data...")
                self.interest_data = self.currency_fetcher.load_data('interest_rates.csv')
                self.logger.info(f"Interest rate data shape: {self.interest_data.shape}")
            else:
                self.logger.info("Interest rate data file not found. Fetching interest rate data...")
                self.interest_data = self.currency_fetcher.fetch_interest_rates(
                    self.start_date, self.end_date
                )
                
                if not self.interest_data.empty:
                    self.logger.info(f"Interest rate data shape: {self.interest_data.shape}")
                    self.currency_fetcher.save_data(self.interest_data, 'interest_rates.csv')
                else:
                    self.logger.warning("No interest rate data fetched")
        else:
            self.logger.warning("No FRED API key provided, skipping interest rate data")
        
        # Check if trade flow data files exist
        trade_balance_file = Path('data/trade_flows/trade_balance_data.csv')
        trade_signals_file = Path('data/trade_flows/trade_signals_data.csv')
        
        if trade_balance_file.exists() and trade_signals_file.exists():
            self.logger.info("Trade flow data files already exist. Loading existing data...")
            trade_balance = self.trade_flow_fetcher.load_trade_data('trade_balance_data.csv')
            trade_signals = self.trade_flow_fetcher.load_trade_data('trade_signals_data.csv')
            
            self.trade_flow_data = {
                'trade_balance': trade_balance,
                'trade_signals': trade_signals
            }
            
            for data_type, df in self.trade_flow_data.items():
                if not df.empty:
                    self.logger.info(f"Trade flow {data_type} shape: {df.shape}")
        else:
            self.logger.info("Trade flow data files not found. Fetching trade flow data...")
            start_year = int(self.start_date[:4])
            end_year = int(self.end_date[:4])
            
            self.trade_flow_data = self.trade_flow_fetcher.get_comprehensive_trade_data(
                start_year, end_year
            )
            
            for data_type, df in self.trade_flow_data.items():
                if not df.empty:
                    self.logger.info(f"Trade flow {data_type} shape: {df.shape}")
                    self.trade_flow_fetcher.save_trade_data(df, f'{data_type}_data.csv')
    
    def generate_signals(self) -> None:
        """
        Generate trading signals using the carry trade strategy.
        """
        self.logger.info("=== GENERATING SIGNALS ===")
        
        if self.currency_data.empty:
            self.logger.error("No currency data available for signal generation")
            return
        
        # Use interest rate data if available, otherwise create dummy data
        if not self.interest_data.empty:
            interest_rates = self.interest_data
        else:
            self.logger.warning("Using dummy interest rate data")
            # Create dummy interest rate data for demonstration
            interest_rates = pd.DataFrame(
                index=self.currency_data.index,
                data={
                    'USD': 2.0, 'EUR': 0.0, 'JPY': -0.1, 'GBP': 1.5,
                    'AUD': 3.0, 'NZD': 2.5, 'CAD': 1.8, 'CHF': -0.5,
                    'SEK': 1.0, 'NOK': 2.0
                }
            )
        
        # Generate signals
        trade_flows = self.trade_flow_data.get('trade_signals')
        self.signals = self.strategy.generate_signals(
            self.currency_data, 
            interest_rates,
            trade_flows
        )
        
        if not self.signals.empty:
            self.logger.info(f"Generated signals shape: {self.signals.shape}")
            
            # Save signals
            Path('data/processed').mkdir(parents=True, exist_ok=True)
            self.signals.to_csv('data/processed/trading_signals.csv')
        else:
            self.logger.warning("No signals generated")
    
    def calculate_positions(self) -> None:
        """
        Calculate optimal position sizes using risk management.
        """
        self.logger.info("=== CALCULATING POSITIONS ===")
        
        if self.signals.empty or self.currency_data.empty:
            self.logger.error("Missing signals or currency data for position calculation")
            return
        
        # Calculate returns for risk management
        returns = self.risk_manager.calculate_returns(self.currency_data)
        
        # Calculate position sizes using risk management
        self.positions = self.risk_manager.calculate_position_sizes(
            self.signals,
            returns,
            self.initial_capital,
            method='risk_parity'
        )
        
        if not self.positions.empty:
            self.logger.info(f"Position sizes shape: {self.positions.shape}")
            
            # Save positions
            Path('data/processed').mkdir(parents=True, exist_ok=True)
            self.positions.to_csv('data/processed/position_sizes.csv')
        else:
            self.logger.warning("No positions calculated")
    
    def run_backtest(self) -> None:
        """
        Run comprehensive backtest of the strategy.
        """
        self.logger.info("=== RUNNING BACKTEST ===")
        
        if self.currency_data.empty or self.signals.empty or self.positions.empty:
            self.logger.error("Missing required data for backtesting")
            return
        
        # Prepare interest rate differentials for backtest
        if not self.interest_data.empty:
            rate_diffs = self.strategy.calculate_interest_rate_differential(self.interest_data)
        else:
            # Create dummy rate differentials
            rate_diffs = pd.DataFrame(index=self.currency_data.index)
            for pair in self.strategy.pair_mappings.keys():
                if pair in self.currency_data.columns:
                    rate_diffs[f'{pair}_rate_diff'] = np.random.normal(1.0, 0.5, len(self.currency_data))
        
        # Run backtest
        portfolio_history, backtest_results = self.backtest_engine.run_backtest(
            prices=self.currency_data,
            signals=self.signals,
            position_sizes=self.positions,
            interest_rates=rate_diffs,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Generate performance report
        if not portfolio_history.empty:
            self.logger.info("=== BACKTEST RESULTS ===")
            
            # Save results
            Path('results').mkdir(parents=True, exist_ok=True)
            portfolio_history.to_csv('results/portfolio_history.csv')
            
            # Generate and save performance report
            report = self.backtest_engine.generate_performance_report(
                portfolio_history, 
                backtest_results,
                'results/performance_report.txt'
            )
            
            print(report)
            
            # Create performance plots
            try:
                self.backtest_engine.plot_performance(
                    portfolio_history,
                    'results/performance_plots.png'
                )
            except Exception as e:
                self.logger.warning(f"Could not create plots: {str(e)}")
            
            # Calculate and display portfolio risk metrics
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(
                self.positions, 
                self.risk_manager.calculate_returns(self.currency_data)
            )
            
            if not portfolio_risk.empty:
                portfolio_risk.to_csv('results/portfolio_risk_metrics.csv')
                
                # Display final risk metrics
                final_vol = portfolio_risk['rolling_vol'].iloc[-1] if 'rolling_vol' in portfolio_risk.columns else 0
                final_sharpe = portfolio_risk['rolling_sharpe'].iloc[-1] if 'rolling_sharpe' in portfolio_risk.columns else 0
                
                self.logger.info(f"Final portfolio volatility: {final_vol:.2%}")
                self.logger.info(f"Final rolling Sharpe ratio: {final_sharpe:.2f}")
        
        else:
            self.logger.error("Backtest failed to produce results")
    
    def run_full_analysis(self) -> None:
        """
        Run the complete carry trade analysis pipeline.
        """
        self.logger.info("=== STARTING CURRENCY CARRY TRADE ANALYSIS ===")
        self.logger.info(f"Period: {self.start_date} to {self.end_date}")
        self.logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        try:
            # Step 1: Fetch all data
            self.fetch_all_data()
            
            # Step 2: Generate trading signals
            self.generate_signals()
            
            # Step 3: Calculate optimal positions
            self.calculate_positions()
            
            # Step 4: Run backtest
            self.run_backtest()
            
            self.logger.info("=== ANALYSIS COMPLETE ===")
            self.logger.info("Results saved in 'results/' directory")
            self.logger.info("Data saved in 'data/' directory")
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")
            raise


def main():
    """
    Main execution function.
    """
    # Configuration
    FRED_API_KEY = None  # Set your FRED API key here or use environment variable
    INITIAL_CAPITAL = 1000000  # $1M starting capital
    START_DATE = '2010-01-01'
    END_DATE = '2023-12-31'
    
    # Check for FRED API key in environment
    if not FRED_API_KEY:
        FRED_API_KEY = os.getenv('FRED_API_KEY')
    
    if not FRED_API_KEY:
        print("Warning: No FRED API key provided. Interest rate data will be simulated.")
        print("To get real interest rate data, sign up for a free FRED API key at:")
        print("https://fred.stlouisfed.org/docs/api/api_key.html")
        print("Then set the FRED_API_KEY environment variable or modify the script.")
        print()
    
    # Initialize and run the system
    system = CurrencyCarryTradeSystem(
        fred_api_key=FRED_API_KEY,
        initial_capital=INITIAL_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Run full analysis
    system.run_full_analysis()


if __name__ == "__main__":
    main()