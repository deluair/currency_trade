import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class BacktestResults:
    """Data class for backtest results."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    transaction_costs: float
    start_date: str
    end_date: str
    
class BacktestEngine:
    """
    Comprehensive backtesting engine for currency carry trade strategies.
    Supports multiple strategies, transaction costs, and detailed performance analysis.
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 transaction_cost_bps: float = 2.0,
                 funding_cost_bps: float = 1.0,
                 rebalance_frequency: str = 'daily',
                 commission_per_trade: float = 0.0):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_bps: Transaction costs in basis points
            funding_cost_bps: Funding costs in basis points
            rebalance_frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')
            commission_per_trade: Fixed commission per trade
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10000
        self.funding_cost = funding_cost_bps / 10000
        self.rebalance_frequency = rebalance_frequency
        self.commission_per_trade = commission_per_trade
        self.logger = self._setup_logger()
        
        # Backtest state
        self.current_positions = {}
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.trade_log = []
        self.portfolio_history = []
        
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
    
    def _should_rebalance(self, current_date: datetime, last_rebalance: datetime) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            current_date: Current date
            last_rebalance: Last rebalancing date
            
        Returns:
            Boolean indicating if rebalancing is needed
        """
        if self.rebalance_frequency == 'daily':
            return True
        elif self.rebalance_frequency == 'weekly':
            return (current_date - last_rebalance).days >= 7
        elif self.rebalance_frequency == 'monthly':
            return current_date.month != last_rebalance.month
        else:
            return True
    
    def _calculate_transaction_costs(self, 
                                   old_positions: Dict[str, float],
                                   new_positions: Dict[str, float],
                                   prices: Dict[str, float]) -> float:
        """
        Calculate transaction costs for position changes.
        
        Args:
            old_positions: Previous positions
            new_positions: New target positions
            prices: Current prices
            
        Returns:
            Total transaction costs
        """
        total_costs = 0.0
        
        all_pairs = set(list(old_positions.keys()) + list(new_positions.keys()))
        
        for pair in all_pairs:
            old_pos = old_positions.get(pair, 0.0)
            new_pos = new_positions.get(pair, 0.0)
            price = prices.get(pair, 1.0)
            
            # Calculate position change
            position_change = abs(new_pos - old_pos)
            
            if position_change > 0:
                # Transaction cost as percentage of traded amount
                trade_value = position_change * price
                cost = trade_value * self.transaction_cost
                total_costs += cost
                
                # Add fixed commission if applicable
                if self.commission_per_trade > 0:
                    total_costs += self.commission_per_trade
        
        return total_costs
    
    def _calculate_funding_costs(self, 
                               positions: Dict[str, float],
                               interest_rates: Dict[str, float],
                               prices: Dict[str, float]) -> float:
        """
        Calculate daily funding costs for carry positions.
        
        Args:
            positions: Current positions
            interest_rates: Interest rate differentials
            prices: Current prices
            
        Returns:
            Daily funding costs/income
        """
        total_funding = 0.0
        
        for pair, position in positions.items():
            if position != 0 and pair in interest_rates and pair in prices:
                # Daily interest rate differential
                daily_rate = interest_rates[pair] / 365 / 100  # Convert to daily decimal
                
                # Funding income/cost
                notional_value = abs(position) * prices[pair]
                funding = position * daily_rate * notional_value
                total_funding += funding
        
        return total_funding
    
    def _update_portfolio_value(self, 
                              positions: Dict[str, float],
                              prices: Dict[str, float],
                              price_changes: Dict[str, float]) -> float:
        """
        Update portfolio value based on position P&L.
        
        Args:
            positions: Current positions
            prices: Current prices
            price_changes: Price changes from previous day
            
        Returns:
            Portfolio P&L for the day
        """
        daily_pnl = 0.0
        
        for pair, position in positions.items():
            if position != 0 and pair in price_changes:
                # P&L from price movement
                pnl = position * price_changes[pair]
                daily_pnl += pnl
        
        return daily_pnl
    
    def run_backtest(self, 
                    prices: pd.DataFrame,
                    signals: pd.DataFrame,
                    position_sizes: pd.DataFrame,
                    interest_rates: Optional[pd.DataFrame] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Tuple[pd.DataFrame, BacktestResults]:
        """
        Run comprehensive backtest.
        
        Args:
            prices: Currency price data
            signals: Trading signals
            position_sizes: Position sizing data
            interest_rates: Interest rate differentials
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Tuple of (portfolio_history, backtest_results)
        """
        self.logger.info("Starting backtest")
        
        # Reset backtest state
        self.current_positions = {}
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.trade_log = []
        self.portfolio_history = []
        
        # Align data
        common_dates = prices.index.intersection(signals.index).intersection(position_sizes.index)
        
        if start_date:
            common_dates = common_dates[common_dates >= start_date]
        if end_date:
            common_dates = common_dates[common_dates <= end_date]
        
        if len(common_dates) == 0:
            self.logger.error("No common dates found in input data")
            return pd.DataFrame(), BacktestResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', '')
        
        common_dates = sorted(common_dates)
        self.logger.info(f"Backtesting from {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} days)")
        
        # Get currency pairs
        signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        pairs = [col.replace('_signal', '') for col in signal_cols]
        available_pairs = [pair for pair in pairs if pair in prices.columns]
        
        last_rebalance = common_dates[0]
        previous_prices = {}
        
        for i, date in enumerate(common_dates):
            # Get current market data
            current_prices = {pair: prices.loc[date, pair] for pair in available_pairs 
                            if not pd.isna(prices.loc[date, pair])}
            
            current_signals = {pair: signals.loc[date, f'{pair}_signal'] for pair in available_pairs 
                             if f'{pair}_signal' in signals.columns and not pd.isna(signals.loc[date, f'{pair}_signal'])}
            
            current_position_sizes = {pair: position_sizes.loc[date, f'{pair}_position_size'] 
                                    for pair in available_pairs 
                                    if f'{pair}_position_size' in position_sizes.columns 
                                    and not pd.isna(position_sizes.loc[date, f'{pair}_position_size'])}
            
            # Calculate price changes
            price_changes = {}
            if previous_prices:
                for pair in current_prices:
                    if pair in previous_prices:
                        price_changes[pair] = current_prices[pair] - previous_prices[pair]
            
            # Update portfolio value from existing positions
            if price_changes and self.current_positions:
                daily_pnl = self._update_portfolio_value(self.current_positions, current_prices, price_changes)
                self.portfolio_value += daily_pnl
            
            # Calculate funding costs/income
            if interest_rates is not None and date in interest_rates.index:
                current_rates = {pair: interest_rates.loc[date, f'{pair}_rate_diff'] 
                               for pair in available_pairs 
                               if f'{pair}_rate_diff' in interest_rates.columns 
                               and not pd.isna(interest_rates.loc[date, f'{pair}_rate_diff'])}
                
                funding_pnl = self._calculate_funding_costs(self.current_positions, current_rates, current_prices)
                self.portfolio_value += funding_pnl
            
            # Rebalancing logic
            should_rebalance = self._should_rebalance(date, last_rebalance)
            
            if should_rebalance and current_signals and current_position_sizes:
                # Calculate new target positions
                new_positions = {}
                for pair in available_pairs:
                    if pair in current_signals and pair in current_position_sizes:
                        signal = current_signals[pair]
                        size = current_position_sizes[pair]
                        
                        # Apply signal to position size
                        target_position = signal * abs(size) if not pd.isna(signal) and not pd.isna(size) else 0
                        new_positions[pair] = target_position
                
                # Calculate transaction costs
                transaction_costs = self._calculate_transaction_costs(
                    self.current_positions, new_positions, current_prices
                )
                
                # Log trades (before updating current_positions)
                for pair, new_pos in new_positions.items():
                    old_pos = self.current_positions.get(pair, 0)
                    if abs(new_pos - old_pos) > 1e-6:  # Significant position change
                        self.trade_log.append({
                            'date': date,
                            'pair': pair,
                            'action': 'buy' if new_pos > old_pos else 'sell',
                            'quantity': abs(new_pos - old_pos),
                            'price': current_prices.get(pair, 0),
                            'transaction_cost': transaction_costs / len(new_positions) if new_positions else 0
                        })
                
                # Update positions and cash
                self.portfolio_value -= transaction_costs
                self.current_positions = new_positions.copy()
                last_rebalance = date
            
            # Record portfolio state
            total_exposure = sum(abs(pos) * current_prices.get(pair, 1) 
                               for pair, pos in self.current_positions.items())
            
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'total_exposure': total_exposure,
                'cash': self.portfolio_value - total_exposure,
                'num_positions': len([pos for pos in self.current_positions.values() if abs(pos) > 1e-6])
            })
            
            previous_prices = current_prices.copy()
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate backtest results
        results = self._calculate_backtest_results(portfolio_df, common_dates[0], common_dates[-1])
        
        self.logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
        
        return portfolio_df, results
    
    def _calculate_backtest_results(self, 
                                  portfolio_df: pd.DataFrame,
                                  start_date: datetime,
                                  end_date: datetime) -> BacktestResults:
        """
        Calculate comprehensive backtest performance metrics.
        
        Args:
            portfolio_df: Portfolio history DataFrame
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResults object
        """
        if portfolio_df.empty:
            return BacktestResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, str(start_date), str(end_date))
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = portfolio_df['returns'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_df['returns'][portfolio_df['returns'] < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(portfolio_df['returns'].dropna(), 5)
        expected_shortfall = portfolio_df['returns'][portfolio_df['returns'] <= var_95].mean()
        
        # Trade statistics
        trade_df = pd.DataFrame(self.trade_log)
        total_trades = len(trade_df)
        
        # Win rate
        if not trade_df.empty:
            # Group trades by pair and calculate P&L
            winning_trades = 0
            total_pnl = 0
            positive_pnl = 0
            negative_pnl = 0
            
            for pair in trade_df['pair'].unique():
                pair_trades = trade_df[trade_df['pair'] == pair].sort_values('date')
                
                for i in range(0, len(pair_trades), 2):  # Assuming buy/sell pairs
                    if i + 1 < len(pair_trades):
                        buy_trade = pair_trades.iloc[i]
                        sell_trade = pair_trades.iloc[i + 1]
                        
                        if buy_trade['action'] == 'buy' and sell_trade['action'] == 'sell':
                            pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['quantity']
                            total_pnl += pnl
                            
                            if pnl > 0:
                                winning_trades += 1
                                positive_pnl += pnl
                            else:
                                negative_pnl += abs(pnl)
            
            win_rate = winning_trades / (total_trades / 2) if total_trades > 0 else 0
            profit_factor = positive_pnl / negative_pnl if negative_pnl > 0 else float('inf')
            
            # Average trade duration (simplified)
            avg_trade_duration = (end_date - start_date).days / (total_trades / 2) if total_trades > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # Transaction costs
        total_transaction_costs = sum(trade['transaction_cost'] for trade in self.trade_log)
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            transaction_costs=total_transaction_costs,
            start_date=str(start_date),
            end_date=str(end_date)
        )
    
    def generate_performance_report(self, 
                                  portfolio_df: pd.DataFrame,
                                  results: BacktestResults,
                                  save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio_df: Portfolio history DataFrame
            results: Backtest results
            save_path: Optional path to save the report
            
        Returns:
            Performance report as string
        """
        report = f"""
=== CURRENCY CARRY TRADE BACKTEST RESULTS ===

Backtest Period: {results.start_date} to {results.end_date}
Initial Capital: ${self.initial_capital:,.2f}

--- PERFORMANCE METRICS ---
Total Return: {results.total_return:.2%}
Annualized Return: {results.annualized_return:.2%}
Volatility: {results.volatility:.2%}
Sharpe Ratio: {results.sharpe_ratio:.2f}
Sortino Ratio: {results.sortino_ratio:.2f}

--- RISK METRICS ---
Maximum Drawdown: {results.max_drawdown:.2%}
95% VaR (Daily): {results.var_95:.2%}
Expected Shortfall: {results.expected_shortfall:.2%}

--- TRADING STATISTICS ---
Total Trades: {results.total_trades}
Win Rate: {results.win_rate:.1%}
Profit Factor: {results.profit_factor:.2f}
Average Trade Duration: {results.avg_trade_duration:.1f} days
Total Transaction Costs: ${results.transaction_costs:,.2f}

--- FINAL PORTFOLIO VALUE ---
Final Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}
Total P&L: ${portfolio_df['portfolio_value'].iloc[-1] - self.initial_capital:,.2f}
        """
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Performance report saved to {save_path}")
        
        return report
    
    def plot_performance(self, 
                        portfolio_df: pd.DataFrame,
                        save_path: Optional[str] = None) -> None:
        """
        Create performance visualization plots.
        
        Args:
            portfolio_df: Portfolio history DataFrame
            save_path: Optional path to save the plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Cumulative returns
        portfolio_df['cumulative_returns'] = (portfolio_df['portfolio_value'] / self.initial_capital) - 1
        axes[0, 1].plot(portfolio_df.index, portfolio_df['cumulative_returns'] * 100)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Returns (%)')
        axes[0, 1].grid(True)
        
        # Drawdown
        cumulative = portfolio_df['portfolio_value'] / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[1, 0].fill_between(portfolio_df.index, drawdown * 100, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True)
        
        # Rolling Sharpe ratio
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        rolling_sharpe = (portfolio_df['returns'].rolling(252).mean() * 252 - 0.02) / (portfolio_df['returns'].rolling(252).std() * np.sqrt(252))
        axes[1, 1].plot(portfolio_df.index, rolling_sharpe)
        axes[1, 1].set_title('Rolling 1-Year Sharpe Ratio')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    backtest_engine = BacktestEngine(
        initial_capital=1000000,
        transaction_cost_bps=2.0,
        funding_cost_bps=1.0,
        rebalance_frequency='daily'
    )
    
    print("Backtest Engine initialized successfully")
    print(f"Initial capital: ${backtest_engine.initial_capital:,.2f}")
    print(f"Transaction cost: {backtest_engine.transaction_cost * 10000:.1f} bps")
    print(f"Rebalance frequency: {backtest_engine.rebalance_frequency}")