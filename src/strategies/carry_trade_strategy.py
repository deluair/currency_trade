import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradeSignal:
    """Data class for trade signals."""
    currency_pair: str
    signal: int  # 1 for long, -1 for short, 0 for no position
    confidence: float  # 0 to 1
    expected_return: float
    risk_score: float
    timestamp: datetime

class CarryTradeStrategy:
    """
    Implements currency carry trade strategy with advanced features:
    - Interest rate differential analysis
    - Momentum and mean reversion filters
    - Risk-adjusted position sizing
    - Transaction cost consideration
    """
    
    def __init__(self, 
                 min_rate_diff: float = 0.25,
                 momentum_window: int = 20,
                 volatility_window: int = 30,
                 max_position_size: float = 0.15,
                 transaction_cost: float = 0.0001):
        """
        Initialize carry trade strategy with optimized parameters.
        
        Args:
            min_rate_diff: Minimum interest rate differential to consider (in %) (optimized: 0.25)
            momentum_window: Window for momentum calculation (optimized: 20)
            volatility_window: Window for volatility calculation (optimized: 30)
            max_position_size: Maximum position size as fraction of portfolio (optimized: 0.15)
            transaction_cost: Transaction cost as fraction (optimized: 0.0001 = 1 bps)
        """
        self.min_rate_diff = min_rate_diff
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.logger = self._setup_logger()
        
        # Currency pair mappings
        self.pair_mappings = {
            'EURUSD': ('EUR', 'USD'),
            'GBPUSD': ('GBP', 'USD'),
            'AUDUSD': ('AUD', 'USD'),
            'NZDUSD': ('NZD', 'USD'),
            'USDCAD': ('USD', 'CAD'),
            'USDCHF': ('USD', 'CHF'),
            'USDJPY': ('USD', 'JPY'),
            'USDSEK': ('USD', 'SEK'),
            'USDNOK': ('USD', 'NOK')
        }
    
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
    
    def calculate_interest_rate_differential(self, 
                                           interest_rates: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interest rate differentials for all currency pairs.
        
        Args:
            interest_rates: DataFrame with interest rates by currency
            
        Returns:
            DataFrame with interest rate differentials
        """
        rate_diffs = pd.DataFrame(index=interest_rates.index)
        
        for pair, (base, quote) in self.pair_mappings.items():
            if base in interest_rates.columns and quote in interest_rates.columns:
                rate_diffs[pair] = interest_rates[base] - interest_rates[quote]
        
        return rate_diffs
    
    def calculate_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price momentum for currency pairs.
        
        Args:
            prices: DataFrame with currency prices
            
        Returns:
            DataFrame with momentum indicators
        """
        momentum = pd.DataFrame(index=prices.index)
        
        for pair in prices.columns:
            if pair in self.pair_mappings:
                # Calculate returns
                returns = prices[pair].pct_change()
                
                # Momentum as cumulative return over window
                momentum[f'{pair}_momentum'] = returns.rolling(
                    window=self.momentum_window
                ).sum()
                
                # Momentum strength (absolute value)
                momentum[f'{pair}_momentum_strength'] = abs(
                    momentum[f'{pair}_momentum']
                )
        
        return momentum
    
    def calculate_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility for currency pairs.
        
        Args:
            prices: DataFrame with currency prices
            
        Returns:
            DataFrame with volatility measures
        """
        volatility = pd.DataFrame(index=prices.index)
        
        for pair in prices.columns:
            if pair in self.pair_mappings:
                returns = prices[pair].pct_change()
                
                # Rolling volatility (annualized)
                vol = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
                volatility[f'{pair}_volatility'] = vol
                
                # Volatility rank (percentile over last year)
                volatility[f'{pair}_vol_rank'] = vol.rolling(
                    window=252, min_periods=30
                ).rank(pct=True)
        
        return volatility
    
    def calculate_carry_score(self, 
                            rate_diffs: pd.DataFrame,
                            momentum: pd.DataFrame,
                            volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive carry trade scores.
        
        Args:
            rate_diffs: Interest rate differentials
            momentum: Momentum indicators
            volatility: Volatility measures
            
        Returns:
            DataFrame with carry trade scores
        """
        scores = pd.DataFrame(index=rate_diffs.index)
        
        for pair in self.pair_mappings.keys():
            if pair in rate_diffs.columns:
                # Base carry score from interest rate differential
                base_score = rate_diffs[pair] / 10.0  # Normalize to 0-1 range
                
                # Momentum adjustment
                momentum_col = f'{pair}_momentum'
                if momentum_col in momentum.columns:
                    momentum_adj = np.tanh(momentum[momentum_col] * 10)  # Bounded adjustment
                    base_score = base_score * (1 + 0.3 * momentum_adj)
                
                # Volatility adjustment (penalize high volatility)
                vol_col = f'{pair}_volatility'
                if vol_col in volatility.columns:
                    vol_penalty = 1 - (volatility[vol_col] / volatility[vol_col].rolling(252).max())
                    base_score = base_score * vol_penalty.fillna(1)
                
                scores[f'{pair}_carry_score'] = base_score
        
        return scores
    
    def generate_signals(self, 
                        prices: pd.DataFrame,
                        interest_rates: pd.DataFrame,
                        trade_flows: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate carry trade signals.
        
        Args:
            prices: Currency price data
            interest_rates: Interest rate data
            trade_flows: Optional trade flow data for additional filtering
            
        Returns:
            DataFrame with trade signals
        """
        self.logger.info("Generating carry trade signals")
        
        # Calculate components
        rate_diffs = self.calculate_interest_rate_differential(interest_rates)
        momentum = self.calculate_momentum(prices)
        volatility = self.calculate_volatility(prices)
        
        # Align all DataFrames to the prices index (common date range)
        common_index = prices.index
        rate_diffs = rate_diffs.reindex(common_index)
        momentum = momentum.reindex(common_index)
        volatility = volatility.reindex(common_index)
        
        carry_scores = self.calculate_carry_score(rate_diffs, momentum, volatility)
        
        # Generate signals
        signals = pd.DataFrame(index=common_index)
        
        for pair in self.pair_mappings.keys():
            score_col = f'{pair}_carry_score'
            if score_col in carry_scores.columns and pair in rate_diffs.columns:
                
                # Basic signal from carry score
                raw_signal = np.where(
                    carry_scores[score_col] > self.min_rate_diff / 100, 1,
                    np.where(carry_scores[score_col] < -self.min_rate_diff / 100, -1, 0)
                )
                
                # Apply filters
                filtered_signal = self._apply_filters(
                    raw_signal, 
                    prices[pair] if pair in prices.columns else None,
                    volatility[f'{pair}_volatility'] if f'{pair}_volatility' in volatility.columns else None
                )
                
                signals[f'{pair}_signal'] = filtered_signal
                signals[f'{pair}_score'] = carry_scores[score_col]
                signals[f'{pair}_rate_diff'] = rate_diffs[pair]
        
        return signals
    
    def _apply_filters(self, 
                      raw_signal: np.ndarray,
                      prices: Optional[pd.Series],
                      volatility: Optional[pd.Series]) -> np.ndarray:
        """
        Apply filters to raw signals.
        
        Args:
            raw_signal: Raw trading signals
            prices: Price data for the currency pair
            volatility: Volatility data for the currency pair
            
        Returns:
            Filtered signals
        """
        filtered_signal = raw_signal.copy()
        
        if prices is not None and volatility is not None:
            # Ensure all arrays have the same length
            min_length = min(len(filtered_signal), len(prices), len(volatility))
            
            # Truncate arrays to the same length
            filtered_signal = filtered_signal[:min_length]
            prices_aligned = prices.iloc[:min_length]
            volatility_aligned = volatility.iloc[:min_length]
            
            # Filter out signals during high volatility periods
            high_vol_threshold = volatility_aligned.rolling(252, min_periods=1).quantile(0.8)
            high_vol_mask = volatility_aligned > high_vol_threshold
            
            # Only apply mask where it's not NaN
            valid_mask = ~high_vol_mask.isna()
            filtered_signal[valid_mask & high_vol_mask] = 0
            
            # Filter out signals during trending markets (momentum too strong)
            returns = prices_aligned.pct_change()
            strong_trend = abs(returns.rolling(10, min_periods=1).sum()) > 0.05  # 5% move in 10 days
            
            # Only apply mask where it's not NaN
            valid_trend_mask = ~strong_trend.isna()
            filtered_signal[valid_trend_mask & strong_trend] = 0
        
        return filtered_signal
    
    def calculate_position_sizes(self, 
                               signals: pd.DataFrame,
                               volatility: pd.DataFrame,
                               portfolio_value: float = 1000000) -> pd.DataFrame:
        """
        Calculate optimal position sizes using risk-based approach.
        
        Args:
            signals: Trading signals
            volatility: Volatility measures
            portfolio_value: Total portfolio value
            
        Returns:
            DataFrame with position sizes
        """
        positions = pd.DataFrame(index=signals.index)
        
        for pair in self.pair_mappings.keys():
            signal_col = f'{pair}_signal'
            vol_col = f'{pair}_volatility'
            
            if signal_col in signals.columns and vol_col in volatility.columns:
                # Risk-adjusted position sizing
                target_vol = 0.15  # 15% annual volatility target
                vol_adj = target_vol / volatility[vol_col].fillna(target_vol)
                
                # Base position size
                base_size = self.max_position_size * vol_adj
                
                # Apply signal
                position_size = signals[signal_col] * base_size
                
                # Cap position size
                position_size = np.clip(position_size, -self.max_position_size, self.max_position_size)
                
                positions[f'{pair}_position'] = position_size
                positions[f'{pair}_notional'] = position_size * portfolio_value
        
        return positions
    
    def calculate_expected_returns(self, 
                                 signals: pd.DataFrame,
                                 positions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate expected returns for carry trade positions.
        
        Args:
            signals: Trading signals with rate differentials
            positions: Position sizes
            
        Returns:
            DataFrame with expected returns
        """
        expected_returns = pd.DataFrame(index=signals.index)
        
        for pair in self.pair_mappings.keys():
            rate_diff_col = f'{pair}_rate_diff'
            position_col = f'{pair}_position'
            
            if (rate_diff_col in signals.columns and 
                position_col in positions.columns):
                
                # Annual carry return (interest rate differential)
                annual_carry = signals[rate_diff_col] / 100
                
                # Daily carry return
                daily_carry = annual_carry / 252
                
                # Expected return adjusted for position size
                expected_return = daily_carry * positions[position_col]
                
                # Subtract transaction costs
                position_changes = positions[position_col].diff().abs()
                transaction_costs = position_changes * self.transaction_cost
                
                expected_returns[f'{pair}_expected_return'] = expected_return - transaction_costs
        
        return expected_returns
    
    def get_portfolio_summary(self, 
                            signals: pd.DataFrame,
                            positions: pd.DataFrame,
                            expected_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio-level summary statistics.
        
        Args:
            signals: Trading signals
            positions: Position sizes
            expected_returns: Expected returns
            
        Returns:
            DataFrame with portfolio summary
        """
        summary = pd.DataFrame(index=signals.index)
        
        # Total portfolio exposure
        position_cols = [col for col in positions.columns if col.endswith('_position')]
        summary['total_exposure'] = positions[position_cols].abs().sum(axis=1)
        
        # Expected portfolio return
        return_cols = [col for col in expected_returns.columns if col.endswith('_expected_return')]
        summary['expected_return'] = expected_returns[return_cols].sum(axis=1)
        
        # Number of active positions
        summary['active_positions'] = (positions[position_cols] != 0).sum(axis=1)
        
        # Risk metrics
        summary['portfolio_risk'] = np.sqrt(
            (expected_returns[return_cols] ** 2).sum(axis=1)
        )
        
        return summary


if __name__ == "__main__":
    # Example usage
    strategy = CarryTradeStrategy(
        min_rate_diff=0.25,
        momentum_window=20,
        volatility_window=30,
        max_position_size=0.15,
        transaction_cost=0.0001
    )
    
    print("Carry Trade Strategy initialized successfully")
    print(f"Minimum rate differential: {strategy.min_rate_diff}%")
    print(f"Maximum position size: {strategy.max_position_size * 100}%")
    print(f"Transaction cost: {strategy.transaction_cost * 10000} bps")