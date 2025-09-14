import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Data class for risk metrics."""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall: float  # Expected Shortfall (CVaR)
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    skewness: float
    kurtosis: float

class RiskManager:
    """
    Comprehensive risk management system for currency carry trades.
    Includes position sizing, risk metrics, and portfolio optimization.
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.15,
                 max_individual_weight: float = 0.2,
                 var_confidence: float = 0.05,
                 rebalance_threshold: float = 0.05,
                 transaction_cost_bps: float = 2.0):
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio volatility (annualized)
            max_individual_weight: Maximum weight for individual position
            var_confidence: Confidence level for VaR calculation (0.05 = 95% VaR)
            rebalance_threshold: Threshold for rebalancing (5% deviation)
            transaction_cost_bps: Transaction costs in basis points
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_individual_weight = max_individual_weight
        self.var_confidence = var_confidence
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost_bps / 10000  # Convert to decimal
        self.logger = self._setup_logger()
        
        # Risk-free rate (will be updated with actual data)
        self.risk_free_rate = 0.02
    
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
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            DataFrame with returns
        """
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_volatility(self, returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            returns: DataFrame with returns
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility measures
        """
        # Annualized volatility
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.
        
        Args:
            returns: DataFrame with returns
            window: Rolling window for correlation calculation
            
        Returns:
            Latest correlation matrix
        """
        # Use the most recent window for correlation
        recent_returns = returns.tail(window)
        correlation_matrix = recent_returns.corr()
        return correlation_matrix
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """
        Calculate Value at Risk using historical simulation.
        
        Args:
            returns: Series of returns
            confidence: Confidence level (0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        if len(returns) < 30:
            return np.nan
        
        return np.percentile(returns.dropna(), confidence * 100)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        if len(returns) < 30:
            return np.nan
        
        var = self.calculate_var(returns, confidence)
        expected_shortfall = returns[returns <= var].mean()
        return expected_shortfall
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown
        """
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            RiskMetrics object with all risk measures
        """
        if len(returns) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic statistics
        volatility = returns.std() * np.sqrt(252)
        mean_return = returns.mean() * 252
        
        # Risk metrics
        var_95 = self.calculate_var(returns, 0.05)
        var_99 = self.calculate_var(returns, 0.01)
        expected_shortfall = self.calculate_expected_shortfall(returns, 0.05)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Performance ratios
        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def calculate_optimal_weights(self, 
                                expected_returns: pd.Series,
                                covariance_matrix: pd.DataFrame,
                                risk_aversion: float = 3.0) -> pd.Series:
        """
        Calculate optimal portfolio weights using mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion parameter
            
        Returns:
            Series with optimal weights
        """
        n_assets = len(expected_returns)
        
        if n_assets == 0:
            return pd.Series()
        
        # Objective function: maximize utility = expected_return - 0.5 * risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds: each weight between -max_weight and +max_weight
        bounds = [(-self.max_individual_weight, self.max_individual_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
                return optimal_weights
            else:
                self.logger.warning("Optimization failed, using equal weights")
                return pd.Series(x0, index=expected_returns.index)
                
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return pd.Series(x0, index=expected_returns.index)
    
    def calculate_risk_parity_weights(self, covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate risk parity weights where each asset contributes equally to portfolio risk.
        
        Args:
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Series with risk parity weights
        """
        n_assets = len(covariance_matrix)
        
        if n_assets == 0:
            return pd.Series()
        
        # Objective function: minimize sum of squared risk contributions
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds: positive weights only
        bounds = [(0.001, self.max_individual_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                risk_parity_weights = pd.Series(result.x, index=covariance_matrix.index)
                return risk_parity_weights
            else:
                self.logger.warning("Risk parity optimization failed, using equal weights")
                return pd.Series(x0, index=covariance_matrix.index)
                
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {str(e)}")
            return pd.Series(x0, index=covariance_matrix.index)
    
    def calculate_position_sizes(self, 
                               signals: pd.DataFrame,
                               returns: pd.DataFrame,
                               portfolio_value: float = 1000000,
                               method: str = 'risk_parity') -> pd.DataFrame:
        """
        Calculate position sizes based on risk management approach.
        
        Args:
            signals: Trading signals
            returns: Historical returns
            portfolio_value: Total portfolio value
            method: Position sizing method ('risk_parity', 'mean_variance', 'equal_weight')
            
        Returns:
            DataFrame with position sizes
        """
        self.logger.info(f"Calculating position sizes using {method} method")
        
        position_sizes = pd.DataFrame(index=signals.index)
        
        # Get currency pairs from signals
        signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        pairs = [col.replace('_signal', '') for col in signal_cols]
        
        # Filter returns to match available pairs
        available_pairs = [pair for pair in pairs if pair in returns.columns]
        
        if not available_pairs:
            self.logger.warning("No matching pairs found between signals and returns")
            return position_sizes
        
        returns_subset = returns[available_pairs].dropna()
        
        if len(returns_subset) < 30:
            self.logger.warning("Insufficient return data for position sizing")
            return position_sizes
        
        # Calculate covariance matrix
        covariance_matrix = returns_subset.cov() * 252  # Annualized
        
        # Calculate weights based on method
        if method == 'risk_parity':
            weights = self.calculate_risk_parity_weights(covariance_matrix)
        elif method == 'mean_variance':
            expected_returns = returns_subset.mean() * 252  # Annualized
            weights = self.calculate_optimal_weights(expected_returns, covariance_matrix)
        else:  # equal_weight
            weights = pd.Series(1.0 / len(available_pairs), index=available_pairs)
        
        # Apply signals to weights
        for pair in available_pairs:
            signal_col = f'{pair}_signal'
            
            if signal_col in signals.columns:
                # Base position size from weight
                base_size = weights.get(pair, 0) * portfolio_value
                
                # Apply signal direction
                position_size = signals[signal_col] * base_size
                
                # Store position information
                position_sizes[f'{pair}_weight'] = weights.get(pair, 0)
                position_sizes[f'{pair}_position_size'] = position_size
                position_sizes[f'{pair}_notional'] = position_size
        
        return position_sizes
    
    def calculate_portfolio_risk(self, 
                               positions: pd.DataFrame,
                               returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            positions: Position sizes
            returns: Historical returns
            
        Returns:
            DataFrame with portfolio risk metrics
        """
        portfolio_risk = pd.DataFrame(index=positions.index)
        
        # Get position columns
        position_cols = [col for col in positions.columns if col.endswith('_position_size')]
        pairs = [col.replace('_position_size', '') for col in position_cols]
        
        # Calculate portfolio returns
        portfolio_returns = []
        
        for date in positions.index:
            if date in returns.index:
                daily_return = 0
                total_notional = 0
                
                for pair in pairs:
                    position_col = f'{pair}_position_size'
                    
                    if position_col in positions.columns and pair in returns.columns:
                        position = positions.loc[date, position_col]
                        pair_return = returns.loc[date, pair]
                        
                        if not pd.isna(position) and not pd.isna(pair_return):
                            daily_return += position * pair_return
                            total_notional += abs(position)
                
                if total_notional > 0:
                    portfolio_returns.append(daily_return / total_notional)
                else:
                    portfolio_returns.append(0)
            else:
                portfolio_returns.append(0)
        
        portfolio_return_series = pd.Series(portfolio_returns, index=positions.index)
        
        # Calculate risk metrics
        portfolio_risk['portfolio_return'] = portfolio_return_series
        portfolio_risk['cumulative_return'] = (1 + portfolio_return_series).cumprod() - 1
        
        # Rolling risk metrics
        window = min(252, len(portfolio_return_series))
        
        portfolio_risk['rolling_vol'] = portfolio_return_series.rolling(window).std() * np.sqrt(252)
        portfolio_risk['rolling_sharpe'] = (
            portfolio_return_series.rolling(window).mean() * 252 - self.risk_free_rate
        ) / portfolio_risk['rolling_vol']
        
        # VaR and Expected Shortfall
        portfolio_risk['var_95'] = portfolio_return_series.rolling(window).apply(
            lambda x: self.calculate_var(x, 0.05)
        )
        portfolio_risk['expected_shortfall'] = portfolio_return_series.rolling(window).apply(
            lambda x: self.calculate_expected_shortfall(x, 0.05)
        )
        
        return portfolio_risk
    
    def check_risk_limits(self, 
                         positions: pd.DataFrame,
                         portfolio_risk: pd.DataFrame) -> Dict[str, bool]:
        """
        Check if current positions violate risk limits.
        
        Args:
            positions: Current positions
            portfolio_risk: Portfolio risk metrics
            
        Returns:
            Dictionary with risk limit checks
        """
        risk_checks = {
            'portfolio_vol_ok': True,
            'individual_weights_ok': True,
            'var_ok': True,
            'drawdown_ok': True
        }
        
        # Check portfolio volatility
        current_vol = portfolio_risk['rolling_vol'].iloc[-1] if not portfolio_risk.empty else 0
        if current_vol > self.max_portfolio_risk:
            risk_checks['portfolio_vol_ok'] = False
            self.logger.warning(f"Portfolio volatility {current_vol:.2%} exceeds limit {self.max_portfolio_risk:.2%}")
        
        # Check individual position weights
        weight_cols = [col for col in positions.columns if col.endswith('_weight')]
        if weight_cols:
            max_weight = positions[weight_cols].abs().max().max()
            if max_weight > self.max_individual_weight:
                risk_checks['individual_weights_ok'] = False
                self.logger.warning(f"Individual weight {max_weight:.2%} exceeds limit {self.max_individual_weight:.2%}")
        
        # Check VaR
        current_var = portfolio_risk['var_95'].iloc[-1] if not portfolio_risk.empty else 0
        if current_var < -0.05:  # 5% daily VaR limit
            risk_checks['var_ok'] = False
            self.logger.warning(f"VaR {current_var:.2%} exceeds 5% daily limit")
        
        return risk_checks


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager(
        max_portfolio_risk=0.15,
        max_individual_weight=0.2,
        var_confidence=0.05,
        transaction_cost_bps=2.0
    )
    
    print("Risk Manager initialized successfully")
    print(f"Maximum portfolio risk: {risk_manager.max_portfolio_risk:.1%}")
    print(f"Maximum individual weight: {risk_manager.max_individual_weight:.1%}")
    print(f"Transaction cost: {risk_manager.transaction_cost * 10000:.1f} bps")