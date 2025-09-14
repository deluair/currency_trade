import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import requests
from fredapi import Fred
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

class CurrencyDataFetcher:
    """
    Fetches currency exchange rates and interest rates for G10 currencies.
    G10 currencies: USD, EUR, JPY, GBP, AUD, NZD, CAD, CHF, SEK, NOK
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.g10_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF', 'SEK', 'NOK']
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.logger = self._setup_logger()
        
        # Currency pairs for yfinance (base currency is USD)
        self.currency_pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'AUDUSD': 'AUDUSD=X',
            'NZDUSD': 'NZDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'USDJPY': 'USDJPY=X',
            'USDSEK': 'USDSEK=X',
            'USDNOK': 'USDNOK=X'
        }
        
        # FRED series codes for G10 interest rates (verified working series)
        self.interest_rate_series = {
            'USD': 'FEDFUNDS',           # US Federal Funds Rate
            'EUR': 'ECBDFR',             # ECB Deposit Facility Rate
            'GBP': 'BOEPRUKA',           # Bank of England Policy Rate
            'JPY': 'IRSTCB01JPM156N',    # Japan Central Bank Rate (OECD)
            'CHF': 'IRSTCB01CHM156N',    # Switzerland Central Bank Rate (OECD) - Monthly
            'CAD': 'IRSTCB01CAQ156N',    # Canada Central Bank Rate (OECD)
            'AUD': 'IRSTCB01AUM156N',    # Australia Central Bank Rate (OECD) - Monthly
            'NZD': 'IRSTCB01NZM156N',    # New Zealand Central Bank Rate (OECD) - Monthly
            'SEK': 'IRSTCB01SEM156N',    # Sweden Central Bank Rate (OECD) - Monthly
            'NOK': 'IRSTCB01NOM156N'     # Norway Central Bank Rate (OECD) - Monthly
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
    
    def fetch_currency_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch currency exchange rate data from Yahoo Finance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with currency exchange rates
        """
        self.logger.info(f"Fetching currency data from {start_date} to {end_date}")
        
        currency_data = {}
        
        for pair_name, yahoo_symbol in self.currency_pairs.items():
            try:
                ticker = yf.Ticker(yahoo_symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    currency_data[pair_name] = data['Close']
                    self.logger.info(f"Successfully fetched {pair_name} data: {len(data)} records")
                else:
                    self.logger.warning(f"No data found for {pair_name}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {pair_name}: {str(e)}")
        
        if currency_data:
            df = pd.DataFrame(currency_data)
            df.index.name = 'Date'
            return df.ffill()  # Forward fill missing values
        else:
            return pd.DataFrame()
    
    def fetch_interest_rates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch interest rate data from FRED API.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with interest rates for G10 currencies
        """
        if not self.fred:
            self.logger.error("FRED API key not provided. Cannot fetch interest rate data.")
            return pd.DataFrame()
        
        self.logger.info(f"Fetching interest rate data from {start_date} to {end_date}")
        
        interest_data = {}
        
        for currency, series_id in self.interest_rate_series.items():
            try:
                data = self.fred.get_series(series_id, start=start_date, end=end_date)
                
                if not data.empty:
                    interest_data[currency] = data
                    self.logger.info(f"Successfully fetched {currency} interest rate data: {len(data)} records")
                else:
                    self.logger.warning(f"No interest rate data found for {currency}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching interest rate for {currency}: {str(e)}")
        
        if interest_data:
            df = pd.DataFrame(interest_data)
            df.index.name = 'Date'
            return df.ffill()  # Forward fill missing values
        else:
            return pd.DataFrame()
    
    def calculate_carry_signals(self, currency_df: pd.DataFrame, interest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate carry trade signals based on interest rate differentials.
        
        Args:
            currency_df: DataFrame with currency exchange rates
            interest_df: DataFrame with interest rates
            
        Returns:
            DataFrame with carry trade signals
        """
        self.logger.info("Calculating carry trade signals")
        
        # Align dates between currency and interest rate data
        common_dates = currency_df.index.intersection(interest_df.index)
        
        if len(common_dates) == 0:
            self.logger.error("No common dates between currency and interest rate data")
            return pd.DataFrame()
        
        currency_aligned = currency_df.loc[common_dates]
        interest_aligned = interest_df.loc[common_dates]
        
        carry_signals = pd.DataFrame(index=common_dates)
        
        # Calculate interest rate differentials for each currency pair
        pair_mappings = {
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
        
        for pair, (base, quote) in pair_mappings.items():
            if pair in currency_aligned.columns and base in interest_aligned.columns and quote in interest_aligned.columns:
                # Interest rate differential (base - quote)
                rate_diff = interest_aligned[base] - interest_aligned[quote]
                carry_signals[f'{pair}_rate_diff'] = rate_diff
                
                # Carry signal: positive when base currency has higher interest rate
                carry_signals[f'{pair}_carry_signal'] = np.where(rate_diff > 0, 1, -1)
        
        return carry_signals
    
    def save_data(self, data: pd.DataFrame, filename: str, data_dir: str = 'data/processed') -> None:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
            data_dir: Directory to save the file
        """
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(data_dir) / filename
        
        # Create a copy to avoid modifying the original data
        data_to_save = data.copy()
        
        # Normalize date index to remove timezone info for consistency
        if hasattr(data_to_save.index, 'tz') and data_to_save.index.tz is not None:
            data_to_save.index = data_to_save.index.tz_localize(None)
        
        data_to_save.to_csv(filepath)
        self.logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str, data_dir: str = 'data/processed') -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename: Name of the file
            data_dir: Directory containing the file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = Path(data_dir) / filename
        
        if filepath.exists():
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Normalize date index to remove timezone info for consistency
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Ensure index is datetime type
            data.index = pd.to_datetime(data.index)
            
            self.logger.info(f"Data loaded from {filepath}")
            return data
        else:
            self.logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def get_latest_data(self, days_back: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the latest currency and interest rate data.
        
        Args:
            days_back: Number of days to look back from today
            
        Returns:
            Tuple of (currency_data, interest_rate_data)
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        currency_data = self.fetch_currency_data(start_date, end_date)
        interest_data = self.fetch_interest_rates(start_date, end_date)
        
        return currency_data, interest_data


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Example usage
    fred_api_key = os.getenv('FRED_API_KEY')
    fetcher = CurrencyDataFetcher(fred_api_key=fred_api_key)
    
    # Check if data files already exist
    currency_file = Path('data/processed/currency_prices.csv')
    interest_file = Path('data/processed/interest_rates.csv')
    
    if currency_file.exists() and interest_file.exists():
        print("Data files already exist. Loading existing data...")
        currency_data = fetcher.load_data('currency_prices.csv')
        interest_data = fetcher.load_data('interest_rates.csv')
    else:
        print("Data files not found. Downloading fresh data...")
        # Get latest 30 days of data
        currency_data, interest_data = fetcher.get_latest_data(30)
        
        # Save the data
        if not currency_data.empty:
            fetcher.save_data(currency_data, 'currency_prices.csv')
        if not interest_data.empty:
            fetcher.save_data(interest_data, 'interest_rates.csv')
    
    print("Currency Data Shape:", currency_data.shape)
    print("Interest Rate Data Shape:", interest_data.shape)
    
    if not currency_data.empty and not interest_data.empty:
        carry_signals = fetcher.calculate_carry_signals(currency_data, interest_data)
        print("Carry Signals Shape:", carry_signals.shape)
        print(carry_signals.head())