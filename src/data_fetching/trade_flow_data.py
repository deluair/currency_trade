import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import time

class TradeFlowDataFetcher:
    """
    Fetches and processes bilateral trade flow data for G10 countries.
    Uses various data sources including OECD, World Bank, and national statistics.
    """
    
    def __init__(self):
        self.g10_countries = {
            'USD': 'USA',
            'EUR': 'DEU',  # Using Germany as proxy for Eurozone
            'JPY': 'JPN',
            'GBP': 'GBR',
            'AUD': 'AUS',
            'NZD': 'NZL',
            'CAD': 'CAN',
            'CHF': 'CHE',
            'SEK': 'SWE',
            'NOK': 'NOR'
        }
        
        self.country_codes_iso3 = {
            'USA': 'United States',
            'DEU': 'Germany',
            'JPN': 'Japan',
            'GBR': 'United Kingdom',
            'AUS': 'Australia',
            'NZL': 'New Zealand',
            'CAN': 'Canada',
            'CHE': 'Switzerland',
            'SWE': 'Sweden',
            'NOR': 'Norway'
        }
        
        self.logger = self._setup_logger()
        
        # OECD API base URL
        self.oecd_base_url = "https://stats.oecd.org/SDMX-JSON/data"
        
        # World Bank API base URL
        self.wb_base_url = "https://api.worldbank.org/v2"
    
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
    
    def fetch_oecd_trade_data(self, 
                             start_year: int = 2010, 
                             end_year: int = 2023) -> pd.DataFrame:
        """
        Fetch bilateral trade data from OECD.
        
        Args:
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with bilateral trade flows
        """
        self.logger.info(f"Fetching OECD trade data from {start_year} to {end_year}")
        
        trade_data = []
        
        # OECD International Trade by Commodity Statistics (ITCS)
        dataset = "ITCS"
        
        for reporter in self.g10_countries.values():
            for partner in self.g10_countries.values():
                if reporter != partner:
                    try:
                        # Construct OECD API URL
                        url = f"{self.oecd_base_url}/{dataset}/{reporter}.{partner}.TOT.IMP.VAL/all"
                        
                        params = {
                            'startTime': start_year,
                            'endTime': end_year,
                            'dimensionAtObservation': 'allDimensions'
                        }
                        
                        response = requests.get(url, params=params, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Parse OECD SDMX-JSON format
                            if 'dataSets' in data and len(data['dataSets']) > 0:
                                observations = data['dataSets'][0].get('observations', {})
                                
                                for obs_key, obs_data in observations.items():
                                    if obs_data and len(obs_data) > 0:
                                        value = obs_data[0]
                                        
                                        # Extract time dimension
                                        dimensions = obs_key.split(':')
                                        if len(dimensions) >= 4:
                                            time_idx = int(dimensions[3])
                                            time_values = data['structure']['dimensions']['observation'][3]['values']
                                            
                                            if time_idx < len(time_values):
                                                year = time_values[time_idx]['id']
                                                
                                                trade_data.append({
                                                    'reporter': reporter,
                                                    'partner': partner,
                                                    'year': int(year),
                                                    'trade_value': float(value) if value else 0
                                                })
                        
                        time.sleep(0.1)  # Rate limiting
                        
                    except Exception as e:
                        self.logger.warning(f"Error fetching OECD data for {reporter}-{partner}: {str(e)}")
                        continue
        
        if trade_data:
            df = pd.DataFrame(trade_data)
            return df
        else:
            self.logger.warning("No OECD trade data retrieved")
            return pd.DataFrame()
    
    def fetch_worldbank_trade_data(self, 
                                  start_year: int = 2010, 
                                  end_year: int = 2023) -> pd.DataFrame:
        """
        Fetch trade data from World Bank API.
        
        Args:
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with trade statistics
        """
        self.logger.info(f"Fetching World Bank trade data from {start_year} to {end_year}")
        
        trade_indicators = {
            'NE.EXP.GNFS.CD': 'exports_goods_services',
            'NE.IMP.GNFS.CD': 'imports_goods_services',
            'TX.VAL.MRCH.CD.WT': 'merchandise_exports',
            'TM.VAL.MRCH.CD.WT': 'merchandise_imports'
        }
        
        trade_data = []
        
        for country_code, country_name in self.country_codes_iso3.items():
            for indicator_code, indicator_name in trade_indicators.items():
                try:
                    url = f"{self.wb_base_url}/country/{country_code}/indicator/{indicator_code}"
                    
                    params = {
                        'date': f"{start_year}:{end_year}",
                        'format': 'json',
                        'per_page': 1000
                    }
                    
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if len(data) > 1 and data[1]:  # World Bank returns metadata in first element
                            for record in data[1]:
                                if record['value'] is not None:
                                    trade_data.append({
                                        'country': country_name,
                                        'country_code': country_code,
                                        'indicator': indicator_name,
                                        'year': int(record['date']),
                                        'value': float(record['value'])
                                    })
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching World Bank data for {country_code}-{indicator_code}: {str(e)}")
                    continue
        
        if trade_data:
            df = pd.DataFrame(trade_data)
            return df
        else:
            self.logger.warning("No World Bank trade data retrieved")
            return pd.DataFrame()
    
    def calculate_trade_balance(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trade balance indicators.
        
        Args:
            trade_data: DataFrame with trade data
            
        Returns:
            DataFrame with trade balance metrics
        """
        self.logger.info("Calculating trade balance indicators")
        
        # Check if required columns exist
        required_columns = ['country', 'country_code', 'year', 'indicator', 'value']
        missing_columns = [col for col in required_columns if col not in trade_data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            self.logger.error(f"Available columns: {list(trade_data.columns)}")
            return pd.DataFrame()
        
        if trade_data.empty:
            self.logger.warning("Empty trade data provided")
            return pd.DataFrame()
        
        # Pivot data to have exports and imports as columns
        try:
            pivot_data = trade_data.pivot_table(
                index=['country', 'country_code', 'year'],
                columns='indicator',
                values='value',
                aggfunc='first'
            ).reset_index()
        except Exception as e:
            self.logger.error(f"Error pivoting trade data: {str(e)}")
            return pd.DataFrame()
        
        # Calculate trade balance
        if 'exports_goods_services' in pivot_data.columns and 'imports_goods_services' in pivot_data.columns:
            pivot_data['trade_balance'] = (pivot_data['exports_goods_services'] - 
                                         pivot_data['imports_goods_services'])
            
            pivot_data['trade_balance_ratio'] = (pivot_data['trade_balance'] / 
                                               (pivot_data['exports_goods_services'] + 
                                                pivot_data['imports_goods_services']))
        
        # Calculate merchandise trade balance
        if 'merchandise_exports' in pivot_data.columns and 'merchandise_imports' in pivot_data.columns:
            pivot_data['merchandise_trade_balance'] = (pivot_data['merchandise_exports'] - 
                                                     pivot_data['merchandise_imports'])
        
        return pivot_data
    
    def calculate_bilateral_trade_intensity(self, bilateral_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bilateral trade intensity measures.
        
        Args:
            bilateral_data: DataFrame with bilateral trade flows
            
        Returns:
            DataFrame with trade intensity metrics
        """
        self.logger.info("Calculating bilateral trade intensity")
        
        # Check if data is empty
        if bilateral_data.empty:
            self.logger.warning("Empty bilateral data provided")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_columns = ['reporter', 'partner', 'year', 'trade_value']
        missing_columns = [col for col in required_columns if col not in bilateral_data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns in bilateral data: {missing_columns}")
            self.logger.error(f"Available columns: {list(bilateral_data.columns)}")
            return pd.DataFrame()
        
        # Calculate total trade for each country-year
        total_trade = bilateral_data.groupby(['reporter', 'year'])['trade_value'].sum().reset_index()
        total_trade.rename(columns={'trade_value': 'total_trade'}, inplace=True)
        
        # Merge with bilateral data
        intensity_data = bilateral_data.merge(
            total_trade, 
            on=['reporter', 'year'], 
            how='left'
        )
        
        # Calculate trade intensity (bilateral trade / total trade)
        intensity_data['trade_intensity'] = (intensity_data['trade_value'] / 
                                           intensity_data['total_trade'])
        
        # Calculate year-over-year growth
        intensity_data = intensity_data.sort_values(['reporter', 'partner', 'year'])
        intensity_data['trade_growth'] = intensity_data.groupby(['reporter', 'partner'])['trade_value'].pct_change()
        
        return intensity_data
    
    def create_trade_flow_signals(self, 
                                trade_balance: pd.DataFrame,
                                bilateral_intensity: pd.DataFrame) -> pd.DataFrame:
        """
        Create trade flow signals for currency prediction.
        
        Args:
            trade_balance: Trade balance data
            bilateral_intensity: Bilateral trade intensity data
            
        Returns:
            DataFrame with trade flow signals
        """
        self.logger.info("Creating trade flow signals")
        
        signals = []
        
        # Map country codes to currencies
        country_to_currency = {v: k for k, v in self.g10_countries.items()}
        
        for year in trade_balance['year'].unique():
            year_data = trade_balance[trade_balance['year'] == year]
            
            for _, row in year_data.iterrows():
                country_code = row['country_code']
                
                if country_code in country_to_currency:
                    currency = country_to_currency[country_code]
                    
                    # Trade balance signal
                    trade_balance_signal = np.tanh(row.get('trade_balance_ratio', 0) * 10)
                    
                    # Merchandise trade balance signal
                    merch_balance = row.get('merchandise_trade_balance', 0)
                    merch_signal = np.tanh(merch_balance / 1e11)  # Normalize by 100B
                    
                    signals.append({
                        'currency': currency,
                        'year': year,
                        'trade_balance_signal': trade_balance_signal,
                        'merchandise_signal': merch_signal,
                        'exports_usd': row.get('exports_goods_services', 0),
                        'imports_usd': row.get('imports_goods_services', 0)
                    })
        
        # Add bilateral trade intensity signals (only if data exists)
        if not bilateral_intensity.empty and 'year' in bilateral_intensity.columns:
            for year in bilateral_intensity['year'].unique():
                year_data = bilateral_intensity[bilateral_intensity['year'] == year]
                
                # Calculate average trade intensity for each reporter
                avg_intensity = year_data.groupby('reporter').agg({
                    'trade_intensity': 'mean',
                    'trade_growth': 'mean'
                }).reset_index()
                
                for _, row in avg_intensity.iterrows():
                    reporter = row['reporter']
                    
                    if reporter in country_to_currency:
                        currency = country_to_currency[reporter]
                        
                        # Find existing signal for this currency-year
                        existing_idx = None
                        for i, signal in enumerate(signals):
                            if signal['currency'] == currency and signal['year'] == year:
                                existing_idx = i
                                break
                        
                        if existing_idx is not None:
                            signals[existing_idx]['trade_intensity'] = row['trade_intensity']
                            signals[existing_idx]['trade_growth'] = row.get('trade_growth', 0)
                        else:
                            signals.append({
                                'currency': currency,
                                'year': year,
                                'trade_balance_signal': 0,
                                'merchandise_signal': 0,
                                'trade_intensity': row['trade_intensity'],
                                'trade_growth': row.get('trade_growth', 0),
                                'exports_usd': 0,
                                'imports_usd': 0
                        })
        
        return pd.DataFrame(signals)
    
    def save_trade_data(self, data: pd.DataFrame, filename: str, data_dir: str = 'data/trade_flows') -> None:
        """
        Save trade data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
            data_dir: Directory to save the file
        """
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(data_dir) / filename
        data.to_csv(filepath, index=False)
        self.logger.info(f"Trade data saved to {filepath}")
    
    def load_trade_data(self, filename: str, data_dir: str = 'data/trade_flows') -> pd.DataFrame:
        """
        Load trade data from CSV file.
        
        Args:
            filename: Name of the file
            data_dir: Directory containing the file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = Path(data_dir) / filename
        
        if filepath.exists():
            data = pd.read_csv(filepath)
            self.logger.info(f"Trade data loaded from {filepath}")
            return data
        else:
            self.logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def get_comprehensive_trade_data(self, 
                                   start_year: int = 2010, 
                                   end_year: int = 2023) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive trade data from multiple sources.
        
        Args:
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            Dictionary with different types of trade data
        """
        self.logger.info("Fetching comprehensive trade data")
        
        results = {}
        
        # Fetch World Bank data (more reliable)
        wb_data = self.fetch_worldbank_trade_data(start_year, end_year)
        if not wb_data.empty:
            trade_balance = self.calculate_trade_balance(wb_data)
            results['trade_balance'] = trade_balance
        
        # Try to fetch OECD bilateral data
        try:
            oecd_data = self.fetch_oecd_trade_data(start_year, end_year)
            if not oecd_data.empty:
                bilateral_intensity = self.calculate_bilateral_trade_intensity(oecd_data)
                results['bilateral_intensity'] = bilateral_intensity
        except Exception as e:
            self.logger.warning(f"Could not fetch OECD data: {str(e)}")
        
        # Create trade flow signals
        if 'trade_balance' in results:
            bilateral_data = results.get('bilateral_intensity', pd.DataFrame())
            trade_signals = self.create_trade_flow_signals(
                results['trade_balance'], 
                bilateral_data
            )
            results['trade_signals'] = trade_signals
        
        return results


if __name__ == "__main__":
    # Example usage
    fetcher = TradeFlowDataFetcher()
    
    # Check if trade data files already exist
    trade_balance_file = Path('data/trade_flows/trade_balance_data.csv')
    trade_signals_file = Path('data/trade_flows/trade_signals_data.csv')
    
    if trade_balance_file.exists() and trade_signals_file.exists():
        print("Trade data files already exist. Loading existing data...")
        trade_balance = fetcher.load_trade_data('trade_balance_data.csv')
        trade_signals = fetcher.load_trade_data('trade_signals_data.csv')
        
        print(f"trade_balance: {trade_balance.shape}")
        if not trade_balance.empty:
            print(trade_balance.head())
            print("---")
        
        print(f"trade_signals: {trade_signals.shape}")
        if not trade_signals.empty:
            print(trade_signals.head())
            print("---")
    else:
        print("Trade data files not found. Downloading fresh data...")
        # Get comprehensive trade data
        trade_data = fetcher.get_comprehensive_trade_data(2015, 2023)
        
        for data_type, df in trade_data.items():
            print(f"{data_type}: {df.shape}")
            if not df.empty:
                print(df.head())
                print("---")
                
                # Save the data to CSV files
                filename = f"{data_type}_data.csv"
                fetcher.save_trade_data(df, filename)
                print(f"Saved {data_type} data to {filename}")
        
        print("\nAll trade data has been saved to the data/trade_flows directory.")