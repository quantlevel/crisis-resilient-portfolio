"""
Data Module for Robust Portfolio Engineering

This module provides a DataLoader class that handles:
- Fetching historical adjusted close prices from Yahoo Finance
- Computing log returns for statistical analysis
- Local caching to avoid redundant API calls
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class DataLoader:
    """
    A class to load and preprocess financial data for portfolio analysis.
    
    This class fetches adjusted close prices from Yahoo Finance for specified
    assets, calculates log returns, and provides local caching functionality
    to avoid redundant downloads.
    
    Attributes:
        tickers (List[str]): List of asset tickers to fetch.
        start_date (str): Start date for data fetch in 'YYYY-MM-DD' format.
        end_date (str): End date for data fetch in 'YYYY-MM-DD' format.
        data_dir (Path): Directory path for caching data locally.
    
    Example:
        >>> loader = DataLoader(['SPY', 'TLT', 'GLD', 'BTC-USD'])
        >>> prices = loader.get_data()
        >>> returns = loader.calculate_log_returns(prices)
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        start_date: str = "2019-01-01",
        end_date: str = "2024-01-01",
        data_dir: str = "data"
    ):
        """
        Initialize the DataLoader.
        
        Args:
            tickers: List of Yahoo Finance ticker symbols.
                     Defaults to ['SPY', 'TLT', 'GLD', 'BTC-USD'].
            start_date: Start date for historical data in 'YYYY-MM-DD' format.
            end_date: End date for historical data in 'YYYY-MM-DD' format.
            data_dir: Directory path for caching CSV files.
        """
        self.tickers = tickers or ['SPY', 'TLT', 'GLD', 'BTC-USD']
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_filename(self) -> Path:
        """
        Generate a unique filename for the cached data based on tickers and dates.
        
        Returns:
            Path to the cache file.
        """
        tickers_str = "_".join(sorted(self.tickers)).replace("-", "")
        return self.data_dir / f"prices_{tickers_str}_{self.start_date}_{self.end_date}.csv"
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch adjusted close prices from Yahoo Finance.
        
        Downloads historical data for all specified tickers using the yfinance
        library. Only adjusted close prices are retained to account for
        dividends and stock splits.
        
        Returns:
            DataFrame with dates as index and tickers as columns,
            containing adjusted close prices.
        
        Raises:
            ValueError: If no data could be fetched for any ticker.
        """
        print(f"Fetching data from Yahoo Finance for: {self.tickers}")
        
        # Download data for all tickers
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,  # Use adjusted prices
            progress=True
        )
        
        # Handle single ticker vs multiple tickers
        if len(self.tickers) == 1:
            prices = data[['Close']].copy()
            prices.columns = self.tickers
        else:
            prices = data['Close'].copy()
        
        # Drop rows with any missing values
        prices = prices.dropna()
        
        if prices.empty:
            raise ValueError("No data fetched. Check tickers and date range.")
        
        print(f"Fetched {len(prices)} trading days from {prices.index[0].date()} to {prices.index[-1].date()}")
        
        return prices
    
    def save_data(self, data: pd.DataFrame, filepath: Optional[Path] = None) -> None:
        """
        Save price data to a CSV file.
        
        Args:
            data: DataFrame containing price data to save.
            filepath: Optional custom path. Defaults to auto-generated cache path.
        """
        if filepath is None:
            filepath = self._get_cache_filename()
        
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load price data from a cached CSV file.
        
        Args:
            filepath: Optional custom path. Defaults to auto-generated cache path.
        
        Returns:
            DataFrame with dates as index and tickers as columns.
        
        Raises:
            FileNotFoundError: If the cache file doesn't exist.
        """
        if filepath is None:
            filepath = self._get_cache_filename()
        
        if not filepath.exists():
            raise FileNotFoundError(f"Cache file not found: {filepath}")
        
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Data loaded from {filepath}")
        
        return data
    
    def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get price data, either from cache or by fetching from Yahoo Finance.
        
        This is the main entry point for obtaining price data. It first checks
        for a cached file and loads it if available. Otherwise, it fetches
        fresh data and caches it for future use.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data.
        
        Returns:
            DataFrame with dates as index and tickers as columns,
            containing adjusted close prices.
        """
        cache_file = self._get_cache_filename()
        
        if not force_refresh and cache_file.exists():
            return self.load_data(cache_file)
        
        # Fetch fresh data
        prices = self.fetch_data()
        
        # Cache for future use
        self.save_data(prices, cache_file)
        
        return prices
    
    @staticmethod
    def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate logarithmic returns from price data.
        
        Log returns are preferred for portfolio analysis because:
        1. They are time-additive (sum of log returns = log return of product)
        2. They are more normally distributed than simple returns
        3. They prevent negative prices in simulations
        
        The formula used is: r_t = ln(P_t / P_{t-1})
        
        Args:
            prices: DataFrame of asset prices with dates as index.
        
        Returns:
            DataFrame of log returns with the same structure as input.
            First row is dropped as it will be NaN.
        
        Example:
            >>> prices = pd.DataFrame({'SPY': [100, 102, 101]})
            >>> returns = DataLoader.calculate_log_returns(prices)
            >>> returns['SPY'].iloc[0]  # ~0.0198
        """
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.dropna()
    
    @staticmethod
    def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simple (arithmetic) returns from price data.
        
        Simple returns are computed as: r_t = (P_t - P_{t-1}) / P_{t-1}
        
        Args:
            prices: DataFrame of asset prices with dates as index.
        
        Returns:
            DataFrame of simple returns.
        """
        simple_returns = prices.pct_change()
        return simple_returns.dropna()
