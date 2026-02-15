from logging import exception

import pandas as pd
import yfinance as yf
from pathlib import Path
from config.Config import config
from typing import Optional
import os
import time
import random

PARENT_DIR = Path(__file__).parent.parent
def fetch_ticker_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical data for a single ticker using yfinance.
    Args:
        ticker (str): The stock symbol (e.g., 'RELIANCE.NS')
    Returns:
        pd.DataFrame: The dataframe with the OHLCV data, or None if empty
    """
    start_date: str = config['data']['start_date']
    end_date  : str = config['data']['end_date']

    print(f"Downloading Historical OHLCV Data for {ticker}")

    try:
        data = yf.download(tickers=ticker, start=start_date, end=end_date)
        if data.shape[0] == 0:
            print(f"No data found for ticker {ticker}")
            return None
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            return data
    except Exception as e:
        print(f"Critical Error | An unexpected error occurred while fetching data for {ticker} from yfinance: {e}")

def save_data(data: pd.DataFrame, ticker: str) -> None:
    """
    Saves the raw OHLCV data from yfinance to a csv file.
    Args:
        data (pd.DataFrame): The OHLCV data for the stock
        ticker (str)       : The ticker of the stock to which the data belongs.
    """
    try:
        raw_path : str  = config['data'].get('raw_path')
        file_dir : Path = PARENT_DIR / raw_path
        os.makedirs(file_dir, exist_ok=True)

        safe_ticker = ticker.replace(".NS", "_NS").replace("^", "")
        file_path = file_dir / f"{safe_ticker}.csv"

        data.to_csv(file_path)
        print(f"Raw OHLCV data for {ticker} saved successfully {raw_path}")

    except Exception as e:
        print(f"Critical Error | An unexpected error occurred while saving data for {ticker} to csv: {e}")

def fetch_index_data(index: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical data for a index using yfinance
    Args:
        index (str): The stock symbol (e.g., 'RELIANCE.NS')
    Returns:
        pd.DataFrame: The dataframe with the OHLCV data, or None if empty
    """
    if not index.startswith("^"):
        f"^{index}"
    start_date: str = config['data']['start_date']
    end_date  : str = config['data']['end_date']

    try:
        print(f"Downloading Historical OHLC Data for {index}")
        data = yf.download(tickers=index, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        print(f"Critical Error | An unexpected error occurred while fetching data for {index} from yfinance: {e}")



def main():
    for ticker in config.tickers:
        time.sleep(random.uniform(1, 3))
        data = fetch_ticker_data(ticker)
        if data is not None:
            save_data(data, ticker)
    for index in config.indices:
        time.sleep(random.uniform(1, 3))
        data = fetch_index_data(index)
        if data is not None:
            save_data(data, index)

if __name__ == "__main__":
    main()