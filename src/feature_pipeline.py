import pandas as pd
from src.indicators import calculate_sma, calculate_rsi, calculate_ema, calculate_macd, calculate_bollinger_bands
from pathlib import Path
from config.Config import config
import os

PARENT_DIR         = Path(__file__).parent.parent
RAW_DATA_DIR       = PARENT_DIR / config['data']['raw_path']
PROCESSED_DATA_DIR = PARENT_DIR / config['data']['processed_path']
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
def load_dataframe(ticker) -> pd.DataFrame:
    """
    Loads the raw OHLCV data for the ticker from the raw folder
    Args:
        ticker: The ticker of the stock for which the raw data is to be retrieved
    Return:
        pd.DataFrame
    """
    safe_ticker: str  = ticker.replace(".NS", "_NS")
    file_path  : Path = RAW_DATA_DIR / f"{safe_ticker}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, parse_dates=True, index_col='Date')
        return df
    raise FileNotFoundError(f'Raw CSV file for {ticker} not found in the raw folder.')

def apply_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the required technical indicators to the dataframe and returns it.
    Args:
        df (pd.DataFrame): The raw OHLCV data to which the technical indicators will be applied.
    Return:
        pd.DataFrame
    """
    # Trend indicators
    ## 200 SMA
    df = calculate_sma(df, window=200)

    ## 50 SMA
    df = calculate_sma(df, window=50)

    ## 20 SMA
    df = calculate_sma(df, window=20)

    ## 20 EMA
    df = calculate_ema(df, window=20)

    # Momentum
    ## RSI for 14-day window
    df = calculate_rsi(df, window=14)

    ## MACD
    df = calculate_macd(df)

    # Volatility
    df = calculate_bollinger_bands(df, window=20)

    # Custom feature - Distance to 200 SMA
    df['Dist_200SMA'] = (df['Close'] / df['200_SMA']) - 1

    return df

def save_processed_dataframe(df: pd.DataFrame, ticker: str):
    """
    Save the DataFrame to which the technical indicators are applied in a csv file within
    the processed data directory.
    Args:
        df (pd.DataFrame): The transformed dataframe to which the technical indicators are applied.
        ticker (str)     : The ticker of the stock to which the data belongs
    """
    safe_ticker: str  = ticker.replace(".NS", "_NS")
    file_path  : Path = PROCESSED_DATA_DIR / f"{safe_ticker}.csv"
    sliced_df = df.loc["2018-01-01":].dropna()

    sliced_df.to_csv(file_path)

def main():
    for ticker in config.tickers:
        try:
            df             = load_dataframe(ticker)
            transformed_df = apply_technical_indicators(df)
            save_processed_dataframe(transformed_df, ticker)
        except FileNotFoundError:
            print(f"Skipping {ticker} as raw data file was not found")

if __name__ == "__main__":
    main()