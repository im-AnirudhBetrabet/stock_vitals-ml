import pandas as pd
from src.indicators import (calculate_sma, calculate_rsi,
                            calculate_ema, calculate_macd, calculate_bollinger_bands,
                            calculate_relative_volume, calculate_atr, calculate_adx)
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
    ## Distance to 200 SMA ( Long term indicator )
    df = calculate_sma(df, window=200)
    df['Dist_SMA_200'] = (df['Close'] / df['200_SMA']) - 1

    ## Distance to 50 SMA ( Mid-term indicator )
    df = calculate_sma(df, window=50)
    df['Dist_SMA_50'] = (df['Close'] / df['50_SMA']) - 1

    ## Distance to 20 SMA ( Short term indicator )
    df = calculate_sma(df, window=20)
    df['Dist_SMA_20'] = (df['Close'] / df['20_SMA']) - 1

    ## Trend speed - How Fast is 20 EMA moving when compared to 20 SMA
    df = calculate_ema(df, window=20)
    df['Trend_Speed'] = ( df['20_EMA'] - df['20_SMA'] ) / df['20_SMA']

    # Momentum
    ## Relative strength indicator for 14-day window
    df = calculate_rsi(df, window=14)
    df['RSI_Delta'] = df['14_RSI'].diff(3)

    ## Normalized Moving Average Convergence divergence
    df = calculate_macd(df)
    df['MACD_Hist_Delta'] = df['MACD_Hist_Norm'].diff(2)
    ## Normalized Bollinger bands
    df = calculate_bollinger_bands(df, window=20)

    ## Volume
    df = calculate_relative_volume(df, 20)
    df['Vol_Surge'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    ## Average True range
    df = calculate_atr(df, window=14)

    ## Average Directional Index
    df = calculate_adx(df, window=14)
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
    sliced_df         = df.loc["2018-01-01":]
    features_to_keep = config.feature_columns

    final_cols = [c for c in features_to_keep if c in sliced_df.columns]
    sliced_df = sliced_df[final_cols]

    sliced_df.to_csv(file_path)

def save_processed_index_df(df: pd.DataFrame, index: str):
    """
    Saves the processed index data to a csv file in the processed data directory.
    """
    safe_ticker: str       = index.replace("^", "")
    file_path: Path        = PROCESSED_DATA_DIR / f"{safe_ticker}.csv"
    sliced_df:pd.DataFrame = df.loc["2018-01-01":].dropna()
    features_to_drop: list = config.index_features_to_drop

    # final_cols = [c for c in features_to_drop if c in sliced_df.columns]
    sliced_df = sliced_df.drop(columns=features_to_drop)

    sliced_df.to_csv(file_path)

def apply_technical_indicators_to_indices(index: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the following indicators to the index.
    1. ret_5d     : percentage returns for a 5-day window.
    2. ret_20d    : percentage returns for a 20-day window.
    3. above_50SMA: boolean indicating if the current value is over the 50-day SMA.
    4. 20d_vol    : volatility of the index over a 20-day window.
    5. rsi_14     : relative strength index for a 14-day window.
    Args:
        index            : The name of the index
        df (pd.DataFrame): Historical OHLC data for the index.
    Returns:
        pd.DataFrame
    """
    df = df.copy()

    df[f'{index}_ret_5d']      = df['Close'].pct_change(periods=5)
    df[f'{index}_ret_20d']     = df['Close'].pct_change(periods=20)
    sma_50                     = df['Close'].rolling(window=50).mean()
    df[f'{index}_above_50SMA'] = ( df['Close'] > sma_50 ).astype(int)
    df[f'{index}_20d_vol']     = df['Close'].pct_change().rolling(window=20).std()
    df                         = calculate_rsi(df, window=14)
    if index == "NSEI":
        df[f"{index}_200SMA"] = df['Close'].rolling(window=200).mean()
        df['Trend_Regime'] = (df['Close'] > df[f"{index}_200SMA"]).astype(int)
        df = df.drop(columns=[f"{index}_200SMA"])
    df.rename(columns={'14_RSI' : f'{index}_rsi_14'}, inplace=True)
    return df

def main():
    for ticker in config.tickers:
        print(f"Processing raw data for [{ticker}]")
        try:
            df             = load_dataframe(ticker)
            transformed_df = apply_technical_indicators(df)
            save_processed_dataframe(transformed_df, ticker)
            print(f"[{ticker}] processed successfully")
        except FileNotFoundError:
            print(f"Skipping {ticker} as raw data file was not found")
    for index in config.indices:
        safe_index = index.replace("^", "")
        try:
            df             = load_dataframe(safe_index)
            transformed_df = apply_technical_indicators_to_indices(safe_index, df)
            save_processed_index_df(transformed_df, safe_index)
        except FileNotFoundError:
            print(f"Skipping {index} as raw data file was not found")

if __name__ == "__main__":
    main()
