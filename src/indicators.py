import pandas as pd

def calculate_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Function to calculate the simple moving average (SMA).
    Formula = (C1 + C2 + .... + Cn-1 + Cn) / n
    """
    if 'Close' not in df.columns:
        raise ValueError("Close not found in DataFrame")
    df[f"{window}_SMA"] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Function to calculate the relative strength index.
    Formula = (100 - (100 / ( 1 + RS ) ))
              where RS = ( Average Gain / Average Loss ) for the look back period (window)
    """
    if 'Close' not in df.columns:
        raise ValueError("Close not found in DataFrame")
    delta_df = df['Close'].diff()

    # Separate gain and loss
    ## If change is positive, gain = change, loss = 0
    ## If change is negative, gain = 0, loss = abs(change)
    gain = delta_df.clip(lower=0)
    loss = -1 * delta_df.clip(upper=0)

    # Calculate average gain and loss
    average_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    average_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    relative_strength = average_gain / average_loss

    df[f'{window}_RSI'] = 100 - (100 / (1 + relative_strength))
    return df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Function to calculate the bollinger bands for the given window.
    Formula:
        1. Upper  band = SMA(window) + ( 2 * std(window) )
        2. Middle band = SMA(window
        3. Lower band  = SMA(window) - ( 2 * std(window) )
    """
    if 'Close' not in df.columns:
        raise ValueError("Close not found in DataFrame")

    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()

    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std

    df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    df['BB_Width']    = (bb_upper - bb_lower) / sma
    return df

def calculate_ema(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculates the EMA of the stock over the given window
    """
    if 'Close' not in df.columns:
        raise ValueError("Close not found in DataFrame")

    df[f'{window}_EMA'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the moving average convergence divergence trend.
    Determines the 9, 12, 26 EMAs and calculates the MACD_Line, MACD_Signal, MACD_Hist
    """
    if 'Close' not in df.columns:
        raise ValueError("Close not found in DataFrame")

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD line
    macd_line = ema_12 - ema_26

    # MACD Signal
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    macd_hist = macd_line - macd_signal

    df['MACD_Line_Norm']   = macd_line   / df['Close']
    df['MACD_Signal_Norm'] = macd_signal / df['Close']
    df['MACD_Hist_Norm']   = macd_hist   / df['Close']

    return df


def calculate_relative_volume(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Function to calculate the relative volume over the given window
    Formula : Volume / Average volume for window
    Args:
        df (pd.DataFrame): The raw OHLCV data for the stock
        window (int)     : The window over which the average volume will be calculated
    """
    if 'Volume' not in df.columns:
        raise ValueError("Volume not found in dataframe. Cannot determine relative volume.")
    average_volume = df['Volume'].rolling(window=window).mean()
    df[f'{window}_RVol'] = df['Volume'] / average_volume

    return df