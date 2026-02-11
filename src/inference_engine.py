from typing import Optional, Dict

import joblib
from src.indicators import calculate_sma, calculate_rsi, calculate_ema, calculate_macd, calculate_bollinger_bands, \
    calculate_relative_volume, calculate_atr, calculate_adx
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta



class StockPredictor:
    def __init__(self):
        self.PARENT_DIR : Path     = Path(__file__).parent.parent
        self.MODEL_PATH : Path     = self.PARENT_DIR / "models" / "voting_classifier.pkl"
        self._current_date         = datetime.now()
        self._date_minus_1_5_years = self._current_date - relativedelta(years=1, months=6)
        self._next_day             = self._current_date + timedelta(days=1)

        if not self.MODEL_PATH.exists():
            raise FileNotFoundError("Voting classifier model not found in models directory")

        self.model = joblib.load(self.MODEL_PATH)

    def _calculator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df['Trend_Speed'] = (df['20_EMA'] - df['20_SMA']) / df['20_SMA']

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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:

        features_to_keep = [
            'Dist_SMA_200', 'Dist_SMA_50', 'Dist_SMA_20', 'Trend_Speed',
            '14_RSI',
            'MACD_Line_Norm', 'MACD_Signal_Norm', 'MACD_Hist_Norm',
            'BB_Position', 'BB_Width',
            '20_RVol', 'RSI_Delta', 'MACD_Hist_Delta', 'Vol_Surge'
        ]
        final_cols  = [c for c in features_to_keep if c in df.columns]
        latest_data = df[final_cols].copy()
        latest_data.fillna(0, inplace=True)
        latest_data = latest_data.iloc[[-1]]
        return latest_data

    def _load_data(self, ticker) -> Optional[pd.DataFrame]:
        data = yf.download(tickers=ticker, interval='1d', start=self._date_minus_1_5_years, end=self._next_day)
        if data.shape[0] == 0:
            print(f"No data found for ticker {ticker}")
            return None
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
        return data

    def _calculate_trade_levels(self, price: float, atr: float) -> Dict[str, float]:
        trade_levels = {}
        stop_loss: float = price - (2.0 * atr)
        target_1 : float = price + (2.0 * atr)
        target_2 : float = price + (4.0 * atr)
        target_3 : float = price + (8.0 * atr)
        trade_levels['stop_loss'] = stop_loss
        trade_levels['target_1']  = target_1
        trade_levels['target_2']  = target_2
        trade_levels['target_3']  = target_3
        return trade_levels

    def _get_signal_strength(self, prob: float, adx: float) -> str:
        if adx < 25:
            return "NEUTRAL (CHOPPY)"
        if prob >= 0.60:
            return "STRONG BUY"
        elif prob >= 0.57:
            return "BUY"
        elif prob >= 0.54:
            return "WEAK BUY"
        else:
            return "HOLD"

    def predict(self, ticker):
        data = self._load_data(ticker)
        if data is not None:
            processed_data   : pd.DataFrame     = self._calculator_indicators(data)
            features         : pd.DataFrame     = self._prepare_features(processed_data)
            regime_prediction: list             = self.model.predict_proba(features)
            bullish_regime   : float            = regime_prediction[0][1]
            bearish_regime   : float            = regime_prediction[0][0]
            latest_indicators: pd.DataFrame     = processed_data.iloc[[-1]]
            trade_levels     : dict[str, float] = self._calculate_trade_levels(latest_indicators['Close'].item(), latest_indicators['ATR'].item())
            signal           : str              = self._get_signal_strength(bullish_regime, latest_indicators['ADX'].item())

            return {
                "ticker"             : ticker,
                "bullish_probability": bullish_regime,
                "bearish_probability": bearish_regime,
                "trade_levels"       : trade_levels,
                "signal"             : signal
            }

def inference_engine(ticker: str):
    predictor = StockPredictor()
    print(predictor.predict(ticker))



if __name__ == "__main__":
    inference_engine("SAATVIKGL.NS")
    inference_engine("PREMIERENE.NS")
    inference_engine("BLACKBUCK.NS")
    inference_engine("GROWW.NS")
    inference_engine("INFY.NS")
    inference_engine("ASHOKLEY.NS")





