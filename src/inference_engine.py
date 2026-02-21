import json
from typing import Optional, Dict, Any

import joblib

from config.Config import config
from src.feature_pipeline import apply_technical_indicators, apply_technical_indicators_to_indices
from src.indicators import calculate_sma, calculate_rsi, calculate_ema, calculate_macd, calculate_bollinger_bands, \
    calculate_relative_volume, calculate_atr, calculate_adx
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from config.Config import config
from src.support_resistance import level_predictor


class StockPredictor:
    def __init__(self):
        self.PARENT_DIR : Path     = Path(__file__).parent.parent
        self.MODEL_PATH : Path     = self.PARENT_DIR / "models" / config.current_model_version / f"voting_classifier_{config.current_model_version}.pkl"
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

        features_to_keep = config.feature_columns
        final_cols  = [c for c in features_to_keep if c in df.columns]
        latest_data = df[final_cols].copy()
        latest_data.fillna(0, inplace=True)
        latest_data.drop(columns=['Close'], inplace=True)
        latest_data = latest_data.iloc[[-1]]
        return latest_data

    def _load_data(self, ticker) -> Optional[pd.DataFrame]:
        data = yf.download(progress=False,tickers=ticker, interval='1d', start=self._date_minus_1_5_years, end=self._next_day)
        if data.shape[0] == 0:
            print(f"No data found for ticker {ticker}")
            return None
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
        return data

    def _load_indices(self) -> Optional[pd.DataFrame]:
        index_data = []
        for index in config.indices:
            if not index.startswith("^"):
                f"^{index}"
            try:
                data = yf.download(progress=False, tickers=index, start=self._date_minus_1_5_years, end=self._next_day, interval='1d')
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                processed_data = apply_technical_indicators_to_indices(index.replace("^", ""), data)
                sliced_df = processed_data.drop(columns=config.index_features_to_drop)
                if isinstance(sliced_df.columns, pd.MultiIndex):
                    sliced_df.columns = sliced_df.columns.droplevel(1)
                index_data.append(sliced_df)
            except Exception as e:
                print(f"Critical Error | An unexpected error occurred while fetching data for {index} from yfinance: {e}")
        index_master = index_data[0]
        for df in index_data[1:]:
            index_master = index_master.join(df, how='left')
        return index_master.iloc[[-1]]

    def _calculate_trade_levels(self, price: float, atr: float) -> Dict[str, float]:
        trade_levels = {}
        stop_loss_1: float = price - (1.0 * atr)
        stop_loss_2: float = price - (1.5 * atr)
        stop_loss_3: float = price - (2.0 * atr)

        target_1 : float = price + (1.5 * atr)
        target_2 : float = price + (2.0 * atr)
        target_3 : float = price + (2.5 * atr)

        trade_levels['stop_loss_1'] = stop_loss_1
        trade_levels['stop_loss_2'] = stop_loss_2
        trade_levels['stop_loss_3'] = stop_loss_3
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
            index_data       : pd.DataFrame     = self._load_indices()
            processed_data   : pd.DataFrame     = apply_technical_indicators(data)
            features         : pd.DataFrame     = self._prepare_features(processed_data)
            features = features.join(index_data, how='left')
            features.to_csv("featurs.csv")
            regime_prediction: list             = self.model.predict_proba(features)[0]
            class_map        : Dict[Any,Any]    = {self.model.classes_[0]: regime_prediction[0],  self.model.classes_[1]: regime_prediction[1] }
            bullish_regime   : float            = class_map[1]
            bearish_regime   : float            = class_map[0]
            latest_indicators: pd.DataFrame     = processed_data.iloc[[-1]]
            trade_levels     : dict[str, float] = self._calculate_trade_levels(latest_indicators['Close'].item(), latest_indicators['ATR'].item())
            signal           : str              = self._get_signal_strength(bullish_regime, latest_indicators['ADX'].item())
            sprt_rst_levels  : dict[str, Any]   = level_predictor.get_levels(data)
            return {
                "ticker"             : ticker,
                "bullish_probability": bullish_regime,
                "bearish_probability": bearish_regime,
                "trade_levels"       : trade_levels,
                "signal"             : signal,
                "support_resistance" : sprt_rst_levels
            }

def inference_engine(ticker: str):
    predictor = StockPredictor()
    return predictor.predict(ticker)



if __name__ == "__main__":
    print(json.dumps(inference_engine("SAATVIKGL.NS")))
    print(json.dumps(inference_engine("INOXWIND.NS")))
    print(json.dumps(inference_engine("MARKSANS.NS")))
    print(json.dumps(inference_engine("BLACKBUCK.NS")))
    print(json.dumps(inference_engine("PROSTARM.NS")))
    print(json.dumps(inference_engine("EBGNG.NS")))
    print(json.dumps(inference_engine("JUNIPER.NS")))
    print(json.dumps(inference_engine("SKYGOLD.NS")))
    print(json.dumps(inference_engine("BLACKBUCK.NS")))
    print(json.dumps(inference_engine("GICRE.NS")))
    print(json.dumps(inference_engine("GPTINFRA.NS")))
    print(json.dumps(inference_engine("GARUDA.NS")))
    print(json.dumps(inference_engine("FISCHER.NS")))
    print(json.dumps(inference_engine("LTFOODS.NS")))






