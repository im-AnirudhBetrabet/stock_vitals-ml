import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
from typing import List, Dict

class LevelPredictor:
    def __init__(self, n_clusters: int= 6, lookback_period: int = 300):
        """
        Args:
            n_clusters (int)    : How many major levels do you want to find? (Default: 6)
            lookback_period (int): How many days of history to scan? (Default: 300 days)
        """
        self.n_clusters     : int  = n_clusters
        self.lookback_period: int  = lookback_period

    def _find_fractals(self, df: pd.DataFrame, window: int = 5) -> np.ndarray:
        """
        Identifies 'V-shaped' (Lows) and 'A-shaped' (Highs) in price actions.
        """
        # Use the last 'lookback' days only.
        data = df.iloc[-self.lookback_period:].copy()

        ## Find Local Maxima (Resistance Candidates)
        high_idx = argrelextrema(data['High'].values, np.greater, order=window)[0]
        highs    = data['High'].iloc[high_idx].values

        ## Find Local Minima (Support Candidates)
        low_idx = argrelextrema(data['Low'].values, np.less, order=window)[0]
        lows    = data['Low'].iloc[low_idx].values

        highs_lows = np.concatenate([highs, lows])
        ## Flatten into a single array and reshape to 2D array.
        return highs_lows.reshape(-1, 1) if len(highs_lows) > 0 else np.array([]).reshape(-1, 1)

    def _cluster_levels(self, price_points: np.array) -> List[float]:
        """
        Uses K-Means clustering to find the center of gravity for the fractal points.
        """
        # Check if we have at-least n_cluster points identified.
        if len(price_points) < self.n_clusters:
            return []

        k_means = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        k_means.fit(price_points)

        centers = k_means.cluster_centers_.flatten()
        centers.sort()

        return [round(x, 2) for x in centers.tolist()]

    def get_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        current_price = data['Close'].iloc[-1]

        points = self._find_fractals(data)

        levels = self._cluster_levels(points)

        supports   = [lvl for lvl in levels if lvl < current_price]
        resistance = [lvl for lvl in levels if lvl > current_price]

        return {
            "current_price": current_price,
            "supports"     : sorted(supports, reverse=True),
            "resistance"   : sorted(resistance)
        }

level_predictor = LevelPredictor()

if __name__ == "__main__":
    levels = LevelPredictor()
    print(levels.get_levels("GROWW.NS"))
    print(levels.get_levels("PREMIERENE.NS"))
    print(levels.get_levels("BLACKBUCK.NS"))
    print(levels.get_levels("SAATVIKGL.NS"))

