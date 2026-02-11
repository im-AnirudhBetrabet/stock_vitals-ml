from config.Config import Config, config
from pathlib import Path
import pandas as pd


class Data:
    def __init__(self):
        self.PARENT_DIR : Path    = Path(__file__).parent.parent
        self.DATA_DIR   : Path    = self.PARENT_DIR / config['data']['processed_path']
        self._data : pd.DataFrame = self._load_data()
        self._cutoff_date: str    = "2024-12-31"

    def _load_data(self) -> pd.DataFrame:
        if not self.DATA_DIR.exists():
            raise RuntimeError("Critical error. Processed directory is not found.")
        master_data: list = []
        for ticker in config.tickers:
            safe_ticker     = ticker.replace(".NS", "_NS")
            curr_file_path  = self.DATA_DIR / f"{safe_ticker}.csv"
            if not curr_file_path.exists():
                print(f"Skipping data for [{safe_ticker}] as corresponding file was not found in processed data folder")
                continue

            temp_df = pd.read_csv(curr_file_path, index_col=['Date'], parse_dates=True)

            temp_df = temp_df.apply(pd.to_numeric, errors='coerce')

            ## Calculate the target column: Did tomorrow close higher that today.
            temp_df['Target'] = ( temp_df['Close'].shift(-1) > temp_df['Close'] ).astype(int)

            ## Dropping the last row as it is redundant
            temp_df = temp_df.iloc[:-1]

            ## Dropping 'Close'
            temp_df.drop(columns=['Close'], inplace=True)
            temp_df.dropna(inplace=True)
            master_data.append(temp_df)

        if not master_data or len(master_data) == 0:
            raise ValueError("No data loaded. Please check processed_data folder for valid data")

        master_df = pd.concat(master_data)
        print(f"Loaded {len(master_data)} datasets.")

        master_df.to_csv(self.PARENT_DIR / "data" / "master.csv")
        master_df = master_df.astype(float)
        return master_df

    @property
    def training_data(self):
        return self._data[self._data.index <= self._cutoff_date]

    @property
    def test_data(self):
        return self._data[self._data.index > self._cutoff_date]

data = Data()

