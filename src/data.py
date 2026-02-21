from config.Config import config
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
        index_data : list = []
        for index in config.indices:
            safe_index = index.replace("^", "")
            index_path = self.DATA_DIR / f"{safe_index}.csv"
            if not index_path.exists():
                print(f"Skipping data for [{safe_index}] as corresponding file was not found in processed data folder")
                continue
            index_df = pd.read_csv(index_path, index_col=['Date'], parse_dates=True)
            index_data.append(index_df)
        index_master = index_data[0]
        for df in index_data[1:]:
            index_master = index_master.join(df, how='left')

        for ticker in config.tickers:
            safe_ticker     = ticker.replace(".NS", "_NS")
            curr_file_path  = self.DATA_DIR / f"{safe_ticker}.csv"
            if not curr_file_path.exists():
                print(f"Skipping data for [{safe_ticker}] as corresponding file was not found in processed data folder")
                continue

            temp_df = pd.read_csv(curr_file_path, index_col=['Date'], parse_dates=True)

            temp_df = temp_df.apply(pd.to_numeric, errors='coerce')

            ## Calculate the target column: Did price 5 days ahead close higher than today
            temp_df = temp_df.join(index_master, how='left')
            temp_df[index_master.columns] = temp_df[index_master.columns].ffill()

            temp_df['Target'] = ( temp_df['Close'].shift(-int(config.returns_horizon)) > temp_df['Close'] ).astype(int)
            ## Dropping redundant rows that may contain NaN due to target calculation
            temp_df = temp_df.dropna(subset=['Target'])

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
if __name__ == "__main__":
    print(data.training_data.shape)
    print(data.test_data.shape)
    print(data.training_data.isna().sum().sum())
    print(data.test_data.isna().sum().sum())
    print(data.training_data['Target'].mean())
    print(data.test_data['Target'].mean())

