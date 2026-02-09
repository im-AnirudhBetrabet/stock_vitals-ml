import yaml
from pathlib import Path

class Config:
    def __init__(self):
        self._project_root = Path(__file__).parent.parent
        self.config_path   = Path.joinpath(self._project_root, "config.yaml")
        self._config       = {}
        self._load_yaml()

    def _load_yaml(self):
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.load(f, Loader=yaml.SafeLoader)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Critical: Config file not found at {self.config_path}")

    def __getitem__(self, key):
        """
        Dunder function to allow retrieving data from the config dict using indexes
        """
        try:
            return self._config[key]
        except KeyError as e:
            raise KeyError(f"{key} not found in the config file")

    @property
    def tickers(self):
        """
        Helper function to get all the tickers
        """
        self._tickers            = self._config['tickers']
        nifty_fifty_tickers      = self._tickers.get('nifty_50', [])
        nifty_next_fifty_tickers = self._tickers.get('nifty_next_50', [])
        historical_constituents  = self._tickers.get("historical_constituents", [])
        tickers = nifty_fifty_tickers + nifty_next_fifty_tickers + historical_constituents
        return sorted(list(set(tickers)))


config = Config()


if __name__ == "__main__":
    print(f"Loaded {len(config.tickers)} tickers from config file")