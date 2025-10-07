"""
Data collection with absolute imports
"""
import ccxt
import pandas as pd
import time
import yaml
from pathlib import Path
import sys
import os

# Add src to path for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.file_handling import FileManager
from src.utils.logging import log_step, log_success, log_error

class DataCollector:
    def __init__(self):
        self.file_mgr = FileManager()
        self.exchange = self._initialize_exchange()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML"""
        config_path = Path("config/settings.yaml")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        log_success("Configuration loaded")
    
    def _initialize_exchange(self):
        """Initialize Kraken exchange connection"""
        log_step("Initializing Kraken exchange", "🔌")
        
        try:
            exchange = ccxt.kraken({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            # Test connection
            exchange.load_markets()
            log_success("Kraken exchange connected successfully")
            return exchange
            
        except Exception as e:
            log_error(f"Failed to initialize exchange: {e}")
            raise
    
    def fetch_ohlcv(self, symbol, days=365):
        """Fetch OHLCV data for a symbol"""
        log_step(f"Fetching {symbol} data", "📊")
        
        try:
            since = self.exchange.parse8601(
                (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
            )
            
            all_ohlcv = []
            current_since = since
            
            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                
                if not ohlcv or len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1  # Next timestamp
                
                # Rate limiting
                time.sleep(0.5)
                
                # Check if we've reached present or empty response
                if len(ohlcv) < 1000 or current_since > self.exchange.milliseconds():
                    break
            
            if not all_ohlcv:
                log_error(f"No data received for {symbol}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            
            log_success(f"{symbol}: {len(df)} candles from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            log_error(f"Error fetching {symbol}: {e}")
            return None
    
    def collect_assets(self, assets=None):
        """Collect data for multiple assets"""
        if assets is None:
            assets = self.config['data']['assets']
        
        log_step(f"Collecting data for {len(assets)} assets", "🎯")
        
        all_data = {}
        successful_assets = []
        
        for asset in assets:
            data = self.fetch_ohlcv(asset, days=self.config['data']['historical_days'])
            
            if data is not None:
                all_data[asset] = data
                successful_assets.append(asset)
                
                # Save raw data immediately
                self.file_mgr.save_dataframe(data, asset, "ohlcv.csv", data_type="raw")
            
            time.sleep(1)  # Rate limiting between assets
        
        log_success(f"Data collection complete: {len(successful_assets)}/{len(assets)} assets")
        return all_data

def main():
    """Main function for standalone execution"""
    from src.utils.logging import setup_logging
    
    setup_logging()
    collector = DataCollector()
    data = collector.collect_assets(["BTC/USD", "ETH/USD"])
    return data

if __name__ == "__main__":
    main()