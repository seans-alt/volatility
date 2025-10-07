"""
Data processing and returns calculation for volatility modeling
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.file_handling import FileManager
from src.utils.logging import log_step, log_success, log_error

class DataProcessor:
    def __init__(self):
        self.file_mgr = FileManager()
    
    def load_raw_data(self, assets=None):
        """Load raw OHLCV data from organized folders"""
        log_step("Loading raw data", "📂")
        
        if assets is None:
            # Auto-discover assets from latest data folder
            latest_data_path = self.file_mgr.base_path / "data" / "raw" / self.file_mgr.current_date
            assets = [f.name.replace('_', '/') for f in latest_data_path.iterdir() if f.is_dir()]
        
        raw_data = {}
        for asset in assets:
            try:
                data_path = self.file_mgr.get_data_path(asset, "raw", create=False) / "ohlcv.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
                    raw_data[asset] = df
                    log_success(f"Loaded {asset}: {len(df)} records")
                else:
                    log_error(f"Data not found for {asset}")
            except Exception as e:
                log_error(f"Error loading {asset}: {e}")
        
        return raw_data
    
    def calculate_returns(self, price_series, method='log'):
        """Calculate returns for volatility modeling"""
        if method == 'log':
            returns = np.log(price_series / price_series.shift(1))
        elif method == 'simple':
            returns = price_series.pct_change()
        else:
            raise ValueError("Method must be 'log' or 'simple'")
        
        returns = returns.dropna()
        return returns
    
    def calculate_volatility_measures(self, returns_series, window=24):
        """Calculate various volatility measures"""
        volatility_measures = {}
        
        # Realized volatility (rolling standard deviation)
        volatility_measures['realized_vol'] = returns_series.rolling(window=window).std() * np.sqrt(365 * 24)  # Annualized
        
        # Parkinson volatility (using high-low range)
        # volatility_measures['parkinson_vol'] = ...  # We'll implement this later
        
        # Garman-Klass volatility
        # volatility_measures['garman_klass_vol'] = ...  # We'll implement this later
        
        return pd.DataFrame(volatility_measures)
    
    def process_asset(self, asset, raw_data):
        """Process a single asset: calculate returns and volatility measures"""
        log_step(f"Processing {asset}", "🔧")
        
        try:
            df = raw_data[asset].copy()
            
            # Calculate log returns (standard for volatility modeling)
            df['log_returns'] = self.calculate_returns(df['close'], method='log')
            
            # Calculate volatility measures
            vol_measures = self.calculate_volatility_measures(df['log_returns'])
            df = pd.concat([df, vol_measures], axis=1)
            
            # Remove any remaining NaN values
            df = df.dropna()
            
            log_success(f"Processed {asset}: {len(df)} clean records")
            return df
            
        except Exception as e:
            log_error(f"Error processing {asset}: {e}")
            return None
    
    def process_all_assets(self, assets=None):
        """Process all assets and save processed data"""
        log_step("Processing all assets", "⚙️")
        
        # Load raw data
        raw_data = self.load_raw_data(assets)
        
        processed_data = {}
        for asset in raw_data.keys():
            processed_df = self.process_asset(asset, raw_data)
            if processed_df is not None:
                processed_data[asset] = processed_df
                
                # Save processed data
                self.file_mgr.save_dataframe(processed_df, asset, "processed.csv", data_type="processed")
        
        log_success(f"Processing complete: {len(processed_data)}/{len(raw_data)} assets")
        return processed_data
    
    def create_summary_statistics(self, processed_data):
        """Create summary statistics for all processed assets"""
        log_step("Creating summary statistics", "📊")
        
        summary_data = []
        for asset, df in processed_data.items():
            stats = {
                'asset': asset,
                'total_records': len(df),
                'mean_return': df['log_returns'].mean(),
                'volatility': df['log_returns'].std(),
                'sharpe_ratio': df['log_returns'].mean() / df['log_returns'].std() * np.sqrt(365 * 24),
                'min_return': df['log_returns'].min(),
                'max_return': df['log_returns'].max(),
                'skewness': df['log_returns'].skew(),
                'kurtosis': df['log_returns'].kurtosis()
            }
            summary_data.append(stats)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = self.file_mgr.get_results_path(model_type="summary")
        summary_file = summary_path / "asset_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        log_success(f"Summary saved: {summary_file}")
        return summary_df

def main():
    """Main function for standalone execution"""
    from src.utils.logging import setup_logging
    
    setup_logging()
    processor = DataProcessor()
    
    # Process all assets
    processed_data = processor.process_all_assets()
    
    # Create summary
    summary = processor.create_summary_statistics(processed_data)
    
    print("\n📈 PROCESSING SUMMARY:")
    print(summary[['asset', 'mean_return', 'volatility', 'sharpe_ratio']].round(6))
    
    return processed_data

if __name__ == "__main__":
    main()