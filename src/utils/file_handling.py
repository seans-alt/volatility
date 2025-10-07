"""
Smart file organization - foundation of everything
"""
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd

class FileManager:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self._setup_directories()
    
    def _setup_directories(self):
        """Create all necessary directories"""
        dirs = [
            self.base_path / "data" / "raw" / self.current_date,
            self.base_path / "data" / "processed" / self.current_date, 
            self.base_path / "data" / "cache",
            self.base_path / "results" / self.current_date,
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {dir_path}")
    
    def get_data_path(self, asset=None, data_type="raw", create=True):
        """Get organized data path"""
        path = self.base_path / "data" / data_type / self.current_date
        if asset:
            # Convert BTC/USD -> BTC_USD for filenames
            asset_safe = asset.replace("/", "_")
            path = path / asset_safe
            
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_results_path(self, asset=None, model_type="garch", create=True):
        """Get organized results path"""
        path = self.base_path / "results" / self.current_date
        if asset:
            asset_safe = asset.replace("/", "_")
            path = path / asset_safe / model_type
            
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_dataframe(self, df, asset, filename, data_type="processed"):
        """Save DataFrame with proper organization"""
        path = self.get_data_path(asset, data_type)
        filepath = path / filename
        df.to_csv(filepath, index=True)
        print(f"💾 Saved: {filepath}")
        return filepath
    
    def save_model(self, model, asset, model_name, metadata=None):
        """Save trained model with metadata"""
        path = self.get_results_path(asset, "models")
        
        # Save model
        model_path = path / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metadata:
            meta_path = path / f"{model_name}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"🤖 Saved model: {model_path}")
    
    def update_latest_symlink(self):
        """Update results/latest symlink to current run"""
        latest_path = self.base_path / "results" / "latest"
        current_path = self.base_path / "results" / self.current_date
        
        if latest_path.exists():
            if latest_path.is_symlink() or latest_path.is_file():
                latest_path.unlink()
            elif latest_path.is_dir():
                import shutil
                shutil.rmtree(latest_path)
        
        # Create relative symlink
        latest_path.symlink_to(current_path.relative_to(latest_path.parent))
        print(f"🔗 Updated symlink: results/latest -> {self.current_date}")