#!/usr/bin/env python3
"""
Data collection entry point - use this to run data collection
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.collector import DataCollector
from src.utils.logging import setup_logging

def main():
    print("🚀 CRYPTO VOLATILITY LAB - DATA COLLECTION")
    print("=" * 50)
    
    setup_logging()
    
    try:
        collector = DataCollector()
        
        # Collect all assets from config
        data = collector.collect_assets()
        
        print(f"\n✅ DATA COLLECTION COMPLETE!")
        print(f"📊 Collected {len(data)} assets")
        
        for asset, df in data.items():
            print(f"   {asset}: {len(df)} records")
            
        # Update latest symlink
        collector.file_mgr.update_latest_symlink()
        
    except Exception as e:
        print(f"❌ DATA COLLECTION FAILED: {e}")
        raise

if __name__ == "__main__":
    main()