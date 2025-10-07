#!/usr/bin/env python3
"""
Data processing test script
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.processor import DataProcessor
from src.utils.logging import setup_logging

def main():
    print("🚀 CRYPTO VOLATILITY - DATA PROCESSING TEST")
    print("=" * 50)
    
    setup_logging()
    
    try:
        processor = DataProcessor()
        
        # Process all assets
        processed_data = processor.process_all_assets()
        
        # Create summary
        summary = processor.create_summary_statistics(processed_data)
        
        print(f"\n✅ DATA PROCESSING COMPLETE!")
        print(f"📊 Processed {len(processed_data)} assets")
        
        print("\n📈 SUMMARY STATISTICS:")
        print(summary.to_string(index=False))
        
        # Show first few returns for each asset
        print("\n🔍 SAMPLE RETURNS:")
        for asset, df in processed_data.items():
            print(f"   {asset}: {len(df['log_returns'])} returns, mean={df['log_returns'].mean():.6f}")
            
    except Exception as e:
        print(f"❌ DATA PROCESSING FAILED: {e}")
        raise

if __name__ == "__main__":
    main()