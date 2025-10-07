#!/usr/bin/env python3
"""
GARCH Modeling Test Script
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.garch import GARCHModelSuite
from src.utils.logging import setup_logging

def main():
    print("🚀 CRYPTO VOLATILITY - GARCH MODELING")
    print("=" * 50)
    
    setup_logging()
    
    try:
        garch_suite = GARCHModelSuite()
        
        # Run GARCH analysis on all assets
        results = garch_suite.analyze_all_assets()
        
        print(f"\n✅ GARCH MODELING COMPLETE!")
        
        if results is not None:
            print("\n🏆 BEST MODELS SUMMARY:")
            best_models = results.loc[results.groupby('asset')['aic'].idxmin()]
            
            for _, row in best_models.iterrows():
                print(f"   {row['asset']}: {row['model']} (AIC: {row['aic']:.1f})")
            
            print(f"\n📊 Full results saved to: results/latest/comparison/")
            print(f"🎨 Plots saved to: results/latest/[asset]/plots/")
        
    except Exception as e:
        print(f"❌ GARCH MODELING FAILED: {e}")
        raise

if __name__ == "__main__":
    main()