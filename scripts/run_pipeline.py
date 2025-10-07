#!/usr/bin/env python3
"""
Final Complete Pipeline - Data → Processing → GARCH → Reporting
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logging import setup_logging, log_step, log_success

def main():
    print("🚀 CRYPTO VOLATILITY LAB - COMPLETE PIPELINE")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # 1. Data Collection
        log_step("Phase 1: Data Collection", "📊")
        from src.data.collector import DataCollector
        collector = DataCollector()
        raw_data = collector.collect_assets()
        
        # 2. Data Processing 
        log_step("Phase 2: Data Processing", "🔧")
        from src.data.processor import DataProcessor
        processor = DataProcessor()
        processed_data = processor.process_all_assets()
        summary = processor.create_summary_statistics(processed_data)
        
        # 3. GARCH Modeling
        log_step("Phase 3: GARCH Modeling", "📈")
        from src.models.garch import GARCHModelSuite
        garch_suite = GARCHModelSuite()
        model_results = garch_suite.analyze_all_assets()
        
        # 4. Comprehensive Reporting
        log_step("Phase 4: Comprehensive Reporting", "📋")
        from src.analysis.reporting import GARCHReporter
        reporter = GARCHReporter()
        reporter.create_comprehensive_report()

        # Add this after GARCH modeling:
        log_step("Phase 5: Notebook Generation", "📓")
        from scripts.generate_notebook import NotebookGenerator
        notebook_gen = NotebookGenerator()
        notebooks = notebook_gen.generate_all_notebooks()
        
        # Update symlink
        collector.file_mgr.update_latest_symlink()
        
        log_success("🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        
        # Final summary
        if model_results is not None:
            best_models = model_results.loc[model_results.groupby('asset')['aic'].idxmin()]
            print(f"\n🏆 FINAL RESULTS:")
            for _, row in best_models.iterrows():
                print(f"   {row['asset']}: {row['model']} (AIC: {row['aic']:.1f})")
        
        print(f"\n📁 All results saved to: results/latest/")
        print(f"📋 Comprehensive report: results/latest/reports/garch_analysis_report.md")
        
    except Exception as e:
        log_step(f"Pipeline failed: {e}", "❌")
        raise

if __name__ == "__main__":
    main()