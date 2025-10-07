"""
GARCH Model Suite - The Main Event!
"""
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.file_handling import FileManager
from src.utils.logging import log_step, log_success, log_error

class GARCHModelSuite:
    def __init__(self):
        self.file_mgr = FileManager()
        self.models = {}
        
    def load_processed_data(self, assets=None):
        """Load processed returns data"""
        log_step("Loading processed returns data", "📂")
        
        if assets is None:
            # Auto-discover assets from processed data
            processed_path = self.file_mgr.base_path / "data" / "processed" / self.file_mgr.current_date
            assets = [f.name.replace('_', '/') for f in processed_path.iterdir() if f.is_dir()]
        
        returns_data = {}
        for asset in assets:
            try:
                data_path = self.file_mgr.get_data_path(asset, "processed", create=False) / "processed.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
                    returns_data[asset] = df['log_returns']
                    log_success(f"Loaded {asset} returns: {len(df)} records")
                else:
                    log_error(f"Processed data not found for {asset}")
            except Exception as e:
                log_error(f"Error loading {asset}: {e}")
        
        return returns_data
    
    def fit_garch(self, returns, p=1, q=1, dist='normal'):
        """Public GARCH fit - handles scaling automatically"""
        scaled_returns = returns * 100
        return self._fit_garch_internal(scaled_returns, p, q, dist)
    
    def fit_egarch(self, returns, p=1, q=1, dist='normal'):
        """Public EGARCH fit - handles scaling automatically"""
        scaled_returns = returns * 100
        return self._fit_egarch_internal(scaled_returns, p, q, dist)
    
    def fit_gjrgarch(self, returns, p=1, q=1, dist='normal'):
        """Public GJR-GARCH fit - handles scaling automatically"""
        scaled_returns = returns * 100
        return self._fit_gjrgarch_internal(scaled_returns, p, q, dist)
    def _fit_garch_internal(self, scaled_returns, p=1, q=1, dist='normal'):
        """Internal GARCH fit with scaled returns"""
        try:
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist=dist, rescale=False)
            fitted = model.fit(disp='off', show_warning=False)
            return fitted
        except Exception as e:
            return None

    def _fit_egarch_internal(self, scaled_returns, p=1, q=1, dist='normal'):
        """Internal EGARCH fit with scaled returns"""
        try:
            model = arch_model(scaled_returns, vol='EGarch', p=p, q=q, dist=dist, rescale=False)
            fitted = model.fit(disp='off', show_warning=False)
            return fitted
        except Exception as e:
            return None

    def _fit_gjrgarch_internal(self, scaled_returns, p=1, q=1, dist='normal'):
        """Internal GJR-GARCH fit with scaled returns"""
        try:
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q, o=1, dist=dist, rescale=False)
            fitted = model.fit(disp='off', show_warning=False)
            return fitted
        except Exception as e:
            return None
    
    def compare_models(self, returns, asset_name):
        """Compare all GARCH variants for a single asset with proper scaling"""
        log_step(f"Comparing GARCH models for {asset_name}", "📊")
        # Scale returns once to avoid warnings - THIS IS THE KEY FIX
        scaled_returns = returns * 100
        models = {
            'GARCH': self._fit_garch_internal(scaled_returns),
            'EGARCH': self._fit_egarch_internal(scaled_returns),
            'GJR-GARCH': self._fit_gjrgarch_internal(scaled_returns)
        }
        # Remove failed models
        models = {k: v for k, v in models.items() if v is not None}
        comparison = []
        for name, model in models.items():
            comparison.append({
                'asset': asset_name,
                'model': name,
                'aic': model.aic,
                'bic': model.bic,
                'log_likelihood': model.loglikelihood,
                'persistence': self._calculate_persistence(model),
                'params': model.params.to_dict()
            })
        return pd.DataFrame(comparison)
    
    def _calculate_persistence(self, model):
        """Calculate volatility persistence"""
        try:
            if hasattr(model, 'alpha') and hasattr(model, 'beta'):
                return model.alpha.sum() + model.beta.sum()
            else:
                # For EGARCH, persistence calculation is different
                return None
        except:
            return None
    
    def forecast_volatility(self, model, steps=100):
        """Generate volatility forecasts"""
        try:
            forecast = model.forecast(horizon=steps, reindex=False)
            return forecast.variance.iloc[-1].values
        except Exception as e:
            log_error(f"Forecasting failed: {e}")
            return None
    
    def create_volatility_plots(self, returns, models, asset_name):
        """Create beautiful volatility visualization plots"""
        log_step(f"Creating volatility plots for {asset_name}", "🎨")
        # Scale returns for plotting consistency
        scaled_returns = returns * 100
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'GARCH Model Analysis - {asset_name}', fontsize=16, fontweight='bold')
        # Plot 1: Returns series (scaled)
        axes[0, 0].plot(returns.index, scaled_returns.values, color='blue', alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title('Log Returns (Scaled ×100)', fontweight='bold')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].grid(True, alpha=0.3)
        # Plot 2: Conditional volatility (GARCH)
        if 'GARCH' in models:
            garch_vol = np.sqrt(models['GARCH'].conditional_volatility)
            axes[0, 1].plot(returns.index, garch_vol, color='red', linewidth=1)
            axes[0, 1].set_title('GARCH Conditional Volatility', fontweight='bold')
            axes[0, 1].set_ylabel('Volatility')
            axes[0, 1].grid(True, alpha=0.3)
        # Plot 3: Model comparison (AIC)
        model_names = [m for m in models.keys() if models[m] is not None]
        aic_values = [models[m].aic for m in model_names]
        axes[1, 0].bar(model_names, aic_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Model Comparison (AIC)', fontweight='bold')
        axes[1, 0].set_ylabel('AIC (lower is better)')
        # Plot 4: Volatility forecasts
        if 'GARCH' in models and models['GARCH'] is not None:
            forecasts = self.forecast_volatility(models['GARCH'], steps=50)
            if forecasts is not None:
                axes[1, 1].plot(range(len(forecasts)), np.sqrt(forecasts), color='purple', linewidth=2)
                axes[1, 1].set_title('Volatility Forecast (50 steps)', fontweight='bold')
                axes[1, 1].set_ylabel('Forecasted Volatility')
                axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        # Save plot
        plot_path = self.file_mgr.get_results_path(asset_name, "plots")
        plot_file = plot_path / "volatility_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        log_success(f"Plot saved: {plot_file}")
    
    def analyze_all_assets(self, assets=None):
        """Complete GARCH analysis for all assets"""
        log_step("Starting GARCH analysis for all assets", "🚀")
        
        # Load returns data
        returns_data = self.load_processed_data(assets)
        
        all_comparisons = []
        
        for asset_name, returns in returns_data.items():
            log_step(f"Analyzing {asset_name}", "📈")
            
            # Compare models
            comparison_df = self.compare_models(returns, asset_name)
            all_comparisons.append(comparison_df)
            
            # Get the best model (lowest AIC)
            best_model_row = comparison_df.loc[comparison_df['aic'].idxmin()]
            best_model_name = best_model_row['model']
            
            # Fit the best model
            if best_model_name == 'GARCH':
                best_model = self.fit_garch(returns)
            elif best_model_name == 'EGARCH':
                best_model = self.fit_egarch(returns)
            elif best_model_name == 'GJR-GARCH':
                best_model = self.fit_gjrgarch(returns)
            
            # Save model
            if best_model is not None:
                self.file_mgr.save_model(best_model, asset_name, f"best_{best_model_name.lower()}")
                
                # Create volatility plots
                models_dict = {
                    'GARCH': self.fit_garch(returns),
                    'EGARCH': self.fit_egarch(returns),
                    'GJR-GARCH': self.fit_gjrgarch(returns)
                }
                self.create_volatility_plots(returns, models_dict, asset_name)
            
            log_success(f"Completed {asset_name} - Best model: {best_model_name}")
        
        # Combine all comparisons
        if all_comparisons:
            full_comparison = pd.concat(all_comparisons, ignore_index=True)
            
            # Save comparison results
            comp_path = self.file_mgr.get_results_path(model_type="comparison")
            comp_file = comp_path / "garch_model_comparison.csv"
            full_comparison.to_csv(comp_file, index=False)
            
            log_success(f"Model comparison saved: {comp_file}")
            
            return full_comparison
        
        return None

def main():
    """Main function for standalone execution"""
    from src.utils.logging import setup_logging
    
    setup_logging()
    garch_suite = GARCHModelSuite()
    
    # Analyze all assets
    comparison_results = garch_suite.analyze_all_assets()
    
    if comparison_results is not None:
        print("\n🏆 BEST MODELS BY ASSET:")
        best_models = comparison_results.loc[comparison_results.groupby('asset')['aic'].idxmin()]
        print(best_models[['asset', 'model', 'aic', 'persistence']].to_string(index=False))
    
    return comparison_results

if __name__ == "__main__":
    main()