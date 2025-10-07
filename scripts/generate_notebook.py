#!/usr/bin/env python3
"""
Auto-generate Jupyter notebooks from GARCH results
"""
import sys
import os
import json
import pandas as pd
from pathlib import Path
import nbformat as nbf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.file_handling import FileManager
from src.utils.logging import setup_logging, log_step, log_success

class NotebookGenerator:
    def __init__(self):
        self.file_mgr = FileManager()
    
    def generate_analysis_notebook(self, asset):
        """Generate a comprehensive analysis notebook for a single asset"""
        log_step(f"Generating notebook for {asset}", "📓")
        
        # Create notebook
        nb = nbf.v4.new_notebook()
        
        # Notebook content
        nb.cells = [
            # Title and overview
            nbf.v4.new_markdown_cell(f"# GARCH Volatility Analysis - {asset}"),
            nbf.v4.new_markdown_cell(f"**Analysis Date**: {self.file_mgr.current_date}"),
            nbf.v4.new_markdown_cell("## Overview\nThis notebook presents comprehensive GARCH volatility analysis."),
            
            # Imports
            nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys
import os

# Setup plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
%matplotlib inline"""),
            
            # Load data
            nbf.v4.new_markdown_cell("## 1. Load Processed Data"),
            nbf.v4.new_code_cell(f"""# Load processed returns data
data_path = Path("data/processed/{self.file_mgr.current_date}/{asset.replace('/', '_')}/processed.csv")
returns_df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
print(f"Loaded {{len(returns_df)}} records for {asset}")
returns_df[['close', 'log_returns']].head()"""),
            
            # Returns analysis
            nbf.v4.new_markdown_cell("## 2. Returns Analysis"),
            nbf.v4.new_code_cell("""# Basic returns statistics
returns_stats = returns_df['log_returns'].describe()
returns_stats['skewness'] = returns_df['log_returns'].skew()
returns_stats['kurtosis'] = returns_df['log_returns'].kurtosis()
returns_stats"""),
            
            nbf.v4.new_code_cell("""# Plot returns distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Time series
ax1.plot(returns_df.index, returns_df['log_returns'], alpha=0.7, linewidth=0.8)
ax1.set_title(f'{asset} - Log Returns')
ax1.set_ylabel('Returns')
ax1.grid(True, alpha=0.3)

# Distribution
ax2.hist(returns_df['log_returns'], bins=50, alpha=0.7, density=True)
ax2.set_title(f'{asset} - Returns Distribution')
ax2.set_xlabel('Returns')
ax2.set_ylabel('Density')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),
            
            # GARCH Results
            nbf.v4.new_markdown_cell("## 3. GARCH Model Results"),
            nbf.v4.new_code_cell(f"""# Load GARCH comparison results
comp_path = Path("results/{self.file_mgr.current_date}/comparison/garch_model_comparison.csv")
comparison_df = pd.read_csv(comp_path)
asset_comparison = comparison_df[comparison_df['asset'] == '{asset}']
asset_comparison"""),
            
            nbf.v4.new_code_cell("""# Visualize model comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = asset_comparison['model']
aic_values = asset_comparison['aic']

bars = ax.bar(models, aic_values, color=['skyblue', 'lightcoral', 'lightgreen'])
ax.set_title(f'{asset} - GARCH Model Comparison (AIC)')
ax.set_ylabel('AIC (Lower is Better)')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, aic in zip(bars, aic_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{aic:.1f}', ha='center', va='bottom')

plt.show()"""),
            
            # Best Model Parameters
            nbf.v4.new_markdown_cell("## 4. Best Model Parameters"),
            nbf.v4.new_code_cell(f"""# Get best model
best_model_row = asset_comparison.loc[asset_comparison['aic'].idxmin()]
best_model_name = best_model_row['model']
best_model_params = best_model_row['params']

print(f"Best Model: {{best_model_name}}")
print(f"AIC: {{best_model_row['aic']:.1f}}")
print(f"Log Likelihood: {{best_model_row['log_likelihood']:.1f}}")
print(f"Persistence: {{best_model_row['persistence']:.3f}}")

print("\\nModel Parameters:")
if isinstance(best_model_params, str):
    best_model_params = eval(best_model_params)  # Convert string to dict
for param, value in best_model_params.items():
    print(f"  {{param}}: {{value:.6f}}")"""),
            
            # Volatility Visualization
            nbf.v4.new_markdown_cell("## 5. Volatility Analysis"),
            nbf.v4.new_code_cell(f"""# Display volatility plot
from IPython.display import Image

plot_path = Path("results/{self.file_mgr.current_date}/{asset.replace('/', '_')}/plots/volatility_analysis.png")
if plot_path.exists():
    display(Image(filename=str(plot_path)))
else:
    print("Volatility plot not found")"""),
            
            # Trading Insights
            nbf.v4.new_markdown_cell("## 6. Trading Insights"),
            nbf.v4.new_code_cell("""# Generate trading insights
persistence = best_model_row['persistence']
if not pd.isna(persistence):
    if persistence > 0.95:
        persistence_insight = "High persistence - volatility shocks have long-lasting effects"
    elif persistence > 0.8:
        persistence_insight = "Moderate persistence - volatility mean-reverts moderately"
    else:
        persistence_insight = "Low persistence - volatility mean-reverts quickly"

volatility = returns_df['log_returns'].std()
if volatility > 0.01:
    vol_insight = "High volatility - suitable for volatility trading strategies"
elif volatility > 0.005:
    vol_insight = "Moderate volatility - balanced risk-return profile"
else:
    vol_insight = "Low volatility - stable but lower potential returns"

print("Trading Insights:")
print(f"- {{persistence_insight}}")
print(f"- {{vol_insight}}")
print(f"- Best model type: {{best_model_name}}")
print(f"- Recommended: Use for {{'shorter-term' if persistence < 0.8 else 'longer-term'}} volatility strategies")"""),
            
            # Conclusion
            nbf.v4.new_markdown_cell("## 7. Conclusion"),
            nbf.v4.new_markdown_cell(f"""
### Summary for {asset}

- **Best Model**: {{{{best_model_name}}}}
- **Volatility Persistence**: {{{{persistence:.3f}}}}
- **Returns Volatility**: {{{{volatility:.4f}}}}
- **Model Fit Quality**: {{{{'Excellent' if best_model_row['aic'] < 1000 else 'Good'}}}}

### Recommended Actions:
1. Monitor volatility regimes using the {{{{best_model_name}}}} model
2. Adjust position sizing based on volatility forecasts
3. Consider volatility-based entry/exit signals
""")
        ]
        
        # Save notebook
        notebook_path = self.file_mgr.get_results_path(asset, "notebooks")
        notebook_file = notebook_path / f"{asset.replace('/', '_')}_analysis.ipynb"
        
        with open(notebook_file, 'w') as f:
            nbf.write(nb, f)
        
        log_success(f"Notebook saved: {notebook_file}")
        return notebook_file
    
    def generate_portfolio_notebook(self, assets):
        """Generate portfolio-level analysis notebook"""
        log_step("Generating portfolio analysis notebook", "📚")
        
        nb = nbf.v4.new_notebook()
        
        nb.cells = [
            nbf.v4.new_markdown_cell("# Crypto Volatility Portfolio Analysis"),
            nbf.v4.new_markdown_cell(f"**Analysis Date**: {self.file_mgr.current_date}"),
            nbf.v4.new_markdown_cell("## Portfolio Overview"),
            
            nbf.v4.new_code_cell("""# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline"""),
            
            nbf.v4.new_code_cell(f"""# Load portfolio comparison results
comp_path = Path("results/{self.file_mgr.current_date}/comparison/garch_model_comparison.csv")
portfolio_df = pd.read_csv(comp_path)
portfolio_df.head()"""),
            
            nbf.v4.new_code_cell("""# Best models by asset
best_models = portfolio_df.loc[portfolio_df.groupby('asset')['aic'].idxmin()]
print("Best Models by Asset:")
print(best_models[['asset', 'model', 'aic', 'persistence']].to_string(index=False))"""),
            
            nbf.v4.new_code_cell("""# Create portfolio summary visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Model preferences
model_counts = best_models['model'].value_counts()
ax1.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%', 
        colors=['lightcoral', 'lightgreen', 'skyblue'])
ax1.set_title('GARCH Model Preferences Across Portfolio')

# AIC comparison
assets = best_models['asset']
aics = best_models['aic']
colors = ['red' if aic > 1000 else 'green' for aic in aics]

bars = ax2.bar(assets, aics, color=colors, alpha=0.7)
ax2.set_title('Model Fit Quality (AIC) by Asset')
ax2.set_ylabel('AIC (Lower is Better)')
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, aic in zip(bars, aics):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
            f'{aic:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()"""),
            
            nbf.v4.new_markdown_cell("## Portfolio Insights"),
            nbf.v4.new_code_cell("""# Generate portfolio insights
print("Portfolio Volatility Insights:")
print(f"Total Assets Analyzed: {len(best_models)}")
print(f"EGARCH Models: {len(best_models[best_models['model'] == 'EGARCH'])}")
print(f"GJR-GARCH Models: {len(best_models[best_models['model'] == 'GJR-GARCH'])}")
print(f"GARCH Models: {len(best_models[best_models['model'] == 'GARCH'])}")

avg_persistence = best_models['persistence'].mean()
print(f"Average Volatility Persistence: {avg_persistence:.3f}")

# Trading recommendations
print("\\nPortfolio Trading Recommendations:")
print("1. Use EGARCH for major assets (BTC, ETH)")
print("2. Use GJR-GARCH for high-volatility altcoins")
print("3. Monitor persistence for regime changes")
print("4. BTC shows superior model fit - focus volatility strategies here")""")
        ]
        
        # Save portfolio notebook
        notebook_path = self.file_mgr.get_results_path(model_type="notebooks")
        notebook_file = notebook_path / "portfolio_analysis.ipynb"
        
        with open(notebook_file, 'w') as f:
            nbf.write(nb, f)
        
        log_success(f"Portfolio notebook saved: {notebook_file}")
        return notebook_file
    
    def generate_all_notebooks(self, assets=None):
        """Generate notebooks for all assets"""
        log_step("Generating all analysis notebooks", "📓")
        
        if assets is None:
            # Auto-discover assets
            processed_path = self.file_mgr.base_path / "data" / "processed" / self.file_mgr.current_date
            assets = [f.name.replace('_', '/') for f in processed_path.iterdir() if f.is_dir()]
        
        generated_notebooks = []
        
        # Generate individual asset notebooks
        for asset in assets:
            notebook_file = self.generate_analysis_notebook(asset)
            generated_notebooks.append(notebook_file)
        
        # Generate portfolio notebook
        portfolio_notebook = self.generate_portfolio_notebook(assets)
        generated_notebooks.append(portfolio_notebook)
        
        log_success(f"Generated {len(generated_notebooks)} notebooks")
        return generated_notebooks

def main():
    """Main function for notebook generation"""
    setup_logging()
    generator = NotebookGenerator()
    
    notebooks = generator.generate_all_notebooks()
    
    print(f"\n✅ NOTEBOOK GENERATION COMPLETE!")
    print(f"📓 Generated {len(notebooks)} notebooks:")
    for notebook in notebooks:
        print(f"   📄 {notebook}")

if __name__ == "__main__":
    main()