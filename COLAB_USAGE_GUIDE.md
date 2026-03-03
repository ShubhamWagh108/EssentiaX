# Google Colab Usage Guide - EssentiaX with Fixed Plotly Rendering

## 🎯 Quick Start in Colab

### 1. Install EssentiaX

```python
!pip install essentiax
```

### 2. Import and Use

```python
import pandas as pd
import numpy as np
from essentiax.visuals.smartViz import smart_viz

# Load your data
df = pd.read_csv('your_data.csv')

# Generate visualizations with AI insights
smart_viz(df, mode='auto')
```

That's it! You'll now see:
- ✅ Beautiful Rich console output with colors and formatting
- ✅ Interactive Plotly graphs that are fully visible
- ✅ AI-powered insights and recommendations

## 📊 Complete Example

```python
# Install (first time only)
!pip install essentiax

# Import libraries
import pandas as pd
import numpy as np
from essentiax.visuals.smartViz import smart_viz
from essentiax.visuals.advanced_viz import advanced_viz
from essentiax.eda.smart_eda import smart_eda

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'revenue': np.random.exponential(1000, 500),
    'cost': np.random.normal(500, 100, 500),
    'quantity': np.random.poisson(50, 500),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 500),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
    'rating': np.random.uniform(1, 5, 500)
})

# Option 1: Smart Visualization (Auto Mode)
print("🎨 Smart Visualization with AI Selection")
smart_viz(df, mode='auto', max_plots=5)

# Option 2: Manual Column Selection
print("\n🎯 Manual Visualization")
smart_viz(df, mode='manual', columns=['revenue', 'cost', 'quantity', 'category'])

# Option 3: Advanced 3D Visualizations
print("\n🚀 Advanced 3D Scatter Plot")
advanced_viz(df, viz_type='3d_scatter', columns=['revenue', 'cost', 'quantity'])

# Option 4: Complete EDA Analysis
print("\n🧠 Complete EDA Analysis")
smart_eda(df, mode='all', target='rating')
```

## 🎨 Available Visualization Types

### SmartViz (Automatic + Manual)

```python
from essentiax.visuals.smartViz import smart_viz

# Auto mode - AI selects best variables
smart_viz(df, mode='auto', max_plots=10)

# Manual mode - You choose variables
smart_viz(df, mode='manual', columns=['col1', 'col2', 'col3'])

# With target variable for supervised analysis
smart_viz(df, mode='auto', target='target_column')
```

### Advanced Visualizations

```python
from essentiax.visuals.advanced_viz import advanced_viz

# 3D Scatter with Clustering
advanced_viz(df, viz_type='3d_scatter', columns=['x', 'y', 'z'], n_clusters=3)

# 3D Surface Plot
advanced_viz(df, viz_type='3d_surface', x_col='x', y_col='y', z_col='z')

# Sunburst Chart (Hierarchical)
advanced_viz(df, viz_type='sunburst', path_columns=['category', 'subcategory', 'item'])

# Sankey Diagram (Flow)
advanced_viz(df, viz_type='sankey', source_col='from', target_col='to')

# Advanced Violin Plots
advanced_viz(df, viz_type='violin', columns=['col1', 'col2', 'col3'])

# Parallel Coordinates
advanced_viz(df, viz_type='parallel', columns=['col1', 'col2', 'col3'], color_col='target')

# Treemap
advanced_viz(df, viz_type='treemap', path_columns=['category', 'subcategory'])

# Animated Scatter
advanced_viz(df, viz_type='animated', x_col='x', y_col='y', animation_col='time')

# Advanced Correlation
advanced_viz(df, viz_type='correlation', columns=['col1', 'col2', 'col3'])

# Ridge Plot
advanced_viz(df, viz_type='ridge', column='value', group_by='category')

# Auto mode - Creates multiple stunning visualizations
advanced_viz(df, viz_type='auto')
```

### Complete EDA

```python
from essentiax.eda.smart_eda import smart_eda

# Console output only
smart_eda(df, mode='console')

# Interactive plots only
smart_eda(df, mode='plots')

# HTML report only
smart_eda(df, mode='html', output_file='report.html')

# Everything!
smart_eda(df, mode='all', target='target_column', output_file='full_report.html')
```

## 🔍 What Makes This Special

### Before the Fix:
❌ Plotly graphs were invisible in Colab
❌ Only Rich console output showed up
❌ Frustrating debugging experience

### After the Fix:
✅ All Plotly graphs render perfectly
✅ Rich console output displays beautifully
✅ Seamless integration of text and interactive plots
✅ Works automatically - no configuration needed

## 💡 Pro Tips for Colab

### 1. Large Datasets
```python
# EssentiaX automatically samples large datasets
smart_viz(df, mode='auto', sample_size=5000)  # Limit to 5000 rows
```

### 2. Control Plot Count
```python
# Limit number of plots for faster execution
smart_viz(df, mode='auto', max_plots=5)
```

### 3. Interactive vs Static
```python
# Interactive Plotly (default)
smart_viz(df, mode='auto', interactive=True)

# Static Matplotlib (faster for many plots)
smart_viz(df, mode='auto', interactive=False)
```

### 4. Target Variable Analysis
```python
# Analyze relationships with target variable
smart_viz(df, mode='auto', target='price')
```

### 5. Save HTML Reports
```python
# Generate downloadable HTML report
smart_eda(df, mode='html', output_file='analysis.html')

# Download in Colab
from google.colab import files
files.download('analysis.html')
```

## 🎯 Common Use Cases

### 1. Quick Data Exploration
```python
# Just uploaded a CSV? Start here:
df = pd.read_csv('data.csv')
smart_viz(df, mode='auto', max_plots=5)
```

### 2. Feature Analysis for ML
```python
# Analyze features before modeling
smart_eda(df, mode='all', target='target_variable')
```

### 3. Presentation-Ready Visualizations
```python
# Create stunning 3D visualizations
advanced_viz(df, viz_type='3d_scatter', columns=['feature1', 'feature2', 'feature3'])
```

### 4. Distribution Analysis
```python
# Focus on distributions
smart_viz(df, mode='manual', columns=['numeric_col1', 'numeric_col2'])
```

### 5. Categorical Analysis
```python
# Analyze categorical variables
smart_viz(df, mode='manual', columns=['category1', 'category2', 'category3'])
```

## 🐛 Troubleshooting

### Issue: Plots still not showing
**Solution**: Make sure you're using the latest version
```python
!pip install --upgrade essentiax
```

### Issue: Import errors
**Solution**: Restart runtime and reinstall
```python
# In Colab: Runtime → Restart runtime
!pip uninstall essentiax -y
!pip install essentiax
```

### Issue: Memory errors with large datasets
**Solution**: Use sampling
```python
smart_viz(df, mode='auto', sample_size=10000)
```

### Issue: Too many plots
**Solution**: Limit plot count
```python
smart_viz(df, mode='auto', max_plots=3)
```

## 📚 Additional Resources

- **GitHub**: [Your GitHub URL]
- **PyPI**: https://pypi.org/project/essentiax/
- **Documentation**: [Your docs URL]
- **Examples**: See `COLAB_PLOTLY_TEST.py` for complete test suite

## 🎉 What You Get

✅ **Automatic Environment Detection**: Works in Colab, Jupyter, IPython
✅ **Buffer Management**: No stream conflicts between Rich and Plotly
✅ **Reliable Rendering**: Uses IPython.display.display() for guaranteed visibility
✅ **Fallback Mechanisms**: Multiple fallbacks ensure plots always render
✅ **Zero Configuration**: Just import and use
✅ **Production Ready**: Tested and documented

## 🚀 Ready to Use!

Your EssentiaX library now provides the best data visualization experience in Google Colab:

1. **Beautiful Console Output**: Rich formatting with colors, tables, and panels
2. **Interactive Plots**: Fully visible and functional Plotly graphs
3. **AI Insights**: Intelligent analysis and recommendations
4. **Zero Hassle**: Works automatically without configuration

Start exploring your data with confidence! 🎨📊✨
