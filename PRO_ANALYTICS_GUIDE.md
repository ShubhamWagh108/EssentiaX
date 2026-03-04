# Professional Analytics Visualizations Guide

## 🎨 YouTube-Style Analytics Dashboard

EssentiaX now includes professional, production-ready visualizations inspired by YouTube Analytics, Google Analytics, and modern data dashboards.

---

## 🚀 Quick Start

```python
from essentiax.visuals.pro_analytics_viz import (
    create_3d_bubble_scatter,
    create_sunburst_hierarchy,
    create_correlation_heatmap_pro,
    create_temporal_trend_bubble,
    create_scatter_matrix_pro,
    pro_analytics_dashboard
)

import pandas as pd

# Your data
df = pd.read_csv('your_data.csv')

# Create professional visualizations
create_3d_bubble_scatter(
    df,
    x_col='sentiment',
    y_col='engagement',
    z_col='title_length',
    size_col='views',
    color_col='views',
    dark_theme=True
)
```

---

## 📊 Available Visualizations

### 1. 3D Bubble Scatter
**Perfect for**: Multi-dimensional analysis, YouTube-style structure analytics

```python
create_3d_bubble_scatter(
    df,
    x_col='sentiment',           # X axis
    y_col='engagement_rate',     # Y axis
    z_col='title_length',        # Z axis
    size_col='views',            # Bubble size
    color_col='views',           # Color gradient
    title="YouTube Structure Analytics",
    dark_theme=True
)
```

**Features**:
- Interactive 3D rotation
- Bubble sizes represent magnitude
- Color gradients for additional dimension
- Hover tooltips with detailed info
- Professional dark theme

---

### 2. Sunburst Hierarchy Chart
**Perfect for**: Hierarchical data, category breakdowns, nested distributions

```python
create_sunburst_hierarchy(
    df,
    path_columns=['sentiment', 'category', 'subcategory'],  # Hierarchy levels
    value_column='views',                                    # Size of segments
    color_column='engagement_rate',                          # Color metric
    title="Hierarchical View Distribution",
    dark_theme=True
)
```

**Features**:
- Multi-level hierarchy visualization
- Click to zoom into segments
- Color-coded by any metric
- Beautiful radial layout
- Interactive exploration

---

### 3. Professional Correlation Heatmap
**Perfect for**: Feature relationships, "what drives what" analysis

```python
create_correlation_heatmap_pro(
    df,
    columns=['views', 'likes', 'comments', 'sentiment'],
    title="Feature Correlation Matrix",
    dark_theme=True,
    annotate=True  # Show correlation values
)
```

**Features**:
- Annotated correlation values
- Red-Blue diverging colorscale
- Professional styling
- Easy to identify strong correlations
- Dark theme optimized

---

### 4. Temporal Trend Bubble Chart
**Perfect for**: Time series analysis, views over time, trend identification

```python
create_temporal_trend_bubble(
    df,
    time_col='publish_date',
    y_col='views',
    size_col='engagement',
    color_col='sentiment',
    title="Temporal Trend Analysis",
    dark_theme=True
)
```

**Features**:
- Time-based scatter plot
- Bubble sizes for additional metric
- Color coding for sentiment/category
- Hover for detailed timestamps
- Trend identification

---

### 5. Scatter Matrix (SPLOM)
**Perfect for**: Multi-variable comparison, all-vs-all relationships

```python
create_scatter_matrix_pro(
    df,
    columns=['views', 'likes', 'sentiment', 'engagement'],
    color_col='engagement_rate',
    title="Multi-Variable Scatter Matrix",
    dark_theme=True
)
```

**Features**:
- All-vs-all scatter plots
- Color-coded by metric
- Interactive brushing
- Identify patterns across variables
- Professional layout

---

### 6. Distribution Histogram Pro
**Perfect for**: Understanding data distribution, identifying outliers

```python
create_distribution_histogram_pro(
    df,
    column='engagement_rate',
    bins=50,
    title="Engagement Rate Distribution",
    dark_theme=True
)
```

**Features**:
- Mean and median lines
- Statistical annotations
- Professional styling
- Customizable bins
- Dark theme optimized

---

### 7. Category Bar Chart Pro
**Perfect for**: Category comparisons, top performers

```python
create_category_bar_pro(
    df,
    category_col='category',
    value_col='views',
    top_n=15,
    title="Top Categories by Views",
    dark_theme=True,
    horizontal=False
)
```

**Features**:
- Color gradient by value
- Show top N categories
- Horizontal or vertical
- Value annotations
- Professional styling

---

### 8. Complete Analytics Dashboard
**Perfect for**: Automatic comprehensive analysis

```python
pro_analytics_dashboard(
    df,
    config='auto',  # Automatically creates relevant visualizations
    dark_theme=True
)
```

**Features**:
- Automatic visualization selection
- Multiple charts in sequence
- Comprehensive analysis
- Professional presentation
- One-line solution

---

## 🎨 Styling Options

### Dark Theme (Default)
```python
dark_theme=True  # Professional dark background
```

**Features**:
- Background: `#1a1a1a`
- Grid: `#333333`
- Text: White
- Optimized for presentations
- Modern aesthetic

### Light Theme
```python
dark_theme=False  # Traditional light background
```

**Features**:
- White background
- Black text
- Traditional styling
- Print-friendly

---

## 💡 Use Cases

### YouTube Analytics Dashboard
```python
# 3D structure analysis
create_3d_bubble_scatter(
    df,
    x_col='sentiment',
    y_col='engagement_rate',
    z_col='title_length',
    size_col='views',
    color_col='views'
)

# Hierarchical breakdown
create_sunburst_hierarchy(
    df,
    path_columns=['sentiment_category', 'category', 'success_level'],
    value_column='views',
    color_column='engagement_rate'
)

# Temporal trends
create_temporal_trend_bubble(
    df,
    time_col='publish_date',
    y_col='views',
    size_col='engagement',
    color_col='sentiment'
)
```

### E-commerce Analytics
```python
# Product performance
create_3d_bubble_scatter(
    df,
    x_col='price',
    y_col='rating',
    z_col='reviews_count',
    size_col='sales',
    color_col='profit_margin'
)

# Category hierarchy
create_sunburst_hierarchy(
    df,
    path_columns=['department', 'category', 'subcategory'],
    value_column='revenue',
    color_column='profit_margin'
)
```

### Social Media Analytics
```python
# Engagement analysis
create_scatter_matrix_pro(
    df,
    columns=['likes', 'shares', 'comments', 'reach', 'engagement_rate'],
    color_col='sentiment'
)

# Temporal trends
create_temporal_trend_bubble(
    df,
    time_col='post_date',
    y_col='engagement',
    size_col='reach',
    color_col='sentiment'
)
```

---

## 🔧 Advanced Customization

### Custom Color Schemes
```python
# Modify in pro_analytics_viz.py
DARK_THEME = {
    'bgcolor': '#1a1a1a',      # Background color
    'gridcolor': '#333333',     # Grid lines
    'textcolor': '#ffffff',     # Text color
    'accent1': '#00d4ff',       # Accent color 1
    'accent2': '#ff6b6b',       # Accent color 2
    'accent3': '#4ecdc4',       # Accent color 3
    'accent4': '#ffe66d'        # Accent color 4
}
```

### Custom Colorscales
Available colorscales:
- `'Viridis'` - Purple to yellow
- `'Plasma'` - Purple to orange
- `'Inferno'` - Black to yellow
- `'Magma'` - Black to white
- `'Cividis'` - Blue to yellow (colorblind-friendly)
- `'RdBu_r'` - Red to blue (diverging)

---

## 📈 Performance Tips

### Large Datasets
```python
# Sample data for visualization
df_sample = df.sample(n=1000, random_state=42)

create_3d_bubble_scatter(
    df_sample,
    x_col='x',
    y_col='y',
    z_col='z',
    size_col='size',
    color_col='color'
)
```

### Optimize Scatter Matrix
```python
# Limit to 5-6 columns max
columns = df.select_dtypes(include=[np.number]).columns[:5]

create_scatter_matrix_pro(
    df,
    columns=columns,
    color_col='target'
)
```

---

## 🎯 Best Practices

### 1. Choose the Right Visualization
- **3D Bubble**: 4-5 dimensions, structure analysis
- **Sunburst**: Hierarchical data, nested categories
- **Heatmap**: Correlations, relationships
- **Temporal**: Time series, trends
- **Scatter Matrix**: Multi-variable exploration
- **Histogram**: Distribution analysis
- **Bar Chart**: Category comparison

### 2. Color Coding
- Use color for additional dimension
- Choose colorblind-friendly palettes
- Consistent color meaning across charts

### 3. Interactivity
- All charts are interactive by default
- Hover for details
- Zoom and pan
- Click to filter (sunburst)

### 4. Dark vs Light Theme
- **Dark**: Presentations, modern look
- **Light**: Reports, printing

---

## 🚀 Complete Example

```python
import pandas as pd
import numpy as np
from essentiax.visuals.pro_analytics_viz import *

# Load data
df = pd.read_csv('analytics_data.csv')

# 1. Overview with dashboard
pro_analytics_dashboard(df, config='auto', dark_theme=True)

# 2. Deep dive into structure
create_3d_bubble_scatter(
    df,
    x_col='metric1',
    y_col='metric2',
    z_col='metric3',
    size_col='importance',
    color_col='performance',
    title="Performance Structure Analysis",
    dark_theme=True
)

# 3. Hierarchical breakdown
create_sunburst_hierarchy(
    df,
    path_columns=['level1', 'level2', 'level3'],
    value_column='value',
    color_column='performance',
    title="Hierarchical Performance View",
    dark_theme=True
)

# 4. Correlation analysis
create_correlation_heatmap_pro(
    df,
    columns=['metric1', 'metric2', 'metric3', 'metric4'],
    title="Feature Correlations",
    dark_theme=True
)

# 5. Temporal trends
create_temporal_trend_bubble(
    df,
    time_col='date',
    y_col='performance',
    size_col='importance',
    color_col='category',
    title="Performance Over Time",
    dark_theme=True
)
```

---

## 📚 Resources

- **Demo Script**: `PRO_ANALYTICS_DEMO.py`
- **Source Code**: `essentiax/visuals/pro_analytics_viz.py`
- **Plotly Docs**: https://plotly.com/python/

---

## 🎉 Summary

EssentiaX Professional Analytics Visualizations provide:

✅ YouTube-style 3D bubble charts  
✅ Hierarchical sunburst diagrams  
✅ Professional correlation heatmaps  
✅ Temporal trend analysis  
✅ Multi-variable scatter matrices  
✅ Beautiful dark theme  
✅ Interactive and production-ready  
✅ One-line dashboard creation  

**Ready to create stunning analytics dashboards!** 🚀
