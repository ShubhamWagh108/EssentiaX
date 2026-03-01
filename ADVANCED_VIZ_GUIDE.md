# 🎨 EssentiaX Advanced Visualization Guide

## Overview

EssentiaX now includes **stunning 3D and interactive visualizations** that go far beyond basic charts. All visualizations are fully interactive with Plotly.

## Quick Start

```python
from essentiax.visuals import advanced_viz

# Auto mode - AI selects best visualizations
advanced_viz(df, viz_type='auto')
```

## Available Visualization Types

### 1. 🎨 3D Scatter with Clustering

Creates beautiful 3D scatter plots with automatic K-means clustering.

```python
from essentiax.visuals import Advanced3DViz

engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(
    df,
    columns=['feature1', 'feature2', 'feature3'],
    n_clusters=3,
    title='My 3D Analysis'
)
```

**Features:**
- Automatic clustering with K-means
- Interactive 3D rotation
- Cluster centroids highlighted
- Color-coded clusters
- Hover for details

---

### 2. 🌊 3D Surface Plot

Creates stunning 3D surface plots for density or relationship visualization.

```python
# Density surface (auto-calculated)
engine.plot_3d_surface(
    df,
    x_col='feature1',
    y_col='feature2'
)

# Or with explicit Z values
engine.plot_3d_surface(
    df,
    x_col='feature1',
    y_col='feature2',
    z_col='target'
)
```

**Features:**
- Smooth 3D surfaces
- Density estimation
- Interactive rotation
- Beautiful color gradients

---

### 3. ☀️ Sunburst Chart

Hierarchical visualization perfect for categorical data.

```python
engine.plot_sunburst(
    df,
    path_columns=['category1', 'category2', 'category3'],
    value_column='amount'  # Optional
)
```

**Features:**
- Multi-level hierarchy
- Interactive drill-down
- Percentage calculations
- Beautiful color schemes

---

### 4. 🌊 Sankey Diagram

Flow visualization showing relationships between categories.

```python
engine.plot_sankey(
    df,
    source_col='from_category',
    target_col='to_category',
    value_col='flow_amount'  # Optional
)
```

**Features:**
- Flow visualization
- Automatic node positioning
- Interactive hover
- Clear relationship mapping

---

### 5. 🎻 Advanced Violin Plots

Distribution comparison with statistical overlays.

```python
engine.plot_violin_advanced(
    df,
    columns=['feature1', 'feature2', 'feature3']
)
```

**Features:**
- Box plot overlay
- Mean line
- Full distribution shape
- Multiple variables comparison

---

### 6. 📊 Parallel Coordinates

Multi-dimensional data visualization.

```python
engine.plot_parallel_coordinates(
    df,
    color_col='target',
    columns=['f1', 'f2', 'f3', 'f4']
)
```

**Features:**
- Multi-dimensional analysis
- Color-coded by category
- Interactive filtering
- Pattern detection

---

### 7. 🗺️ Treemap

Hierarchical data as nested rectangles.

```python
engine.plot_treemap(
    df,
    path_columns=['category1', 'category2'],
    value_column='size'
)
```

**Features:**
- Hierarchical structure
- Size-based visualization
- Interactive zoom
- Hover details

---

### 8. 🎬 Animated Scatter

Time-series or sequential data animation.

```python
engine.plot_animated_scatter(
    df,
    x_col='feature1',
    y_col='feature2',
    animation_col='year',
    size_col='population',
    color_col='category'
)
```

**Features:**
- Smooth animations
- Play/pause controls
- Bubble sizing
- Color coding

---

### 9. 🎭 Advanced Correlation

Beautiful correlation matrix with network visualization.

```python
engine.plot_correlation_chord(
    df,
    columns=['f1', 'f2', 'f3', 'f4'],
    threshold=0.5
)
```

**Features:**
- Interactive heatmap
- Strong correlation highlighting
- Network-style connections
- Threshold filtering

---

### 10. 🏔️ Ridge Plot

Distribution comparison across categories (joyplot).

```python
engine.plot_ridge(
    df,
    column='value',
    group_by='category'
)
```

**Features:**
- Overlapping distributions
- Category comparison
- Beautiful layering
- Clear separation

---

## Complete Example

```python
from essentiax.visuals import advanced_viz, Advanced3DViz
import pandas as pd
from sklearn.datasets import load_wine

# Load data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Method 1: Auto mode (easiest)
advanced_viz(df, viz_type='auto')

# Method 2: Specific visualization
advanced_viz(df, viz_type='3d_scatter', 
            columns=['alcohol', 'flavanoids', 'color_intensity'],
            n_clusters=3)

# Method 3: Using engine directly (most control)
engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(df, n_clusters=3)
engine.plot_violin_advanced(df, columns=['alcohol', 'malic_acid'])
engine.plot_parallel_coordinates(df, color_col='target')
```

## All Viz Types

| Type | Description | Best For |
|------|-------------|----------|
| `auto` | AI-powered selection | Quick exploration |
| `3d_scatter` | 3D scatter + clustering | Multi-dimensional patterns |
| `3d_surface` | 3D surface plot | Density visualization |
| `sunburst` | Hierarchical sunburst | Categorical hierarchy |
| `sankey` | Flow diagram | Category relationships |
| `violin` | Advanced violin plots | Distribution comparison |
| `parallel` | Parallel coordinates | Multi-dimensional data |
| `treemap` | Hierarchical treemap | Nested categories |
| `animated` | Animated scatter | Time-series data |
| `correlation` | Advanced correlation | Feature relationships |
| `ridge` | Ridge plot | Category distributions |

## Interactive Features

All visualizations support:
- ✅ **Hover** - Detailed information on hover
- ✅ **Zoom** - Zoom in/out with mouse wheel
- ✅ **Pan** - Click and drag to pan
- ✅ **Rotate** - 3D plots can be rotated
- ✅ **Export** - Save as PNG/SVG
- ✅ **Filter** - Click legend to filter

## Tips for Best Results

1. **3D Plots**: Work best with 3-6 features
2. **Hierarchical**: Need 2+ categorical columns
3. **Animated**: Requires sequential/time column
4. **Parallel**: Limit to 8 columns for readability
5. **Auto Mode**: Great for initial exploration

## Comparison: Basic vs Advanced

### Before (Basic)
```python
from essentiax.visuals import smart_viz
smart_viz(df)  # Basic histograms and bar charts
```

### After (Advanced)
```python
from essentiax.visuals import advanced_viz
advanced_viz(df, viz_type='auto')  # 3D, interactive, stunning!
```

## Performance Notes

- Large datasets (>10K rows) are automatically sampled
- 3D plots work best with <50K points
- Animations may be slow with >100 frames
- All plots are optimized for Jupyter/Colab

## Requirements

All dependencies are included in EssentiaX:
- plotly >= 5.0
- scipy >= 1.7
- scikit-learn >= 1.0
- pandas >= 1.0
- numpy >= 1.20

## Next Steps

1. Try `advanced_viz(df, viz_type='auto')` on your data
2. Experiment with specific viz types
3. Customize with parameters
4. Export for presentations
5. Share interactive HTML files

---

**Made with ❤️ by EssentiaX**

*Transform boring charts into stunning visualizations!*
