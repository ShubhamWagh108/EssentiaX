# Unified Smart Visualization Guide

## 🎨 ONE Function for Everything

EssentiaX now has **ONE unified function** - `smart_viz()` - that handles both basic 2D and advanced 3D visualizations!

---

## 🚀 Quick Start

```python
from essentiax.visuals import smart_viz
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Create advanced 3D visualizations (default)
smart_viz(df, mode='auto', viz_type='advanced', dark_theme=True)
```

That's it! ONE function, professional results.

---

## 📊 Function Parameters

```python
smart_viz(
    df,                    # Your DataFrame
    mode='auto',           # 'auto' or 'manual'
    columns=None,          # List of columns (for manual mode)
    target=None,           # Target variable
    viz_type='advanced',   # 'advanced' or 'basic'
    dark_theme=True        # True (dark) or False (light)
)
```

### Parameters Explained

| Parameter | Options | Description |
|-----------|---------|-------------|
| `mode` | `'auto'` or `'manual'` | Auto: AI selects variables<br>Manual: You specify columns |
| `viz_type` | `'advanced'` or `'basic'` | Advanced: 3D professional charts<br>Basic: Simple 2D charts |
| `dark_theme` | `True` or `False` | True: Dark theme (presentations)<br>False: Light theme (reports) |
| `columns` | `list` or `None` | Required for manual mode |
| `target` | `str` or `None` | Target variable for analysis |

---

## 🎨 Visualization Types

### Advanced 3D Visualizations (viz_type='advanced')

Creates **4+ professional 3D charts**:

1. **3D Bubble Scatter** - Multi-dimensional structure analytics
2. **Sunburst Hierarchy** - Interactive nested circular charts
3. **Correlation Heatmap Pro** - Annotated correlation matrix
4. **Scatter Matrix Pro** - Multi-variable comparison with color coding
5. **Distribution Pro** - Histograms with mean/median lines and statistics

### Basic 2D Visualizations (viz_type='basic')

Creates standard charts:

1. Distribution plots
2. Correlation heatmap
3. Categorical bar charts
4. Scatter matrix

---

## 💡 Usage Examples

### Example 1: Auto Mode + Advanced (Recommended)

```python
from essentiax.visuals import smart_viz

# AI selects best variables and creates 3D visualizations
smart_viz(df, mode='auto', viz_type='advanced', dark_theme=True)
```

**Output**:
- ✅ Visualization Setup panel
- ✅ AI Variable Selection
- ✅ 3D Bubble Scatter
- ✅ Sunburst Hierarchy
- ✅ Correlation Heatmap Pro
- ✅ Scatter Matrix Pro
- ✅ Distribution with Statistics
- ✅ Statistical Summaries
- ✅ Distribution Insights
- ✅ Visualization Summary

---

### Example 2: Manual Mode + Advanced

```python
# You specify which columns to visualize
smart_viz(
    df,
    mode='manual',
    columns=['age', 'income', 'score', 'category', 'type'],
    viz_type='advanced',
    dark_theme=True
)
```

---

### Example 3: Basic 2D Visualizations

```python
# Simple 2D charts for quick analysis
smart_viz(df, mode='auto', viz_type='basic', dark_theme=False)
```

---

### Example 4: With Target Variable

```python
# Supervised analysis with target
smart_viz(
    df,
    mode='auto',
    target='outcome',
    viz_type='advanced',
    dark_theme=True
)
```

---

## 🎨 Theme Options

### Dark Theme (Recommended for Presentations)

```python
smart_viz(df, dark_theme=True)
```

**Features**:
- Background: `#1a1a1a`
- Grid: `#333333`
- Text: White
- Modern aesthetic
- Professional look

### Light Theme (For Reports/Printing)

```python
smart_viz(df, dark_theme=False)
```

**Features**:
- Background: White
- Grid: Light gray
- Text: Black
- Traditional styling
- Print-friendly

---

## 📊 Rich Console Output

The function provides comprehensive console output:

### 1. Visualization Setup
```
┌─────────────────────────────────────┐
│     📊 Visualization Setup          │
├─────────────────────────────────────┤
│ Dataset Shape    │ 500 × 9          │
│ Mode             │ AUTO             │
│ Visualization    │ ADVANCED         │
│ Theme            │ 🌙 Dark          │
│ Target Variable  │ None             │
└─────────────────────────────────────┘
```

### 2. AI Variable Selection (Auto Mode)
```
┌─────────────────────────────────────┐
│     🤖 AI Variable Selection        │
├─────────────────────────────────────┤
│ Numeric      │ age, income, score   │
│ Categorical  │ category, type       │
│ Correlations │ 3 pairs found        │
└─────────────────────────────────────┘
```

### 3. Statistical Summary (Per Chart)
```
┌─────────────────────────────────────┐
│  📊 Statistical Summary: age        │
├─────────────────────────────────────┤
│ Count    │ 500                      │
│ Mean     │ 48.52                    │
│ Median   │ 49.00                    │
│ Std Dev  │ 17.89                    │
│ Min      │ 18.00                    │
│ Max      │ 79.00                    │
│ Skewness │ -0.023                   │
│ Kurtosis │ -1.201                   │
└─────────────────────────────────────┘
```

### 4. Distribution Insights
```
┌─────────────────────────────────────┐
│   🔍 Distribution Insights          │
├─────────────────────────────────────┤
│ 📊 Symmetric Distribution           │
│ 🎯 Unimodal Distribution            │
│ ✅ Clean Distribution: 2.1% outliers│
│ 🎯 Balanced Central Tendency        │
└─────────────────────────────────────┘
```

### 5. Visualization Summary
```
┌─────────────────────────────────────┐
│   🎉 Visualization Summary          │
├─────────────────────────────────────┤
│ ✨ Visualization Complete!          │
│                                     │
│ 📊 Total Plots: 5                   │
│ 🎯 Variables: 7                     │
│ 🔍 Insights: 10+                    │
│ ⚡ Type: ADVANCED                   │
│ 🎨 Theme: Dark                      │
│                                     │
│ 💡 Next Steps:                      │
│ • Feature engineering               │
│ • Data cleaning                     │
│ • Model selection                   │
└─────────────────────────────────────┘
```

---

## 🎯 Advanced 3D Visualizations Details

### 1. 3D Bubble Scatter

**What it shows**: Multi-dimensional relationships with bubble sizes and colors

**Features**:
- X, Y, Z axes for 3 variables
- Bubble size represents magnitude
- Color gradient for additional dimension
- Interactive 3D rotation
- Hover tooltips

**Use cases**:
- YouTube analytics (sentiment × engagement × title_length)
- Product analysis (price × rating × reviews)
- Performance metrics

---

### 2. Sunburst Hierarchy

**What it shows**: Hierarchical category distribution

**Features**:
- Multi-level nested circles
- Click to zoom into segments
- Color-coded by metric
- Interactive exploration

**Use cases**:
- Category breakdowns
- Hierarchical data
- Nested classifications

---

### 3. Correlation Heatmap Pro

**What it shows**: Feature relationships and correlations

**Features**:
- Annotated correlation values
- Red-Blue diverging colors
- Strong correlation highlights
- Professional styling

**Use cases**:
- Feature selection
- Multicollinearity detection
- Relationship analysis

---

### 4. Scatter Matrix Pro

**What it shows**: All-vs-all variable relationships

**Features**:
- Multiple scatter plots
- Color-coded by metric
- Interactive brushing
- Pattern identification

**Use cases**:
- Multi-variable exploration
- Relationship discovery
- Outlier detection

---

### 5. Distribution Pro

**What it shows**: Data distribution with statistics

**Features**:
- Histogram with bins
- Mean line (dashed)
- Median line (dotted)
- Statistical summary table
- Distribution insights

**Use cases**:
- Understanding data spread
- Identifying outliers
- Checking normality

---

## 🔧 Advanced Usage

### Large Datasets

```python
# Automatically samples to 10,000 rows
smart_viz(df, mode='auto', viz_type='advanced')

# Custom sample size
smart_viz(df, mode='auto', viz_type='advanced', sample_size=5000)
```

### Specific Columns Only

```python
# Analyze only specific columns
smart_viz(
    df,
    mode='manual',
    columns=['col1', 'col2', 'col3'],
    viz_type='advanced'
)
```

### With Target Variable

```python
# Supervised analysis
smart_viz(
    df,
    mode='auto',
    target='outcome',
    viz_type='advanced'
)
```

---

## 📈 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| **Dimensions** | 2D | 3D |
| **Interactivity** | Standard | Enhanced |
| **Styling** | Simple | Professional |
| **Statistics** | Basic | Comprehensive |
| **Insights** | Limited | Detailed |
| **Theme** | Light only | Dark + Light |
| **Charts** | 4 types | 5+ types |
| **Use Case** | Quick analysis | Presentations |

---

## 💡 Best Practices

### 1. Start with Auto + Advanced

```python
# Best for exploration
smart_viz(df, mode='auto', viz_type='advanced', dark_theme=True)
```

### 2. Use Manual for Specific Analysis

```python
# When you know what to analyze
smart_viz(df, mode='manual', columns=['x', 'y', 'z'], viz_type='advanced')
```

### 3. Dark Theme for Presentations

```python
# Professional look
smart_viz(df, dark_theme=True)
```

### 4. Light Theme for Reports

```python
# Print-friendly
smart_viz(df, dark_theme=False)
```

---

## 🎓 Complete Example

```python
import pandas as pd
import numpy as np
from essentiax.visuals import smart_viz

# Create sample data
df = pd.DataFrame({
    'age': np.random.randint(18, 80, 500),
    'income': np.random.normal(50000, 15000, 500),
    'score': np.random.uniform(0, 100, 500),
    'category': np.random.choice(['A', 'B', 'C'], 500),
    'type': np.random.choice(['X', 'Y'], 500)
})

# ONE function call - creates everything!
smart_viz(
    df,
    mode='auto',           # AI selects variables
    viz_type='advanced',   # 3D professional charts
    dark_theme=True        # Dark theme
)
```

**Output**: Complete professional analytics dashboard with:
- 3D visualizations
- Statistical summaries
- Distribution insights
- Correlation analysis
- Rich console output

---

## 🎉 Summary

### Why ONE Function?

✅ **Simplicity** - One function to remember  
✅ **Flexibility** - Auto or manual, basic or advanced  
✅ **Professional** - Production-ready visualizations  
✅ **Comprehensive** - Statistics + insights + charts  
✅ **Beautiful** - Dark theme + rich console output  

### Quick Reference

```python
# Default (recommended)
smart_viz(df)

# Full control
smart_viz(
    df,
    mode='auto',           # or 'manual'
    columns=None,          # or ['col1', 'col2']
    target=None,           # or 'target_col'
    viz_type='advanced',   # or 'basic'
    dark_theme=True        # or False
)
```

---

**ONE function. Infinite possibilities. Professional results.** 🚀
