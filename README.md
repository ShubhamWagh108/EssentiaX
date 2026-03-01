# 🎨 EssentiaX - Next-Generation Data Analysis Library

> **Smart EDA, Cleaning, and Visualization with AI-Powered Insights**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.5-green.svg)](https://github.com/ShubhamWagh/EssentiaX)

## 🚀 What Makes EssentiaX Special?

EssentiaX is not just another data analysis library. It's a **next-generation toolkit** that combines:

- 🤖 **AI-Powered Variable Selection** - Let AI choose the best variables to visualize
- 🎨 **Stunning Interactive Visualizations** - Beautiful Plotly charts with insights
- 🧠 **Smart Insights Generation** - Automatic interpretation of every chart
- 🧹 **Intelligent Data Cleaning** - One-function ML-ready preprocessing
- 📊 **Professional EDA Reports** - HTML reports that impress stakeholders
- 💡 **ML Model Recommendations** - Get model suggestions based on your data

## 🎯 Quick Start

```bash
pip install essentiax
```

```python
from essentiax import smart_read, smart_viz, smart_clean, problem_card

# 1. Load data with beautiful console output
df = smart_read("your_data.csv")

# 2. AI-powered visualization with insights
smart_viz(df, mode="auto", interactive=True)

# 3. Get ML insights and model recommendations
problem_card(df, target="your_target_column")

# 4. Clean data for ML in one line
clean_df = smart_clean(df)
```

## 🎨 Smart Visualization Engine

### 🤖 Automatic Mode (AI Selection)
Let AI choose the best variables and create stunning visualizations:

```python
smart_viz(
    df=df,
    mode="auto",           # AI selects best variables
    target="target_col",   # Optional target variable
    max_plots=8,          # Control number of plots
    interactive=True      # Beautiful interactive charts
)
```

**What you get:**
- 📊 **Smart Variable Selection** - AI picks the most informative variables
- 🔥 **Interactive Correlation Heatmaps** - Hover for detailed insights
- 📈 **Distribution Analysis** - With statistical interpretations
- 🎯 **Multi-variable Relationships** - Scatter plot matrices
- 💡 **AI-Generated Insights** - Automatic interpretation of every chart

### 👤 Manual Mode (User Selection)
Choose specific variables you want to analyze:

```python
smart_viz(
    df=df,
    mode="manual",
    columns=["age", "salary", "department"],  # Your chosen variables
    target="promotion",
    interactive=True
)
```

### 🎨 Features That Make It GOATED

#### 1. **AI-Powered Insights** 🧠
Every chart comes with automatic interpretation:
- Statistical significance analysis
- Pattern recognition
- Outlier detection
- Correlation explanations
- Feature engineering suggestions

#### 2. **Interactive Visualizations** ⚡
- **Plotly-powered** interactive charts
- Hover for detailed information
- Zoom, pan, and explore your data
- Professional styling that impresses

#### 3. **Beautiful Console UI** 🎨
- Rich console output with colors and formatting
- Progress bars and spinners
- Organized panels and tables
- Professional presentation

#### 4. **Smart Chart Selection** 🎯
AI automatically chooses the best chart type:
- Distribution plots for continuous variables
- Box plots for outlier detection
- Correlation heatmaps for relationships
- Categorical analysis for discrete variables
- Scatter matrices for multi-variable analysis

## 🎨 NEW: Advanced 3D & Interactive Visualizations

**Transform boring charts into stunning visualizations!**

```python
from essentiax.visuals import advanced_viz

# Auto mode - AI selects best advanced visualizations
advanced_viz(df, viz_type='auto')
```

### 🚀 10 Advanced Visualization Types

| Visualization | Description | Perfect For |
|--------------|-------------|-------------|
| 🎨 **3D Scatter + Clustering** | 3D scatter with K-means clustering | Multi-dimensional patterns |
| 🌊 **3D Surface Plot** | Beautiful 3D surfaces | Density visualization |
| ☀️ **Sunburst Chart** | Hierarchical circular viz | Categorical hierarchy |
| 🌊 **Sankey Diagram** | Flow visualization | Process flows |
| 🎻 **Advanced Violin Plots** | Distribution + statistics | Feature comparison |
| 📊 **Parallel Coordinates** | Multi-dimensional data | High-dim exploration |
| 🗺️ **Treemap** | Hierarchical rectangles | Category proportions |
| 🎬 **Animated Scatter** | Time-series animations | Temporal analysis |
| 🎭 **Advanced Correlation** | Interactive correlation | Feature relationships |
| 🏔️ **Ridge Plot** | Overlapping distributions | Category comparison |

### ✨ Key Features

- ✅ **Fully Interactive** - Hover, zoom, pan, rotate
- ✅ **3D Capabilities** - True 3D with rotation
- ✅ **Auto Clustering** - K-means built-in
- ✅ **Production Ready** - Beautiful aesthetics
- ✅ **One-Line Usage** - Simple API

### 📝 Quick Examples

```python
from essentiax.visuals import Advanced3DViz

engine = Advanced3DViz()

# 3D scatter with clustering
engine.plot_3d_scatter_clusters(df, n_clusters=3)

# 3D surface plot
engine.plot_3d_surface(df, x_col='feature1', y_col='feature2')

# Advanced violin plots
engine.plot_violin_advanced(df, columns=['f1', 'f2', 'f3'])

# Parallel coordinates
engine.plot_parallel_coordinates(df, color_col='target')

# Sunburst chart
engine.plot_sunburst(df, path_columns=['cat1', 'cat2'])
```

**See `ADVANCED_VIZ_GUIDE.md` for complete documentation!**

## 🧹 Smart Data Cleaning

Transform messy data into ML-ready datasets:

```python
clean_df = smart_clean(
    df,
    missing_strategy="auto",    # Smart missing value handling
    outlier_strategy="iqr",     # Intelligent outlier removal
    scale_numeric=True,         # Automatic scaling
    encode_categorical=True,    # Smart encoding
    verbose=True               # Beautiful progress output
)
```

## 📊 Problem Card & Model Recommendations

Get instant ML insights:

```python
problem_card(df, target="your_target")
```

**Provides:**
- 🎯 **Problem Type Detection** (Classification/Regression/NLP)
- 🤖 **Model Recommendations** (Baseline + Advanced)
- ⚖️ **Class Imbalance Analysis**
- 🔍 **Data Quality Score**
- 💡 **Actionable Insights**

## 📈 Professional EDA Reports

Generate stunning HTML reports:

```python
from essentiax import smart_eda_pro

smart_eda_pro(
    df, 
    target="target_column",
    report_path="my_analysis.html"
)
```

## 🎯 Real-World Examples

### Example 1: Sales Data Analysis
```python
# Load sales data
df = smart_read("sales_data.csv")

# AI-powered visualization
smart_viz(df, mode="auto", target="revenue")

# Get insights and recommendations
problem_card(df, target="revenue")
```

### Example 2: Customer Segmentation
```python
# Manual analysis of specific features
smart_viz(
    df=customer_df,
    mode="manual", 
    columns=["age", "income", "spending_score", "loyalty_years"],
    interactive=True
)
```

### Example 3: ML Pipeline
```python
# Complete ML preprocessing pipeline
df = smart_read("dataset.csv")
problem_card(df, target="target")
clean_df = smart_clean(df)
# Now ready for model training!
```

## 🆚 Why Choose EssentiaX?

| Feature | EssentiaX | pandas-profiling | sweetviz |
|---------|-----------|------------------|----------|
| AI Variable Selection | ✅ | ❌ | ❌ |
| Interactive Charts | ✅ | ❌ | ❌ |
| Real-time Insights | ✅ | ❌ | ❌ |
| ML Recommendations | ✅ | ❌ | ❌ |
| Beautiful Console UI | ✅ | ❌ | ❌ |
| One-line Cleaning | ✅ | ❌ | ❌ |

## 📦 Installation

```bash
# Basic installation
pip install essentiax

# With all dependencies
pip install essentiax[complete]
```

## 🛠️ Requirements

- Python 3.7+
- pandas >= 1.0
- numpy >= 1.20
- matplotlib >= 3.0
- seaborn >= 0.11
- plotly >= 5.0
- rich >= 10.0
- scikit-learn >= 1.0

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ by [Shubham Wagh](https://github.com/ShubhamWagh)
- Powered by the amazing Python data science ecosystem
- Special thanks to the Plotly and Rich communities

---

**⭐ Star this repo if EssentiaX helps you build better ML models!**

[🔗 GitHub](https://github.com/ShubhamWagh/EssentiaX) | [📧 Contact](mailto:waghshubham197@gmail.com) | [🐦 Twitter](https://twitter.com/your_handle)