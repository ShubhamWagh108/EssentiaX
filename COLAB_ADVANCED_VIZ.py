"""
EssentiaX v1.1.0 - Advanced Visualization Demo for Google Colab
================================================================
Copy each cell into separate Colab cells for stunning visualizations!
"""

# ============================================================================
# CELL 1: Installation
# ============================================================================
"""
# 📦 Install EssentiaX v1.1.1
"""
!pip install --upgrade Essentiax

# Setup for Colab (ensures plots display properly)
from essentiax.visuals import setup_colab
setup_colab()

print("✅ EssentiaX v1.1.1 installed with Advanced Visualizations!")

# ============================================================================
# CELL 2: Load Sample Data
# ============================================================================
"""
# 📊 Load Wine Dataset
"""
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['target_name'] = df['target'].map({0: 'Class_0', 1: 'Class_1', 2: 'Class_2'})

print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
df.head()

# ============================================================================
# CELL 3: Auto Mode - AI Selects Best Visualizations (ONE LINE!)
# ============================================================================
"""
# 🎨 Auto Mode - ONE LINE!
"""
from essentiax.visuals import advanced_viz

advanced_viz(df, viz_type='auto')

# ============================================================================
# CELL 4: 3D Scatter with Clustering (STUNNING!)
# ============================================================================
"""
# 🎨 3D Scatter + Clustering
"""
from essentiax.visuals import Advanced3DViz

engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(
    df,
    columns=['alcohol', 'flavanoids', 'color_intensity'],
    n_clusters=3,
    title='🎨 Wine Chemical Analysis - 3D Clustering'
)

# ============================================================================
# CELL 5: 3D Surface Plot (BEAUTIFUL!)
# ============================================================================
"""
# 🌊 3D Surface Plot
"""
engine.plot_3d_surface(
    df,
    x_col='alcohol',
    y_col='flavanoids',
    title='🌊 Density Surface: Alcohol vs Flavanoids'
)

# ============================================================================
# CELL 6: Advanced Violin Plots
# ============================================================================
"""
# 🎻 Advanced Violin Plots
"""
engine.plot_violin_advanced(
    df,
    columns=['alcohol', 'malic_acid', 'ash', 'magnesium', 'proline'],
    title='🎻 Distribution Analysis - Top 5 Features'
)

# ============================================================================
# CELL 7: Parallel Coordinates (Multi-Dimensional)
# ============================================================================
"""
# 📊 Parallel Coordinates
"""
engine.plot_parallel_coordinates(
    df,
    color_col='target',
    columns=['alcohol', 'flavanoids', 'color_intensity', 'hue', 'proline'],
    title='📊 Multi-Dimensional Analysis - Colored by Wine Class'
)

# ============================================================================
# CELL 8: Sunburst Chart (Hierarchical)
# ============================================================================
"""
# ☀️ Sunburst Chart
"""
# Create categorical bins
df['alcohol_level'] = pd.cut(df['alcohol'], bins=3, labels=['Low', 'Medium', 'High'])
df['flavanoid_level'] = pd.cut(df['flavanoids'], bins=3, labels=['Low', 'Medium', 'High'])

engine.plot_sunburst(
    df,
    path_columns=['target_name', 'alcohol_level', 'flavanoid_level'],
    title='☀️ Wine Classification Hierarchy'
)

# ============================================================================
# CELL 9: Treemap
# ============================================================================
"""
# 🗺️ Interactive Treemap
"""
engine.plot_treemap(
    df,
    path_columns=['target_name', 'alcohol_level'],
    title='🗺️ Wine Distribution Treemap'
)

# ============================================================================
# CELL 10: Advanced Correlation Matrix
# ============================================================================
"""
# 🎭 Advanced Correlation
"""
engine.plot_correlation_chord(
    df,
    columns=['alcohol', 'malic_acid', 'ash', 'flavanoids', 'color_intensity', 'hue'],
    threshold=0.3,
    title='🎭 Correlation Network (|r| ≥ 0.3)'
)

# ============================================================================
# CELL 11: Ridge Plot
# ============================================================================
"""
# 🏔️ Ridge Plot
"""
engine.plot_ridge(
    df,
    column='alcohol',
    group_by='target_name',
    title='🏔️ Alcohol Distribution by Wine Class'
)

# ============================================================================
# CELL 12: One-Line Specific Visualizations
# ============================================================================
"""
# 🚀 One-Line Usage
"""
# 3D scatter
advanced_viz(df, viz_type='3d_scatter', 
            columns=['alcohol', 'flavanoids', 'color_intensity'],
            n_clusters=3)

# Violin plots
advanced_viz(df, viz_type='violin', 
            columns=['alcohol', 'malic_acid', 'ash'])

# Parallel coordinates
advanced_viz(df, viz_type='parallel',
            color_col='target',
            columns=['alcohol', 'flavanoids', 'hue'])

# ============================================================================
# CELL 13: Summary
# ============================================================================
"""
# 🎉 Summary
"""
print("""
✨ ADVANCED VISUALIZATIONS COMPLETE!

🎨 Visualizations Created:
1. ✅ Auto Mode (AI-powered)
2. ✅ 3D Scatter with Clustering
3. ✅ 3D Surface Plot
4. ✅ Advanced Violin Plots
5. ✅ Parallel Coordinates
6. ✅ Sunburst Chart
7. ✅ Interactive Treemap
8. ✅ Advanced Correlation Matrix
9. ✅ Ridge Plot
10. ✅ One-Line Usage Examples

💡 All plots are FULLY INTERACTIVE:
   • Hover for details
   • Zoom and pan
   • Rotate 3D plots
   • Click to filter
   • Export as PNG/HTML

📦 Installation:
   pip install --upgrade Essentiax

📚 Documentation:
   See ADVANCED_VIZ_GUIDE.md

⭐ GitHub:
   github.com/ShubhamWagh108/EssentiaX

🚀 Transform boring charts into stunning visualizations!
""")
