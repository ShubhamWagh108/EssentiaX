"""
EssentiaX Advanced Visualization Demo
======================================
Showcase stunning 3D and interactive visualizations
"""

# ============================================================================
# SETUP
# ============================================================================
from sklearn.datasets import load_wine, load_iris
import pandas as pd
from essentiax.visuals import advanced_viz, Advanced3DViz

# Load sample data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['target_name'] = df['target'].map({0: 'Class_0', 1: 'Class_1', 2: 'Class_2'})

print("🎨 EssentiaX Advanced Visualization Demo")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ============================================================================
# DEMO 1: AUTO MODE - Let AI choose the best visualizations
# ============================================================================
print("\n" + "="*80)
print("DEMO 1: AUTO MODE - AI-Powered Visualization Selection")
print("="*80)

advanced_viz(df, viz_type='auto')

# ============================================================================
# DEMO 2: 3D SCATTER WITH CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("DEMO 2: 3D Scatter Plot with Automatic Clustering")
print("="*80)

engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(
    df, 
    columns=['alcohol', 'flavanoids', 'color_intensity'],
    n_clusters=3,
    title='🎨 Wine Dataset - 3D Cluster Analysis'
)

# ============================================================================
# DEMO 3: 3D SURFACE PLOT
# ============================================================================
print("\n" + "="*80)
print("DEMO 3: 3D Surface Plot (Density Visualization)")
print("="*80)

engine.plot_3d_surface(
    df,
    x_col='alcohol',
    y_col='flavanoids',
    title='🌊 Density Surface: Alcohol vs Flavanoids'
)

# ============================================================================
# DEMO 4: ADVANCED VIOLIN PLOTS
# ============================================================================
print("\n" + "="*80)
print("DEMO 4: Advanced Violin Plots with Statistical Overlays")
print("="*80)

engine.plot_violin_advanced(
    df,
    columns=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium'],
    title='🎻 Distribution Comparison - Top 5 Features'
)

# ============================================================================
# DEMO 5: PARALLEL COORDINATES
# ============================================================================
print("\n" + "="*80)
print("DEMO 5: Parallel Coordinates (Multi-Dimensional Analysis)")
print("="*80)

engine.plot_parallel_coordinates(
    df,
    color_col='target',
    columns=['alcohol', 'flavanoids', 'color_intensity', 'hue', 'proline'],
    title='📊 Parallel Coordinates - Colored by Wine Class'
)

# ============================================================================
# DEMO 6: ADVANCED CORRELATION VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("DEMO 6: Advanced Correlation Matrix")
print("="*80)

engine.plot_correlation_chord(
    df,
    columns=['alcohol', 'malic_acid', 'ash', 'flavanoids', 'color_intensity', 'hue'],
    threshold=0.3,
    title='🎭 Correlation Network (|r| ≥ 0.3)'
)

# ============================================================================
# DEMO 7: SUNBURST CHART (with categorical data)
# ============================================================================
print("\n" + "="*80)
print("DEMO 7: Sunburst Chart (Hierarchical Visualization)")
print("="*80)

# Create categorical bins for demo
df['alcohol_level'] = pd.cut(df['alcohol'], bins=3, labels=['Low', 'Medium', 'High'])
df['flavanoid_level'] = pd.cut(df['flavanoids'], bins=3, labels=['Low', 'Medium', 'High'])

engine.plot_sunburst(
    df,
    path_columns=['target_name', 'alcohol_level', 'flavanoid_level'],
    title='☀️ Wine Classification Hierarchy'
)

# ============================================================================
# DEMO 8: TREEMAP
# ============================================================================
print("\n" + "="*80)
print("DEMO 8: Interactive Treemap")
print("="*80)

engine.plot_treemap(
    df,
    path_columns=['target_name', 'alcohol_level'],
    title='🗺️ Wine Distribution Treemap'
)

# ============================================================================
# DEMO 9: RIDGE PLOT
# ============================================================================
print("\n" + "="*80)
print("DEMO 9: Ridge Plot (Distribution Comparison)")
print("="*80)

engine.plot_ridge(
    df,
    column='alcohol',
    group_by='target_name',
    title='🏔️ Alcohol Distribution by Wine Class'
)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✨ DEMO COMPLETE!")
print("="*80)
print("""
🎨 Advanced Visualizations Demonstrated:
1. ✅ Auto Mode (AI-powered selection)
2. ✅ 3D Scatter with Clustering
3. ✅ 3D Surface Plot
4. ✅ Advanced Violin Plots
5. ✅ Parallel Coordinates
6. ✅ Advanced Correlation Matrix
7. ✅ Sunburst Chart
8. ✅ Interactive Treemap
9. ✅ Ridge Plot

💡 All plots are fully interactive:
   • Hover for details
   • Zoom and pan
   • Rotate 3D plots
   • Click to filter

📦 Usage:
   from essentiax.visuals import advanced_viz
   
   # Auto mode
   advanced_viz(df, viz_type='auto')
   
   # Specific visualization
   advanced_viz(df, viz_type='3d_scatter', columns=['col1', 'col2', 'col3'])

🚀 Ready for production use!
""")
