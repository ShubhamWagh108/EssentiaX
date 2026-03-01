"""
EssentiaX - Simple Colab Demo (No setup_colab needed!)
=======================================================
The visualizations now work automatically in Colab!
"""

# ============================================================================
# CELL 1: Installation
# ============================================================================
"""
# 📦 Install EssentiaX
"""
!pip install --upgrade Essentiax
print("✅ EssentiaX installed!")

# ============================================================================
# CELL 2: Load Data
# ============================================================================
"""
# 📊 Load Sample Data
"""
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
df.head()

# ============================================================================
# CELL 3: Basic Visualization (Works automatically!)
# ============================================================================
"""
# 🎨 Smart Visualization
"""
from essentiax.visuals import smart_viz

# No setup needed - works automatically in Colab!
smart_viz(df, mode='auto', interactive=True)

# ============================================================================
# CELL 4: Advanced 3D Visualization
# ============================================================================
"""
# 🎨 Advanced 3D Visualization
"""
from essentiax.visuals import advanced_viz

# Automatically detects Colab and displays properly!
advanced_viz(df, viz_type='auto')

# ============================================================================
# CELL 5: Specific 3D Plot
# ============================================================================
"""
# 🎨 3D Scatter with Clustering
"""
from essentiax.visuals import Advanced3DViz

engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(
    df,
    columns=['alcohol', 'flavanoids', 'color_intensity'],
    n_clusters=3
)

# ============================================================================
# CELL 6: Summary
# ============================================================================
"""
# 🎉 Complete!
"""
print("""
✅ All visualizations should display properly!

The latest version (v1.1.1) automatically detects Google Colab
and uses the correct renderer - no setup needed!

If graphs still don't show:
1. Make sure you're using v1.1.1 or higher
2. Try restarting the runtime
3. Clear output and re-run cells
""")
