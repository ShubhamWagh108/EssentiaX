"""
✅ READY TO TEST IN COLAB - Copy this entire file into Colab!
================================================================
This version works WITHOUT needing setup_colab() import
"""

# ============================================================================
# CELL 1: Installation
# ============================================================================
!pip install --upgrade Essentiax
print("✅ EssentiaX installed!")

# ============================================================================
# CELL 2: Verify Version
# ============================================================================
import essentiax
print(f"Version: {essentiax.__version__}")
print("Expected: 1.1.1 or higher")

# ============================================================================
# CELL 3: Load Sample Data
# ============================================================================
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['target_name'] = df['target'].map({0: 'Class_0', 1: 'Class_1', 2: 'Class_2'})

print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
df.head()

# ============================================================================
# CELL 4: Test Basic Visualization
# ============================================================================
print("🎨 Testing basic visualization...")
from essentiax.visuals import smart_viz

# This should show graphs automatically in Colab!
smart_viz(df, mode='auto', interactive=True)

print("✅ If you see graphs above, basic visualization works!")

# ============================================================================
# CELL 5: Test Advanced 3D Visualization
# ============================================================================
print("\n🎨 Testing advanced 3D visualization...")
from essentiax.visuals import advanced_viz

# This should show 3D interactive plots!
advanced_viz(df, viz_type='auto')

print("✅ If you see 3D plots above, advanced visualization works!")

# ============================================================================
# CELL 6: Test Specific 3D Scatter
# ============================================================================
print("\n🎨 Testing 3D scatter with clustering...")
from essentiax.visuals import Advanced3DViz

engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(
    df,
    columns=['alcohol', 'flavanoids', 'color_intensity'],
    n_clusters=3,
    title='🎨 Wine Dataset - 3D Cluster Analysis'
)

print("✅ If you see a 3D scatter plot above, everything works!")

# ============================================================================
# CELL 7: Test 3D Surface
# ============================================================================
print("\n🌊 Testing 3D surface plot...")
engine.plot_3d_surface(
    df,
    x_col='alcohol',
    y_col='flavanoids',
    title='🌊 Density Surface: Alcohol vs Flavanoids'
)

print("✅ If you see a 3D surface plot above, perfect!")

# ============================================================================
# CELL 8: Test Violin Plots
# ============================================================================
print("\n🎻 Testing advanced violin plots...")
engine.plot_violin_advanced(
    df,
    columns=['alcohol', 'malic_acid', 'ash', 'magnesium'],
    title='🎻 Distribution Comparison'
)

print("✅ If you see violin plots above, excellent!")

# ============================================================================
# CELL 9: Final Summary
# ============================================================================
print("\n" + "="*60)
print("🎉 TESTING COMPLETE!")
print("="*60)
print("""
If you saw all the graphs above, then:

✅ Basic visualizations work
✅ Advanced visualizations work
✅ 3D scatter plots work
✅ 3D surface plots work
✅ Violin plots work
✅ All interactive features work

The automatic Colab detection is working perfectly!

No setup_colab() needed - it works automatically! 🎨✨

If you DIDN'T see graphs:
1. Check that you're using v1.1.1 or higher
2. Try: Runtime → Restart runtime
3. Re-run all cells
4. Check COLAB_TROUBLESHOOTING.md
""")

print(f"\nYour version: {essentiax.__version__}")
print("Required: 1.1.1+")
