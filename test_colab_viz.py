"""
Test script to verify visualizations work in Colab
Copy this into Google Colab to test
"""

# ============================================================================
# STEP 1: Install
# ============================================================================
print("Step 1: Installing EssentiaX...")
# !pip install --upgrade Essentiax  # Uncomment in Colab

# ============================================================================
# STEP 2: Setup for Colab (CRITICAL!)
# ============================================================================
print("\nStep 2: Setting up for Colab...")
from essentiax.visuals import setup_colab
setup_colab()

# ============================================================================
# STEP 3: Load Sample Data
# ============================================================================
print("\nStep 3: Loading sample data...")
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# STEP 4: Test Basic Visualization
# ============================================================================
print("\nStep 4: Testing basic visualization...")
from essentiax.visuals import smart_viz

print("Creating basic visualizations...")
smart_viz(df, mode='auto', interactive=True)
print("✅ Basic visualization complete!")

# ============================================================================
# STEP 5: Test Advanced Visualization
# ============================================================================
print("\nStep 5: Testing advanced visualization...")
from essentiax.visuals import advanced_viz

print("Creating advanced visualizations...")
advanced_viz(df, viz_type='auto')
print("✅ Advanced visualization complete!")

# ============================================================================
# STEP 6: Test Specific 3D Plot
# ============================================================================
print("\nStep 6: Testing 3D scatter plot...")
from essentiax.visuals import Advanced3DViz

engine = Advanced3DViz()
engine.plot_3d_scatter_clusters(
    df,
    columns=['alcohol', 'flavanoids', 'color_intensity'],
    n_clusters=3
)
print("✅ 3D scatter plot complete!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print("""
If you can see the graphs above, everything is working!

✅ Basic visualizations
✅ Advanced visualizations  
✅ 3D interactive plots

If you only see text and no graphs:
1. Make sure you ran setup_colab() first
2. Check COLAB_TROUBLESHOOTING.md for solutions
3. Try restarting the runtime
""")
