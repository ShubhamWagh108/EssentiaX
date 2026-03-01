"""
Quick test script for advanced visualizations
Run this to verify everything works!
"""

from sklearn.datasets import load_wine
import pandas as pd

# Load data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['target_name'] = df['target'].map({0: 'Class_0', 1: 'Class_1', 2: 'Class_2'})

print("🎨 Testing EssentiaX Advanced Visualizations")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

# Test 1: Auto mode
print("Test 1: Auto Mode")
print("-" * 50)
try:
    from essentiax.visuals import advanced_viz
    advanced_viz(df, viz_type='auto')
    print("✅ Auto mode works!\n")
except Exception as e:
    print(f"❌ Auto mode failed: {e}\n")

# Test 2: 3D scatter
print("Test 2: 3D Scatter with Clustering")
print("-" * 50)
try:
    from essentiax.visuals import Advanced3DViz
    engine = Advanced3DViz()
    engine.plot_3d_scatter_clusters(df, n_clusters=3)
    print("✅ 3D scatter works!\n")
except Exception as e:
    print(f"❌ 3D scatter failed: {e}\n")

# Test 3: Violin plots
print("Test 3: Advanced Violin Plots")
print("-" * 50)
try:
    engine.plot_violin_advanced(df, columns=['alcohol', 'malic_acid', 'ash'])
    print("✅ Violin plots work!\n")
except Exception as e:
    print(f"❌ Violin plots failed: {e}\n")

# Test 4: Parallel coordinates
print("Test 4: Parallel Coordinates")
print("-" * 50)
try:
    engine.plot_parallel_coordinates(df, color_col='target')
    print("✅ Parallel coordinates work!\n")
except Exception as e:
    print(f"❌ Parallel coordinates failed: {e}\n")

# Test 5: Correlation
print("Test 5: Advanced Correlation")
print("-" * 50)
try:
    engine.plot_correlation_chord(df, threshold=0.3)
    print("✅ Correlation visualization works!\n")
except Exception as e:
    print(f"❌ Correlation visualization failed: {e}\n")

print("=" * 50)
print("🎉 Testing Complete!")
print(f"Total plots created: {engine.plot_count}")
print("=" * 50)
