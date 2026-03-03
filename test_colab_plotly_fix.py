"""
Test script to verify Plotly rendering fix in Colab/Jupyter environments
"""

import pandas as pd
import numpy as np
from essentiax.visuals.smartViz import smart_viz
from essentiax.visuals.advanced_viz import advanced_viz

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'numeric_1': np.random.randn(500),
    'numeric_2': np.random.randn(500) * 2 + 5,
    'numeric_3': np.random.exponential(2, 500),
    'category_1': np.random.choice(['A', 'B', 'C', 'D'], 500),
    'category_2': np.random.choice(['X', 'Y', 'Z'], 500),
    'target': np.random.choice([0, 1], 500)
})

print("="*80)
print("Testing Plotly Rendering Fix for Colab/Jupyter")
print("="*80)

print("\n1. Testing smartViz with auto mode...")
try:
    smart_viz(df, mode='auto', max_plots=3)
    print("✅ smartViz test passed!")
except Exception as e:
    print(f"❌ smartViz test failed: {e}")

print("\n2. Testing advanced_viz with 3D scatter...")
try:
    advanced_viz(df, viz_type='3d_scatter', columns=['numeric_1', 'numeric_2', 'numeric_3'])
    print("✅ advanced_viz 3D scatter test passed!")
except Exception as e:
    print(f"❌ advanced_viz test failed: {e}")

print("\n3. Testing advanced_viz with violin plots...")
try:
    advanced_viz(df, viz_type='violin', columns=['numeric_1', 'numeric_2', 'numeric_3'])
    print("✅ advanced_viz violin test passed!")
except Exception as e:
    print(f"❌ advanced_viz violin test failed: {e}")

print("\n" + "="*80)
print("All tests completed!")
print("="*80)
print("\n💡 If running in Colab:")
print("   - Rich console output should display correctly")
print("   - Plotly graphs should render below the console output")
print("   - No invisible/missing plots")
print("\n💡 The fix includes:")
print("   - Environment detection (Colab/Jupyter/Terminal)")
print("   - Stream buffer flushing before plot rendering")
print("   - IPython.display.display() for reliable rendering")
print("   - Fallback mechanisms for edge cases")
