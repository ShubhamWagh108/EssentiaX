"""
🎨 EssentiaX Colab Plotly Rendering Test
========================================
Copy this entire file into a Google Colab cell to test the fix!

This demonstrates that Rich console output and Plotly graphs
now render correctly together in Colab.
"""

# Install EssentiaX (uncomment if needed)
# !pip install essentiax

import pandas as pd
import numpy as np
from essentiax.visuals.smartViz import smart_viz
from essentiax.visuals.advanced_viz import advanced_viz

print("="*80)
print("🎨 EssentiaX Plotly Rendering Test for Google Colab")
print("="*80)
print("\n✅ If you can see this text AND the plots below, the fix works!\n")

# Create sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'sales': np.random.exponential(100, 300),
    'profit': np.random.normal(50, 20, 300),
    'customers': np.random.poisson(30, 300),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 300),
    'product': np.random.choice(['A', 'B', 'C'], 300),
    'rating': np.random.uniform(1, 5, 300)
})

print("\n" + "="*80)
print("TEST 1: SmartViz with Auto Mode")
print("="*80)
print("Expected: Rich console output + 2-3 interactive Plotly graphs\n")

smart_viz(df, mode='auto', max_plots=3)

print("\n" + "="*80)
print("TEST 2: Advanced 3D Scatter Plot")
print("="*80)
print("Expected: Rich console output + 1 interactive 3D Plotly graph\n")

advanced_viz(df, viz_type='3d_scatter', columns=['sales', 'profit', 'customers'], n_clusters=3)

print("\n" + "="*80)
print("TEST 3: Advanced Violin Plots")
print("="*80)
print("Expected: Rich console output + 1 interactive Plotly violin plot\n")

advanced_viz(df, viz_type='violin', columns=['sales', 'profit', 'rating'])

print("\n" + "="*80)
print("✅ TEST COMPLETE!")
print("="*80)
print("\n📊 What you should see:")
print("   ✅ Colored Rich console output (cyan, green, yellow text)")
print("   ✅ Beautiful formatted panels and tables")
print("   ✅ Interactive Plotly graphs (you can hover, zoom, pan)")
print("   ✅ All plots are VISIBLE (not invisible/missing)")
print("   ✅ Proper ordering: Console output → Plot → Console output")
print("\n🎉 If all of the above is true, the Colab rendering fix is working!")
print("\n💡 The fix includes:")
print("   • Automatic Colab environment detection")
print("   • Buffer flushing to prevent stream conflicts")
print("   • IPython.display.display() for reliable rendering")
print("   • Fallback mechanisms for edge cases")
print("\n🚀 Your EssentiaX library is now production-ready for Colab!")
