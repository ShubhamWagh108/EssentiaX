"""
Test Script: Verify Rich Progress + Plotly Coexistence in Colab
================================================================
This script tests that Plotly graphs render correctly after rich.progress
animations in both auto and manual modes.

Run this in Google Colab to verify the fix.
"""

import pandas as pd
import numpy as np
from essentiax.visuals.smartViz import smart_viz

# Create test dataset
np.random.seed(42)
test_data = pd.DataFrame({
    'age': np.random.randint(18, 80, 500),
    'income': np.random.normal(50000, 15000, 500),
    'score': np.random.uniform(0, 100, 500),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 500),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 500),
    'satisfaction': np.random.randint(1, 6, 500)
})

print("="*80)
print("TEST 1: AUTO MODE (Uses rich.progress - Previously Broken)")
print("="*80)
print("\n🔍 Expected: Rich text outputs + Plotly graphs both render\n")

smart_viz(
    df=test_data,
    mode="auto",
    interactive=True,
    max_plots=6
)

print("\n\n")
print("="*80)
print("TEST 2: MANUAL MODE (No rich.progress - Previously Worked)")
print("="*80)
print("\n🔍 Expected: Rich text outputs + Plotly graphs both render\n")

smart_viz(
    df=test_data,
    mode="manual",
    columns=['age', 'income', 'score', 'category'],
    interactive=True
)

print("\n\n")
print("="*80)
print("✅ TEST COMPLETE")
print("="*80)
print("\n📊 Verification Checklist:")
print("  ✓ Rich console output displays correctly")
print("  ✓ Plotly graphs render in AUTO mode")
print("  ✓ Plotly graphs render in MANUAL mode")
print("  ✓ No graphs are missing or hidden")
print("\n💡 If all graphs rendered, the fix is successful!")
