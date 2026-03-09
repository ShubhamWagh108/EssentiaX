"""
Unified Smart Visualization Demo
=================================
ONE function for all visualizations - basic and advanced 3D
"""

import pandas as pd
import numpy as np
from essentiax.visuals import smart_viz

# Create sample data
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.normal(50000, 15000, n),
    'score': np.random.uniform(0, 100, n),
    'satisfaction': np.random.uniform(1, 5, n),
    'engagement_rate': np.random.uniform(0, 0.5, n),
    'views': np.random.exponential(scale=10000, n).astype(int),
    'category': np.random.choice(['Tech', 'Gaming', 'Education', 'Entertainment'], n),
    'type': np.random.choice(['Video', 'Article', 'Tutorial'], n),
    'sentiment': np.random.uniform(-0.5, 0.5, n)
})

print("="*80)
print("🎨 UNIFIED SMART VISUALIZATION DEMO")
print("="*80)
print("\nONE function - smart_viz() - for everything!\n")

# ============================================================================
# DEMO 1: AUTO MODE + ADVANCED 3D VISUALIZATIONS (DEFAULT)
# ============================================================================
print("\n" + "="*80)
print("DEMO 1: Auto Mode + Advanced 3D Visualizations")
print("="*80)
print("\nThis creates:")
print("  • 3D Bubble Scatter")
print("  • Sunburst Hierarchy")
print("  • Correlation Heatmap Pro")
print("  • Scatter Matrix Pro")
print("  • Distribution with Statistics")
print("\nAll with dark theme and professional styling!\n")

smart_viz(
    df,
    mode='auto',
    viz_type='advanced',  # Advanced 3D visualizations
    dark_theme=True       # Professional dark theme
)

# ============================================================================
# DEMO 2: MANUAL MODE + ADVANCED VISUALIZATIONS
# ============================================================================
print("\n\n" + "="*80)
print("DEMO 2: Manual Mode + Advanced Visualizations")
print("="*80)
print("\nYou specify which columns to visualize\n")

smart_viz(
    df,
    mode='manual',
    columns=['age', 'income', 'score', 'satisfaction', 'category', 'type'],
    viz_type='advanced',
    dark_theme=True
)

# ============================================================================
# DEMO 3: BASIC 2D VISUALIZATIONS (if you prefer simple charts)
# ============================================================================
print("\n\n" + "="*80)
print("DEMO 3: Basic 2D Visualizations")
print("="*80)
print("\nSimple 2D charts for quick analysis\n")

smart_viz(
    df,
    mode='auto',
    viz_type='basic',     # Basic 2D visualizations
    dark_theme=False      # Light theme
)

# ============================================================================
# DEMO 4: LIGHT THEME (for printing/reports)
# ============================================================================
print("\n\n" + "="*80)
print("DEMO 4: Light Theme for Reports")
print("="*80)
print("\nProfessional light theme for printing\n")

smart_viz(
    df,
    mode='auto',
    viz_type='advanced',
    dark_theme=False      # Light theme
)

print("\n" + "="*80)
print("✅ DEMO COMPLETE!")
print("="*80)
print("\n💡 Key Features:")
print("  • ONE function for everything")
print("  • Auto or manual mode")
print("  • Basic 2D or Advanced 3D")
print("  • Dark or light theme")
print("  • Rich console output with insights")
print("  • Statistical summaries")
print("  • Distribution insights")
print("\n🚀 Ready for production!")
