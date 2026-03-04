"""
Professional Analytics Visualizations Demo
==========================================
YouTube-Style Analytics Dashboard Examples
"""

import pandas as pd
import numpy as np
from essentiax.visuals.pro_analytics_viz import (
    create_3d_bubble_scatter,
    create_sunburst_hierarchy,
    create_correlation_heatmap_pro,
    create_temporal_trend_bubble,
    create_scatter_matrix_pro,
    create_distribution_histogram_pro,
    create_category_bar_pro,
    pro_analytics_dashboard
)

# Create sample YouTube-style analytics data
np.random.seed(42)
n_videos = 200

data = {
    'video_id': [f'VID_{i:04d}' for i in range(n_videos)],
    'title': [f'Video Title {i}' for i in range(n_videos)],
    'views': np.random.exponential(scale=10000, size=n_videos).astype(int),
    'likes': np.random.exponential(scale=500, size=n_videos).astype(int),
    'comments': np.random.exponential(scale=100, size=n_videos).astype(int),
    'shares': np.random.exponential(scale=50, size=n_videos).astype(int),
    'watch_time_hours': np.random.exponential(scale=1000, size=n_videos),
    'sentiment': np.random.uniform(-0.5, 0.5, n_videos),
    'title_length': np.random.randint(20, 100, n_videos),
    'category': np.random.choice(['Tech', 'Gaming', 'Education', 'Entertainment', 'Music'], n_videos),
    'subcategory': np.random.choice(['Tutorial', 'Review', 'Vlog', 'News', 'Comedy'], n_videos),
    'publish_date': pd.date_range(start='2023-01-01', periods=n_videos, freq='3D'),
    'duration_minutes': np.random.uniform(5, 60, n_videos),
    'thumbnail_quality': np.random.uniform(0.5, 1.0, n_videos),
}

# Calculate derived metrics
df = pd.DataFrame(data)
df['engagement_rate'] = (df['likes'] + df['comments'] * 2 + df['shares'] * 3) / df['views']
df['engagement_rate'] = df['engagement_rate'].clip(0, 0.5)  # Cap at 50%
df['ctr'] = np.random.uniform(0.02, 0.15, n_videos)  # Click-through rate
df['avg_view_duration'] = df['watch_time_hours'] * 60 / df['views']  # minutes
df['subjectivity'] = np.random.uniform(0, 1, n_videos)

print("="*80)
print("🎬 PROFESSIONAL ANALYTICS VISUALIZATIONS DEMO")
print("="*80)
print(f"\n📊 Dataset: {len(df)} videos with {df.shape[1]} metrics\n")

# ============================================================================
# 1. 3D BUBBLE SCATTER (Like YouTube Structure Analytics)
# ============================================================================
print("\n" + "="*80)
print("1️⃣  3D BUBBLE SCATTER - YouTube Structure Analytics")
print("="*80)

create_3d_bubble_scatter(
    df,
    x_col='sentiment',
    y_col='engagement_rate',
    z_col='title_length',
    size_col='views',
    color_col='views',
    title="YouTube Structure Analytics (X=Sentiment, Y=Engagement, Z=Title Length)",
    dark_theme=True
)

# ============================================================================
# 2. SUNBURST HIERARCHY (Sentiment > Category > Success)
# ============================================================================
print("\n" + "="*80)
print("2️⃣  SUNBURST HIERARCHY - Hierarchical View Distribution")
print("="*80)

# Create sentiment categories
df['sentiment_category'] = pd.cut(
    df['sentiment'],
    bins=[-1, -0.2, 0.2, 1],
    labels=['Negative', 'Neutral', 'Positive']
)

# Create view categories
df['view_category'] = pd.cut(
    df['views'],
    bins=[0, 5000, 15000, np.inf],
    labels=['Low Views', 'Medium Views', 'High Views']
)

create_sunburst_hierarchy(
    df,
    path_columns=['sentiment_category', 'category', 'view_category'],
    value_column='views',
    color_column='engagement_rate',
    title="Hierarchical View Distribution (Sentiment > Category > Success)",
    dark_theme=True
)

# ============================================================================
# 3. CORRELATION HEATMAP (What drives what?)
# ============================================================================
print("\n" + "="*80)
print("3️⃣  CORRELATION HEATMAP - Feature Correlation Matrix")
print("="*80)

correlation_cols = [
    'views', 'likes', 'comments', 'sentiment', 
    'title_length', 'engagement_rate', 'ctr', 'subjectivity'
]

create_correlation_heatmap_pro(
    df,
    columns=correlation_cols,
    title="Feature Correlation Matrix (What drives what?)",
    dark_theme=True,
    annotate=True
)

# ============================================================================
# 4. TEMPORAL TREND (Views over Time)
# ============================================================================
print("\n" + "="*80)
print("4️⃣  TEMPORAL TREND - Views Over Time")
print("="*80)

create_temporal_trend_bubble(
    df,
    time_col='publish_date',
    y_col='views',
    size_col='engagement_rate',
    color_col='sentiment',
    title="Temporal Trend Analysis (Views over Time)",
    dark_theme=True
)

# ============================================================================
# 5. SCATTER MATRIX (All-vs-All Comparison)
# ============================================================================
print("\n" + "="*80)
print("5️⃣  SCATTER MATRIX - Multi-Variable Comparison")
print("="*80)

scatter_cols = ['views', 'likes', 'sentiment', 'title_length', 'engagement_rate']

create_scatter_matrix_pro(
    df,
    columns=scatter_cols,
    color_col='engagement_rate',
    title="Scatter Matrix (All-vs-All Comparison)",
    dark_theme=True
)

# ============================================================================
# 6. DISTRIBUTION HISTOGRAM (Engagement Rate)
# ============================================================================
print("\n" + "="*80)
print("6️⃣  DISTRIBUTION HISTOGRAM - Engagement Rate Distribution")
print("="*80)

create_distribution_histogram_pro(
    df,
    column='engagement_rate',
    bins=50,
    title="Engagement Rate Distribution",
    dark_theme=True
)

# ============================================================================
# 7. CATEGORY BAR CHART (Top Categories)
# ============================================================================
print("\n" + "="*80)
print("7️⃣  CATEGORY BAR CHART - Category Distribution")
print("="*80)

create_category_bar_pro(
    df,
    category_col='category',
    value_col='views',
    top_n=10,
    title="Category Distribution by Views",
    dark_theme=True,
    horizontal=False
)

# ============================================================================
# 8. COMPLETE DASHBOARD (Auto Mode)
# ============================================================================
print("\n" + "="*80)
print("8️⃣  COMPLETE DASHBOARD - Auto Analytics")
print("="*80)

# Create a smaller dataset for dashboard demo
dashboard_df = df[['views', 'likes', 'engagement_rate', 'sentiment', 'category']].copy()

pro_analytics_dashboard(
    dashboard_df,
    config='auto',
    dark_theme=True
)

print("\n" + "="*80)
print("✅ DEMO COMPLETE!")
print("="*80)
print("\n💡 Tips:")
print("  • All visualizations are interactive (zoom, pan, hover)")
print("  • Dark theme is optimized for presentations")
print("  • Use light theme with dark_theme=False")
print("  • Customize colors and styling as needed")
print("\n🚀 Ready for production use!")
