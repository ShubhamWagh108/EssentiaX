# EssentiaX Version History

## v1.1.6 (Current) - March 4, 2026
**Major Refactor: Unified Visualization Function**

### Changed (Breaking)
- 🔄 **UNIFIED FUNCTION**: Merged `smart_viz()` and `pro_analytics_dashboard()` into ONE function
- 🔄 Removed separate `pro_analytics_viz` module functions
- 🔄 Simplified API - ONE function for everything

### Added
- ✨ **viz_type parameter**: Choose 'basic' (2D) or 'advanced' (3D)
- ✨ **Advanced 3D Mode**: Creates 4+ professional visualizations:
  - 3D Bubble Scatter (structure analytics)
  - Sunburst Hierarchy (category distribution)
  - Correlation Heatmap Pro (feature relationships)
  - Scatter Matrix Pro (multi-variable analysis)
  - Distribution Pro (with statistics)
- ✨ **Rich Console Output**:
  - Visualization Setup panel
  - Statistical Summary tables (Count, Mean, Median, Std, Min, Max, Skewness, Kurtosis)
  - Distribution Insights (Symmetric/Skewed, Outliers, Central Tendency)
  - Visualization Summary
- ✨ **Dark Theme**: Professional dark theme for presentations
- ✨ **Light Theme**: Traditional light theme for reports
- 📚 **UNIFIED_VIZ_DEMO.py** - Demo showing unified function
- 📚 **UNIFIED_VIZ_GUIDE.md** - Comprehensive guide

### Migration Guide
```python
# OLD (v1.1.5)
from essentiax.visuals.pro_analytics_viz import pro_analytics_dashboard
pro_analytics_dashboard(df, config='auto', dark_theme=True)

# NEW (v1.1.6)
from essentiax.visuals import smart_viz
smart_viz(df, mode='auto', viz_type='advanced', dark_theme=True)
```

### API
```python
smart_viz(
    df,
    mode='auto',           # 'auto' or 'manual'
    columns=None,          # List for manual mode
    target=None,           # Target variable
    viz_type='advanced',   # 'advanced' (3D) or 'basic' (2D)
    dark_theme=True        # True (dark) or False (light)
)
```

### Benefits
- ✅ **Simplicity**: ONE function to remember
- ✅ **Flexibility**: Auto/manual, basic/advanced, dark/light
- ✅ **Professional**: Production-ready 3D visualizations
- ✅ **Comprehensive**: Statistics + insights + charts
- ✅ **Beautiful**: Dark theme + rich console output

### Performance
- ⚡ Same performance as v1.1.5
- ⚡ Optimized for datasets up to 10,000 rows
- ⚡ Automatic sampling for larger datasets

### Compatibility
- ✅ Google Colab
- ✅ Jupyter Notebook
- ✅ JupyterLab
- ✅ IPython Terminal
- ✅ Python Scripts

---

## v1.1.5 - March 4, 2026
**Major Feature: Professional Analytics Visualizations (YouTube-Style)**

### Added
- 🎨 **NEW MODULE**: `pro_analytics_viz.py` - Professional analytics visualizations
- ✨ **3D Bubble Scatter** - YouTube-style structure analytics with bubble sizes and color gradients
- ✨ **Sunburst Hierarchy Charts** - Interactive nested circular charts for hierarchical data
- ✨ **Professional Correlation Heatmap** - Annotated heatmaps with Red-Blue diverging colors
- ✨ **Temporal Trend Bubble** - Time series visualization with bubble scatter and color coding
- ✨ **Scatter Matrix Pro (SPLOM)** - Multi-variable all-vs-all comparison with color coding
- ✨ **Distribution Histogram Pro** - Professional histograms with mean/median lines and statistics
- ✨ **Category Bar Chart Pro** - Gradient color bars with value annotations
- ✨ **Complete Analytics Dashboard** - Auto-generates comprehensive analytics in one line
- 🎨 **Dark Theme** - Professional dark theme optimized for presentations
- 📚 **PRO_ANALYTICS_DEMO.py** - Complete demo with sample YouTube-style data
- 📚 **PRO_ANALYTICS_GUIDE.md** - Comprehensive guide with examples and best practices

### Features
- 🎯 **8 Professional Visualization Types** - Production-ready charts
- 🌙 **Dark Theme Optimized** - Beautiful dark backgrounds like YouTube Analytics
- 🎨 **Color Gradients** - Plasma, Viridis, and custom color schemes
- 📊 **Interactive** - Zoom, pan, hover, drill-down capabilities
- 💎 **Production-Ready** - Professional styling and annotations
- 🔄 **Auto Dashboard** - Automatically creates relevant visualizations
- 🎭 **Customizable** - Full control over colors, themes, and styling

### Technical Details
- All visualizations use Plotly for interactivity
- Compatible with Colab, Jupyter, and terminal environments
- Stream cleanup for reliable rendering
- Professional color schemes and gradients
- Optimized for large datasets

### Use Cases
- YouTube/Social Media Analytics
- E-commerce Performance Dashboards
- Business Intelligence Reports
- Scientific Data Visualization
- Marketing Analytics
- Product Analytics

### API
```python
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
```

### Performance
- ⚡ Optimized for datasets up to 10,000 rows
- ⚡ Sampling recommended for larger datasets
- ⚡ Interactive rendering with minimal lag

### Compatibility
- ✅ Google Colab
- ✅ Jupyter Notebook
- ✅ JupyterLab
- ✅ IPython Terminal
- ✅ Python Scripts

---

## v1.1.4 - March 4, 2026
**Critical Fix: Rich Progress + Plotly Coexistence in Colab**

### Fixed
- 🐛 **MAJOR**: Plotly graphs now render correctly after Rich progress animations in auto mode
- 🐛 Stream corruption caused by `rich.progress.Progress` in Google Colab
- 🐛 IOPub message bus getting stuck after progress animations
- 🐛 Graphs disappearing silently in `smart_viz(mode="auto")`

### Added
- ✨ Stream cleanup after `rich.progress.Progress` context closes
- ✨ IPython output context reset with `clear_output(wait=False)`
- ✨ Enhanced `_display_plotly_figure()` with 4-level fallback chain
- ✨ Direct widget injection using `display(fig)` instead of `fig.show()`
- ✨ Timing delays for proper stream sequencing
- 📚 COLAB_RICH_PLOTLY_FIX.md - Technical deep dive
- 📚 RICH_PLOTLY_COEXISTENCE_GUIDE.md - Developer best practices
- 📚 FIX_IMPLEMENTATION_SUMMARY.md - Executive summary
- 📚 QUICK_FIX_GUIDE.md - User quick-start guide
- 📚 FIX_FLOW_DIAGRAM.md - Visual flow diagrams
- 📚 SOLUTION_SUMMARY.md - Quick overview
- 📚 DEPLOYMENT_CHECKLIST.md - Deployment guide
- 📚 FIX_DOCUMENTATION_INDEX.md - Documentation index
- 🧪 test_colab_rich_plotly_fix.py - Comprehensive test script

### Changed
- 🔧 Enhanced `_display_plotly_figure()` in smartViz.py
- 🔧 Enhanced `_auto_select_variables()` with stream cleanup
- 🔧 Improved reliability in Colab environment
- 📝 Updated all visualization documentation

### Performance
- ⚡ Minimal overhead: ~0.1s per graph (negligible)
- ⚡ No impact on manual mode
- ⚡ Optimized stream flush operations

### Impact
- ✅ Rich progress animations display correctly
- ✅ Plotly graphs render after progress animations
- ✅ Auto mode now works perfectly in Colab
- ✅ Manual mode unchanged (still works)
- ✅ No breaking changes - fully backward compatible
- ✅ Works across all environments (Colab, Jupyter, IPython, terminal)

### Technical Details
**Root Cause**: `rich.progress.Progress` modifies the IPython output stream state. When the context closes, the stream remains corrupted, causing Plotly's HTML/JS payload to be silently dropped.

**Solution**: 
1. Flush all output buffers after progress closes
2. Reset IPython context with `clear_output(wait=False)`
3. Use `display(fig)` for direct widget injection
4. Add timing delays for proper sequencing

---

## v1.1.3 - March 3, 2026
**Critical Fix: Plotly Rendering in Google Colab**

### Fixed
- 🐛 **MAJOR**: Plotly graphs now render correctly in Google Colab (were completely invisible)
- 🐛 Stream conflicts between Rich console output and Plotly HTML/JS
- 🐛 JavaScript injection timing issues
- 🐛 Environment-specific rendering problems

### Added
- ✨ Automatic environment detection (Colab/Jupyter/IPython/Terminal)
- ✨ IPython.display integration for reliable rendering
- ✨ Buffer flushing mechanism to prevent stream conflicts
- ✨ Multiple fallback mechanisms for maximum compatibility
- 📚 COLAB_PLOTLY_FIX.md - Technical deep dive
- 📚 PLOTLY_FIX_SUMMARY.md - Quick reference
- 📚 COLAB_USAGE_GUIDE.md - Complete usage guide
- 📚 PLOTLY_FIX_DIAGRAM.md - Visual explanation
- 🧪 test_colab_plotly_fix.py - Test script
- 🧪 COLAB_PLOTLY_TEST.py - Colab-ready test

### Changed
- 🔧 Replaced `fig.show()` with `_display_plotly_figure()` in all visualization modules
- 🔧 Enhanced error handling with helpful messages
- 🔧 Improved reliability across all environments
- 📝 Updated all visualization files (smartViz.py, advanced_viz.py, smart_eda.py)

### Performance
- ⚡ Minimal overhead: ~0.1s per plot
- ⚡ No impact on existing functionality
- ⚡ Optimized buffer flushing

### Impact
- ✅ Rich console output displays beautifully
- ✅ Plotly graphs render perfectly below console output
- ✅ All plots are visible and interactive
- ✅ Works automatically - zero configuration needed
- ✅ Backward compatible - no code changes required

---

## v1.1.2 - Earlier
**Attempted Fix: Colab Renderer Configuration**

### Changed
- 🔧 More aggressive Colab renderer configuration (partial fix)

---

## v1.1.1 - February 27, 2026
**Bug Fix: Colab Visualization Display**

### Fixed
- 🐛 Visualizations not displaying in Google Colab (only text output)
- 🐛 Plotly renderer compatibility issues

### Added
- ✨ Smart environment detection for automatic renderer selection
- ✨ `setup_colab()` helper function for explicit Colab setup
- ✨ `enable_plotly_colab()` for Plotly-specific configuration
- 📚 COLAB_TROUBLESHOOTING.md - Complete troubleshooting guide
- 📚 COLAB_FIX_SUMMARY.md - Technical implementation details
- 🧪 test_colab_viz.py - Test script

### Changed
- 🔧 All visualization files now use smart display function
- 📝 Updated all demo files with setup instructions
- 📝 Updated COLAB_DEMO.md with new instructions

---

## v1.1.0 - February 27, 2026
**Major Feature: Advanced 3D & Interactive Visualizations**

### Added
- 🎨 10 new advanced visualization types
- 🎨 3D Scatter plots with automatic K-means clustering
- 🌊 3D Surface plots with density estimation
- ☀️ Sunburst charts for hierarchical data
- 🌊 Sankey diagrams for flow visualization
- 🎻 Advanced violin plots with statistics
- 📊 Parallel coordinates for multi-dimensional data
- 🗺️ Treemaps for hierarchical categories
- 🎬 Animated scatter plots for time-series
- 🎭 Advanced correlation visualizations
- 🏔️ Ridge plots for distribution comparison
- 🤖 AI-powered auto mode for visualization selection
- 📚 ADVANCED_VIZ_GUIDE.md - Comprehensive documentation
- 📚 ADVANCED_VIZ_DEMO.py - Complete demo
- 📚 COLAB_ADVANCED_VIZ.py - Colab-ready demo

### Changed
- 📝 Updated README.md with advanced visualization section
- 🔧 Enhanced visualization engine with production-ready aesthetics

---

## v1.0.9 - February 26, 2026
**Bug Fix: SmartViz Import**

### Fixed
- 🐛 ImportError when importing SmartViz from essentiax.visuals
- 🐛 Demo files using incorrect import statements

### Changed
- 🔧 Updated COLAB_DEMO.py to use correct imports
- 🔧 Updated COLAB_DEMO.md with correct usage
- 🔧 Updated linkedin_demo_complete.py
- 🔧 Updated linkedin_demo_visual.py

### Added
- 📚 V1.0.9_RELEASE_NOTES.md

---

## v1.0.8 - Earlier
**Bug Fix: SmartEDA Verbose Parameter**

### Fixed
- 🐛 Removed verbose parameter from SmartEDA

---

## v1.0.6 - Earlier
**Feature: Auto-detect Target Column**

### Added
- ✨ Automatic target column detection in SmartEDA
- 🤖 AI-powered target identification

---

## Earlier Versions

### v1.0.0 - Initial Release
**Complete ML Automation Platform**

### Core Features
- 📊 Smart EDA with AI insights
- 🧹 Intelligent data cleaning
- 🎨 Smart visualizations
- 🔧 Feature engineering
- 🤖 AutoML capabilities
- 📈 Model explainability
- 🚀 Production deployment tools

---

## Version Numbering

EssentiaX follows semantic versioning:
- **Major (X.0.0)**: Breaking changes
- **Minor (1.X.0)**: New features, backward compatible
- **Patch (1.1.X)**: Bug fixes, backward compatible

---

## Upgrade Guide

### To v1.1.3 (Current) ⭐ RECOMMENDED
```bash
pip install --upgrade Essentiax
```

**What's Fixed**:
- ✅ Plotly graphs now render in Colab!
- ✅ No more invisible plots
- ✅ Perfect integration with Rich output

**No code changes required!** Just upgrade and enjoy:
```python
from essentiax.visuals.smartViz import smart_viz
smart_viz(df, mode='auto')  # Now works perfectly in Colab!
```

### To v1.1.1
```bash
pip install --upgrade Essentiax
```

No code changes required! All existing code works.

**Optional**: Add `setup_colab()` for Colab notebooks:
```python
from essentiax.visuals import setup_colab
setup_colab()
```

### To v1.1.0
```bash
pip install --upgrade Essentiax
```

New features available:
```python
from essentiax.visuals import advanced_viz
advanced_viz(df, viz_type='auto')
```

### To v1.0.9
```bash
pip install --upgrade Essentiax
```

Update imports:
```python
# Old (incorrect)
from essentiax.visuals import SmartViz

# New (correct)
from essentiax.visuals import smart_viz
```

---

## Compatibility

### Python Versions
- ✅ Python 3.7+
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11

### Environments
- ✅ Google Colab (v1.1.1+)
- ✅ Jupyter Notebook
- ✅ JupyterLab
- ✅ IPython
- ✅ Python scripts
- ✅ VS Code notebooks

### Operating Systems
- ✅ Windows
- ✅ macOS
- ✅ Linux

---

## Dependencies

### Core Dependencies
- pandas >= 1.0
- numpy >= 1.20
- matplotlib >= 3.0
- seaborn >= 0.11
- scikit-learn >= 1.0
- rich >= 10.0
- plotly >= 5.0
- scipy >= 1.7

### Optional Dependencies
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- catboost >= 1.0.0
- optuna >= 3.0.0
- shap >= 0.41.0

---

## Roadmap

### Upcoming Features
- 🔮 More 3D visualization types
- 🎨 Custom color schemes
- 🎬 Animation controls
- 📊 Dashboard builder
- 🔄 Real-time data support
- 🤖 Enhanced AI insights
- 📈 Advanced statistical tests

---

## Support

- **GitHub**: https://github.com/ShubhamWagh108/EssentiaX
- **Issues**: Report bugs or request features
- **Documentation**: See README.md and guides

---

**Current Version**: 1.1.6  
**Latest Stable**: 1.1.6  
**Recommended**: 1.1.6 ⭐ (Unified Function - ONE smart_viz() for Everything)
