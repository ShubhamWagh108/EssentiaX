# EssentiaX Version History

## v1.1.3 (Current) - March 3, 2026
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

**Current Version**: 1.1.3  
**Latest Stable**: 1.1.3  
**Recommended**: 1.1.3 ⭐ (Fixes critical Colab rendering issue)
