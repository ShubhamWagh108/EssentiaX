# 🔧 Google Colab Troubleshooting Guide

## Issue: Visualizations Not Displaying (Only Text Output)

If you see only text output and no graphs in Google Colab, follow these solutions:

### ✅ Solution 1: Use setup_colab() (Recommended)

Add this at the start of your notebook (right after installation):

```python
# Cell 1: Installation
!pip install --upgrade Essentiax

# Cell 2: Setup for Colab
from essentiax.visuals import setup_colab
setup_colab()
```

This automatically configures Plotly to display properly in Colab.

---

### ✅ Solution 2: Manual Plotly Configuration

If Solution 1 doesn't work, manually configure Plotly:

```python
import plotly.io as pio
pio.renderers.default = 'colab'
```

Add this before running any visualization code.

---

### ✅ Solution 3: Use Colab Renderer Explicitly

When creating visualizations, you can also specify the renderer:

```python
from essentiax.visuals import advanced_viz

# The updated code now automatically detects Colab
advanced_viz(df, viz_type='auto')
```

The latest version (v1.1.0+) automatically detects Colab and uses the correct renderer.

---

### ✅ Solution 4: Check Plotly Installation

Ensure Plotly is properly installed:

```python
!pip install --upgrade plotly
import plotly
print(f"Plotly version: {plotly.__version__}")
```

You need Plotly >= 5.0 for best results.

---

### ✅ Solution 5: Restart Runtime

Sometimes Colab needs a fresh start:

1. Click **Runtime** → **Restart runtime**
2. Re-run all cells from the beginning
3. Make sure to run `setup_colab()` first

---

## Complete Working Example for Colab

```python
# ============================================================================
# CELL 1: Installation
# ============================================================================
!pip install --upgrade Essentiax

# ============================================================================
# CELL 2: Setup (IMPORTANT!)
# ============================================================================
from essentiax.visuals import setup_colab
setup_colab()

# ============================================================================
# CELL 3: Load Data
# ============================================================================
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(f"✅ Data loaded: {df.shape}")

# ============================================================================
# CELL 4: Visualize (Should show graphs now!)
# ============================================================================
from essentiax.visuals import advanced_viz

advanced_viz(df, viz_type='auto')
```

---

## Why This Happens

Google Colab uses a different rendering system than regular Jupyter notebooks. Plotly needs to be explicitly told to use the 'colab' renderer, otherwise it tries to open plots in a separate window (which doesn't exist in Colab).

---

## Verification

After running `setup_colab()`, you should see:

```
🎨 Setting up EssentiaX for Google Colab...
✅ EssentiaX visualization setup complete!
📊 All plots will now display properly in Colab
```

---

## Alternative: Use smart_viz with interactive=True

The basic visualization also works:

```python
from essentiax.visuals import smart_viz, setup_colab

setup_colab()  # Run this first!
smart_viz(df, mode='auto', interactive=True)
```

---

## Still Not Working?

If visualizations still don't display:

1. **Check Colab Version**: Ensure you're using the latest Colab
2. **Clear Output**: Click the trash icon to clear cell output, then re-run
3. **Check Browser**: Try a different browser (Chrome works best)
4. **Check Console**: Press F12 and check for JavaScript errors
5. **Use Static Plots**: As a fallback, use `interactive=False`:

```python
from essentiax.visuals import smart_viz
smart_viz(df, mode='auto', interactive=False)  # Uses matplotlib
```

---

## Quick Test

Run this to test if Plotly works:

```python
import plotly.express as px
import plotly.io as pio

# Set renderer
pio.renderers.default = 'colab'

# Create simple plot
fig = px.scatter(x=[1, 2, 3], y=[1, 2, 3])
fig.show()
```

If you see a scatter plot, Plotly is working correctly!

---

## Updated Demo Files

All demo files have been updated to include `setup_colab()`:

- `COLAB_DEMO.py` - Basic demo with setup
- `COLAB_ADVANCED_VIZ.py` - Advanced viz demo with setup

Just copy the cells into Colab and run them in order.

---

## Summary

**The key is to run `setup_colab()` before any visualizations!**

```python
# Always do this first in Colab:
from essentiax.visuals import setup_colab
setup_colab()
```

This ensures all EssentiaX visualizations display properly in Google Colab.

---

**Need more help?** Open an issue on GitHub with:
- Your Colab notebook link
- Error messages (if any)
- Screenshot of the output
