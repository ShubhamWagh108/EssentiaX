# Testing Guide for EssentiaX v1.1.1

## Why You're Seeing the Import Error

The error `cannot import name 'setup_colab'` happens because you're testing the code locally before building and installing the package. Python can't find the new files until the package is properly installed.

---

## Solution 1: Test Without setup_colab() (Easiest)

The good news: **You don't actually need `setup_colab()`!** 

The visualizations now work automatically because we added smart environment detection. Just skip the setup call:

### Use This Instead:
```python
# ❌ Don't use this (causes import error in local testing)
from essentiax.visuals import setup_colab
setup_colab()

# ✅ Use this instead (works automatically!)
from essentiax.visuals import smart_viz
smart_viz(df)  # Automatically detects Colab and displays properly!
```

### Simple Colab Demo (No Setup Needed)
```python
# Cell 1: Install
!pip install --upgrade Essentiax

# Cell 2: Load data
from sklearn.datasets import load_wine
import pandas as pd
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Cell 3: Visualize (works automatically!)
from essentiax.visuals import smart_viz
smart_viz(df)  # Graphs will display!
```

---

## Solution 2: Build and Install the Package

If you want to test `setup_colab()`, you need to build and install the package first:

### Step 1: Build the Package
```bash
# In your project root directory
python setup.py sdist bdist_wheel
```

### Step 2: Install Locally
```bash
pip install -e .
# or
pip install dist/Essentiax-1.1.1-py3-none-any.whl
```

### Step 3: Test in Colab
```python
# Now this will work:
from essentiax.visuals import setup_colab
setup_colab()
```

---

## Solution 3: Upload to PyPI (For Production)

### Step 1: Build
```bash
python setup.py sdist bdist_wheel
```

### Step 2: Upload to PyPI
```bash
pip install twine
twine upload dist/*
```

### Step 3: Install from PyPI
```python
# In Colab
!pip install --upgrade Essentiax

# Now everything works:
from essentiax.visuals import setup_colab
setup_colab()
```

---

## Recommended Testing Approach

### For Now (Before Publishing)
Use `COLAB_SIMPLE_DEMO.py` which doesn't require `setup_colab()`:

```python
# Just install and use
!pip install --upgrade Essentiax

from essentiax.visuals import smart_viz
smart_viz(df)  # Works automatically!
```

### After Publishing to PyPI
Users can use the full demo with `setup_colab()`:

```python
!pip install --upgrade Essentiax

from essentiax.visuals import setup_colab
setup_colab()  # Now available!

from essentiax.visuals import smart_viz
smart_viz(df)
```

---

## Quick Test Script

Copy this into Colab to test (no setup_colab needed):

```python
# ============================================================================
# Quick Test - No setup_colab() needed!
# ============================================================================

# Install
!pip install --upgrade Essentiax

# Load data
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print(f"Data loaded: {df.shape}")

# Test basic visualization (works automatically!)
from essentiax.visuals import smart_viz
print("\nTesting smart_viz...")
smart_viz(df, mode='auto', interactive=True)

# Test advanced visualization (works automatically!)
from essentiax.visuals import advanced_viz
print("\nTesting advanced_viz...")
advanced_viz(df, viz_type='auto')

print("\n✅ If you see graphs above, everything works!")
```

---

## Why Automatic Detection is Better

The automatic detection we implemented means:
- ✅ No setup needed
- ✅ Works in Colab automatically
- ✅ Works in Jupyter automatically
- ✅ Works everywhere
- ✅ Simpler for users

The `setup_colab()` function is now **optional** - it's there for users who want explicit control, but it's not required.

---

## Summary

**Current Situation**: Testing locally before package is built  
**Error**: `cannot import name 'setup_colab'`  
**Solution**: Skip `setup_colab()` - visualizations work automatically!  
**Alternative**: Build and install package first

**Recommended**: Use `COLAB_SIMPLE_DEMO.py` for testing - no setup needed!

---

## Files to Use

### For Testing Now (Before Publishing)
- ✅ `COLAB_SIMPLE_DEMO.py` - Works without setup_colab()

### For Users (After Publishing)
- ✅ `COLAB_DEMO.py` - Full demo with setup_colab()
- ✅ `COLAB_ADVANCED_VIZ.py` - Advanced viz demo

Both will work, but the simple demo works immediately without building the package.
