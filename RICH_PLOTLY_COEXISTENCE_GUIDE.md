# Rich + Plotly Coexistence Guide for Colab

## Quick Reference: Making Rich and Plotly Work Together

### The Problem

When using `rich.progress.Progress` before Plotly graphs in Google Colab, the graphs disappear due to output stream corruption.

### The Solution Pattern

```python
from rich.progress import Progress
from IPython.display import clear_output, display
import sys
import time

# 1. Use rich.progress as normal
with Progress(...) as progress:
    # Your progress animation
    pass

# 2. CRITICAL: Clean up immediately after
sys.stdout.flush()
sys.stderr.flush()

# 3. Reset IPython output context (Colab only)
try:
    clear_output(wait=False)
    time.sleep(0.05)
except:
    pass

# 4. Display Plotly with display() instead of fig.show()
display(fig)
time.sleep(0.05)
```

## Best Practices

### ✅ DO

1. **Always flush streams after rich.progress:**
   ```python
   with Progress(...) as progress:
       # ... work ...
   
   sys.stdout.flush()
   sys.stderr.flush()
   ```

2. **Use clear_output() in Colab:**
   ```python
   from IPython.display import clear_output
   clear_output(wait=False)  # Resets stream without clearing visible output
   ```

3. **Use display() instead of fig.show():**
   ```python
   from IPython.display import display
   display(fig)  # More reliable than fig.show()
   ```

4. **Add small timing delays:**
   ```python
   time.sleep(0.05)  # Allows stream to reset
   ```

5. **Detect environment:**
   ```python
   def is_colab():
       try:
           import google.colab
           return True
       except:
           return False
   ```

### ❌ DON'T

1. **Don't use fig.show() directly after rich.progress in Colab**
   ```python
   # BAD
   with Progress(...) as progress:
       pass
   fig.show()  # Will fail in Colab
   ```

2. **Don't skip stream cleanup**
   ```python
   # BAD
   with Progress(...) as progress:
       pass
   # Missing cleanup!
   display(fig)  # May still fail
   ```

3. **Don't use clear_output(wait=True)**
   ```python
   # BAD
   clear_output(wait=True)  # Clears visible output
   
   # GOOD
   clear_output(wait=False)  # Only resets stream
   ```

## Complete Example

```python
import pandas as pd
import plotly.express as px
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
import sys
import time

console = Console()

def plot_with_progress(df):
    """Example: Plotting with progress animation"""
    
    # 1. Show progress animation
    console.print("🔄 Processing data...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)
        time.sleep(1)  # Simulate work
    
    # 2. CRITICAL: Clean up streams
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        console.file.flush()
    except:
        pass
    
    # 3. Reset IPython context (Colab)
    try:
        from IPython.display import clear_output, display
        clear_output(wait=False)
        time.sleep(0.05)
    except:
        pass
    
    # 4. Create and display plot
    fig = px.scatter(df, x='x', y='y', title='My Plot')
    
    try:
        from IPython.display import display
        display(fig)
        time.sleep(0.05)
    except:
        fig.show()
    
    console.print("✅ Plot complete!")

# Test it
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
plot_with_progress(df)
```

## Environment Detection

```python
def detect_environment():
    """Detect runtime environment"""
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return 'jupyter'
        elif shell == 'TerminalInteractiveShell':
            return 'ipython'
    except NameError:
        pass
    
    return 'terminal'

# Use it
env = detect_environment()
if env == 'colab':
    # Apply Colab-specific fixes
    pass
```

## Reusable Helper Function

```python
def display_plotly_safe(fig, environment='colab'):
    """
    Safely display Plotly figure after rich.progress usage.
    
    Args:
        fig: Plotly figure object
        environment: 'colab', 'jupyter', or 'terminal'
    """
    import sys
    import time
    
    # Flush all streams
    sys.stdout.flush()
    sys.stderr.flush()
    
    if environment == 'colab':
        try:
            from IPython.display import clear_output, display, HTML
            
            # Reset output context
            clear_output(wait=False)
            time.sleep(0.05)
            
            # Display figure
            display(fig)
            time.sleep(0.05)
            
        except Exception:
            # Fallback chain
            try:
                fig.show(renderer='colab')
            except:
                try:
                    html = fig.to_html(include_plotlyjs='cdn')
                    display(HTML(html))
                except:
                    fig.show()
    
    elif environment == 'jupyter':
        try:
            from IPython.display import display
            display(fig)
        except:
            fig.show(renderer='notebook')
    
    else:
        fig.show()

# Usage
with Progress(...) as progress:
    # ... work ...

display_plotly_safe(fig, environment='colab')
```

## Common Pitfalls

### Pitfall 1: Forgetting to flush console.file

```python
# INCOMPLETE
sys.stdout.flush()
sys.stderr.flush()
# Missing console.file.flush()!

# COMPLETE
sys.stdout.flush()
sys.stderr.flush()
try:
    console.file.flush()  # Don't forget this!
except:
    pass
```

### Pitfall 2: Using clear_output() too early

```python
# BAD - Clears the progress output
with Progress(...) as progress:
    clear_output(wait=False)  # Too early!
    # ... work ...

# GOOD - Clears after progress completes
with Progress(...) as progress:
    # ... work ...
clear_output(wait=False)  # After the block
```

### Pitfall 3: Not handling exceptions

```python
# BAD - Crashes if IPython not available
from IPython.display import clear_output
clear_output(wait=False)

# GOOD - Graceful fallback
try:
    from IPython.display import clear_output
    clear_output(wait=False)
except:
    pass  # Not in IPython environment
```

## Testing Checklist

When implementing this pattern, verify:

- [ ] Rich progress animation displays correctly
- [ ] Progress animation completes and disappears
- [ ] Plotly graph renders after progress
- [ ] Graph is interactive (zoom, pan, hover)
- [ ] No error messages in console
- [ ] Works in Colab
- [ ] Works in Jupyter
- [ ] Works in terminal (with appropriate fallbacks)

## Performance Notes

- Stream flush operations: ~0.001s each
- `clear_output()`: ~0.01s
- Timing delays: 0.05s each (configurable)
- Total overhead: ~0.1s per graph (negligible)

## Debugging

If graphs still don't render:

```python
# 1. Check environment
print(f"Environment: {detect_environment()}")

# 2. Check Plotly version
import plotly
print(f"Plotly version: {plotly.__version__}")

# 3. Check IPython version
import IPython
print(f"IPython version: {IPython.__version__}")

# 4. Test basic display
from IPython.display import display, HTML
display(HTML("<h1>Test</h1>"))  # Should show "Test"

# 5. Test Plotly without rich
import plotly.express as px
fig = px.scatter(x=[1,2,3], y=[4,5,6])
display(fig)  # Should show graph
```

## Additional Resources

- [EssentiaX Documentation](https://github.com/yourusername/essentiax)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [IPython Display System](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)

## Summary

The key to making Rich and Plotly coexist in Colab:

1. **Flush streams** after rich.progress
2. **Reset IPython context** with clear_output(wait=False)
3. **Use display()** instead of fig.show()
4. **Add small delays** for timing
5. **Handle exceptions** gracefully

Follow this pattern and your visualizations will render perfectly! 🎨
