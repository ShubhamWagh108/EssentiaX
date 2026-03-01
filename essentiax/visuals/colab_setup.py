"""
colab_setup.py — Google Colab Setup for EssentiaX Visualizations
================================================================
Ensures visualizations display properly in Google Colab
"""

def setup_colab():
    """
    Setup Google Colab environment for EssentiaX visualizations
    Call this at the start of your Colab notebook
    """
    try:
        # Check if we're in Colab
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        print("🎨 Setting up EssentiaX for Google Colab...")
        
        # Configure Plotly for Colab
        import plotly.io as pio
        pio.renderers.default = 'colab'
        
        # Enable inline plotting for matplotlib
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                ipython.run_line_magic('matplotlib', 'inline')
        except:
            pass
        
        print("✅ EssentiaX visualization setup complete!")
        print("📊 All plots will now display properly in Colab")
    else:
        print("ℹ️ Not in Google Colab - no setup needed")


def enable_plotly_colab():
    """
    Enable Plotly rendering in Google Colab
    Alternative to setup_colab() if you only need Plotly
    """
    try:
        import plotly.io as pio
        pio.renderers.default = 'colab'
        print("✅ Plotly renderer set to 'colab'")
    except Exception as e:
        print(f"⚠️ Could not set Plotly renderer: {e}")


# Auto-setup when imported in Colab
try:
    import google.colab
    # Automatically configure when imported in Colab
    import plotly.io as pio
    pio.renderers.default = 'colab'
except ImportError:
    pass
