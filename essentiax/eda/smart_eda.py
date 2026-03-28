"""
smart_eda.py - EssentiaX Unified EDA Engine
===========================================
🧠 Next-Generation Exploratory Data Analysis with AI Insights

PHASE 1 UNIFIED FEATURES:
• Smart EDA with enhanced insights
• Problem Card with ML recommendations  
• EDA Pro with HTML reports
• Interactive visualizations
• Multiple output modes
• AI-powered variable selection

Combines the power of smart_eda, problem_card, eda_pro, and smart_viz
into one unified, intelligent EDA engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import base64
from datetime import datetime
from scipy import stats
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

# Additional imports for enhanced analysis
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    SCIPY_ADVANCED_AVAILABLE = True
except ImportError:
    SCIPY_ADVANCED_AVAILABLE = False

# Import advanced EssentiaX modules
try:
    from ..ai.advanced_insights import AdvancedInsightsEngine
    from ..eda.advanced_stats import AdvancedStatistics
    from ..visuals.big_data_plots import BigDataPlotter
    from ..eda.smart_variable_detector import SmartVariableDetector, analyze_variables
    ADVANCED_MODULES_AVAILABLE = True
except ImportError:
    ADVANCED_MODULES_AVAILABLE = False

warnings.filterwarnings("ignore")

# Initialize Rich Console for beautiful output
console = Console()

# Configure Plotly rendering with robust environment detection
try:
    import plotly.io as pio
    from IPython.display import display, HTML
    import sys
    
    # Detect environment
    def _detect_environment():
        """Detect if running in Colab, Jupyter, or terminal"""
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
    
    _ENVIRONMENT = _detect_environment()
    
    # Set default renderer based on environment
    if _ENVIRONMENT == 'colab':
        pio.renderers.default = 'colab'
    elif _ENVIRONMENT == 'jupyter':
        pio.renderers.default = 'notebook'
    else:
        pio.renderers.default = 'browser'
        
except ImportError:
    _ENVIRONMENT = 'terminal'
    pio = None


def _display_plotly_figure(fig):
    """
    Display Plotly figure with guaranteed rendering in Colab/Jupyter environments.
    
    This function handles the timing and stream issues that prevent Plotly graphs
    from rendering in Colab when mixed with rich console output.
    """
    try:
        # Flush any pending console output to prevent stream clashing
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Force console buffer flush if using rich
        try:
            console.file.flush()
        except:
            pass
        
        # Environment-specific rendering
        if _ENVIRONMENT == 'colab':
            # Colab-specific: Force display with explicit HTML injection
            try:
                from IPython.display import display, HTML
                
                # Method 1: Use display() with the figure directly (most reliable)
                display(fig)
                
                # Small delay to ensure rendering completes
                import time
                time.sleep(0.1)
                
            except Exception as e:
                # Fallback: Use fig.show() with explicit renderer
                try:
                    fig.show(renderer='colab')
                except:
                    # Last resort: Generate HTML and display
                    html_str = fig.to_html(include_plotlyjs='cdn', div_id=f'plotly-div-{id(fig)}')
                    display(HTML(html_str))
                    
        elif _ENVIRONMENT == 'jupyter':
            # Jupyter notebook: Use display for better reliability
            try:
                from IPython.display import display
                display(fig)
            except:
                fig.show(renderer='notebook')
                
        else:
            # Terminal or other: Use standard show
            fig.show()
            
    except Exception as e:
        # Ultimate fallback: standard show method
        try:
            fig.show()
        except Exception as show_error:
            console.print(f"[yellow]⚠️ Warning: Could not display plot. Error: {show_error}[/yellow]")
            console.print("[yellow]💡 Tip: Try running in Jupyter/Colab for interactive plots[/yellow]")

# Set modern styling for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


class EDAInsights:
    """AI-Powered EDA Insights Generator"""
    
    @staticmethod
    def _detect_multimodal(data):
        """Advanced multimodal distribution detection using kernel density estimation"""
        if not SCIPY_ADVANCED_AVAILABLE or len(data) < 50:
            return 1, [], [], []
        
        try:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            density = kde(x_range)
            peaks, properties = find_peaks(density, height=np.max(density)*0.1, distance=20)
            return len(peaks), peaks, density, x_range
        except Exception:
            return 1, [], [], []
    
    @staticmethod
    def _advanced_outlier_detection(data):
        """Enhanced outlier detection using multiple methods"""
        outlier_methods = {}
        
        # Method 1: IQR (Traditional)
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        if IQR > 0:
            iqr_outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            outlier_methods['IQR'] = len(iqr_outliers)
        else:
            outlier_methods['IQR'] = 0
        
        # Method 2: Z-Score (for normal distributions)
        try:
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]
            outlier_methods['Z-Score'] = len(z_outliers)
        except:
            outlier_methods['Z-Score'] = 0
        
        # Method 3: Modified Z-Score (more robust)
        try:
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad != 0:
                modified_z_scores = 0.6745 * (data - median) / mad
                modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
                outlier_methods['Modified Z-Score'] = len(modified_z_outliers)
            else:
                outlier_methods['Modified Z-Score'] = 0
        except:
            outlier_methods['Modified Z-Score'] = 0
        
        # Method 4: Isolation Forest (for complex patterns)
        if SKLEARN_AVAILABLE and len(data) > 50:
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
                iso_outliers = np.sum(outlier_labels == -1)
                outlier_methods['Isolation Forest'] = iso_outliers
            except:
                pass
        
        return outlier_methods
    
    @staticmethod
    def _infer_problem_type(df: pd.DataFrame, target: str):
        """Infer ML problem type based on target column."""
        if target is None or target not in df.columns:
            return None, "no_target"

        y = df[target]
        n_unique = y.nunique()

        # Pure numeric target
        if np.issubdtype(y.dtype, np.number):
            if n_unique <= 1:
                return None, "invalid"
            if n_unique <= 20:
                return "classification", "numeric_low_card"
            return "regression", "numeric_regression"

        # Non-numeric → could be classification or NLP
        avg_len = y.astype(str).str.len().mean()
        if n_unique <= 1:
            return None, "invalid"
        
        # Check if target itself looks like text (long strings with spaces/punctuation)
        if avg_len >= 30:
            sample_text = y.astype(str).iloc[0] if len(y) > 0 else ""
            has_spaces = ' ' in sample_text
            has_punctuation = any(char in sample_text for char in '.,!?;:')
            
            if has_spaces or has_punctuation or avg_len >= 50:
                return "nlp", "long_text_target"
        
        # Enhanced NLP detection - only if we have clear text features AND text-like target
        text_columns = []
        for col in df.columns:
            if col != target and df[col].dtype == 'object':
                col_avg_len = df[col].astype(str).str.len().mean()
                if col_avg_len >= 30:
                    sample_text = df[col].astype(str).iloc[0] if len(df) > 0 else ""
                    has_spaces = ' ' in sample_text
                    has_punctuation = any(char in sample_text for char in '.,!?;:')
                    if has_spaces or has_punctuation:
                        text_columns.append(col)
        
        # Only classify as NLP if target is short categorical AND we have clear text features
        if text_columns and n_unique <= 10 and avg_len < 20:
            # Additional check: does the target look like labels rather than text?
            sample_target = str(y.iloc[0]) if len(y) > 0 else ""
            if not any(char in sample_target for char in ' .,!?;:'):
                return "nlp", "text_classification"
        
        # Default to classification for categorical targets
        if n_unique <= 30 and avg_len < 30:
            return "classification", "categorical_labels"
        return "classification", "categorical_high_card"


def _fig_to_base64(fig):
    """Convert a Matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _model_recommendations(problem_type, df: pd.DataFrame, target: str, imbalance_flag: bool):
    """Return model recommendations based on problem type and dataset characteristics"""
    if problem_type is None:
        return {
            "type": "unknown",
            "baseline": [],
            "advanced": [],
            "notes": ["No target or invalid target — cannot recommend models."]
        }

    rows, cols = df.shape
    feature_cols = [c for c in df.columns if c != target]
    n_features = len(feature_cols)

    small_data = rows < 10_000
    wide_data = n_features > 100

    rec = {
        "type": problem_type,
        "baseline": [],
        "advanced": [],
        "notes": []
    }

    if problem_type == "classification":
        rec["baseline"].append("LogisticRegression (with scaling)")
        rec["baseline"].append("RandomForestClassifier")

        if wide_data:
            rec["advanced"].append("LinearSVC / SGDClassifier (handles many features)")
        else:
            rec["advanced"].append("GradientBoostingClassifier / XGBoost (if available)")

        if imbalance_flag:
            rec["notes"].append("Use class_weight='balanced' or resampling (SMOTE/undersampling).")

    elif problem_type == "regression":
        rec["baseline"].append("LinearRegression (check assumptions)")
        rec["baseline"].append("RandomForestRegressor")

        if wide_data:
            rec["advanced"].append("Lasso / ElasticNet (for feature selection)")
        else:
            rec["advanced"].append("GradientBoostingRegressor / XGBoostRegressor (if available)")

        if rows < 1000:
            rec["notes"].append("Small data — prefer simpler models, strong regularization, cross-validation.")

    elif problem_type == "nlp":
        rec["baseline"].append("TF-IDF + LogisticRegression / LinearSVC")
        rec["baseline"].append("TF-IDF + NaiveBayes (for text classification)")

        rec["advanced"].append("Pretrained transformers (e.g., BERT) via HuggingFace")
        rec["notes"].append("Do proper train/val/test split by time or document source to avoid leakage.")

    else:
        rec["notes"].append("Unknown problem type — treat as EDA-only for now.")

    if rows > 200_000:
        rec["notes"].append("Large dataset — consider subsampling for prototyping, then scale with efficient models.")

    return rec


def _check_imbalance(y: pd.Series):
    """Return imbalance flag + ratio of majority class if classification."""
    if y is None:
        return False, None

    if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
        return False, None  # regression, ignore

    counts = y.value_counts(normalize=True)
    if len(counts) < 2:
        return False, None

    top = counts.iloc[0]
    return bool(top > 0.8), float(top)


def _create_interactive_distribution(df, column):
    """Create interactive distribution plot with Plotly"""
    data = df[column].dropna()
    
    # Create subplot with histogram and box plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Distribution: {column}', f'Box Plot: {column}'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=data, name='Distribution', nbinsx=30, 
                    marker_color='rgba(255, 107, 107, 0.7)'),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(x=data, name='Box Plot', marker_color='rgba(78, 205, 196, 0.7)'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"📊 Distribution Analysis: {column}",
        template="plotly_white",
        height=600,
        showlegend=False
    )
    
    return fig


def _create_interactive_correlation(df, columns):
    """Create interactive correlation heatmap"""
    corr_data = df[columns].corr()
    
    fig = px.imshow(
        corr_data,
        title="🔥 Interactive Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect="auto",
        text_auto=True
    )
    
    fig.update_layout(
        title_font_size=16,
        width=800,
        height=600
    )
    
    return fig


def _create_interactive_categorical(df, column, target=None):
    """Create interactive categorical plot"""
    data = df[column].value_counts().head(15)
    
    if target and target in df.columns:
        fig = px.histogram(
            df, x=column, color=target,
            title=f"📊 {column} Distribution by {target}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        fig = px.bar(
            x=data.index, y=data.values,
            title=f"📊 Category Distribution: {column}",
            color=data.values,
            color_continuous_scale='viridis'
        )
    
    fig.update_layout(
        title_font_size=16,
        xaxis_tickangle=-45
    )
    
    return fig
def smart_eda(
    df: pd.DataFrame, 
    target: str = None,
    mode: str = "console",  # "console", "html", "interactive", "all"
    sample_size: int = 50000,
    report_path: str = "essentiax_eda_report.html",
    max_plots: int = 8,
    show_visualizations: bool = True,
    advanced_stats: bool = True,  # New parameter for advanced statistics
    ai_insights: bool = True,     # New parameter for AI insights
    auto_detect: bool = True      # New parameter for automatic variable detection
):
    """
    🧠 EssentiaX Unified Smart EDA Engine - Phase 1 Enhanced
    
    Combines smart_eda, problem_card, eda_pro, and smart_viz with advanced statistics and AI insights.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target : str, optional
        Target column for ML-focused analysis (auto-detected if None)
    mode : str
        Output mode: "console", "html", "interactive", "all"
    sample_size : int
        Sample size for large datasets
    report_path : str
        Path for HTML report (when mode includes "html")
    max_plots : int
        Maximum number of interactive plots to show
    show_visualizations : bool
        Whether to display interactive visualizations
    advanced_stats : bool
        Whether to include advanced statistical analysis
    ai_insights : bool
        Whether to generate AI-powered insights and recommendations
    auto_detect : bool
        Whether to automatically detect target and meaningful features
    
    Returns:
    --------
    dict : Analysis results and insights
    """
    
    # Initialize engines
    insights_engine = EDAInsights()
    advanced_stats_engine = None
    advanced_insights_engine = None
    big_data_plotter = None
    variable_detector = None
    
    if ADVANCED_MODULES_AVAILABLE and advanced_stats:
        advanced_stats_engine = AdvancedStatistics()
    if ADVANCED_MODULES_AVAILABLE and ai_insights:
        advanced_insights_engine = AdvancedInsightsEngine()
    if ADVANCED_MODULES_AVAILABLE and show_visualizations:
        big_data_plotter = BigDataPlotter(max_points=sample_size)
    if ADVANCED_MODULES_AVAILABLE and auto_detect:
        variable_detector = SmartVariableDetector()
    
    # Beautiful header for console mode
    if mode in ["console", "all"]:
        console.print("\n" + "="*80)
        console.print("🧠 [bold magenta]EssentiaX Unified Smart EDA Engine[/bold magenta] 🧠", justify="center")
        console.print("="*80)
        
        # Dataset info panel
        info_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="bold green")
        
        info_table.add_row("Dataset Shape", f"{df.shape[0]:,} × {df.shape[1]}")
        info_table.add_row("Mode", mode.upper())
        info_table.add_row("Target Variable", target if target else "Auto-Detect")
        info_table.add_row("Sample Size", f"{sample_size:,}")
        info_table.add_row("Auto Detection", "✅ Enabled" if auto_detect else "❌ Disabled")
        
        console.print(Panel(info_table, title="📊 EDA Configuration", border_style="blue"))

    # =========================================================
    # 🔍 SMART VARIABLE DETECTION & ANALYSIS (NEW)
    # =========================================================
    variable_analysis = None
    if variable_detector and auto_detect:
        if mode in ["console", "all"]:
            console.print("\n🔍 [bold cyan]SMART VARIABLE DETECTION & ANALYSIS[/bold cyan]")
            console.print("-" * 70)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("🔍 Analyzing variables...", total=None)
                variable_analysis = variable_detector.analyze_dataset(df)
                progress.update(task, description="✅ Variable analysis completed")
        else:
            variable_analysis = variable_detector.analyze_dataset(df)
        
        # Auto-detect target if not provided
        if not target and variable_analysis['recommended_target']:
            target = variable_analysis['recommended_target']
            if mode in ["console", "all"]:
                console.print(f"🎯 [bold yellow]Auto-detected target column:[/bold yellow] [bold green]{target}[/bold green]")
        
        # Display variable categorization
        if mode in ["console", "all"]:
            categories = variable_analysis['variable_categories']
            
            # Create variable summary table
            var_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
            var_table.add_column("Category", style="cyan")
            var_table.add_column("Count", style="bold green")
            var_table.add_column("Columns", style="dim")
            
            var_table.add_row("📊 Numeric Continuous", str(len(categories['numeric_continuous'])), 
                             ", ".join(categories['numeric_continuous'][:3]) + ("..." if len(categories['numeric_continuous']) > 3 else ""))
            var_table.add_row("🔢 Numeric Discrete", str(len(categories['numeric_discrete'])), 
                             ", ".join(categories['numeric_discrete'][:3]) + ("..." if len(categories['numeric_discrete']) > 3 else ""))
            var_table.add_row("🏷️ Categorical (Low)", str(len(categories['categorical_low_card'])), 
                             ", ".join(categories['categorical_low_card'][:3]) + ("..." if len(categories['categorical_low_card']) > 3 else ""))
            var_table.add_row("🏷️ Categorical (High)", str(len(categories['categorical_high_card'])), 
                             ", ".join(categories['categorical_high_card'][:2]) + ("..." if len(categories['categorical_high_card']) > 2 else ""))
            var_table.add_row("📅 DateTime", str(len(categories['datetime'])), 
                             ", ".join(categories['datetime']))
            var_table.add_row("📝 Text", str(len(categories['text'])), 
                             ", ".join(categories['text']))
            
            console.print(var_table)
            
            # Show exclusion recommendations
            exclusions = variable_analysis['columns_to_exclude']
            total_exclusions = sum(len(cols) for cols in exclusions.values())
            
            if total_exclusions > 0:
                console.print(f"\n⚠️ [bold yellow]Columns recommended for exclusion:[/bold yellow] {total_exclusions}")
                for category, cols in exclusions.items():
                    if cols:
                        console.print(f"   • {category.replace('_', ' ').title()}: {', '.join(cols[:3])}" + 
                                    ("..." if len(cols) > 3 else ""))
            
            # Show meaningful features
            meaningful = variable_analysis['meaningful_features']['all_meaningful']
            console.print(f"\n✅ [bold green]Meaningful features selected:[/bold green] {len(meaningful)}")
            if len(meaningful) > 10:
                console.print(f"   Top features: {', '.join(meaningful[:10])}...")
            else:
                console.print(f"   Features: {', '.join(meaningful)}")
        
        # Update results with variable analysis
        results = {
            "variable_analysis": variable_analysis,
            "dataset_info": {},
            "missing_analysis": {},
            "numeric_analysis": {},
            "categorical_analysis": {},
            "correlation_analysis": {},
            "target_analysis": {},
            "problem_type": variable_analysis['analysis_recommendations']['problem_type'],
            "model_recommendations": {},
            "data_quality_score": 0,
            "insights": []
        }
        
        # Use meaningful features for analysis
        meaningful_features = variable_analysis['meaningful_features']
        numeric_cols = meaningful_features['numeric_features']
        categorical_cols = meaningful_features['categorical_features']
        datetime_cols = meaningful_features['datetime_features']
    else:
        # Fallback to original column detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        
        # Initialize results dictionary
        results = {
            "dataset_info": {},
            "missing_analysis": {},
            "numeric_analysis": {},
            "categorical_analysis": {},
            "correlation_analysis": {},
            "target_analysis": {},
            "problem_type": None,
            "model_recommendations": {},
            "data_quality_score": 0,
            "insights": []
        }

    # Optional Sampling (for large datasets)
    original_rows = len(df)
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
        if mode in ["console", "all"]:
            console.print(f"\n⚡ [yellow]Dataset sampled to {sample_size:,} rows for performance[/yellow]")
    else:
        df_sample = df.copy()

    # =========================================================
    # 1️⃣ DATASET OVERVIEW
    # =========================================================
    n_rows, n_cols = df.shape
    dtypes = df.dtypes
    
    # Data types breakdown (like manual df.dtypes.value_counts())
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_counts_str = {str(k): int(v) for k, v in dtype_counts.items()}
    
    # Duplicate analysis
    n_duplicates = int(df.duplicated().sum())
    dup_pct = (n_duplicates / n_rows * 100) if n_rows > 0 else 0
    
    # Constant columns detection
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    near_constant_cols = [col for col in df.columns if 1 < df[col].nunique() <= 2 and col not in constant_cols]
    
    # High-zero columns
    high_zero_cols = []
    for col in numeric_cols:
        zero_ratio = (df[col] == 0).sum() / n_rows
        if zero_ratio > 0.5:
            high_zero_cols.append((col, zero_ratio))
    
    results["dataset_info"] = {
        "shape": (n_rows, n_cols),
        "memory_mb": df.memory_usage(deep=True).sum()/1024**2,
        "duplicates": n_duplicates,
        "duplicate_pct": round(dup_pct, 2),
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
        "datetime_cols": len(datetime_cols),
        "dtype_counts": dtype_counts_str,
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "high_zero_columns": [(c, round(r, 3)) for c, r in high_zero_cols]
    }

    if mode in ["console", "all"]:
        console.print("\n1️⃣ [bold cyan]DATASET OVERVIEW[/bold cyan]")
        console.print("-" * 70)
        console.print(f"• Rows: {n_rows:,}")
        console.print(f"• Columns: {n_cols}")
        console.print(f"• Total Cells: {df.size:,}")
        console.print(f"• Memory Usage: {results['dataset_info']['memory_mb']:.2f} MB")
        console.print(f"• Duplicate Rows: {n_duplicates:,} ({dup_pct:.2f}%)")
        
        # Data Types Breakdown
        console.print("\n📋 [bold yellow]Data Types Breakdown:[/bold yellow]")
        dtype_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        dtype_table.add_column("Data Type", style="cyan")
        dtype_table.add_column("Count", style="bold green")
        dtype_table.add_column("Percentage", style="dim")
        for dt, cnt in dtype_counts_str.items():
            pct = cnt / n_cols * 100
            dtype_table.add_row(str(dt), str(cnt), f"{pct:.1f}%")
        console.print(dtype_table)
        
        console.print(f"\n• Meaningful Numeric Columns: {len(numeric_cols)}")
        console.print(f"• Meaningful Categorical Columns: {len(categorical_cols)}")
        console.print(f"• Date Columns: {len(datetime_cols)}")
        
        # Constant/near-constant warnings
        if constant_cols:
            console.print(f"\n⚠️ [bold red]Constant Columns ({len(constant_cols)}):[/bold red] {', '.join(constant_cols[:5])}{'...' if len(constant_cols) > 5 else ''}")
        if near_constant_cols:
            console.print(f"⚠️ [yellow]Near-Constant Columns ({len(near_constant_cols)}):[/yellow] {', '.join(near_constant_cols[:5])}{'...' if len(near_constant_cols) > 5 else ''}")
        if high_zero_cols:
            console.print(f"⚠️ [yellow]High-Zero Columns ({len(high_zero_cols)}):[/yellow] {', '.join([f'{c} ({r*100:.0f}%)' for c,r in high_zero_cols[:5]])}")
        
        if variable_analysis:
            total_excluded = sum(len(cols) for cols in variable_analysis['columns_to_exclude'].values())
            console.print(f"• Excluded Columns: {total_excluded}")
        
        # Sample Data Preview (like df.head())
        console.print("\n📄 [bold yellow]Sample Data Preview (first 5 rows):[/bold yellow]")
        try:
            preview_cols = list(df.columns[:10])  # Limit to 10 columns for readability
            preview_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE, show_lines=True)
            for col in preview_cols:
                preview_table.add_column(str(col), style="dim", max_width=20, overflow="ellipsis")
            if len(df.columns) > 10:
                preview_table.add_column(f"... +{len(df.columns)-10} more", style="dim")
            for idx in range(min(5, len(df))):
                row_vals = [str(df.iloc[idx][col])[:20] for col in preview_cols]
                if len(df.columns) > 10:
                    row_vals.append("...")
                preview_table.add_row(*row_vals)
            console.print(preview_table)
        except Exception:
            console.print("   (Could not render preview)")

    # =========================================================
    # 2️⃣ MISSING VALUES ANALYSIS
    # =========================================================
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows) * 100
    missing_df = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    results["missing_analysis"] = {
        "total_missing": int(missing.sum()),
        "missing_percentage": float(missing.sum() / (n_rows * n_cols) * 100),
        "columns_with_missing": len(missing_df),
        "missing_by_column": missing_df.to_dict()
    }

    if mode in ["console", "all"]:
        console.print("\n2️⃣ [bold cyan]MISSING VALUE ANALYSIS[/bold cyan]")
        console.print("-" * 70)
        if missing_df.empty:
            console.print("✔ No missing values in dataset.")
        else:
            total_missing = missing.sum()
            console.print(f"⚠ Missing Values Found: {total_missing:,}")
            console.print("\nTop Missing Columns:")
            for col, pct in missing_df.head(8).items():
                val = missing[col]
                console.print(f"• {col:20s} → {val:8,} missing ({pct:.2f}%)")

    # =========================================================
    # 3️⃣ NUMERIC ANALYSIS WITH ENHANCED INSIGHTS
    # =========================================================
    if numeric_cols:
        desc = df_sample[numeric_cols].describe().T
        desc["skew"] = df_sample[numeric_cols].skew()
        desc["missing_%"] = (df[numeric_cols].isnull().sum() / len(df)) * 100
        # Enhanced stats: IQR, Range, Coefficient of Variation
        desc["IQR"] = desc["75%"] - desc["25%"]
        desc["range"] = desc["max"] - desc["min"]
        desc["CV"] = (desc["std"] / desc["mean"]).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        results["numeric_analysis"] = {
            "descriptive_stats": desc.to_dict(),
            "outlier_analysis": {},
            "distribution_insights": {}
        }

        if mode in ["console", "all"]:
            console.print("\n3️⃣ [bold cyan]NUMERIC FEATURE PROFILE[/bold cyan]")
            console.print("-" * 70)
            console.print(desc[["mean", "std", "min", "25%", "50%", "75%", "max", "IQR", "range", "CV", "skew", "missing_%"]]
                          .head(10)
                          .round(3)
                          .to_string())

            # Enhanced outlier discovery
            console.print("\n📌 [bold yellow]Advanced Outlier Detection[/bold yellow]")
            for col in numeric_cols[:5]:  # Limit for performance
                outlier_methods = insights_engine._advanced_outlier_detection(df[col].dropna())
                results["numeric_analysis"]["outlier_analysis"][col] = outlier_methods
                
                consensus_outliers = np.mean(list(outlier_methods.values()))
                if consensus_outliers > 0:
                    console.print(f"• {col}: {consensus_outliers:.0f} outliers (consensus)")

            # Enhanced skewness analysis
            console.print("\n📊 [bold yellow]Distribution Insights[/bold yellow]")
            for col in numeric_cols[:5]:
                clean_data = df[col].dropna()
                if len(clean_data) > 10:
                    skewness = stats.skew(clean_data)
                    num_modes, _, _, _ = insights_engine._detect_multimodal(clean_data)
                    
                    results["numeric_analysis"]["distribution_insights"][col] = {
                        "skewness": float(skewness),
                        "modes": int(num_modes)
                    }
                    
                    skew_desc = "symmetric" if abs(skewness) < 0.5 else ("right-skewed" if skewness > 0 else "left-skewed")
                    mode_desc = f"{num_modes} mode(s)"
                    console.print(f"• {col:20s} → {skew_desc}, {mode_desc}")

    # =========================================================
    # 4️⃣ CATEGORICAL ANALYSIS
    # =========================================================
    if categorical_cols:
        results["categorical_analysis"] = {
            "cardinality": {},
            "balance_analysis": {},
            "high_cardinality_cols": [],
            "entropy": {}
        }

        if mode in ["console", "all"]:
            console.print("\n4️⃣ [bold cyan]CATEGORICAL FEATURE PROFILE[/bold cyan]")
            console.print("-" * 70)
            
            for col in categorical_cols[:8]:
                unique = df[col].nunique()
                results["categorical_analysis"]["cardinality"][col] = int(unique)
                
                # High cardinality check
                if unique > n_rows * 0.5:
                    results["categorical_analysis"]["high_cardinality_cols"].append(col)
                
                # Entropy and balance analysis
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    probabilities = value_counts / value_counts.sum()
                    entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-8)))
                    max_entropy = float(np.log2(len(value_counts)))
                    balance_ratio = round(entropy / max_entropy, 3) if max_entropy > 0 else 0
                    imbalance_ratio = round(float(value_counts.iloc[0] / value_counts.iloc[-1]), 2) if value_counts.iloc[-1] > 0 else float('inf')
                else:
                    entropy = 0.0
                    balance_ratio = 0.0
                    imbalance_ratio = float('inf')
                
                results["categorical_analysis"]["entropy"][col] = {
                    "entropy": round(entropy, 3),
                    "balance_ratio": balance_ratio,
                    "imbalance_ratio": imbalance_ratio
                }
                results["categorical_analysis"]["balance_analysis"][col] = balance_ratio
                
                top = df[col].value_counts().head(3)
                console.print(f"\n📌 {col}")
                console.print(f"• Unique Values: {unique}")
                console.print(f"• Entropy: {entropy:.3f} | Balance: {balance_ratio:.3f} | Top/Bottom Ratio: {imbalance_ratio:.1f}x")
                for val, cnt in top.items():
                    pct = 100 * cnt / len(df)
                    console.print(f"   - {val}  ({pct:.2f}%)")

    # =========================================================
    # 5️⃣ CORRELATION ANALYSIS
    # =========================================================
    if len(numeric_cols) > 1:
        corr = df_sample[numeric_cols].corr()
        strong_correlations = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append((corr.index[i], corr.columns[j], val))

        results["correlation_analysis"] = {
            "correlation_matrix": corr.to_dict(),
            "strong_correlations": strong_correlations
        }

        if mode in ["console", "all"]:
            console.print("\n5️⃣ [bold cyan]CORRELATION INTELLIGENCE[/bold cyan]")
            console.print("-" * 70)
            if strong_correlations:
                strong_correlations = sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, val in strong_correlations[:10]:
                    relation = "Positive" if val > 0 else "Negative"
                    console.print(f"• {col1} ↔ {col2} → {val:.3f} ({relation})")
            else:
                console.print("No strong correlations found (|r| ≥ 0.5).")

    # =========================================================
    # 6️⃣ ADVANCED STATISTICAL ANALYSIS (NEW)
    # =========================================================
    if advanced_stats_engine and numeric_cols:
        if mode in ["console", "all"]:
            console.print("\n6️⃣ [bold cyan]ADVANCED STATISTICAL ANALYSIS[/bold cyan]")
            console.print("-" * 70)
        
        advanced_results = {
            'normality_tests': {},
            'advanced_correlations': {},
            'statistical_tests': {},
            'outlier_analysis': {}
        }
        
        # Normality tests for key numeric columns
        for col in numeric_cols[:5]:  # Limit for performance
            normality_result = advanced_stats_engine.test_normality(df[col], col)
            advanced_results['normality_tests'][col] = normality_result
            
            if mode in ["console", "all"]:
                console.print(f"\n📊 Normality Test: {col}")
                console.print(f"   • Normal Distribution: {'✅ Yes' if normality_result['is_normal'] else '❌ No'}")
                if 'consensus_score' in normality_result:
                    console.print(f"   • Consensus Score: {normality_result['consensus_score']:.2f}")
        
        # Advanced correlation analysis
        if len(numeric_cols) > 1:
            adv_corr_result = advanced_stats_engine.advanced_correlation_analysis(df_sample, numeric_cols[:10])
            advanced_results['advanced_correlations'] = adv_corr_result
            
            if mode in ["console", "all"] and 'significant_pairs' in adv_corr_result:
                console.print(f"\n🔍 Advanced Correlations: {len(adv_corr_result['significant_pairs'])} significant pairs")
                for pair in adv_corr_result['significant_pairs'][:3]:
                    console.print(f"   • {pair['variable1']} ↔ {pair['variable2']}: r={pair['pearson_r']:.3f} ({pair['strength']})")
        
        # Statistical tests suite
        stats_tests_result = advanced_stats_engine.statistical_tests_suite(df_sample, target)
        advanced_results['statistical_tests'] = stats_tests_result
        
        if mode in ["console", "all"]:
            console.print(f"\n📈 Statistical Tests: {len(stats_tests_result['tests_performed'])} tests performed")
            console.print(f"   • Significant Results: {len(stats_tests_result['significant_results'])}")
        
        # Enhanced outlier detection
        for col in numeric_cols[:3]:  # Limit for performance
            outlier_result = advanced_stats_engine.detect_outliers_advanced(df[col], col)
            advanced_results['outlier_analysis'][col] = outlier_result
            
            if mode in ["console", "all"]:
                outlier_pct = outlier_result.get('outlier_percentage', 0)
                console.print(f"   • {col}: {outlier_pct:.1f}% outliers detected")
        
        # Add to main results
        results['advanced_statistical_analysis'] = advanced_results

    # =========================================================
    # 7️⃣ AI-POWERED INSIGHTS & RECOMMENDATIONS (NEW)
    # =========================================================
    if advanced_insights_engine and ai_insights:
        if mode in ["console", "all"]:
            console.print("\n7️⃣ [bold cyan]AI-POWERED INSIGHTS & RECOMMENDATIONS[/bold cyan]")
            console.print("-" * 70)
        
        # Data quality assessment
        quality_assessment = advanced_insights_engine.assess_data_quality(df, target)
        results['data_quality_assessment'] = quality_assessment
        
        if mode in ["console", "all"]:
            console.print(f"📊 Data Quality Score: {quality_assessment['overall_score']:.1f}/100")
            console.print(f"🎯 Model Readiness: {quality_assessment['model_readiness'].replace('_', ' ').title()}")
            
            if quality_assessment['quality_issues']:
                console.print("\n⚠️ Quality Issues Detected:")
                for issue in quality_assessment['quality_issues'][:3]:
                    console.print(f"   • {issue}")
        
        # Generate comprehensive recommendations
        recommendations = advanced_insights_engine.generate_recommendations(df, results, target)
        results['ai_recommendations'] = recommendations
        
        if mode in ["console", "all"]:
            console.print(f"\n🚀 Recommendations Priority: {recommendations['priority_level'].upper()}")
            console.print(f"📋 Preprocessing Steps: {len(recommendations['preprocessing_steps'])}")
            console.print(f"🔧 Feature Engineering: {len(recommendations['feature_engineering'])}")
            console.print(f"🤖 Model Suggestions: {len(recommendations['model_selection'])}")
            
            if recommendations['next_steps']:
                console.print("\n💡 Next Steps Roadmap:")
                for step in recommendations['next_steps'][:5]:
                    console.print(f"   {step}")

    # =========================================================
    # 8️⃣ TARGET ANALYSIS & PROBLEM TYPE INFERENCE
    # =========================================================
    if variable_analysis and variable_analysis['recommended_target']:
        # Use the auto-detected target
        target = variable_analysis['recommended_target']
        problem_type = variable_analysis['analysis_recommendations']['problem_type']
        reason = f"auto_detected_{problem_type}"
    else:
        # Fallback to original logic
        problem_type, reason = insights_engine._infer_problem_type(df, target)
    
    y = df[target] if target and target in df.columns else None
    imbalance_flag, majority_ratio = _check_imbalance(y)
    
    results["problem_type"] = problem_type
    results["target_analysis"] = {
        "target_column": target,
        "problem_type": problem_type,
        "inference_reason": reason,
        "imbalance_detected": imbalance_flag,
        "majority_class_ratio": majority_ratio,
        "auto_detected": variable_analysis is not None and variable_analysis['recommended_target'] == target
    }

    if mode in ["console", "all"]:
        console.print("\n8️⃣ [bold cyan]TARGET ANALYSIS & ML PROBLEM TYPE[/bold cyan]")
        console.print("-" * 70)
        
        if target is None:
            console.print("🎯 Target: NOT PROVIDED — cannot infer ML problem type.")
        elif target not in df.columns:
            console.print(f"🎯 Target: '{target}' ❌ (column not found)")
        else:
            console.print(f"🎯 Target Column: {target}")
            if results["target_analysis"].get("auto_detected", False):
                console.print("   🤖 [bold green]Auto-detected by Smart Variable Detector[/bold green]")
            console.print(f"   • Dtype: {y.dtype}")
            console.print(f"   • Unique Values: {y.nunique()}")
            console.print(f"   • Missing: {y.isnull().sum()} ({y.isnull().mean()*100:.2f}%)")

            if problem_type is None:
                console.print("   ❌ Invalid or constant target — cannot define a problem.")
            else:
                console.print(f"   ✔ Inferred Problem Type: {problem_type.upper()} ({reason})")

            if problem_type == "classification" and imbalance_flag:
                console.print(f"   ⚠ Class Imbalance Detected — majority class = {majority_ratio*100:.1f}%")
            
            # Target distribution summary
            if problem_type == "classification" or (y.nunique() <= 20):
                console.print("\n   📊 [bold yellow]Target Distribution:[/bold yellow]")
                target_vc = y.value_counts()
                for val, cnt in target_vc.head(10).items():
                    pct = cnt / len(y) * 100
                    bar_len = int(pct / 2)
                    bar = '█' * bar_len
                    console.print(f"      {str(val):15s} → {cnt:8,} ({pct:5.1f}%) {bar}")
                if len(target_vc) > 10:
                    console.print(f"      ... and {len(target_vc) - 10} more classes")
            elif problem_type == "regression":
                console.print("\n   📊 [bold yellow]Target Distribution (Regression):[/bold yellow]")
                console.print(f"      Mean: {y.mean():.4f} | Median: {y.median():.4f} | Std: {y.std():.4f}")
                console.print(f"      Min: {y.min():.4f} | Max: {y.max():.4f}")
                console.print(f"      Skewness: {y.skew():.4f} | Kurtosis: {y.kurtosis():.4f}")

    # =========================================================
    # 9️⃣ MODEL RECOMMENDATIONS
    # =========================================================
    model_rec = _model_recommendations(problem_type, df, target, imbalance_flag)
    results["model_recommendations"] = model_rec

    if mode in ["console", "all"]:
        console.print("\n9️⃣ [bold cyan]AI MODEL RECOMMENDATIONS[/bold cyan]")
        console.print("-" * 70)
        console.print(f"   • Problem Type: {model_rec['type']}")
        if model_rec["baseline"]:
            console.print("   • Baseline Models:")
            for m in model_rec["baseline"]:
                console.print(f"       - {m}")
        if model_rec["advanced"]:
            console.print("   • Advanced Models:")
            for m in model_rec["advanced"]:
                console.print(f"       - {m}")
        if model_rec["notes"]:
            console.print("   • Notes:")
            for n in model_rec["notes"]:
                console.print(f"       - {n}")

    # =========================================================
    # 🔟 DATA QUALITY SCORE (Proportional)
    # =========================================================
    score = 100.0
    missing_pct_total = results["missing_analysis"]["missing_percentage"]
    high_card_count = len(results["categorical_analysis"].get("high_cardinality_cols", []))
    n_constant = len(results["dataset_info"].get("constant_columns", []))
    dup_pct_val = results["dataset_info"].get("duplicate_pct", 0)
    
    quality_breakdown = {}
    
    # Missing values: proportional penalty (max -30)
    missing_penalty = min(30, missing_pct_total * 0.6)
    score -= missing_penalty
    quality_breakdown["Missing Values"] = f"-{missing_penalty:.1f}"
    
    # Duplicates: proportional penalty (max -15)
    dup_penalty = min(15, dup_pct_val * 0.3)
    score -= dup_penalty
    quality_breakdown["Duplicates"] = f"-{dup_penalty:.1f}"
    
    # Constant columns penalty (max -15)
    if n_constant > 0:
        const_penalty = min(15, n_constant * 5)
        score -= const_penalty
        quality_breakdown["Constant Columns"] = f"-{const_penalty:.1f}"
    
    # High cardinality (max -10)
    if high_card_count > 0:
        hc_penalty = min(10, high_card_count * 3)
        score -= hc_penalty
        quality_breakdown["High Cardinality"] = f"-{hc_penalty:.1f}"
    
    # No numeric columns
    if len(numeric_cols) == 0:
        score -= 15
        quality_breakdown["No Numeric Cols"] = "-15.0"
    
    # Memory usage
    if results["dataset_info"]["memory_mb"] > 200:
        score -= 5
        quality_breakdown["High Memory"] = "-5.0"
    
    # Invalid target
    if problem_type is None and target is not None:
        score -= 10
        quality_breakdown["Invalid Target"] = "-10.0"
    
    # Highly skewed features penalty (max -10)
    if numeric_cols:
        n_highly_skewed = sum(1 for col in numeric_cols[:10] if abs(df_sample[col].skew()) > 2)
        if n_highly_skewed > 0:
            skew_penalty = min(10, n_highly_skewed * 2)
            score -= skew_penalty
            quality_breakdown["Highly Skewed"] = f"-{skew_penalty:.1f}"

    results["data_quality_score"] = max(int(round(score)), 1)
    results["quality_breakdown"] = quality_breakdown

    if mode in ["console", "all"]:
        console.print(f"\n🔟 [bold cyan]DATA QUALITY SCORE[/bold cyan]")
        console.print("-" * 70)
        
        # Color the score
        q_score = results['data_quality_score']
        if q_score >= 80:
            score_color = "bold green"
        elif q_score >= 60:
            score_color = "bold yellow"
        elif q_score >= 40:
            score_color = "bold orange1"
        else:
            score_color = "bold red"
        
        console.print(f"💯 Dataset Quality Score: [{score_color}]{q_score} / 100[/{score_color}]")
        
        if quality_breakdown:
            console.print("\n   Breakdown:")
            for factor, penalty in quality_breakdown.items():
                console.print(f"      • {factor}: {penalty} pts")

    # =========================================================
    # 1️⃣1️⃣ ENHANCED INTERACTIVE VISUALIZATIONS
    # =========================================================
    if show_visualizations and mode in ["interactive", "all"]:
        console.print("\n1️⃣1️⃣ [bold cyan]ENHANCED INTERACTIVE VISUALIZATIONS[/bold cyan]")
        console.print("-" * 70)
        
        plot_count = 0
        
        # Use advanced plotter if available
        if big_data_plotter and ADVANCED_MODULES_AVAILABLE:
            console.print("🚀 [bold yellow]Using Advanced Big Data Plotting Engine...[/bold yellow]")
            
            # Smart sample the data for visualization
            df_viz = big_data_plotter.smart_sample_for_visualization(df_sample, target_col=target)
            console.print(f"📊 Sampled {len(df_viz):,} rows for optimal visualization performance")
            
            # Advanced distribution plots
            if numeric_cols and plot_count < max_plots:
                console.print("\n📊 [bold yellow]Advanced Distribution Analysis...[/bold yellow]")
                for col in numeric_cols[:min(2, max_plots - plot_count)]:
                    try:
                        # Create diagnostic plots
                        fig_diag = big_data_plotter.create_diagnostic_plots(df_viz[col], col)
                        fig_diag.show()
                        
                        # Create distribution plots
                        fig_dist = big_data_plotter.create_distribution_plots(df_viz, col, target)
                        fig_dist.show()
                        
                        plot_count += 2
                    except Exception as e:
                        console.print(f"⚠️ Could not create advanced plots for {col}: {str(e)}")
            
            # Advanced relationship plots
            if len(numeric_cols) > 1 and plot_count < max_plots:
                console.print("\n🔗 [bold yellow]Advanced Relationship Analysis...[/bold yellow]")
                try:
                    # Create relationship plot with regression
                    fig_rel = big_data_plotter.create_relationship_plots(
                        df_viz, numeric_cols[0], numeric_cols[1], target, plot_type='regression'
                    )
                    fig_rel.show()
                    plot_count += 1
                except Exception as e:
                    console.print(f"⚠️ Could not create relationship plot: {str(e)}")
            
            # Advanced categorical plots
            if categorical_cols and plot_count < max_plots:
                console.print("\n🏷️ [bold yellow]Advanced Categorical Analysis...[/bold yellow]")
                for col in categorical_cols[:min(1, max_plots - plot_count)]:
                    try:
                        fig_cat = big_data_plotter.create_categorical_plots(df_viz, col, target)
                        fig_cat.show()
                        plot_count += 1
                    except Exception as e:
                        console.print(f"⚠️ Could not create categorical plot for {col}: {str(e)}")
        
        else:
            # Fallback to original plotting
            console.print("📊 [bold yellow]Using Standard Plotting Engine...[/bold yellow]")
            
            # Distribution plots for numeric variables
            if numeric_cols and plot_count < max_plots:
                console.print("📊 [bold yellow]Generating Distribution Plots...[/bold yellow]")
                for col in numeric_cols[:min(3, max_plots - plot_count)]:
                    fig = _create_interactive_distribution(df_sample, col)
                    _display_plotly_figure(fig)
                    plot_count += 1
            
            # Correlation heatmap
            if len(numeric_cols) > 1 and plot_count < max_plots:
                console.print("\n🔥 [bold yellow]Generating Correlation Heatmap...[/bold yellow]")
                fig = _create_interactive_correlation(df_sample, numeric_cols[:10])
                _display_plotly_figure(fig)
                plot_count += 1
            
            # Categorical plots
            if categorical_cols and plot_count < max_plots:
                console.print("\n🏷️ [bold yellow]Generating Categorical Plots...[/bold yellow]")
                for col in categorical_cols[:min(2, max_plots - plot_count)]:
                    fig = _create_interactive_categorical(df_sample, col, target)
                    _display_plotly_figure(fig)
                    plot_count += 1
        
        console.print(f"\n✅ Generated {plot_count} interactive visualizations")

    # =========================================================
    # 1️⃣2️⃣ HTML REPORT GENERATION
    # =========================================================
    if mode in ["html", "all"]:
        console.print("\n1️⃣2️⃣ [bold cyan]GENERATING ENHANCED HTML REPORT[/bold cyan]")
        console.print("-" * 70)
        
        # Generate comprehensive HTML report
        html_sections = []
        
        # Dataset overview section
        overview_html = f"""
        <section>
          <h2>1. Dataset Overview</h2>
          <ul>
            <li><b>Shape:</b> {n_rows:,} rows × {n_cols} columns</li>
            <li><b>Total cells:</b> {n_rows * n_cols:,}</li>
            <li><b>Duplicates:</b> {results['dataset_info']['duplicates']:,} rows</li>
            <li><b>Memory usage:</b> {results['dataset_info']['memory_mb']:.2f} MB</li>
            <li><b>Data Quality Score:</b> {results['data_quality_score']}/100</li>
          </ul>
          <h3>Column Types</h3>
          {dtypes.to_frame("dtype").to_html(classes="simple-table", border=0)}
        </section>
        """
        html_sections.append(overview_html)
        
        # Missing values section
        if not missing_df.empty:
            missing_html = f"""
            <section>
              <h2>2. Missing Values Analysis</h2>
              <p><b>Total missing cells:</b> {results['missing_analysis']['total_missing']:,} 
                 ({results['missing_analysis']['missing_percentage']:.2f}% of all cells)</p>
              {pd.DataFrame(list(results['missing_analysis']['missing_by_column'].items()), 
                           columns=['Column', 'Missing_%']).to_html(classes="simple-table", border=0, index=False)}
            </section>
            """
        else:
            missing_html = """
            <section>
              <h2>2. Missing Values Analysis</h2>
              <p><b>No missing values detected.</b></p>
            </section>
            """
        html_sections.append(missing_html)
        
        # Target analysis section
        if target and problem_type:
            baseline_models_html = ''.join([f'<li>{model}</li>' for model in model_rec['baseline']])
            advanced_models_html = ''.join([f'<li>{model}</li>' for model in model_rec['advanced']])
            target_html = f"""
            <section>
              <h2>3. Target Analysis & ML Recommendations</h2>
              <p><b>Target Column:</b> {target}</p>
              <p><b>Problem Type:</b> {problem_type.upper()}</p>
              <p><b>Imbalance Detected:</b> {'Yes' if imbalance_flag else 'No'}</p>
              <h3>Recommended Models</h3>
              <h4>Baseline Models:</h4>
              <ul>
                {baseline_models_html}
              </ul>
              <h4>Advanced Models:</h4>
              <ul>
                {advanced_models_html}
              </ul>
            </section>
            """
            html_sections.append(target_html)
        
        # Generate final HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>EssentiaX Unified EDA Report</title>
          <style>
            body {{
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              margin: 0;
              padding: 0 20px 30px 20px;
              background: #f7f9fb;
              color: #2c3e50;
            }}
            h1 {{
              text-align: center;
              margin-top: 20px;
              color: #3498db;
            }}
            h2 {{
              border-bottom: 2px solid #3498db;
              padding-bottom: 4px;
              margin-top: 30px;
            }}
            section {{
              background: #ffffff;
              margin-top: 20px;
              padding: 15px 20px;
              border-radius: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .simple-table {{
              border-collapse: collapse;
              width: 100%;
              margin-top: 10px;
              font-size: 13px;
            }}
            .simple-table th, .simple-table td {{
              border: 1px solid #ecf0f1;
              padding: 6px 8px;
              text-align: left;
            }}
            .simple-table th {{
              background-color: #f0f4f8;
            }}
          </style>
        </head>
        <body>
          <h1>🧠 EssentiaX Unified EDA Report</h1>
          <p style="text-align:center; color:#7f8c8d;">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            — Rows: {n_rows:,}, Columns: {n_cols}
          </p>
          {''.join(html_sections)}
        </body>
        </html>
        """
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        console.print(f"✅ [green]HTML report saved to: {report_path}[/green]")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    if mode in ["console", "all"]:
        console.print("\n" + "="*80)
        console.print("✅ [bold green]EDA COMPLETED — EssentiaX Intelligence Report[/bold green]")
        console.print("="*80)

        console.print("\n💡 [bold yellow]Recommended Next Steps:[/bold yellow]")
        if advanced_insights_engine and 'ai_recommendations' in results:
            # Use AI-generated recommendations if available
            ai_recs = results['ai_recommendations']
            if ai_recs['next_steps']:
                for step in ai_recs['next_steps'][:5]:
                    console.print(f"{step}")
        else:
            # Fallback to standard recommendations
            console.print("1. Use smart_clean() to handle missing values & encode data.")
            console.print("2. Remove outliers if they distort your model.")
            console.print("3. Normalize numerical columns before ML.")
            console.print("4. Perform feature engineering on categorical values.")
            if problem_type:
                console.print(f"5. Try recommended {problem_type} models.")
        
        # Add information about advanced features
        if ADVANCED_MODULES_AVAILABLE:
            console.print(f"\n🚀 [bold green]Advanced Features Used:[/bold green]")
            if advanced_stats:
                console.print("   ✅ Advanced Statistical Analysis")
            if ai_insights:
                console.print("   ✅ AI-Powered Insights & Recommendations")
            if show_visualizations:
                console.print("   ✅ Big Data Optimized Visualizations")
        else:
            console.print(f"\n💡 [bold yellow]Tip:[/bold yellow] Install advanced modules for enhanced analysis:")
            console.print("   pip install scikit-learn scipy plotly")
        
        console.print("\n")

    return results


# Legacy function for backward compatibility
def smart_eda_legacy(df: pd.DataFrame, sample_size: int = 50000):
    """Legacy function - redirects to new unified smart_eda"""
    return smart_eda(df=df, mode="console", sample_size=sample_size)



# Additional unified functions for backward compatibility and extended functionality

def problem_card(df: pd.DataFrame, target: str = None):
    """
    Unified Problem Card - now redirects to smart_eda with console mode
    """
    return smart_eda(df=df, target=target, mode="console", show_visualizations=False)


def smart_eda_pro(
    df: pd.DataFrame,
    target: str = None,
    report_path: str = "essentiax_report.html",
    sample_size: int = 5000,
    max_cat_unique: int = 50,
):
    """
    Unified EDA Pro - now redirects to smart_eda with HTML mode
    """
    return smart_eda(
        df=df, 
        target=target, 
        mode="html", 
        sample_size=sample_size,
        report_path=report_path,
        show_visualizations=False
    )


def smart_viz(
    df: pd.DataFrame,
    mode: str = "auto",
    columns: list = None,
    target: str = None,
    max_plots: int = 8,
    interactive: bool = True,
    sample_size: int = 10000
):
    """
    Unified Smart Visualization - now redirects to smart_eda with interactive mode
    """
    if mode == "manual" and columns:
        # For manual mode, we'll create a subset and run EDA
        df_subset = df[columns] if columns else df
        return smart_eda(
            df=df_subset,
            target=target,
            mode="interactive",
            sample_size=sample_size,
            max_plots=max_plots,
            show_visualizations=True
        )
    else:
        # Auto mode
        return smart_eda(
            df=df,
            target=target,
            mode="interactive",
            sample_size=sample_size,
            max_plots=max_plots,
            show_visualizations=True
        )


# For local testing
if __name__ == "__main__":
    # Create test dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "numeric_normal": np.random.normal(50, 10, 1000),
        "numeric_skewed": np.random.exponential(2, 1000),
        "categorical_balanced": np.random.choice(["A", "B", "C"], 1000),
        "categorical_imbalanced": np.random.choice(["X", "Y", "Z"], 1000, p=[0.7, 0.2, 0.1]),
        "target_classification": np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
        "high_cardinality": [f"item_{i}" for i in np.random.randint(0, 500, 1000)]
    })
    
    # Add some missing values
    df.loc[50:100, "numeric_normal"] = np.nan
    df.loc[200:250, "categorical_balanced"] = np.nan
    
    # Add some duplicates
    df = pd.concat([df, df.iloc[:50]], ignore_index=True)
    
    console.print("\n🧪 [bold cyan]Testing EssentiaX Unified Smart EDA[/bold cyan]")
    console.print("="*60)
    
    # Test different modes
    console.print("\n1️⃣ [bold yellow]Testing Console Mode[/bold yellow]")
    results_console = smart_eda(df, target="target_classification", mode="console")
    
    console.print("\n2️⃣ [bold yellow]Testing Interactive Mode[/bold yellow]")
    results_interactive = smart_eda(df, target="target_classification", mode="interactive", max_plots=3)
    
    console.print("\n3️⃣ [bold yellow]Testing HTML Mode[/bold yellow]")
    results_html = smart_eda(df, target="target_classification", mode="html", report_path="test_report.html")
    
    console.print("\n4️⃣ [bold yellow]Testing All Modes[/bold yellow]")
    results_all = smart_eda(df, target="target_classification", mode="all", max_plots=2)
    
    console.print("\n✅ [bold green]All tests completed successfully![/bold green]")
    console.print(f"📊 Data Quality Score: {results_console['data_quality_score']}/100")
    console.print(f"🎯 Problem Type: {results_console['problem_type']}")
    console.print(f"📈 Strong Correlations: {len(results_console['correlation_analysis']['strong_correlations'])}")
