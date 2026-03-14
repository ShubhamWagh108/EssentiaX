"""
smartViz.py — EssentiaX Next-Gen Smart Visualization Engine
=========================================================
🎨 GOATED UI/UX with AI-Powered Insights
🤖 Automatic + Manual Variable Selection
📊 Intelligent Chart Selection
💡 Real-time Interpretation & Insights
🎯 Professional Grade Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
from scipy import stats
from sklearn.preprocessing import LabelEncoder

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


# Module-level flag: ensures stream cleanup happens only ONCE after rich.progress
_stream_cleaned_after_progress = False

def _display_plotly_figure(fig):
    """
    Display Plotly figure with guaranteed rendering in Colab/Jupyter environments.
    
    CRITICAL: Does NOT call clear_output() — that would destroy all previous charts.
    Instead, uses targeted stream flushing + direct display() for reliable rendering.
    """
    global _stream_cleaned_after_progress
    
    try:
        # Flush all output streams to prevent corruption
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Flush rich console buffer
        try:
            console.file.flush()
        except:
            pass
        
        if _ENVIRONMENT == 'colab':
            try:
                from IPython.display import display, HTML
                
                # One-time stream reset after rich.progress (only first chart)
                if not _stream_cleaned_after_progress:
                    time.sleep(0.1)
                    _stream_cleaned_after_progress = True
                
                # Display using IPython display — preserves all prior output
                display(fig)
                
                # Small delay to let Colab finish rendering before next chart
                time.sleep(0.1)
                
            except Exception as e:
                # Fallback: explicit renderer
                try:
                    fig.show(renderer='colab')
                except:
                    try:
                        html_str = fig.to_html(
                            include_plotlyjs='cdn',
                            div_id=f'plotly-div-{id(fig)}',
                            config={'displayModeBar': True, 'responsive': True}
                        )
                        from IPython.display import display, HTML
                        display(HTML(html_str))
                    except:
                        fig.show()
                    
        elif _ENVIRONMENT == 'jupyter':
            try:
                from IPython.display import display
                display(fig)
            except:
                fig.show(renderer='notebook')
                
        else:
            # Terminal or other: Use standard show
            fig.show()
            
    except Exception as e:
        try:
            fig.show()
        except Exception as show_error:
            console.print(f"[yellow]⚠️ Warning: Could not display plot. Error: {show_error}[/yellow]")
            console.print("[yellow]💡 Tip: Try running in Jupyter/Colab for interactive plots[/yellow]")


# Set modern styling
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


class VizInsights:
    """Enhanced AI-Powered Visualization Insights Generator"""
    
    @staticmethod
    def _detect_multimodal(data):
        """Advanced multimodal distribution detection using kernel density estimation"""
        if not SCIPY_ADVANCED_AVAILABLE or len(data) < 50:
            return 1, [], [], []
        
        try:
            # Create KDE
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            density = kde(x_range)
            
            # Find peaks in the density
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
    def _refined_categorical_balance(data):
        """Enhanced categorical balance analysis"""
        value_counts = data.value_counts()
        total_count = len(data)
        unique_count = data.nunique()
        
        # Calculate various balance metrics
        proportions = value_counts / total_count
        
        # Gini coefficient for inequality measurement
        sorted_props = np.sort(proportions.values)
        n = len(sorted_props)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_props))) / (n * np.sum(sorted_props)) - (n + 1) / n
        
        # Entropy for information content
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(unique_count) if unique_count > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Simpson's diversity index
        simpson_index = 1 - np.sum(proportions ** 2)
        
        return {
            'gini_coefficient': gini,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'simpson_diversity': simpson_index,
            'top_category_pct': proportions.iloc[0] * 100,
            'effective_categories': 1 / np.sum(proportions ** 2)  # Effective number of categories
        }
    
    @staticmethod
    def analyze_distribution(data, column_name):
        """Enhanced distribution analysis with advanced insights"""
        clean_data = data.dropna()
        mean_val = clean_data.mean()
        median_val = clean_data.median()
        std_val = clean_data.std()
        skewness = stats.skew(clean_data)
        kurtosis = stats.kurtosis(clean_data)
        
        insights = []
        
        # Enhanced skewness analysis with confidence
        skew_abs = abs(skewness)
        if skew_abs < 0.5:
            confidence = "high" if skew_abs < 0.2 else "moderate"
            insights.append(f"📊 **Symmetric Distribution**: {column_name} shows symmetric distribution ({confidence} confidence)")
        elif skewness > 0.5:
            if skewness > 2:
                insights.append(f"📈 **Heavily Right Skewed**: {column_name} has extreme positive skew (skew={skewness:.2f})")
            else:
                insights.append(f"📈 **Right Skewed**: {column_name} has moderate positive skew (skew={skewness:.2f})")
        else:
            if skewness < -2:
                insights.append(f"📉 **Heavily Left Skewed**: {column_name} has extreme negative skew (skew={skewness:.2f})")
            else:
                insights.append(f"📉 **Left Skewed**: {column_name} has moderate negative skew (skew={skewness:.2f})")
        
        # Multimodal detection
        num_modes, peaks, density, x_range = VizInsights._detect_multimodal(clean_data)
        if num_modes > 1:
            insights.append(f"🎭 **Multimodal Distribution**: Detected {num_modes} distinct peaks - suggests multiple subgroups")
        elif num_modes == 1:
            insights.append(f"🎯 **Unimodal Distribution**: Single peak detected - homogeneous population")
        
        # Enhanced outlier detection
        outlier_results = VizInsights._advanced_outlier_detection(clean_data)
        outlier_consensus = np.mean(list(outlier_results.values()))
        outlier_pct = (outlier_consensus / len(clean_data)) * 100
        
        if outlier_pct > 10:
            method_agreement = len([v for v in outlier_results.values() if v > len(clean_data) * 0.05])
            if method_agreement >= 2:
                insights.append(f"🚨 **High Outlier Content**: {outlier_pct:.1f}% outliers detected (multiple methods agree)")
            else:
                insights.append(f"⚠️ **Potential Outliers**: {outlier_pct:.1f}% outliers detected (method-dependent)")
        elif outlier_pct > 5:
            insights.append(f"📊 **Moderate Outliers**: {outlier_pct:.1f}% outliers detected - typical for real data")
        else:
            insights.append(f"✅ **Clean Distribution**: {outlier_pct:.1f}% outliers - well-behaved data")
        
        # Kurtosis analysis (tail behavior)
        if kurtosis > 3:
            insights.append(f"📏 **Heavy Tails**: High kurtosis ({kurtosis:.2f}) indicates fat tails and extreme values")
        elif kurtosis < -1:
            insights.append(f"📏 **Light Tails**: Low kurtosis ({kurtosis:.2f}) indicates thin tails and bounded distribution")
        
        # Central tendency relationship with enhanced interpretation
        mean_median_diff = abs(mean_val - median_val)
        relative_diff = mean_median_diff / std_val if std_val > 0 else 0
        
        if relative_diff < 0.1:
            insights.append(f"🎯 **Balanced Central Tendency**: Mean ({mean_val:.2f}) ≈ Median ({median_val:.2f})")
        elif relative_diff < 0.3:
            insights.append(f"⚖️ **Slight Asymmetry**: Mean ({mean_val:.2f}) vs Median ({median_val:.2f}) - minor skewness")
        else:
            direction = "higher" if mean_val > median_val else "lower"
            insights.append(f"⚖️ **Significant Asymmetry**: Mean {direction} than median - strong skewness or outliers")
        
        return insights
    
    @staticmethod
    def analyze_correlation(corr_value, var1, var2):
        """Enhanced correlation analysis with statistical significance"""
        insights = []
        
        # Enhanced strength classification with more granular levels
        abs_corr = abs(corr_value)
        if abs_corr >= 0.95:
            strength = "Nearly Perfect"
            emoji = "🔥"
            interpretation = "Variables are almost perfectly related"
        elif abs_corr >= 0.8:
            strength = "Very Strong"
            emoji = "💪"
            interpretation = "Strong linear relationship - high predictive power"
        elif abs_corr >= 0.6:
            strength = "Strong"
            emoji = "📊"
            interpretation = "Substantial relationship - good for modeling"
        elif abs_corr >= 0.4:
            strength = "Moderate"
            emoji = "📈"
            interpretation = "Noticeable relationship - worth investigating"
        elif abs_corr >= 0.2:
            strength = "Weak"
            emoji = "📉"
            interpretation = "Slight relationship - limited practical significance"
        elif abs_corr >= 0.1:
            strength = "Very Weak"
            emoji = "⚪"
            interpretation = "Minimal relationship - likely not meaningful"
        else:
            strength = "Negligible"
            emoji = "❌"
            interpretation = "No meaningful linear relationship"
        
        direction = "Positive" if corr_value > 0 else "Negative"
        
        # Main correlation insight
        insights.append(f"{emoji} **{strength} {direction} Correlation** (r = {corr_value:.3f})")
        insights.append(f"📋 **Interpretation**: {interpretation}")
        
        # Practical implications based on strength
        if abs_corr >= 0.9:
            insights.append(f"🚨 **Multicollinearity Alert**: Extremely high correlation - consider removing one variable")
        elif abs_corr >= 0.7:
            insights.append(f"⚠️ **Multicollinearity Risk**: High correlation - check for redundancy in modeling")
        elif abs_corr >= 0.5:
            insights.append(f"🔍 **Feature Engineering**: Consider creating interaction terms or combined features")
        elif abs_corr >= 0.3:
            insights.append(f"📊 **Modeling Potential**: Moderate correlation - useful for predictive modeling")
        
        # Statistical significance context (rough estimate)
        if abs_corr >= 0.5:
            insights.append(f"📈 **Statistical Note**: Likely statistically significant in most sample sizes")
        elif abs_corr >= 0.3:
            insights.append(f"📊 **Statistical Note**: May be significant with adequate sample size (n>100)")
        else:
            insights.append(f"📉 **Statistical Note**: Likely not significant unless very large sample")
        
        # Coefficient of determination (R²)
        r_squared = corr_value ** 2
        variance_explained = r_squared * 100
        if variance_explained >= 25:
            insights.append(f"🎯 **Variance Explained**: {variance_explained:.1f}% of variance in one variable explained by the other")
        
        return insights
    
    @staticmethod
    def analyze_categorical(data, column_name):
        """Enhanced categorical data analysis"""
        value_counts = data.value_counts()
        total_count = len(data)
        unique_count = data.nunique()
        
        insights = []
        
        # Enhanced cardinality analysis
        if unique_count == 1:
            insights.append(f"⚠️ **Constant Variable**: {column_name} has only one unique value - consider removal")
        elif unique_count == 2:
            insights.append(f"🔄 **Binary Variable**: {column_name} is binary - perfect for encoding")
        elif unique_count <= 5:
            insights.append(f"🏷️ **Low Cardinality**: {unique_count} categories - excellent for one-hot encoding")
        elif unique_count <= 10:
            insights.append(f"📊 **Low-Medium Cardinality**: {unique_count} categories - good for dummy encoding")
        elif unique_count <= 20:
            insights.append(f"📊 **Medium Cardinality**: {unique_count} categories - consider target/ordinal encoding")
        elif unique_count <= 50:
            insights.append(f"🌊 **High Cardinality**: {unique_count} categories - use advanced encoding (target/hash)")
        else:
            insights.append(f"🌊 **Very High Cardinality**: {unique_count} categories - likely needs dimensionality reduction")
        
        # Enhanced balance analysis using multiple metrics
        balance_metrics = VizInsights._refined_categorical_balance(data)
        
        # Gini-based balance assessment (more accurate than simple percentage)
        gini = balance_metrics['gini_coefficient']
        entropy = balance_metrics['normalized_entropy']
        top_pct = balance_metrics['top_category_pct']
        effective_cats = balance_metrics['effective_categories']
        
        if gini < 0.2 and entropy > 0.8:
            insights.append(f"✅ **Well Balanced**: High diversity (Gini={gini:.2f}, Entropy={entropy:.2f})")
        elif gini < 0.4 and entropy > 0.6:
            insights.append(f"📊 **Moderately Balanced**: Good diversity (Gini={gini:.2f}, {effective_cats:.1f} effective categories)")
        elif gini < 0.6:
            insights.append(f"⚠️ **Moderately Imbalanced**: Dominant category {top_pct:.1f}% (Gini={gini:.2f})")
        elif gini < 0.8:
            insights.append(f"🚨 **Highly Imbalanced**: Very dominant category {top_pct:.1f}% (Gini={gini:.2f})")
        else:
            insights.append(f"🚨 **Extremely Imbalanced**: Single category dominates {top_pct:.1f}% - consider resampling")
        
        # Rare category detection
        rare_threshold = 0.01  # 1%
        rare_categories = value_counts[value_counts / total_count < rare_threshold]
        if len(rare_categories) > 0:
            rare_pct = (rare_categories.sum() / total_count) * 100
            insights.append(f"🔍 **Rare Categories**: {len(rare_categories)} categories with <1% frequency ({rare_pct:.1f}% total)")
        
        # Missing value context
        missing_count = data.isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / (total_count + missing_count)) * 100
            if missing_pct > 20:
                insights.append(f"⚠️ **High Missing Rate**: {missing_pct:.1f}% missing values - investigate data quality")
            elif missing_pct > 5:
                insights.append(f"📊 **Moderate Missing Rate**: {missing_pct:.1f}% missing values - consider imputation")
        
        return insights


class SmartVizEngine:
    """Next-Generation Visualization Engine with AI Insights"""
    
    def __init__(self):
        self.insights = VizInsights()
        self.plot_count = 0
        
    def _create_insight_panel(self, insights, title):
        """Create a beautiful insight panel using Rich"""
        insight_text = Text()
        for i, insight in enumerate(insights):
            if i > 0:
                insight_text.append("\n")
            insight_text.append(insight, style="cyan")
        
        return Panel(
            insight_text,
            title=f"🧠 {title}",
            border_style="blue",
            padding=(1, 2)
        )
    
    def _create_statistical_summary(self, data, column_name):
        """Create detailed statistical summary panel"""
        
        # Calculate comprehensive statistics
        stats_data = {
            "Count": f"{len(data):,}",
            "Mean": f"{data.mean():.2f}",
            "Median": f"{data.median():.2f}",
            "Std Dev": f"{data.std():.2f}",
            "Min": f"{data.min():.2f}",
            "Max": f"{data.max():.2f}",
            "Skewness": f"{stats.skew(data):.3f}",
            "Kurtosis": f"{stats.kurtosis(data):.3f}"
        }
        
        # Create statistics table
        stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="bold green")
        
        for stat, value in stats_data.items():
            stats_table.add_row(stat, value)
        
        # Detect outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(data)) * 100
        
        stats_table.add_row("Outliers", f"{outlier_count} ({outlier_pct:.1f}%)")
        
        return Panel(
            stats_table,
            title=f"📊 Statistical Summary: {column_name}",
            border_style="magenta"
        )
    
    def _create_categorical_summary(self, data, column_name):
        """Create detailed categorical summary panel"""
        
        value_counts = data.value_counts()
        total_count = len(data)
        unique_count = data.nunique()
        missing_count = data.isnull().sum()
        
        # Create summary table
        summary_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", style="bold yellow")
        summary_table.add_column("Percentage", style="bold green")
        
        # Add top categories
        for category, count in value_counts.head(10).items():
            percentage = (count / total_count) * 100
            summary_table.add_row(
                str(category), 
                f"{count:,}", 
                f"{percentage:.1f}%"
            )
        
        if len(value_counts) > 10:
            others_count = value_counts.tail(-10).sum()
            others_pct = (others_count / total_count) * 100
            summary_table.add_row(
                f"Others ({len(value_counts) - 10} categories)",
                f"{others_count:,}",
                f"{others_pct:.1f}%"
            )
        
        # Add summary statistics
        summary_table.add_row("─" * 20, "─" * 10, "─" * 10)
        summary_table.add_row("TOTAL UNIQUE", f"{unique_count:,}", "100.0%")
        if missing_count > 0:
            missing_pct = (missing_count / (total_count + missing_count)) * 100
            summary_table.add_row("MISSING VALUES", f"{missing_count:,}", f"{missing_pct:.1f}%")
        
        return Panel(
            summary_table,
            title=f"📊 Category Breakdown: {column_name}",
            border_style="green"
        )
    
    def _auto_select_variables(self, df, max_vars=8):
        """Intelligently select variables for visualization"""
        console.print("\n🤖 [bold cyan]AI Variable Selection in Progress...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing data patterns...", total=None)
            time.sleep(1)  # Simulate analysis
        
        # CRITICAL: Clean up stream after rich.progress to fix Colab rendering
        # This clear_output runs ONCE here (not per-chart) so it doesn't destroy charts
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            console.file.flush()
        except:
            pass
        
        if _ENVIRONMENT == 'colab':
            try:
                from IPython.display import clear_output
                clear_output(wait=True)  # wait=True preserves queued output
                time.sleep(0.1)  # Let stream fully reset
            except:
                pass
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        selected = {
            'numeric': [],
            'categorical': [],
            'correlations': []
        }
        
        # Select top numeric variables by variance and non-null ratio
        if numeric_cols:
            numeric_scores = {}
            for col in numeric_cols:
                variance_score = df[col].var() if df[col].var() > 0 else 0
                completeness_score = (1 - df[col].isnull().sum() / len(df))
                numeric_scores[col] = variance_score * completeness_score
            
            selected['numeric'] = sorted(numeric_scores.items(), 
                                       key=lambda x: x[1], reverse=True)[:max_vars//2]
            selected['numeric'] = [col for col, _ in selected['numeric']]
        
        # Select categorical variables by cardinality and completeness
        if categorical_cols:
            cat_scores = {}
            for col in categorical_cols:
                cardinality = df[col].nunique()
                if cardinality > 50:
                    continue  # Skip high-cardinality columns (title, show_id, etc.)
                completeness = (1 - df[col].isnull().sum() / len(df))
                # Prefer medium cardinality (2-20 categories)
                cardinality_score = 1 / (1 + abs(cardinality - 10)) if cardinality <= 50 else 0.1
                cat_scores[col] = cardinality_score * completeness
            
            selected['categorical'] = sorted(cat_scores.items(), 
                                           key=lambda x: x[1], reverse=True)[:max_vars//2]
            selected['categorical'] = [col for col, _ in selected['categorical']]
        
        # Find interesting correlations
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= 0.5:  # Moderate to strong correlation
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_val
                        ))
            
            selected['correlations'] = sorted(high_corr_pairs, 
                                            key=lambda x: abs(x[2]), reverse=True)[:5]
        
        return selected
    
    def _plot_distribution(self, df, column, interactive=True):
        """Create enhanced distribution plot with insights"""
        data = df[column].dropna()
        
        # Generate insights FIRST
        insights = self.insights.analyze_distribution(data, column)
        
        # Display insights panel BEFORE the chart
        console.print(f"\n🔍 [bold cyan]Analyzing {column}...[/bold cyan]")
        console.print(self._create_insight_panel(insights, f"Distribution Insights: {column}"))
        
        if interactive:
            # Plotly interactive histogram
            fig = px.histogram(
                df, x=column, 
                title=f"📊 Distribution Analysis: {column}",
                marginal="box",
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(
                template="plotly_white",
                title_font_size=16,
                showlegend=False
            )
            _display_plotly_figure(fig)
        else:
            # Matplotlib version
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with KDE
            sns.histplot(data, kde=True, ax=ax1, color='#FF6B6B', alpha=0.7)
            ax1.set_title(f'Distribution: {column}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            sns.boxplot(y=data, ax=ax2, color='#4ECDC4')
            ax2.set_title(f'Box Plot: {column}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Display additional statistical insights AFTER the chart
        console.print(self._create_statistical_summary(data, column))
        console.print("\n" + "─" * 80 + "\n")
        
        self.plot_count += 1
    
    def _plot_correlation_heatmap(self, df, columns):
        """Create enhanced correlation heatmap"""
        console.print(f"\n🔥 [bold cyan]Analyzing Correlations...[/bold cyan]")
        
        corr_data = df[columns].corr()
        
        # Plotly interactive heatmap
        fig = px.imshow(
            corr_data,
            title="🔥 Correlation Matrix - Interactive Heatmap",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(
            title_font_size=16,
            width=800,
            height=600
        )
        _display_plotly_figure(fig)
        
        # Find and display top correlations
        high_corr = []
        for i in range(len(corr_data.columns)):
            for j in range(i+1, len(corr_data.columns)):
                corr_val = corr_data.iloc[i, j]
                if abs(corr_val) >= 0.3:
                    high_corr.append((corr_data.columns[i], corr_data.columns[j], corr_val))
        
        if high_corr:
            high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Create correlation insights table
            corr_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
            corr_table.add_column("Variable 1", style="cyan")
            corr_table.add_column("Variable 2", style="cyan") 
            corr_table.add_column("Correlation", style="bold green")
            corr_table.add_column("Strength", style="yellow")
            corr_table.add_column("Interpretation", style="magenta")
            
            for var1, var2, corr_val in high_corr[:8]:  # Show top 8 correlations
                if abs(corr_val) >= 0.8:
                    strength = "Very Strong"
                    interpretation = "Consider multicollinearity"
                elif abs(corr_val) >= 0.6:
                    strength = "Strong"
                    interpretation = "Significant relationship"
                elif abs(corr_val) >= 0.4:
                    strength = "Moderate"
                    interpretation = "Notable association"
                else:
                    strength = "Weak"
                    interpretation = "Slight relationship"
                
                corr_table.add_row(
                    var1, var2, f"{corr_val:.3f}", 
                    strength, interpretation
                )
            
            console.print(Panel(
                corr_table,
                title="🔍 Correlation Analysis Results",
                border_style="blue"
            ))
            
            # Generate detailed insights for top correlations
            for var1, var2, corr_val in high_corr[:3]:
                insights = self.insights.analyze_correlation(corr_val, var1, var2)
                console.print(self._create_insight_panel(insights, f"Correlation: {var1} ↔ {var2}"))
        else:
            console.print(Panel(
                "No significant correlations found (|r| ≥ 0.3)",
                title="🔍 Correlation Analysis Results",
                border_style="yellow"
            ))
        
        console.print("\n" + "─" * 80 + "\n")
        self.plot_count += 1
    
    def _plot_categorical(self, df, column, target=None):
        """Create enhanced categorical visualization"""
        console.print(f"\n🏷️ [bold cyan]Analyzing Categorical Variable: {column}...[/bold cyan]")
        
        data = df[column].value_counts().head(15)  # Top 15 categories
        
        # Generate insights FIRST
        insights = self.insights.analyze_categorical(df[column], column)
        console.print(self._create_insight_panel(insights, f"Categorical Insights: {column}"))
        
        if target and target in df.columns:
            # Grouped bar chart
            fig = px.histogram(
                df, x=column, color=target,
                title=f"📊 {column} Distribution by {target}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        else:
            # Simple bar chart
            fig = px.bar(
                x=data.index, y=data.values,
                title=f"📊 Category Distribution: {column}",
                color=data.values,
                color_continuous_scale='viridis'
            )
        
        fig.update_layout(
            template="plotly_white",
            title_font_size=16,
            xaxis_tickangle=-45
        )
        _display_plotly_figure(fig)
        
        # Create detailed categorical summary
        cat_summary = self._create_categorical_summary(df[column], column)
        console.print(cat_summary)
        console.print("\n" + "─" * 80 + "\n")
        
        self.plot_count += 1
    
    def _plot_scatter_matrix(self, df, columns):
        """Create interactive scatter plot matrix"""
        console.print(f"\n🎯 [bold cyan]Creating Multi-Variable Relationship Matrix...[/bold cyan]")
        
        if len(columns) > 6:
            columns = columns[:6]  # Limit for readability
            console.print(f"📊 [yellow]Limited to first 6 variables for readability[/yellow]")
        
        fig = px.scatter_matrix(
            df[columns],
            title="🎯 Multi-Variable Relationship Matrix",
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            title_font_size=16,
            width=1000,
            height=800
        )
        _display_plotly_figure(fig)
        
        # Analyze relationships between variables
        relationship_insights = []
        corr_matrix = df[columns].corr()
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr_val = corr_matrix.iloc[i, j]
                var1, var2 = columns[i], columns[j]
                
                if abs(corr_val) >= 0.5:
                    if corr_val > 0:
                        relationship_insights.append(f"🔗 **{var1}** and **{var2}** show strong positive relationship (r={corr_val:.3f})")
                    else:
                        relationship_insights.append(f"🔗 **{var1}** and **{var2}** show strong negative relationship (r={corr_val:.3f})")
                elif abs(corr_val) >= 0.3:
                    relationship_insights.append(f"📊 **{var1}** and **{var2}** show moderate correlation (r={corr_val:.3f})")
        
        if not relationship_insights:
            relationship_insights.append("📊 No strong linear relationships detected between variables")
        
        # Create insights panel
        insights_text = Text()
        insights_text.append("🔍 **Pattern Recognition Guide**\n\n", style="bold cyan")
        insights_text.append("• Look for linear patterns indicating strong relationships\n", style="cyan")
        insights_text.append("• Curved patterns suggest non-linear relationships\n", style="cyan")
        insights_text.append("• Clustered points indicate potential groupings\n", style="cyan")
        insights_text.append("• Outliers appear as isolated points\n\n", style="cyan")
        
        insights_text.append("🔗 **Detected Relationships**\n\n", style="bold yellow")
        for insight in relationship_insights[:5]:  # Show top 5
            insights_text.append(f"{insight}\n", style="yellow")
        
        console.print(Panel(
            insights_text,
            title="🧠 Scatter Matrix Analysis",
            border_style="green"
        ))
        
        console.print("\n" + "─" * 80 + "\n")
        self.plot_count += 1
    
    def _create_basic_visualizations(self, df, selected_vars, target):
        """Create basic 2D visualizations"""
        # 1. Distributions
        if selected_vars['numeric']:
            console.print("📊 [bold]Generating Distribution Plots...[/bold]")
            for col in selected_vars['numeric'][:6]:
                self._plot_distribution(df, col, interactive=True)
        
        # 2. Correlation
        if len(selected_vars['numeric']) > 1:
            console.print("\n🔥 [bold]Generating Correlation Analysis...[/bold]")
            self._plot_correlation_heatmap(df, selected_vars['numeric'])
        
        # 3. Categorical
        if selected_vars['categorical']:
            console.print("\n🏷️ [bold]Generating Categorical Analysis...[/bold]")
            for col in selected_vars['categorical'][:4]:
                self._plot_categorical(df, col, target)
        
        # 4. Scatter matrix
        if len(selected_vars['numeric']) >= 3:
            console.print("\n🎯 [bold]Generating Multi-Variable Analysis...[/bold]")
            self._plot_scatter_matrix(df, selected_vars['numeric'][:6])
    
    def _create_advanced_visualizations(self, df, selected_vars, dark_theme, target):
        """Create advanced 3D professional visualizations — always aims for 5 charts"""
        console.print("🎨 [bold magenta]Creating Advanced 3D Visualizations...[/bold magenta]\n")
        
        numeric_cols = selected_vars['numeric']
        categorical_cols = selected_vars['categorical']
        charts_created = 0
        
        # 1. 3D SURFACE DENSITY PLOT (works with ≥1 numeric)
        if len(numeric_cols) >= 1:
            try:
                self._create_3d_surface_density(df, numeric_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ 3D Surface skipped: {e}[/yellow]\n")
        
        # 2. INTERACTIVE TREEMAP (works with ≥1 categorical)
        if len(categorical_cols) >= 1:
            try:
                self._create_treemap_chart(df, categorical_cols, numeric_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Treemap skipped: {e}[/yellow]\n")
        
        # 3. VIOLIN BOX COMPARISON (works with ≥1 numeric + ≥1 categorical)
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            try:
                self._create_violin_box_chart(df, numeric_cols, categorical_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Violin chart skipped: {e}[/yellow]\n")
        
        # 4. DONUT PIE BREAKDOWN (works with ≥1 categorical)
        if len(categorical_cols) >= 1:
            try:
                self._create_donut_pie_chart(df, categorical_cols, numeric_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Donut chart skipped: {e}[/yellow]\n")
        
        # 5. RIDGELINE 3D DISTRIBUTION (works with ≥1 numeric + ≥1 categorical)
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            try:
                self._create_ridgeline_3d(df, numeric_cols, categorical_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Ridgeline chart skipped: {e}[/yellow]\n")
        
        # 6. 3D RADAR / SPIDER CHART (works with ≥3 numeric — bonus chart)
        if len(numeric_cols) >= 3:
            try:
                self._create_3d_radar_chart(df, numeric_cols, categorical_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Radar chart skipped: {e}[/yellow]\n")
        
        # 7. PARALLEL COORDINATES (works with ≥2 numeric — bonus chart)
        if len(numeric_cols) >= 2:
            try:
                self._create_parallel_coordinates(df, numeric_cols, categorical_cols, dark_theme)
                charts_created += 1
            except Exception as e:
                console.print(f"[yellow]⚠️ Parallel coordinates skipped: {e}[/yellow]\n")
        
        # FALLBACK: If very few charts, add distribution
        if charts_created < 3 and len(numeric_cols) >= 1:
            for col in numeric_cols[:max(1, 3 - charts_created)]:
                try:
                    self._create_distribution_pro(df, col, dark_theme)
                    charts_created += 1
                except Exception:
                    pass
    
    # ─── CHART 1: 3D SURFACE DENSITY ───────────────────────────────────────────
    
    def _create_3d_surface_density(self, df, numeric_cols, dark_theme):
        """Create a 3D surface density landscape using KDE"""
        console.print("\n🌊 [bold cyan]1. 3D Density Surface — Data Landscape[/bold cyan]")
        
        # Pick best 2 numeric columns (or use column + its index)
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            x_data = df[x_col].dropna().values.astype(float)
            y_data = df[y_col].dropna().values.astype(float)
            # Align lengths
            min_len = min(len(x_data), len(y_data))
            x_data, y_data = x_data[:min_len], y_data[:min_len]
            x_label, y_label = x_col, y_col
        else:
            x_col = numeric_cols[0]
            x_data = df[x_col].dropna().values.astype(float)
            # Create synthetic y-axis from row index
            y_data = np.arange(len(x_data)).astype(float)
            x_label, y_label = x_col, "Index"
        
        # Sample for performance
        if len(x_data) > 5000:
            idx = np.random.choice(len(x_data), 5000, replace=False)
            x_data, y_data = x_data[idx], y_data[idx]
        
        # Build 2D KDE surface
        try:
            from scipy.stats import gaussian_kde
            xy_stack = np.vstack([x_data, y_data])
            kde = gaussian_kde(xy_stack, bw_method='scott')
            
            grid_size = 60
            x_grid = np.linspace(x_data.min(), x_data.max(), grid_size)
            y_grid = np.linspace(y_data.min(), y_data.max(), grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
        except Exception:
            # Fallback: histogram-based surface
            grid_size = 40
            Z, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=grid_size)
            x_grid = (x_edges[:-1] + x_edges[1:]) / 2
            y_grid = (y_edges[:-1] + y_edges[1:]) / 2
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = Z.T  # Align axes
        
        # Neon color scales for dark theme
        colorscale = [
            [0.0, '#0d0221'], [0.1, '#0a0440'], [0.2, '#150050'],
            [0.3, '#3f0071'], [0.4, '#6100a1'], [0.5, '#a200d1'],
            [0.6, '#d000ff'], [0.7, '#ff00e4'], [0.8, '#ff6bf5'],
            [0.9, '#ffa6f6'], [1.0, '#ffe0fc']
        ] if dark_theme else 'Viridis'
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Density",
                titlefont=dict(color='white' if dark_theme else 'black'),
                tickfont=dict(color='white' if dark_theme else 'black')
            ),
            lighting=dict(
                ambient=0.4, diffuse=0.6, specular=0.3,
                roughness=0.5, fresnel=0.2
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True, highlightcolor='#ff00e4' if dark_theme else '#636EFA')
            ),
            opacity=0.95
        )])
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        gridcolor = '#1a0a3e' if dark_theme else '#e0e0e0'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        
        fig.update_layout(
            title=dict(text=f"🌊 3D Density Surface: {x_label} × {y_label}",
                      font=dict(size=20, color=textcolor, family='Arial Black')),
            scene=dict(
                xaxis=dict(title=x_label, backgroundcolor=bgcolor, gridcolor=gridcolor,
                          titlefont=dict(color=textcolor), tickfont=dict(color=textcolor)),
                yaxis=dict(title=y_label, backgroundcolor=bgcolor, gridcolor=gridcolor,
                          titlefont=dict(color=textcolor), tickfont=dict(color=textcolor)),
                zaxis=dict(title="Density", backgroundcolor=bgcolor, gridcolor=gridcolor,
                          titlefont=dict(color=textcolor), tickfont=dict(color=textcolor)),
                bgcolor=bgcolor,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            font=dict(color=textcolor),
            width=1000, height=750
        )
        
        _display_plotly_figure(fig)
        console.print("✅ [green]3D Density Surface created![/green]\n")
        self.plot_count += 1

    # ─── CHART 2: TREEMAP (replaces buggy sunburst) ────────────────────────────

    def _create_treemap_chart(self, df, categorical_cols, numeric_cols, dark_theme):
        """Create an interactive treemap hierarchy chart"""
        console.print("\n🌳 [bold cyan]2. Interactive Treemap — Hierarchical Distribution[/bold cyan]")
        
        # Filter to low-cardinality categoricals (≤ 30 unique) for readable charts
        usable_cats = [c for c in categorical_cols if df[c].nunique() <= 30]
        if not usable_cats:
            # Fallback: bin the top categorical by top-10 values
            col = categorical_cols[0]
            top_vals = df[col].value_counts().head(10).index.tolist()
            df = df.copy()
            df[col] = df[col].where(df[col].isin(top_vals), other='Other')
            usable_cats = [col]
        
        # Build path: use up to 2 categorical columns
        path_cols = usable_cats[:2]
        
        # Aggregate: count rows per group (safe — no negative values issue)
        if len(path_cols) == 2:
            agg_df = df.groupby(path_cols).size().reset_index(name='count')
        else:
            agg_df = df.groupby(path_cols).size().reset_index(name='count')
        
        # Color by count or a numeric column
        if numeric_cols:
            # Compute mean of first numeric col per group
            mean_df = df.groupby(path_cols)[numeric_cols[0]].mean().reset_index()
            mean_df.columns = list(path_cols) + ['color_val']
            agg_df = agg_df.merge(mean_df, on=list(path_cols), how='left')
            color_col_name = 'color_val'
            color_label = f"Avg {numeric_cols[0]}"
        else:
            agg_df['color_val'] = agg_df['count']
            color_col_name = 'color_val'
            color_label = "Count"
        
        # Neon treemap palette
        neon_colors = [
            '#ff006e', '#8338ec', '#3a86ff', '#00f5d4', '#fee440',
            '#fb5607', '#ff006e', '#7209b7', '#560bad', '#480ca8'
        ] if dark_theme else px.colors.qualitative.Set3
        
        fig = px.treemap(
            agg_df,
            path=[px.Constant("All")] + path_cols,
            values='count',
            color=color_col_name,
            color_continuous_scale='Plasma' if dark_theme else 'Viridis',
            title=f"Hierarchical Distribution: {' → '.join(path_cols)}",
            hover_data={'count': ':,'}
        )
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        
        fig.update_layout(
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            font=dict(color=textcolor, size=12, family='Arial'),
            title=dict(font=dict(size=20, color=textcolor, family='Arial Black')),
            coloraxis_colorbar=dict(
                title=color_label,
                titlefont=dict(color=textcolor),
                tickfont=dict(color=textcolor)
            ),
            width=1000, height=750,
            margin=dict(t=60, l=10, r=10, b=10)
        )
        fig.update_traces(
            textinfo="label+value+percent parent",
            marker=dict(cornerradius=5)
        )
        
        _display_plotly_figure(fig)
        
        # Summary insights
        insights_text = Text()
        for col in path_cols:
            unique_count = df[col].nunique()
            top_cat = df[col].value_counts().index[0]
            top_pct = (df[col].value_counts().iloc[0] / len(df)) * 100
            insights_text.append(f"📊 {col}: {unique_count} categories, top = '{top_cat}' ({top_pct:.1f}%)\n", style="cyan")
        insights_text.append(f"📈 Total records: {len(df):,}\n", style="green")
        
        console.print(Panel(insights_text, title="🔍 Treemap Insights", border_style="cyan"))
        console.print("✅ [green]Interactive Treemap created![/green]\n")
        self.plot_count += 1

    # ─── CHART 3: 3D RADAR / SPIDER ───────────────────────────────────────────

    def _create_3d_radar_chart(self, df, numeric_cols, categorical_cols, dark_theme):
        """Create a multi-dimensional radar/spider chart comparing groups"""
        console.print("\n🕸️ [bold cyan]3. 3D Radar Chart — Multi-Dimensional Profiles[/bold cyan]")
        
        # Pick up to 8 numeric dimensions
        radar_cols = numeric_cols[:8]
        
        # Normalize all columns to 0-1 for fair comparison
        normalized = pd.DataFrame()
        for col in radar_cols:
            col_data = df[col].dropna()
            col_min, col_max = col_data.min(), col_data.max()
            if col_max - col_min > 0:
                normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = 0.5
        
        # If we have a categorical column, compare group profiles
        group_col = None
        if categorical_cols:
            # Find a categorical with 2-6 groups
            for cat in categorical_cols:
                if 2 <= df[cat].nunique() <= 6:
                    group_col = cat
                    break
            if not group_col:
                # Use top 4 values of first categorical
                group_col = categorical_cols[0]
                top_vals = df[group_col].value_counts().head(4).index.tolist()
                normalized[group_col] = df[group_col].where(df[group_col].isin(top_vals), other=None)
            else:
                normalized[group_col] = df[group_col]
        
        # Neon colors for radar traces
        neon_palette = ['#ff006e', '#00f5d4', '#3a86ff', '#fee440', '#fb5607', '#8338ec']
        light_palette = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261', '#264653']
        colors = neon_palette if dark_theme else light_palette
        
        fig = go.Figure()
        
        if group_col and group_col in normalized.columns:
            groups = normalized[group_col].dropna().unique()[:6]
            for i, group in enumerate(groups):
                group_data = normalized[normalized[group_col] == group][radar_cols].mean()
                values = group_data.values.tolist()
                values.append(values[0])  # Close the polygon
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_cols + [radar_cols[0]],
                    name=str(group),
                    fill='toself',
                    fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)}, {int(colors[i % len(colors)][3:5], 16)}, {int(colors[i % len(colors)][5:7], 16)}, 0.15)",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    marker=dict(size=6, color=colors[i % len(colors)])
                ))
        else:
            # Overall profile: mean, median, std bands
            mean_vals = normalized[radar_cols].mean().values.tolist()
            mean_vals.append(mean_vals[0])
            median_vals = normalized[radar_cols].median().values.tolist()
            median_vals.append(median_vals[0])
            
            fig.add_trace(go.Scatterpolar(
                r=mean_vals, theta=radar_cols + [radar_cols[0]],
                name='Mean Profile', fill='toself',
                fillcolor='rgba(255, 0, 110, 0.15)' if dark_theme else 'rgba(99, 110, 250, 0.15)',
                line=dict(color='#ff006e' if dark_theme else '#636EFA', width=2.5),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatterpolar(
                r=median_vals, theta=radar_cols + [radar_cols[0]],
                name='Median Profile', fill='toself',
                fillcolor='rgba(0, 245, 212, 0.1)' if dark_theme else 'rgba(0, 204, 150, 0.1)',
                line=dict(color='#00f5d4' if dark_theme else '#00CC96', width=2, dash='dash'),
                marker=dict(size=5)
            ))
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        gridcolor = '#2a1052' if dark_theme else '#e0e0e0'
        
        fig.update_layout(
            polar=dict(
                bgcolor=bgcolor,
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor=gridcolor, linecolor=gridcolor,
                    tickfont=dict(color=textcolor, size=9)
                ),
                angularaxis=dict(
                    gridcolor=gridcolor, linecolor=gridcolor,
                    tickfont=dict(color=textcolor, size=11)
                )
            ),
            title=dict(
                text=f"🕸️ Multi-Dimensional Radar: {len(radar_cols)} Features" +
                     (f" by {group_col}" if group_col else ""),
                font=dict(size=20, color=textcolor, family='Arial Black')
            ),
            paper_bgcolor=bgcolor,
            font=dict(color=textcolor),
            legend=dict(font=dict(color=textcolor, size=11)),
            width=900, height=750
        )
        
        _display_plotly_figure(fig)
        
        # Feature importance ranking
        if group_col and group_col in normalized.columns:
            groups_data = normalized.dropna(subset=[group_col])
            variance_by_feature = {}
            for col in radar_cols:
                group_means = groups_data.groupby(group_col)[col].mean()
                variance_by_feature[col] = group_means.var()
            
            sorted_features = sorted(variance_by_feature.items(), key=lambda x: x[1], reverse=True)
            feat_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            feat_table.add_column("Feature", style="cyan")
            feat_table.add_column("Discriminative Power", style="bold green")
            for feat, var in sorted_features[:5]:
                bar = "█" * int(var / max(v for _, v in sorted_features) * 20) if max(v for _, v in sorted_features) > 0 else ""
                feat_table.add_row(feat, f"{var:.4f}  {bar}")
            console.print(Panel(feat_table, title="🎯 Feature Discriminative Power", border_style="magenta"))
        
        console.print("✅ [green]Radar Chart created![/green]\n")
        self.plot_count += 1

    # ─── CHART 4: PARALLEL COORDINATES ─────────────────────────────────────────

    def _create_parallel_coordinates(self, df, numeric_cols, categorical_cols, dark_theme):
        """Create interactive parallel coordinates plot"""
        console.print("\n🔀 [bold cyan]4. Parallel Coordinates — Multi-Axis Pattern Explorer[/bold cyan]")
        
        # Pick up to 7 numeric columns
        par_cols = numeric_cols[:7]
        
        # Decide color dimension
        color_data = None
        color_label = ""
        colorscale = 'Plasma' if dark_theme else 'Viridis'
        
        if categorical_cols:
            # Encode best categorical as color
            best_cat = None
            for cat in categorical_cols:
                if 2 <= df[cat].nunique() <= 10:
                    best_cat = cat
                    break
            if not best_cat:
                best_cat = categorical_cols[0]
            
            le = LabelEncoder()
            valid_mask = df[best_cat].notna()
            encoded = pd.Series(np.nan, index=df.index)
            encoded[valid_mask] = le.fit_transform(df[best_cat][valid_mask].astype(str))
            color_data = encoded
            color_label = best_cat
            
            # Use a discrete-ish neon scale
            n_cats = int(encoded.max()) + 1 if encoded.notna().any() else 1
            if dark_theme:
                neon = ['#ff006e', '#00f5d4', '#3a86ff', '#fee440', '#fb5607', '#8338ec', '#7209b7', '#560bad', '#f72585', '#4cc9f0']
                colorscale = [[i / max(n_cats - 1, 1), neon[i % len(neon)]] for i in range(n_cats)]
            else:
                colorscale = 'Turbo'
        else:
            color_data = df[par_cols[0]]
            color_label = par_cols[0]
        
        # Build dimensions
        dimensions = []
        for col in par_cols:
            col_data = df[col].dropna()
            dimensions.append(dict(
                range=[col_data.min(), col_data.max()],
                label=col,
                values=df[col]
            ))
        
        # Sample for performance
        sample_size = min(len(df), 5000)
        if len(df) > sample_size:
            sample_idx = np.random.choice(len(df), sample_size, replace=False)
            sampled_dims = []
            for d in dimensions:
                sampled_dims.append(dict(
                    range=d['range'], label=d['label'],
                    values=d['values'].iloc[sample_idx]
                ))
            dimensions = sampled_dims
            color_sampled = color_data.iloc[sample_idx] if color_data is not None else None
        else:
            color_sampled = color_data
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        line_color_bg = '#1a0a3e' if dark_theme else '#f0f0f0'
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=color_sampled,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=color_label,
                    titlefont=dict(color=textcolor),
                    tickfont=dict(color=textcolor)
                )
            ),
            dimensions=dimensions,
            labelfont=dict(color=textcolor, size=12),
            tickfont=dict(color=textcolor, size=10),
            rangefont=dict(color=textcolor, size=10)
        ))
        
        fig.update_layout(
            title=dict(text=f"🔀 Parallel Coordinates: {len(par_cols)} Dimensions" +
                       (f" colored by {color_label}" if color_label else ""),
                      font=dict(size=20, color=textcolor, family='Arial Black')),
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            font=dict(color=textcolor),
            width=1100, height=650,
            margin=dict(l=80, r=80, t=80, b=40)
        )
        
        _display_plotly_figure(fig)
        
        # Insights
        insights_text = Text()
        insights_text.append(f"📊 Dimensions: {len(par_cols)} numeric features\n", style="cyan")
        if color_label:
            insights_text.append(f"🎨 Color: {color_label}\n", style="cyan")
        insights_text.append("💡 Tip: Drag axes to reorder. Click+drag on an axis to filter ranges.\n", style="green")
        console.print(Panel(insights_text, title="🔍 Parallel Coordinates Guide", border_style="cyan"))
        console.print("✅ [green]Parallel Coordinates created![/green]\n")
        self.plot_count += 1

    # ─── CHART 3: VIOLIN BOX COMPARISON ────────────────────────────────────────

    def _create_violin_box_chart(self, df, numeric_cols, categorical_cols, dark_theme):
        """Create advanced violin + box plot comparison across categories"""
        console.print("\n🎻 [bold cyan]3. Violin Box Comparison — Distribution by Category[/bold cyan]")
        
        num_col = numeric_cols[0]
        
        # Pick best categorical (2-10 groups)
        cat_col = None
        for cat in categorical_cols:
            n = df[cat].nunique()
            if 2 <= n <= 10:
                cat_col = cat
                break
        if not cat_col:
            cat_col = categorical_cols[0]
            top_vals = df[cat_col].value_counts().head(8).index.tolist()
            df = df.copy()
            df[cat_col] = df[cat_col].where(df[cat_col].isin(top_vals), other=None)
            df = df.dropna(subset=[cat_col])
        
        # Neon palette
        neon = ['#ff006e', '#00f5d4', '#3a86ff', '#fee440', '#fb5607',
                '#8338ec', '#f72585', '#4cc9f0', '#7209b7', '#06d6a0']
        light = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261',
                 '#264653', '#a8dadc', '#fca311', '#6d6875', '#b5838d']
        palette = neon if dark_theme else light
        
        categories = df[cat_col].value_counts().head(10).index.tolist()
        
        fig = go.Figure()
        for i, cat in enumerate(categories):
            cat_data = df[df[cat_col] == cat][num_col].dropna()
            if len(cat_data) < 3:
                continue
            color = palette[i % len(palette)]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            
            fig.add_trace(go.Violin(
                y=cat_data,
                name=str(cat)[:15],
                box_visible=True,
                meanline_visible=True,
                fillcolor=f'rgba({r}, {g}, {b}, 0.3)',
                line_color=color,
                marker=dict(color=color, size=3, opacity=0.3),
                points='outliers',
                scalemode='width',
                width=0.8
            ))
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        gridcolor = '#1a0a3e' if dark_theme else '#e0e0e0'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        
        fig.update_layout(
            title=dict(text=f"🎻 {num_col} Distribution by {cat_col}",
                      font=dict(size=20, color=textcolor, family='Arial Black')),
            yaxis=dict(title=num_col, gridcolor=gridcolor,
                      titlefont=dict(color=textcolor), tickfont=dict(color=textcolor)),
            xaxis=dict(title=cat_col, tickfont=dict(color=textcolor),
                      titlefont=dict(color=textcolor)),
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            font=dict(color=textcolor),
            showlegend=False,
            width=1000, height=650
        )
        
        _display_plotly_figure(fig)
        
        # Quick stats
        stats_text = Text()
        for cat in categories[:5]:
            cat_data = df[df[cat_col] == cat][num_col].dropna()
            if len(cat_data) > 0:
                stats_text.append(
                    f"📊 {str(cat)[:15]}: mean={cat_data.mean():.1f}, "
                    f"median={cat_data.median():.1f}, std={cat_data.std():.1f}\n",
                    style="cyan"
                )
        console.print(Panel(stats_text, title="🔍 Category Statistics", border_style="cyan"))
        console.print("✅ [green]Violin Box chart created![/green]\n")
        self.plot_count += 1

    # ─── CHART 4: DONUT PIE BREAKDOWN ──────────────────────────────────────────

    def _create_donut_pie_chart(self, df, categorical_cols, numeric_cols, dark_theme):
        """Create a neon-styled donut pie chart for category breakdown"""
        console.print("\n🍩 [bold cyan]4. Donut Breakdown — Category Proportions[/bold cyan]")
        
        # Pick the best categorical (prefer 3-12 unique)
        cat_col = None
        for cat in categorical_cols:
            n = df[cat].nunique()
            if 3 <= n <= 12:
                cat_col = cat
                break
        if not cat_col:
            cat_col = categorical_cols[0]
        
        value_counts = df[cat_col].value_counts().head(10)
        
        # If there are remaining categories beyond top 10, group as "Other"
        if df[cat_col].nunique() > 10:
            other_count = len(df) - value_counts.sum()
            if other_count > 0:
                value_counts = pd.concat([value_counts, pd.Series({'Other': other_count})])
        
        labels = [str(l)[:20] for l in value_counts.index]
        values = value_counts.values
        
        # Neon colors
        neon = ['#ff006e', '#00f5d4', '#3a86ff', '#fee440', '#fb5607',
                '#8338ec', '#f72585', '#4cc9f0', '#7209b7', '#06d6a0', '#ffbe0b']
        light = px.colors.qualitative.Set3
        colors = neon[:len(labels)] if dark_theme else light[:len(labels)]
        
        # If we have a numeric column, add mean as hover info
        hover_text = None
        if numeric_cols:
            hover_parts = []
            for cat in value_counts.index:
                cat_data = df[df[cat_col] == cat][numeric_cols[0]].dropna()
                if len(cat_data) > 0:
                    hover_parts.append(f"Avg {numeric_cols[0]}: {cat_data.mean():.1f}")
                else:
                    hover_parts.append("")
            hover_text = hover_parts
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(
                colors=colors,
                line=dict(color='#0d0221' if dark_theme else '#ffffff', width=2)
            ),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(color='#e0d0ff' if dark_theme else 'black', size=11),
            hovertext=hover_text,
            hoverinfo='label+value+text' if hover_text else 'label+value+percent',
            pull=[0.05 if i == 0 else 0 for i in range(len(labels))]
        )])
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        
        fig.update_layout(
            title=dict(text=f"🍩 Category Breakdown: {cat_col}",
                      font=dict(size=20, color=textcolor, family='Arial Black')),
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            font=dict(color=textcolor),
            legend=dict(
                font=dict(color=textcolor, size=11),
                bgcolor='rgba(13, 2, 33, 0.7)' if dark_theme else 'rgba(255,255,255,0.9)'
            ),
            annotations=[dict(
                text=f'{cat_col}',
                x=0.5, y=0.5,
                font=dict(size=14, color=textcolor, family='Arial Black'),
                showarrow=False
            )],
            width=900, height=700
        )
        
        _display_plotly_figure(fig)
        
        # Summary table
        summary_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count", style="yellow")
        summary_table.add_column("Percentage", style="bold green")
        
        total = values.sum()
        for label, val in zip(labels[:8], values[:8]):
            pct = (val / total) * 100
            summary_table.add_row(label, f"{val:,}", f"{pct:.1f}%")
        
        console.print(Panel(summary_table, title=f"📊 {cat_col} Distribution", border_style="green"))
        console.print("✅ [green]Donut Pie chart created![/green]\n")
        self.plot_count += 1

    # ─── CHART 5: RIDGELINE 3D DISTRIBUTION ───────────────────────────────────

    def _create_ridgeline_3d(self, df, numeric_cols, categorical_cols, dark_theme):
        """Create a 3D ridgeline (joy) plot showing distributions per category"""
        console.print("\n🏔️ [bold cyan]5. 3D Ridgeline — Distribution by Category[/bold cyan]")
        
        # Pick numeric column
        num_col = numeric_cols[0]
        
        # Pick categorical with 3-12 unique values
        cat_col = None
        for cat in categorical_cols:
            n_unique = df[cat].nunique()
            if 3 <= n_unique <= 12:
                cat_col = cat
                break
        if not cat_col:
            # Use top 8 values of first categorical
            cat_col = categorical_cols[0]
            top_vals = df[cat_col].value_counts().head(8).index.tolist()
            df = df.copy()
            df[cat_col] = df[cat_col].where(df[cat_col].isin(top_vals), other=None)
            df = df.dropna(subset=[cat_col])
        
        categories = df[cat_col].value_counts().head(10).index.tolist()
        
        # Neon color palette
        neon_palette = [
            '#ff006e', '#00f5d4', '#3a86ff', '#fee440', '#fb5607',
            '#8338ec', '#f72585', '#4cc9f0', '#7209b7', '#06d6a0'
        ]
        light_palette = [
            '#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261',
            '#264653', '#a8dadc', '#fca311', '#6d6875', '#b5838d'
        ]
        palette = neon_palette if dark_theme else light_palette
        
        fig = go.Figure()
        
        # Create KDE for each category and stack them in 3D
        x_range = np.linspace(df[num_col].min(), df[num_col].max(), 200)
        
        for i, cat in enumerate(categories):
            cat_data = df[df[cat_col] == cat][num_col].dropna().values
            if len(cat_data) < 5:
                continue
            
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(cat_data, bw_method='scott')
                density = kde(x_range)
            except Exception:
                # Fallback: simple histogram-based density
                hist, edges = np.histogram(cat_data, bins=50, range=(x_range[0], x_range[-1]), density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                density = np.interp(x_range, centers, hist)
            
            # Normalize density for visual consistency
            if density.max() > 0:
                density = density / density.max()
            
            color = palette[i % len(palette)]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            
            # Filled area as a surface-like trace
            fig.add_trace(go.Scatter3d(
                x=x_range, y=[i] * len(x_range), z=density,
                mode='lines',
                line=dict(color=color, width=3),
                name=str(cat),
                showlegend=True
            ))
            
            # Fill: create a filled polygon by adding zero-line
            fig.add_trace(go.Mesh3d(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate([[i] * len(x_range), [i] * len(x_range)]),
                z=np.concatenate([density, np.zeros(len(x_range))]),
                color=f'rgba({r}, {g}, {b}, 0.3)',
                flatshading=True,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        gridcolor = '#1a0a3e' if dark_theme else '#e0e0e0'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        
        fig.update_layout(
            title=dict(text=f"🏔️ 3D Ridgeline: {num_col} by {cat_col}",
                      font=dict(size=20, color=textcolor, family='Arial Black')),
            scene=dict(
                xaxis=dict(title=num_col, backgroundcolor=bgcolor, gridcolor=gridcolor,
                          titlefont=dict(color=textcolor), tickfont=dict(color=textcolor)),
                yaxis=dict(
                    title=cat_col,
                    backgroundcolor=bgcolor, gridcolor=gridcolor,
                    titlefont=dict(color=textcolor),
                    tickvals=list(range(len(categories))),
                    ticktext=[str(c)[:15] for c in categories],
                    tickfont=dict(color=textcolor, size=9)
                ),
                zaxis=dict(title="Density", backgroundcolor=bgcolor, gridcolor=gridcolor,
                          titlefont=dict(color=textcolor), tickfont=dict(color=textcolor)),
                bgcolor=bgcolor,
                camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0))
            ),
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            font=dict(color=textcolor),
            legend=dict(font=dict(color=textcolor, size=10)),
            width=1100, height=750
        )
        
        _display_plotly_figure(fig)
        
        # Category comparison insights
        stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        stats_table.add_column("Category", style="cyan")
        stats_table.add_column("Count", style="yellow")
        stats_table.add_column(f"Mean {num_col}", style="green")
        stats_table.add_column(f"Std {num_col}", style="green")
        
        for cat in categories[:8]:
            cat_data = df[df[cat_col] == cat][num_col].dropna()
            if len(cat_data) > 0:
                stats_table.add_row(
                    str(cat)[:20],
                    f"{len(cat_data):,}",
                    f"{cat_data.mean():.2f}",
                    f"{cat_data.std():.2f}"
                )
        
        console.print(Panel(stats_table, title=f"📊 {num_col} by {cat_col}", border_style="magenta"))
        console.print("✅ [green]3D Ridgeline created![/green]\n")
        self.plot_count += 1

    # ─── FALLBACK: DISTRIBUTION PRO ────────────────────────────────────────────

    def _create_distribution_pro(self, df, column, dark_theme):
        """Create professional distribution with statistics (fallback chart)"""
        console.print(f"\n📊 [bold cyan]Distribution Analysis: {column}[/bold cyan]")
        
        data = df[column].dropna()
        
        fig = go.Figure()
        
        # Histogram + KDE overlay
        fig.add_trace(go.Histogram(
            x=data, nbinsx=50, name='Distribution',
            marker=dict(
                color='#a200d1' if dark_theme else '#636EFA',
                line=dict(color='#d000ff' if dark_theme else '#4a5568', width=0.5)
            ),
            opacity=0.75
        ))
        
        # Mean & Median lines
        mean_val = data.mean()
        median_val = data.median()
        fig.add_vline(x=mean_val, line_dash="dash",
                     line_color='#ff006e' if dark_theme else 'red',
                     annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top")
        fig.add_vline(x=median_val, line_dash="dot",
                     line_color='#00f5d4' if dark_theme else 'green',
                     annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom")
        
        bgcolor = '#0d0221' if dark_theme else '#ffffff'
        gridcolor = '#1a0a3e' if dark_theme else '#e0e0e0'
        textcolor = '#e0d0ff' if dark_theme else 'black'
        
        fig.update_layout(
            title=dict(text=f"Distribution: {column}", font=dict(size=20, color=textcolor)),
            xaxis=dict(title=column, gridcolor=gridcolor, titlefont=dict(color=textcolor),
                      tickfont=dict(color=textcolor)),
            yaxis=dict(title='Count', gridcolor=gridcolor, titlefont=dict(color=textcolor),
                      tickfont=dict(color=textcolor)),
            paper_bgcolor=bgcolor, plot_bgcolor=bgcolor,
            font=dict(color=textcolor),
            width=1000, height=600, showlegend=True
        )
        
        _display_plotly_figure(fig)
        
        # Stats panel
        stats_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="bold green")
        
        for label, val in [("Count", f"{len(data):,}"), ("Mean", f"{mean_val:.2f}"),
                           ("Median", f"{median_val:.2f}"), ("Std Dev", f"{data.std():.2f}"),
                           ("Min", f"{data.min():.2f}"), ("Max", f"{data.max():.2f}"),
                           ("Skewness", f"{stats.skew(data):.3f}"),
                           ("Kurtosis", f"{stats.kurtosis(data):.3f}")]:
            stats_table.add_row(label, val)
        
        console.print(Panel(stats_table, title=f"📊 {column}", border_style="magenta"))
        console.print("─" * 80 + "\n")
        self.plot_count += 1


def smart_viz(
    df: pd.DataFrame,
    mode: str = "auto",
    columns: list = None,
    target: str = None,
    viz_type: str = "advanced",  # "basic" or "advanced"
    dark_theme: bool = True,
    sample_size: int = 10000
):
    """
    🎨 EssentiaX Unified Smart Visualization Engine
    
    ONE function for all visualizations - basic and advanced 3D analytics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    mode : str
        "auto" - AI selects variables | "manual" - You specify columns
    columns : list
        Specific columns (required for manual mode)
    target : str
        Target variable for supervised analysis
    viz_type : str
        "basic" - Standard 2D plots | "advanced" - Professional 3D analytics
    dark_theme : bool
        True - Dark theme (presentation) | False - Light theme (print)
    sample_size : int
        Sample size for large datasets
        
    Examples:
    ---------
    # Auto mode with advanced 3D visualizations
    smart_viz(df, mode='auto', viz_type='advanced', dark_theme=True)
    
    # Manual mode with specific columns
    smart_viz(df, mode='manual', columns=['age', 'income'], viz_type='advanced')
    
    # Basic 2D visualizations
    smart_viz(df, mode='auto', viz_type='basic')
    """
    
    # Reset stream cleanup flag for this new call
    global _stream_cleaned_after_progress
    _stream_cleaned_after_progress = False
    
    # Initialize engine
    engine = SmartVizEngine()
    
    # Header
    console.print("\n" + "="*80)
    console.print("🎨 [bold magenta]EssentiaX Smart Visualization Engine[/bold magenta] 🎨", justify="center")
    console.print("="*80)
    
    # Visualization Setup Panel
    setup_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
    setup_table.add_column("Setting", style="cyan")
    setup_table.add_column("Value", style="bold green")
    
    setup_table.add_row("Dataset Shape", f"{df.shape[0]:,} × {df.shape[1]}")
    setup_table.add_row("Mode", mode.upper())
    setup_table.add_row("Visualization Type", viz_type.upper())
    setup_table.add_row("Theme", "🌙 Dark" if dark_theme else "☀️ Light")
    setup_table.add_row("Target Variable", target if target else "None")
    
    console.print(Panel(setup_table, title="📊 Visualization Setup", border_style="blue"))
    
    # Sample large datasets
    if len(df) > sample_size:
        df_viz = df.sample(sample_size, random_state=42)
        console.print(f"\n⚡ [yellow]Sampled {sample_size:,} rows for performance[/yellow]")
    else:
        df_viz = df.copy()
    
    # Variable selection
    if mode == "auto":
        selected_vars = engine._auto_select_variables(df_viz, 12)
        
        summary_table = Table(show_header=True, header_style="bold cyan")
        summary_table.add_column("Category", style="yellow")
        summary_table.add_column("Selected Variables", style="green")
        
        summary_table.add_row("Numeric", ", ".join(selected_vars['numeric'][:5]))
        summary_table.add_row("Categorical", ", ".join(selected_vars['categorical'][:5]))
        summary_table.add_row("Correlations", f"{len(selected_vars['correlations'])} pairs found")
        
        console.print(Panel(summary_table, title="🤖 AI Variable Selection", border_style="green"))
        
    elif mode == "manual":
        if not columns:
            console.print("[bold red]❌ Manual mode requires 'columns' parameter![/bold red]")
            return
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            console.print(f"[bold red]❌ Columns not found: {missing_cols}[/bold red]")
            return
        
        numeric_cols = df_viz[columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_viz[columns].select_dtypes(include=['object', 'category']).columns.tolist()
        
        selected_vars = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'correlations': []
        }
        
        console.print(f"✅ [green]Manual selection: {len(columns)} variables[/green]")
    
    # Start visualization
    console.print("\n🎬 [bold cyan]Starting Visualization Process...[/bold cyan]\n")
    
    if viz_type == "advanced":
        # ADVANCED 3D VISUALIZATIONS
        engine._create_advanced_visualizations(df_viz, selected_vars, dark_theme, target)
    else:
        # BASIC 2D VISUALIZATIONS
        engine._create_basic_visualizations(df_viz, selected_vars, target)
    
    # Final Summary
    console.print("\n" + "="*80)
    summary_panel = Panel(
        f"✨ **Visualization Complete!**\n\n"
        f"📊 Total Plots: {engine.plot_count}\n"
        f"🎯 Variables: {len(selected_vars['numeric']) + len(selected_vars['categorical'])}\n"
        f"🔍 Insights: {engine.plot_count * 2}+\n"
        f"⚡ Type: {viz_type.upper()}\n"
        f"🎨 Theme: {'Dark' if dark_theme else 'Light'}\n\n"
        f"💡 **Next Steps:**\n"
        f"• Feature engineering\n"
        f"• Data cleaning\n"
        f"• Model selection",
        title="🎉 Visualization Summary",
        border_style="magenta"
    )
    console.print(summary_panel)
    console.print("="*80 + "\n")


# Legacy function for backward compatibility
def smart_viz_legacy(df, specific_columns, target_column=None, **kwargs):
    """Legacy function - redirects to new smart_viz with manual mode"""
    return smart_viz(
        df=df,
        mode="manual",
        columns=specific_columns,
        target=target_column,
        **kwargs
    )
