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


def smart_viz(
    df: pd.DataFrame,
    mode: str = "auto",  # "auto" or "manual"
    columns: list = None,
    target: str = None,
    max_plots: int = 12,
    interactive: bool = True,
    sample_size: int = 10000
):
    """
    🎨 EssentiaX Next-Gen Smart Visualization Engine
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    mode : str
        "auto" for AI selection, "manual" for user selection
    columns : list
        Specific columns to visualize (required for manual mode)
    target : str
        Target variable for supervised analysis
    max_plots : int
        Maximum number of plots to generate
    interactive : bool
        Use interactive Plotly plots (True) or static Matplotlib (False)
    sample_size : int
        Sample size for large datasets
    """
    
    # Initialize the engine
    engine = SmartVizEngine()
    
    # Beautiful header
    console.print("\n" + "="*80)
    console.print("🎨 [bold magenta]EssentiaX Smart Visualization Engine[/bold magenta] 🎨", justify="center")
    console.print("="*80)
    
    # Dataset info panel
    info_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="bold green")
    
    info_table.add_row("Dataset Shape", f"{df.shape[0]:,} × {df.shape[1]}")
    info_table.add_row("Mode", mode.upper())
    info_table.add_row("Interactive", "✅ Plotly" if interactive else "📊 Matplotlib")
    info_table.add_row("Target Variable", target if target else "None")
    
    console.print(Panel(info_table, title="📊 Visualization Setup", border_style="blue"))
    
    # Sample large datasets
    if len(df) > sample_size:
        df_viz = df.sample(sample_size, random_state=42)
        console.print(f"\n⚡ [yellow]Sampled {sample_size:,} rows for visualization performance[/yellow]")
    else:
        df_viz = df.copy()
    
    # Variable selection based on mode
    if mode == "auto":
        selected_vars = engine._auto_select_variables(df_viz, max_plots)
        
        # Display selection summary
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
        
        # Validate columns
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            console.print(f"[bold red]❌ Columns not found: {missing_cols}[/bold red]")
            return
        
        # Organize manual selection
        numeric_cols = df_viz[columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_viz[columns].select_dtypes(include=['object', 'category']).columns.tolist()
        
        selected_vars = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'correlations': []
        }
        
        console.print(f"✅ [green]Manual selection: {len(columns)} variables[/green]")
    
    # Start visualization process
    console.print("\n🎬 [bold cyan]Starting Visualization Process...[/bold cyan]\n")
    
    # 1. Numeric distributions
    if selected_vars['numeric']:
        console.print("📊 [bold]Generating Distribution Plots...[/bold]")
        for col in selected_vars['numeric'][:6]:  # Limit to prevent overload
            engine._plot_distribution(df_viz, col, interactive)
    
    # 2. Correlation analysis
    if len(selected_vars['numeric']) > 1:
        console.print("\n🔥 [bold]Generating Correlation Analysis...[/bold]")
        engine._plot_correlation_heatmap(df_viz, selected_vars['numeric'])
    
    # 3. Categorical analysis
    if selected_vars['categorical']:
        console.print("\n🏷️ [bold]Generating Categorical Analysis...[/bold]")
        for col in selected_vars['categorical'][:4]:  # Limit categorical plots
            engine._plot_categorical(df_viz, col, target)
    
    # 4. Multi-variable relationships
    if len(selected_vars['numeric']) >= 3:
        console.print("\n🎯 [bold]Generating Multi-Variable Analysis...[/bold]")
        engine._plot_scatter_matrix(df_viz, selected_vars['numeric'][:6])
    
    # Final summary
    console.print("\n" + "="*80)
    summary_panel = Panel(
        f"✨ **Visualization Complete!**\n\n"
        f"📊 Total Plots Generated: {engine.plot_count}\n"
        f"🎯 Variables Analyzed: {len(selected_vars['numeric']) + len(selected_vars['categorical'])}\n"
        f"🔍 Insights Provided: {engine.plot_count * 2}+\n"
        f"⚡ Mode Used: {mode.upper()}\n\n"
        f"💡 **Next Steps:**\n"
        f"• Use insights for feature engineering\n"
        f"• Apply data cleaning based on patterns\n"
        f"• Consider correlations for model selection",
        title="🎉 EssentiaX Visualization Summary",
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
