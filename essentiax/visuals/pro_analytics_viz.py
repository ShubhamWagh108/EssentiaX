














































































































































"""
pro_analytics_viz.py — EssentiaX Professional Analytics Visualizations
======================================================================
🎨 YouTube-Style Analytics Dashboards
📊 Production-Ready Professional Charts
🚀 Advanced 3D & Interactive Visualizations
💎 Beautiful Dark-Themed Plots
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import time

warnings.filterwarnings("ignore")
console = Console()

# Professional color schemes
DARK_THEME = {
    'bgcolor': '#1a1a1a',
    'gridcolor': '#333333',
    'textcolor': '#ffffff',
    'accent1': '#00d4ff',
    'accent2': '#ff6b6b',
    'accent3': '#4ecdc4',
    'accent4': '#ffe66d'
}

GRADIENT_COLORS = [
    '#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
    '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'
]

# Environment detection and display setup
try:
    import plotly.io as pio
    from IPython.display import display, HTML, clear_output
    
    def _detect_environment():
        try:
            import google.colab
            return 'colab'
        except ImportError:
            pass
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return 'jupyter'
        except NameError:
            pass
        return 'terminal'
    
    _ENVIRONMENT = _detect_environment()
    
    if _ENVIRONMENT == 'colab':
        pio.renderers.default = 'colab'
    elif _ENVIRONMENT == 'jupyter':
        pio.renderers.default = 'notebook'
    else:
        pio.renderers.default = 'browser'
except ImportError:
    _ENVIRONMENT = 'terminal'


def _display_plotly_figure(fig):
    """Enhanced display with stream cleanup for Colab compatibility"""
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        console.file.flush()
    except:
        pass
    
    if _ENVIRONMENT == 'colab':
        try:
            clear_output(wait=False)
            time.sleep(0.05)
            display(fig)
            time.sleep(0.05)
        except:
            fig.show()
    else:
        try:
            display(fig)
        except:
            fig.show()


def create_3d_bubble_scatter(df, x_col, y_col, z_col, size_col=None, color_col=None, 
                             title="3D Analytics Visualization", dark_theme=True):
    """
    Create professional 3D bubble scatter plot like YouTube analytics
    
    Parameters:
    -----------
    df : DataFrame
    x_col : str - X axis column (e.g., 'sentiment')
    y_col : str - Y axis column (e.g., 'engagement')
    z_col : str - Z axis column (e.g., 'title_length')
    size_col : str - Bubble size column (e.g., 'views')
    color_col : str - Color column (e.g., 'views' or 'engagement_rate')
    title : str
    dark_theme : bool
    """
    console.print(f"\n🎨 [bold cyan]Creating 3D Bubble Scatter: {title}[/bold cyan]")
    
    # Prepare data
    plot_df = df[[x_col, y_col, z_col]].copy()
    
    if size_col:
        plot_df['size'] = df[size_col]
        # Normalize sizes for better visualization
        plot_df['size_normalized'] = (plot_df['size'] - plot_df['size'].min()) / (plot_df['size'].max() - plot_df['size'].min()) * 50 + 10
    else:
        plot_df['size_normalized'] = 20
    
    if color_col:
        plot_df['color'] = df[color_col]
    else:
        plot_df['color'] = plot_df[y_col]
    
    # Create 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df[x_col],
        y=plot_df[y_col],
        z=plot_df[z_col],
        mode='markers',
        marker=dict(
            size=plot_df['size_normalized'],
            color=plot_df['color'],
            colorscale='Viridis' if not dark_theme else 'Plasma',
            showscale=True,
            colorbar=dict(
                title=color_col if color_col else y_col,
                titlefont=dict(color='white' if dark_theme else 'black'),
                tickfont=dict(color='white' if dark_theme else 'black')
            ),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)' if dark_theme else 'rgba(0,0,0,0.3)'),
            opacity=0.8
        ),
        text=[f"{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}" + 
              (f"<br>{size_col}: {s:,.0f}" if size_col else "") 
              for x, y, z, s in zip(plot_df[x_col], plot_df[y_col], plot_df[z_col], 
                                   plot_df['size'] if size_col else [0]*len(plot_df))],
        hovertemplate='<b>%{text}</b><extra></extra>'
    )])
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='white')),
            scene=dict(
                xaxis=dict(title=x_col, backgroundcolor=DARK_THEME['bgcolor'], 
                          gridcolor=DARK_THEME['gridcolor'], titlefont=dict(color='white')),
                yaxis=dict(title=y_col, backgroundcolor=DARK_THEME['bgcolor'], 
                          gridcolor=DARK_THEME['gridcolor'], titlefont=dict(color='white')),
                zaxis=dict(title=z_col, backgroundcolor=DARK_THEME['bgcolor'], 
                          gridcolor=DARK_THEME['gridcolor'], titlefont=dict(color='white')),
                bgcolor=DARK_THEME['bgcolor']
            ),
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white'),
            width=1000,
            height=700
        )
    else:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            scene=dict(
                xaxis=dict(title=x_col),
                yaxis=dict(title=y_col),
                zaxis=dict(title=z_col)
            ),
            width=1000,
            height=700
        )
    
    _display_plotly_figure(fig)
    console.print("✅ [green]3D Bubble Scatter created successfully![/green]\n")
    return fig


def create_sunburst_hierarchy(df, path_columns, value_column, color_column=None,
                               title="Hierarchical Distribution", dark_theme=True):
    """
    Create professional sunburst chart for hierarchical data
    
    Parameters:
    -----------
    df : DataFrame
    path_columns : list - Hierarchy levels (e.g., ['sentiment', 'category', 'subcategory'])
    value_column : str - Size of segments (e.g., 'views')
    color_column : str - Color metric (e.g., 'engagement_rate')
    title : str
    dark_theme : bool
    """
    console.print(f"\n☀️ [bold cyan]Creating Sunburst Chart: {title}[/bold cyan]")
    
    # Prepare data for sunburst
    plot_df = df[path_columns + [value_column]].copy()
    if color_column and color_column in df.columns:
        plot_df[color_column] = df[color_column]
    
    fig = px.sunburst(
        plot_df,
        path=path_columns,
        values=value_column,
        color=color_column if color_column else value_column,
        color_continuous_scale='Plasma' if dark_theme else 'Viridis',
        title=title
    )
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white', size=12),
            title=dict(font=dict(size=20, color='white')),
            width=900,
            height=900
        )
    else:
        fig.update_layout(width=900, height=900)
    
    _display_plotly_figure(fig)
    console.print("✅ [green]Sunburst chart created successfully![/green]\n")
    return fig


def create_correlation_heatmap_pro(df, columns=None, title="Feature Correlation Matrix",
                                    dark_theme=True, annotate=True):
    """
    Create professional correlation heatmap with annotations
    
    Parameters:
    -----------
    df : DataFrame
    columns : list - Columns to include (None = all numeric)
    title : str
    dark_theme : bool
    annotate : bool - Show correlation values
    """
    console.print(f"\n🔥 [bold cyan]Creating Correlation Heatmap: {title}[/bold cyan]")
    
    # Select numeric columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    # Create annotations
    annotations = []
    if annotate:
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.index[i],
                        text=f"{value:.2f}",
                        showarrow=False,
                        font=dict(
                            color='white' if abs(value) > 0.5 else ('white' if dark_theme else 'black'),
                            size=10
                        )
                    )
                )
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}' if annotate else None,
        textfont={"size": 10},
        colorbar=dict(
            title="Correlation",
            titlefont=dict(color='white' if dark_theme else 'black'),
            tickfont=dict(color='white' if dark_theme else 'black')
        )
    ))
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='white')),
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white'),
            xaxis=dict(tickangle=-45, tickfont=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white')),
            width=900,
            height=800
        )
    else:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(tickangle=-45),
            width=900,
            height=800
        )
    
    _display_plotly_figure(fig)
    console.print("✅ [green]Correlation heatmap created successfully![/green]\n")
    return fig


def create_temporal_trend_bubble(df, time_col, y_col, size_col=None, color_col=None,
                                  title="Temporal Trend Analysis", dark_theme=True):
    """
    Create temporal trend visualization with bubbles (like YouTube views over time)
    
    Parameters:
    -----------
    df : DataFrame
    time_col : str - Time column (will be converted to datetime)
    y_col : str - Y axis metric (e.g., 'views')
    size_col : str - Bubble size (e.g., 'engagement')
    color_col : str - Color metric (e.g., 'sentiment')
    title : str
    dark_theme : bool
    """
    console.print(f"\n📈 [bold cyan]Creating Temporal Trend: {title}[/bold cyan]")
    
    # Prepare data
    plot_df = df.copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
    plot_df = plot_df.sort_values(time_col)
    
    if size_col:
        plot_df['size_normalized'] = (plot_df[size_col] - plot_df[size_col].min()) / \
                                     (plot_df[size_col].max() - plot_df[size_col].min()) * 40 + 10
    else:
        plot_df['size_normalized'] = 15
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_df[time_col],
        y=plot_df[y_col],
        mode='markers',
        marker=dict(
            size=plot_df['size_normalized'],
            color=plot_df[color_col] if color_col else plot_df[y_col],
            colorscale='Plasma' if dark_theme else 'Viridis',
            showscale=True,
            colorbar=dict(
                title=color_col if color_col else y_col,
                titlefont=dict(color='white' if dark_theme else 'black'),
                tickfont=dict(color='white' if dark_theme else 'black')
            ),
            line=dict(width=1, color='rgba(255,255,255,0.3)' if dark_theme else 'rgba(0,0,0,0.3)'),
            opacity=0.7
        ),
        text=[f"Date: {t.strftime('%Y-%m-%d')}<br>{y_col}: {y:,.0f}" + 
              (f"<br>{size_col}: {s:.2f}" if size_col else "") +
              (f"<br>{color_col}: {c:.2f}" if color_col else "")
              for t, y, s, c in zip(plot_df[time_col], plot_df[y_col],
                                   plot_df[size_col] if size_col else [0]*len(plot_df),
                                   plot_df[color_col] if color_col else [0]*len(plot_df))],
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='white')),
            xaxis=dict(
                title=time_col,
                gridcolor=DARK_THEME['gridcolor'],
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title=y_col,
                gridcolor=DARK_THEME['gridcolor'],
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white'),
            width=1200,
            height=600,
            hovermode='closest'
        )
    else:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(title=time_col),
            yaxis=dict(title=y_col),
            width=1200,
            height=600,
            hovermode='closest'
        )
    
    _display_plotly_figure(fig)
    console.print("✅ [green]Temporal trend created successfully![/green]\n")
    return fig


def create_scatter_matrix_pro(df, columns=None, color_col=None, 
                               title="Multi-Variable Scatter Matrix", dark_theme=True):
    """
    Create professional scatter matrix with color coding
    
    Parameters:
    -----------
    df : DataFrame
    columns : list - Columns to include (max 6 recommended)
    color_col : str - Column for color coding
    title : str
    dark_theme : bool
    """
    console.print(f"\n🎯 [bold cyan]Creating Scatter Matrix: {title}[/bold cyan]")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    elif len(columns) > 6:
        console.print("[yellow]⚠️ Limiting to first 6 columns for readability[/yellow]")
        columns = columns[:6]
    
    plot_df = df[columns].copy()
    if color_col and color_col in df.columns:
        plot_df[color_col] = df[color_col]
    
    fig = px.scatter_matrix(
        plot_df,
        dimensions=columns,
        color=color_col if color_col else None,
        color_continuous_scale='Plasma' if dark_theme else 'Viridis',
        title=title
    )
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white', size=10),
            title=dict(font=dict(size=20, color='white')),
            width=1200,
            height=1000
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.6))
    else:
        fig.update_layout(width=1200, height=1000)
        fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.6))
    
    _display_plotly_figure(fig)
    console.print("✅ [green]Scatter matrix created successfully![/green]\n")
    return fig


def create_distribution_histogram_pro(df, column, bins=50, title=None, dark_theme=True):
    """
    Create professional distribution histogram with statistics overlay
    
    Parameters:
    -----------
    df : DataFrame
    column : str - Column to visualize
    bins : int
    title : str
    dark_theme : bool
    """
    if title is None:
        title = f"Distribution Analysis: {column}"
    
    console.print(f"\n📊 [bold cyan]Creating Distribution: {title}[/bold cyan]")
    
    data = df[column].dropna()
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name='Distribution',
        marker=dict(
            color=DARK_THEME['accent1'] if dark_theme else '#636EFA',
            line=dict(color='white' if dark_theme else 'black', width=0.5)
        ),
        opacity=0.7
    ))
    
    # Add mean line
    mean_val = data.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=DARK_THEME['accent2'] if dark_theme else 'red',
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top"
    )
    
    # Add median line
    median_val = data.median()
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=DARK_THEME['accent3'] if dark_theme else 'green',
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="bottom"
    )
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='white')),
            xaxis=dict(
                title=column,
                gridcolor=DARK_THEME['gridcolor'],
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title='Count',
                gridcolor=DARK_THEME['gridcolor'],
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white'),
            width=1000,
            height=600,
            showlegend=True
        )
    else:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(title=column),
            yaxis=dict(title='Count'),
            width=1000,
            height=600,
            showlegend=True
        )
    
    _display_plotly_figure(fig)
    console.print("✅ [green]Distribution histogram created successfully![/green]\n")
    return fig


def create_category_bar_pro(df, category_col, value_col=None, top_n=15,
                             title=None, dark_theme=True, horizontal=False):
    """
    Create professional category bar chart
    
    Parameters:
    -----------
    df : DataFrame
    category_col : str - Category column
    value_col : str - Value column (None = count)
    top_n : int - Show top N categories
    title : str
    dark_theme : bool
    horizontal : bool
    """
    if title is None:
        title = f"Category Distribution: {category_col}"
    
    console.print(f"\n📊 [bold cyan]Creating Category Bar Chart: {title}[/bold cyan]")
    
    if value_col:
        plot_data = df.groupby(category_col)[value_col].sum().sort_values(ascending=False).head(top_n)
    else:
        plot_data = df[category_col].value_counts().head(top_n)
    
    if horizontal:
        fig = go.Figure(go.Bar(
            y=plot_data.index,
            x=plot_data.values,
            orientation='h',
            marker=dict(
                color=plot_data.values,
                colorscale='Plasma' if dark_theme else 'Viridis',
                showscale=True,
                colorbar=dict(
                    title="Value",
                    titlefont=dict(color='white' if dark_theme else 'black'),
                    tickfont=dict(color='white' if dark_theme else 'black')
                )
            ),
            text=plot_data.values,
            texttemplate='%{text:,.0f}',
            textposition='outside'
        ))
    else:
        fig = go.Figure(go.Bar(
            x=plot_data.index,
            y=plot_data.values,
            marker=dict(
                color=plot_data.values,
                colorscale='Plasma' if dark_theme else 'Viridis',
                showscale=True,
                colorbar=dict(
                    title="Value",
                    titlefont=dict(color='white' if dark_theme else 'black'),
                    tickfont=dict(color='white' if dark_theme else 'black')
                )
            ),
            text=plot_data.values,
            texttemplate='%{text:,.0f}',
            textposition='outside'
        ))
    
    # Professional styling
    if dark_theme:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='white')),
            xaxis=dict(
                title=category_col if not horizontal else (value_col or 'Count'),
                gridcolor=DARK_THEME['gridcolor'],
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                tickangle=-45 if not horizontal else 0
            ),
            yaxis=dict(
                title=value_col or 'Count' if not horizontal else category_col,
                gridcolor=DARK_THEME['gridcolor'],
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            paper_bgcolor=DARK_THEME['bgcolor'],
            plot_bgcolor=DARK_THEME['bgcolor'],
            font=dict(color='white'),
            width=1000,
            height=600
        )
    else:
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(title=category_col if not horizontal else (value_col or 'Count'), tickangle=-45 if not horizontal else 0),
            yaxis=dict(title=value_col or 'Count' if not horizontal else category_col),
            width=1000,
            height=600
        )
    
    _display_plotly_figure(fig)
    console.print("✅ [green]Category bar chart created successfully![/green]\n")
    return fig


# Main function for easy access
def pro_analytics_dashboard(df, config='auto', dark_theme=True):
    """
    Create a complete professional analytics dashboard
    
    Parameters:
    -----------
    df : DataFrame
    config : str or dict
        'auto' - Automatically detect and create relevant visualizations
        dict - Custom configuration
    dark_theme : bool
    """
    console.print("\n" + "="*80)
    console.print("🎨 [bold magenta]Professional Analytics Dashboard[/bold magenta]", justify="center")
    console.print("="*80 + "\n")
    
    # Dataset info
    info_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="bold green")
    info_table.add_row("Rows", f"{len(df):,}")
    info_table.add_row("Columns", f"{df.shape[1]}")
    info_table.add_row("Theme", "Dark" if dark_theme else "Light")
    
    console.print(Panel(info_table, title="📊 Dataset Info", border_style="blue"))
    
    figures = []
    
    if config == 'auto':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 3:
            fig = create_correlation_heatmap_pro(df, numeric_cols[:10], dark_theme=dark_theme)
            figures.append(fig)
        
        # 2. Scatter matrix for top numeric columns
        if len(numeric_cols) >= 3:
            fig = create_scatter_matrix_pro(df, numeric_cols[:5], 
                                           color_col=numeric_cols[0] if numeric_cols else None,
                                           dark_theme=dark_theme)
            figures.append(fig)
        
        # 3. Distribution for top numeric columns
        for col in numeric_cols[:3]:
            fig = create_distribution_histogram_pro(df, col, dark_theme=dark_theme)
            figures.append(fig)
        
        # 4. Category bars for categorical columns
        for col in categorical_cols[:2]:
            fig = create_category_bar_pro(df, col, dark_theme=dark_theme)
            figures.append(fig)
    
    console.print("\n" + "="*80)
    console.print(f"✨ [bold green]Dashboard Complete! Created {len(figures)} visualizations[/bold green]", justify="center")
    console.print("="*80 + "\n")
    
    return figures
