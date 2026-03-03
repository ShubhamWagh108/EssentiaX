"""
advanced_viz.py — EssentiaX Advanced 3D & Interactive Visualizations
====================================================================
🎨 STUNNING 3D Visualizations
🚀 Next-Gen Interactive Charts
💎 Production-Ready Beautiful Plots
🎭 Advanced Statistical Visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
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


class Advanced3DViz:
    """Advanced 3D and Interactive Visualization Engine"""
    
    def __init__(self):
        self.plot_count = 0
        
    def plot_3d_scatter_clusters(self, df, columns=None, n_clusters=3, title=None):
        """
        Create stunning 3D scatter plot with automatic clustering
        
        Parameters:
        -----------
        df : DataFrame
        columns : list of 3 column names (if None, auto-select top 3 by variance)
        n_clusters : number of clusters to identify
        title : custom title
        """
        console.print("\n🎨 [bold cyan]Creating 3D Scatter Plot with Clustering...[/bold cyan]")
        
        # Auto-select columns if not provided
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns is None:
            # Select top 3 by variance
            variances = df[numeric_cols].var().sort_values(ascending=False)
            columns = variances.head(3).index.tolist()
        
        if len(columns) < 3:
            console.print("[yellow]⚠️ Need at least 3 numeric columns for 3D plot[/yellow]")
            return
        
        # Prepare data
        X = df[columns[:3]].dropna()
        
        # Perform clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:n_clusters]
        
        for i in range(n_clusters):
            mask = clusters == i
            fig.add_trace(go.Scatter3d(
                x=X.iloc[mask, 0],
                y=X.iloc[mask, 1],
                z=X.iloc[mask, 2],
                mode='markers',
                name=f'Cluster {i+1}',
                marker=dict(
                    size=6,
                    color=colors[i],
                    opacity=0.8,
                    line=dict(color='white', width=0.5)
                ),
                hovertemplate=f'<b>Cluster {i+1}</b><br>' +
                             f'{columns[0]}: %{{x:.2f}}<br>' +
                             f'{columns[1]}: %{{y:.2f}}<br>' +
                             f'{columns[2]}: %{{z:.2f}}<extra></extra>'
            ))
        
        # Add cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers',
            name='Centroids',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=2)
            ),
            hovertemplate='<b>Centroid</b><extra></extra>'
        ))
        
        fig.update_layout(
            title=title or f'🎨 3D Cluster Analysis: {", ".join(columns[:3])}',
            scene=dict(
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                zaxis_title=columns[2],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                bgcolor='rgba(240, 240, 255, 0.9)'
            ),
            font=dict(size=12),
            showlegend=True,
            height=700,
            template='plotly_white'
        )
        
        _display_plotly_figure(fig)
        console.print(f"✅ [green]3D scatter plot created with {n_clusters} clusters![/green]\n")
        self.plot_count += 1
        
    def plot_3d_surface(self, df, x_col, y_col, z_col=None, title=None):
        """
        Create beautiful 3D surface plot
        
        Parameters:
        -----------
        df : DataFrame
        x_col, y_col : column names for X and Y axes
        z_col : column for Z axis (if None, uses density estimation)
        """
        console.print("\n🌊 [bold cyan]Creating 3D Surface Plot...[/bold cyan]")
        
        if z_col:
            # Direct surface from data
            pivot_data = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
            
            fig = go.Figure(data=[go.Surface(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z:.2f}}<extra></extra>'
            )])
            
            z_title = z_col
        else:
            # Create density surface
            x_data = df[x_col].dropna()
            y_data = df[y_col].dropna()
            
            # Create grid
            x_grid = np.linspace(x_data.min(), x_data.max(), 50)
            y_grid = np.linspace(y_data.min(), y_data.max(), 50)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            # Calculate 2D density
            from scipy.stats import gaussian_kde
            xy = np.vstack([x_data, y_data])
            kde = gaussian_kde(xy)
            Z_grid = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
            
            fig = go.Figure(data=[go.Surface(
                z=Z_grid,
                x=x_grid,
                y=y_grid,
                colorscale='Plasma',
                hovertemplate=f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<br>Density: %{{z:.4f}}<extra></extra>'
            )])
            
            z_title = 'Density'
        
        fig.update_layout(
            title=title or f'🌊 3D Surface: {x_col} × {y_col}',
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_title,
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.3)),
                bgcolor='rgba(240, 240, 255, 0.9)'
            ),
            height=700,
            template='plotly_white'
        )
        
        _display_plotly_figure(fig)
        console.print("✅ [green]3D surface plot created![/green]\n")
        self.plot_count += 1
        
    def plot_sunburst(self, df, path_columns, value_column=None, title=None):
        """
        Create stunning sunburst chart for hierarchical categorical data
        
        Parameters:
        -----------
        df : DataFrame
        path_columns : list of columns defining hierarchy (outer to inner)
        value_column : column for sizing (if None, uses count)
        """
        console.print("\n☀️ [bold cyan]Creating Sunburst Chart...[/bold cyan]")
        
        if value_column:
            fig = px.sunburst(
                df,
                path=path_columns,
                values=value_column,
                title=title or f'☀️ Hierarchical Sunburst: {" → ".join(path_columns)}',
                color=value_column,
                color_continuous_scale='RdYlBu_r',
                height=700
            )
        else:
            # Count-based
            df_counts = df.groupby(path_columns).size().reset_index(name='count')
            fig = px.sunburst(
                df_counts,
                path=path_columns,
                values='count',
                title=title or f'☀️ Hierarchical Sunburst: {" → ".join(path_columns)}',
                color='count',
                color_continuous_scale='Viridis',
                height=700
            )
        
        fig.update_traces(
            textinfo='label+percent parent',
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percent: %{percentParent}<extra></extra>'
        )
        
        fig.update_layout(template='plotly_white')
        _display_plotly_figure(fig)
        console.print("✅ [green]Sunburst chart created![/green]\n")
        self.plot_count += 1
        
    def plot_sankey(self, df, source_col, target_col, value_col=None, title=None):
        """
        Create beautiful Sankey diagram for flow visualization
        
        Parameters:
        -----------
        df : DataFrame
        source_col : source category column
        target_col : target category column
        value_col : flow value column (if None, uses count)
        """
        console.print("\n🌊 [bold cyan]Creating Sankey Diagram...[/bold cyan]")
        
        if value_col:
            flow_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
        else:
            flow_data = df.groupby([source_col, target_col]).size().reset_index(name='value')
            value_col = 'value'
        
        # Create node labels
        all_nodes = list(set(flow_data[source_col].unique()) | set(flow_data[target_col].unique()))
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Map to indices
        source_indices = flow_data[source_col].map(node_dict).tolist()
        target_indices = flow_data[target_col].map(node_dict).tolist()
        values = flow_data[value_col].tolist()
        
        # Create colors
        colors = px.colors.qualitative.Set3 * (len(all_nodes) // len(px.colors.qualitative.Set3) + 1)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_nodes,
                color=colors[:len(all_nodes)]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color='rgba(0,0,0,0.2)'
            )
        )])
        
        fig.update_layout(
            title=title or f'🌊 Flow Diagram: {source_col} → {target_col}',
            font=dict(size=12),
            height=600,
            template='plotly_white'
        )
        
        _display_plotly_figure(fig)
        console.print("✅ [green]Sankey diagram created![/green]\n")
        self.plot_count += 1
        
    def plot_violin_advanced(self, df, columns=None, title=None):
        """
        Create advanced violin plots with box plots and statistical overlays
        
        Parameters:
        -----------
        df : DataFrame
        columns : list of numeric columns (if None, auto-select)
        """
        console.print("\n🎻 [bold cyan]Creating Advanced Violin Plots...[/bold cyan]")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns is None:
            columns = numeric_cols[:6]  # Limit to 6 for readability
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for idx, col in enumerate(columns):
            data = df[col].dropna()
            
            fig.add_trace(go.Violin(
                y=data,
                name=col,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[idx % len(colors)],
                opacity=0.7,
                x0=col,
                hovertemplate=f'<b>{col}</b><br>Value: %{{y:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title or '🎻 Advanced Violin Plot Analysis',
            yaxis_title='Value',
            showlegend=False,
            height=600,
            template='plotly_white',
            violinmode='group'
        )
        
        _display_plotly_figure(fig)
        console.print("✅ [green]Violin plots created![/green]\n")
        self.plot_count += 1
        
    def plot_parallel_coordinates(self, df, color_col=None, columns=None, title=None):
        """
        Create interactive parallel coordinates plot
        
        Parameters:
        -----------
        df : DataFrame
        color_col : column to use for coloring
        columns : list of columns to include (if None, uses all numeric)
        """
        console.print("\n📊 [bold cyan]Creating Parallel Coordinates Plot...[/bold cyan]")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns is None:
            columns = numeric_cols[:8]  # Limit for readability
        
        if color_col and color_col not in columns:
            columns = [color_col] + columns
        
        df_plot = df[columns].dropna()
        
        if color_col:
            fig = px.parallel_coordinates(
                df_plot,
                color=color_col,
                dimensions=columns,
                title=title or f'📊 Parallel Coordinates (colored by {color_col})',
                color_continuous_scale='Viridis',
                height=600
            )
        else:
            fig = px.parallel_coordinates(
                df_plot,
                dimensions=columns,
                title=title or '📊 Parallel Coordinates Analysis',
                height=600
            )
        
        fig.update_layout(template='plotly_white')
        _display_plotly_figure(fig)
        console.print("✅ [green]Parallel coordinates plot created![/green]\n")
        self.plot_count += 1
        
    def plot_treemap(self, df, path_columns, value_column=None, title=None):
        """
        Create interactive treemap for hierarchical data
        
        Parameters:
        -----------
        df : DataFrame
        path_columns : list of columns defining hierarchy
        value_column : column for sizing (if None, uses count)
        """
        console.print("\n🗺️ [bold cyan]Creating Treemap...[/bold cyan]")
        
        if value_column:
            fig = px.treemap(
                df,
                path=path_columns,
                values=value_column,
                title=title or f'🗺️ Treemap: {" → ".join(path_columns)}',
                color=value_column,
                color_continuous_scale='RdYlGn',
                height=600
            )
        else:
            df_counts = df.groupby(path_columns).size().reset_index(name='count')
            fig = px.treemap(
                df_counts,
                path=path_columns,
                values='count',
                title=title or f'🗺️ Treemap: {" → ".join(path_columns)}',
                color='count',
                color_continuous_scale='Blues',
                height=600
            )
        
        fig.update_traces(
            textinfo='label+value+percent parent',
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percent: %{percentParent}<extra></extra>'
        )
        
        fig.update_layout(template='plotly_white')
        _display_plotly_figure(fig)
        console.print("✅ [green]Treemap created![/green]\n")
        self.plot_count += 1
        
    def plot_animated_scatter(self, df, x_col, y_col, animation_col, size_col=None, color_col=None, title=None):
        """
        Create animated scatter plot (great for time series or sequential data)
        
        Parameters:
        -----------
        df : DataFrame
        x_col, y_col : columns for X and Y axes
        animation_col : column to animate over (e.g., time, year)
        size_col : column for bubble size
        color_col : column for coloring
        """
        console.print("\n🎬 [bold cyan]Creating Animated Scatter Plot...[/bold cyan]")
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            animation_frame=animation_col,
            size=size_col,
            color=color_col,
            hover_name=color_col,
            title=title or f'🎬 Animated: {x_col} vs {y_col} over {animation_col}',
            height=600,
            range_x=[df[x_col].min() * 0.9, df[x_col].max() * 1.1],
            range_y=[df[y_col].min() * 0.9, df[y_col].max() * 1.1]
        )
        
        fig.update_layout(template='plotly_white')
        _display_plotly_figure(fig)
        console.print("✅ [green]Animated scatter plot created![/green]\n")
        self.plot_count += 1
        
    def plot_correlation_chord(self, df, columns=None, threshold=0.5, title=None):
        """
        Create beautiful correlation visualization with chord-like connections
        
        Parameters:
        -----------
        df : DataFrame
        columns : list of columns (if None, uses all numeric)
        threshold : minimum correlation to display
        """
        console.print("\n🎭 [bold cyan]Creating Advanced Correlation Visualization...[/bold cyan]")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns is None:
            columns = numeric_cols[:10]  # Limit for readability
        
        corr_matrix = df[columns].corr()
        
        # Create network-style visualization
        fig = go.Figure()
        
        # Add heatmap base
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
            hovertemplate='%{x} × %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title or '🎭 Advanced Correlation Matrix',
            xaxis_title='',
            yaxis_title='',
            height=700,
            width=800,
            template='plotly_white'
        )
        
        _display_plotly_figure(fig)
        
        # Also create a network graph for strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corrs.append({
                        'source': corr_matrix.columns[i],
                        'target': corr_matrix.columns[j],
                        'value': corr_val
                    })
        
        if strong_corrs:
            console.print(f"📊 [yellow]Found {len(strong_corrs)} strong correlations (|r| ≥ {threshold})[/yellow]")
        
        console.print("✅ [green]Correlation visualization created![/green]\n")
        self.plot_count += 1
        
    def plot_ridge(self, df, column, group_by, title=None):
        """
        Create ridge plot (joyplot) for distribution comparison
        
        Parameters:
        -----------
        df : DataFrame
        column : numeric column to plot
        group_by : categorical column to group by
        """
        console.print("\n🏔️ [bold cyan]Creating Ridge Plot...[/bold cyan]")
        
        groups = df[group_by].unique()
        colors = px.colors.qualitative.Set3
        
        fig = go.Figure()
        
        for idx, group in enumerate(groups[:10]):  # Limit to 10 groups
            data = df[df[group_by] == group][column].dropna()
            
            fig.add_trace(go.Violin(
                x=data,
                name=str(group),
                fillcolor=colors[idx % len(colors)],
                opacity=0.7,
                orientation='h',
                side='positive',
                width=3,
                points=False
            ))
        
        fig.update_layout(
            title=title or f'🏔️ Ridge Plot: {column} by {group_by}',
            xaxis_title=column,
            yaxis_title=group_by,
            showlegend=True,
            height=max(400, len(groups) * 50),
            template='plotly_white',
            violingap=0,
            violinmode='overlay'
        )
        
        _display_plotly_figure(fig)
        console.print("✅ [green]Ridge plot created![/green]\n")
        self.plot_count += 1


def advanced_viz(df, viz_type='auto', **kwargs):
    """
    Main function for advanced visualizations
    
    Parameters:
    -----------
    df : DataFrame
    viz_type : str
        - 'auto': automatically create best visualizations
        - '3d_scatter': 3D scatter with clustering
        - '3d_surface': 3D surface plot
        - 'sunburst': hierarchical sunburst
        - 'sankey': flow diagram
        - 'violin': advanced violin plots
        - 'parallel': parallel coordinates
        - 'treemap': hierarchical treemap
        - 'animated': animated scatter
        - 'correlation': advanced correlation viz
        - 'ridge': ridge plot
    **kwargs : additional parameters for specific plot types
    """
    
    engine = Advanced3DViz()
    
    console.print("\n" + "="*80)
    console.print("🎨 [bold magenta]EssentiaX Advanced Visualization Engine[/bold magenta] 🎨", justify="center")
    console.print("="*80)
    
    if viz_type == 'auto':
        console.print("\n🤖 [cyan]Auto mode: Creating stunning visualizations...[/cyan]\n")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. 3D Scatter with clustering
        if len(numeric_cols) >= 3:
            engine.plot_3d_scatter_clusters(df)
        
        # 2. Advanced violin plots
        if len(numeric_cols) >= 2:
            engine.plot_violin_advanced(df, columns=numeric_cols[:5])
        
        # 3. Parallel coordinates
        if len(numeric_cols) >= 4:
            engine.plot_parallel_coordinates(df, columns=numeric_cols[:6])
        
        # 4. Sunburst if hierarchical categorical data
        if len(categorical_cols) >= 2:
            engine.plot_sunburst(df, path_columns=categorical_cols[:3])
        
        # 5. Advanced correlation
        if len(numeric_cols) >= 3:
            engine.plot_correlation_chord(df)
            
    elif viz_type == '3d_scatter':
        engine.plot_3d_scatter_clusters(df, **kwargs)
    elif viz_type == '3d_surface':
        engine.plot_3d_surface(df, **kwargs)
    elif viz_type == 'sunburst':
        engine.plot_sunburst(df, **kwargs)
    elif viz_type == 'sankey':
        engine.plot_sankey(df, **kwargs)
    elif viz_type == 'violin':
        engine.plot_violin_advanced(df, **kwargs)
    elif viz_type == 'parallel':
        engine.plot_parallel_coordinates(df, **kwargs)
    elif viz_type == 'treemap':
        engine.plot_treemap(df, **kwargs)
    elif viz_type == 'animated':
        engine.plot_animated_scatter(df, **kwargs)
    elif viz_type == 'correlation':
        engine.plot_correlation_chord(df, **kwargs)
    elif viz_type == 'ridge':
        engine.plot_ridge(df, **kwargs)
    else:
        console.print(f"[red]❌ Unknown viz_type: {viz_type}[/red]")
        return
    
    # Summary
    console.print("\n" + "="*80)
    summary = Panel(
        f"✨ **Advanced Visualization Complete!**\n\n"
        f"📊 Total Plots: {engine.plot_count}\n"
        f"🎨 Visualization Type: {viz_type.upper()}\n"
        f"💎 All plots are interactive - hover, zoom, rotate!\n\n"
        f"💡 **Available viz_types:**\n"
        f"• 'auto' - Smart selection\n"
        f"• '3d_scatter' - 3D clustering\n"
        f"• '3d_surface' - 3D surface\n"
        f"• 'sunburst' - Hierarchical sunburst\n"
        f"• 'sankey' - Flow diagram\n"
        f"• 'violin' - Advanced violin\n"
        f"• 'parallel' - Parallel coordinates\n"
        f"• 'treemap' - Hierarchical treemap\n"
        f"• 'animated' - Animated scatter\n"
        f"• 'correlation' - Advanced correlation\n"
        f"• 'ridge' - Ridge plot",
        title="🎉 EssentiaX Advanced Viz Summary",
        border_style="magenta"
    )
    console.print(summary)
    console.print("="*80 + "\n")
