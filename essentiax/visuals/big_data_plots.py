"""
Big Data Optimized Visualizations for EssentiaX
==============================================
Memory-efficient plotting for large datasets with smart sampling

Features:
- Smart sampling for visualizations
- Statistical diagnostic plots
- Enhanced distribution plots  
- Relationship analysis plots
- Interactive Plotly charts optimized for big data
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


class BigDataPlotter:
    """Memory-efficient plotting engine for large datasets"""
    
    def __init__(self, max_points=10000):
        self.max_points = max_points
        self.color_palette = px.colors.qualitative.Set3
    
    def smart_sample_for_visualization(self, df, plot_type='scatter', target_col=None, preserve_outliers=True):
        """
        Intelligent sampling that preserves data distribution for visualization
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        plot_type : str
            Type of plot ('scatter', 'histogram', 'box', 'categorical')
        target_col : str, optional
            Target column for stratified sampling
        preserve_outliers : bool
            Whether to preserve outliers in sample
            
        Returns:
        --------
        pandas.DataFrame : Sampled dataframe optimized for visualization
        """
        if len(df) <= self.max_points:
            return df.copy()
        
        sample_size = min(self.max_points, len(df))
        
        # Stratified sampling if target column provided
        if target_col and target_col in df.columns:
            try:
                # Stratified sampling by target
                sampled_dfs = []
                for group in df[target_col].unique():
                    group_df = df[df[target_col] == group]
                    group_sample_size = int(sample_size * len(group_df) / len(df))
                    if group_sample_size > 0:
                        if len(group_df) <= group_sample_size:
                            sampled_dfs.append(group_df)
                        else:
                            sampled_dfs.append(group_df.sample(n=group_sample_size, random_state=42))
                
                sampled_df = pd.concat(sampled_dfs, ignore_index=True)
            except:
                # Fallback to random sampling
                sampled_df = df.sample(n=sample_size, random_state=42)
        else:
            # Random sampling
            sampled_df = df.sample(n=sample_size, random_state=42)
        
        # Preserve outliers for certain plot types
        if preserve_outliers and plot_type in ['scatter', 'box']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_indices = set()
            
            for col in numeric_cols:
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                        outlier_indices.update(outliers.index[:50])  # Limit outliers
                except:
                    pass
            
            if outlier_indices:
                outlier_df = df.loc[list(outlier_indices)]
                # Combine sample with outliers, remove duplicates
                sampled_df = pd.concat([sampled_df, outlier_df]).drop_duplicates().reset_index(drop=True)
        
        return sampled_df
    
    def create_diagnostic_plots(self, data, column_name, distribution='norm'):
        """
        Create statistical diagnostic plots for distribution analysis
        
        Parameters:
        -----------
        data : array-like
            Data to analyze
        column_name : str
            Name of the column
        distribution : str
            Distribution to compare against ('norm', 'uniform', 'exponential')
            
        Returns:
        --------
        plotly.graph_objects.Figure : Subplot figure with diagnostic plots
        """
        clean_data = pd.Series(data).dropna()
        
        if len(clean_data) < 10:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for diagnostic plots (n={len(clean_data)})",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=f"Diagnostic Plots: {column_name}")
            return fig
        
        # Sample data if too large
        if len(clean_data) > self.max_points:
            clean_data = clean_data.sample(n=self.max_points, random_state=42)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Q-Q Plot', 'P-P Plot',
                'Histogram with Normal Curve', 'Box Plot'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Q-Q Plot
        if distribution == 'norm':
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_data)))
            sample_quantiles = np.sort(clean_data)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color='blue', size=4, opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Add reference line
            min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val * clean_data.std() + clean_data.mean(), 
                       max_val * clean_data.std() + clean_data.mean()],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # P-P Plot
        sorted_data = np.sort(clean_data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        if distribution == 'norm':
            theoretical_cdf = stats.norm.cdf(sorted_data, loc=clean_data.mean(), scale=clean_data.std())
        else:
            theoretical_cdf = empirical_cdf  # Fallback
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_cdf,
                y=empirical_cdf,
                mode='markers',
                name='P-P Plot',
                marker=dict(color='green', size=4, opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Add reference line for P-P plot
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Histogram with normal curve
        fig.add_trace(
            go.Histogram(
                x=clean_data,
                nbinsx=30,
                name='Histogram',
                opacity=0.7,
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Add normal curve
        if distribution == 'norm':
            x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
            normal_curve = stats.norm.pdf(x_range, loc=clean_data.mean(), scale=clean_data.std())
            # Scale to match histogram
            normal_curve = normal_curve * len(clean_data) * (clean_data.max() - clean_data.min()) / 30
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=normal_curve,
                    mode='lines',
                    name='Normal Curve',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ),
                row=2, col=1, secondary_y=True
            )
        
        # Box Plot
        fig.add_trace(
            go.Box(
                y=clean_data,
                name='Box Plot',
                marker_color='orange',
                boxpoints='outliers'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Diagnostic Plots: {column_name}",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical CDF", row=1, col=2)
        fig.update_yaxes(title_text="Empirical CDF", row=1, col=2)
        fig.update_xaxes(title_text=column_name, row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Density", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text=column_name, row=2, col=2)
        
        return fig
    
    def create_distribution_plots(self, data, column_name, group_col=None):
        """
        Create advanced distribution visualizations
        
        Parameters:
        -----------
        data : pandas.DataFrame or array-like
            Data to visualize
        column_name : str
            Name of the column to plot
        group_col : str, optional
            Column to group by for multiple distributions
            
        Returns:
        --------
        plotly.graph_objects.Figure : Distribution plots
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame({column_name: data})
        
        # Sample if too large
        if len(df) > self.max_points:
            df = self.smart_sample_for_visualization(df, plot_type='histogram')
        
        clean_data = df[column_name].dropna()
        
        if len(clean_data) < 10:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for distribution plots (n={len(clean_data)})",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        if group_col and group_col in df.columns:
            # Multiple distributions by group
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Violin Plot by Group', 'Box Plot by Group',
                    'Histogram by Group', 'Empirical CDF by Group'
                )
            )
            
            groups = df[group_col].unique()[:5]  # Limit to 5 groups
            colors = self.color_palette[:len(groups)]
            
            # Violin plots
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group][column_name].dropna()
                if len(group_data) > 0:
                    fig.add_trace(
                        go.Violin(
                            y=group_data,
                            name=str(group),
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=colors[i],
                            opacity=0.6,
                            x0=str(group)
                        ),
                        row=1, col=1
                    )
            
            # Box plots
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group][column_name].dropna()
                if len(group_data) > 0:
                    fig.add_trace(
                        go.Box(
                            y=group_data,
                            name=str(group),
                            marker_color=colors[i],
                            boxpoints='outliers',
                            x0=str(group)
                        ),
                        row=1, col=2
                    )
            
            # Histograms
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group][column_name].dropna()
                if len(group_data) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=group_data,
                            name=str(group),
                            opacity=0.7,
                            marker_color=colors[i],
                            nbinsx=20
                        ),
                        row=2, col=1
                    )
            
            # Empirical CDF
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group][column_name].dropna()
                if len(group_data) > 0:
                    sorted_data = np.sort(group_data)
                    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_data,
                            y=y_vals,
                            mode='lines',
                            name=str(group),
                            line=dict(color=colors[i], width=2)
                        ),
                        row=2, col=2
                    )
            
        else:
            # Single distribution
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Violin Plot', 'Histogram with KDE',
                    'Empirical CDF', 'Box Plot with Statistics'
                )
            )
            
            # Violin plot
            fig.add_trace(
                go.Violin(
                    y=clean_data,
                    name=column_name,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor='lightblue',
                    opacity=0.6
                ),
                row=1, col=1
            )
            
            # Histogram with KDE
            fig.add_trace(
                go.Histogram(
                    x=clean_data,
                    nbinsx=30,
                    name='Histogram',
                    opacity=0.7,
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # Add KDE curve if possible
            try:
                if len(clean_data) > 10:
                    kde = gaussian_kde(clean_data)
                    x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                    kde_values = kde(x_range)
                    # Scale KDE to match histogram
                    kde_values = kde_values * len(clean_data) * (clean_data.max() - clean_data.min()) / 30
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=kde_values,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=2)
                        ),
                        row=1, col=2
                    )
            except:
                pass
            
            # Empirical CDF
            sorted_data = np.sort(clean_data)
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            fig.add_trace(
                go.Scatter(
                    x=sorted_data,
                    y=y_vals,
                    mode='lines',
                    name='Empirical CDF',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Box plot with statistics
            fig.add_trace(
                go.Box(
                    y=clean_data,
                    name=column_name,
                    marker_color='orange',
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Distribution Analysis: {column_name}",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_relationship_plots(self, df, x_col, y_col, hue_col=None, plot_type='scatter'):
        """
        Create advanced relationship visualizations
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        x_col : str
            X-axis column
        y_col : str
            Y-axis column
        hue_col : str, optional
            Column for color coding
        plot_type : str
            Type of plot ('scatter', 'hexbin', 'regression')
            
        Returns:
        --------
        plotly.graph_objects.Figure : Relationship plot
        """
        # Sample if too large
        if len(df) > self.max_points:
            df_plot = self.smart_sample_for_visualization(df, plot_type='scatter', target_col=hue_col)
        else:
            df_plot = df.copy()
        
        # Remove rows with missing values in key columns
        cols_to_check = [x_col, y_col]
        if hue_col:
            cols_to_check.append(hue_col)
        df_plot = df_plot.dropna(subset=cols_to_check)
        
        if len(df_plot) < 10:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for relationship plots (n={len(df_plot)})",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        if plot_type == 'hexbin' and len(df_plot) > 1000:
            # Hexbin plot for large datasets
            fig = ff.create_hexbin_mapbox(
                data_frame=df_plot,
                lat=y_col,
                lon=x_col,
                nx_hexagon=20,
                opacity=0.7,
                labels={"color": "Count"},
                color_continuous_scale="Blues"
            )
            fig.update_layout(title=f"Hexbin Plot: {x_col} vs {y_col}")
            
        elif plot_type == 'regression':
            # Scatter plot with regression line
            if hue_col:
                fig = px.scatter(
                    df_plot, x=x_col, y=y_col, color=hue_col,
                    trendline="ols",
                    title=f"Regression Plot: {x_col} vs {y_col}",
                    opacity=0.6
                )
            else:
                fig = px.scatter(
                    df_plot, x=x_col, y=y_col,
                    trendline="ols",
                    title=f"Regression Plot: {x_col} vs {y_col}",
                    opacity=0.6
                )
            
            # Add correlation coefficient
            try:
                corr_coef = df_plot[x_col].corr(df_plot[y_col])
                fig.add_annotation(
                    text=f"Correlation: r = {corr_coef:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            except:
                pass
                
        else:
            # Standard scatter plot
            if hue_col:
                fig = px.scatter(
                    df_plot, x=x_col, y=y_col, color=hue_col,
                    title=f"Scatter Plot: {x_col} vs {y_col}",
                    opacity=0.6,
                    hover_data=[hue_col]
                )
            else:
                fig = px.scatter(
                    df_plot, x=x_col, y=y_col,
                    title=f"Scatter Plot: {x_col} vs {y_col}",
                    opacity=0.6
                )
        
        fig.update_layout(height=600)
        return fig
    
    def create_categorical_plots(self, df, cat_col, value_col=None, plot_type='bar'):
        """
        Create categorical data visualizations
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        cat_col : str
            Categorical column
        value_col : str, optional
            Value column for aggregation
        plot_type : str
            Type of plot ('bar', 'pie', 'sunburst')
            
        Returns:
        --------
        plotly.graph_objects.Figure : Categorical plot
        """
        # Sample if too large
        if len(df) > self.max_points:
            df_plot = self.smart_sample_for_visualization(df, plot_type='categorical')
        else:
            df_plot = df.copy()
        
        df_plot = df_plot.dropna(subset=[cat_col])
        
        if len(df_plot) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for categorical plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Limit categories to top N to avoid overcrowding
        top_categories = df_plot[cat_col].value_counts().head(20).index
        df_plot = df_plot[df_plot[cat_col].isin(top_categories)]
        
        if plot_type == 'pie':
            value_counts = df_plot[cat_col].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {cat_col}"
            )
            
        elif plot_type == 'sunburst' and value_col:
            # Sunburst requires hierarchical data
            agg_data = df_plot.groupby(cat_col)[value_col].mean().reset_index()
            fig = px.sunburst(
                agg_data,
                path=[cat_col],
                values=value_col,
                title=f"Sunburst: {cat_col} by {value_col}"
            )
            
        else:
            # Bar plot
            if value_col:
                agg_data = df_plot.groupby(cat_col)[value_col].mean().reset_index()
                fig = px.bar(
                    agg_data, x=cat_col, y=value_col,
                    title=f"Average {value_col} by {cat_col}"
                )
            else:
                value_counts = df_plot[cat_col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Count by {cat_col}",
                    labels={'x': cat_col, 'y': 'Count'}
                )
        
        fig.update_layout(height=500)
        return fig


# Convenience functions for direct use
def create_diagnostic_plots(data, column_name, distribution='norm', max_points=10000):
    """Create statistical diagnostic plots"""
    plotter = BigDataPlotter(max_points=max_points)
    return plotter.create_diagnostic_plots(data, column_name, distribution)

def create_distribution_plots(data, column_name, group_col=None, max_points=10000):
    """Create advanced distribution plots"""
    plotter = BigDataPlotter(max_points=max_points)
    return plotter.create_distribution_plots(data, column_name, group_col)

def create_relationship_plots(df, x_col, y_col, hue_col=None, plot_type='scatter', max_points=10000):
    """Create relationship analysis plots"""
    plotter = BigDataPlotter(max_points=max_points)
    return plotter.create_relationship_plots(df, x_col, y_col, hue_col, plot_type)

def create_categorical_plots(df, cat_col, value_col=None, plot_type='bar', max_points=10000):
    """Create categorical data plots"""
    plotter = BigDataPlotter(max_points=max_points)
    return plotter.create