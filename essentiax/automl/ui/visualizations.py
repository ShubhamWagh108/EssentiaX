"""
EssentiaX AutoML - Advanced Visualizations
==========================================

Comprehensive visualization capabilities for AutoML training and results.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AutoMLVisualizer:
    """
    📊 Advanced Visualization Engine for AutoML
    
    Provides comprehensive visualization capabilities including:
    - Training progress and learning curves
    - Model performance comparisons
    - Hyperparameter optimization landscapes
    - Feature importance and SHAP visualizations
    - Prediction analysis and error diagnostics
    - Interactive dashboards and reports
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the AutoML visualizer.
        
        Parameters:
        -----------
        style : str, default='seaborn-v0_8'
            Matplotlib style to use
        figsize : tuple, default=(12, 8)
            Default figure size for plots
        """
        self.style = style
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        
        # Set plotting style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        # Configure seaborn
        sns.set_palette(self.color_palette)
        
    def plot_training_progress(self, progress_data: Dict[str, Any], 
                             save_path: Optional[str] = None) -> Figure:
        """
        Plot training progress across different phases.
        
        Parameters:
        -----------
        progress_data : dict
            Dictionary containing progress information for each phase
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extract data
        phases = list(progress_data.keys())
        progress_values = [data.get('progress', 0) for data in progress_data.values()]
        elapsed_times = [data.get('elapsed_time', 0) for data in progress_data.values()]
        
        # Progress bar chart
        bars = ax1.barh(phases, progress_values, color=self.color_palette[:len(phases)])
        ax1.set_xlabel('Progress (%)')
        ax1.set_title('Training Phase Progress')
        ax1.set_xlim(0, 100)
        
        # Add progress labels
        for i, (bar, progress) in enumerate(zip(bars, progress_values)):
            ax1.text(progress + 2, i, f'{progress:.1f}%', 
                    va='center', fontweight='bold')
        
        # Time consumption chart
        ax2.pie(elapsed_times, labels=phases, autopct='%1.1f%%', 
               colors=self.color_palette[:len(phases)])
        ax2.set_title('Time Distribution Across Phases')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, model_results: List[Dict[str, Any]], 
                            metric: str = 'score', save_path: Optional[str] = None) -> Figure:
        """
        Plot model performance comparison.
        
        Parameters:
        -----------
        model_results : list
            List of model result dictionaries
        metric : str, default='score'
            Metric to compare models on
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        if not model_results:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No model results available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Sort models by performance
        sorted_results = sorted(model_results, key=lambda x: x.get(metric, 0), reverse=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Performance comparison
        model_names = [result['name'] for result in sorted_results[:10]]
        scores = [result.get(metric, 0) for result in sorted_results[:10]]
        training_times = [result.get('training_time', 0) for result in sorted_results[:10]]
        
        bars = ax1.bar(range(len(model_names)), scores, 
                      color=self.color_palette[:len(model_names)])
        ax1.set_xlabel('Models')
        ax1.set_ylabel(f'{metric.title()}')
        ax1.set_title(f'Model Performance Comparison ({metric.title()})')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add score labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Performance vs Training Time scatter
        ax2.scatter(training_times, scores, s=100, alpha=0.7, 
                   c=range(len(scores)), cmap='viridis')
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel(f'{metric.title()}')
        ax2.set_title('Performance vs Training Time')
        
        # Add model name annotations
        for i, (name, time, score) in enumerate(zip(model_names, training_times, scores)):
            ax2.annotate(name, (time, score), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_hyperparameter_optimization(self, hyperopt_history: List[Dict[str, Any]], 
                                       save_path: Optional[str] = None) -> Figure:
        """
        Plot hyperparameter optimization progress.
        
        Parameters:
        -----------
        hyperopt_history : list
            List of hyperparameter optimization results
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        if not hyperopt_history:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No hyperparameter optimization data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Extract data
        iterations = [result.get('iteration', i) for i, result in enumerate(hyperopt_history)]
        scores = [result.get('score', 0) for result in hyperopt_history]
        
        # Optimization progress
        ax1.plot(iterations, scores, 'o-', linewidth=2, markersize=6, alpha=0.7)
        
        # Highlight best scores
        best_scores = []
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        ax1.plot(iterations, best_scores, 'r-', linewidth=3, alpha=0.8, label='Best Score')
        ax1.set_xlabel('Optimization Iteration')
        ax1.set_ylabel('Score')
        ax1.set_title('Hyperparameter Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Score distribution
        ax2.hist(scores, bins=min(20, len(scores)//2), alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores):.4f}')
        ax2.axvline(np.max(scores), color='green', linestyle='--', 
                   label=f'Best: {np.max(scores):.4f}')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 20, save_path: Optional[str] = None) -> Figure:
        """
        Plot feature importance analysis.
        
        Parameters:
        -----------
        feature_importance : dict
            Dictionary mapping feature names to importance scores
        top_n : int, default=20
            Number of top features to display
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        if not feature_importance:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No feature importance data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Horizontal bar chart
        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]
        
        bars = ax1.barh(range(len(feature_names)), importance_values, 
                       color=self.color_palette[0])
        ax1.set_xlabel('Importance Score')
        ax1.set_ylabel('Features')
        ax1.set_title(f'Top {top_n} Feature Importance')
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        
        # Add importance labels
        for bar, importance in zip(bars, importance_values):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        # Cumulative importance
        cumulative_importance = np.cumsum(importance_values) / np.sum(importance_values) * 100
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                'o-', linewidth=2, markersize=6)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Threshold')
        ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (%)')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float], 
                           train_sizes: Optional[List[int]] = None, 
                           save_path: Optional[str] = None) -> Figure:
        """
        Plot learning curves showing training and validation performance.
        
        Parameters:
        -----------
        train_scores : list
            Training scores
        val_scores : list
            Validation scores
        train_sizes : list, optional
            Training set sizes
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if train_sizes is None:
            train_sizes = range(1, len(train_scores) + 1)
        
        ax.plot(train_sizes, train_scores, 'o-', linewidth=2, markersize=6, 
               label='Training Score', color=self.color_palette[0])
        ax.plot(train_sizes, val_scores, 'o-', linewidth=2, markersize=6, 
               label='Validation Score', color=self.color_palette[1])
        
        ax.set_xlabel('Training Set Size' if train_sizes != range(1, len(train_scores) + 1) else 'Iteration')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fill between curves to show gap
        ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> Figure:
        """
        Plot confusion matrix for classification results.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list, optional
            Names of classes
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       class_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None) -> Figure:
        """
        Plot ROC curves for classification results.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        class_names : list, optional
            Names of classes
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Handle binary and multiclass cases
        if y_proba.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, 
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')
        else:  # Multiclass
            for i in range(y_proba.shape[1]):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_name = class_names[i] if class_names else f'Class {i}'
                ax.plot(fpr, tpr, linewidth=2, 
                       label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, automl_results: Dict[str, Any]) -> go.Figure:
        """
        Create an interactive Plotly dashboard with multiple visualizations.
        
        Parameters:
        -----------
        automl_results : dict
            Dictionary containing AutoML results and metrics
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance', 'Training Progress', 
                          'Feature Importance', 'Hyperparameter Optimization'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Model performance (if available)
        if 'model_results' in automl_results:
            model_results = automl_results['model_results']
            model_names = [r['name'] for r in model_results[:10]]
            scores = [r.get('score', 0) for r in model_results[:10]]
            
            fig.add_trace(
                go.Bar(x=model_names, y=scores, name='Model Scores',
                      marker_color='lightblue'),
                row=1, col=1
            )
        
        # Training progress (if available)
        if 'progress_data' in automl_results:
            progress_data = automl_results['progress_data']
            phases = list(progress_data.keys())
            progress_values = [data.get('progress', 0) for data in progress_data.values()]
            
            fig.add_trace(
                go.Scatter(x=phases, y=progress_values, mode='lines+markers',
                          name='Training Progress', line=dict(color='green')),
                row=1, col=2
            )
        
        # Feature importance (if available)
        if 'feature_importance' in automl_results:
            feature_importance = automl_results['feature_importance']
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:15]
            feature_names = [item[0] for item in sorted_features]
            importance_values = [item[1] for item in sorted_features]
            
            fig.add_trace(
                go.Bar(x=importance_values, y=feature_names, 
                      orientation='h', name='Feature Importance',
                      marker_color='orange'),
                row=2, col=1
            )
        
        # Hyperparameter optimization (if available)
        if 'hyperopt_history' in automl_results:
            hyperopt_history = automl_results['hyperopt_history']
            iterations = [r.get('iteration', i) for i, r in enumerate(hyperopt_history)]
            scores = [r.get('score', 0) for r in hyperopt_history]
            
            fig.add_trace(
                go.Scatter(x=iterations, y=scores, mode='lines+markers',
                          name='Optimization Progress', line=dict(color='red')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="AutoML Interactive Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def save_all_plots(self, automl_results: Dict[str, Any], output_dir: str = "./automl_plots"):
        """
        Save all available plots to files.
        
        Parameters:
        -----------
        automl_results : dict
            Dictionary containing AutoML results
        output_dir : str, default="./automl_plots"
            Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Training progress
        if 'progress_data' in automl_results:
            self.plot_training_progress(
                automl_results['progress_data'],
                save_path=f"{output_dir}/training_progress.png"
            )
        
        # Model comparison
        if 'model_results' in automl_results:
            self.plot_model_comparison(
                automl_results['model_results'],
                save_path=f"{output_dir}/model_comparison.png"
            )
        
        # Hyperparameter optimization
        if 'hyperopt_history' in automl_results:
            self.plot_hyperparameter_optimization(
                automl_results['hyperopt_history'],
                save_path=f"{output_dir}/hyperopt_progress.png"
            )
        
        # Feature importance
        if 'feature_importance' in automl_results:
            self.plot_feature_importance(
                automl_results['feature_importance'],
                save_path=f"{output_dir}/feature_importance.png"
            )
        
        # Interactive dashboard
        if any(key in automl_results for key in ['model_results', 'progress_data', 
                                                'feature_importance', 'hyperopt_history']):
            interactive_fig = self.create_interactive_dashboard(automl_results)
            interactive_fig.write_html(f"{output_dir}/interactive_dashboard.html")
        
        print(f"📊 All plots saved to {output_dir}/")