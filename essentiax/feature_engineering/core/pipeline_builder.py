"""
Pipeline Builder
===============

Automated pipeline construction for feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings

from .base_transformer import BaseFeatureTransformer
from ..transformers.numerical import NumericalTransformer
from ..transformers.categorical import CategoricalTransformer
from ..selectors.smart_selector import SmartFeatureSelector

class PipelineBuilder:
    """
    🔧 Automated Feature Engineering Pipeline Builder
    
    Intelligently constructs feature engineering pipelines based on:
    - Data characteristics analysis
    - Problem type detection
    - Performance requirements
    - Resource constraints
    
    Parameters:
    -----------
    strategy : str, default='auto'
        Pipeline building strategy:
        - 'auto': AI-powered automatic pipeline
        - 'fast': Quick pipeline for prototyping
        - 'comprehensive': Maximum feature engineering
        - 'memory_efficient': Optimized for large datasets
        
    max_pipeline_steps : int, default=10
        Maximum number of pipeline steps
        
    enable_feature_selection : bool, default=True
        Whether to include feature selection
        
    enable_scaling : bool, default=True
        Whether to include feature scaling
        
    enable_encoding : bool, default=True
        Whether to include categorical encoding
        
    optimize_for : str, default='accuracy'
        Optimization target:
        - 'accuracy': Optimize for model accuracy
        - 'speed': Optimize for processing speed
        - 'memory': Optimize for memory usage
        - 'interpretability': Optimize for interpretability
        
    verbose : bool, default=True
        Whether to show pipeline construction details
    """
    
    def __init__(
        self,
        strategy: str = 'auto',
        max_pipeline_steps: int = 10,
        enable_feature_selection: bool = True,
        enable_scaling: bool = True,
        enable_encoding: bool = True,
        optimize_for: str = 'accuracy',
        verbose: bool = True
    ):
        self.strategy = strategy
        self.max_pipeline_steps = max_pipeline_steps
        self.enable_feature_selection = enable_feature_selection
        self.enable_scaling = enable_scaling
        self.enable_encoding = enable_encoding
        self.optimize_for = optimize_for
        self.verbose = verbose
        
        # Pipeline components
        self.pipeline_ = None
        self.pipeline_steps_ = []
        self.data_analysis_ = {}
        
    def build_pipeline(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        problem_type: Optional[str] = None
    ) -> Pipeline:
        """
        Build an optimized feature engineering pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
        problem_type : str, optional
            Problem type ('classification', 'regression', 'unsupervised')
            
        Returns:
        --------
        pipeline : Pipeline
            Constructed feature engineering pipeline
        """
        
        if self.verbose:
            print("🔧 Building feature engineering pipeline...")
        
        # Analyze data characteristics
        self._analyze_data(X, y, problem_type)
        
        # Determine pipeline strategy
        pipeline_config = self._determine_pipeline_config()
        
        # Build pipeline steps
        self._build_pipeline_steps(pipeline_config)
        
        # Create sklearn pipeline
        self.pipeline_ = Pipeline(self.pipeline_steps_)
        
        if self.verbose:
            self._display_pipeline_summary()
        
        return self.pipeline_
    
    def _analyze_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None, problem_type: Optional[str] = None):
        """Analyze data characteristics for pipeline optimization."""
        
        self.data_analysis_ = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / 1024**2,
            'numerical_features': list(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(X.select_dtypes(include=['object', 'category']).columns),
            'missing_values_total': X.isnull().sum().sum(),
            'missing_ratio': X.isnull().sum().sum() / (len(X) * len(X.columns)),
            'high_cardinality_features': [],
            'skewed_features': [],
            'outlier_features': []
        }
        
        # Analyze numerical features
        for col in self.data_analysis_['numerical_features']:
            series = X[col]
            
            # Check skewness
            if abs(series.skew()) > 1.5:
                self.data_analysis_['skewed_features'].append(col)
            
            # Check outliers
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(series) * 0.1:
                self.data_analysis_['outlier_features'].append(col)
        
        # Analyze categorical features
        for col in self.data_analysis_['categorical_features']:
            if X[col].nunique() > 50:
                self.data_analysis_['high_cardinality_features'].append(col)
        
        # Determine problem type
        if problem_type:
            self.data_analysis_['problem_type'] = problem_type
        elif y is not None:
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                self.data_analysis_['problem_type'] = 'classification'
            else:
                self.data_analysis_['problem_type'] = 'regression'
        else:
            self.data_analysis_['problem_type'] = 'unsupervised'
        
        # Performance considerations
        self.data_analysis_['is_large_dataset'] = len(X) > 100000
        self.data_analysis_['is_wide_dataset'] = len(X.columns) > 1000
        self.data_analysis_['is_memory_constrained'] = self.data_analysis_['memory_usage_mb'] > 1000
    
    def _determine_pipeline_config(self) -> Dict[str, Any]:
        """Determine optimal pipeline configuration."""
        
        config = {
            'numerical_transformer_config': {},
            'categorical_transformer_config': {},
            'feature_selector_config': {},
            'include_numerical_transformer': len(self.data_analysis_['numerical_features']) > 0,
            'include_categorical_transformer': len(self.data_analysis_['categorical_features']) > 0,
            'include_feature_selector': self.enable_feature_selection
        }
        
        # Configure based on strategy
        if self.strategy == 'fast':
            config.update({
                'numerical_transformer_config': {
                    'strategy': 'conservative',
                    'generate_interactions': False,
                    'generate_polynomials': False,
                    'transform_skewed': False
                },
                'categorical_transformer_config': {
                    'strategy': 'conservative',
                    'encoding_method': 'auto',
                    'handle_rare_categories': False
                },
                'feature_selector_config': {
                    'selection_methods': ['statistical'],
                    'max_features': 0.8
                }
            })
        
        elif self.strategy == 'comprehensive':
            config.update({
                'numerical_transformer_config': {
                    'strategy': 'aggressive',
                    'generate_interactions': True,
                    'generate_polynomials': True,
                    'transform_skewed': True
                },
                'categorical_transformer_config': {
                    'strategy': 'aggressive',
                    'encoding_method': 'auto',
                    'handle_rare_categories': True,
                    'text_features': True
                },
                'feature_selector_config': {
                    'selection_methods': ['statistical', 'model_based', 'recursive'],
                    'ensemble_voting': True
                }
            })
        
        elif self.strategy == 'memory_efficient':
            config.update({
                'numerical_transformer_config': {
                    'strategy': 'conservative',
                    'generate_interactions': False,
                    'generate_polynomials': False,
                    'scaling_method': 'robust'  # More memory efficient
                },
                'categorical_transformer_config': {
                    'strategy': 'conservative',
                    'encoding_method': 'frequency',  # More memory efficient than one-hot
                    'max_categories': 10
                },
                'feature_selector_config': {
                    'selection_methods': ['statistical'],
                    'max_features': 0.5  # Aggressive feature reduction
                }
            })
        
        else:  # auto strategy
            # Adaptive configuration based on data characteristics
            if self.data_analysis_['is_large_dataset']:
                # Large dataset - prioritize speed and memory
                config['numerical_transformer_config'].update({
                    'generate_interactions': False,
                    'generate_polynomials': False
                })
                config['feature_selector_config']['max_features'] = 0.7
            
            if self.data_analysis_['is_wide_dataset']:
                # Wide dataset - aggressive feature selection
                config['feature_selector_config']['max_features'] = 0.3
                config['include_feature_selector'] = True
            
            if len(self.data_analysis_['skewed_features']) > 0:
                config['numerical_transformer_config']['transform_skewed'] = True
            
            if len(self.data_analysis_['high_cardinality_features']) > 0:
                config['categorical_transformer_config']['encoding_method'] = 'target'
        
        # Optimization-specific adjustments
        if self.optimize_for == 'speed':
            config['numerical_transformer_config']['generate_polynomials'] = False
            config['categorical_transformer_config']['text_features'] = False
            config['feature_selector_config']['selection_methods'] = ['statistical']
        
        elif self.optimize_for == 'memory':
            config['numerical_transformer_config']['generate_interactions'] = False
            config['categorical_transformer_config']['encoding_method'] = 'frequency'
            config['feature_selector_config']['max_features'] = 0.5
        
        elif self.optimize_for == 'interpretability':
            config['numerical_transformer_config']['transform_skewed'] = False
            config['categorical_transformer_config']['encoding_method'] = 'onehot'
            config['feature_selector_config']['selection_methods'] = ['statistical', 'model_based']
        
        return config
    
    def _build_pipeline_steps(self, config: Dict[str, Any]):
        """Build pipeline steps based on configuration."""
        
        self.pipeline_steps_ = []
        
        # Numerical transformer
        if config['include_numerical_transformer'] and self.enable_scaling:
            numerical_transformer = NumericalTransformer(
                verbose=False,
                **config['numerical_transformer_config']
            )
            self.pipeline_steps_.append(('numerical_transformer', numerical_transformer))
        
        # Categorical transformer
        if config['include_categorical_transformer'] and self.enable_encoding:
            categorical_transformer = CategoricalTransformer(
                verbose=False,
                **config['categorical_transformer_config']
            )
            self.pipeline_steps_.append(('categorical_transformer', categorical_transformer))
        
        # Feature selector
        if config['include_feature_selector'] and self.enable_feature_selection:
            feature_selector = SmartFeatureSelector(
                verbose=False,
                **config['feature_selector_config']
            )
            self.pipeline_steps_.append(('feature_selector', feature_selector))
        
        # Ensure we don't exceed max steps
        if len(self.pipeline_steps_) > self.max_pipeline_steps:
            self.pipeline_steps_ = self.pipeline_steps_[:self.max_pipeline_steps]
    
    def _display_pipeline_summary(self):
        """Display pipeline construction summary."""
        
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Pipeline summary
        table = Table(title="🔧 Pipeline Construction Summary", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Configuration", style="yellow")
        
        for step_name, transformer in self.pipeline_steps_:
            status = "✅ Included"
            config_info = f"{transformer.__class__.__name__}"
            if hasattr(transformer, 'strategy'):
                config_info += f" ({transformer.strategy})"
            
            table.add_row(step_name.replace('_', ' ').title(), status, config_info)
        
        console.print(table)
        
        # Data analysis summary
        analysis_info = [
            f"📊 Dataset: {self.data_analysis_['n_samples']:,} samples × {self.data_analysis_['n_features']} features",
            f"💾 Memory: {self.data_analysis_['memory_usage_mb']:.1f} MB",
            f"🎯 Problem: {self.data_analysis_['problem_type'].title()}",
            f"⚡ Strategy: {self.strategy.title()}",
            f"🎨 Optimized for: {self.optimize_for.title()}"
        ]
        
        console.print(Panel(
            "\n".join(analysis_info),
            title="📈 Data Analysis",
            border_style="blue"
        ))
    
    def get_pipeline(self) -> Optional[Pipeline]:
        """Get the constructed pipeline."""
        return self.pipeline_
    
    def get_pipeline_steps(self) -> List[Tuple[str, Any]]:
        """Get pipeline steps."""
        return self.pipeline_steps_
    
    def get_data_analysis(self) -> Dict[str, Any]:
        """Get data analysis results."""
        return self.data_analysis_
    
    def estimate_pipeline_performance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate pipeline performance characteristics.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        performance_estimate : dict
            Estimated performance metrics
        """
        
        estimate = {
            'estimated_transform_time_seconds': 0.0,
            'estimated_memory_usage_mb': 0.0,
            'estimated_output_features': 0,
            'complexity_score': 0.0
        }
        
        # Base processing time (per 1000 samples)
        base_time_per_1k = 0.1
        n_samples_k = len(X) / 1000
        
        # Estimate for each pipeline step
        for step_name, transformer in self.pipeline_steps_:
            if 'numerical' in step_name:
                # Numerical transformer
                time_multiplier = 1.0
                if hasattr(transformer, 'generate_polynomials') and transformer.generate_polynomials:
                    time_multiplier *= 2.0
                if hasattr(transformer, 'generate_interactions') and transformer.generate_interactions:
                    time_multiplier *= 1.5
                
                estimate['estimated_transform_time_seconds'] += base_time_per_1k * n_samples_k * time_multiplier
                estimate['complexity_score'] += time_multiplier
                
            elif 'categorical' in step_name:
                # Categorical transformer
                time_multiplier = 1.0
                if hasattr(transformer, 'text_features') and transformer.text_features:
                    time_multiplier *= 3.0
                
                estimate['estimated_transform_time_seconds'] += base_time_per_1k * n_samples_k * time_multiplier
                estimate['complexity_score'] += time_multiplier
                
            elif 'selector' in step_name:
                # Feature selector
                time_multiplier = 2.0  # Selection is more expensive
                if hasattr(transformer, 'selection_methods'):
                    time_multiplier *= len(transformer.selection_methods)
                
                estimate['estimated_transform_time_seconds'] += base_time_per_1k * n_samples_k * time_multiplier
                estimate['complexity_score'] += time_multiplier
        
        # Estimate output features
        n_numerical = len(self.data_analysis_['numerical_features'])
        n_categorical = len(self.data_analysis_['categorical_features'])
        
        estimated_features = n_numerical + n_categorical
        
        # Adjust for transformations
        for step_name, transformer in self.pipeline_steps_:
            if 'numerical' in step_name:
                if hasattr(transformer, 'generate_polynomials') and transformer.generate_polynomials:
                    estimated_features += n_numerical * 2  # Rough estimate
                if hasattr(transformer, 'generate_interactions') and transformer.generate_interactions:
                    estimated_features += min(n_numerical * (n_numerical - 1) // 2, 20)  # Cap interactions
            
            elif 'categorical' in step_name:
                # One-hot encoding can significantly increase features
                for col in self.data_analysis_['categorical_features']:
                    if col in self.data_analysis_['high_cardinality_features']:
                        estimated_features += 1  # Target/frequency encoding
                    else:
                        estimated_features += min(X[col].nunique(), 20)  # One-hot encoding
            
            elif 'selector' in step_name:
                # Feature selection reduces features
                if hasattr(transformer, 'max_features'):
                    if isinstance(transformer.max_features, float):
                        estimated_features = int(estimated_features * transformer.max_features)
                    elif isinstance(transformer.max_features, int):
                        estimated_features = min(estimated_features, transformer.max_features)
        
        estimate['estimated_output_features'] = max(1, estimated_features)
        
        # Estimate memory usage
        base_memory_mb = X.memory_usage(deep=True).sum() / 1024**2
        memory_multiplier = estimate['estimated_output_features'] / len(X.columns)
        estimate['estimated_memory_usage_mb'] = base_memory_mb * memory_multiplier
        
        return estimate