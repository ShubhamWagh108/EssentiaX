"""
Main Feature Engineering Class
=============================

The core FeatureEngineer class that orchestrates all feature engineering operations.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .base_transformer import BaseFeatureTransformer, FeatureQualityMixin, SmartTransformationMixin
from ..transformers.numerical import NumericalTransformer
from ..transformers.categorical import CategoricalTransformer
from ..selectors.smart_selector import SmartFeatureSelector
from ..utils.metrics import FeatureQualityMetrics

console = Console()

class FeatureEngineer(BaseFeatureTransformer, FeatureQualityMixin, SmartTransformationMixin):
    """
    🚀 Advanced Feature Engineering Pipeline
    
    The most intelligent feature engineering system that automatically:
    - Detects optimal transformations for each feature type
    - Generates new features based on data patterns
    - Selects the most important features using multiple algorithms
    - Provides real-time quality assessment and recommendations
    
    Parameters:
    -----------
    strategy : str, default='auto'
        Feature engineering strategy:
        - 'auto': AI-powered automatic feature engineering
        - 'conservative': Safe transformations only
        - 'aggressive': Maximum feature generation
        - 'custom': User-defined transformations
        
    feature_selection : bool, default=True
        Whether to perform automatic feature selection
        
    max_features : int or float, default=None
        Maximum number of features to keep:
        - int: Exact number of features
        - float: Proportion of features (0.0 to 1.0)
        - None: Keep all generated features
        
    generate_interactions : bool, default=True
        Whether to generate interaction features
        
    generate_polynomials : bool, default=False
        Whether to generate polynomial features
        
    handle_missing : str, default='auto'
        Missing value handling strategy:
        - 'auto': Smart imputation based on data type
        - 'drop': Drop features with missing values
        - 'simple': Simple imputation (mean/mode)
        - 'advanced': Advanced imputation (KNN, iterative)
        
    scale_features : bool, default=True
        Whether to scale numerical features
        
    encode_categoricals : bool, default=True
        Whether to encode categorical features
        
    remove_correlated : bool, default=True
        Whether to remove highly correlated features
        
    correlation_threshold : float, default=0.95
        Correlation threshold for feature removal
        
    random_state : int, default=42
        Random state for reproducibility
        
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
        
    verbose : bool, default=True
        Whether to show progress and results
    """
    
    def __init__(
        self,
        strategy: str = 'auto',
        feature_selection: bool = True,
        max_features: Optional[Union[int, float]] = None,
        generate_interactions: bool = True,
        generate_polynomials: bool = False,
        handle_missing: str = 'auto',
        scale_features: bool = True,
        encode_categoricals: bool = True,
        remove_correlated: bool = True,
        correlation_threshold: float = 0.95,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        
        self.strategy = strategy
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.generate_interactions = generate_interactions
        self.generate_polynomials = generate_polynomials
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.encode_categoricals = encode_categoricals
        self.remove_correlated = remove_correlated
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize components
        self.numerical_transformer_ = None
        self.categorical_transformer_ = None
        self.feature_selector_ = None
        self.quality_metrics_ = FeatureQualityMetrics()
        
        # Results storage
        self.feature_importance_ = None
        self.quality_scores_ = None
        self.transformation_summary_ = None
        self.performance_improvement_ = None
        
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline.
        """
        if self.verbose:
            console.print(Panel.fit(
                "🚀 [bold blue]EssentiaX Feature Engineering Pipeline[/bold blue]\n"
                f"Strategy: {self.strategy} | Features: {X.shape[1]} | Samples: {X.shape[0]}",
                border_style="blue"
            ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            
            # Step 1: Analyze data characteristics
            task1 = progress.add_task("🔍 Analyzing data characteristics...", total=None)
            self._analyze_data(X, y)
            progress.update(task1, completed=True)
            
            # Step 2: Handle missing values
            task2 = progress.add_task("🧹 Handling missing values...", total=None)
            X_clean = self._handle_missing_values(X)
            progress.update(task2, completed=True)
            
            # Step 3: Transform numerical features
            task3 = progress.add_task("🔢 Transforming numerical features...", total=None)
            self._fit_numerical_transformer(X_clean, y)
            progress.update(task3, completed=True)
            
            # Step 4: Transform categorical features
            task4 = progress.add_task("📊 Transforming categorical features...", total=None)
            self._fit_categorical_transformer(X_clean, y)
            progress.update(task4, completed=True)
            
            # Step 5: Feature selection setup
            if self.feature_selection:
                task5 = progress.add_task("🎯 Setting up feature selection...", total=None)
                self._fit_feature_selector(X_clean, y)
                progress.update(task5, completed=True)
        
        # Generate transformation summary
        self._generate_transformation_summary(X)
        
        if self.verbose:
            self._display_results()
            
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted pipeline.
        """
        if self.verbose:
            console.print("🔄 [bold green]Applying feature transformations...[/bold green]")
        
        # Handle missing values
        X_clean = self._handle_missing_values(X)
        
        # Apply transformations
        X_transformed = X_clean.copy()
        
        # Transform numerical features
        if self.numerical_transformer_ is not None:
            numerical_cols = X_clean.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_num_transformed = self.numerical_transformer_.transform(X_clean[numerical_cols])
                X_transformed = X_transformed.drop(columns=numerical_cols)
                X_transformed = pd.concat([X_transformed, X_num_transformed], axis=1)
        
        # Transform categorical features
        if self.categorical_transformer_ is not None:
            categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                X_cat_transformed = self.categorical_transformer_.transform(X_clean[categorical_cols])
                X_transformed = X_transformed.drop(columns=categorical_cols)
                X_transformed = pd.concat([X_transformed, X_cat_transformed], axis=1)
        
        # Apply feature selection
        if self.feature_selector_ is not None:
            X_transformed = self.feature_selector_.transform(X_transformed)
        
        return X_transformed
    
    def _analyze_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyze data characteristics and determine optimal strategy."""
        self.data_analysis_ = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'numerical_features': list(X.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(X.select_dtypes(include=['object', 'category']).columns),
            'missing_values': X.isnull().sum().sum(),
            'missing_ratio': X.isnull().sum().sum() / (len(X) * len(X.columns)),
            'memory_usage': X.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Determine problem type
        if y is not None:
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                self.problem_type_ = 'classification'
            else:
                self.problem_type_ = 'regression'
        else:
            self.problem_type_ = 'unsupervised'
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        if self.handle_missing == 'drop':
            return X.dropna()
        elif self.handle_missing == 'simple':
            X_filled = X.copy()
            for col in X.columns:
                if X[col].dtype in [np.number]:
                    X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
                else:
                    X_filled[col] = X_filled[col].fillna(X_filled[col].mode().iloc[0] if not X_filled[col].mode().empty else 'missing')
            return X_filled
        else:  # auto or advanced
            # For now, use simple strategy - can be enhanced later
            X_filled = X.copy()
            for col in X.columns:
                if X[col].dtype in [np.number]:
                    X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
                else:
                    X_filled[col] = X_filled[col].fillna(X_filled[col].mode().iloc[0] if not X_filled[col].mode().empty else 'missing')
            return X_filled
    
    def _fit_numerical_transformer(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit numerical feature transformer."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self.numerical_transformer_ = NumericalTransformer(
                strategy=self.strategy,
                scale_features=self.scale_features,
                generate_interactions=self.generate_interactions,
                generate_polynomials=self.generate_polynomials,
                verbose=False
            )
            self.numerical_transformer_.fit(X[numerical_cols], y)
    
    def _fit_categorical_transformer(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit categorical feature transformer."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.categorical_transformer_ = CategoricalTransformer(
                strategy=self.strategy,
                encode_categoricals=self.encode_categoricals,
                verbose=False
            )
            self.categorical_transformer_.fit(X[categorical_cols], y)
    
    def _fit_feature_selector(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit feature selector."""
        if self.feature_selection and y is not None:
            self.feature_selector_ = SmartFeatureSelector(
                max_features=self.max_features,
                remove_correlated=self.remove_correlated,
                correlation_threshold=self.correlation_threshold,
                random_state=self.random_state,
                verbose=False
            )
            # Transform data first to get all features
            X_temp = self._transform_without_selection(X)
            self.feature_selector_.fit(X_temp, y)
    
    def _transform_without_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data without feature selection."""
        X_clean = self._handle_missing_values(X)
        X_transformed = X_clean.copy()
        
        # Transform numerical features
        if self.numerical_transformer_ is not None:
            numerical_cols = X_clean.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_num_transformed = self.numerical_transformer_.transform(X_clean[numerical_cols])
                X_transformed = X_transformed.drop(columns=numerical_cols)
                X_transformed = pd.concat([X_transformed, X_num_transformed], axis=1)
        
        # Transform categorical features
        if self.categorical_transformer_ is not None:
            categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                X_cat_transformed = self.categorical_transformer_.transform(X_clean[categorical_cols])
                X_transformed = X_transformed.drop(columns=categorical_cols)
                X_transformed = pd.concat([X_transformed, X_cat_transformed], axis=1)
        
        return X_transformed
    
    def _generate_transformation_summary(self, X: pd.DataFrame):
        """Generate summary of transformations applied."""
        self.transformation_summary_ = {
            'original_features': len(X.columns),
            'final_features': len(self.feature_names_out_) if self.feature_names_out_ else 0,
            'numerical_transformations': [],
            'categorical_transformations': [],
            'feature_selection_applied': self.feature_selection,
            'missing_values_handled': self.handle_missing != 'none'
        }
        
        # Add transformation details from components
        if self.numerical_transformer_:
            self.transformation_summary_['numerical_transformations'] = self.numerical_transformer_.get_transformation_log()
        if self.categorical_transformer_:
            self.transformation_summary_['categorical_transformations'] = self.categorical_transformer_.get_transformation_log()
    
    def _display_results(self):
        """Display feature engineering results."""
        # Create results table
        table = Table(title="🎯 Feature Engineering Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Original Features", str(self.transformation_summary_['original_features']))
        table.add_row("Final Features", str(self.transformation_summary_['final_features']))
        table.add_row("Feature Selection", "✅" if self.feature_selection else "❌")
        table.add_row("Missing Values Handled", "✅" if self.transformation_summary_['missing_values_handled'] else "❌")
        table.add_row("Strategy", self.strategy.title())
        
        console.print(table)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.
        
        Returns:
        --------
        importance_df : pd.DataFrame or None
            Feature importance scores
        """
        if self.feature_selector_ and hasattr(self.feature_selector_, 'feature_importance_'):
            return self.feature_selector_.feature_importance_
        return None
    
    def get_quality_scores(self) -> Optional[Dict[str, float]]:
        """
        Get feature quality scores.
        
        Returns:
        --------
        quality_scores : dict or None
            Feature quality scores
        """
        return self.quality_scores_
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all transformations applied.
        
        Returns:
        --------
        summary : dict
            Transformation summary
        """
        return self.transformation_summary_ or {}
    
    def evaluate_improvement(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Evaluate performance improvement from feature engineering.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Original features
        y : pd.Series
            Target variable
        cv : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        improvement : dict
            Performance improvement metrics
        """
        if self.verbose:
            console.print("📊 [bold yellow]Evaluating feature engineering improvement...[/bold yellow]")
        
        # Choose appropriate model
        if self.problem_type_ == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs)
            scoring = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs)
            scoring = 'r2'
        
        # Evaluate original features
        original_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        
        # Evaluate transformed features
        X_transformed = self.transform(X)
        transformed_scores = cross_val_score(model, X_transformed, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
        
        improvement = {
            'original_score': original_scores.mean(),
            'original_std': original_scores.std(),
            'transformed_score': transformed_scores.mean(),
            'transformed_std': transformed_scores.std(),
            'improvement': transformed_scores.mean() - original_scores.mean(),
            'improvement_pct': ((transformed_scores.mean() - original_scores.mean()) / original_scores.mean()) * 100
        }
        
        self.performance_improvement_ = improvement
        
        if self.verbose:
            console.print(f"📈 Performance improvement: {improvement['improvement_pct']:.2f}%")
            console.print(f"   Original: {improvement['original_score']:.4f} ± {improvement['original_std']:.4f}")
            console.print(f"   Transformed: {improvement['transformed_score']:.4f} ± {improvement['transformed_std']:.4f}")
        
        return improvement