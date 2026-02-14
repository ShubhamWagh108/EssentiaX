"""
Advanced Numerical Feature Transformer
=====================================

Intelligent transformations for numerical features with AI-powered optimization.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import warnings

from ..core.base_transformer import BaseFeatureTransformer, SmartTransformationMixin

class NumericalTransformer(BaseFeatureTransformer, SmartTransformationMixin):
    """
    🔢 Advanced Numerical Feature Transformer
    
    Intelligently transforms numerical features using:
    - Smart scaling method selection
    - Advanced mathematical transformations
    - Polynomial feature generation
    - Interaction feature creation
    - Outlier-aware processing
    
    Parameters:
    -----------
    strategy : str, default='auto'
        Transformation strategy:
        - 'auto': AI-powered automatic selection
        - 'conservative': Safe transformations only
        - 'aggressive': Maximum transformations
        
    scale_features : bool, default=True
        Whether to scale numerical features
        
    scaling_method : str, default='auto'
        Scaling method:
        - 'auto': Automatically select best method
        - 'standard': StandardScaler
        - 'robust': RobustScaler (outlier-resistant)
        - 'minmax': MinMaxScaler
        - 'none': No scaling
        
    transform_skewed : bool, default=True
        Whether to transform skewed distributions
        
    skewness_threshold : float, default=1.0
        Threshold for skewness transformation
        
    generate_interactions : bool, default=False
        Whether to generate interaction features
        
    max_interactions : int, default=10
        Maximum number of interaction features to generate
        
    generate_polynomials : bool, default=False
        Whether to generate polynomial features
        
    polynomial_degree : int, default=2
        Degree for polynomial features
        
    handle_outliers : bool, default=True
        Whether to handle outliers in transformations
        
    outlier_method : str, default='iqr'
        Outlier detection method:
        - 'iqr': Interquartile range
        - 'zscore': Z-score method
        - 'isolation': Isolation forest
        
    verbose : bool, default=True
        Whether to show transformation details
    """
    
    def __init__(
        self,
        strategy: str = 'auto',
        scale_features: bool = True,
        scaling_method: str = 'auto',
        transform_skewed: bool = True,
        skewness_threshold: float = 1.0,
        generate_interactions: bool = False,
        max_interactions: int = 10,
        generate_polynomials: bool = False,
        polynomial_degree: int = 2,
        handle_outliers: bool = True,
        outlier_method: str = 'iqr',
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        
        self.strategy = strategy
        self.scale_features = scale_features
        self.scaling_method = scaling_method
        self.transform_skewed = transform_skewed
        self.skewness_threshold = skewness_threshold
        self.generate_interactions = generate_interactions
        self.max_interactions = max_interactions
        self.generate_polynomials = generate_polynomials
        self.polynomial_degree = polynomial_degree
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        
        # Fitted components
        self.scalers_ = {}
        self.transformers_ = {}
        self.polynomial_features_ = None
        self.interaction_features_ = []
        self.outlier_bounds_ = {}
        self.feature_stats_ = {}
        
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NumericalTransformer':
        """Fit the numerical transformer."""
        
        # Analyze each feature
        for col in X.columns:
            self._analyze_feature(X[col], col)
        
        # Fit scalers
        if self.scale_features:
            self._fit_scalers(X)
        
        # Fit transformers for skewed features
        if self.transform_skewed:
            self._fit_skewness_transformers(X)
        
        # Fit polynomial features
        if self.generate_polynomials:
            self._fit_polynomial_features(X)
        
        # Identify interaction features
        if self.generate_interactions:
            self._identify_interactions(X, y)
        
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the numerical features."""
        
        X_transformed = X.copy()
        
        # Handle outliers
        if self.handle_outliers:
            X_transformed = self._handle_outliers(X_transformed)
        
        # Apply skewness transformations
        if self.transform_skewed:
            X_transformed = self._apply_skewness_transformations(X_transformed)
        
        # Apply scaling
        if self.scale_features:
            X_transformed = self._apply_scaling(X_transformed)
        
        # Generate polynomial features
        if self.generate_polynomials and self.polynomial_features_:
            X_poly = self._generate_polynomial_features(X_transformed)
            X_transformed = pd.concat([X_transformed, X_poly], axis=1)
        
        # Generate interaction features
        if self.generate_interactions:
            X_interactions = self._generate_interaction_features(X_transformed)
            X_transformed = pd.concat([X_transformed, X_interactions], axis=1)
        
        return X_transformed
    
    def _analyze_feature(self, series: pd.Series, col_name: str):
        """Analyze a single numerical feature."""
        
        stats_dict = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'missing_ratio': series.isnull().sum() / len(series),
            'zero_ratio': (series == 0).sum() / len(series),
            'unique_ratio': series.nunique() / len(series)
        }
        
        # Detect outliers
        if self.outlier_method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            self.outlier_bounds_[col_name] = (lower_bound, upper_bound)
        elif self.outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = (z_scores > 3).sum()
            self.outlier_bounds_[col_name] = (-3, 3)  # Z-score bounds
        else:
            outliers = 0
            
        stats_dict['outlier_ratio'] = outliers / len(series)
        
        self.feature_stats_[col_name] = stats_dict
        
        # Log analysis
        self._log_transformation('analyze_feature', {
            'feature': col_name,
            'stats': stats_dict
        })
    
    def _fit_scalers(self, X: pd.DataFrame):
        """Fit appropriate scalers for each feature."""
        
        for col in X.columns:
            stats = self.feature_stats_[col]
            
            # Choose scaler based on data characteristics
            if self.scaling_method == 'auto':
                if stats['outlier_ratio'] > 0.1:  # Many outliers
                    scaler = RobustScaler()
                elif stats['min'] >= 0 and stats['max'] <= 1:  # Already normalized
                    scaler = None
                elif abs(stats['skewness']) > 2:  # Highly skewed
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
            elif self.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.scaling_method == 'robust':
                scaler = RobustScaler()
            elif self.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = None
            
            if scaler is not None:
                scaler.fit(X[[col]])
                self.scalers_[col] = scaler
                
                self._log_transformation('fit_scaler', {
                    'feature': col,
                    'scaler_type': type(scaler).__name__
                })
    
    def _fit_skewness_transformers(self, X: pd.DataFrame):
        """Fit transformers for skewed features."""
        
        for col in X.columns:
            stats = self.feature_stats_[col]
            
            if abs(stats['skewness']) > self.skewness_threshold:
                series = X[col].dropna()
                
                # Try different transformations
                transformations = []
                
                # Log transformation (for positive values)
                if series.min() > 0:
                    try:
                        log_transformed = np.log1p(series)
                        log_skew = abs(log_transformed.skew())
                        transformations.append(('log', log_skew))
                    except:
                        pass
                
                # Square root transformation (for non-negative values)
                if series.min() >= 0:
                    try:
                        sqrt_transformed = np.sqrt(series)
                        sqrt_skew = abs(sqrt_transformed.skew())
                        transformations.append(('sqrt', sqrt_skew))
                    except:
                        pass
                
                # Box-Cox transformation (for positive values)
                if series.min() > 0:
                    try:
                        boxcox_transformed, lambda_param = boxcox(series)
                        boxcox_skew = abs(pd.Series(boxcox_transformed).skew())
                        transformations.append(('boxcox', boxcox_skew, lambda_param))
                    except:
                        pass
                
                # Yeo-Johnson transformation (for any values)
                try:
                    yj_transformed, lambda_param = yeojohnson(series)
                    yj_skew = abs(pd.Series(yj_transformed).skew())
                    transformations.append(('yeojohnson', yj_skew, lambda_param))
                except:
                    pass
                
                # Choose best transformation
                if transformations:
                    best_transform = min(transformations, key=lambda x: x[1])
                    self.transformers_[col] = best_transform
                    
                    self._log_transformation('fit_skewness_transformer', {
                        'feature': col,
                        'original_skewness': stats['skewness'],
                        'transformation': best_transform[0],
                        'new_skewness': best_transform[1]
                    })
    
    def _fit_polynomial_features(self, X: pd.DataFrame):
        """Fit polynomial feature generator."""
        
        # Only use features with reasonable cardinality
        suitable_features = []
        for col in X.columns:
            stats = self.feature_stats_[col]
            if stats['unique_ratio'] > 0.1:  # Not too categorical
                suitable_features.append(col)
        
        if len(suitable_features) > 0:
            self.polynomial_features_ = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=False
            )
            
            # Fit on a subset to avoid memory issues
            sample_size = min(1000, len(X))
            X_sample = X[suitable_features].sample(n=sample_size, random_state=42)
            self.polynomial_features_.fit(X_sample)
            
            self._log_transformation('fit_polynomial_features', {
                'features_used': suitable_features,
                'degree': self.polynomial_degree,
                'n_output_features': self.polynomial_features_.n_output_features_
            })
    
    def _identify_interactions(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Identify important interaction features."""
        
        features = list(X.columns)
        interactions = []
        
        # Generate potential interactions
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features[i+1:], i+1):
                if len(interactions) >= self.max_interactions:
                    break
                    
                # Calculate interaction strength
                interaction_strength = self._calculate_interaction_strength(
                    X[feat1], X[feat2], y
                )
                
                if interaction_strength > 0.1:  # Threshold for meaningful interaction
                    interactions.append((feat1, feat2, interaction_strength))
        
        # Sort by strength and keep top interactions
        interactions.sort(key=lambda x: x[2], reverse=True)
        self.interaction_features_ = interactions[:self.max_interactions]
        
        if self.interaction_features_:
            self._log_transformation('identify_interactions', {
                'n_interactions': len(self.interaction_features_),
                'interactions': [(f1, f2) for f1, f2, _ in self.interaction_features_]
            })
    
    def _calculate_interaction_strength(self, feat1: pd.Series, feat2: pd.Series, y: Optional[pd.Series] = None) -> float:
        """Calculate the strength of interaction between two features."""
        
        # Simple correlation-based interaction strength
        try:
            # Multiplicative interaction
            interaction = feat1 * feat2
            
            if y is not None:
                # Correlation with target
                corr_interaction = abs(interaction.corr(y))
                corr_feat1 = abs(feat1.corr(y))
                corr_feat2 = abs(feat2.corr(y))
                
                # Interaction strength is how much the interaction adds
                strength = max(0, corr_interaction - max(corr_feat1, corr_feat2))
            else:
                # Variance-based strength
                strength = interaction.var() / (feat1.var() * feat2.var() + 1e-8)
                
            return min(strength, 1.0)
        except:
            return 0.0
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        
        X_clean = X.copy()
        
        for col in X.columns:
            if col in self.outlier_bounds_:
                lower_bound, upper_bound = self.outlier_bounds_[col]
                
                if self.outlier_method == 'iqr':
                    # Clip outliers to bounds
                    X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
                elif self.outlier_method == 'zscore':
                    # Clip based on z-score
                    mean_val = X_clean[col].mean()
                    std_val = X_clean[col].std()
                    X_clean[col] = X_clean[col].clip(
                        mean_val - 3 * std_val,
                        mean_val + 3 * std_val
                    )
        
        return X_clean
    
    def _apply_skewness_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply skewness transformations."""
        
        X_transformed = X.copy()
        
        for col, transform_info in self.transformers_.items():
            if col in X_transformed.columns:
                series = X_transformed[col]
                
                if transform_info[0] == 'log':
                    X_transformed[col] = np.log1p(series.clip(lower=0))
                elif transform_info[0] == 'sqrt':
                    X_transformed[col] = np.sqrt(series.clip(lower=0))
                elif transform_info[0] == 'boxcox':
                    lambda_param = transform_info[2]
                    try:
                        X_transformed[col] = boxcox(series.clip(lower=1e-8), lmbda=lambda_param)
                    except:
                        pass
                elif transform_info[0] == 'yeojohnson':
                    lambda_param = transform_info[2]
                    try:
                        X_transformed[col] = yeojohnson(series, lmbda=lambda_param)
                    except:
                        pass
        
        return X_transformed
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformations."""
        
        X_scaled = X.copy()
        
        for col, scaler in self.scalers_.items():
            if col in X_scaled.columns:
                X_scaled[col] = scaler.transform(X_scaled[[col]]).flatten()
        
        return X_scaled
    
    def _generate_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        
        if self.polynomial_features_ is None:
            return pd.DataFrame()
        
        # Get suitable features
        suitable_features = []
        for col in X.columns:
            if col in self.feature_stats_:
                stats = self.feature_stats_[col]
                if stats['unique_ratio'] > 0.1:
                    suitable_features.append(col)
        
        if not suitable_features:
            return pd.DataFrame()
        
        # Generate polynomial features
        X_poly_array = self.polynomial_features_.transform(X[suitable_features])
        
        # Create feature names
        feature_names = self.polynomial_features_.get_feature_names_out(suitable_features)
        
        # Remove original features (they're already in X)
        original_feature_names = suitable_features
        poly_feature_names = [name for name in feature_names if name not in original_feature_names]
        
        # Get corresponding columns
        original_indices = [i for i, name in enumerate(feature_names) if name in original_feature_names]
        poly_indices = [i for i, name in enumerate(feature_names) if name not in original_feature_names]
        
        if poly_indices:
            X_poly = pd.DataFrame(
                X_poly_array[:, poly_indices],
                columns=poly_feature_names,
                index=X.index
            )
            return X_poly
        
        return pd.DataFrame()
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        
        if not self.interaction_features_:
            return pd.DataFrame()
        
        X_interactions = pd.DataFrame(index=X.index)
        
        for feat1, feat2, strength in self.interaction_features_:
            if feat1 in X.columns and feat2 in X.columns:
                # Multiplicative interaction
                interaction_name = f"{feat1}_x_{feat2}"
                X_interactions[interaction_name] = X[feat1] * X[feat2]
                
                # Additive interaction (if different from multiplicative)
                if not X[feat1].equals(X[feat2]):
                    add_interaction_name = f"{feat1}_plus_{feat2}"
                    X_interactions[add_interaction_name] = X[feat1] + X[feat2]
        
        return X_interactions
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all features."""
        return self.feature_stats_
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of all transformations applied."""
        return {
            'scalers_applied': list(self.scalers_.keys()),
            'skewness_transformers_applied': list(self.transformers_.keys()),
            'polynomial_features_generated': self.polynomial_features_ is not None,
            'interaction_features_generated': len(self.interaction_features_),
            'outlier_handling_applied': self.handle_outliers
        }