"""
Base Transformer Classes for Feature Engineering
===============================================

Provides the foundation for all feature engineering transformations.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for all feature transformers in EssentiaX.
    
    Provides common functionality and interface for feature engineering.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.is_fitted_ = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.transformation_log_ = []
        
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseFeatureTransformer':
        """
        Internal fit method to be implemented by subclasses.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
            
        Returns:
        --------
        self : BaseFeatureTransformer
        """
        pass
    
    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Internal transform method to be implemented by subclasses.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        X_transformed : pd.DataFrame
            Transformed features
        """
        pass
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'BaseFeatureTransformer':
        """
        Fit the transformer to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input features
        y : pd.Series or np.ndarray, optional
            Target variable
            
        Returns:
        --------
        self : BaseFeatureTransformer
        """
        # Convert to DataFrame if needed
        X = self._ensure_dataframe(X)
        if y is not None:
            y = self._ensure_series(y)
            
        # Store feature names
        self.feature_names_in_ = list(X.columns)
        
        # Validate input
        self._validate_input(X, y)
        
        # Fit the transformer
        self._fit(X, y)
        
        # Mark as fitted
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"✅ {self.__class__.__name__} fitted successfully")
            
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform the input data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input features
            
        Returns:
        --------
        X_transformed : pd.DataFrame
            Transformed features
        """
        # Check if fitted
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before transform")
            
        # Convert to DataFrame if needed
        X = self._ensure_dataframe(X)
        
        # Validate input
        self._validate_input(X)
        
        # Transform the data
        X_transformed = self._transform(X)
        
        # Store output feature names
        self.feature_names_out_ = list(X_transformed.columns)
        
        if self.verbose:
            print(f"✅ {self.__class__.__name__} transformed {X.shape[0]} samples")
            print(f"   Features: {X.shape[1]} → {X_transformed.shape[1]}")
            
        return X_transformed
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """
        Fit the transformer and transform the data in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input features
        y : pd.Series or np.ndarray, optional
            Target variable
            
        Returns:
        --------
        X_transformed : pd.DataFrame
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def _ensure_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame if needed."""
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
                columns = self.feature_names_in_
            else:
                columns = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=columns)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be pandas DataFrame or numpy array")
        return X
    
    def _ensure_series(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Convert target to Series if needed."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        elif not isinstance(y, pd.Series):
            raise ValueError("Target must be pandas Series or numpy array")
        return y
    
    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Validate input data."""
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
            missing_features = set(self.feature_names_in_) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
                
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names.
        
        Parameters:
        -----------
        input_features : list of str, optional
            Input feature names
            
        Returns:
        --------
        feature_names_out : list of str
            Output feature names
        """
        if not self.is_fitted_:
            raise ValueError("Transformer must be fitted before getting feature names")
            
        return self.feature_names_out_ or []
    
    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """
        Get log of all transformations applied.
        
        Returns:
        --------
        log : list of dict
            Transformation log
        """
        return self.transformation_log_
    
    def _log_transformation(self, operation: str, details: Dict[str, Any]):
        """Log a transformation operation."""
        log_entry = {
            'operation': operation,
            'transformer': self.__class__.__name__,
            'details': details,
            'timestamp': pd.Timestamp.now()
        }
        self.transformation_log_.append(log_entry)


class FeatureQualityMixin:
    """
    Mixin class for feature quality assessment.
    """
    
    def assess_feature_quality(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Assess the quality of features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
            
        Returns:
        --------
        quality_scores : dict
            Feature quality scores
        """
        quality_scores = {}
        
        for col in X.columns:
            score = 0.0
            
            # Missing value penalty
            missing_ratio = X[col].isnull().sum() / len(X)
            score += (1 - missing_ratio) * 0.3
            
            # Variance check
            if X[col].dtype in ['int64', 'float64']:
                if X[col].var() > 0:
                    score += 0.3
                    
            # Uniqueness for categorical
            elif X[col].dtype == 'object':
                unique_ratio = X[col].nunique() / len(X)
                if 0.01 < unique_ratio < 0.9:  # Not too unique, not too constant
                    score += 0.3
                    
            # Target correlation (if available)
            if y is not None:
                try:
                    if X[col].dtype in ['int64', 'float64'] and y.dtype in ['int64', 'float64']:
                        corr = abs(X[col].corr(y))
                        score += min(corr, 0.4)
                except:
                    pass
                    
            quality_scores[col] = min(score, 1.0)
            
        return quality_scores


class SmartTransformationMixin:
    """
    Mixin class for smart transformation selection.
    """
    
    def _detect_data_type(self, series: pd.Series) -> str:
        """
        Detect the semantic data type of a series.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
            
        Returns:
        --------
        data_type : str
            Detected data type
        """
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
            
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually categorical (few unique values)
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.05 and series.nunique() < 20:
                return 'categorical_numeric'
            return 'numeric'
            
        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'
            
        # Check for categorical
        if pd.api.types.is_categorical_dtype(series):
            return 'categorical'
            
        # Check for text
        if series.dtype == 'object':
            # Check if it's actually numeric stored as string
            try:
                pd.to_numeric(series.dropna())
                return 'numeric_string'
            except:
                pass
                
            # Check average string length to distinguish categorical vs text
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 50:
                return 'text'
            else:
                return 'categorical'
                
        return 'unknown'
    
    def _recommend_transformation(self, series: pd.Series, data_type: str) -> List[str]:
        """
        Recommend transformations based on data characteristics.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
        data_type : str
            Detected data type
            
        Returns:
        --------
        recommendations : list of str
            Recommended transformations
        """
        recommendations = []
        
        if data_type == 'numeric':
            # Check skewness
            skewness = abs(series.skew())
            if skewness > 1:
                recommendations.append('log_transform')
                
            # Check for outliers
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(series) * 0.05:
                recommendations.append('robust_scaling')
            else:
                recommendations.append('standard_scaling')
                
        elif data_type in ['categorical', 'categorical_numeric']:
            # Check cardinality
            cardinality = series.nunique()
            if cardinality > 10:
                recommendations.append('target_encoding')
            else:
                recommendations.append('onehot_encoding')
                
        elif data_type == 'datetime':
            recommendations.extend(['extract_date_features', 'cyclical_encoding'])
            
        elif data_type == 'text':
            recommendations.extend(['tfidf_vectorization', 'sentiment_analysis'])
            
        return recommendations