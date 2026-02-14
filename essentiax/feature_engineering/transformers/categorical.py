"""
Advanced Categorical Feature Transformer
=======================================

Intelligent transformations for categorical features with smart encoding selection.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

from ..core.base_transformer import BaseFeatureTransformer, SmartTransformationMixin

class CategoricalTransformer(BaseFeatureTransformer, SmartTransformationMixin):
    """
    📊 Advanced Categorical Feature Transformer
    
    Intelligently transforms categorical features using:
    - Smart encoding method selection
    - Rare category handling
    - Target encoding for high-cardinality features
    - Text feature extraction
    - Ordinal relationship detection
    
    Parameters:
    -----------
    strategy : str, default='auto'
        Transformation strategy:
        - 'auto': AI-powered automatic selection
        - 'conservative': Safe transformations only
        - 'aggressive': Maximum transformations
        
    encode_categoricals : bool, default=True
        Whether to encode categorical features
        
    encoding_method : str, default='auto'
        Encoding method:
        - 'auto': Automatically select best method
        - 'onehot': One-hot encoding
        - 'label': Label encoding
        - 'target': Target encoding (requires y)
        - 'frequency': Frequency encoding
        - 'binary': Binary encoding
        
    handle_rare_categories : bool, default=True
        Whether to group rare categories
        
    rare_threshold : float, default=0.01
        Threshold for rare category grouping (as fraction of total)
        
    max_categories : int, default=20
        Maximum categories for one-hot encoding
        
    detect_ordinal : bool, default=True
        Whether to detect ordinal relationships
        
    text_features : bool, default=True
        Whether to extract features from text columns
        
    max_text_features : int, default=100
        Maximum number of text features to extract
        
    verbose : bool, default=True
        Whether to show transformation details
    """
    
    def __init__(
        self,
        strategy: str = 'auto',
        encode_categoricals: bool = True,
        encoding_method: str = 'auto',
        handle_rare_categories: bool = True,
        rare_threshold: float = 0.01,
        max_categories: int = 20,
        detect_ordinal: bool = True,
        text_features: bool = True,
        max_text_features: int = 100,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        
        self.strategy = strategy
        self.encode_categoricals = encode_categoricals
        self.encoding_method = encoding_method
        self.handle_rare_categories = handle_rare_categories
        self.rare_threshold = rare_threshold
        self.max_categories = max_categories
        self.detect_ordinal = detect_ordinal
        self.text_features = text_features
        self.max_text_features = max_text_features
        
        # Fitted components
        self.encoders_ = {}
        self.rare_category_mappings_ = {}
        self.ordinal_mappings_ = {}
        self.text_vectorizers_ = {}
        self.feature_stats_ = {}
        self.encoding_strategies_ = {}
        
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalTransformer':
        """Fit the categorical transformer."""
        
        # Analyze each feature
        for col in X.columns:
            self._analyze_categorical_feature(X[col], col, y)
        
        # Determine encoding strategies
        self._determine_encoding_strategies(X, y)
        
        # Fit encoders
        if self.encode_categoricals:
            self._fit_encoders(X, y)
        
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the categorical features."""
        
        X_transformed = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            if col in self.encoding_strategies_:
                strategy = self.encoding_strategies_[col]
                
                # Handle rare categories first
                series = self._handle_rare_categories(X[col], col)
                
                # Apply encoding
                if strategy == 'onehot':
                    encoded = self._apply_onehot_encoding(series, col)
                elif strategy == 'label':
                    encoded = self._apply_label_encoding(series, col)
                elif strategy == 'target':
                    encoded = self._apply_target_encoding(series, col)
                elif strategy == 'frequency':
                    encoded = self._apply_frequency_encoding(series, col)
                elif strategy == 'ordinal':
                    encoded = self._apply_ordinal_encoding(series, col)
                elif strategy == 'text':
                    encoded = self._apply_text_encoding(series, col)
                else:
                    # Keep original
                    encoded = pd.DataFrame({col: series})
                
                # Add to result
                if isinstance(encoded, pd.DataFrame):
                    X_transformed = pd.concat([X_transformed, encoded], axis=1)
                else:
                    X_transformed[col] = encoded
        
        return X_transformed
    
    def _analyze_categorical_feature(self, series: pd.Series, col_name: str, y: Optional[pd.Series] = None):
        """Analyze a single categorical feature."""
        
        # Basic statistics
        value_counts = series.value_counts()
        stats_dict = {
            'unique_count': series.nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'missing_ratio': series.isnull().sum() / len(series),
            'cardinality_ratio': series.nunique() / len(series),
            'avg_string_length': series.astype(str).str.len().mean(),
            'max_string_length': series.astype(str).str.len().max(),
            'rare_categories': (value_counts < len(series) * self.rare_threshold).sum()
        }
        
        # Detect if it's actually text
        if stats_dict['avg_string_length'] > 20:
            stats_dict['is_text'] = True
        else:
            stats_dict['is_text'] = False
        
        # Detect ordinal patterns
        if self.detect_ordinal:
            stats_dict['is_ordinal'] = self._detect_ordinal_pattern(series)
        else:
            stats_dict['is_ordinal'] = False
        
        # Target correlation (if available)
        if y is not None:
            stats_dict['target_correlation'] = self._calculate_categorical_target_correlation(series, y)
        else:
            stats_dict['target_correlation'] = 0.0
        
        self.feature_stats_[col_name] = stats_dict
        
        # Log analysis
        self._log_transformation('analyze_categorical_feature', {
            'feature': col_name,
            'stats': stats_dict
        })
    
    def _detect_ordinal_pattern(self, series: pd.Series) -> bool:
        """Detect if a categorical feature has ordinal relationships."""
        
        unique_values = series.dropna().unique()
        
        # Common ordinal patterns
        ordinal_patterns = [
            # Size patterns
            ['xs', 's', 'm', 'l', 'xl', 'xxl'],
            ['small', 'medium', 'large'],
            ['low', 'medium', 'high'],
            ['poor', 'fair', 'good', 'excellent'],
            # Grade patterns
            ['f', 'd', 'c', 'b', 'a'],
            ['fail', 'pass', 'good', 'excellent'],
            # Numeric strings
            ['1', '2', '3', '4', '5'],
            # Education levels
            ['elementary', 'high school', 'college', 'graduate'],
            # Frequency patterns
            ['never', 'rarely', 'sometimes', 'often', 'always']
        ]
        
        # Check if values match any ordinal pattern
        unique_lower = [str(v).lower() for v in unique_values]
        
        for pattern in ordinal_patterns:
            if set(unique_lower).issubset(set(pattern)):
                return True
        
        # Check for numeric strings
        try:
            numeric_values = [float(v) for v in unique_values if str(v).replace('.', '').isdigit()]
            if len(numeric_values) == len(unique_values) and len(numeric_values) > 2:
                return True
        except:
            pass
        
        return False
    
    def _calculate_categorical_target_correlation(self, series: pd.Series, y: pd.Series) -> float:
        """Calculate correlation between categorical feature and target."""
        
        try:
            # For classification targets
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                # Use Cramér's V
                return self._cramers_v(series, y)
            else:
                # For regression targets, use ANOVA F-statistic
                from scipy.stats import f_oneway
                groups = [y[series == cat].dropna() for cat in series.unique()]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) > 1:
                    f_stat, p_value = f_oneway(*groups)
                    return min(f_stat / 100, 1.0)  # Normalize
                else:
                    return 0.0
        except:
            return 0.0
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramér's V statistic for categorical association."""
        
        try:
            confusion_matrix = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        except:
            return 0.0
    
    def _determine_encoding_strategies(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Determine the best encoding strategy for each feature."""
        
        for col in X.columns:
            stats = self.feature_stats_[col]
            
            if self.encoding_method == 'auto':
                # AI-powered strategy selection
                if stats['is_text']:
                    strategy = 'text'
                elif stats['is_ordinal']:
                    strategy = 'ordinal'
                elif stats['unique_count'] <= 2:
                    strategy = 'label'  # Binary categorical
                elif stats['unique_count'] <= self.max_categories:
                    strategy = 'onehot'
                elif y is not None and stats['target_correlation'] > 0.1:
                    strategy = 'target'  # High correlation with target
                elif stats['unique_count'] > 50:
                    strategy = 'frequency'  # High cardinality
                else:
                    strategy = 'onehot'
            else:
                strategy = self.encoding_method
            
            self.encoding_strategies_[col] = strategy
            
            self._log_transformation('determine_encoding_strategy', {
                'feature': col,
                'strategy': strategy,
                'reason': self._get_strategy_reason(stats, strategy)
            })
    
    def _get_strategy_reason(self, stats: Dict[str, Any], strategy: str) -> str:
        """Get reason for choosing encoding strategy."""
        
        if strategy == 'text':
            return f"Text feature (avg length: {stats['avg_string_length']:.1f})"
        elif strategy == 'ordinal':
            return "Detected ordinal pattern"
        elif strategy == 'label':
            return f"Binary categorical ({stats['unique_count']} categories)"
        elif strategy == 'onehot':
            return f"Low cardinality ({stats['unique_count']} categories)"
        elif strategy == 'target':
            return f"High target correlation ({stats['target_correlation']:.3f})"
        elif strategy == 'frequency':
            return f"High cardinality ({stats['unique_count']} categories)"
        else:
            return "Default strategy"
    
    def _fit_encoders(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit encoders for each feature."""
        
        for col in X.columns:
            strategy = self.encoding_strategies_[col]
            series = self._handle_rare_categories(X[col], col)
            
            if strategy == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(series.values.reshape(-1, 1))
                self.encoders_[col] = encoder
                
            elif strategy == 'label':
                encoder = LabelEncoder()
                encoder.fit(series.dropna())
                self.encoders_[col] = encoder
                
            elif strategy == 'target' and y is not None:
                # Target encoding
                target_means = series.groupby(series).apply(lambda x: y[x.index].mean())
                self.encoders_[col] = target_means
                
            elif strategy == 'frequency':
                # Frequency encoding
                freq_map = series.value_counts().to_dict()
                self.encoders_[col] = freq_map
                
            elif strategy == 'ordinal':
                # Ordinal encoding
                ordinal_map = self._create_ordinal_mapping(series)
                self.encoders_[col] = ordinal_map
                
            elif strategy == 'text':
                # Text vectorization
                vectorizer = TfidfVectorizer(
                    max_features=self.max_text_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                vectorizer.fit(series.astype(str))
                self.encoders_[col] = vectorizer
    
    def _handle_rare_categories(self, series: pd.Series, col_name: str) -> pd.Series:
        """Handle rare categories by grouping them."""
        
        if not self.handle_rare_categories:
            return series
        
        if col_name not in self.rare_category_mappings_:
            # Create rare category mapping during fit
            value_counts = series.value_counts()
            rare_threshold_count = len(series) * self.rare_threshold
            rare_categories = value_counts[value_counts < rare_threshold_count].index.tolist()
            
            if rare_categories:
                self.rare_category_mappings_[col_name] = rare_categories
                self._log_transformation('handle_rare_categories', {
                    'feature': col_name,
                    'rare_categories_count': len(rare_categories),
                    'rare_categories': rare_categories[:10]  # Log first 10
                })
        
        # Apply rare category mapping
        if col_name in self.rare_category_mappings_:
            rare_categories = self.rare_category_mappings_[col_name]
            series_handled = series.copy()
            series_handled = series_handled.replace(rare_categories, 'RARE_CATEGORY')
            return series_handled
        
        return series
    
    def _create_ordinal_mapping(self, series: pd.Series) -> Dict[str, int]:
        """Create ordinal mapping for a series."""
        
        unique_values = series.dropna().unique()
        
        # Try to sort intelligently
        try:
            # Try numeric sorting
            sorted_values = sorted(unique_values, key=lambda x: float(x))
        except:
            try:
                # Try alphabetic sorting
                sorted_values = sorted(unique_values)
            except:
                # Fallback to original order
                sorted_values = list(unique_values)
        
        return {val: i for i, val in enumerate(sorted_values)}
    
    def _apply_onehot_encoding(self, series: pd.Series, col_name: str) -> pd.DataFrame:
        """Apply one-hot encoding."""
        
        encoder = self.encoders_[col_name]
        encoded_array = encoder.transform(series.values.reshape(-1, 1))
        
        # Create column names
        feature_names = [f"{col_name}_{cat}" for cat in encoder.categories_[0]]
        
        return pd.DataFrame(encoded_array, columns=feature_names, index=series.index)
    
    def _apply_label_encoding(self, series: pd.Series, col_name: str) -> pd.Series:
        """Apply label encoding."""
        
        encoder = self.encoders_[col_name]
        
        # Handle unknown categories
        series_encoded = series.copy()
        mask = series_encoded.isin(encoder.classes_)
        series_encoded[mask] = encoder.transform(series_encoded[mask])
        series_encoded[~mask] = -1  # Unknown category
        
        return series_encoded.astype(int)
    
    def _apply_target_encoding(self, series: pd.Series, col_name: str) -> pd.Series:
        """Apply target encoding."""
        
        target_means = self.encoders_[col_name]
        
        # Map values to target means
        series_encoded = series.map(target_means)
        
        # Fill unknown categories with overall mean
        overall_mean = target_means.mean()
        series_encoded = series_encoded.fillna(overall_mean)
        
        return series_encoded
    
    def _apply_frequency_encoding(self, series: pd.Series, col_name: str) -> pd.Series:
        """Apply frequency encoding."""
        
        freq_map = self.encoders_[col_name]
        
        # Map values to frequencies
        series_encoded = series.map(freq_map)
        
        # Fill unknown categories with 0
        series_encoded = series_encoded.fillna(0)
        
        return series_encoded
    
    def _apply_ordinal_encoding(self, series: pd.Series, col_name: str) -> pd.Series:
        """Apply ordinal encoding."""
        
        ordinal_map = self.encoders_[col_name]
        
        # Map values to ordinal numbers
        series_encoded = series.map(ordinal_map)
        
        # Fill unknown categories with -1
        series_encoded = series_encoded.fillna(-1)
        
        return series_encoded.astype(int)
    
    def _apply_text_encoding(self, series: pd.Series, col_name: str) -> pd.DataFrame:
        """Apply text encoding (TF-IDF)."""
        
        vectorizer = self.encoders_[col_name]
        
        # Transform text to TF-IDF features
        tfidf_matrix = vectorizer.transform(series.astype(str))
        
        # Create feature names
        feature_names = [f"{col_name}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        
        return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=series.index)
    
    def get_feature_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all features."""
        return self.feature_stats_
    
    def get_encoding_strategies(self) -> Dict[str, str]:
        """Get encoding strategies for all features."""
        return self.encoding_strategies_
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Get summary of all transformations applied."""
        return {
            'encoding_strategies': self.encoding_strategies_,
            'rare_categories_handled': list(self.rare_category_mappings_.keys()),
            'ordinal_features_detected': [col for col, stats in self.feature_stats_.items() if stats.get('is_ordinal', False)],
            'text_features_detected': [col for col, stats in self.feature_stats_.items() if stats.get('is_text', False)]
        }