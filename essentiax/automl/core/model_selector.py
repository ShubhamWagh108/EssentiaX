"""
Intelligent Model Selection for AutoML
=====================================

Smart algorithm selection based on dataset characteristics and problem requirements.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from rich.console import Console

console = Console()

class ModelSelector:
    """
    🎯 Intelligent Model Selection Engine
    
    Automatically selects the most promising machine learning algorithms
    based on dataset characteristics, problem type, and performance requirements.
    
    Selection Strategy:
    - Dataset size and dimensionality analysis
    - Problem complexity assessment
    - Interpretability requirements
    - Computational constraints
    """
    
    def __init__(
        self,
        task: str = 'classification',
        interpretability: str = 'medium',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.task = task
        self.interpretability = interpretability
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Model priority rankings based on general performance
        self.classification_models = self._get_classification_models()
        self.regression_models = self._get_regression_models()
    
    def get_candidate_models(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """
        Select candidate models based on dataset characteristics.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        candidates : list
            List of candidate models with metadata
        """
        # Analyze dataset characteristics
        dataset_profile = self._analyze_dataset(X, y)
        
        # Get base model pool
        if self.task == 'classification':
            model_pool = self.classification_models
        elif self.task == 'regression':
            model_pool = self.regression_models
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
        # Filter and rank models based on dataset characteristics
        candidates = self._select_optimal_models(model_pool, dataset_profile)
        
        return candidates
    
    def _analyze_dataset(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze dataset characteristics for model selection."""
        n_samples, n_features = X.shape
        
        profile = {
            'n_samples': n_samples,
            'n_features': n_features,
            'dimensionality_ratio': n_features / n_samples,
            'is_high_dimensional': n_features > n_samples,
            'is_large_dataset': n_samples > 10000,
            'is_small_dataset': n_samples < 1000,
            'has_categorical': len(X.select_dtypes(include=['object', 'category']).columns) > 0,
            'missing_ratio': X.isnull().sum().sum() / (n_samples * n_features),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Task-specific analysis
        if self.task == 'classification':
            profile.update({
                'n_classes': y.nunique(),
                'is_binary': y.nunique() == 2,
                'is_imbalanced': self._check_class_imbalance(y),
                'class_distribution': y.value_counts(normalize=True).to_dict()
            })
        elif self.task == 'regression':
            profile.update({
                'target_range': y.max() - y.min(),
                'target_std': y.std(),
                'target_skewness': y.skew(),
                'has_outliers': self._check_outliers(y)
            })
        
        return profile
    
    def _check_class_imbalance(self, y: pd.Series) -> bool:
        """Check if dataset has class imbalance."""
        class_counts = y.value_counts(normalize=True)
        min_class_ratio = class_counts.min()
        return min_class_ratio < 0.1  # Less than 10% for minority class
    
    def _check_outliers(self, y: pd.Series) -> bool:
        """Check if target has significant outliers."""
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((y < lower_bound) | (y > upper_bound)).sum()
        return outliers / len(y) > 0.05  # More than 5% outliers
    
    def _get_classification_models(self) -> List[Dict[str, Any]]:
        """Get classification model pool with metadata."""
        models = []
        
        # LightGBM (if available) - Best overall performance
        if LIGHTGBM_AVAILABLE:
            models.append({
                'name': 'LightGBM',
                'model_class': LGBMClassifier,
                'priority': 1,
                'interpretability': 'medium',
                'speed': 'fast',
                'memory_efficient': True,
                'handles_categorical': True,
                'handles_missing': True,
                'good_for': ['large_datasets', 'mixed_features', 'general_purpose'],
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbose': -1
                }
            })
        
        # XGBoost (if available) - Excellent performance
        if XGBOOST_AVAILABLE:
            models.append({
                'name': 'XGBoost',
                'model_class': XGBClassifier,
                'priority': 2,
                'interpretability': 'medium',
                'speed': 'medium',
                'memory_efficient': True,
                'handles_categorical': False,
                'handles_missing': True,
                'good_for': ['structured_data', 'competitions', 'general_purpose'],
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbosity': 0
                }
            })
        
        # Random Forest - Robust and interpretable
        models.append({
            'name': 'RandomForest',
            'model_class': RandomForestClassifier,
            'priority': 3,
            'interpretability': 'high',
            'speed': 'medium',
            'memory_efficient': False,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['interpretability', 'feature_importance', 'robust'],
            'params': {
                'n_estimators': 100,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }
        })
        
        # Logistic Regression - Fast and interpretable
        models.append({
            'name': 'LogisticRegression',
            'model_class': LogisticRegression,
            'priority': 4,
            'interpretability': 'high',
            'speed': 'fast',
            'memory_efficient': True,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['linear_relationships', 'baseline', 'interpretability'],
            'params': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'max_iter': 1000
            }
        })
        
        # Support Vector Machine - Good for high-dimensional data
        models.append({
            'name': 'SVM',
            'model_class': SVC,
            'priority': 5,
            'interpretability': 'low',
            'speed': 'slow',
            'memory_efficient': False,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['high_dimensional', 'small_datasets', 'non_linear'],
            'params': {
                'random_state': self.random_state,
                'probability': True
            }
        })
        
        # K-Nearest Neighbors - Simple and effective
        models.append({
            'name': 'KNN',
            'model_class': KNeighborsClassifier,
            'priority': 6,
            'interpretability': 'medium',
            'speed': 'slow',
            'memory_efficient': False,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['small_datasets', 'local_patterns', 'non_parametric'],
            'params': {
                'n_jobs': self.n_jobs
            }
        })
        
        # Naive Bayes - Fast for text and categorical data
        models.append({
            'name': 'NaiveBayes',
            'model_class': GaussianNB,
            'priority': 7,
            'interpretability': 'high',
            'speed': 'fast',
            'memory_efficient': True,
            'handles_categorical': True,
            'handles_missing': False,
            'good_for': ['text_data', 'categorical_features', 'baseline'],
            'params': {}
        })
        
        return models
    
    def _get_regression_models(self) -> List[Dict[str, Any]]:
        """Get regression model pool with metadata."""
        models = []
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            models.append({
                'name': 'LightGBM',
                'model_class': LGBMRegressor,
                'priority': 1,
                'interpretability': 'medium',
                'speed': 'fast',
                'memory_efficient': True,
                'handles_categorical': True,
                'handles_missing': True,
                'good_for': ['large_datasets', 'mixed_features', 'general_purpose'],
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbose': -1
                }
            })
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models.append({
                'name': 'XGBoost',
                'model_class': XGBRegressor,
                'priority': 2,
                'interpretability': 'medium',
                'speed': 'medium',
                'memory_efficient': True,
                'handles_categorical': False,
                'handles_missing': True,
                'good_for': ['structured_data', 'competitions', 'general_purpose'],
                'params': {
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbosity': 0
                }
            })
        
        # Random Forest
        models.append({
            'name': 'RandomForest',
            'model_class': RandomForestRegressor,
            'priority': 3,
            'interpretability': 'high',
            'speed': 'medium',
            'memory_efficient': False,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['interpretability', 'feature_importance', 'robust'],
            'params': {
                'n_estimators': 100,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            }
        })
        
        # Ridge Regression
        models.append({
            'name': 'Ridge',
            'model_class': Ridge,
            'priority': 4,
            'interpretability': 'high',
            'speed': 'fast',
            'memory_efficient': True,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['linear_relationships', 'regularization', 'multicollinearity'],
            'params': {
                'random_state': self.random_state
            }
        })
        
        # Lasso Regression
        models.append({
            'name': 'Lasso',
            'model_class': Lasso,
            'priority': 5,
            'interpretability': 'high',
            'speed': 'fast',
            'memory_efficient': True,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['feature_selection', 'sparse_solutions', 'regularization'],
            'params': {
                'random_state': self.random_state,
                'max_iter': 1000
            }
        })
        
        # Support Vector Regression
        models.append({
            'name': 'SVR',
            'model_class': SVR,
            'priority': 6,
            'interpretability': 'low',
            'speed': 'slow',
            'memory_efficient': False,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['high_dimensional', 'small_datasets', 'non_linear'],
            'params': {}
        })
        
        # K-Nearest Neighbors
        models.append({
            'name': 'KNN',
            'model_class': KNeighborsRegressor,
            'priority': 7,
            'interpretability': 'medium',
            'speed': 'slow',
            'memory_efficient': False,
            'handles_categorical': False,
            'handles_missing': False,
            'good_for': ['small_datasets', 'local_patterns', 'non_parametric'],
            'params': {
                'n_jobs': self.n_jobs
            }
        })
        
        return models
    
    def _select_optimal_models(self, model_pool: List[Dict], dataset_profile: Dict) -> List[Dict]:
        """Select and rank models based on dataset characteristics."""
        scored_models = []
        
        for model_info in model_pool:
            score = self._score_model_fit(model_info, dataset_profile)
            model_info['selection_score'] = score
            scored_models.append(model_info)
        
        # Sort by selection score (higher is better)
        scored_models.sort(key=lambda x: x['selection_score'], reverse=True)
        
        # Apply interpretability filter
        if self.interpretability == 'high':
            scored_models = [m for m in scored_models if m['interpretability'] in ['high', 'medium']]
        elif self.interpretability == 'medium':
            # Keep all models but prefer interpretable ones
            pass
        
        # Return top candidates (limit to reasonable number)
        max_candidates = min(5, len(scored_models))
        return scored_models[:max_candidates]
    
    def _score_model_fit(self, model_info: Dict, dataset_profile: Dict) -> float:
        """Score how well a model fits the dataset characteristics."""
        score = 0.0
        
        # Base priority score (higher priority = higher score)
        max_priority = 7
        score += (max_priority - model_info['priority']) * 10
        
        # Dataset size considerations
        if dataset_profile['is_large_dataset']:
            if model_info['speed'] == 'fast':
                score += 15
            elif model_info['speed'] == 'medium':
                score += 10
            else:
                score -= 5
        
        if dataset_profile['is_small_dataset']:
            if model_info['name'] in ['KNN', 'SVM', 'SVR']:
                score += 10
            if model_info['name'] in ['LightGBM', 'XGBoost']:
                score -= 5  # May overfit on small data
        
        # High-dimensional data
        if dataset_profile['is_high_dimensional']:
            if model_info['name'] in ['SVM', 'SVR', 'Lasso']:
                score += 15
            if model_info['name'] in ['KNN']:
                score -= 10  # Curse of dimensionality
        
        # Memory efficiency
        if dataset_profile['memory_usage_mb'] > 100:  # Large memory usage
            if model_info['memory_efficient']:
                score += 10
            else:
                score -= 5
        
        # Categorical features
        if dataset_profile['has_categorical']:
            if model_info['handles_categorical']:
                score += 10
            else:
                score -= 2
        
        # Missing values
        if dataset_profile['missing_ratio'] > 0.01:  # More than 1% missing
            if model_info['handles_missing']:
                score += 8
            else:
                score -= 3
        
        # Task-specific scoring
        if self.task == 'classification':
            # Imbalanced data
            if dataset_profile.get('is_imbalanced', False):
                if model_info['name'] in ['RandomForest', 'LightGBM', 'XGBoost']:
                    score += 10
            
            # Binary vs multiclass
            if dataset_profile.get('is_binary', True):
                if model_info['name'] in ['LogisticRegression', 'SVM']:
                    score += 5
        
        elif self.task == 'regression':
            # Outliers
            if dataset_profile.get('has_outliers', False):
                if model_info['name'] in ['RandomForest', 'LightGBM', 'XGBoost']:
                    score += 8
                if model_info['name'] in ['Ridge', 'Lasso']:
                    score -= 3
        
        # Interpretability preference
        interpretability_scores = {'high': 10, 'medium': 5, 'low': 0}
        if self.interpretability == 'high':
            score += interpretability_scores[model_info['interpretability']]
        
        return score