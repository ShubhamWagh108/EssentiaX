"""
Smart Feature Selector
======================

AI-powered feature selection using multiple algorithms and ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import accuracy_score, r2_score
import warnings

from ..core.base_transformer import BaseFeatureTransformer

class SmartFeatureSelector(BaseFeatureTransformer):
    """
    🎯 Smart Feature Selector
    
    AI-powered feature selection that combines multiple algorithms:
    - Statistical tests (F-test, mutual information)
    - Model-based selection (Random Forest, Lasso)
    - Recursive feature elimination
    - Correlation-based filtering
    - Ensemble voting across methods
    
    Parameters:
    -----------
    max_features : int or float, optional
        Maximum number of features to select:
        - int: Exact number of features
        - float: Proportion of features (0.0 to 1.0)
        - None: Select optimal number automatically
        
    selection_methods : list of str, default=['statistical', 'model_based', 'recursive']
        Feature selection methods to use:
        - 'statistical': F-test and mutual information
        - 'model_based': Random Forest and Lasso importance
        - 'recursive': Recursive feature elimination
        - 'correlation': Correlation-based filtering
        
    remove_correlated : bool, default=True
        Whether to remove highly correlated features
        
    correlation_threshold : float, default=0.95
        Correlation threshold for feature removal
        
    ensemble_voting : bool, default=True
        Whether to use ensemble voting across methods
        
    voting_threshold : float, default=0.5
        Minimum vote proportion to select a feature
        
    cv_folds : int, default=5
        Number of cross-validation folds for model-based selection
        
    random_state : int, default=42
        Random state for reproducibility
        
    n_jobs : int, default=-1
        Number of parallel jobs
        
    verbose : bool, default=True
        Whether to show selection details
    """
    
    def __init__(
        self,
        max_features: Optional[Union[int, float]] = None,
        selection_methods: List[str] = ['statistical', 'model_based', 'recursive'],
        remove_correlated: bool = True,
        correlation_threshold: float = 0.95,
        ensemble_voting: bool = True,
        voting_threshold: float = 0.5,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        
        self.max_features = max_features
        self.selection_methods = selection_methods
        self.remove_correlated = remove_correlated
        self.correlation_threshold = correlation_threshold
        self.ensemble_voting = ensemble_voting
        self.voting_threshold = voting_threshold
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Results storage
        self.feature_importance_ = None
        self.selected_features_ = None
        self.selection_scores_ = {}
        self.correlation_matrix_ = None
        self.method_votes_ = {}
        
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> 'SmartFeatureSelector':
        """Fit the feature selector."""
        
        if y is None:
            raise ValueError("Target variable y is required for feature selection")
        
        # Determine problem type
        if y.dtype in ['object', 'category'] or y.nunique() < 20:
            self.problem_type_ = 'classification'
        else:
            self.problem_type_ = 'regression'
        
        if self.verbose:
            print(f"🎯 Smart Feature Selection ({self.problem_type_})")
            print(f"   Original features: {X.shape[1]}")
        
        # Initialize feature importance DataFrame
        self.feature_importance_ = pd.DataFrame(index=X.columns)
        
        # Apply each selection method
        if 'statistical' in self.selection_methods:
            self._apply_statistical_selection(X, y)
        
        if 'model_based' in self.selection_methods:
            self._apply_model_based_selection(X, y)
        
        if 'recursive' in self.selection_methods:
            self._apply_recursive_selection(X, y)
        
        if 'correlation' in self.selection_methods:
            self._apply_correlation_filtering(X)
        
        # Ensemble voting
        if self.ensemble_voting:
            self._apply_ensemble_voting()
        else:
            # Use the first method's results
            method_name = list(self.method_votes_.keys())[0]
            self.selected_features_ = self.method_votes_[method_name]
        
        # Apply max_features constraint
        self._apply_max_features_constraint(X)
        
        if self.verbose:
            print(f"   Selected features: {len(self.selected_features_)}")
            print(f"   Selection ratio: {len(self.selected_features_) / X.shape[1]:.2%}")
        
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        
        if self.selected_features_ is None:
            raise ValueError("Feature selector must be fitted before transform")
        
        # Select features
        available_features = [f for f in self.selected_features_ if f in X.columns]
        
        if len(available_features) == 0:
            raise ValueError("No selected features found in input data")
        
        return X[available_features]
    
    def _apply_statistical_selection(self, X: pd.DataFrame, y: pd.Series):
        """Apply statistical feature selection methods."""
        
        if self.verbose:
            print("   📊 Applying statistical selection...")
        
        # F-test
        if self.problem_type_ == 'classification':
            f_selector = SelectKBest(score_func=f_classif, k='all')
        else:
            f_selector = SelectKBest(score_func=f_regression, k='all')
        
        f_selector.fit(X, y)
        f_scores = f_selector.scores_
        f_scores_normalized = f_scores / f_scores.max()
        
        self.feature_importance_['f_test_score'] = f_scores_normalized
        self.selection_scores_['f_test'] = dict(zip(X.columns, f_scores_normalized))
        
        # Mutual Information
        if self.problem_type_ == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        mi_scores_normalized = mi_scores / (mi_scores.max() + 1e-8)
        
        self.feature_importance_['mutual_info_score'] = mi_scores_normalized
        self.selection_scores_['mutual_info'] = dict(zip(X.columns, mi_scores_normalized))
        
        # Combined statistical score
        combined_scores = (f_scores_normalized + mi_scores_normalized) / 2
        self.feature_importance_['statistical_score'] = combined_scores
        
        # Select top features based on statistical scores
        n_features = min(int(len(X.columns) * 0.8), len(X.columns))  # Top 80%
        top_features = pd.Series(combined_scores, index=X.columns).nlargest(n_features).index.tolist()
        self.method_votes_['statistical'] = top_features
    
    def _apply_model_based_selection(self, X: pd.DataFrame, y: pd.Series):
        """Apply model-based feature selection."""
        
        if self.verbose:
            print("   🌲 Applying model-based selection...")
        
        # Random Forest importance
        if self.problem_type_ == 'classification':
            rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        rf_model.fit(X, y)
        rf_importance = rf_model.feature_importances_
        rf_importance_normalized = rf_importance / rf_importance.max()
        
        self.feature_importance_['rf_importance'] = rf_importance_normalized
        self.selection_scores_['random_forest'] = dict(zip(X.columns, rf_importance_normalized))
        
        # Lasso/LogisticRegression importance
        try:
            if self.problem_type_ == 'classification':
                lasso_model = LogisticRegressionCV(
                    cv=self.cv_folds,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    max_iter=1000
                )
            else:
                lasso_model = LassoCV(
                    cv=self.cv_folds,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            
            lasso_model.fit(X, y)
            lasso_coef = np.abs(lasso_model.coef_)
            if lasso_coef.ndim > 1:
                lasso_coef = lasso_coef[0]  # For multiclass classification
            
            lasso_importance_normalized = lasso_coef / (lasso_coef.max() + 1e-8)
            
            self.feature_importance_['lasso_importance'] = lasso_importance_normalized
            self.selection_scores_['lasso'] = dict(zip(X.columns, lasso_importance_normalized))
            
            # Combined model-based score
            combined_scores = (rf_importance_normalized + lasso_importance_normalized) / 2
        except:
            # Fallback to RF only
            combined_scores = rf_importance_normalized
        
        self.feature_importance_['model_based_score'] = combined_scores
        
        # Select top features based on model importance
        n_features = min(int(len(X.columns) * 0.7), len(X.columns))  # Top 70%
        top_features = pd.Series(combined_scores, index=X.columns).nlargest(n_features).index.tolist()
        self.method_votes_['model_based'] = top_features
    
    def _apply_recursive_selection(self, X: pd.DataFrame, y: pd.Series):
        """Apply recursive feature elimination."""
        
        if self.verbose:
            print("   🔄 Applying recursive selection...")
        
        # Use Random Forest as base estimator
        if self.problem_type_ == 'classification':
            base_estimator = RandomForestClassifier(
                n_estimators=50,  # Fewer trees for speed
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            base_estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        
        # Select top 60% of features
        n_features_to_select = max(int(len(X.columns) * 0.6), 1)
        
        rfe = RFE(
            estimator=base_estimator,
            n_features_to_select=n_features_to_select,
            step=0.1  # Remove 10% of features at each step
        )
        
        rfe.fit(X, y)
        
        # Get feature rankings
        rfe_rankings = rfe.ranking_
        rfe_scores = 1.0 / rfe_rankings  # Convert rankings to scores (higher is better)
        rfe_scores_normalized = rfe_scores / rfe_scores.max()
        
        self.feature_importance_['rfe_score'] = rfe_scores_normalized
        self.selection_scores_['rfe'] = dict(zip(X.columns, rfe_scores_normalized))
        
        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()
        self.method_votes_['recursive'] = selected_features
    
    def _apply_correlation_filtering(self, X: pd.DataFrame):
        """Apply correlation-based feature filtering."""
        
        if self.verbose:
            print("   🔗 Applying correlation filtering...")
        
        # Calculate correlation matrix
        self.correlation_matrix_ = X.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix_.columns)):
            for j in range(i+1, len(self.correlation_matrix_.columns)):
                if self.correlation_matrix_.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((
                        self.correlation_matrix_.columns[i],
                        self.correlation_matrix_.columns[j],
                        self.correlation_matrix_.iloc[i, j]
                    ))
        
        # Remove features with high correlation
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            # Keep the feature with higher average correlation with other features
            avg_corr_feat1 = self.correlation_matrix_[feat1].mean()
            avg_corr_feat2 = self.correlation_matrix_[feat2].mean()
            
            if avg_corr_feat1 > avg_corr_feat2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        # Features to keep
        features_to_keep = [f for f in X.columns if f not in features_to_remove]
        self.method_votes_['correlation'] = features_to_keep
        
        # Store correlation scores (inverse of max correlation with other features)
        corr_scores = []
        for col in X.columns:
            max_corr_with_others = self.correlation_matrix_[col].drop(col).max()
            corr_score = 1.0 - max_corr_with_others  # Higher score for less correlated features
            corr_scores.append(corr_score)
        
        self.feature_importance_['correlation_score'] = corr_scores
        self.selection_scores_['correlation'] = dict(zip(X.columns, corr_scores))
    
    def _apply_ensemble_voting(self):
        """Apply ensemble voting across all methods."""
        
        if self.verbose:
            print("   🗳️ Applying ensemble voting...")
        
        # Count votes for each feature
        feature_votes = {}
        total_methods = len(self.method_votes_)
        
        for feature in self.feature_names_in_:
            votes = 0
            for method, selected_features in self.method_votes_.items():
                if feature in selected_features:
                    votes += 1
            feature_votes[feature] = votes / total_methods
        
        # Select features that meet voting threshold
        selected_features = [
            feature for feature, vote_ratio in feature_votes.items()
            if vote_ratio >= self.voting_threshold
        ]
        
        # If no features meet threshold, lower it gradually
        if len(selected_features) == 0:
            for threshold in [0.4, 0.3, 0.2, 0.1]:
                selected_features = [
                    feature for feature, vote_ratio in feature_votes.items()
                    if vote_ratio >= threshold
                ]
                if len(selected_features) > 0:
                    break
        
        # If still no features, select top features by vote count
        if len(selected_features) == 0:
            selected_features = sorted(feature_votes.keys(), key=lambda x: feature_votes[x], reverse=True)[:10]
        
        self.selected_features_ = selected_features
        
        # Store voting scores
        self.feature_importance_['ensemble_vote'] = [feature_votes[f] for f in self.feature_names_in_]
        self.selection_scores_['ensemble'] = feature_votes
    
    def _apply_max_features_constraint(self, X: pd.DataFrame):
        """Apply max_features constraint."""
        
        if self.max_features is None:
            return
        
        if isinstance(self.max_features, float):
            max_features_count = int(len(X.columns) * self.max_features)
        else:
            max_features_count = self.max_features
        
        if len(self.selected_features_) > max_features_count:
            # Rank features by ensemble vote score
            if 'ensemble_vote' in self.feature_importance_.columns:
                feature_scores = self.feature_importance_['ensemble_vote']
            else:
                # Fallback to first available score
                score_col = self.feature_importance_.columns[0]
                feature_scores = self.feature_importance_[score_col]
            
            # Select top features
            top_features = feature_scores.nlargest(max_features_count).index.tolist()
            self.selected_features_ = [f for f in self.selected_features_ if f in top_features]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from all methods."""
        return self.feature_importance_
    
    def get_selection_scores(self) -> Dict[str, Dict[str, float]]:
        """Get selection scores from all methods."""
        return self.selection_scores_
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features_ or []
    
    def get_method_votes(self) -> Dict[str, List[str]]:
        """Get feature selections from each method."""
        return self.method_votes_
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance scores."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.feature_importance_ is None:
                print("No feature importance data available. Fit the selector first.")
                return
            
            # Get top features
            if 'ensemble_vote' in self.feature_importance_.columns:
                top_features = self.feature_importance_['ensemble_vote'].nlargest(top_n)
            else:
                score_col = self.feature_importance_.columns[0]
                top_features = self.feature_importance_[score_col].nlargest(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.title(f'Top {top_n} Feature Importance Scores')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix if correlation filtering was applied."""
        return self.correlation_matrix_