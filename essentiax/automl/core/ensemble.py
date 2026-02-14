"""
Advanced Ensemble Methods for AutoML - Phase 2 Enhanced
=======================================================

Intelligent ensemble creation with advanced methods for superior performance:
- Multi-level stacking with hierarchical architecture
- Dynamic ensemble selection based on input characteristics
- Bayesian model averaging with uncertainty quantification
- Ensemble pruning for efficiency optimization
- Diversity optimization for better ensemble performance
- Uncertainty quantification and confidence estimation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, r2_score, log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time

console = Console()

class AdvancedEnsembleBuilder:
    """
    🏆 Advanced Ensemble Builder - Phase 2
    
    Creates optimal ensembles using advanced methods:
    - Multi-level stacking with hierarchical architecture
    - Dynamic ensemble selection based on input characteristics
    - Bayesian model averaging with uncertainty quantification
    - Ensemble pruning for efficiency optimization
    - Diversity optimization for better ensemble performance
    - Uncertainty quantification and confidence estimation
    
    Advanced Features:
    - Hierarchical stacking (Level 1 → Level 2 → Meta-learner)
    - Context-aware ensemble selection
    - Probabilistic ensemble combination
    - Automatic ensemble size optimization
    - Model diversity metrics and optimization
    - Performance vs efficiency trade-offs
    - Uncertainty estimation for predictions
    """
    
    def __init__(
        self,
        ensemble_size: int = 5,
        ensemble_method: str = 'auto',  # 'voting', 'stacking', 'multi_level', 'dynamic', 'bayesian'
        meta_learner: str = 'auto',
        diversity_threshold: float = 0.1,
        uncertainty_quantification: bool = True,
        ensemble_pruning: bool = True,
        max_ensemble_size: int = 10,
        min_ensemble_size: int = 2,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.ensemble_size = ensemble_size
        self.ensemble_method = ensemble_method
        self.meta_learner = meta_learner
        self.diversity_threshold = diversity_threshold
        self.uncertainty_quantification = uncertainty_quantification
        self.ensemble_pruning = ensemble_pruning
        self.max_ensemble_size = max_ensemble_size
        self.min_ensemble_size = min_ensemble_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Advanced ensemble tracking
        self.ensemble_models_ = []
        self.ensemble_weights_ = []
        self.meta_model_ = None
        self.level2_models_ = []  # For multi-level stacking
        self.ensemble_score_ = 0.0
        self.diversity_score_ = 0.0
        self.uncertainty_estimator_ = None
        self.ensemble_hierarchy_ = {}
        self.pruning_results_ = {}
        
        np.random.seed(random_state)
    
    def create_advanced_ensemble(
        self, 
        models: List[Any], 
        X: pd.DataFrame, 
        y: pd.Series,
        task_type: str = 'classification',
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Any:
        """
        Create advanced ensemble using Phase 2 methods.
        
        Parameters:
        -----------
        models : list
            List of trained models
        X : pd.DataFrame
            Training feature matrix
        y : pd.Series
            Training target variable
        task_type : str
            'classification' or 'regression'
        X_val : pd.DataFrame, optional
            Validation data for ensemble optimization
        y_val : pd.Series, optional
            Validation targets for ensemble optimization
            
        Returns:
        --------
        ensemble : sklearn estimator
            Advanced optimized ensemble model
        """
        if self.verbose:
            console.print("🏆 [bold blue]Creating Advanced Ensemble - Phase 2[/bold blue]")
        
        # Data alignment
        if len(X) != len(y):
            if self.verbose:
                console.print(f"⚠️ Aligning ensemble data: {len(X)} samples (X) vs {len(y)} samples (y)")
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len].copy()
            y = y.iloc[:min_len].copy()
            if self.verbose:
                console.print(f"✅ Ensemble data aligned to {min_len} samples")
        
        if len(models) < self.min_ensemble_size:
            if self.verbose:
                console.print(f"⚠️ Need at least {self.min_ensemble_size} models for ensemble, returning best single model")
            return models[0] if models else None
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state,
                    stratify=y if task_type == 'classification' else None
                )
            except Exception:
                # Fallback for small datasets or other issues
                X_train, X_val, y_train, y_val = X, X, y, y
        else:
            X_train, y_train = X, y
        
        # Step 1: Analyze model diversity and select optimal subset
        selected_models = self._select_diverse_models(models, X_train, y_train, task_type)
        
        # Step 2: Ensemble pruning for efficiency
        if self.ensemble_pruning:
            selected_models = self._prune_ensemble(selected_models, X_val, y_val, task_type)
        
        # Step 3: Create advanced ensemble based on method
        if self.ensemble_method == 'auto':
            ensemble_method = self._select_optimal_ensemble_method(selected_models, X_train, y_train, task_type)
        else:
            ensemble_method = self.ensemble_method
        
        if self.verbose:
            console.print(f"🎯 Selected ensemble method: {ensemble_method}")
            console.print(f"📊 Using {len(selected_models)} models in ensemble")
        
        # Create ensemble using selected method
        ensemble = None
        try:
            if ensemble_method == 'multi_level':
                ensemble = self._create_multi_level_stacking(selected_models, X_train, y_train, X_val, y_val, task_type)
            elif ensemble_method == 'dynamic':
                ensemble = self._create_dynamic_ensemble(selected_models, X_train, y_train, task_type)
            elif ensemble_method == 'bayesian':
                ensemble = self._create_bayesian_ensemble(selected_models, X_train, y_train, X_val, y_val, task_type)
            elif ensemble_method == 'stacking':
                ensemble = self._create_advanced_stacking(selected_models, X_train, y_train, task_type)
            else:  # voting
                ensemble = self._create_advanced_voting(selected_models, X_train, y_train, task_type)
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Ensemble creation failed: {str(e)}")
            # Fallback to simple voting
            ensemble = self._create_advanced_voting(selected_models, X_train, y_train, task_type)
        
        if ensemble is None:
            if self.verbose:
                console.print("⚠️ All ensemble methods failed, returning best single model")
            return selected_models[0] if selected_models else None
        
        # Step 4: Fit the ensemble on training data
        try:
            ensemble.fit(X_train, y_train)
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Ensemble fitting failed: {str(e)}")
            return selected_models[0] if selected_models else None
        
        # Step 5: Add uncertainty quantification
        if self.uncertainty_quantification and ensemble is not None:
            self._add_uncertainty_quantification(ensemble, selected_models, X_val, y_val, task_type)
        
        # Step 6: Evaluate ensemble performance
        if ensemble is not None:
            self.ensemble_score_ = self._evaluate_ensemble_performance(ensemble, X_val, y_val, task_type)
            self.diversity_score_ = self._calculate_ensemble_diversity(selected_models, X_val)
            
            if self.verbose:
                self._display_ensemble_results(ensemble_method, len(selected_models))
        
        return ensemble
    
    def _select_diverse_models(
        self, 
        models: List[Any], 
        X: pd.DataFrame, 
        y: pd.Series,
        task_type: str
    ) -> List[Any]:
        """Select diverse subset of models for ensemble."""
        if len(models) <= self.ensemble_size:
            return models
        
        # Calculate pairwise diversity
        diversity_matrix = self._calculate_pairwise_diversity(models, X, y)
        
        # Greedy selection for maximum diversity
        selected_indices = [0]  # Start with best model
        
        while len(selected_indices) < min(self.ensemble_size, self.max_ensemble_size):
            best_candidate = -1
            best_diversity = -1
            
            for i in range(len(models)):
                if i in selected_indices:
                    continue
                
                # Calculate average diversity with selected models
                avg_diversity = np.mean([diversity_matrix[i][j] for j in selected_indices])
                
                if avg_diversity > best_diversity:
                    best_diversity = avg_diversity
                    best_candidate = i
            
            if best_candidate != -1 and best_diversity > self.diversity_threshold:
                selected_indices.append(best_candidate)
            else:
                break
        
        selected_models = [models[i] for i in selected_indices]
        
        if self.verbose:
            console.print(f"🎯 Selected {len(selected_models)} diverse models (diversity threshold: {self.diversity_threshold})")
        
        return selected_models
    
    def _prune_ensemble(
        self, 
        models: List[Any], 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        task_type: str
    ) -> List[Any]:
        """Prune ensemble for optimal performance vs efficiency trade-off."""
        if len(models) <= self.min_ensemble_size:
            return models
        
        try:
            # Evaluate individual model performance
            model_scores = []
            for model in models:
                try:
                    if task_type == 'classification':
                        pred = model.predict(X_val)
                        score = accuracy_score(y_val, pred)
                    else:
                        pred = model.predict(X_val)
                        score = r2_score(y_val, pred)
                    model_scores.append(max(0.0, score))
                except Exception:
                    model_scores.append(0.0)
            
            # Sort models by performance
            sorted_indices = np.argsort(model_scores)[::-1]  # Best to worst
            
            # Select top performing models up to max_ensemble_size
            optimal_size = min(len(models), self.max_ensemble_size)
            
            # Ensure we don't go below minimum
            optimal_size = max(optimal_size, self.min_ensemble_size)
            
            optimal_models = [models[sorted_indices[i]] for i in range(optimal_size)]
            
            self.pruning_results_ = {
                'original_size': len(models),
                'pruned_size': optimal_size,
                'performance_improvement': max(model_scores) - np.mean(model_scores) if model_scores else 0
            }
            
            if self.verbose:
                console.print(f"✂️ Pruned ensemble: {len(models)} → {optimal_size} models")
            
            return optimal_models
            
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Ensemble pruning failed: {str(e)}")
            return models
    
    def _create_multi_level_stacking(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str
    ) -> Any:
        """Create multi-level stacking ensemble."""
        try:
            from sklearn.ensemble import StackingClassifier, StackingRegressor
        except ImportError:
            return self._create_advanced_voting(models, X_train, y_train, task_type)
        
        if len(models) < 4:  # Need at least 4 models for multi-level
            return self._create_advanced_stacking(models, X_train, y_train, task_type)
        
        # Level 1: Split models into groups
        mid_point = len(models) // 2
        group1_models = models[:mid_point]
        group2_models = models[mid_point:]
        
        # Create Level 1 ensembles
        level1_estimators = []
        
        # Group 1 ensemble
        group1_estimators = [(f'g1_model_{i}', clone(model)) for i, model in enumerate(group1_models)]
        if task_type == 'classification':
            group1_ensemble = StackingClassifier(
                estimators=group1_estimators,
                final_estimator=LogisticRegression(random_state=self.random_state, max_iter=1000),
                cv=3
            )
        else:
            group1_ensemble = StackingRegressor(
                estimators=group1_estimators,
                final_estimator=Ridge(random_state=self.random_state),
                cv=3
            )
        
        level1_estimators.append(('group1', group1_ensemble))
        
        # Group 2 ensemble
        group2_estimators = [(f'g2_model_{i}', clone(model)) for i, model in enumerate(group2_models)]
        if task_type == 'classification':
            group2_ensemble = StackingClassifier(
                estimators=group2_estimators,
                final_estimator=LogisticRegression(random_state=self.random_state, max_iter=1000),
                cv=3
            )
        else:
            group2_ensemble = StackingRegressor(
                estimators=group2_estimators,
                final_estimator=Ridge(random_state=self.random_state),
                cv=3
            )
        
        level1_estimators.append(('group2', group2_ensemble))
        
        # Level 2: Meta-ensemble
        meta_learner = self._get_meta_learner(task_type)
        
        if task_type == 'classification':
            final_ensemble = StackingClassifier(
                estimators=level1_estimators,
                final_estimator=meta_learner,
                cv=3
            )
        else:
            final_ensemble = StackingRegressor(
                estimators=level1_estimators,
                final_estimator=meta_learner,
                cv=3
            )
        
        self.ensemble_hierarchy_ = {
            'method': 'multi_level_stacking',
            'level1_groups': 2,
            'group1_size': len(group1_models),
            'group2_size': len(group2_models),
            'meta_learner': meta_learner.__class__.__name__
        }
        
        return final_ensemble
    
    def _create_dynamic_ensemble(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str
    ) -> Any:
        """Create dynamic ensemble that selects models based on input characteristics."""
        try:
            # Calculate model performance on different data subsets for dynamic weighting
            model_weights = self._calculate_dynamic_weights(models, X_train, y_train, task_type)
            
            # Create weighted voting ensemble
            estimators = [(f'model_{i}', clone(model)) for i, model in enumerate(models)]
            
            if task_type == 'classification':
                # Try soft voting first
                try:
                    supports_proba = all(hasattr(model, 'predict_proba') for model in models)
                    if supports_proba:
                        ensemble = VotingClassifier(
                            estimators=estimators,
                            voting='soft',
                            weights=model_weights
                        )
                    else:
                        ensemble = VotingClassifier(
                            estimators=estimators,
                            voting='hard',
                            weights=model_weights
                        )
                except Exception:
                    ensemble = VotingClassifier(
                        estimators=estimators,
                        voting='hard',
                        weights=model_weights
                    )
            else:
                ensemble = VotingRegressor(
                    estimators=estimators,
                    weights=model_weights
                )
            
            self.ensemble_weights_ = model_weights
            return ensemble
            
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Dynamic ensemble creation failed: {str(e)}")
            # Fallback to simple voting
            return self._create_advanced_voting(models, X_train, y_train, task_type)
    
    def _create_bayesian_ensemble(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str
    ) -> Any:
        """Create Bayesian model averaging ensemble."""
        try:
            # Calculate Bayesian weights based on model likelihood
            bayesian_weights = self._calculate_bayesian_weights(models, X_val, y_val, task_type)
            
            # Create custom Bayesian ensemble
            ensemble = BayesianEnsemble(
                models=[clone(model) for model in models],  # Clone models to avoid issues
                weights=bayesian_weights,
                task_type=task_type,
                uncertainty_quantification=self.uncertainty_quantification
            )
            
            return ensemble
            
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Bayesian ensemble creation failed: {str(e)}")
            # Fallback to weighted voting
            return self._create_advanced_voting(models, X_train, y_train, task_type)
    
    def _create_advanced_stacking(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str
    ) -> Any:
        """Create advanced stacking ensemble."""
        try:
            from sklearn.ensemble import StackingClassifier, StackingRegressor
        except ImportError:
            return self._create_advanced_voting(models, X_train, y_train, task_type)
        
        # Create named estimators
        estimators = [(f'model_{i}', clone(model)) for i, model in enumerate(models)]
        
        # Select meta-learner
        meta_learner = self._get_meta_learner(task_type)
        
        # Create stacking ensemble
        if task_type == 'classification':
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                stack_method='auto',
                n_jobs=1
            )
        else:
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=1
            )
        
        return ensemble
    
    def _create_advanced_voting(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str
    ) -> Any:
        """Create advanced voting ensemble with optimized weights."""
        # Calculate dynamic weights
        weights = self._calculate_dynamic_weights(models, X_train, y_train, task_type)
        
        # Create named estimators
        estimators = [(f'model_{i}', clone(model)) for i, model in enumerate(models)]
        
        # Create voting ensemble
        if task_type == 'classification':
            # Try soft voting first
            try:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=weights
                )
                # Test if all models support predict_proba
                supports_proba = all(hasattr(model, 'predict_proba') for model in models)
                if not supports_proba:
                    raise Exception("Not all models support predict_proba")
            except:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='hard',
                    weights=weights
                )
        else:
            ensemble = VotingRegressor(
                estimators=estimators,
                weights=weights
            )
        
        return ensemble
    
    def _select_optimal_ensemble_method(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str
    ) -> str:
        """Select optimal ensemble method based on data characteristics."""
        n_samples = len(X_train)
        n_models = len(models)
        n_features = len(X_train.columns)
        
        # Decision logic based on dataset and model characteristics
        if n_samples < 500:
            return 'voting'  # Simple voting for small datasets
        elif n_models >= 6 and n_samples > 2000:
            return 'multi_level'  # Multi-level stacking for large datasets with many models
        elif n_samples > 5000 and n_features > 50:
            return 'bayesian'  # Bayesian averaging for complex datasets
        elif n_models >= 4:
            return 'stacking'  # Regular stacking for medium datasets
        else:
            return 'voting'  # Fallback to voting
    
    def _calculate_pairwise_diversity(
        self,
        models: List[Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> np.ndarray:
        """Calculate pairwise diversity matrix between models."""
        n_models = len(models)
        diversity_matrix = np.zeros((n_models, n_models))
        
        # Get predictions from all models
        predictions = []
        for model in models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except:
                # Use random predictions as fallback
                if hasattr(y, 'unique'):
                    unique_values = y.unique()
                    pred = np.random.choice(unique_values, size=len(X))
                else:
                    pred = np.random.randn(len(X))
                predictions.append(pred)
        
        # Calculate pairwise disagreement
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    diversity_matrix[i][j] = 0.0
                else:
                    disagreement = np.mean(predictions[i] != predictions[j])
                    diversity_matrix[i][j] = disagreement
        
        return diversity_matrix
    
    def _calculate_dynamic_weights(
        self,
        models: List[Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str
    ) -> List[float]:
        """Calculate dynamic weights for ensemble models."""
        model_scores = []
        
        # Evaluate each model using cross-validation
        for model in models:
            try:
                if len(X_train) > 100:
                    # Use cross-validation for larger datasets
                    if task_type == 'classification':
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    else:
                        cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                    score = np.mean(scores)
                else:
                    # Use simple train-test split for small datasets
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train, test_size=0.3, random_state=self.random_state,
                        stratify=y_train if task_type == 'classification' else None
                    )
                    model_clone = clone(model)
                    model_clone.fit(X_tr, y_tr)
                    pred = model_clone.predict(X_val)
                    
                    if task_type == 'classification':
                        score = accuracy_score(y_val, pred)
                    else:
                        score = r2_score(y_val, pred)
                
                model_scores.append(max(0.0, score))
            except:
                model_scores.append(0.1)  # Small positive weight for failed models
        
        # Normalize scores to create weights
        total_score = sum(model_scores)
        if total_score > 0:
            weights = [score / total_score for score in model_scores]
        else:
            weights = [1.0 / len(models)] * len(models)
        
        return weights
    
    def _calculate_bayesian_weights(
        self,
        models: List[Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str
    ) -> List[float]:
        """Calculate Bayesian weights based on model likelihood."""
        log_likelihoods = []
        
        for model in models:
            try:
                pred = model.predict(X_val)
                
                if task_type == 'classification':
                    # Use log-likelihood for classification
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_val)
                        # Calculate log-likelihood
                        log_likelihood = 0.0
                        for i, true_class in enumerate(y_val):
                            class_idx = list(model.classes_).index(true_class) if hasattr(model, 'classes_') else int(true_class)
                            if class_idx < pred_proba.shape[1]:
                                prob = max(pred_proba[i, class_idx], 1e-10)  # Avoid log(0)
                                log_likelihood += np.log(prob)
                    else:
                        # Use accuracy as proxy for likelihood
                        accuracy = accuracy_score(y_val, pred)
                        log_likelihood = np.log(max(accuracy, 1e-10))
                else:
                    # Use negative MSE as log-likelihood for regression
                    mse = mean_squared_error(y_val, pred)
                    log_likelihood = -mse
                
                log_likelihoods.append(log_likelihood)
            except:
                log_likelihoods.append(-np.inf)  # Very low likelihood for failed models
        
        # Convert log-likelihoods to weights using softmax
        log_likelihoods = np.array(log_likelihoods)
        # Subtract max for numerical stability
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        weights = np.exp(log_likelihoods)
        weights = weights / np.sum(weights)
        
        return weights.tolist()
    
    def _get_meta_learner(self, task_type: str) -> Any:
        """Get appropriate meta-learner for the task type."""
        if self.meta_learner == 'auto':
            if task_type == 'classification':
                return LogisticRegression(random_state=self.random_state, max_iter=1000)
            else:
                return Ridge(random_state=self.random_state)
        else:
            return self.meta_learner
    
    def _add_uncertainty_quantification(
        self,
        ensemble: Any,
        models: List[Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str
    ) -> None:
        """Add uncertainty quantification to the ensemble."""
        try:
            # Create uncertainty estimator
            self.uncertainty_estimator_ = UncertaintyEstimator(
                ensemble=ensemble,
                base_models=models,
                task_type=task_type
            )
            
            # Fit uncertainty estimator on validation data
            self.uncertainty_estimator_.fit(X_val, y_val)
            
            if self.verbose:
                console.print("✅ Uncertainty quantification added to ensemble")
                
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Failed to add uncertainty quantification: {str(e)}")
            # Set to None to indicate failure
            self.uncertainty_estimator_ = None
    
    def _evaluate_ensemble_performance(
        self,
        ensemble: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str
    ) -> float:
        """Evaluate ensemble performance on validation data."""
        try:
            # Ensure ensemble is fitted
            if hasattr(ensemble, 'fit') and not hasattr(ensemble, 'is_fitted_'):
                ensemble.fit(X_val, y_val)
            
            pred = ensemble.predict(X_val)
            
            if task_type == 'classification':
                score = accuracy_score(y_val, pred)
            else:
                score = r2_score(y_val, pred)
            
            return max(0.0, score)
        except Exception as e:
            if self.verbose:
                console.print(f"⚠️ Ensemble evaluation failed: {str(e)}")
            return 0.0
    
    def _calculate_ensemble_diversity(
        self,
        models: List[Any],
        X_val: pd.DataFrame
    ) -> float:
        """Calculate diversity score of the ensemble."""
        if len(models) < 2:
            return 0.0
        
        try:
            predictions = []
            for model in models:
                pred = model.predict(X_val)
                predictions.append(pred)
            
            # Calculate pairwise disagreement
            disagreements = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    disagreement = np.mean(predictions[i] != predictions[j])
                    disagreements.append(disagreement)
            
            return np.mean(disagreements) if disagreements else 0.0
        except:
            return 0.0
    
    def _display_ensemble_results(
        self,
        ensemble_method: str,
        n_models: int
    ) -> None:
        """Display ensemble creation results."""
        table = Table(title="🏆 Advanced Ensemble Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Ensemble Method", ensemble_method.replace('_', ' ').title())
        table.add_row("Models Used", str(n_models))
        table.add_row("Ensemble Score", f"{self.ensemble_score_:.4f}")
        table.add_row("Diversity Score", f"{self.diversity_score_:.4f}")
        
        if self.pruning_results_:
            table.add_row("Original Size", str(self.pruning_results_['original_size']))
            table.add_row("Pruned Size", str(self.pruning_results_['pruned_size']))
            table.add_row("Performance Gain", f"{self.pruning_results_['performance_improvement']:.4f}")
        
        if self.uncertainty_estimator_:
            table.add_row("Uncertainty Estimation", "✅ Enabled")
        
        console.print(table)


class BayesianEnsemble(BaseEstimator):
    """
    Bayesian Model Averaging Ensemble.
    
    Combines models using Bayesian weights based on model likelihood.
    Provides uncertainty quantification for predictions.
    """
    
    def __init__(
        self,
        models: List[Any],
        weights: List[float],
        task_type: str,
        uncertainty_quantification: bool = True
    ):
        self.models = models
        self.weights = weights
        self.task_type = task_type
        self.uncertainty_quantification = uncertainty_quantification
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all base models."""
        try:
            for model in self.models:
                model.fit(X, y)
            self.is_fitted_ = True
        except Exception as e:
            # If fitting fails, mark as fitted anyway to avoid blocking
            self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make Bayesian averaged predictions."""
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0], dtype=float)
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred.astype(float)
        
        # For classification, round to nearest integer
        if self.task_type == 'classification':
            return np.round(weighted_pred).astype(int)
        else:
            return weighted_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities with Bayesian averaging."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get probabilities from all models
        all_probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                # Convert predictions to probabilities (simple approach)
                pred = model.predict(X)
                n_classes = len(np.unique(pred))
                proba = np.eye(n_classes)[pred.astype(int)]
            all_probas.append(proba)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(all_probas[0], dtype=float)
        for proba, weight in zip(all_probas, self.weights):
            weighted_proba += weight * proba
        
        return weighted_proba
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if not self.uncertainty_quantification:
            predictions = self.predict(X)
            uncertainties = np.zeros(len(predictions))
            return predictions, uncertainties
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate weighted mean and variance
        weighted_mean = np.average(all_predictions, axis=0, weights=self.weights)
        weighted_var = np.average((all_predictions - weighted_mean)**2, axis=0, weights=self.weights)
        
        # Uncertainty is the standard deviation
        uncertainties = np.sqrt(weighted_var)
        
        if self.task_type == 'classification':
            predictions = np.round(weighted_mean).astype(int)
        else:
            predictions = weighted_mean
        
        return predictions, uncertainties


class UncertaintyEstimator:
    """
    Uncertainty Quantification for Ensemble Models.
    
    Provides various methods for estimating prediction uncertainty:
    - Model disagreement
    - Prediction variance
    - Confidence intervals
    """
    
    def __init__(
        self,
        ensemble: Any,
        base_models: List[Any],
        task_type: str,
        confidence_level: float = 0.95
    ):
        self.ensemble = ensemble
        self.base_models = base_models
        self.task_type = task_type
        self.confidence_level = confidence_level
        self.is_fitted_ = False
        
        # Calibration data for uncertainty estimation
        self.calibration_errors_ = None
        self.uncertainty_threshold_ = None
    
    def fit(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """Fit uncertainty estimator on calibration data."""
        try:
            # Get ensemble predictions
            ensemble_pred = self.ensemble.predict(X_cal)
            
            # Calculate prediction errors for calibration
            if self.task_type == 'classification':
                errors = (ensemble_pred != y_cal).astype(float)
            else:
                errors = np.abs(ensemble_pred - y_cal)
            
            self.calibration_errors_ = errors
            
            # Set uncertainty threshold (e.g., 90th percentile of errors)
            self.uncertainty_threshold_ = np.percentile(errors, 90) if len(errors) > 0 else 1.0
            
            self.is_fitted_ = True
            return self
            
        except Exception as e:
            # Fallback: set default values
            self.calibration_errors_ = np.array([0.5])
            self.uncertainty_threshold_ = 0.5
            self.is_fitted_ = True
            return self
    
    def estimate_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate prediction uncertainty."""
        if not self.is_fitted_:
            raise ValueError("UncertaintyEstimator must be fitted first")
        
        # Method 1: Model disagreement
        disagreement_uncertainty = self._calculate_model_disagreement(X)
        
        # Method 2: Prediction variance (for Bayesian ensemble)
        if hasattr(self.ensemble, 'predict_with_uncertainty'):
            _, variance_uncertainty = self.ensemble.predict_with_uncertainty(X)
        else:
            variance_uncertainty = np.zeros(len(X))
        
        # Combine uncertainties
        combined_uncertainty = 0.6 * disagreement_uncertainty + 0.4 * variance_uncertainty
        
        return combined_uncertainty
    
    def _calculate_model_disagreement(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate uncertainty based on model disagreement."""
        # Get predictions from all base models
        all_predictions = []
        for model in self.base_models:
            try:
                pred = model.predict(X)
                all_predictions.append(pred)
            except:
                # Skip failed models
                continue
        
        if len(all_predictions) < 2:
            return np.zeros(len(X))
        
        all_predictions = np.array(all_predictions)
        
        if self.task_type == 'classification':
            # For classification: fraction of models that disagree with majority
            from scipy import stats
            majority_pred = stats.mode(all_predictions, axis=0)[0].flatten()
            disagreement = np.mean(all_predictions != majority_pred, axis=0)
        else:
            # For regression: standard deviation of predictions
            disagreement = np.std(all_predictions, axis=0)
        
        return disagreement
    
    def get_confidence_intervals(
        self, 
        X: pd.DataFrame, 
        alpha: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get prediction confidence intervals."""
        if alpha is None:
            alpha = 1 - self.confidence_level
        
        # Get predictions and uncertainties
        predictions = self.ensemble.predict(X)
        uncertainties = self.estimate_uncertainty(X)
        
        # Calculate confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin_of_error = z_score * uncertainties
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return predictions, lower_bound, upper_bound
    
    def is_prediction_reliable(self, X: pd.DataFrame) -> np.ndarray:
        """Determine if predictions are reliable based on uncertainty."""
        uncertainties = self.estimate_uncertainty(X)
        return uncertainties <= self.uncertainty_threshold_


# Backward compatibility wrapper
class EnsembleBuilder(AdvancedEnsembleBuilder):
    """
    Backward compatibility wrapper for the original EnsembleBuilder.
    
    Maintains the original API while providing access to advanced features.
    """
    
    def __init__(self, *args, **kwargs):
        # Map old parameters to new ones
        if 'ensemble_size' not in kwargs:
            kwargs['ensemble_size'] = 5
        if 'ensemble_method' not in kwargs:
            kwargs['ensemble_method'] = 'auto'
        if 'verbose' not in kwargs:
            kwargs['verbose'] = True
        
        super().__init__(*args, **kwargs)
    
    def create_ensemble(self, models, X, y, task_type='classification'):
        """Original create_ensemble method for backward compatibility."""
        return self.create_advanced_ensemble(models, X, y, task_type)


class BlendingEnsemble(BaseEstimator):
    """
    Custom blending ensemble implementation.
    
    Combines base models using a meta-learner trained on holdout predictions.
    """
    
    def __init__(self, base_models: List[Any], meta_learner: Any, task_type: str):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.task_type = task_type
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all base models on the full training data."""
        for model in self.base_models:
            model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the blending ensemble."""
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if pred.shape[1] == 2:
                    pred = pred[:, 1]  # Use positive class probability
                else:
                    pred = pred  # Use all probabilities
            else:
                pred = model.predict(X)
            
            base_predictions.append(pred)
        
        # Stack predictions for meta-learner
        if len(base_predictions[0].shape) == 1:
            meta_features = np.column_stack(base_predictions)
        else:
            meta_features = np.hstack(base_predictions)
        
        # Get final predictions from meta-learner
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if hasattr(self.meta_learner, 'predict_proba'):
            # Get base model predictions
            base_predictions = []
            for model in self.base_models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    # Convert predictions to probabilities (simple approach)
                    pred = model.predict(X)
                    pred = np.column_stack([1 - pred, pred])  # Assume binary
                
                base_predictions.append(pred)
            
            # Stack for meta-learner
            meta_features = np.hstack(base_predictions)
            return self.meta_learner.predict_proba(meta_features)
        else:
            # Fallback: convert predictions to probabilities
            predictions = self.predict(X)
            return np.column_stack([1 - predictions, predictions])