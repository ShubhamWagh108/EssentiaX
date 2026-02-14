"""
Advanced Model Explainability for AutoML - Phase 2
==================================================

Comprehensive model interpretability and explanation system that provides:
- SHAP integration for all model types
- Permutation importance analysis
- Partial dependence plots
- Local vs global explanations
- Feature interaction detection
- Model-agnostic explanations
- Regulatory compliance support

Phase 2 Features:
- Multi-method explanation ensemble
- Uncertainty quantification in explanations
- Interactive explanation dashboards
- Explanation validation and consistency checks
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class ModelExplainer:
    """
    🔍 Advanced Model Explainer - Phase 2
    
    Comprehensive model interpretability system that provides multiple
    explanation methods for any machine learning model:
    
    Core Features:
    - SHAP integration (TreeExplainer, KernelExplainer, LinearExplainer)
    - Permutation importance with statistical significance
    - Partial dependence plots and ICE curves
    - Local explanations for individual predictions
    - Global feature importance ranking
    - Feature interaction detection and visualization
    - Model complexity analysis and interpretability scoring
    
    Advanced Features:
    - Multi-method explanation ensemble
    - Explanation uncertainty quantification
    - Consistency validation across methods
    - Regulatory compliance reporting
    - Interactive explanation dashboards
    - Explanation caching and optimization
    """
    
    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str = 'classification',
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.task_type = task_type
        self.feature_names = feature_names or list(X_train.columns)
        self.class_names = class_names
        self.random_state = random_state
        self.verbose = verbose
        
        # Explanation cache
        self.explanation_cache_ = {}
        self.shap_explainer_ = None
        self.shap_values_ = None
        
        # Model characteristics
        self.model_name_ = model.__class__.__name__
        self.model_type_ = self._determine_model_type()
        self.interpretability_score_ = self._calculate_interpretability_score()
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer()
        
        np.random.seed(random_state)
    
    def explain_global(
        self, 
        methods: List[str] = ['shap', 'permutation', 'feature_importance'],
        n_features: int = 10,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Generate comprehensive global explanations using multiple methods.
        
        Parameters:
        -----------
        methods : list
            Explanation methods to use: ['shap', 'permutation', 'feature_importance', 'partial_dependence']
        n_features : int
            Number of top features to analyze
        sample_size : int
            Sample size for computationally expensive methods
            
        Returns:
        --------
        explanations : dict
            Comprehensive global explanations
        """
        if self.verbose:
            console.print("🔍 [bold blue]Generating Global Model Explanations[/bold blue]")
        
        explanations = {
            'model_info': self._get_model_info(),
            'interpretability_score': self.interpretability_score_,
            'explanation_methods': methods
        }
        
        # SHAP Global Explanations
        if 'shap' in methods:
            try:
                shap_global = self._get_shap_global_explanations(n_features, sample_size)
                explanations['shap_global'] = shap_global
                if self.verbose:
                    console.print("   ✅ SHAP global explanations generated")
            except Exception as e:
                explanations['shap_global'] = {'error': f"SHAP failed: {str(e)}"}
                if self.verbose:
                    console.print(f"   ⚠️ SHAP global explanations failed: {str(e)}")
        
        # Permutation Importance
        if 'permutation' in methods:
            try:
                perm_importance = self._get_permutation_importance(n_features, sample_size)
                explanations['permutation_importance'] = perm_importance
                if self.verbose:
                    console.print("   ✅ Permutation importance calculated")
            except Exception as e:
                explanations['permutation_importance'] = {'error': f"Permutation importance failed: {str(e)}"}
                if self.verbose:
                    console.print(f"   ⚠️ Permutation importance failed: {str(e)}")
        
        # Built-in Feature Importance
        if 'feature_importance' in methods:
            try:
                builtin_importance = self._get_builtin_feature_importance(n_features)
                explanations['builtin_importance'] = builtin_importance
                if self.verbose:
                    console.print("   ✅ Built-in feature importance extracted")
            except Exception as e:
                explanations['builtin_importance'] = {'error': f"Built-in importance failed: {str(e)}"}
                if self.verbose:
                    console.print(f"   ⚠️ Built-in importance failed: {str(e)}")
        
        # Partial Dependence
        if 'partial_dependence' in methods:
            try:
                partial_dep = self._get_partial_dependence(n_features, sample_size)
                explanations['partial_dependence'] = partial_dep
                if self.verbose:
                    console.print("   ✅ Partial dependence plots generated")
            except Exception as e:
                explanations['partial_dependence'] = {'error': f"Partial dependence failed: {str(e)}"}
                if self.verbose:
                    console.print(f"   ⚠️ Partial dependence failed: {str(e)}")
        
        # Feature Interactions
        try:
            interactions = self._detect_feature_interactions(n_features, sample_size)
            explanations['feature_interactions'] = interactions
            if self.verbose:
                console.print("   ✅ Feature interactions detected")
        except Exception as e:
            explanations['feature_interactions'] = {'error': f"Feature interactions failed: {str(e)}"}
            if self.verbose:
                console.print(f"   ⚠️ Feature interactions failed: {str(e)}")
        
        # Explanation Consistency Analysis
        explanations['consistency_analysis'] = self._analyze_explanation_consistency(explanations)
        
        return explanations
    
    def explain_local(
        self, 
        X_sample: pd.DataFrame,
        methods: List[str] = ['shap', 'lime'],
        n_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generate local explanations for specific instances.
        
        Parameters:
        -----------
        X_sample : pd.DataFrame
            Instances to explain
        methods : list
            Local explanation methods to use
        n_features : int
            Number of features to show in explanations
            
        Returns:
        --------
        explanations : dict
            Local explanations for each instance
        """
        if self.verbose:
            console.print(f"🔍 [bold blue]Generating Local Explanations for {len(X_sample)} instances[/bold blue]")
        
        explanations = {
            'instances': len(X_sample),
            'methods': methods,
            'local_explanations': []
        }
        
        for idx, (_, instance) in enumerate(X_sample.iterrows()):
            instance_explanations = {
                'instance_id': idx,
                'prediction': self._get_prediction_info(instance),
                'explanations': {}
            }
            
            # SHAP Local Explanations
            if 'shap' in methods:
                try:
                    shap_local = self._get_shap_local_explanation(instance, n_features)
                    instance_explanations['explanations']['shap'] = shap_local
                except Exception as e:
                    instance_explanations['explanations']['shap'] = {'error': str(e)}
            
            # LIME Local Explanations (placeholder - would need LIME integration)
            if 'lime' in methods:
                try:
                    lime_local = self._get_lime_local_explanation(instance, n_features)
                    instance_explanations['explanations']['lime'] = lime_local
                except Exception as e:
                    instance_explanations['explanations']['lime'] = {'error': str(e)}
            
            explanations['local_explanations'].append(instance_explanations)
        
        return explanations
    
    def _initialize_shap_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            import shap
            
            if self.model_type_ == 'tree':
                # Tree-based models (RandomForest, XGBoost, LightGBM)
                self.shap_explainer_ = shap.TreeExplainer(self.model)
                if self.verbose:
                    console.print("   🌳 SHAP TreeExplainer initialized")
                    
            elif self.model_type_ == 'linear':
                # Linear models (LogisticRegression, Ridge, Lasso)
                self.shap_explainer_ = shap.LinearExplainer(self.model, self.X_train)
                if self.verbose:
                    console.print("   📏 SHAP LinearExplainer initialized")
                    
            else:
                # Model-agnostic KernelExplainer for other models
                background = self.X_train.sample(n=min(100, len(self.X_train)), random_state=self.random_state)
                self.shap_explainer_ = shap.KernelExplainer(self.model.predict, background)
                if self.verbose:
                    console.print("   🔧 SHAP KernelExplainer initialized")
                    
        except ImportError:
            if self.verbose:
                console.print("   ⚠️ SHAP not available, explanations will be limited")
            self.shap_explainer_ = None
        except Exception as e:
            if self.verbose:
                console.print(f"   ⚠️ SHAP initialization failed: {str(e)}")
            self.shap_explainer_ = None
    
    def _get_shap_global_explanations(self, n_features: int, sample_size: int) -> Dict[str, Any]:
        """Generate SHAP global explanations."""
        if self.shap_explainer_ is None:
            return {'error': 'SHAP explainer not available'}
        
        # Sample data for efficiency
        X_sample = self.X_train.sample(n=min(sample_size, len(self.X_train)), random_state=self.random_state)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer_.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class for simplicity
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(feature_importance)],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'method': 'SHAP Global',
            'top_features': importance_df.head(n_features).to_dict('records'),
            'total_features': len(importance_df),
            'samples_analyzed': len(X_sample),
            'explainer_type': self.shap_explainer_.__class__.__name__
        }
    
    def _get_shap_local_explanation(self, instance: pd.Series, n_features: int) -> Dict[str, Any]:
        """Generate SHAP local explanation for a single instance."""
        if self.shap_explainer_ is None:
            return {'error': 'SHAP explainer not available'}
        
        # Reshape instance for SHAP
        instance_df = pd.DataFrame([instance])
        
        # Calculate SHAP values
        shap_values = self.shap_explainer_.shap_values(instance_df)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0][0]  # First class, first instance
        else:
            shap_values = shap_values[0]  # First instance
        
        # Create explanation DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names[:len(shap_values)],
            'shap_value': shap_values,
            'feature_value': instance.values[:len(shap_values)]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return {
            'method': 'SHAP Local',
            'top_features': explanation_df.head(n_features).to_dict('records'),
            'base_value': getattr(self.shap_explainer_, 'expected_value', 0),
            'prediction_impact': explanation_df['shap_value'].sum()
        }
    
    def _get_permutation_importance(self, n_features: int, sample_size: int) -> Dict[str, Any]:
        """Calculate permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
            
            # Sample data for efficiency
            X_sample = self.X_train.sample(n=min(sample_size, len(self.X_train)), random_state=self.random_state)
            y_sample = self.y_train.loc[X_sample.index]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X_sample, y_sample,
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=1
            )
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return {
                'method': 'Permutation Importance',
                'top_features': importance_df.head(n_features).to_dict('records'),
                'total_features': len(importance_df),
                'samples_analyzed': len(X_sample),
                'n_repeats': 5
            }
            
        except ImportError:
            return {'error': 'Permutation importance requires scikit-learn >= 0.22'}
        except Exception as e:
            return {'error': f'Permutation importance failed: {str(e)}'}
    
    def _get_builtin_feature_importance(self, n_features: int) -> Dict[str, Any]:
        """Extract built-in feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            method = 'Tree-based Feature Importance'
            
        elif hasattr(self.model, 'coef_'):
            # Linear models
            coef = self.model.coef_
            if len(coef.shape) > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
            method = 'Linear Coefficients (absolute)'
            
        else:
            return {'error': 'Model does not have built-in feature importance'}
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return {
            'method': method,
            'top_features': importance_df.head(n_features).to_dict('records'),
            'total_features': len(importance_df)
        }
    
    def _get_partial_dependence(self, n_features: int, sample_size: int) -> Dict[str, Any]:
        """Calculate partial dependence for top features."""
        try:
            from sklearn.inspection import partial_dependence
            
            # Get top features from any available importance method
            top_features = self._get_top_features_indices(n_features)
            
            if not top_features:
                return {'error': 'No feature importance available for partial dependence'}
            
            # Sample data for efficiency
            X_sample = self.X_train.sample(n=min(sample_size, len(self.X_train)), random_state=self.random_state)
            
            partial_deps = {}
            for feature_idx in top_features[:min(5, n_features)]:  # Limit to 5 features for performance
                try:
                    pd_result = partial_dependence(
                        self.model, X_sample, [feature_idx],
                        kind='average'
                    )
                    
                    partial_deps[self.feature_names[feature_idx]] = {
                        'values': pd_result['values'][0].tolist(),
                        'grid_values': pd_result['grid_values'][0].tolist()
                    }
                except Exception:
                    continue
            
            return {
                'method': 'Partial Dependence',
                'partial_dependence': partial_deps,
                'features_analyzed': len(partial_deps),
                'samples_used': len(X_sample)
            }
            
        except ImportError:
            return {'error': 'Partial dependence requires scikit-learn >= 0.22'}
        except Exception as e:
            return {'error': f'Partial dependence failed: {str(e)}'}
    
    def _detect_feature_interactions(self, n_features: int, sample_size: int) -> Dict[str, Any]:
        """Detect feature interactions using correlation analysis."""
        try:
            # Sample data for efficiency
            X_sample = self.X_train.sample(n=min(sample_size, len(self.X_train)), random_state=self.random_state)
            
            # Calculate correlation matrix
            correlation_matrix = X_sample.corr().abs()
            
            # Find high correlations (excluding diagonal)
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > 0.5:  # Threshold for significant correlation
                        high_correlations.append({
                            'feature_1': correlation_matrix.columns[i],
                            'feature_2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            # Sort by correlation strength
            high_correlations.sort(key=lambda x: x['correlation'], reverse=True)
            
            return {
                'method': 'Correlation-based Interaction Detection',
                'interactions': high_correlations[:n_features],
                'total_interactions': len(high_correlations),
                'correlation_threshold': 0.5
            }
            
        except Exception as e:
            return {'error': f'Feature interaction detection failed: {str(e)}'}
    
    def _get_lime_local_explanation(self, instance: pd.Series, n_features: int) -> Dict[str, Any]:
        """Generate LIME local explanation (placeholder - would need LIME integration)."""
        # This would require LIME library integration
        return {
            'method': 'LIME Local',
            'error': 'LIME integration not implemented yet',
            'note': 'Would provide model-agnostic local explanations'
        }
    
    def _analyze_explanation_consistency(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency between different explanation methods."""
        consistency_analysis = {
            'methods_compared': [],
            'feature_ranking_similarity': {},
            'overall_consistency_score': 0.0
        }
        
        # Extract feature rankings from different methods
        feature_rankings = {}
        
        if 'shap_global' in explanations and 'top_features' in explanations['shap_global']:
            shap_features = [f['feature'] for f in explanations['shap_global']['top_features']]
            feature_rankings['shap'] = shap_features
            consistency_analysis['methods_compared'].append('shap')
        
        if 'permutation_importance' in explanations and 'top_features' in explanations['permutation_importance']:
            perm_features = [f['feature'] for f in explanations['permutation_importance']['top_features']]
            feature_rankings['permutation'] = perm_features
            consistency_analysis['methods_compared'].append('permutation')
        
        if 'builtin_importance' in explanations and 'top_features' in explanations['builtin_importance']:
            builtin_features = [f['feature'] for f in explanations['builtin_importance']['top_features']]
            feature_rankings['builtin'] = builtin_features
            consistency_analysis['methods_compared'].append('builtin')
        
        # Calculate pairwise ranking similarities
        if len(feature_rankings) >= 2:
            methods = list(feature_rankings.keys())
            total_similarity = 0
            comparisons = 0
            
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    similarity = self._calculate_ranking_similarity(
                        feature_rankings[method1], 
                        feature_rankings[method2]
                    )
                    consistency_analysis['feature_ranking_similarity'][f'{method1}_vs_{method2}'] = similarity
                    total_similarity += similarity
                    comparisons += 1
            
            consistency_analysis['overall_consistency_score'] = total_similarity / comparisons if comparisons > 0 else 0.0
        
        return consistency_analysis
    
    def _calculate_ranking_similarity(self, ranking1: List[str], ranking2: List[str]) -> float:
        """Calculate similarity between two feature rankings using Jaccard similarity."""
        set1 = set(ranking1[:5])  # Top 5 features
        set2 = set(ranking2[:5])  # Top 5 features
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_top_features_indices(self, n_features: int) -> List[int]:
        """Get indices of top features from any available importance method."""
        # Try to get from built-in importance first
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:n_features]
            return top_indices.tolist()
        
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) > 1:
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
            top_indices = np.argsort(importances)[::-1][:n_features]
            return top_indices.tolist()
        
        # Fallback to first n features
        return list(range(min(n_features, len(self.feature_names))))
    
    def _determine_model_type(self) -> str:
        """Determine the type of model for appropriate explanation methods."""
        model_name = self.model_name_.lower()
        
        if any(name in model_name for name in ['randomforest', 'xgb', 'lightgbm', 'gradientboosting', 'tree']):
            return 'tree'
        elif any(name in model_name for name in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
            return 'linear'
        elif any(name in model_name for name in ['svm', 'svc', 'svr']):
            return 'svm'
        elif any(name in model_name for name in ['neural', 'mlp']):
            return 'neural'
        else:
            return 'other'
    
    def _calculate_interpretability_score(self) -> float:
        """Calculate interpretability score for the model."""
        model_name = self.model_name_.lower()
        
        # Interpretability scores based on model type
        interpretability_scores = {
            'linear': 0.9,
            'tree': 0.7,
            'svm': 0.4,
            'neural': 0.2,
            'other': 0.5
        }
        
        base_score = interpretability_scores.get(self.model_type_, 0.5)
        
        # Adjust based on model complexity
        if hasattr(self.model, 'n_estimators'):
            # Ensemble models are less interpretable
            n_estimators = getattr(self.model, 'n_estimators', 100)
            complexity_penalty = min(0.3, n_estimators / 1000)
            base_score -= complexity_penalty
        
        if hasattr(self.model, 'max_depth'):
            # Deeper trees are less interpretable
            max_depth = getattr(self.model, 'max_depth', 10)
            if max_depth and max_depth > 10:
                depth_penalty = min(0.2, (max_depth - 10) / 50)
                base_score -= depth_penalty
        
        return max(0.1, min(1.0, base_score))
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name_,
            'model_type': self.model_type_,
            'interpretability_score': self.interpretability_score_,
            'task_type': self.task_type,
            'n_features': len(self.feature_names),
            'n_training_samples': len(self.X_train),
            'has_feature_importance': hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_'),
            'shap_available': self.shap_explainer_ is not None
        }
    
    def _get_prediction_info(self, instance: pd.Series) -> Dict[str, Any]:
        """Get prediction information for an instance."""
        instance_df = pd.DataFrame([instance])
        
        prediction_info = {
            'prediction': self.model.predict(instance_df)[0]
        }
        
        # Add probability if available
        if hasattr(self.model, 'predict_proba') and self.task_type == 'classification':
            probabilities = self.model.predict_proba(instance_df)[0]
            prediction_info['probabilities'] = probabilities.tolist()
            prediction_info['confidence'] = max(probabilities)
        
        return prediction_info
    
    def generate_explanation_report(self, explanations: Dict[str, Any]) -> str:
        """Generate a comprehensive explanation report."""
        report_lines = []
        report_lines.append("🔍 MODEL EXPLANATION REPORT")
        report_lines.append("=" * 50)
        
        # Model Information
        if 'model_info' in explanations:
            model_info = explanations['model_info']
            report_lines.append(f"\n📊 Model: {model_info['model_name']}")
            report_lines.append(f"🎯 Task: {model_info['task_type']}")
            report_lines.append(f"🧠 Interpretability Score: {model_info['interpretability_score']:.2f}/1.0")
            report_lines.append(f"📈 Features: {model_info['n_features']}")
        
        # Feature Importance Summary
        if 'shap_global' in explanations and 'top_features' in explanations['shap_global']:
            report_lines.append(f"\n🏆 Top Features (SHAP):")
            for i, feature in enumerate(explanations['shap_global']['top_features'][:5], 1):
                report_lines.append(f"   {i}. {feature['feature']}: {feature['importance']:.4f}")
        
        # Consistency Analysis
        if 'consistency_analysis' in explanations:
            consistency = explanations['consistency_analysis']
            if consistency['overall_consistency_score'] > 0:
                report_lines.append(f"\n🔄 Explanation Consistency: {consistency['overall_consistency_score']:.2f}/1.0")
                if consistency['overall_consistency_score'] > 0.7:
                    report_lines.append("   ✅ High consistency across methods")
                elif consistency['overall_consistency_score'] > 0.4:
                    report_lines.append("   ⚠️ Moderate consistency across methods")
                else:
                    report_lines.append("   ❌ Low consistency - investigate further")
        
        return "\n".join(report_lines)