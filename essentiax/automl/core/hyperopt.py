"""
Advanced Hyperparameter Optimization for AutoML
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import cross_val_score
from rich.console import Console

console = Console()

class HyperOptimizer:
    """Hyperparameter optimization engine."""
    
    def __init__(self, metric='accuracy', cv_folds=5, time_budget=3600, 
                 random_state=42, n_jobs=-1, verbose=True):
        self.metric = metric
        self.cv_folds = cv_folds
        self.time_budget = time_budget
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_params_ = {}
        self.best_scores_ = {}
    
    def optimize_models(self, candidate_models, X, y):
        """Optimize hyperparameters for candidate models."""
        model_rankings = []
        
        if self.verbose:
            console.print(f"🎯 Evaluating {len(candidate_models)} models")
        
        for model_info in candidate_models:
            try:
                model_class = model_info['model_class']
                base_params = model_info.get('params', {})
                model = model_class(**base_params)
                
                score = self._evaluate_model(model, X, y)
                
                model_rankings.append({
                    'model': model,
                    'score': score,
                    'name': model_info['name'],
                    'params': base_params
                })
                
                if self.verbose:
                    console.print(f"✅ {model_info['name']}: {score:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    console.print(f"⚠️ Failed: {model_info['name']}: {str(e)}")
                continue
        
        model_rankings.sort(key=lambda x: x['score'], reverse=True)
        return model_rankings
    
    def _evaluate_model(self, model, X, y):
        """Evaluate model using cross-validation."""
        try:
            # Determine scoring based on task type
            if hasattr(self, 'task'):
                scoring = 'r2' if self.task == 'regression' else 'accuracy'
            else:
                scoring = 'accuracy' if self.metric == 'accuracy' else 'r2'
            
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=1)
            mean_score = scores.mean()
            
            # Return the score as-is (don't force positive)
            return mean_score if not (np.isnan(mean_score) or np.isinf(mean_score)) else 0.0
        except:
            return 0.0