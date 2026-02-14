"""
🚀 EssentiaX Feature Engineering Module
=======================================

The most advanced and accurate feature engineering library for Python.

Key Features:
- AI-Powered Feature Selection
- Automated Feature Generation  
- Advanced Statistical Transformations
- Real-time Feature Quality Assessment
- Interactive Feature Engineering Pipeline

Usage:
    from essentiax.feature_engineering import FeatureEngineer, smart_features
    
    # One-line feature engineering
    fe = FeatureEngineer()
    X_transformed = fe.fit_transform(X, y)
    
    # Smart feature engineering with AI
    X_smart = smart_features(X, y, mode='auto')
"""

from .core.feature_engineer import FeatureEngineer
from .core.smart_features import smart_features, quick_features, comprehensive_features, ml_ready_features
from .core.pipeline_builder import PipelineBuilder
from .transformers.numerical import NumericalTransformer
from .transformers.categorical import CategoricalTransformer
from .selectors.smart_selector import SmartFeatureSelector
from .utils.metrics import FeatureQualityMetrics

__version__ = "1.0.0"

__all__ = [
    "FeatureEngineer",
    "smart_features", 
    "quick_features",
    "comprehensive_features", 
    "ml_ready_features",
    "PipelineBuilder",
    "NumericalTransformer",
    "CategoricalTransformer", 
    "SmartFeatureSelector",
    "FeatureQualityMetrics"
]