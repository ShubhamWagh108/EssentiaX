"""
EssentiaX AutoML Module
======================

Intelligent Automated Machine Learning that builds upon Essentiax's 
feature engineering and EDA capabilities.

Main Components:
- AutoML: Main automated machine learning class
- ModelSelector: Intelligent model selection
- HyperOptimizer: Advanced hyperparameter optimization
- EnsembleBuilder: Automated ensemble creation (optional)
"""

# Import core components individually to avoid circular imports
from .core.model_selector import ModelSelector
from .core.hyperopt import HyperOptimizer

# Optional ensemble import
try:
    from .core.ensemble import EnsembleBuilder
    ENSEMBLE_AVAILABLE = True
except ImportError:
    EnsembleBuilder = None
    ENSEMBLE_AVAILABLE = False

# Import AutoML last to avoid circular dependency
from .core.auto_ml import AutoML

# Build __all__ list
__all__ = ['AutoML', 'ModelSelector', 'HyperOptimizer']

if ENSEMBLE_AVAILABLE:
    __all__.append('EnsembleBuilder')