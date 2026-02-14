from .io.smart_read import smart_read
from .summary.problem_card import problem_card
from .summary.eda_pro import smart_eda_pro
from .visuals.smartViz import smart_viz
from .cleaning.smart_clean import smart_clean
from .eda import smart_eda

# Feature Engineering Module
from .feature_engineering import (
    FeatureEngineer,
    smart_features,
    quick_features,
    comprehensive_features,
    ml_ready_features
)

# AutoML Module (optional - graceful fallback if dependencies missing)
try:
    from .automl import AutoML
    AUTOML_AVAILABLE = True
except ImportError as e:
    AutoML = None
    AUTOML_AVAILABLE = False

# Build __all__ list dynamically
__all__ = [
    "smart_read",
    "problem_card", 
    "smart_eda_pro",
    "smart_viz",
    "smart_clean",
    "smart_eda",
    # Feature Engineering
    "FeatureEngineer",
    "smart_features",
    "quick_features", 
    "comprehensive_features",
    "ml_ready_features"
]

# Add AutoML if available
if AUTOML_AVAILABLE:
    __all__.append("AutoML")
