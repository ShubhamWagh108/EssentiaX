"""
EssentiaX EDA Module - Unified Smart EDA Engine
"""

from .smart_eda import smart_eda, problem_card, smart_eda_pro, smart_viz, smart_eda_legacy

__all__ = [
    'smart_eda',           # Main unified function
    'problem_card',        # Legacy compatibility
    'smart_eda_pro',       # Legacy compatibility  
    'smart_viz',           # Legacy compatibility
    'smart_eda_legacy'     # Legacy compatibility
]

__all__ = ["smart_eda"]
