"""
EssentiaX EDA Module - Unified Smart EDA Engine
"""

from .smart_eda import smart_eda, problem_card, smart_eda_pro, smart_viz, smart_eda_legacy

class SmartEDA:
    """
    SmartEDA class wrapper for sklearn-style usage.
    
    Usage:
        eda = SmartEDA()
        report = eda.analyze(df, target_column='target')
    """
    def __init__(self, mode='auto', output_format='html', verbose=True):
        self.mode = mode
        self.output_format = output_format
        self.verbose = verbose
        self.report = None
    
    def analyze(self, df, target_column=None, output_file=None):
        """Perform EDA analysis"""
        self.report = smart_eda(
            df, 
            target_column=target_column,
            mode=self.mode,
            output_format=self.output_format,
            output_file=output_file,
            verbose=self.verbose
        )
        return self.report
    
    def fit(self, df, target_column=None):
        """Fit method (for sklearn compatibility)"""
        return self.analyze(df, target_column)

__all__ = [
    'smart_eda',
    'SmartEDA',
    'problem_card',
    'smart_eda_pro',  
    'smart_viz',
    'smart_eda_legacy'
]
