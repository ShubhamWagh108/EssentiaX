"""
EssentiaX EDA Module - Unified Smart EDA Engine
"""

from .smart_eda import smart_eda, problem_card, smart_eda_pro, smart_viz, smart_eda_legacy

class SmartEDA:
    """
    SmartEDA class wrapper for sklearn-style usage with automatic target detection.
    
    Usage:
        # Auto-detect target column
        eda = SmartEDA()
        report = eda.analyze(df)  # Target will be auto-detected!
        
        # Or specify target manually
        eda = SmartEDA()
        report = eda.analyze(df, target='your_column')
    """
    def __init__(self, mode='console', auto_detect=True):
        self.mode = mode
        self.auto_detect = auto_detect
        self.report = None
        self.detected_target = None
    
    def analyze(self, df, target=None, output_file=None):
        """
        Perform EDA analysis with automatic target detection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target : str, optional
            Target column name. If None, will be auto-detected.
        output_file : str, optional
            Path to save HTML report (uses report_path parameter)
            
        Returns:
        --------
        dict : Analysis results and insights
        """
        self.report = smart_eda(
            df, 
            target=target,
            mode=self.mode,
            report_path=output_file if output_file else "essentiax_eda_report.html",
            auto_detect=self.auto_detect
        )
        
        # Store detected target for reference
        if target is None and self.report and 'variable_analysis' in self.report:
            self.detected_target = self.report['variable_analysis'].get('recommended_target')
        
        return self.report
    
    def fit(self, df, target=None):
        """Fit method (for sklearn compatibility)"""
        return self.analyze(df, target)

__all__ = [
    'smart_eda',
    'SmartEDA',
    'problem_card',
    'smart_eda_pro',  
    'smart_viz',
    'smart_eda_legacy'
]
