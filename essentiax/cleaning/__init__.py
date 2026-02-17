from .smart_clean import smart_clean

class SmartClean:
    """
    SmartClean class wrapper for easy sklearn-style usage.
    
    Usage:
        cleaner = SmartClean()
        df_clean = cleaner.fit_transform(df)
    """
    def __init__(self, missing_strategy="auto", outlier_strategy="iqr", 
                 scale_numeric=True, encode_categorical=True, 
                 max_cardinality=50, verbose=True):
        self.missing_strategy = missing_strategy
        self.outlier_strategy = outlier_strategy
        self.scale_numeric = scale_numeric
        self.encode_categorical = encode_categorical
        self.max_cardinality = max_cardinality
        self.verbose = verbose
    
    def fit(self, df):
        """Fit method (for sklearn compatibility)"""
        return self
    
    def transform(self, df):
        """Transform the dataframe"""
        return smart_clean(
            df,
            missing_strategy=self.missing_strategy,
            outlier_strategy=self.outlier_strategy,
            scale_numeric=self.scale_numeric,
            encode_categorical=self.encode_categorical,
            max_cardinality=self.max_cardinality,
            verbose=self.verbose
        )
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        return self.transform(df)

__all__ = ['smart_clean', 'SmartClean']
