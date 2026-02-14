"""
smart_clean_enhanced.py — EssentiaX Clean Pro Enhanced
=======================================================
Advanced ML-ready Data Cleaner with multiple strategies
• Multiple outlier detection methods (IQR, Z-score, Isolation Forest, LOF)
• Advanced categorical encoding (Target, Binary, Frequency encoding)
• Multiple scaling options (Standard, MinMax, Robust, Power, Quantile)
• Advanced imputation methods (KNN, MICE/Iterative, Forward/Backward fill)
• Intelligent automatic strategy selection
• Performance optimizations and memory efficiency
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder, QuantileTransformer, PowerTransformer
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class AdvancedOutlierDetector:
    """Advanced outlier detection with multiple methods"""
    
    @staticmethod
    def detect_iqr_outliers(series, factor=1.5):
        """IQR method for outlier detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return pd.Series([False] * len(series), index=series.index)
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return (series < lower) | (series > upper)
    
    @staticmethod
    def detect_zscore_outliers(series, threshold=3):
        """Z-score method for outlier detection"""
        if series.std() == 0:
            return pd.Series([False] * len(series), index=series.index)
        z_scores = np.abs(zscore(series, nan_policy='omit'))
        return z_scores > threshold
    
    @staticmethod
    def detect_modified_zscore_outliers(series, threshold=3.5):
        """Modified Z-score using median absolute deviation"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return pd.Series([False] * len(series), index=series.index)
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    @staticmethod
    def detect_isolation_forest_outliers(df_numeric, contamination=0.1, random_state=42):
        """Isolation Forest for multivariate outlier detection"""
        if df_numeric.shape[1] == 0 or df_numeric.shape[0] < 10:
            return pd.Series([False] * len(df_numeric), index=df_numeric.index)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        outliers = iso_forest.fit_predict(df_numeric)
        return pd.Series(outliers == -1, index=df_numeric.index)
    
    @staticmethod
    def detect_lof_outliers(df_numeric, n_neighbors=20, contamination=0.1):
        """Local Outlier Factor for density-based outlier detection"""
        if df_numeric.shape[1] == 0 or df_numeric.shape[0] < n_neighbors + 1:
            return pd.Series([False] * len(df_numeric), index=df_numeric.index)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outliers = lof.fit_predict(df_numeric)
        return pd.Series(outliers == -1, index=df_numeric.index)


class AdvancedCategoricalEncoder:
    """Advanced categorical encoding with multiple strategies"""
    
    @staticmethod
    def frequency_encoding(series):
        """Frequency/Count encoding"""
        freq_map = series.value_counts().to_dict()
        return series.map(freq_map)
    
    @staticmethod
    def target_encoding(series, target, smoothing=1.0):
        """Target encoding with smoothing"""
        if target is None:
            return AdvancedCategoricalEncoder.frequency_encoding(series)
        
        # Calculate global mean
        global_mean = target.mean()
        
        # Calculate category means and counts
        agg = pd.DataFrame({'target': target, 'category': series}).groupby('category')['target'].agg(['mean', 'count'])
        
        # Apply smoothing
        smoothed_means = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        
        return series.map(smoothed_means.to_dict()).fillna(global_mean)
    
    @staticmethod
    def binary_encoding(series):
        """Binary encoding for medium cardinality"""
        # Get unique values
        unique_vals = series.unique()
        n_bits = int(np.ceil(np.log2(len(unique_vals))))
        
        # Create mapping
        mapping = {val: i for i, val in enumerate(unique_vals)}
        
        # Convert to binary
        encoded_df = pd.DataFrame(index=series.index)
        for i in range(n_bits):
            bit_col = f"{series.name}_bit_{i}"
            encoded_df[bit_col] = series.map(mapping).apply(lambda x: (x >> i) & 1)
        
        return encoded_df


class AdvancedImputer:
    """Advanced imputation methods"""
    
    @staticmethod
    def knn_imputation(df, n_neighbors=5):
        """KNN imputation for numeric data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df
    
    @staticmethod
    def iterative_imputation(df, max_iter=10, random_state=42):
        """MICE/Iterative imputation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df
    
    @staticmethod
    def forward_fill_imputation(df):
        """Forward fill imputation"""
        return df.fillna(method='ffill')
    
    @staticmethod
    def backward_fill_imputation(df):
        """Backward fill imputation"""
        return df.fillna(method='bfill')


def auto_select_strategies(df, target_column=None):
    """Automatically select optimal cleaning strategies based on data characteristics"""
    
    # Analyze data characteristics
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    missing_rate = df.isnull().sum().sum() / (n_rows * n_cols)
    
    strategies = {}
    
    # Missing value strategy
    if missing_rate > 0.3:
        strategies['missing'] = 'iterative'  # High missing rate - use advanced method
    elif missing_rate > 0.1:
        strategies['missing'] = 'knn'        # Medium missing rate - use KNN
    else:
        strategies['missing'] = 'auto'       # Low missing rate - use simple methods
    
    # Outlier detection strategy
    if len(numeric_cols) > 3 and n_rows > 100:
        strategies['outlier'] = 'isolation'  # Multivariate data - use Isolation Forest
    elif n_rows > 1000:
        strategies['outlier'] = 'zscore'     # Large dataset - use Z-score
    else:
        strategies['outlier'] = 'iqr'        # Small dataset - use IQR
    
    # Scaling strategy
    outlier_rate = 0
    for col in numeric_cols:
        if len(df[col].dropna()) > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                outlier_rate += outliers / len(df[col].dropna())
    
    outlier_rate /= max(len(numeric_cols), 1)
    
    if outlier_rate > 0.1:
        strategies['scaling'] = 'robust'     # High outlier rate - use RobustScaler
    else:
        # Check for skewness
        skewed_cols = 0
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                skewness = abs(df[col].skew())
                if skewness > 1:
                    skewed_cols += 1
        
        if skewed_cols / max(len(numeric_cols), 1) > 0.5:
            strategies['scaling'] = 'power'  # High skewness - use PowerTransformer
        else:
            strategies['scaling'] = 'standard'  # Normal distribution - use StandardScaler
    
    # Encoding strategy
    encoding_strategies = {}
    for col in categorical_cols:
        cardinality = df[col].nunique()
        if cardinality <= 5:
            encoding_strategies[col] = 'onehot'
        elif cardinality <= 20:
            encoding_strategies[col] = 'binary'
        elif target_column is not None and target_column in df.columns:
            encoding_strategies[col] = 'target'
        else:
            encoding_strategies[col] = 'frequency'
    
    strategies['encoding'] = encoding_strategies
    
    return strategies


def smart_clean_enhanced(
    df: pd.DataFrame,
    missing_strategy: str = "auto",           # auto | mean | median | mode | knn | iterative | forward | backward | drop
    outlier_strategy: str = "auto",           # auto | iqr | zscore | modified_zscore | isolation | lof | none
    outlier_action: str = "remove",           # remove | cap | transform
    scale_method: str = "auto",               # auto | standard | minmax | robust | quantile | power
    encode_method: str = "auto",              # auto | onehot | label | target | binary | frequency
    max_cardinality: int = 50,
    target_column: str = None,                # For target encoding
    inplace: bool = False,
    verbose: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Enhanced data cleaning with multiple advanced strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    missing_strategy : str
        Strategy for missing values: 'auto', 'mean', 'median', 'mode', 'knn', 'iterative', 'forward', 'backward', 'drop'
    outlier_strategy : str
        Outlier detection method: 'auto', 'iqr', 'zscore', 'modified_zscore', 'isolation', 'lof', 'none'
    outlier_action : str
        Action for outliers: 'remove', 'cap', 'transform'
    scale_method : str
        Scaling method: 'auto', 'standard', 'minmax', 'robust', 'quantile', 'power'
    encode_method : str
        Encoding method: 'auto', 'onehot', 'label', 'target', 'binary', 'frequency'
    max_cardinality : int
        Maximum unique values for one-hot encoding
    target_column : str
        Target column name for target encoding
    inplace : bool
        Modify original dataframe
    verbose : bool
        Print detailed progress information
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    pd.DataFrame : Enhanced cleaned, ML-ready dataset
    """
    
    if not inplace:
        df = df.copy()
    
    if verbose:
        print("\n🚀 **EssentiaX Clean Pro Enhanced: Starting Advanced Cleaning Pipeline...**")
        print("=" * 80)
    
    original_shape = df.shape
    
    # Auto-select strategies if needed
    if any(param == "auto" for param in [missing_strategy, outlier_strategy, scale_method, encode_method]):
        auto_strategies = auto_select_strategies(df, target_column)
        
        if missing_strategy == "auto":
            missing_strategy = auto_strategies['missing']
        if outlier_strategy == "auto":
            outlier_strategy = auto_strategies['outlier']
        if scale_method == "auto":
            scale_method = auto_strategies['scaling']
        if encode_method == "auto":
            encode_strategies = auto_strategies['encoding']
        
        if verbose:
            print(f"🧠 Auto-selected strategies:")
            print(f"   • Missing values: {missing_strategy}")
            print(f"   • Outlier detection: {outlier_strategy}")
            print(f"   • Scaling method: {scale_method}")
            if encode_method == "auto":
                print(f"   • Encoding strategies: {len(set(encode_strategies.values()))} different methods")
    
    # Column Types Detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if verbose:
        print(f"\n🔍 Dataset Analysis:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Numeric columns: {len(numeric_cols)}")
        print(f"   • Categorical columns: {len(categorical_cols)}")
        print(f"   • Missing values: {df.isnull().sum().sum():,}")
    
    # ═══════════════════════════════════════════════════════
    # 1️⃣ ADVANCED MISSING VALUE HANDLING
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n1️⃣ Advanced Missing Value Handling ({missing_strategy})...")
    
    missing_info = df.isna().sum()
    total_missing = missing_info.sum()
    
    if total_missing == 0:
        if verbose:
            print("   ✅ No missing values found.")
    else:
        if verbose:
            print(f"   ⚠️  Found {total_missing:,} missing values")
        
        if missing_strategy == "drop":
            df = df.dropna()
        elif missing_strategy == "knn":
            df = AdvancedImputer.knn_imputation(df)
        elif missing_strategy == "iterative":
            df = AdvancedImputer.iterative_imputation(df, random_state=random_state)
        elif missing_strategy == "forward":
            df = AdvancedImputer.forward_fill_imputation(df)
        elif missing_strategy == "backward":
            df = AdvancedImputer.backward_fill_imputation(df)
        else:
            # Traditional imputation (mean/median/mode)
            for col in df.columns:
                if df[col].isna().sum() == 0:
                    continue
                
                if col in numeric_cols:
                    if missing_strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:  # median
                        df[col] = df[col].fillna(df[col].median())
                else:
                    # Mode for categorical
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        
        remaining_missing = df.isna().sum().sum()
        if verbose:
            print(f"   ✅ Missing values handled: {total_missing - remaining_missing:,} filled")
    
    # ═══════════════════════════════════════════════════════
    # 2️⃣ ADVANCED OUTLIER DETECTION & HANDLING
    # ═══════════════════════════════════════════════════════
    if outlier_strategy != "none" and len(numeric_cols) > 0:
        if verbose:
            print(f"\n2️⃣ Advanced Outlier Detection ({outlier_strategy})...")
        
        before_rows = df.shape[0]
        outlier_detector = AdvancedOutlierDetector()
        
        if outlier_strategy == "isolation":
            # Multivariate outlier detection
            numeric_data = df[numeric_cols].dropna()
            if len(numeric_data) > 10:
                outlier_mask = outlier_detector.detect_isolation_forest_outliers(
                    numeric_data, random_state=random_state
                )
                outlier_indices = outlier_mask[outlier_mask].index
            else:
                outlier_indices = []
        
        elif outlier_strategy == "lof":
            # Local Outlier Factor
            numeric_data = df[numeric_cols].dropna()
            if len(numeric_data) > 20:
                outlier_mask = outlier_detector.detect_lof_outliers(numeric_data)
                outlier_indices = outlier_mask[outlier_mask].index
            else:
                outlier_indices = []
        
        else:
            # Univariate methods (IQR, Z-score, Modified Z-score)
            outlier_indices = set()
            
            for col in numeric_cols:
                if outlier_strategy == "iqr":
                    outliers = outlier_detector.detect_iqr_outliers(df[col])
                elif outlier_strategy == "zscore":
                    outliers = outlier_detector.detect_zscore_outliers(df[col])
                elif outlier_strategy == "modified_zscore":
                    outliers = outlier_detector.detect_modified_zscore_outliers(df[col])
                
                outlier_indices.update(outliers[outliers].index)
            
            outlier_indices = list(outlier_indices)
        
        # Handle outliers based on action
        if len(outlier_indices) > 0:
            if outlier_action == "remove":
                df = df.drop(outlier_indices)
            elif outlier_action == "cap":
                # Cap outliers at 5th and 95th percentiles
                for col in numeric_cols:
                    lower_cap = df[col].quantile(0.05)
                    upper_cap = df[col].quantile(0.95)
                    df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
            elif outlier_action == "transform":
                # Log transform for positive skewed data
                for col in numeric_cols:
                    if df[col].min() > 0 and df[col].skew() > 1:
                        df[col] = np.log1p(df[col])
        
        removed = before_rows - df.shape[0]
        if verbose:
            print(f"   🧮 Outliers detected: {len(outlier_indices):,}")
            if outlier_action == "remove":
                print(f"   🧮 Rows removed: {removed:,} ({100*removed/before_rows:.2f}%)")
            else:
                print(f"   🧮 Outliers {outlier_action}ped in place")
    
    # ═══════════════════════════════════════════════════════
    # 3️⃣ ADVANCED SCALING
    # ═══════════════════════════════════════════════════════
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Update after outlier removal
    
    if len(numeric_cols) > 0:
        if verbose:
            print(f"\n3️⃣ Advanced Scaling ({scale_method})...")
        
        # Select scaler
        if scale_method == "standard":
            scaler = StandardScaler()
        elif scale_method == "minmax":
            scaler = MinMaxScaler()
        elif scale_method == "robust":
            scaler = RobustScaler()
        elif scale_method == "quantile":
            scaler = QuantileTransformer(output_distribution='uniform', random_state=random_state)
        elif scale_method == "power":
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        
        # Apply scaling
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        if verbose:
            mean_check = df[numeric_cols].mean().abs().mean()
            std_check = df[numeric_cols].std().mean()
            print(f"   ✅ Scaled {len(numeric_cols)} numeric features")
            print(f"   📊 Verification: mean ≈ {mean_check:.6f}, std ≈ {std_check:.3f}")
    
    # ═══════════════════════════════════════════════════════
    # 4️⃣ ADVANCED CATEGORICAL ENCODING
    # ═══════════════════════════════════════════════════════
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()  # Update
    
    if len(categorical_cols) > 0:
        if verbose:
            print(f"\n4️⃣ Advanced Categorical Encoding...")
        
        encoder = AdvancedCategoricalEncoder()
        encoded_features = 0
        
        for col in categorical_cols:
            cardinality = df[col].nunique()
            
            # Determine encoding strategy
            if encode_method == "auto":
                strategy = encode_strategies.get(col, 'frequency')
            else:
                strategy = encode_method
            
            if verbose:
                print(f"   • {col} ({cardinality} unique) → {strategy} encoding")
            
            if strategy == "onehot" and cardinality <= max_cardinality:
                # One-hot encoding
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.int8)
                encoded = ohe.fit_transform(df[[col]])
                
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                    index=df.index
                )
                
                df = df.drop(columns=[col])
                df = pd.concat([df, encoded_df], axis=1)
                encoded_features += encoded_df.shape[1]
            
            elif strategy == "binary":
                # Binary encoding
                encoded_df = encoder.binary_encoding(df[col])
                df = df.drop(columns=[col])
                df = pd.concat([df, encoded_df], axis=1)
                encoded_features += encoded_df.shape[1]
            
            elif strategy == "target" and target_column is not None and target_column in df.columns:
                # Target encoding
                target_series = df[target_column] if target_column in df.columns else None
                df[f"{col}_target_encoded"] = encoder.target_encoding(df[col], target_series)
                df = df.drop(columns=[col])
                encoded_features += 1
            
            elif strategy == "frequency":
                # Frequency encoding
                df[f"{col}_frequency"] = encoder.frequency_encoding(df[col])
                df = df.drop(columns=[col])
                encoded_features += 1
            
            elif strategy == "label":
                # Label encoding
                le = LabelEncoder()
                df[f"{col}_label"] = le.fit_transform(df[col].astype(str))
                df = df.drop(columns=[col])
                encoded_features += 1
        
        if verbose:
            print(f"   ✅ Encoded {len(categorical_cols)} categorical columns")
            print(f"   📊 Created {encoded_features} new features")
    
    # ═══════════════════════════════════════════════════════
    # 5️⃣ FINAL OPTIMIZATION & SUMMARY
    # ═══════════════════════════════════════════════════════
    
    # Memory optimization
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    if verbose:
        print(f"\n" + "=" * 80)
        print("🎯 **ENHANCED CLEANING SUMMARY**")
        print(f"🧾 Original shape: {original_shape}")
        print(f"🧾 Final shape:    {df.shape}")
        print(f"📉 Rows removed:   {original_shape[0] - df.shape[0]:,} ({100*(original_shape[0] - df.shape[0])/original_shape[0]:.2f}%)")
        print(f"📈 Features added: {df.shape[1] - original_shape[1]}")
        print(f"📦 Memory usage:   {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Final verification
        print(f"\n🔍 Final Data Quality:")
        final_missing = df.isna().sum().sum()
        final_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        print(f"   • Missing values:     {final_missing:,} {'✅' if final_missing == 0 else '⚠️'}")
        print(f"   • Categorical columns: {final_categorical}")
        print(f"   • Numeric columns:    {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"   • ML Ready:           {'✅ YES' if final_missing == 0 and final_categorical == 0 else '⚠️ PARTIAL'}")
        
        print(f"\n✨ **Enhanced Cleaning Complete!**")
        print("=" * 80 + "\n")
    
    return df


# ═══════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🧪 Testing Enhanced Smart Clean with sample data...\n")
    
    # Create sample dataset with various problems
    np.random.seed(42)
    data = {
        'Age': [25, 30, 120, 22, 29, np.nan, 28, 150, 26, 31] * 50,
        'Salary': [50000, 60000, 1000000, 45000, 52000, np.nan, 51000, 2000000, 47000, 55000] * 50,
        'City': ['Mumbai', 'Delhi', 'Delhi', 'Pune', np.nan, 'Mumbai', 'Bangalore', 'Chennai', 'Mumbai', 'Delhi'] * 50,
        'Department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'IT', 'HR', 'IT', 'Finance', 'HR'] * 50,
        'Score': [75, 80, 95, 60, 88, 72, np.nan, 91, 85, 78] * 50,
        'Target': np.random.choice([0, 1], 500)
    }
    
    df = pd.DataFrame(data)
    
    print("🔹 Original Data:")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values: {df.isna().sum().sum()}")
    
    # Test enhanced cleaning
    cleaned_df = smart_clean_enhanced(
        df,
        missing_strategy="auto",
        outlier_strategy="auto",
        outlier_action="remove",
        scale_method="auto",
        encode_method="auto",
        target_column="Target",
        verbose=True
    )
    
    print(f"\n🔹 Enhanced Cleaned Data:")
    print(f"   Shape: {cleaned_df.shape}")
    print(f"   Missing values: {cleaned_df.isna().sum().sum()}")
    print(f"   Data types: {cleaned_df.dtypes.value_counts().to_dict()}")
    print("\n✅ Enhanced test completed!")