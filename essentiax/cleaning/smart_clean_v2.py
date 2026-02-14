"""
smart_clean_v2.py — EssentiaX Clean Pro Enhanced (Working Version)
================================================================
Advanced ML-ready Data Cleaner with multiple strategies
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

def smart_clean_v2(
    df: pd.DataFrame,
    missing_strategy: str = "auto",
    outlier_strategy: str = "auto", 
    scale_method: str = "auto",
    encode_method: str = "auto",
    target_column: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Enhanced data cleaning with multiple advanced strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    missing_strategy : str
        'auto', 'mean', 'median', 'mode', 'knn', 'drop'
    outlier_strategy : str
        'auto', 'iqr', 'zscore', 'isolation', 'none'
    scale_method : str
        'auto', 'standard', 'minmax', 'robust'
    encode_method : str
        'auto', 'onehot', 'label', 'target', 'frequency'
    target_column : str
        Target column for target encoding
    verbose : bool
        Print progress information
    
    Returns:
    --------
    pd.DataFrame : Enhanced cleaned dataset
    """
    
    df = df.copy()
    original_shape = df.shape
    
    if verbose:
        print("\n🚀 EssentiaX Clean Pro Enhanced v2.0")
        print("=" * 60)
        print(f"📊 Original Shape: {original_shape}")
    
    # ═══════════════════════════════════════════════════════
    # 1️⃣ MISSING VALUE HANDLING
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n1️⃣ Missing Value Handling ({missing_strategy})...")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        if missing_strategy == "drop":
            df = df.dropna()
        elif missing_strategy == "knn":
            # KNN Imputation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # Mode for categorical
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        else:
            # Traditional imputation
            for col in df.columns:
                if df[col].isnull().sum() == 0:
                    continue
                
                if df[col].dtype in ['int64', 'float64']:
                    if missing_strategy == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:  # median or auto
                        df[col] = df[col].fillna(df[col].median())
                else:
                    # Mode for categorical
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        
        if verbose:
            remaining = df.isnull().sum().sum()
            print(f"   ✅ Handled {missing_count - remaining:,} missing values")
    else:
        if verbose:
            print("   ✅ No missing values found")
    
    # ═══════════════════════════════════════════════════════
    # 2️⃣ OUTLIER DETECTION & REMOVAL
    # ═══════════════════════════════════════════════════════
    if outlier_strategy != "none":
        if verbose:
            print(f"\n2️⃣ Outlier Detection ({outlier_strategy})...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        before_rows = df.shape[0]
        
        if len(numeric_cols) > 0:
            if outlier_strategy == "isolation":
                # Isolation Forest for multivariate outliers
                if df.shape[0] > 10 and len(numeric_cols) > 1:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(df[numeric_cols])
                    df = df[outliers != -1]
            
            elif outlier_strategy == "zscore":
                # Z-score method
                from scipy.stats import zscore
                outlier_indices = set()
                for col in numeric_cols:
                    if df[col].std() > 0:
                        z_scores = np.abs(zscore(df[col], nan_policy='omit'))
                        outlier_mask = z_scores > 3
                        outlier_indices.update(df[outlier_mask].index)
                
                if outlier_indices:
                    df = df.drop(list(outlier_indices))
            
            else:  # IQR method (default)
                outlier_indices = set()
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outlier_mask = (df[col] < lower) | (df[col] > upper)
                        outlier_indices.update(df[outlier_mask].index)
                
                if outlier_indices:
                    df = df.drop(list(outlier_indices))
        
        removed = before_rows - df.shape[0]
        if verbose:
            print(f"   ✅ Removed {removed:,} outlier rows ({100*removed/before_rows:.2f}%)")
    
    # ═══════════════════════════════════════════════════════
    # 3️⃣ SCALING
    # ═══════════════════════════════════════════════════════
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        if verbose:
            print(f"\n3️⃣ Scaling ({scale_method})...")
        
        # Select scaler
        if scale_method == "minmax":
            scaler = MinMaxScaler()
        elif scale_method == "robust":
            scaler = RobustScaler()
        else:  # standard or auto
            scaler = StandardScaler()
        
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        if verbose:
            mean_check = df[numeric_cols].mean().abs().mean()
            print(f"   ✅ Scaled {len(numeric_cols)} features (mean: {mean_check:.4f})")
    
    # ═══════════════════════════════════════════════════════
    # 4️⃣ CATEGORICAL ENCODING
    # ═══════════════════════════════════════════════════════
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        if verbose:
            print(f"\n4️⃣ Categorical Encoding ({encode_method})...")
        
        encoded_features = 0
        
        for col in categorical_cols:
            cardinality = df[col].nunique()
            
            if encode_method == "target" and target_column and target_column in df.columns:
                # Target encoding
                target_mean = df[target_column].mean()
                encoding_map = df.groupby(col)[target_column].mean().to_dict()
                df[f"{col}_target"] = df[col].map(encoding_map).fillna(target_mean)
                df = df.drop(col, axis=1)
                encoded_features += 1
            
            elif encode_method == "frequency":
                # Frequency encoding
                freq_map = df[col].value_counts().to_dict()
                df[f"{col}_freq"] = df[col].map(freq_map)
                df = df.drop(col, axis=1)
                encoded_features += 1
            
            elif encode_method == "label":
                # Label encoding
                le = LabelEncoder()
                df[f"{col}_label"] = le.fit_transform(df[col].astype(str))
                df = df.drop(col, axis=1)
                encoded_features += 1
            
            elif cardinality <= 20:  # One-hot for low cardinality
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = ohe.fit_transform(df[[col]])
                
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                    index=df.index
                )
                
                df = df.drop(col, axis=1)
                df = pd.concat([df, encoded_df], axis=1)
                encoded_features += encoded_df.shape[1]
            
            else:
                # Skip high cardinality columns
                if verbose:
                    print(f"   ⚠️  Skipped {col} (cardinality: {cardinality})")
        
        if verbose:
            print(f"   ✅ Encoded categorical features (+{encoded_features} features)")
    
    # ═══════════════════════════════════════════════════════
    # 5️⃣ FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n" + "=" * 60)
        print("🎯 ENHANCED CLEANING SUMMARY")
        print(f"📊 Original: {original_shape} → Final: {df.shape}")
        print(f"📉 Rows removed: {original_shape[0] - df.shape[0]:,}")
        print(f"📈 Features added: {df.shape[1] - original_shape[1]}")
        
        final_missing = df.isnull().sum().sum()
        final_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        
        print(f"\n🔍 Final Quality:")
        print(f"   • Missing values: {final_missing:,} {'✅' if final_missing == 0 else '⚠️'}")
        print(f"   • Categorical cols: {final_categorical}")
        print(f"   • ML Ready: {'✅ YES' if final_missing == 0 and final_categorical == 0 else '⚠️ PARTIAL'}")
        print("=" * 60)
    
    return df


# Test function
def test_enhanced_cleaning():
    """Test the enhanced cleaning function"""
    print("🧪 Testing Enhanced Smart Clean v2.0")
    
    # Create test data
    np.random.seed(42)
    data = {
        'age': [25, 30, 120, 22, 29, np.nan, 28, 150, 26, 31] * 20,
        'salary': [50000, 60000, 1000000, 45000, 52000, np.nan, 51000, 2000000, 47000, 55000] * 20,
        'city': ['Mumbai', 'Delhi', 'Pune', 'Mumbai', np.nan, 'Bangalore', 'Chennai', 'Mumbai', 'Delhi', 'Pune'] * 20,
        'department': ['HR', 'IT', 'Finance'] * 67,  # 201 total, trim to 200
        'target': np.random.choice([0, 1], 200)
    }
    
    # Trim to exact size
    for key in data:
        data[key] = data[key][:200]
    
    df = pd.DataFrame(data)
    
    print(f"Original: {df.shape}, Missing: {df.isnull().sum().sum()}")
    
    # Test different strategies
    strategies = [
        ("Auto", {"missing_strategy": "auto", "outlier_strategy": "auto"}),
        ("KNN + Isolation", {"missing_strategy": "knn", "outlier_strategy": "isolation"}),
        ("Target Encoding", {"encode_method": "target", "target_column": "target"}),
        ("Robust Scaling", {"scale_method": "robust"}),
    ]
    
    for name, params in strategies:
        try:
            cleaned = smart_clean_v2(df.copy(), verbose=False, **params)
            print(f"✅ {name:15}: {cleaned.shape}, Missing: {cleaned.isnull().sum().sum()}")
        except Exception as e:
            print(f"❌ {name:15}: Error - {e}")
    
    print("\n🎉 Enhanced cleaning test completed!")


if __name__ == "__main__":
    test_enhanced_cleaning()