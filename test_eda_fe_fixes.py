"""
Test Script: EDA & Feature Engineering Fixes and Enhancements
=============================================================
Verifies all bug fixes and new features work correctly.
"""

import pandas as pd
import numpy as np
import sys
import traceback

np.random.seed(42)

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")

# =================================================================
# 1. Create synthetic test DataFrame
# =================================================================
print("\n🔧 Creating synthetic test DataFrame...")
n = 500
df = pd.DataFrame({
    "age": np.random.normal(35, 10, n).clip(18, 80),
    "income": np.random.exponential(50000, n),  # skewed
    "score": np.random.uniform(0, 100, n),
    "constant_col": 1,  # constant
    "near_constant": np.random.choice([0, 1], n, p=[0.99, 0.01]),
    "category_balanced": np.random.choice(["A", "B", "C"], n),
    "category_imbalanced": np.random.choice(["X", "Y", "Z"], n, p=[0.8, 0.15, 0.05]),
    "high_card": [f"item_{i}" for i in np.random.randint(0, 200, n)],
    "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
})

# Add missing values
df.loc[50:80, "age"] = np.nan
df.loc[100:130, "income"] = np.nan
df.loc[200:220, "category_balanced"] = np.nan

# Add duplicates
df = pd.concat([df, df.iloc[:30]], ignore_index=True)

print(f"   Shape: {df.shape}")

# =================================================================
# 2. Test Smart EDA
# =================================================================
print("\n" + "="*60)
print("📊 TESTING SMART EDA")
print("="*60)

try:
    from essentiax.eda import smart_eda
    results = smart_eda(df, target="target", mode="console", show_visualizations=False, advanced_stats=False, ai_insights=False, auto_detect=False)
    
    # Verify returned dict structure
    print("\n📋 Verifying EDA Results Structure:")
    check("Returns dict", isinstance(results, dict))
    check("Has dataset_info", "dataset_info" in results)
    check("Has missing_analysis", "missing_analysis" in results)
    check("Has numeric_analysis", "numeric_analysis" in results)
    check("Has categorical_analysis", "categorical_analysis" in results)
    check("Has data_quality_score", "data_quality_score" in results)
    check("Has quality_breakdown", "quality_breakdown" in results)
    
    # Verify new dataset_info fields
    di = results["dataset_info"]
    check("Has dtype_counts", "dtype_counts" in di)
    check("dtype_counts is dict", isinstance(di["dtype_counts"], dict))
    check("Has duplicate_pct", "duplicate_pct" in di)
    check("Has constant_columns", "constant_columns" in di)
    check("constant_col detected", "constant_col" in di["constant_columns"])
    check("Has near_constant_columns", "near_constant_columns" in di)
    check("Has high_zero_columns", "high_zero_columns" in di)
    
    # Verify numeric analysis has new stats
    if results["numeric_analysis"] and "descriptive_stats" in results["numeric_analysis"]:
        desc = results["numeric_analysis"]["descriptive_stats"]
        check("Has IQR in stats", "IQR" in desc)
        check("Has range in stats", "range" in desc)
        check("Has CV in stats", "CV" in desc)
    
    # Verify categorical analysis has entropy
    if results["categorical_analysis"] and "entropy" in results["categorical_analysis"]:
        check("Has entropy analysis", len(results["categorical_analysis"]["entropy"]) > 0)
    
    # Verify quality breakdown
    check("Quality breakdown is dict", isinstance(results.get("quality_breakdown"), dict))
    check("Quality score is int", isinstance(results["data_quality_score"], int))
    check("Quality score in range", 1 <= results["data_quality_score"] <= 100)
    
    # Verify target analysis
    ta = results.get("target_analysis", {})
    check("Target analysis present", bool(ta))
    check("Problem type inferred", results.get("problem_type") is not None)
    
    print(f"\n   Data Quality Score: {results['data_quality_score']}/100")
    print(f"   Problem Type: {results['problem_type']}")
    
except Exception as e:
    FAIL += 1
    print(f"\n❌ EDA Test FAILED with error:")
    traceback.print_exc()

# =================================================================
# 3. Test Feature Engineering
# =================================================================
print("\n" + "="*60)
print("🔧 TESTING FEATURE ENGINEERING")
print("="*60)

try:
    from essentiax.feature_engineering import FeatureEngineer
    
    # Create FE test data (separate X and y)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    fe = FeatureEngineer(
        strategy='auto',
        feature_selection=False,  # Disable for simpler test
        generate_interactions=False,
        generate_polynomials=False,
        handle_missing='auto',
        scale_features=False,  # Disable to see raw features
        encode_categoricals=True,
        verbose=True
    )
    
    X_transformed = fe.fit_transform(X, y)
    
    print("\n📋 Verifying Feature Engineering Results:")
    check("Returns DataFrame", isinstance(X_transformed, pd.DataFrame))
    check("No NaN in output", X_transformed.isnull().sum().sum() == 0)
    check("Output has rows", len(X_transformed) == len(X))
    
    # Check missing indicators were created
    missing_indicator_cols = [c for c in X_transformed.columns if c.endswith("_is_missing")]
    check("Missing indicators created", len(missing_indicator_cols) > 0)
    print(f"   Missing indicator columns: {missing_indicator_cols}")
    
    # Check transformation summary
    summary = fe.get_transformation_summary()
    check("Summary has missing_indicators_added", "missing_indicators_added" in summary)
    check("Summary has datetime_features_engineered", "datetime_features_engineered" in summary)
    check("Missing indicators count > 0", summary.get("missing_indicators_added", 0) > 0)
    
    print(f"\n   Original features: {X.shape[1]}")
    print(f"   Transformed features: {X_transformed.shape[1]}")
    print(f"   Missing indicators: {summary.get('missing_indicators_added', 0)}")
    print(f"   Datetime features: {summary.get('datetime_features_engineered', 0)}")
    
except Exception as e:
    FAIL += 1
    print(f"\n❌ Feature Engineering Test FAILED with error:")
    traceback.print_exc()

# =================================================================
# 4. Test smart_features convenience function
# =================================================================
print("\n" + "="*60)
print("🚀 TESTING SMART FEATURES")
print("="*60)

try:
    from essentiax.feature_engineering import smart_features
    
    # Quick test with fast mode
    X_small = df[["age", "income", "score"]].copy()
    y_small = df["target"].copy()
    
    X_smart = smart_features(X_small, y_small, mode='fast', verbose=False)
    
    check("smart_features returns DataFrame", isinstance(X_smart, pd.DataFrame))
    check("smart_features no NaN", X_smart.isnull().sum().sum() == 0)
    check("smart_features has rows", len(X_smart) == len(X_small))
    
    print(f"   Input: {X_small.shape} → Output: {X_smart.shape}")
    
except Exception as e:
    FAIL += 1
    print(f"\n❌ Smart Features Test FAILED with error:")
    traceback.print_exc()

# =================================================================
# 5. Test scipy import fix in CategoricalTransformer
# =================================================================
print("\n" + "="*60)
print("🔬 TESTING CATEGORICAL TRANSFORMER (scipy fix)")
print("="*60)

try:
    from essentiax.feature_engineering.transformers.categorical import CategoricalTransformer
    
    cat_data = pd.DataFrame({
        "color": np.random.choice(["red", "blue", "green"], 100),
        "size": np.random.choice(["S", "M", "L"], 100),
    })
    cat_target = pd.Series(np.random.choice([0, 1], 100))
    
    ct = CategoricalTransformer(strategy='auto', verbose=False)
    ct.fit(cat_data, cat_target)
    result = ct.transform(cat_data)
    
    check("CategoricalTransformer works", isinstance(result, pd.DataFrame))
    check("CategoricalTransformer no crash", True)
    
except Exception as e:
    FAIL += 1
    print(f"\n❌ Categorical Transformer Test FAILED:")
    traceback.print_exc()

# =================================================================
# FINAL SUMMARY
# =================================================================
print("\n" + "="*60)
total = PASS + FAIL
print(f"📊 TEST RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("✅ ALL TESTS PASSED!")
else:
    print(f"⚠️ {FAIL} test(s) failed")
print("="*60)
