import pandas as pd
import numpy as np
import sys
import os
import codecs

# Reconfigure stdout for emoji/charmap issues
sys.stdout = codecs.open("automl_test_output.txt", "w", encoding="utf-8")

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from essentiax.automl import AutoML
    
    # Create dataset with mixed types
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'numeric_feature': np.random.normal(0, 1, n),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n),
        'datetime_feature': pd.date_range(start='2020-01-01', periods=n, freq='D'),
        'missing_feature': np.random.normal(10, 2, n),
        'text_feature': np.random.choice(['hello world', 'foo bar', 'test string', 'another text'], n),
        'target': np.random.choice([0, 1], n)
    })
    
    # Introduce missing values
    df.loc[10:30, 'missing_feature'] = np.nan
    df.loc[50:60, 'categorical_feature'] = np.nan
    
    print("=== Testing AutoML Classification ===")
    X = df.drop('target', axis=1)
    y = df['target']
    
    automl = AutoML(task='classification', time_budget=30)
    automl.fit(X, y)
    
    print("\n=== AutoML Testing Completed Successfully ===")
except Exception as e:
    import traceback
    print(f"Exception Encountered:\n{traceback.format_exc()}")
