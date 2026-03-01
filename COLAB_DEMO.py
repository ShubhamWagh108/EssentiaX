"""
EssentiaX v1.0.9 - Google Colab Demo
====================================
Copy each cell below into separate Colab cells for your LinkedIn video.
Minimal code, maximum impact! 🚀

NEW in v1.0.9: Fixed SmartViz import issues! 🎯

IMPORTANT: Make sure to upgrade to v1.0.9 first:
!pip install --upgrade Essentiax
"""

# ============================================================================
# CELL 1: Installation & Setup
# ============================================================================
"""
# 📦 Install EssentiaX v1.1.1
"""
!pip install --upgrade Essentiax

# Setup for Colab (ensures plots display properly)
from essentiax.visuals import setup_colab
setup_colab()

print("✅ EssentiaX v1.1.1 installed!")

# ============================================================================
# CELL 2: Load Sample Data
# ============================================================================
"""
# 📊 Load Dataset
"""
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
df.head()

# ============================================================================
# CELL 3: Smart Data Cleaning (1 line!)
# ============================================================================
"""
# 🧹 Smart Cleaning - One Line!
"""
from essentiax.cleaning import SmartClean

df_clean = SmartClean().fit_transform(df)
print(f"✅ Cleaned: {df_clean.shape}")

# ============================================================================
# CELL 4: Smart EDA (2 lines!)
# ============================================================================
"""
# 📊 Smart EDA - Two Lines! (Auto-detects target!)
"""
from essentiax.eda import SmartEDA

eda = SmartEDA()
report = eda.analyze(df_clean)  # Target auto-detected! ✨
print(f"✅ EDA Complete! Auto-detected target: {eda.detected_target}")

# ============================================================================
# CELL 5: AI Insights (2 lines!)
# ============================================================================
"""
# 🤖 AI-Powered Insights - Two Lines!
"""
from essentiax.ai import InsightsEngine

insights = InsightsEngine().generate_insights(df_clean)
print(f"✅ Generated {len(insights.get('key_findings', []))} AI insights!")

# ============================================================================
# CELL 6: Feature Engineering (3 lines!)
# ============================================================================
"""
# 🔧 Auto Feature Engineering - Three Lines!
"""
from essentiax.feature_engineering import FeatureEngineer

# Use auto-detected target from EDA
target_col = eda.detected_target or 'target'
X, y = df_clean.drop(target_col, axis=1), df_clean[target_col]
X_new = FeatureEngineer().fit_transform(X, y)
print(f"✅ Features: {X.shape[1]} → {X_new.shape[1]}")

# ============================================================================
# CELL 7: AutoML (4 lines!)
# ============================================================================
"""
# 🤖 AutoML - Four Lines!
"""
from essentiax.automl import AutoML
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
automl = AutoML(task='classification', time_budget=30)
automl.fit(X_train, y_train)
print(f"✅ Best Model: {automl.best_model_name} | Score: {automl.best_score:.3f}")

# ============================================================================
# CELL 8: Model Explainability (2 lines!)
# ============================================================================
"""
# 🔍 Model Explainability - Two Lines!
"""
from essentiax.automl.core import ModelExplainer

explainer = ModelExplainer(automl.best_model)
explanations = explainer.explain(X_test)
print("✅ SHAP values & feature importance calculated!")

# ============================================================================
# CELL 9: Ensemble Model (3 lines!)
# ============================================================================
"""
# 🎭 Ensemble Model - Three Lines!
"""
from essentiax.automl.core import EnsembleBuilder

ensemble = EnsembleBuilder(method='stacking')
ensemble.fit(X_train, y_train)
print(f"✅ Ensemble Score: {ensemble.score(X_test, y_test):.3f}")

# ============================================================================
# CELL 10: Production Deployment (2 lines!)
# ============================================================================
"""
# 🚀 Production Ready - Two Lines!
"""
from essentiax.automl.core import ProductionModel

prod = ProductionModel(automl.best_model)
prod.save('wine_model.pkl')
print("✅ Model saved with API code & Docker config!")

# ============================================================================
# CELL 11: Visualizations
# ============================================================================
"""
# 📊 Smart Visualizations
"""
from essentiax.visuals import smart_viz

smart_viz(df_clean)

# ============================================================================
# CELL 12: Final Summary
# ============================================================================
"""
# 🎉 Summary
"""
print("""
✅ COMPLETE ML PIPELINE IN MINIMAL CODE!

Features Demonstrated:
1. Smart Data Cleaning (1 line)
2. Automated EDA (2 lines)
3. AI Insights (2 lines)
4. Feature Engineering (3 lines)
5. AutoML (4 lines)
6. Model Explainability (2 lines)
7. Ensemble Methods (3 lines)
8. Production Deployment (2 lines)

📦 pip install Essentiax
⭐ github.com/ShubhamWagh108/EssentiaX
""")
