"""
EssentiaX v1.0.0 - Quick Demo (3-5 minutes)
===========================================
Perfect for a short LinkedIn video showcasing key features.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("\n" + "="*70)
print("  🚀 EssentiaX v1.0.0 - Complete ML Automation Platform")
print("="*70 + "\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("📂 Step 1: Loading Data")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ============================================================================
# 2. SMART CLEANING
# ============================================================================
print("🧹 Step 2: Smart Data Cleaning")
from essentiax.cleaning import SmartClean

cleaner = SmartClean()
df_clean = cleaner.fit_transform(df)
print(f"   ✓ Data cleaned and preprocessed\n")

# ============================================================================
# 3. SMART EDA
# ============================================================================
print("📊 Step 3: Exploratory Data Analysis")
from essentiax.eda import SmartEDA

eda = SmartEDA()
report = eda.analyze(df_clean, target_column='target')
print(f"   ✓ EDA complete with rich visualizations\n")

# ============================================================================
# 4. AI INSIGHTS
# ============================================================================
print("🤖 Step 4: AI-Powered Insights")
from essentiax.ai import InsightsEngine

insights = InsightsEngine()
ai_insights = insights.generate_insights(df_clean)
print(f"   ✓ {len(ai_insights.get('key_findings', []))} insights generated\n")

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================
print("🔧 Step 5: Automated Feature Engineering")
from essentiax.feature_engineering import FeatureEngineer

X = df_clean.drop('target', axis=1)
y = df_clean['target']

engineer = FeatureEngineer()
X_engineered = engineer.fit_transform(X, y)
print(f"   ✓ Features: {X.shape[1]} → {X_engineered.shape[1]}\n")

# ============================================================================
# 6. AUTOML
# ============================================================================
print("🤖 Step 6: AutoML - Automated Model Training")
from essentiax.automl import AutoML

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

automl = AutoML(task='classification', time_budget=30)
automl.fit(X_train, y_train)
score = automl.score(X_test, y_test)

print(f"   ✓ Best Model: {automl.best_model_name}")
print(f"   ✓ Accuracy: {score:.2%}\n")

# ============================================================================
# 7. MODEL EXPLAINABILITY
# ============================================================================
print("🔍 Step 7: Model Explainability (SHAP)")
from essentiax.automl.core import ModelExplainer

explainer = ModelExplainer(automl.best_model)
explanations = explainer.explain(X_test)
print(f"   ✓ Feature importance calculated\n")

# ============================================================================
# 8. PRODUCTION DEPLOYMENT
# ============================================================================
print("🚀 Step 8: Production-Ready Deployment")
from essentiax.automl.core import ProductionModel

prod = ProductionModel(automl.best_model)
prod.save('models/iris_model.pkl')
print(f"   ✓ Model saved with API code & Docker config\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("  ✅ COMPLETE ML PIPELINE IN 8 STEPS!")
print("="*70)
print("""
  📦 Install: pip install Essentiax
  ⭐ GitHub: https://github.com/ShubhamWagh108/EssentiaX
  📚 Features:
     • Smart Data Cleaning
     • Automated EDA
     • AI-Powered Insights
     • Feature Engineering
     • AutoML
     • Model Explainability
     • Production Deployment
     • Interactive Dashboards
""")
print("="*70 + "\n")
