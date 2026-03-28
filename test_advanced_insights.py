import pandas as pd
import numpy as np
import sys
import codecs
import os
sys.stdout = codecs.open("test_output_utf8.txt", "w", encoding="utf-8")

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from essentiax.ai.advanced_insights import AdvancedInsightsEngine, assess_data_quality, generate_recommendations

# Create a challenging dataset
df = pd.DataFrame({
    'age': [25, 30, -5, 150, 200, 25, 30, 25, pd.NA, 45], # Negative age, extreme age, missing
    'price': [-100, 50, 60, 70, 80, 50, 60, pd.NA, 90, 100], # Negative price, missing
    'category': ['A', 'a', 'B', 'b', 'C', 'c', 123, 'A', 'B', 'A'], # Case inconsistencies, mixed types
    'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Add complete duplicate row
df.loc[10] = df.loc[0]

print("=== Testing AdvancedInsightsEngine ===")
quality_assessment = assess_data_quality(df, target_col='target')
print(f"Data Quality Score: {quality_assessment['overall_score']:.1f}/100")
print(f"Dimension Scores: {quality_assessment['dimension_scores']}")
print(f"Quality Issues: {quality_assessment['quality_issues']}")
print(f"Improvement Recommendations: {quality_assessment['improvement_recommendations']}")
print(f"Model Readiness: {quality_assessment['model_readiness']}")

mock_analysis = {
    'problem_type': 'classification',
    'data_quality_score': quality_assessment['overall_score'],
    'missing_analysis': {'missing_percentage': df.isna().sum().sum() / df.size * 100, 'total_missing': df.isna().sum().sum()},
    'categorical_analysis': {'categorical_cols': 1},
    'numeric_analysis': {'outlier_analysis': True}
}
recommendations = generate_recommendations(df, mock_analysis, target_col='target')
print("\nRecommendations Priority:", recommendations['priority_level'])
print("Preprocessing:", recommendations['preprocessing_steps'])
print("Next Steps:", recommendations['next_steps'])

engine = AdvancedInsightsEngine()
test_results = {
    'normality_results': {
        'age': {'is_normal': False, 'skewness': 2.5},
        'price': {'is_normal': True}
    },
    'correlation_results': {
        'significant_pairs': [
            {'variable1': 'age', 'variable2': 'price', 'pearson_r': 0.85, 'strength': 'strong', 'direction': 'positive'}
        ]
    },
    'outlier_results': {
        'age': {'outlier_percentage': 16.0}
    }
}
context = {'sample_size': len(df), 'missing_percentage': 5.0}

insights = engine.interpret_statistical_results(test_results, context)

print("\nInsights Generation:")
print("Normality:", insights['normality_insights'])
print("Correlation:", insights['correlation_insights'])
print("Outliers:", insights['outlier_insights'])
print("Overall:", insights['overall_interpretation'])
print("Confidence:", insights['confidence_score'])
