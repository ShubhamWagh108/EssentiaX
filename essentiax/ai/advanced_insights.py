"""
Advanced AI Insights Engine for EssentiaX
=========================================
Template-based statistical interpretation and actionable recommendations

Features:
- Statistical interpretation templates
- Data quality assessment
- Actionable recommendations engine
- Business impact insights
- Model readiness assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedInsightsEngine:
    """AI-powered insights and recommendations engine"""
    
    def __init__(self):
        self.interpretation_templates = self._load_interpretation_templates()
        self.recommendation_templates = self._load_recommendation_templates()
    
    def interpret_statistical_results(self, test_results: Dict, context: Dict) -> Dict[str, Any]:
        """
        Convert statistical results to plain English interpretations
        
        Parameters:
        -----------
        test_results : dict
            Results from statistical tests
        context : dict
            Dataset context information
            
        Returns:
        --------
        dict : Human-readable insights and recommendations
        """
        insights = {
            'normality_insights': [],
            'correlation_insights': [],
            'statistical_test_insights': [],
            'outlier_insights': [],
            'overall_interpretation': '',
            'confidence_score': 0.0
        }
        
        # Interpret normality test results
        if 'normality_results' in test_results:
            insights['normality_insights'] = self._interpret_normality_tests(
                test_results['normality_results'], context
            )
        
        # Interpret correlation results
        if 'correlation_results' in test_results:
            insights['correlation_insights'] = self._interpret_correlation_analysis(
                test_results['correlation_results'], context
            )
        
        # Interpret statistical tests
        if 'statistical_tests' in test_results:
            insights['statistical_test_insights'] = self._interpret_statistical_tests(
                test_results['statistical_tests'], context
            )
        
        # Interpret outlier detection
        if 'outlier_results' in test_results:
            insights['outlier_insights'] = self._interpret_outlier_detection(
                test_results['outlier_results'], context
            )
        
        # Generate overall interpretation
        insights['overall_interpretation'] = self._generate_overall_interpretation(insights, context)
        insights['confidence_score'] = self._calculate_confidence_score(test_results, context)
        
        return insights
    
    def assess_data_quality(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_col : str, optional
            Target column for supervised learning assessment
            
        Returns:
        --------
        dict : Quality score and improvement recommendations
        """
        assessment = {
            'overall_score': 0,
            'dimension_scores': {},
            'quality_issues': [],
            'improvement_recommendations': [],
            'model_readiness': 'not_ready'
        }
        
        # Assess different quality dimensions
        completeness_score = self._assess_completeness(df)
        consistency_score = self._assess_consistency(df)
        validity_score = self._assess_validity(df)
        uniqueness_score = self._assess_uniqueness(df)
        accuracy_score = self._assess_accuracy(df, target_col)
        
        assessment['dimension_scores'] = {
            'completeness': completeness_score,
            'consistency': consistency_score,
            'validity': validity_score,
            'uniqueness': uniqueness_score,
            'accuracy': accuracy_score
        }
        
        # Calculate overall score (weighted average)
        weights = {'completeness': 0.25, 'consistency': 0.2, 'validity': 0.2, 
                  'uniqueness': 0.15, 'accuracy': 0.2}
        assessment['overall_score'] = sum(
            score * weights[dim] for dim, score in assessment['dimension_scores'].items()
        )
        
        # Identify quality issues and recommendations
        assessment['quality_issues'] = self._identify_quality_issues(df, assessment['dimension_scores'])
        assessment['improvement_recommendations'] = self._generate_quality_recommendations(
            assessment['quality_issues'], df, target_col
        )
        
        # Assess model readiness
        assessment['model_readiness'] = self._assess_model_readiness(assessment['overall_score'], df, target_col)
        
        return assessment
    
    def generate_recommendations(self, df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> Dict[str, Any]:
        """
        Generate actionable data science recommendations
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        analysis_results : dict
            Results from EDA analysis
        target_col : str, optional
            Target column
            
        Returns:
        --------
        dict : Preprocessing, modeling, and business recommendations
        """
        recommendations = {
            'preprocessing_steps': [],
            'feature_engineering': [],
            'model_selection': [],
            'validation_strategy': [],
            'business_insights': [],
            'next_steps': [],
            'priority_level': 'medium'
        }
        
        # Generate preprocessing recommendations
        recommendations['preprocessing_steps'] = self._recommend_preprocessing(df, analysis_results, target_col)
        
        # Generate feature engineering recommendations
        recommendations['feature_engineering'] = self._recommend_feature_engineering(df, analysis_results, target_col)
        
        # Generate model selection recommendations
        recommendations['model_selection'] = self._recommend_models(df, analysis_results, target_col)
        
        # Generate validation strategy recommendations
        recommendations['validation_strategy'] = self._recommend_validation_strategy(df, analysis_results, target_col)
        
        # Generate business insights
        recommendations['business_insights'] = self._generate_business_insights(df, analysis_results, target_col)
        
        # Generate next steps roadmap
        recommendations['next_steps'] = self._generate_next_steps(recommendations, df, target_col)
        
        # Determine priority level
        recommendations['priority_level'] = self._determine_priority_level(analysis_results, df)
        
        return recommendations
    
    def _load_interpretation_templates(self) -> Dict[str, Dict]:
        """Load statistical interpretation templates"""
        return {
            'normality': {
                'normal': "The data follows a normal distribution, making it suitable for parametric statistical tests and linear models.",
                'not_normal': "The data does not follow a normal distribution. Consider non-parametric tests or data transformation.",
                'skewed_right': "The data is right-skewed with a long tail extending to higher values. Consider log transformation.",
                'skewed_left': "The data is left-skewed with a long tail extending to lower values. Consider square or cube transformation.",
                'multimodal': "The data shows multiple peaks, suggesting distinct subgroups or mixed populations."
            },
            'correlation': {
                'strong_positive': "Strong positive correlation indicates that as one variable increases, the other tends to increase proportionally.",
                'strong_negative': "Strong negative correlation indicates that as one variable increases, the other tends to decrease proportionally.",
                'moderate': "Moderate correlation suggests a meaningful but not overwhelming relationship between variables.",
                'weak': "Weak correlation indicates little to no linear relationship between variables.",
                'multicollinearity': "High correlations between predictors may cause multicollinearity issues in linear models."
            },
            'outliers': {
                'few_outliers': "Few outliers detected. These may represent natural variation or measurement errors.",
                'many_outliers': "Many outliers detected. Investigate data collection process and consider outlier treatment.",
                'extreme_outliers': "Extreme outliers detected. These may significantly impact model performance.",
                'no_outliers': "No significant outliers detected. Data appears clean and well-behaved."
            }
        }
    
    def _load_recommendation_templates(self) -> Dict[str, List]:
        """Load recommendation templates"""
        return {
            'missing_data': [
                "Use median imputation for numeric variables with < 30% missing values",
                "Use mode imputation for categorical variables with < 20% missing values",
                "Consider dropping columns with > 50% missing values",
                "Use advanced imputation (KNN, iterative) for important features with moderate missingness"
            ],
            'outliers': [
                "Use IQR method to cap outliers at 1.5*IQR boundaries",
                "Apply log transformation to reduce impact of extreme values",
                "Use robust scalers (RobustScaler) instead of StandardScaler",
                "Consider outlier detection algorithms (Isolation Forest) for complex patterns"
            ],
            'skewed_data': [
                "Apply log transformation for right-skewed data",
                "Apply square root transformation for moderate skewness",
                "Use Box-Cox transformation for optimal normalization",
                "Consider quantile transformation for non-parametric normalization"
            ],
            'categorical_data': [
                "Use one-hot encoding for low cardinality categorical variables (< 10 categories)",
                "Use target encoding for high cardinality categorical variables",
                "Consider frequency encoding for ordinal relationships",
                "Use embedding layers for very high cardinality categories in neural networks"
            ]
        }
    
    def _interpret_normality_tests(self, normality_results: Dict, context: Dict) -> List[str]:
        """Interpret normality test results"""
        insights = []
        
        for column, result in normality_results.items():
            if result.get('is_normal', False):
                insights.append(f"✅ {column}: {self.interpretation_templates['normality']['normal']}")
                insights.append(f"   → Suitable for parametric tests and linear regression assumptions")
            else:
                insights.append(f"❌ {column}: {self.interpretation_templates['normality']['not_normal']}")
                
                # Check for skewness
                if 'skewness' in result:
                    skew = result['skewness']
                    if skew > 1:
                        insights.append(f"   → {self.interpretation_templates['normality']['skewed_right']}")
                    elif skew < -1:
                        insights.append(f"   → {self.interpretation_templates['normality']['skewed_left']}")
                
                # Check for multimodality
                if result.get('modes', 1) > 1:
                    insights.append(f"   → {self.interpretation_templates['normality']['multimodal']}")
        
        return insights
    
    def _interpret_correlation_analysis(self, correlation_results: Dict, context: Dict) -> List[str]:
        """Interpret correlation analysis results"""
        insights = []
        
        if 'significant_pairs' in correlation_results:
            significant_pairs = correlation_results['significant_pairs']
            
            if not significant_pairs:
                insights.append("ℹ️ No strong correlations found between variables")
                insights.append("   → Variables appear to be independent, reducing multicollinearity concerns")
            else:
                insights.append(f"🔍 Found {len(significant_pairs)} significant correlations:")
                
                for pair in significant_pairs[:5]:  # Top 5 correlations
                    var1, var2 = pair['variable1'], pair['variable2']
                    corr = pair['pearson_r']
                    strength = pair['strength']
                    direction = pair['direction']
                    
                    if abs(corr) >= 0.8:
                        template = self.interpretation_templates['correlation']['strong_positive' if corr > 0 else 'strong_negative']
                        insights.append(f"   → {var1} ↔ {var2}: {template} (r={corr:.3f})")
                        if context.get('target_col') not in [var1, var2]:
                            insights.append(f"     ⚠️ Consider removing one variable to avoid multicollinearity")
                    elif abs(corr) >= 0.5:
                        insights.append(f"   → {var1} ↔ {var2}: {self.interpretation_templates['correlation']['moderate']} (r={corr:.3f})")
        
        return insights
    
    def _interpret_statistical_tests(self, statistical_tests: Dict, context: Dict) -> List[str]:
        """Interpret statistical test results"""
        insights = []
        
        if 'significant_results' in statistical_tests:
            significant_results = statistical_tests['significant_results']
            
            if not significant_results:
                insights.append("ℹ️ No statistically significant relationships detected")
                insights.append("   → Variables may be independent or relationships may be non-linear")
            else:
                insights.append(f"📊 Found {len(significant_results)} statistically significant results:")
                
                for result in significant_results[:5]:  # Top 5 results
                    test_type = result['test_type']
                    interpretation = result['interpretation']
                    p_value = result['p_value']
                    
                    insights.append(f"   → {interpretation} (p={p_value:.4f})")
                    
                    # Add context-specific advice
                    if test_type == 'two_sample_ttest' and p_value < 0.001:
                        insights.append(f"     💡 Very strong evidence of difference - important for feature selection")
                    elif test_type == 'chi_square_independence' and p_value < 0.05:
                        insights.append(f"     💡 Variables are associated - consider interaction terms")
        
        return insights
    
    def _interpret_outlier_detection(self, outlier_results: Dict, context: Dict) -> List[str]:
        """Interpret outlier detection results"""
        insights = []
        
        for column, result in outlier_results.items():
            outlier_pct = result.get('outlier_percentage', 0)
            
            if outlier_pct == 0:
                insights.append(f"✅ {column}: {self.interpretation_templates['outliers']['no_outliers']}")
            elif outlier_pct < 5:
                insights.append(f"⚠️ {column}: {self.interpretation_templates['outliers']['few_outliers']} ({outlier_pct:.1f}%)")
                insights.append(f"   → Consider capping at 95th/5th percentiles or using robust scaling")
            elif outlier_pct < 15:
                insights.append(f"🚨 {column}: {self.interpretation_templates['outliers']['many_outliers']} ({outlier_pct:.1f}%)")
                insights.append(f"   → Investigate data collection process and apply outlier treatment")
            else:
                insights.append(f"🔥 {column}: {self.interpretation_templates['outliers']['extreme_outliers']} ({outlier_pct:.1f}%)")
                insights.append(f"   → Consider data transformation or robust modeling approaches")
        
        return insights
    
    def _generate_overall_interpretation(self, insights: Dict, context: Dict) -> str:
        """Generate overall interpretation summary"""
        total_insights = sum(len(insight_list) for insight_list in insights.values() if isinstance(insight_list, list))
        
        if total_insights == 0:
            return "Dataset appears to be well-behaved with no major statistical concerns detected."
        
        interpretation = f"Statistical analysis revealed {total_insights} key insights across multiple dimensions. "
        
        # Prioritize insights by importance
        if insights['outlier_insights']:
            interpretation += "Outlier patterns require attention for robust modeling. "
        
        if insights['normality_insights']:
            interpretation += "Distribution characteristics will influence choice of statistical methods. "
        
        if insights['correlation_insights']:
            interpretation += "Variable relationships detected may impact feature selection and model interpretation. "
        
        interpretation += "Review detailed insights for specific recommendations."
        
        return interpretation
    
    def _calculate_confidence_score(self, test_results: Dict, context: Dict) -> float:
        """Calculate confidence score for interpretations"""
        score = 0.8  # Base confidence
        
        # Adjust based on sample size
        sample_size = context.get('sample_size', 0)
        if sample_size < 30:
            score -= 0.3
        elif sample_size < 100:
            score -= 0.1
        elif sample_size > 1000:
            score += 0.1
        
        # Adjust based on data quality
        missing_pct = context.get('missing_percentage', 0)
        if missing_pct > 20:
            score -= 0.2
        elif missing_pct > 10:
            score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def _assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        return completeness * 100
    
    def _assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency"""
        score = 100.0
        
        # Check for mixed data types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric
                pd.to_numeric(df[col], errors='raise')
                score -= 5  # Mixed numeric/text in object column
            except:
                pass
        
        # Check for inconsistent categorical values
        for col in df.select_dtypes(include=['object']).columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                # Check for case inconsistencies
                lower_vals = [str(val).lower() for val in unique_vals]
                if len(set(lower_vals)) < len(unique_vals):
                    score -= 10
        
        return max(0, score)
    
    def _assess_validity(self, df: pd.DataFrame) -> float:
        """Assess data validity"""
        score = 100.0
        
        # Check for negative values in columns that should be positive
        potential_positive_cols = [col for col in df.columns if any(keyword in col.lower() 
                                  for keyword in ['age', 'price', 'count', 'amount', 'quantity'])]
        
        for col in potential_positive_cols:
            if col in df.select_dtypes(include=[np.number]).columns:
                if (df[col] < 0).any():
                    score -= 15
        
        # Check for unrealistic values
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'age' in col.lower():
                if (df[col] > 150).any() or (df[col] < 0).any():
                    score -= 10
        
        return max(0, score)
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> float:
        """Assess data uniqueness"""
        duplicate_rows = df.duplicated().sum()
        uniqueness = (len(df) - duplicate_rows) / len(df)
        return uniqueness * 100
    
    def _assess_accuracy(self, df: pd.DataFrame, target_col: str = None) -> float:
        """Assess data accuracy (proxy measures)"""
        score = 100.0
        
        # Check for extreme outliers that might indicate errors
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                extreme_outliers = df[(df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)]
                if len(extreme_outliers) > len(df) * 0.05:  # More than 5% extreme outliers
                    score -= 20
        
        return max(0, score)
    
    def _identify_quality_issues(self, df: pd.DataFrame, dimension_scores: Dict) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        if dimension_scores['completeness'] < 90:
            missing_pct = 100 - dimension_scores['completeness']
            issues.append(f"High missing data rate: {missing_pct:.1f}% of cells are missing")
        
        if dimension_scores['consistency'] < 80:
            issues.append("Data consistency issues detected (mixed types, case inconsistencies)")
        
        if dimension_scores['validity'] < 80:
            issues.append("Data validity concerns (negative values in positive fields, unrealistic values)")
        
        if dimension_scores['uniqueness'] < 95:
            duplicate_pct = 100 - dimension_scores['uniqueness']
            issues.append(f"Duplicate records detected: {duplicate_pct:.1f}% of rows are duplicates")
        
        if dimension_scores['accuracy'] < 80:
            issues.append("Potential accuracy issues (extreme outliers suggesting data errors)")
        
        return issues
    
    def _generate_quality_recommendations(self, quality_issues: List[str], df: pd.DataFrame, target_col: str = None) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        for issue in quality_issues:
            if "missing data" in issue.lower():
                recommendations.extend(self.recommendation_templates['missing_data'])
            elif "consistency" in issue.lower():
                recommendations.append("Standardize categorical values (case, spelling, format)")
                recommendations.append("Convert mixed-type columns to appropriate data types")
            elif "validity" in issue.lower():
                recommendations.append("Investigate and correct unrealistic values")
                recommendations.append("Apply domain-specific validation rules")
            elif "duplicate" in issue.lower():
                recommendations.append("Remove duplicate records after careful investigation")
                recommendations.append("Implement data deduplication procedures")
            elif "accuracy" in issue.lower():
                recommendations.extend(self.recommendation_templates['outliers'])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_model_readiness(self, overall_score: float, df: pd.DataFrame, target_col: str = None) -> str:
        """Assess readiness for machine learning modeling"""
        if overall_score >= 85:
            return "ready"
        elif overall_score >= 70:
            return "needs_minor_cleanup"
        elif overall_score >= 50:
            return "needs_major_cleanup"
        else:
            return "not_ready"
    
    def _recommend_preprocessing(self, df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> List[str]:
        """Recommend preprocessing steps"""
        steps = []
        
        # Missing value handling
        missing_analysis = analysis_results.get('missing_analysis', {})
        if missing_analysis.get('total_missing', 0) > 0:
            steps.append("Handle missing values using appropriate imputation strategies")
            steps.append("Consider creating missing value indicator features for important variables")
        
        # Outlier handling
        numeric_analysis = analysis_results.get('numeric_analysis', {})
        if 'outlier_analysis' in numeric_analysis:
            steps.append("Apply outlier detection and treatment methods")
            steps.append("Use robust scaling methods for features with outliers")
        
        # Categorical encoding
        categorical_analysis = analysis_results.get('categorical_analysis', {})
        if categorical_analysis.get('categorical_cols', 0) > 0:
            steps.append("Encode categorical variables using appropriate methods")
            steps.append("Handle high cardinality categorical variables with target encoding")
        
        # Feature scaling
        if numeric_analysis:
            steps.append("Scale numerical features for distance-based algorithms")
            steps.append("Consider normalization for neural networks")
        
        return steps
    
    def _recommend_feature_engineering(self, df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> List[str]:
        """Recommend feature engineering steps"""
        recommendations = []
        
        # Correlation-based recommendations
        correlation_analysis = analysis_results.get('correlation_analysis', {})
        strong_correlations = correlation_analysis.get('strong_correlations', [])
        
        if strong_correlations:
            recommendations.append("Create interaction terms for strongly correlated variables")
            recommendations.append("Consider polynomial features for non-linear relationships")
        
        # Distribution-based recommendations
        numeric_analysis = analysis_results.get('numeric_analysis', {})
        if 'distribution_insights' in numeric_analysis:
            for col, insights in numeric_analysis['distribution_insights'].items():
                if abs(insights.get('skewness', 0)) > 1:
                    recommendations.append(f"Apply transformation to {col} to reduce skewness")
        
        # Categorical recommendations
        categorical_analysis = analysis_results.get('categorical_analysis', {})
        high_card_cols = categorical_analysis.get('high_cardinality_cols', [])
        if high_card_cols:
            recommendations.append("Create frequency-based features for high cardinality categories")
            recommendations.append("Consider grouping rare categories into 'Other' category")
        
        # Time-based recommendations (if datetime columns exist)
        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            recommendations.append("Extract time-based features (day, month, year, weekday)")
            recommendations.append("Create lag features for time series data")
        
        return recommendations
    
    def _recommend_models(self, df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> List[str]:
        """Recommend appropriate models"""
        recommendations = []
        
        problem_type = analysis_results.get('problem_type')
        n_samples, n_features = df.shape
        
        if problem_type == 'classification':
            if n_samples < 1000:
                recommendations.append("Logistic Regression (good for small datasets)")
                recommendations.append("Random Forest (handles overfitting well)")
            else:
                recommendations.append("Gradient Boosting (XGBoost, LightGBM)")
                recommendations.append("Support Vector Machine for complex boundaries")
            
            # Check for imbalance
            target_analysis = analysis_results.get('target_analysis', {})
            if target_analysis.get('imbalance_detected', False):
                recommendations.append("Use class balancing techniques (SMOTE, class weights)")
        
        elif problem_type == 'regression':
            if n_features > n_samples:
                recommendations.append("Regularized regression (Ridge, Lasso, Elastic Net)")
            else:
                recommendations.append("Random Forest Regressor")
                recommendations.append("Gradient Boosting Regressor")
        
        elif problem_type == 'nlp':
            recommendations.append("TF-IDF + Logistic Regression for baseline")
            recommendations.append("Pre-trained transformers (BERT) for advanced performance")
        
        # General recommendations based on data characteristics
        if n_features > 100:
            recommendations.append("Consider feature selection techniques")
            recommendations.append("Use dimensionality reduction (PCA) if appropriate")
        
        return recommendations
    
    def _recommend_validation_strategy(self, df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> List[str]:
        """Recommend validation strategy"""
        recommendations = []
        
        n_samples = len(df)
        problem_type = analysis_results.get('problem_type')
        
        if n_samples < 1000:
            recommendations.append("Use k-fold cross-validation (k=5 or k=10)")
            recommendations.append("Consider leave-one-out CV for very small datasets")
        else:
            recommendations.append("Use train/validation/test split (60/20/20)")
            recommendations.append("Apply stratified sampling for classification problems")
        
        # Time-based recommendations
        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            recommendations.append("Use time-based splitting to avoid data leakage")
            recommendations.append("Consider walk-forward validation for time series")
        
        # Imbalanced data recommendations
        target_analysis = analysis_results.get('target_analysis', {})
        if target_analysis.get('imbalance_detected', False):
            recommendations.append("Use stratified sampling to maintain class distribution")
            recommendations.append("Monitor precision, recall, and F1-score in addition to accuracy")
        
        return recommendations
    
    def _generate_business_insights(self, df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> List[str]:
        """Generate business-relevant insights"""
        insights = []
        
        # Data volume insights
        n_samples, n_features = df.shape
        if n_samples > 100000:
            insights.append("Large dataset enables complex model training and robust validation")
        elif n_samples < 1000:
            insights.append("Small dataset may limit model complexity and require careful validation")
        
        # Feature richness insights
        if n_features > 50:
            insights.append("Rich feature set provides good modeling potential")
            insights.append("Consider feature importance analysis to identify key drivers")
        
        # Data quality insights
        data_quality_score = analysis_results.get('data_quality_score', 0)
        if data_quality_score >= 90:
            insights.append("High data quality enables reliable model development")
        elif data_quality_score < 70:
            insights.append("Data quality issues may impact model reliability - invest in data cleaning")
        
        # Missing data business impact
        missing_analysis = analysis_results.get('missing_analysis', {})
        missing_pct = missing_analysis.get('missing_percentage', 0)
        if missing_pct > 20:
            insights.append("High missing data rate may indicate data collection process issues")
        
        # Correlation insights for business
        correlation_analysis = analysis_results.get('correlation_analysis', {})
        strong_correlations = correlation_analysis.get('strong_correlations', [])
        if len(strong_correlations) > 5:
            insights.append("Multiple strong correlations suggest underlying business relationships")
            insights.append("Consider domain expertise to validate statistical relationships")
        
        return insights
    
    def _generate_next_steps(self, recommendations: Dict, df: pd.DataFrame, target_col: str = None) -> List[str]:
        """Generate prioritized next steps roadmap"""
        next_steps = []
        
        # Prioritize based on data issues
        if recommendations['preprocessing_steps']:
            next_steps.append("1. Execute data preprocessing pipeline")
            next_steps.append("   - Handle missing values and outliers")
            next_steps.append("   - Encode categorical variables")
        
        if recommendations['feature_engineering']:
            next_steps.append("2. Implement feature engineering")
            next_steps.append("   - Create derived features")
            next_steps.append("   - Apply transformations")
        
        if recommendations['model_selection']:
            next_steps.append("3. Model development and selection")
            next_steps.append("   - Train baseline models")
            next_steps.append("   - Compare model performance")
        
        if recommendations['validation_strategy']:
            next_steps.append("4. Implement robust validation")
            next_steps.append("   - Set up cross-validation")
            next_steps.append("   - Monitor key metrics")
        
        next_steps.append("5. Model interpretation and deployment")
        next_steps.append("   - Analyze feature importance")
        next_steps.append("   - Prepare for production deployment")
        
        return next_steps
    
    def _determine_priority_level(self, analysis_results: Dict, df: pd.DataFrame) -> str:
        """Determine priority level for recommendations"""
        data_quality_score = analysis_results.get('data_quality_score', 100)
        missing_pct = analysis_results.get('missing_analysis', {}).get('missing_percentage', 0)
        
        if data_quality_score < 60 or missing_pct > 30:
            return 'high'
        elif data_quality_score < 80 or missing_pct > 15:
            return 'medium'
        else:
            return 'low'


# Convenience functions for direct use
def interpret_statistical_results(test_results: Dict, context: Dict) -> Dict[str, Any]:
    """Interpret statistical test results"""
    engine = AdvancedInsightsEngine()
    return engine.interpret_statistical_results(test_results, context)

def assess_data_quality(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """Assess data quality comprehensively"""
    engine = AdvancedInsightsEngine()
    return engine.assess_data_quality(df, target_col)

def generate_recommendations(df: pd.DataFrame, analysis_results: Dict, target_col: str = None) -> Dict[str, Any]:
    """Generate actionable recommendations"""
    engine = AdvancedInsightsEngine()
    return engine.generate_recommendations(df, analysis_results, target_col)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    df = pd.DataFrame({
        'normal_feature': np.random.normal(100, 15, 1000),
        'skewed_feature': np.random.exponential(2, 1000),
        'categorical': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Add some missing values and outliers
    df.loc[50:100, 'normal_feature'] = np.nan
    df.loc[900:950, 'skewed_feature'] = df.loc[900:950, 'skewed_feature'] * 10  # Create outliers
    
    print("=== Advanced Insights Engine Test ===")
    
    # Test data quality assessment
    quality_assessment = assess_data_quality(df, target_col='target')
    print(f"\n📊 Data Quality Score: {quality_assessment['overall_score']:.1f}/100")
    print(f"🎯 Model Readiness: {quality_assessment['model_readiness']}")
    
    # Test recommendations
    mock_analysis_results = {
        'problem_type': 'classification',
        'data_quality_score': quality_assessment['overall_score'],
        'missing_analysis': {'missing_percentage': 5.1, 'total_missing': 51},
        'target_analysis': {'imbalance_detected': False}
    }
    
    recommendations = generate_recommendations(df, mock_analysis_results, target_col='target')
    print(f"\n🚀 Priority Level: {recommendations['priority_level']}")
    print(f"📋 Preprocessing Steps: {len(recommendations['preprocessing_steps'])}")
    print(f"🔧 Feature Engineering: {len(recommendations['feature_engineering'])}")
    print(f"🤖 Model Recommendations: {len(recommendations['model_selection'])}")
    
    print("\n✅ Advanced Insights Engine test completed successfully!")