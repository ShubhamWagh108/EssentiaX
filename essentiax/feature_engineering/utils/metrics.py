"""
Feature Quality Metrics
=======================

Comprehensive metrics for assessing feature quality and engineering effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import f_classif, f_regression
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import warnings

class FeatureQualityMetrics:
    """
    📊 Feature Quality Assessment
    
    Comprehensive metrics for evaluating feature quality:
    - Statistical significance
    - Information content
    - Predictive power
    - Stability measures
    - Data quality indicators
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def assess_feature_quality(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        include_target_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Comprehensive feature quality assessment.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
        include_target_metrics : bool, default=True
            Whether to include target-related metrics
            
        Returns:
        --------
        quality_df : pd.DataFrame
            Feature quality metrics
        """
        
        if self.verbose:
            print("📊 Assessing feature quality...")
        
        quality_metrics = {}
        
        for col in X.columns:
            metrics = {}
            series = X[col]
            
            # Basic data quality metrics
            metrics.update(self._calculate_basic_metrics(series))
            
            # Statistical metrics
            metrics.update(self._calculate_statistical_metrics(series))
            
            # Information content metrics
            metrics.update(self._calculate_information_metrics(series))
            
            # Target-related metrics (if target is provided)
            if y is not None and include_target_metrics:
                metrics.update(self._calculate_target_metrics(series, y, col))
            
            quality_metrics[col] = metrics
        
        # Convert to DataFrame
        quality_df = pd.DataFrame(quality_metrics).T
        
        # Calculate overall quality score
        quality_df['overall_quality_score'] = self._calculate_overall_quality_score(quality_df)
        
        # Rank features by quality
        quality_df['quality_rank'] = quality_df['overall_quality_score'].rank(ascending=False)
        
        return quality_df.sort_values('overall_quality_score', ascending=False)
    
    def _calculate_basic_metrics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate basic data quality metrics."""
        
        metrics = {}
        
        # Missing values
        metrics['missing_ratio'] = series.isnull().sum() / len(series)
        metrics['completeness_score'] = 1.0 - metrics['missing_ratio']
        
        # Uniqueness
        metrics['unique_count'] = series.nunique()
        metrics['unique_ratio'] = series.nunique() / len(series)
        
        # Constant values
        metrics['is_constant'] = series.nunique() <= 1
        metrics['constant_score'] = 0.0 if metrics['is_constant'] else 1.0
        
        # Data type consistency
        if pd.api.types.is_numeric_dtype(series):
            metrics['data_type'] = 'numeric'
            metrics['type_consistency_score'] = 1.0
        elif pd.api.types.is_datetime64_any_dtype(series):
            metrics['data_type'] = 'datetime'
            metrics['type_consistency_score'] = 1.0
        else:
            metrics['data_type'] = 'categorical'
            # Check for mixed types in categorical
            try:
                pd.to_numeric(series.dropna())
                metrics['type_consistency_score'] = 0.5  # Mixed numeric/categorical
            except:
                metrics['type_consistency_score'] = 1.0
        
        return metrics
    
    def _calculate_statistical_metrics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate statistical metrics."""
        
        metrics = {}
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric statistics
            metrics['mean'] = series.mean()
            metrics['std'] = series.std()
            metrics['skewness'] = series.skew()
            metrics['kurtosis'] = series.kurtosis()
            
            # Normality indicators
            metrics['skewness_score'] = max(0, 1 - abs(series.skew()) / 3)  # Penalize high skewness
            metrics['kurtosis_score'] = max(0, 1 - abs(series.kurtosis()) / 5)  # Penalize high kurtosis
            
            # Outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            metrics['outlier_ratio'] = outliers / len(series)
            metrics['outlier_score'] = max(0, 1 - metrics['outlier_ratio'] * 2)  # Penalize outliers
            
            # Variance
            metrics['variance'] = series.var()
            metrics['variance_score'] = min(1.0, series.var() / (series.var() + 1))  # Normalized variance score
            
        else:
            # Categorical statistics
            value_counts = series.value_counts()
            
            # Distribution balance
            if len(value_counts) > 1:
                # Calculate entropy for balance
                probabilities = value_counts / len(series)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
                max_entropy = np.log2(len(value_counts))
                metrics['entropy'] = entropy
                metrics['balance_score'] = entropy / max_entropy if max_entropy > 0 else 0
            else:
                metrics['entropy'] = 0
                metrics['balance_score'] = 0
            
            # Cardinality appropriateness
            cardinality = series.nunique()
            if cardinality == 1:
                metrics['cardinality_score'] = 0.0  # Constant
            elif cardinality == 2:
                metrics['cardinality_score'] = 1.0  # Binary - good
            elif cardinality <= 10:
                metrics['cardinality_score'] = 0.9  # Low cardinality - good
            elif cardinality <= 50:
                metrics['cardinality_score'] = 0.7  # Medium cardinality - okay
            else:
                metrics['cardinality_score'] = 0.5  # High cardinality - may need encoding
        
        return metrics
    
    def _calculate_information_metrics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate information content metrics."""
        
        metrics = {}
        
        # Information content based on entropy
        if pd.api.types.is_numeric_dtype(series):
            # For numeric features, discretize first
            try:
                discretized = pd.cut(series.dropna(), bins=10, duplicates='drop')
                value_counts = discretized.value_counts()
            except:
                # Fallback for edge cases
                value_counts = series.value_counts()
        else:
            value_counts = series.value_counts()
        
        if len(value_counts) > 1:
            probabilities = value_counts / value_counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
            max_entropy = np.log2(len(value_counts))
            metrics['information_content'] = entropy / max_entropy if max_entropy > 0 else 0
        else:
            metrics['information_content'] = 0
        
        # Discriminative power (based on value distribution)
        if len(value_counts) > 1:
            # Gini impurity
            gini = 1 - np.sum((probabilities) ** 2)
            metrics['discriminative_power'] = gini
        else:
            metrics['discriminative_power'] = 0
        
        return metrics
    
    def _calculate_target_metrics(self, series: pd.Series, y: pd.Series, feature_name: str) -> Dict[str, float]:
        """Calculate target-related metrics."""
        
        metrics = {}
        
        # Determine target type
        if y.dtype in ['object', 'category'] or y.nunique() < 20:
            target_type = 'classification'
        else:
            target_type = 'regression'
        
        try:
            if pd.api.types.is_numeric_dtype(series) and target_type == 'regression':
                # Numeric feature, numeric target
                correlation, p_value = pearsonr(series.dropna(), y[series.dropna().index])
                metrics['pearson_correlation'] = abs(correlation)
                metrics['correlation_p_value'] = p_value
                metrics['correlation_significance'] = 1.0 if p_value < 0.05 else 0.0
                
                # Spearman correlation (non-linear relationships)
                spearman_corr, spearman_p = spearmanr(series.dropna(), y[series.dropna().index])
                metrics['spearman_correlation'] = abs(spearman_corr)
                
            elif pd.api.types.is_numeric_dtype(series) and target_type == 'classification':
                # Numeric feature, categorical target
                # F-statistic
                f_stat, f_p_value = f_classif(series.values.reshape(-1, 1), y)
                metrics['f_statistic'] = f_stat[0]
                metrics['f_p_value'] = f_p_value[0]
                metrics['f_significance'] = 1.0 if f_p_value[0] < 0.05 else 0.0
                
            else:
                # Categorical feature
                if target_type == 'classification':
                    # Categorical feature, categorical target - Chi-square
                    contingency_table = pd.crosstab(series, y)
                    chi2, chi2_p, dof, expected = chi2_contingency(contingency_table)
                    metrics['chi2_statistic'] = chi2
                    metrics['chi2_p_value'] = chi2_p
                    metrics['chi2_significance'] = 1.0 if chi2_p < 0.05 else 0.0
                    
                    # Cramér's V
                    n = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                    metrics['cramers_v'] = cramers_v
                    
                else:
                    # Categorical feature, numeric target - ANOVA F-test
                    f_stat, f_p_value = f_regression(
                        pd.get_dummies(series, drop_first=True).values,
                        y
                    )
                    metrics['f_statistic'] = np.mean(f_stat)
                    metrics['f_p_value'] = np.mean(f_p_value)
                    metrics['f_significance'] = 1.0 if np.mean(f_p_value) < 0.05 else 0.0
            
            # Mutual information
            if target_type == 'classification':
                mi_score = mutual_info_score(series.astype(str), y.astype(str))
                nmi_score = normalized_mutual_info_score(series.astype(str), y.astype(str))
            else:
                # For regression, discretize target
                y_discretized = pd.cut(y, bins=10, duplicates='drop')
                mi_score = mutual_info_score(series.astype(str), y_discretized.astype(str))
                nmi_score = normalized_mutual_info_score(series.astype(str), y_discretized.astype(str))
            
            metrics['mutual_information'] = mi_score
            metrics['normalized_mutual_information'] = nmi_score
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate target metrics for {feature_name}: {e}")
            # Set default values
            metrics['target_relationship_strength'] = 0.0
        
        return metrics
    
    def _calculate_overall_quality_score(self, quality_df: pd.DataFrame) -> pd.Series:
        """Calculate overall quality score for each feature."""
        
        scores = pd.Series(index=quality_df.index, dtype=float)
        
        for feature in quality_df.index:
            score = 0.0
            
            # Data quality components (40% weight)
            if 'completeness_score' in quality_df.columns:
                score += quality_df.loc[feature, 'completeness_score'] * 0.15
            if 'constant_score' in quality_df.columns:
                score += quality_df.loc[feature, 'constant_score'] * 0.15
            if 'type_consistency_score' in quality_df.columns:
                score += quality_df.loc[feature, 'type_consistency_score'] * 0.10
            
            # Statistical quality components (30% weight)
            if 'balance_score' in quality_df.columns:
                score += quality_df.loc[feature, 'balance_score'] * 0.10
            elif 'variance_score' in quality_df.columns:
                score += quality_df.loc[feature, 'variance_score'] * 0.10
            
            if 'outlier_score' in quality_df.columns:
                score += quality_df.loc[feature, 'outlier_score'] * 0.10
            if 'skewness_score' in quality_df.columns:
                score += quality_df.loc[feature, 'skewness_score'] * 0.10
            
            # Information content (30% weight)
            if 'information_content' in quality_df.columns:
                score += quality_df.loc[feature, 'information_content'] * 0.15
            if 'discriminative_power' in quality_df.columns:
                score += quality_df.loc[feature, 'discriminative_power'] * 0.15
            
            # Ensure score is between 0 and 1
            scores[feature] = max(0.0, min(1.0, score))
        
        return scores
    
    def get_feature_recommendations(self, quality_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Get feature recommendations based on quality assessment.
        
        Parameters:
        -----------
        quality_df : pd.DataFrame
            Feature quality assessment results
        threshold : float, default=0.5
            Quality threshold for recommendations
            
        Returns:
        --------
        recommendations : dict
            Feature recommendations by category
        """
        
        recommendations = {
            'high_quality': [],
            'medium_quality': [],
            'low_quality': [],
            'needs_attention': [],
            'consider_removing': []
        }
        
        for feature in quality_df.index:
            quality_score = quality_df.loc[feature, 'overall_quality_score']
            
            if quality_score >= 0.8:
                recommendations['high_quality'].append(feature)
            elif quality_score >= 0.6:
                recommendations['medium_quality'].append(feature)
            elif quality_score >= 0.4:
                recommendations['low_quality'].append(feature)
            elif quality_score >= 0.2:
                recommendations['needs_attention'].append(feature)
            else:
                recommendations['consider_removing'].append(feature)
        
        # Additional specific recommendations
        for feature in quality_df.index:
            # High missing values
            if quality_df.loc[feature, 'missing_ratio'] > 0.5:
                if feature not in recommendations['consider_removing']:
                    recommendations['needs_attention'].append(feature)
            
            # Constant features
            if quality_df.loc[feature, 'is_constant']:
                if feature not in recommendations['consider_removing']:
                    recommendations['consider_removing'].append(feature)
            
            # High outlier ratio
            if 'outlier_ratio' in quality_df.columns and quality_df.loc[feature, 'outlier_ratio'] > 0.2:
                if feature not in recommendations['needs_attention']:
                    recommendations['needs_attention'].append(feature)
        
        return recommendations
    
    def generate_quality_report(self, quality_df: pd.DataFrame) -> str:
        """
        Generate a comprehensive quality report.
        
        Parameters:
        -----------
        quality_df : pd.DataFrame
            Feature quality assessment results
            
        Returns:
        --------
        report : str
            Formatted quality report
        """
        
        recommendations = self.get_feature_recommendations(quality_df)
        
        report = []
        report.append("📊 FEATURE QUALITY ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("📈 SUMMARY STATISTICS")
        report.append("-" * 30)
        report.append(f"Total features analyzed: {len(quality_df)}")
        report.append(f"Average quality score: {quality_df['overall_quality_score'].mean():.3f}")
        report.append(f"Median quality score: {quality_df['overall_quality_score'].median():.3f}")
        report.append(f"Best feature: {quality_df['overall_quality_score'].idxmax()} ({quality_df['overall_quality_score'].max():.3f})")
        report.append(f"Worst feature: {quality_df['overall_quality_score'].idxmin()} ({quality_df['overall_quality_score'].min():.3f})")
        report.append("")
        
        # Feature recommendations
        report.append("🎯 FEATURE RECOMMENDATIONS")
        report.append("-" * 30)
        
        for category, features in recommendations.items():
            if features:
                category_name = category.replace('_', ' ').title()
                report.append(f"{category_name}: {len(features)} features")
                if len(features) <= 10:
                    report.append(f"  {', '.join(features)}")
                else:
                    report.append(f"  {', '.join(features[:10])}... (and {len(features)-10} more)")
                report.append("")
        
        # Data quality issues
        report.append("⚠️ DATA QUALITY ISSUES")
        report.append("-" * 30)
        
        # Missing values
        high_missing = quality_df[quality_df['missing_ratio'] > 0.1]
        if len(high_missing) > 0:
            report.append(f"Features with >10% missing values: {len(high_missing)}")
            for feature in high_missing.index[:5]:
                missing_pct = high_missing.loc[feature, 'missing_ratio'] * 100
                report.append(f"  {feature}: {missing_pct:.1f}% missing")
            report.append("")
        
        # Constant features
        constant_features = quality_df[quality_df['is_constant'] == True]
        if len(constant_features) > 0:
            report.append(f"Constant features (consider removing): {len(constant_features)}")
            report.append(f"  {', '.join(constant_features.index.tolist())}")
            report.append("")
        
        # High outlier features
        if 'outlier_ratio' in quality_df.columns:
            high_outliers = quality_df[quality_df['outlier_ratio'] > 0.1]
            if len(high_outliers) > 0:
                report.append(f"Features with >10% outliers: {len(high_outliers)}")
                for feature in high_outliers.index[:5]:
                    outlier_pct = high_outliers.loc[feature, 'outlier_ratio'] * 100
                    report.append(f"  {feature}: {outlier_pct:.1f}% outliers")
                report.append("")
        
        return "\n".join(report)