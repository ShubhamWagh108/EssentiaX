"""
Smart Variable Detection & Selection for EssentiaX
=================================================
Intelligent automatic detection and categorization of variables for EDA

Features:
- Automatic target column detection
- Intelligent feature categorization
- Meaningful variable selection
- ID and constant column detection
- High-cardinality filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class SmartVariableDetector:
    """Intelligent variable detection and categorization engine"""
    
    def __init__(self):
        self.target_keywords = [
            'target', 'label', 'class', 'outcome', 'result', 'prediction',
            'y', 'dependent', 'response', 'output', 'goal', 'objective',
            'churn', 'fraud', 'default', 'conversion', 'success', 'failure',
            'price', 'cost', 'revenue', 'profit', 'sales', 'amount',
            'rating', 'score', 'grade', 'rank', 'level'
        ]
        
        self.id_keywords = [
            'id', 'index', 'key', 'identifier', 'uuid', 'guid',
            'customer_id', 'user_id', 'product_id', 'order_id',
            'transaction_id', 'session_id', 'account_id'
        ]
        
        self.datetime_keywords = [
            'date', 'time', 'timestamp', 'created', 'updated',
            'start', 'end', 'birth', 'death', 'expire'
        ]
        
        self.text_keywords = [
            'text', 'description', 'comment', 'review', 'feedback',
            'message', 'content', 'body', 'summary', 'notes'
        ]
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis and variable categorization
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        dict : Complete variable analysis results
        """
        analysis = {
            'dataset_info': self._get_dataset_info(df),
            'variable_categories': self._categorize_variables(df),
            'target_candidates': self._detect_target_candidates(df),
            'recommended_target': self._recommend_target(df),
            'feature_quality': self._assess_feature_quality(df),
            'columns_to_exclude': self._identify_exclusion_candidates(df),
            'meaningful_features': self._select_meaningful_features(df),
            'analysis_recommendations': self._generate_analysis_recommendations(df)
        }
        
        return analysis
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'total_cells': df.size,
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    
    def _categorize_variables(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Intelligent variable categorization"""
        categories = {
            'numeric_continuous': [],
            'numeric_discrete': [],
            'categorical_low_card': [],
            'categorical_high_card': [],
            'datetime': [],
            'text': [],
            'boolean': [],
            'id_columns': [],
            'constant_columns': [],
            'mixed_type': []
        }
        
        for col in df.columns:
            col_data = df[col]
            col_name_lower = col.lower()
            
            # Check for constant columns
            if col_data.nunique() <= 1:
                categories['constant_columns'].append(col)
                continue
            
            # Check for ID columns
            if self._is_id_column(col, col_data):
                categories['id_columns'].append(col)
                continue
            
            # Check data type and patterns
            if pd.api.types.is_datetime64_any_dtype(col_data):
                categories['datetime'].append(col)
            elif pd.api.types.is_bool_dtype(col_data):
                categories['boolean'].append(col)
            elif pd.api.types.is_numeric_dtype(col_data):
                if self._is_discrete_numeric(col_data):
                    categories['numeric_discrete'].append(col)
                else:
                    categories['numeric_continuous'].append(col)
            elif pd.api.types.is_object_dtype(col_data):
                # Check if it's actually datetime
                if any(keyword in col_name_lower for keyword in self.datetime_keywords):
                    if self._try_parse_datetime(col_data):
                        categories['datetime'].append(col)
                        continue
                
                # Check if it's text data
                if self._is_text_column(col, col_data):
                    categories['text'].append(col)
                else:
                    # Categorical data
                    unique_ratio = col_data.nunique() / len(col_data)
                    if unique_ratio > 0.5 or col_data.nunique() > 50:
                        categories['categorical_high_card'].append(col)
                    else:
                        categories['categorical_low_card'].append(col)
            else:
                categories['mixed_type'].append(col)
        
        return categories
    
    def _detect_target_candidates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential target columns"""
        candidates = []
        
        for col in df.columns:
            col_data = df[col]
            col_name_lower = col.lower()
            score = 0
            reasons = []
            
            # Check column name for target keywords
            for keyword in self.target_keywords:
                if keyword in col_name_lower:
                    score += 10
                    reasons.append(f"Contains keyword '{keyword}'")
                    break
            
            # Check if it's at the end (common for target columns)
            if col == df.columns[-1]:
                score += 5
                reasons.append("Last column (common target position)")
            
            # Check data characteristics
            if pd.api.types.is_numeric_dtype(col_data):
                unique_count = col_data.nunique()
                
                # Binary classification candidate
                if unique_count == 2:
                    score += 8
                    reasons.append("Binary variable (classification)")
                
                # Multi-class classification candidate
                elif 2 < unique_count <= 10:
                    score += 6
                    reasons.append(f"Low cardinality ({unique_count} classes)")
                
                # Regression candidate
                elif unique_count > 10:
                    # Check if it looks like a continuous target
                    if col_data.dtype in ['float64', 'float32']:
                        score += 4
                        reasons.append("Continuous numeric (regression)")
                    elif 'price' in col_name_lower or 'amount' in col_name_lower:
                        score += 6
                        reasons.append("Price/amount variable")
            
            elif pd.api.types.is_object_dtype(col_data):
                unique_count = col_data.nunique()
                
                # Categorical target candidate
                if 2 <= unique_count <= 20:
                    score += 5
                    reasons.append(f"Categorical with {unique_count} classes")
            
            # Penalize ID-like columns
            if self._is_id_column(col, col_data):
                score -= 10
                reasons.append("Appears to be ID column")
            
            # Penalize high cardinality
            if col_data.nunique() > len(df) * 0.8:
                score -= 5
                reasons.append("Very high cardinality")
            
            if score > 0:
                candidates.append({
                    'column': col,
                    'score': score,
                    'reasons': reasons,
                    'data_type': str(col_data.dtype),
                    'unique_count': col_data.nunique(),
                    'missing_count': col_data.isnull().sum()
                })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    
    def _recommend_target(self, df: pd.DataFrame) -> Optional[str]:
        """Recommend the best target column"""
        candidates = self._detect_target_candidates(df)
        
        if not candidates:
            return None
        
        # Return the highest scoring candidate
        best_candidate = candidates[0]
        
        # Lower threshold for better detection
        if best_candidate['score'] >= 3:
            return best_candidate['column']
        
        return None
    
    def _assess_feature_quality(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Assess the quality of each feature for analysis"""
        quality_assessment = {}
        
        for col in df.columns:
            col_data = df[col]
            
            assessment = {
                'missing_percentage': (col_data.isnull().sum() / len(df)) * 100,
                'unique_count': col_data.nunique(),
                'unique_ratio': col_data.nunique() / len(df),
                'data_type': str(col_data.dtype),
                'quality_score': 100,  # Start with perfect score
                'quality_issues': [],
                'recommendations': []
            }
            
            # Penalize for missing values
            missing_pct = assessment['missing_percentage']
            if missing_pct > 50:
                assessment['quality_score'] -= 40
                assessment['quality_issues'].append(f"High missing values ({missing_pct:.1f}%)")
                assessment['recommendations'].append("Consider dropping or imputing")
            elif missing_pct > 20:
                assessment['quality_score'] -= 20
                assessment['quality_issues'].append(f"Moderate missing values ({missing_pct:.1f}%)")
                assessment['recommendations'].append("Consider imputation")
            
            # Penalize for constant/near-constant columns
            if assessment['unique_count'] <= 1:
                assessment['quality_score'] -= 50
                assessment['quality_issues'].append("Constant column")
                assessment['recommendations'].append("Remove - no information value")
            elif assessment['unique_ratio'] < 0.01:
                assessment['quality_score'] -= 30
                assessment['quality_issues'].append("Near-constant column")
                assessment['recommendations'].append("Consider removing")
            
            # Penalize for very high cardinality
            if assessment['unique_ratio'] > 0.9:
                assessment['quality_score'] -= 25
                assessment['quality_issues'].append("Very high cardinality")
                assessment['recommendations'].append("Likely ID column - consider removing")
            
            # Check for mixed data types in object columns
            if pd.api.types.is_object_dtype(col_data):
                if self._has_mixed_types(col_data):
                    assessment['quality_score'] -= 15
                    assessment['quality_issues'].append("Mixed data types")
                    assessment['recommendations'].append("Clean and standardize data types")
            
            quality_assessment[col] = assessment
        
        return quality_assessment
    
    def _identify_exclusion_candidates(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify columns that should be excluded from analysis"""
        exclusions = {
            'id_columns': [],
            'constant_columns': [],
            'high_missing': [],
            'high_cardinality': [],
            'mixed_type': []
        }
        
        quality_assessment = self._assess_feature_quality(df)
        
        for col, assessment in quality_assessment.items():
            # ID columns
            if self._is_id_column(col, df[col]):
                exclusions['id_columns'].append(col)
            
            # Constant columns
            elif assessment['unique_count'] <= 1:
                exclusions['constant_columns'].append(col)
            
            # High missing values
            elif assessment['missing_percentage'] > 70:
                exclusions['high_missing'].append(col)
            
            # Very high cardinality
            elif assessment['unique_ratio'] > 0.95:
                exclusions['high_cardinality'].append(col)
            
            # Mixed types
            elif pd.api.types.is_object_dtype(df[col]) and self._has_mixed_types(df[col]):
                exclusions['mixed_type'].append(col)
        
        return exclusions
    
    def _select_meaningful_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Select meaningful features for analysis"""
        categories = self._categorize_variables(df)
        exclusions = self._identify_exclusion_candidates(df)
        
        # Get all columns to exclude
        exclude_cols = set()
        for exclusion_list in exclusions.values():
            exclude_cols.update(exclusion_list)
        
        meaningful_features = {
            'numeric_features': [],
            'categorical_features': [],
            'datetime_features': [],
            'text_features': [],
            'all_meaningful': []
        }
        
        # Select meaningful numeric features
        for col in categories['numeric_continuous'] + categories['numeric_discrete']:
            if col not in exclude_cols:
                meaningful_features['numeric_features'].append(col)
        
        # Select meaningful categorical features
        for col in categories['categorical_low_card']:
            if col not in exclude_cols:
                meaningful_features['categorical_features'].append(col)
        
        # Select datetime features
        for col in categories['datetime']:
            if col not in exclude_cols:
                meaningful_features['datetime_features'].append(col)
        
        # Select text features (limited)
        for col in categories['text'][:3]:  # Limit to 3 text columns
            if col not in exclude_cols:
                meaningful_features['text_features'].append(col)
        
        # Combine all meaningful features
        meaningful_features['all_meaningful'] = (
            meaningful_features['numeric_features'] +
            meaningful_features['categorical_features'] +
            meaningful_features['datetime_features'] +
            meaningful_features['text_features']
        )
        
        return meaningful_features
    
    def _generate_analysis_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate recommendations for analysis approach"""
        target_candidates = self._detect_target_candidates(df)
        meaningful_features = self._select_meaningful_features(df)
        
        recommendations = {
            'analysis_type': 'exploratory',
            'target_recommendation': None,
            'problem_type': None,
            'key_features': meaningful_features['all_meaningful'][:20],  # Top 20
            'analysis_focus': [],
            'preprocessing_needed': [],
            'visualization_priorities': []
        }
        
        # Determine analysis type and problem type
        if target_candidates and target_candidates[0]['score'] >= 5:
            recommendations['analysis_type'] = 'supervised'
            recommendations['target_recommendation'] = target_candidates[0]['column']
            
            target_col = target_candidates[0]['column']
            target_data = df[target_col]
            
            # Determine problem type
            if pd.api.types.is_numeric_dtype(target_data):
                if target_data.nunique() <= 10:
                    recommendations['problem_type'] = 'classification'
                else:
                    recommendations['problem_type'] = 'regression'
            else:
                recommendations['problem_type'] = 'classification'
        
        # Analysis focus recommendations
        if len(meaningful_features['numeric_features']) > 5:
            recommendations['analysis_focus'].append('correlation_analysis')
            recommendations['visualization_priorities'].append('correlation_heatmap')
        
        if len(meaningful_features['categorical_features']) > 3:
            recommendations['analysis_focus'].append('categorical_analysis')
            recommendations['visualization_priorities'].append('categorical_distributions')
        
        if meaningful_features['datetime_features']:
            recommendations['analysis_focus'].append('time_series_analysis')
            recommendations['visualization_priorities'].append('time_series_plots')
        
        # Preprocessing recommendations
        quality_assessment = self._assess_feature_quality(df)
        missing_cols = [col for col, assess in quality_assessment.items() 
                       if assess['missing_percentage'] > 5]
        
        if missing_cols:
            recommendations['preprocessing_needed'].append('missing_value_imputation')
        
        if any('Mixed data types' in assess['quality_issues'] 
               for assess in quality_assessment.values()):
            recommendations['preprocessing_needed'].append('data_type_standardization')
        
        return recommendations
    
    def _is_id_column(self, col_name: str, col_data: pd.Series) -> bool:
        """Check if column is likely an ID column"""
        col_name_lower = col_name.lower()
        
        # Check column name
        if any(keyword in col_name_lower for keyword in self.id_keywords):
            return True
        
        # Check data characteristics
        unique_ratio = col_data.nunique() / len(col_data)
        
        # High uniqueness suggests ID
        if unique_ratio > 0.95:
            return True
        
        # Sequential integers suggest ID
        if pd.api.types.is_integer_dtype(col_data):
            if col_data.nunique() == len(col_data):
                # Check if sequential
                sorted_values = col_data.dropna().sort_values()
                if len(sorted_values) > 1:
                    diffs = sorted_values.diff().dropna()
                    if (diffs == 1).all():
                        return True
        
        return False
    
    def _is_discrete_numeric(self, col_data: pd.Series) -> bool:
        """Check if numeric column is discrete"""
        if not pd.api.types.is_numeric_dtype(col_data):
            return False
        
        # Check if all values are integers
        if pd.api.types.is_integer_dtype(col_data):
            return True
        
        # Check if float values are actually integers
        if pd.api.types.is_float_dtype(col_data):
            non_null_data = col_data.dropna()
            if len(non_null_data) > 0:
                return (non_null_data % 1 == 0).all()
        
        return False
    
    def _is_text_column(self, col_name: str, col_data: pd.Series) -> bool:
        """Check if column contains text data"""
        col_name_lower = col_name.lower()
        
        # Check column name
        if any(keyword in col_name_lower for keyword in self.text_keywords):
            return True
        
        # Check data characteristics
        if pd.api.types.is_object_dtype(col_data):
            sample_data = col_data.dropna().head(100)
            if len(sample_data) > 0:
                avg_length = sample_data.astype(str).str.len().mean()
                
                # Long strings suggest text
                if avg_length > 50:
                    return True
                
                # Check for spaces (indicates sentences/phrases)
                has_spaces = sample_data.astype(str).str.contains(' ').any()
                if has_spaces and avg_length > 20:
                    return True
        
        return False
    
    def _try_parse_datetime(self, col_data: pd.Series) -> bool:
        """Try to parse column as datetime"""
        try:
            sample = col_data.dropna().head(100)
            if len(sample) > 0:
                pd.to_datetime(sample, errors='raise')
                return True
        except:
            pass
        return False
    
    def _has_mixed_types(self, col_data: pd.Series) -> bool:
        """Check if object column has mixed data types"""
        if not pd.api.types.is_object_dtype(col_data):
            return False
        
        sample_data = col_data.dropna().head(1000)
        if len(sample_data) == 0:
            return False
        
        type_counts = Counter(type(val).__name__ for val in sample_data)
        return len(type_counts) > 1


# Convenience functions
def analyze_variables(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and categorize variables in a dataframe"""
    detector = SmartVariableDetector()
    return detector.analyze_dataset(df)

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detect the most likely target column"""
    detector = SmartVariableDetector()
    return detector._recommend_target(df)

def get_meaningful_features(df: pd.DataFrame) -> List[str]:
    """Get list of meaningful features for analysis"""
    detector = SmartVariableDetector()
    meaningful = detector._select_meaningful_features(df)
    return meaningful['all_meaningful']


# Example usage
if __name__ == "__main__":
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'customer_id': range(1000),  # ID column
        'age': np.random.randint(18, 80, 1000),  # Numeric
        'income': np.random.normal(50000, 15000, 1000),  # Numeric continuous
        'category': np.random.choice(['A', 'B', 'C'], 1000),  # Categorical
        'high_card_cat': [f'item_{i}' for i in np.random.randint(0, 800, 1000)],  # High cardinality
        'constant_col': 'same_value',  # Constant
        'target_variable': np.random.choice([0, 1], 1000),  # Target
        'description': [f'This is description {i}' for i in range(1000)],  # Text
        'created_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000]  # Datetime
    })
    
    # Add missing values
    test_df.loc[50:100, 'income'] = np.nan
    
    print("=== Smart Variable Detection Test ===")
    
    detector = SmartVariableDetector()
    analysis = detector.analyze_dataset(test_df)
    
    print(f"Dataset shape: {analysis['dataset_info']['shape']}")
    print(f"Missing percentage: {analysis['dataset_info']['missing_percentage']:.1f}%")
    
    print(f"\nRecommended target: {analysis['recommended_target']}")
    print(f"Problem type: {analysis['analysis_recommendations']['problem_type']}")
    
    print(f"\nMeaningful features ({len(analysis['meaningful_features']['all_meaningful'])}):")
    for feature in analysis['meaningful_features']['all_meaningful']:
        print(f"  - {feature}")
    
    print(f"\nColumns to exclude:")
    for category, cols in analysis['columns_to_exclude'].items():
        if cols:
            print(f"  {category}: {cols}")
    
    print("\n✅ Smart Variable Detection test completed!")