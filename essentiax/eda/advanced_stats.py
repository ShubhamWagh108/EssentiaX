"""
Advanced Statistical Analysis Module for EssentiaX
=================================================
Enhanced statistical tests and analysis for big data EDA

Features:
- Normality testing suite
- Advanced correlation analysis  
- Statistical significance tests
- Enhanced outlier detection
- AI-powered interpretations
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, anderson, kstest, jarque_bera, normaltest
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, chi2_contingency, f_oneway, kruskal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedStatistics:
    """Advanced statistical analysis engine for EssentiaX"""
    
    def __init__(self):
        self.results = {}
        self.interpretations = {}
    
    def test_normality(self, data, column_name, alpha=0.05):
        """
        Comprehensive normality testing using multiple methods
        
        Parameters:
        -----------
        data : array-like
            Data to test for normality
        column_name : str
            Name of the column being tested
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        dict : Test results with interpretations
        """
        clean_data = pd.Series(data).dropna()
        
        if len(clean_data) < 3:
            return {
                'column': column_name,
                'tests': {},
                'interpretation': 'Insufficient data for normality testing',
                'is_normal': None
            }
        
        results = {
            'column': column_name,
            'sample_size': len(clean_data),
            'tests': {},
            'interpretation': '',
            'is_normal': None
        }
        
        # Shapiro-Wilk Test (best for n < 5000)
        if len(clean_data) <= 5000:
            try:
                stat, p_value = shapiro(clean_data)
                results['tests']['shapiro_wilk'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > alpha
                }
            except:
                pass
        
        # Anderson-Darling Test (good for any size)
        try:
            result = anderson(clean_data, dist='norm')
            # Use 5% significance level (index 2)
            critical_value = result.critical_values[2]
            is_normal = result.statistic < critical_value
            results['tests']['anderson_darling'] = {
                'statistic': float(result.statistic),
                'critical_value': float(critical_value),
                'is_normal': is_normal
            }
        except:
            pass
        
        # Kolmogorov-Smirnov Test
        try:
            # Standardize data for KS test
            standardized = (clean_data - clean_data.mean()) / clean_data.std()
            stat, p_value = kstest(standardized, 'norm')
            results['tests']['kolmogorov_smirnov'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > alpha
            }
        except:
            pass
        
        # Jarque-Bera Test
        try:
            stat, p_value = jarque_bera(clean_data)
            results['tests']['jarque_bera'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > alpha
            }
        except:
            pass
        
        # D'Agostino's Test
        try:
            stat, p_value = normaltest(clean_data)
            results['tests']['dagostino'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': p_value > alpha
            }
        except:
            pass
        
        # Consensus decision
        normal_votes = []
        for test_name, test_result in results['tests'].items():
            if 'is_normal' in test_result:
                normal_votes.append(test_result['is_normal'])
        
        if normal_votes:
            consensus = sum(normal_votes) / len(normal_votes)
            results['is_normal'] = consensus >= 0.5
            results['consensus_score'] = consensus
        
        # Generate interpretation
        results['interpretation'] = self._interpret_normality_results(results)
        
        return results
    
    def advanced_correlation_analysis(self, df, columns=None, alpha=0.05):
        """
        Advanced correlation analysis with statistical significance
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        columns : list, optional
            Columns to analyze (default: all numeric columns)
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Correlation results with interpretations
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) < 2:
            return {
                'error': 'Need at least 2 numeric columns for correlation analysis',
                'columns_available': len(columns)
            }
        
        df_clean = df[columns].dropna()
        
        results = {
            'columns': columns,
            'sample_size': len(df_clean),
            'correlations': {},
            'significant_pairs': [],
            'interpretation': ''
        }
        
        # Pearson correlations with p-values
        pearson_corr = df_clean.corr(method='pearson')
        pearson_pvalues = pd.DataFrame(index=columns, columns=columns, dtype=float)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    try:
                        corr, p_val = pearsonr(df_clean[col1], df_clean[col2])
                        pearson_pvalues.loc[col1, col2] = p_val
                    except:
                        pearson_pvalues.loc[col1, col2] = np.nan
                else:
                    pearson_pvalues.loc[col1, col2] = 0.0
        
        # Spearman correlations
        spearman_corr = df_clean.corr(method='spearman')
        
        # Kendall correlations
        kendall_corr = df_clean.corr(method='kendall')
        
        results['correlations'] = {
            'pearson': {
                'correlation_matrix': pearson_corr.to_dict(),
                'p_values': pearson_pvalues.to_dict()
            },
            'spearman': {
                'correlation_matrix': spearman_corr.to_dict()
            },
            'kendall': {
                'correlation_matrix': kendall_corr.to_dict()
            }
        }
        
        # Find significant correlations
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                pearson_r = pearson_corr.loc[col1, col2]
                pearson_p = pearson_pvalues.loc[col1, col2]
                
                if not np.isnan(pearson_p) and pearson_p < alpha and abs(pearson_r) > 0.1:
                    results['significant_pairs'].append({
                        'variable1': col1,
                        'variable2': col2,
                        'pearson_r': float(pearson_r),
                        'pearson_p': float(pearson_p),
                        'spearman_r': float(spearman_corr.loc[col1, col2]),
                        'kendall_tau': float(kendall_corr.loc[col1, col2]),
                        'strength': self._correlation_strength(abs(pearson_r)),
                        'direction': 'positive' if pearson_r > 0 else 'negative'
                    })
        
        # Sort by absolute correlation strength
        results['significant_pairs'].sort(key=lambda x: abs(x['pearson_r']), reverse=True)
        
        # Generate interpretation
        results['interpretation'] = self._interpret_correlation_results(results)
        
        return results
    
    def statistical_tests_suite(self, df, target_col=None, alpha=0.05):
        """
        Comprehensive statistical testing suite
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_col : str, optional
            Target column for supervised tests
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Statistical test results
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        results = {
            'sample_size': len(df),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'tests_performed': [],
            'significant_results': [],
            'interpretation': ''
        }
        
        # One-sample t-tests (test if mean differs from 0)
        for col in numeric_cols:
            clean_data = df[col].dropna()
            if len(clean_data) > 1:
                try:
                    stat, p_value = ttest_1samp(clean_data, 0)
                    test_result = {
                        'test_type': 'one_sample_ttest',
                        'column': col,
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': p_value < alpha,
                        'interpretation': f"Mean of {col} {'significantly' if p_value < alpha else 'not significantly'} differs from 0"
                    }
                    results['tests_performed'].append(test_result)
                    if p_value < alpha:
                        results['significant_results'].append(test_result)
                except:
                    pass
        
        # Two-sample t-tests (if target column is binary)
        if target_col and target_col in df.columns:
            target_unique = df[target_col].nunique()
            
            if target_unique == 2:  # Binary target
                groups = df[target_col].unique()
                for col in numeric_cols:
                    if col != target_col:
                        try:
                            group1 = df[df[target_col] == groups[0]][col].dropna()
                            group2 = df[df[target_col] == groups[1]][col].dropna()
                            
                            if len(group1) > 1 and len(group2) > 1:
                                # T-test
                                stat, p_value = ttest_ind(group1, group2)
                                test_result = {
                                    'test_type': 'two_sample_ttest',
                                    'column': col,
                                    'target': target_col,
                                    'statistic': float(stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < alpha,
                                    'interpretation': f"{col} means {'significantly' if p_value < alpha else 'not significantly'} differ between {target_col} groups"
                                }
                                results['tests_performed'].append(test_result)
                                if p_value < alpha:
                                    results['significant_results'].append(test_result)
                                
                                # Mann-Whitney U test (non-parametric alternative)
                                stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                                test_result = {
                                    'test_type': 'mann_whitney_u',
                                    'column': col,
                                    'target': target_col,
                                    'statistic': float(stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < alpha,
                                    'interpretation': f"{col} distributions {'significantly' if p_value < alpha else 'not significantly'} differ between {target_col} groups"
                                }
                                results['tests_performed'].append(test_result)
                                if p_value < alpha:
                                    results['significant_results'].append(test_result)
                        except:
                            pass
        
        # Chi-square tests for categorical variables
        if len(categorical_cols) >= 2:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            test_result = {
                                'test_type': 'chi_square_independence',
                                'variable1': col1,
                                'variable2': col2,
                                'chi2_statistic': float(chi2),
                                'p_value': float(p_value),
                                'degrees_of_freedom': int(dof),
                                'significant': p_value < alpha,
                                'interpretation': f"{col1} and {col2} are {'significantly' if p_value < alpha else 'not significantly'} associated"
                            }
                            results['tests_performed'].append(test_result)
                            if p_value < alpha:
                                results['significant_results'].append(test_result)
                    except:
                        pass
        
        # ANOVA (if target has multiple categories)
        if target_col and target_col in df.columns:
            target_unique = df[target_col].nunique()
            if target_unique > 2 and target_unique <= 10:  # Multi-class target
                for col in numeric_cols:
                    if col != target_col:
                        try:
                            groups = [df[df[target_col] == group][col].dropna() for group in df[target_col].unique()]
                            groups = [g for g in groups if len(g) > 1]  # Remove empty groups
                            
                            if len(groups) >= 2:
                                # One-way ANOVA
                                stat, p_value = f_oneway(*groups)
                                test_result = {
                                    'test_type': 'one_way_anova',
                                    'column': col,
                                    'target': target_col,
                                    'f_statistic': float(stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < alpha,
                                    'interpretation': f"{col} means {'significantly' if p_value < alpha else 'not significantly'} differ across {target_col} groups"
                                }
                                results['tests_performed'].append(test_result)
                                if p_value < alpha:
                                    results['significant_results'].append(test_result)
                                
                                # Kruskal-Wallis test (non-parametric alternative)
                                stat, p_value = kruskal(*groups)
                                test_result = {
                                    'test_type': 'kruskal_wallis',
                                    'column': col,
                                    'target': target_col,
                                    'h_statistic': float(stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < alpha,
                                    'interpretation': f"{col} distributions {'significantly' if p_value < alpha else 'not significantly'} differ across {target_col} groups"
                                }
                                results['tests_performed'].append(test_result)
                                if p_value < alpha:
                                    results['significant_results'].append(test_result)
                        except:
                            pass
        
        # Generate overall interpretation
        results['interpretation'] = self._interpret_statistical_tests(results)
        
        return results
    
    def detect_outliers_advanced(self, data, column_name, methods=['iqr', 'zscore', 'isolation']):
        """
        Multi-method outlier detection with consensus scoring
        
        Parameters:
        -----------
        data : array-like
            Data to analyze for outliers
        column_name : str
            Name of the column
        methods : list
            Methods to use for outlier detection
            
        Returns:
        --------
        dict : Outlier detection results
        """
        clean_data = pd.Series(data).dropna()
        
        if len(clean_data) < 4:
            return {
                'column': column_name,
                'sample_size': len(clean_data),
                'error': 'Insufficient data for outlier detection'
            }
        
        results = {
            'column': column_name,
            'sample_size': len(clean_data),
            'methods': {},
            'consensus_outliers': [],
            'outlier_percentage': 0.0,
            'interpretation': ''
        }
        
        outlier_indices = {}
        
        # IQR Method
        if 'iqr' in methods:
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                
                results['methods']['iqr'] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(clean_data) * 100
                }
                outlier_indices['iqr'] = set(outliers.index)
        
        # Z-Score Method
        if 'zscore' in methods:
            z_scores = np.abs(stats.zscore(clean_data))
            outliers = clean_data[z_scores > 3]
            
            results['methods']['zscore'] = {
                'threshold': 3.0,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(clean_data) * 100
            }
            outlier_indices['zscore'] = set(outliers.index)
        
        # Modified Z-Score Method
        if 'modified_zscore' in methods:
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))
            
            if mad != 0:
                modified_z_scores = 0.6745 * (clean_data - median) / mad
                outliers = clean_data[np.abs(modified_z_scores) > 3.5]
                
                results['methods']['modified_zscore'] = {
                    'threshold': 3.5,
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(clean_data) * 100
                }
                outlier_indices['modified_zscore'] = set(outliers.index)
        
        # Isolation Forest Method
        if 'isolation' in methods and len(clean_data) >= 10:
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(clean_data.values.reshape(-1, 1))
                outliers = clean_data[outlier_labels == -1]
                
                results['methods']['isolation_forest'] = {
                    'contamination': 0.1,
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(clean_data) * 100
                }
                outlier_indices['isolation_forest'] = set(outliers.index)
            except:
                pass
        
        # Consensus outliers (detected by multiple methods)
        if len(outlier_indices) > 1:
            all_indices = set()
            for indices in outlier_indices.values():
                all_indices.update(indices)
            
            consensus_outliers = []
            for idx in all_indices:
                vote_count = sum(1 for indices in outlier_indices.values() if idx in indices)
                if vote_count >= len(outlier_indices) / 2:  # Majority vote
                    consensus_outliers.append(idx)
            
            results['consensus_outliers'] = consensus_outliers
            results['outlier_percentage'] = len(consensus_outliers) / len(clean_data) * 100
        
        # Generate interpretation
        results['interpretation'] = self._interpret_outlier_results(results)
        
        return results
    
    def _interpret_normality_results(self, results):
        """Generate interpretation for normality test results"""
        if not results['tests']:
            return "No normality tests could be performed on this data."
        
        interpretation = f"Normality Analysis for {results['column']}:\n"
        
        if results['is_normal']:
            interpretation += f"✅ Data appears to be normally distributed (consensus score: {results.get('consensus_score', 0):.2f})\n"
            interpretation += "• Suitable for parametric statistical tests\n"
            interpretation += "• Can use mean and standard deviation as representative statistics\n"
            interpretation += "• Linear regression assumptions likely satisfied\n"
        else:
            interpretation += f"❌ Data does not appear to be normally distributed (consensus score: {results.get('consensus_score', 0):.2f})\n"
            interpretation += "• Consider non-parametric statistical tests\n"
            interpretation += "• Use median and IQR as representative statistics\n"
            interpretation += "• May need data transformation for linear models\n"
        
        return interpretation
    
    def _interpret_correlation_results(self, results):
        """Generate interpretation for correlation analysis"""
        interpretation = f"Correlation Analysis Summary:\n"
        interpretation += f"• Analyzed {len(results['columns'])} numeric variables\n"
        interpretation += f"• Found {len(results['significant_pairs'])} significant correlations\n\n"
        
        if results['significant_pairs']:
            interpretation += "Top Correlations:\n"
            for i, pair in enumerate(results['significant_pairs'][:5]):
                interpretation += f"{i+1}. {pair['variable1']} ↔ {pair['variable2']}: "
                interpretation += f"r = {pair['pearson_r']:.3f} ({pair['strength']} {pair['direction']})\n"
        else:
            interpretation += "No significant correlations found.\n"
        
        return interpretation
    
    def _interpret_statistical_tests(self, results):
        """Generate interpretation for statistical test results"""
        interpretation = f"Statistical Tests Summary:\n"
        interpretation += f"• Performed {len(results['tests_performed'])} statistical tests\n"
        interpretation += f"• Found {len(results['significant_results'])} significant results\n\n"
        
        if results['significant_results']:
            interpretation += "Significant Findings:\n"
            for i, result in enumerate(results['significant_results'][:5]):
                interpretation += f"{i+1}. {result['interpretation']}\n"
        else:
            interpretation += "No statistically significant relationships detected.\n"
        
        return interpretation
    
    def _interpret_outlier_results(self, results):
        """Generate interpretation for outlier detection results"""
        interpretation = f"Outlier Analysis for {results['column']}:\n"
        
        if results['outlier_percentage'] > 10:
            interpretation += f"⚠️ High outlier content: {results['outlier_percentage']:.1f}% of data\n"
            interpretation += "• Consider investigating data quality\n"
            interpretation += "• May need outlier treatment before modeling\n"
        elif results['outlier_percentage'] > 5:
            interpretation += f"📊 Moderate outliers detected: {results['outlier_percentage']:.1f}% of data\n"
            interpretation += "• Normal for real-world data\n"
            interpretation += "• Monitor impact on model performance\n"
        else:
            interpretation += f"✅ Low outlier content: {results['outlier_percentage']:.1f}% of data\n"
            interpretation += "• Data appears clean\n"
            interpretation += "• Suitable for most modeling approaches\n"
        
        return interpretation
    
    def _correlation_strength(self, abs_corr):
        """Classify correlation strength"""
        if abs_corr >= 0.9:
            return "very strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very weak"


# Convenience functions for direct use
def test_normality(data, column_name, alpha=0.05):
    """Test data normality using multiple methods"""
    analyzer = AdvancedStatistics()
    return analyzer.test_normality(data, column_name, alpha)

def analyze_correlations(df, columns=None, alpha=0.05):
    """Perform advanced correlation analysis"""
    analyzer = AdvancedStatistics()
    return analyzer.advanced_correlation_analysis(df, columns, alpha)

def run_statistical_tests(df, target_col=None, alpha=0.05):
    """Run comprehensive statistical test suite"""
    analyzer = AdvancedStatistics()
    return analyzer.statistical_tests_suite(df, target_col, alpha)

def detect_outliers(data, column_name, methods=['iqr', 'zscore', 'isolation']):
    """Detect outliers using multiple methods"""
    analyzer = AdvancedStatistics()
    return analyzer.detect_outliers_advanced(data, column_name, methods)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'normal_data': np.random.normal(100, 15, 1000),
        'skewed_data': np.random.exponential(2, 1000),
        'categorical': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Test normality
    print("=== Normality Test ===")
    norm_result = test_normality(df['normal_data'], 'normal_data')
    print(norm_result['interpretation'])
    
    # Test correlations
    print("\n=== Correlation Analysis ===")
    corr_result = analyze_correlations(df)
    print(corr_result['interpretation'])
    
    # Run statistical tests
    print("\n=== Statistical Tests ===")
    stats_result = run_statistical_tests(df, target_col='target')
    print(stats_result['interpretation'])
    
    # Detect outliers
    print("\n=== Outlier Detection ===")
    outlier_result = detect_outliers(df['normal_data'], 'normal_data')
    print(outlier_result['interpretation'])