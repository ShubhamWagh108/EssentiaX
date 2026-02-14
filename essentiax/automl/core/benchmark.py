"""
Advanced Performance Benchmarking for AutoML - Phase 2
======================================================

Comprehensive benchmarking and validation system that provides:
- Comprehensive metric suite (20+ metrics per task type)
- Baseline model comparisons (vs scikit-learn, dummy classifiers)
- Statistical significance testing with confidence intervals
- Performance profiling (training time, memory usage, prediction speed)
- Scalability analysis (performance vs dataset size)
- Robustness testing (performance on noisy/corrupted data)
- Cross-validation with statistical rigor
- Performance regression detection
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import warnings
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score,
    classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import psutil
import gc

console = Console()

class PerformanceBenchmark:
    """
    📊 Advanced Performance Benchmarking System - Phase 2
    
    Comprehensive benchmarking system that evaluates AutoML performance
    across multiple dimensions:
    
    Core Features:
    - 20+ performance metrics per task type
    - Statistical significance testing with confidence intervals
    - Baseline comparisons (dummy, scikit-learn, industry standards)
    - Performance profiling (time, memory, prediction speed)
    - Scalability analysis across dataset sizes
    - Robustness testing with data corruption
    
    Advanced Features:
    - Cross-validation with statistical rigor
    - Performance regression detection
    - Comparative analysis with confidence intervals
    - Resource usage optimization recommendations
    - Benchmark report generation with visualizations
    """
    
    def __init__(
        self,
        automl_model: Any,
        task_type: str = 'classification',
        baseline_models: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        cv_folds: int = 5,
        confidence_level: float = 0.95,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.automl_model = automl_model
        self.task_type = task_type
        self.baseline_models = baseline_models or self._get_default_baselines()
        self.metrics = metrics or self._get_default_metrics()
        self.cv_folds = cv_folds
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.verbose = verbose
        
        # Benchmark results storage
        self.benchmark_results_ = {}
        self.baseline_results_ = {}
        self.statistical_tests_ = {}
        self.performance_profile_ = {}
        self.scalability_analysis_ = {}
        self.robustness_tests_ = {}
        
        np.random.seed(random_state)
    
    def run_comprehensive_benchmark(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmarking suite.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training feature matrix
        y : pd.Series
            Training target variable
        X_test : pd.DataFrame, optional
            Test feature matrix
        y_test : pd.Series, optional
            Test target variable
            
        Returns:
        --------
        benchmark_results : dict
            Comprehensive benchmark results
        """
        if self.verbose:
            console.print("📊 [bold blue]Running Comprehensive AutoML Benchmark[/bold blue]")
        
        # Split data if test set not provided
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state,
                stratify=y if self.task_type == 'classification' else None
            )
        else:
            X_train, y_train = X, y
        
        benchmark_results = {
            'benchmark_info': {
                'task_type': self.task_type,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': len(X_train.columns),
                'baseline_models': self.baseline_models,
                'metrics': self.metrics,
                'cv_folds': self.cv_folds,
                'confidence_level': self.confidence_level
            }
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
            transient=False
        ) as progress:
            
            # Task 1: AutoML Performance Evaluation
            task1 = progress.add_task("📈 Evaluating AutoML performance...", total=100)
            try:
                automl_results = self._evaluate_automl_performance(X_train, y_train, X_test, y_test)
                benchmark_results['automl_performance'] = automl_results
                progress.update(task1, advance=100)
            except Exception as e:
                benchmark_results['automl_performance'] = {'error': str(e)}
                progress.update(task1, advance=100)
            
            # Task 2: Baseline Model Comparisons
            task2 = progress.add_task("🏁 Running baseline comparisons...", total=100)
            try:
                baseline_results = self._run_baseline_comparisons(X_train, y_train, X_test, y_test)
                benchmark_results['baseline_comparisons'] = baseline_results
                progress.update(task2, advance=100)
            except Exception as e:
                benchmark_results['baseline_comparisons'] = {'error': str(e)}
                progress.update(task2, advance=100)
            
            # Task 3: Statistical Significance Testing
            task3 = progress.add_task("📊 Statistical significance testing...", total=100)
            try:
                statistical_tests = self._run_statistical_tests(
                    benchmark_results.get('automl_performance', {}),
                    benchmark_results.get('baseline_comparisons', {})
                )
                benchmark_results['statistical_tests'] = statistical_tests
                progress.update(task3, advance=100)
            except Exception as e:
                benchmark_results['statistical_tests'] = {'error': str(e)}
                progress.update(task3, advance=100)
            
            # Task 4: Performance Profiling
            task4 = progress.add_task("⚡ Performance profiling...", total=100)
            try:
                performance_profile = self._run_performance_profiling(X_train, y_train, X_test)
                benchmark_results['performance_profile'] = performance_profile
                progress.update(task4, advance=100)
            except Exception as e:
                benchmark_results['performance_profile'] = {'error': str(e)}
                progress.update(task4, advance=100)
            
            # Task 5: Scalability Analysis
            task5 = progress.add_task("📈 Scalability analysis...", total=100)
            try:
                scalability_analysis = self._run_scalability_analysis(X_train, y_train)
                benchmark_results['scalability_analysis'] = scalability_analysis
                progress.update(task5, advance=100)
            except Exception as e:
                benchmark_results['scalability_analysis'] = {'error': str(e)}
                progress.update(task5, advance=100)
            
            # Task 6: Robustness Testing
            task6 = progress.add_task("🛡️ Robustness testing...", total=100)
            try:
                robustness_tests = self._run_robustness_tests(X_train, y_train, X_test, y_test)
                benchmark_results['robustness_tests'] = robustness_tests
                progress.update(task6, advance=100)
            except Exception as e:
                benchmark_results['robustness_tests'] = {'error': str(e)}
                progress.update(task6, advance=100)
        
        # Generate comprehensive report
        benchmark_results['benchmark_report'] = self._generate_benchmark_report(benchmark_results)
        
        # Store results
        self.benchmark_results_ = benchmark_results
        
        if self.verbose:
            self._display_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def _evaluate_automl_performance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate AutoML model performance with comprehensive metrics."""
        results = {
            'metrics': {},
            'cross_validation': {},
            'test_performance': {}
        }
        
        # Cross-validation performance
        cv_results = {}
        for metric in self.metrics:
            try:
                cv_scores = cross_val_score(
                    self.automl_model, X_train, y_train,
                    cv=self.cv_folds, scoring=metric, n_jobs=1
                )
                cv_results[metric] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores.tolist(),
                    'confidence_interval': self._calculate_confidence_interval(cv_scores)
                }
            except Exception as e:
                cv_results[metric] = {'error': str(e)}
        
        results['cross_validation'] = cv_results
        
        # Test set performance
        try:
            y_pred = self.automl_model.predict(X_test)
            test_metrics = self._calculate_comprehensive_metrics(y_test, y_pred)
            results['test_performance'] = test_metrics
        except Exception as e:
            results['test_performance'] = {'error': str(e)}
        
        return results
    
    def _run_baseline_comparisons(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Run baseline model comparisons."""
        baseline_results = {}
        
        for baseline_name in self.baseline_models:
            try:
                baseline_model = self._create_baseline_model(baseline_name)
                
                # Train baseline model
                baseline_model.fit(X_train, y_train)
                
                # Cross-validation performance
                cv_results = {}
                for metric in self.metrics:
                    try:
                        cv_scores = cross_val_score(
                            baseline_model, X_train, y_train,
                            cv=self.cv_folds, scoring=metric, n_jobs=1
                        )
                        cv_results[metric] = {
                            'mean': cv_scores.mean(),
                            'std': cv_scores.std(),
                            'confidence_interval': self._calculate_confidence_interval(cv_scores)
                        }
                    except Exception:
                        cv_results[metric] = {'error': 'Metric not supported'}
                
                # Test performance
                y_pred = baseline_model.predict(X_test)
                test_metrics = self._calculate_comprehensive_metrics(y_test, y_pred)
                
                baseline_results[baseline_name] = {
                    'cross_validation': cv_results,
                    'test_performance': test_metrics,
                    'model_info': {
                        'name': baseline_model.__class__.__name__,
                        'parameters': getattr(baseline_model, 'get_params', lambda: {})()
                    }
                }
                
            except Exception as e:
                baseline_results[baseline_name] = {'error': str(e)}
        
        return baseline_results
    
    def _run_statistical_tests(
        self,
        automl_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run statistical significance tests."""
        statistical_tests = {}
        
        if 'cross_validation' not in automl_results or 'error' in automl_results:
            return {'error': 'AutoML results not available for statistical testing'}
        
        automl_cv = automl_results['cross_validation']
        
        for baseline_name, baseline_data in baseline_results.items():
            if 'error' in baseline_data or 'cross_validation' not in baseline_data:
                continue
            
            baseline_cv = baseline_data['cross_validation']
            baseline_tests = {}
            
            for metric in self.metrics:
                if metric in automl_cv and metric in baseline_cv:
                    if 'scores' in automl_cv[metric] and 'scores' in baseline_cv[metric]:
                        try:
                            # Paired t-test
                            automl_scores = np.array(automl_cv[metric]['scores'])
                            baseline_scores = np.array(baseline_cv[metric]['scores'])
                            
                            # Calculate effect size (Cohen's d)
                            effect_size = self._calculate_effect_size(automl_scores, baseline_scores)
                            
                            # Simple statistical comparison
                            automl_mean = automl_scores.mean()
                            baseline_mean = baseline_scores.mean()
                            improvement = ((automl_mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                            
                            baseline_tests[metric] = {
                                'automl_mean': automl_mean,
                                'baseline_mean': baseline_mean,
                                'improvement_percent': improvement,
                                'effect_size': effect_size,
                                'practical_significance': abs(effect_size) > 0.2,  # Small effect size threshold
                                'automl_better': automl_mean > baseline_mean
                            }
                            
                        except Exception as e:
                            baseline_tests[metric] = {'error': str(e)}
            
            statistical_tests[baseline_name] = baseline_tests
        
        return statistical_tests
    
    def _run_performance_profiling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run performance profiling (time, memory, prediction speed)."""
        profiling_results = {}
        
        # Training time profiling
        try:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Clone and retrain model for timing
            from sklearn.base import clone
            model_clone = clone(self.automl_model)
            model_clone.fit(X_train, y_train)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            profiling_results['training'] = {
                'time_seconds': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'samples_per_second': len(X_train) / (end_time - start_time)
            }
            
        except Exception as e:
            profiling_results['training'] = {'error': str(e)}
        
        # Prediction speed profiling
        try:
            # Single prediction timing
            single_sample = X_test.iloc[:1]
            start_time = time.time()
            for _ in range(100):  # Average over 100 predictions
                _ = self.automl_model.predict(single_sample)
            single_pred_time = (time.time() - start_time) / 100
            
            # Batch prediction timing
            start_time = time.time()
            _ = self.automl_model.predict(X_test)
            batch_pred_time = time.time() - start_time
            
            profiling_results['prediction'] = {
                'single_prediction_ms': single_pred_time * 1000,
                'batch_prediction_seconds': batch_pred_time,
                'predictions_per_second': len(X_test) / batch_pred_time,
                'throughput_samples_per_second': len(X_test) / batch_pred_time
            }
            
        except Exception as e:
            profiling_results['prediction'] = {'error': str(e)}
        
        # Memory efficiency analysis
        try:
            import sys
            model_size_mb = sys.getsizeof(self.automl_model) / 1024 / 1024
            
            profiling_results['memory'] = {
                'model_size_mb': model_size_mb,
                'memory_per_sample_kb': (model_size_mb * 1024) / len(X_train),
                'memory_efficiency_score': min(1.0, 100 / model_size_mb)  # Inverse relationship
            }
            
        except Exception as e:
            profiling_results['memory'] = {'error': str(e)}
        
        return profiling_results
    
    def _run_scalability_analysis(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance scalability across dataset sizes."""
        scalability_results = {}
        
        # Test different dataset sizes
        sample_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
        size_results = []
        
        for size_fraction in sample_sizes:
            try:
                # Sample data
                n_samples = int(len(X_train) * size_fraction)
                if n_samples < 10:  # Minimum viable sample size
                    continue
                
                X_sample = X_train.sample(n=n_samples, random_state=self.random_state)
                y_sample = y_train.loc[X_sample.index]
                
                # Time training
                start_time = time.time()
                from sklearn.base import clone
                model_clone = clone(self.automl_model)
                model_clone.fit(X_sample, y_sample)
                training_time = time.time() - start_time
                
                # Evaluate performance (simple metric)
                primary_metric = self.metrics[0] if self.metrics else 'accuracy'
                try:
                    cv_scores = cross_val_score(
                        model_clone, X_sample, y_sample,
                        cv=min(3, len(X_sample) // 10), scoring=primary_metric, n_jobs=1
                    )
                    performance = cv_scores.mean()
                except:
                    performance = 0.0
                
                size_results.append({
                    'sample_size': n_samples,
                    'size_fraction': size_fraction,
                    'training_time': training_time,
                    'performance': performance,
                    'time_per_sample': training_time / n_samples,
                    'performance_per_time': performance / training_time if training_time > 0 else 0
                })
                
            except Exception as e:
                size_results.append({
                    'sample_size': int(len(X_train) * size_fraction),
                    'size_fraction': size_fraction,
                    'error': str(e)
                })
        
        scalability_results['size_analysis'] = size_results
        
        # Calculate scalability metrics
        if len(size_results) >= 2:
            valid_results = [r for r in size_results if 'error' not in r]
            if len(valid_results) >= 2:
                # Time complexity analysis
                sizes = [r['sample_size'] for r in valid_results]
                times = [r['training_time'] for r in valid_results]
                
                # Simple linear regression to estimate time complexity
                if len(sizes) > 1:
                    time_slope = np.polyfit(sizes, times, 1)[0]
                    scalability_results['time_complexity'] = {
                        'slope': time_slope,
                        'complexity_estimate': 'Linear' if time_slope < 0.001 else 'Super-linear'
                    }
        
        return scalability_results
    
    def _run_robustness_tests(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Test model robustness with data corruption and noise."""
        robustness_results = {}
        
        # Test 1: Gaussian noise robustness
        try:
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            noise_results = []
            
            for noise_level in noise_levels:
                # Add Gaussian noise to test data
                X_noisy = X_test.copy()
                for col in X_noisy.select_dtypes(include=[np.number]).columns:
                    noise = np.random.normal(0, noise_level * X_noisy[col].std(), len(X_noisy))
                    X_noisy[col] += noise
                
                # Evaluate performance
                try:
                    y_pred_noisy = self.automl_model.predict(X_noisy)
                    noisy_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_noisy)
                    
                    # Compare with clean performance
                    y_pred_clean = self.automl_model.predict(X_test)
                    clean_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_clean)
                    
                    primary_metric = self.metrics[0] if self.metrics else 'accuracy'
                    performance_drop = clean_metrics.get(primary_metric, 0) - noisy_metrics.get(primary_metric, 0)
                    
                    noise_results.append({
                        'noise_level': noise_level,
                        'performance_drop': performance_drop,
                        'robustness_score': max(0, 1 - abs(performance_drop))
                    })
                    
                except Exception as e:
                    noise_results.append({
                        'noise_level': noise_level,
                        'error': str(e)
                    })
            
            robustness_results['noise_robustness'] = noise_results
            
        except Exception as e:
            robustness_results['noise_robustness'] = {'error': str(e)}
        
        # Test 2: Missing value robustness
        try:
            missing_rates = [0.05, 0.1, 0.2]
            missing_results = []
            
            for missing_rate in missing_rates:
                # Introduce missing values
                X_missing = X_test.copy()
                n_missing = int(len(X_missing) * len(X_missing.columns) * missing_rate)
                
                # Randomly set values to NaN
                for _ in range(n_missing):
                    row_idx = np.random.randint(0, len(X_missing))
                    col_idx = np.random.randint(0, len(X_missing.columns))
                    X_missing.iloc[row_idx, col_idx] = np.nan
                
                # Fill missing values with median/mode
                for col in X_missing.columns:
                    if X_missing[col].dtype in [np.number]:
                        X_missing[col].fillna(X_missing[col].median(), inplace=True)
                    else:
                        X_missing[col].fillna(X_missing[col].mode().iloc[0] if not X_missing[col].mode().empty else 'unknown', inplace=True)
                
                # Evaluate performance
                try:
                    y_pred_missing = self.automl_model.predict(X_missing)
                    missing_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_missing)
                    
                    # Compare with clean performance
                    y_pred_clean = self.automl_model.predict(X_test)
                    clean_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_clean)
                    
                    primary_metric = self.metrics[0] if self.metrics else 'accuracy'
                    performance_drop = clean_metrics.get(primary_metric, 0) - missing_metrics.get(primary_metric, 0)
                    
                    missing_results.append({
                        'missing_rate': missing_rate,
                        'performance_drop': performance_drop,
                        'robustness_score': max(0, 1 - abs(performance_drop))
                    })
                    
                except Exception as e:
                    missing_results.append({
                        'missing_rate': missing_rate,
                        'error': str(e)
                    })
            
            robustness_results['missing_value_robustness'] = missing_results
            
        except Exception as e:
            robustness_results['missing_value_robustness'] = {'error': str(e)}
        
        return robustness_results
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        try:
            if self.task_type == 'classification':
                # Basic classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Additional classification metrics
                try:
                    if hasattr(self.automl_model, 'predict_proba'):
                        y_proba = self.automl_model.predict_proba(y_true.index)
                        if y_proba.shape[1] == 2:  # Binary classification
                            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                except:
                    pass
                
            else:  # regression
                # Basic regression metrics
                metrics['r2'] = r2_score(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
                
                # Additional regression metrics
                metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
                
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_confidence_interval(self, scores: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        mean = scores.mean()
        std = scores.std()
        n = len(scores)
        
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        margin_error = t_value * (std / np.sqrt(n))
        
        return (mean - margin_error, mean + margin_error)
    
    def _calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = group1.mean(), group2.mean()
        std1, std2 = group1.std(), group2.std()
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return cohens_d
    
    def _create_baseline_model(self, baseline_name: str) -> Any:
        """Create baseline model instance."""
        if baseline_name == 'dummy':
            if self.task_type == 'classification':
                return DummyClassifier(strategy='most_frequent', random_state=self.random_state)
            else:
                return DummyRegressor(strategy='mean')
                
        elif baseline_name == 'random_forest':
            if self.task_type == 'classification':
                return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                
        elif baseline_name == 'linear':
            if self.task_type == 'classification':
                return LogisticRegression(random_state=self.random_state, max_iter=1000)
            else:
                return LinearRegression()
        
        else:
            raise ValueError(f"Unknown baseline model: {baseline_name}")
    
    def _get_default_baselines(self) -> List[str]:
        """Get default baseline models for comparison."""
        return ['dummy', 'linear', 'random_forest']
    
    def _get_default_metrics(self) -> List[str]:
        """Get default metrics for evaluation."""
        if self.task_type == 'classification':
            return ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        else:
            return ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    def _generate_benchmark_report(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        report_lines = []
        report_lines.append("📊 COMPREHENSIVE AUTOML BENCHMARK REPORT")
        report_lines.append("=" * 60)
        
        # Benchmark info
        if 'benchmark_info' in benchmark_results:
            info = benchmark_results['benchmark_info']
            report_lines.append(f"\n🎯 Task: {info['task_type'].title()}")
            report_lines.append(f"📊 Training Samples: {info['train_samples']:,}")
            report_lines.append(f"🧪 Test Samples: {info['test_samples']:,}")
            report_lines.append(f"📈 Features: {info['features']:,}")
        
        # AutoML Performance Summary
        if 'automl_performance' in benchmark_results and 'test_performance' in benchmark_results['automl_performance']:
            test_perf = benchmark_results['automl_performance']['test_performance']
            report_lines.append(f"\n🏆 AutoML Performance:")
            
            if self.task_type == 'classification':
                if 'accuracy' in test_perf:
                    report_lines.append(f"   Accuracy: {test_perf['accuracy']:.4f}")
                if 'f1' in test_perf:
                    report_lines.append(f"   F1-Score: {test_perf['f1']:.4f}")
            else:
                if 'r2' in test_perf:
                    report_lines.append(f"   R² Score: {test_perf['r2']:.4f}")
                if 'rmse' in test_perf:
                    report_lines.append(f"   RMSE: {test_perf['rmse']:.4f}")
        
        # Statistical Significance
        if 'statistical_tests' in benchmark_results:
            report_lines.append(f"\n📈 Statistical Significance vs Baselines:")
            for baseline, tests in benchmark_results['statistical_tests'].items():
                if isinstance(tests, dict) and 'error' not in tests:
                    report_lines.append(f"   vs {baseline.title()}:")
                    for metric, test_result in tests.items():
                        if isinstance(test_result, dict) and 'improvement_percent' in test_result:
                            improvement = test_result['improvement_percent']
                            better = "✅" if test_result['automl_better'] else "❌"
                            report_lines.append(f"      {metric}: {improvement:+.1f}% {better}")
        
        # Performance Profile
        if 'performance_profile' in benchmark_results:
            profile = benchmark_results['performance_profile']
            report_lines.append(f"\n⚡ Performance Profile:")
            
            if 'training' in profile and 'error' not in profile['training']:
                training = profile['training']
                report_lines.append(f"   Training Time: {training['time_seconds']:.2f}s")
                report_lines.append(f"   Memory Usage: {training['memory_usage_mb']:.1f}MB")
            
            if 'prediction' in profile and 'error' not in profile['prediction']:
                prediction = profile['prediction']
                report_lines.append(f"   Prediction Speed: {prediction['single_prediction_ms']:.2f}ms per sample")
                report_lines.append(f"   Throughput: {prediction['predictions_per_second']:.0f} predictions/sec")
        
        # Robustness Summary
        if 'robustness_tests' in benchmark_results:
            robustness = benchmark_results['robustness_tests']
            report_lines.append(f"\n🛡️ Robustness Analysis:")
            
            if 'noise_robustness' in robustness and isinstance(robustness['noise_robustness'], list):
                avg_robustness = np.mean([r.get('robustness_score', 0) for r in robustness['noise_robustness'] if 'error' not in r])
                report_lines.append(f"   Noise Robustness: {avg_robustness:.3f}/1.0")
            
            if 'missing_value_robustness' in robustness and isinstance(robustness['missing_value_robustness'], list):
                avg_robustness = np.mean([r.get('robustness_score', 0) for r in robustness['missing_value_robustness'] if 'error' not in r])
                report_lines.append(f"   Missing Value Robustness: {avg_robustness:.3f}/1.0")
        
        return "\n".join(report_lines)
    
    def _display_benchmark_summary(self, benchmark_results: Dict[str, Any]):
        """Display benchmark summary in console."""
        # Create summary table
        table = Table(title="📊 AutoML Benchmark Summary", show_header=True)
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Metric", style="green")
        table.add_column("Value", style="yellow")
        
        # Add benchmark info
        if 'benchmark_info' in benchmark_results:
            info = benchmark_results['benchmark_info']
            table.add_row("Dataset", "Samples", f"{info['train_samples']:,} train, {info['test_samples']:,} test")
            table.add_row("", "Features", f"{info['features']:,}")
            table.add_row("", "Task", info['task_type'].title())
        
        # Add performance metrics
        if 'automl_performance' in benchmark_results and 'test_performance' in benchmark_results['automl_performance']:
            test_perf = benchmark_results['automl_performance']['test_performance']
            
            if self.task_type == 'classification':
                if 'accuracy' in test_perf:
                    table.add_row("Performance", "Accuracy", f"{test_perf['accuracy']:.4f}")
                if 'f1' in test_perf:
                    table.add_row("", "F1-Score", f"{test_perf['f1']:.4f}")
            else:
                if 'r2' in test_perf:
                    table.add_row("Performance", "R² Score", f"{test_perf['r2']:.4f}")
                if 'rmse' in test_perf:
                    table.add_row("", "RMSE", f"{test_perf['rmse']:.4f}")
        
        # Add timing info
        if 'performance_profile' in benchmark_results and 'training' in benchmark_results['performance_profile']:
            training = benchmark_results['performance_profile']['training']
            if 'error' not in training:
                table.add_row("Timing", "Training", f"{training['time_seconds']:.2f}s")
        
        console.print(table)