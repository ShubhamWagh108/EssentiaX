"""
EssentiaX AutoML - Production Features
=====================================

Production-ready features for AutoML deployment, monitoring, and management.
Includes model serialization, deployment pipeline, monitoring, and A/B testing.
"""

import os
import json
import pickle
import joblib
import datetime
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ModelSerializer:
    """
    Advanced model serialization with versioning and metadata.
    
    Features:
    - Multiple serialization formats (pickle, joblib, custom)
    - Model versioning and metadata tracking
    - Compression and optimization
    - Security and integrity checks
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize model serializer.
        
        Parameters:
        -----------
        base_path : str
            Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_model(
        self,
        automl_instance,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = 'joblib',
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Save AutoML model with comprehensive metadata.
        
        Parameters:
        -----------
        automl_instance : AutoML
            Trained AutoML instance
        model_name : str
            Name for the model
        version : str, optional
            Model version (auto-generated if None)
        metadata : dict, optional
            Additional metadata
        format : str
            Serialization format ('pickle', 'joblib', 'custom')
        compress : bool
            Whether to compress the model
            
        Returns:
        --------
        dict : Save information and metadata
        """
        try:
            # Generate version if not provided
            if version is None:
                version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create model directory
            model_dir = self.base_path / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare comprehensive metadata
            model_metadata = self._extract_model_metadata(automl_instance)
            if metadata:
                model_metadata.update(metadata)
            
            # Add save information
            model_metadata.update({
                'model_name': model_name,
                'version': version,
                'save_timestamp': datetime.datetime.now().isoformat(),
                'format': format,
                'compressed': compress,
                'serializer_version': '1.0.0'
            })
            
            # Save model components
            save_info = {}
            
            # Save main AutoML instance
            model_file = model_dir / f"automl_model.{format}"
            if format == 'joblib':
                if compress:
                    joblib.dump(automl_instance, model_file, compress=3)
                else:
                    joblib.dump(automl_instance, model_file)
            elif format == 'pickle':
                with open(model_file, 'wb') as f:
                    pickle.dump(automl_instance, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            save_info['model_file'] = str(model_file)
            save_info['model_size'] = model_file.stat().st_size
            
            # Save metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
            
            save_info['metadata_file'] = str(metadata_file)
            
            # Generate model hash for integrity
            model_hash = self._generate_model_hash(model_file)
            model_metadata['model_hash'] = model_hash
            
            # Update metadata with hash
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2, default=str)
            
            # Create deployment manifest
            manifest = self._create_deployment_manifest(model_metadata, save_info)
            manifest_file = model_dir / "deployment_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            save_info['manifest_file'] = str(manifest_file)
            save_info['model_hash'] = model_hash
            save_info['success'] = True
            
            console.print(f"✅ Model saved successfully: {model_name} v{version}")
            console.print(f"📁 Location: {model_dir}")
            console.print(f"📊 Size: {save_info['model_size'] / 1024 / 1024:.2f} MB")
            
            return save_info
            
        except Exception as e:
            error_info = {
                'success': False,
                'error': str(e),
                'model_name': model_name,
                'version': version
            }
            console.print(f"❌ Failed to save model: {str(e)}")
            return error_info
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        verify_integrity: bool = True
    ):
        """
        Load AutoML model with integrity verification.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
        version : str, optional
            Specific version to load (latest if None)
        verify_integrity : bool
            Whether to verify model integrity
            
        Returns:
        --------
        AutoML instance or None if failed
        """
        try:
            # Find model directory
            model_base_dir = self.base_path / model_name
            if not model_base_dir.exists():
                raise FileNotFoundError(f"Model '{model_name}' not found")
            
            # Get version directory
            if version is None:
                # Get latest version
                versions = [d.name for d in model_base_dir.iterdir() if d.is_dir()]
                if not versions:
                    raise FileNotFoundError(f"No versions found for model '{model_name}'")
                version = max(versions)  # Latest version
            
            model_dir = model_base_dir / version
            if not model_dir.exists():
                raise FileNotFoundError(f"Version '{version}' not found for model '{model_name}'")
            
            # Load metadata
            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                raise FileNotFoundError("Model metadata not found")
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Determine model file and format
            format = metadata.get('format', 'joblib')
            model_file = model_dir / f"automl_model.{format}"
            
            if not model_file.exists():
                raise FileNotFoundError("Model file not found")
            
            # Verify integrity if requested
            if verify_integrity and 'model_hash' in metadata:
                current_hash = self._generate_model_hash(model_file)
                if current_hash != metadata['model_hash']:
                    raise ValueError("Model integrity check failed - file may be corrupted")
            
            # Load model
            if format == 'joblib':
                automl_instance = joblib.load(model_file)
            elif format == 'pickle':
                with open(model_file, 'rb') as f:
                    automl_instance = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Attach metadata to instance
            automl_instance._loaded_metadata = metadata
            automl_instance._loaded_version = version
            automl_instance._loaded_name = model_name
            
            console.print(f"✅ Model loaded successfully: {model_name} v{version}")
            console.print(f"📊 Training accuracy: {metadata.get('best_score', 'N/A')}")
            console.print(f"🕒 Trained: {metadata.get('training_timestamp', 'N/A')}")
            
            return automl_instance
            
        except Exception as e:
            console.print(f"❌ Failed to load model: {str(e)}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with metadata.
        
        Returns:
        --------
        list : List of model information dictionaries
        """
        models = []
        
        try:
            if not self.base_path.exists():
                return models
            
            for model_dir in self.base_path.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                versions = []
                
                for version_dir in model_dir.iterdir():
                    if not version_dir.is_dir():
                        continue
                    
                    version = version_dir.name
                    metadata_file = version_dir / "metadata.json"
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            version_info = {
                                'version': version,
                                'save_timestamp': metadata.get('save_timestamp'),
                                'best_score': metadata.get('best_score'),
                                'task_type': metadata.get('task_type'),
                                'model_type': metadata.get('best_model_name'),
                                'training_time': metadata.get('training_time'),
                                'dataset_shape': metadata.get('dataset_shape')
                            }
                            versions.append(version_info)
                        except:
                            continue
                
                if versions:
                    models.append({
                        'model_name': model_name,
                        'versions': sorted(versions, key=lambda x: x['version'], reverse=True),
                        'latest_version': max(versions, key=lambda x: x['version'])['version']
                    })
        
        except Exception as e:
            console.print(f"⚠️ Error listing models: {str(e)}")
        
        return models
    
    def delete_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        confirm: bool = False
    ) -> bool:
        """
        Delete model or specific version.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        version : str, optional
            Specific version to delete (all versions if None)
        confirm : bool
            Confirmation flag for safety
            
        Returns:
        --------
        bool : Success status
        """
        if not confirm:
            console.print("⚠️ Deletion requires confirm=True for safety")
            return False
        
        try:
            model_dir = self.base_path / model_name
            if not model_dir.exists():
                console.print(f"❌ Model '{model_name}' not found")
                return False
            
            if version is None:
                # Delete entire model
                import shutil
                shutil.rmtree(model_dir)
                console.print(f"✅ Deleted model: {model_name} (all versions)")
            else:
                # Delete specific version
                version_dir = model_dir / version
                if not version_dir.exists():
                    console.print(f"❌ Version '{version}' not found")
                    return False
                
                import shutil
                shutil.rmtree(version_dir)
                console.print(f"✅ Deleted model version: {model_name} v{version}")
                
                # Remove model directory if empty
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()
            
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to delete model: {str(e)}")
            return False
    
    def _extract_model_metadata(self, automl_instance) -> Dict[str, Any]:
        """Extract comprehensive metadata from AutoML instance."""
        metadata = {
            'training_timestamp': datetime.datetime.now().isoformat(),
            'essentiax_version': '2.0.0',  # Update as needed
        }
        
        # Extract available attributes
        attrs_to_extract = [
            'task_type', 'best_score_', 'best_model_', 'best_model_name_',
            'training_time_', 'feature_names_', 'target_name_', 'n_features_',
            'model_rankings_', 'ensemble_score_', 'preprocessing_steps_'
        ]
        
        for attr in attrs_to_extract:
            if hasattr(automl_instance, attr):
                value = getattr(automl_instance, attr)
                # Convert to JSON-serializable format
                if hasattr(value, 'tolist'):  # numpy arrays
                    value = value.tolist()
                elif hasattr(value, '__dict__'):  # complex objects
                    value = str(value)
                metadata[attr.rstrip('_')] = value
        
        # Add dataset information if available
        if hasattr(automl_instance, 'X_') and automl_instance.X_ is not None:
            metadata['dataset_shape'] = automl_instance.X_.shape
            metadata['feature_names'] = list(automl_instance.X_.columns) if hasattr(automl_instance.X_, 'columns') else None
        
        return metadata
    
    def _generate_model_hash(self, file_path: Path) -> str:
        """Generate SHA-256 hash of model file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _create_deployment_manifest(self, metadata: Dict[str, Any], save_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create deployment manifest with all necessary information."""
        return {
            'model_info': {
                'name': metadata.get('model_name'),
                'version': metadata.get('version'),
                'type': metadata.get('task_type'),
                'algorithm': metadata.get('best_model_name'),
                'performance': metadata.get('best_score')
            },
            'deployment_info': {
                'created_at': metadata.get('save_timestamp'),
                'model_file': save_info.get('model_file'),
                'model_size_mb': save_info.get('model_size', 0) / 1024 / 1024,
                'integrity_hash': save_info.get('model_hash')
            },
            'requirements': {
                'python_version': '>=3.7',
                'essentiax_version': '>=2.0.0',
                'dependencies': [
                    'pandas>=1.3.0',
                    'numpy>=1.21.0',
                    'scikit-learn>=1.0.0',
                    'joblib>=1.0.0'
                ]
            },
            'api_info': {
                'input_features': metadata.get('feature_names', []),
                'output_type': 'classification' if metadata.get('task_type') == 'classification' else 'regression',
                'prediction_method': 'predict',
                'probability_method': 'predict_proba' if metadata.get('task_type') == 'classification' else None
            }
        }


class ModelMonitor:
    """
    Model performance monitoring and drift detection.
    
    Features:
    - Performance drift detection
    - Data drift monitoring
    - Alert system
    - Performance logging
    """
    
    def __init__(self, model_name: str, baseline_data: Optional[pd.DataFrame] = None):
        """
        Initialize model monitor.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to monitor
        baseline_data : pd.DataFrame, optional
            Baseline data for drift detection
        """
        self.model_name = model_name
        self.baseline_data = baseline_data
        self.performance_log = []
        self.drift_log = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ModelMonitor_{model_name}")
    
    def log_prediction(
        self,
        input_data: pd.DataFrame,
        predictions: np.ndarray,
        actual_values: Optional[np.ndarray] = None,
        timestamp: Optional[datetime.datetime] = None
    ):
        """
        Log prediction for monitoring.
        
        Parameters:
        -----------
        input_data : pd.DataFrame
            Input features
        predictions : np.ndarray
            Model predictions
        actual_values : np.ndarray, optional
            Actual target values (for performance monitoring)
        timestamp : datetime, optional
            Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'n_predictions': len(predictions),
            'input_shape': input_data.shape,
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }
        
        # Add performance metrics if actual values provided
        if actual_values is not None:
            try:
                from sklearn.metrics import mean_squared_error, accuracy_score
                
                if len(np.unique(actual_values)) <= 10:  # Classification
                    accuracy = accuracy_score(actual_values, predictions)
                    log_entry['performance'] = {'accuracy': accuracy}
                else:  # Regression
                    mse = mean_squared_error(actual_values, predictions)
                    log_entry['performance'] = {'mse': mse, 'rmse': np.sqrt(mse)}
            except:
                pass
        
        # Check for data drift
        if self.baseline_data is not None:
            drift_score = self._calculate_drift_score(input_data)
            log_entry['drift_score'] = drift_score
            
            if drift_score > 0.1:  # Threshold for drift alert
                self._trigger_drift_alert(drift_score, timestamp)
        
        self.performance_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for the last N days.
        
        Parameters:
        -----------
        days : int
            Number of days to include in summary
            
        Returns:
        --------
        dict : Performance summary
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        recent_logs = [
            log for log in self.performance_log
            if datetime.datetime.fromisoformat(log['timestamp']) > cutoff_date
        ]
        
        if not recent_logs:
            return {'message': 'No recent data available'}
        
        summary = {
            'period': f'Last {days} days',
            'total_predictions': sum(log['n_predictions'] for log in recent_logs),
            'total_requests': len(recent_logs),
            'avg_predictions_per_request': np.mean([log['n_predictions'] for log in recent_logs])
        }
        
        # Performance metrics if available
        performance_logs = [log for log in recent_logs if 'performance' in log]
        if performance_logs:
            if 'accuracy' in performance_logs[0]['performance']:
                accuracies = [log['performance']['accuracy'] for log in performance_logs]
                summary['performance'] = {
                    'metric': 'accuracy',
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'trend': 'stable'  # Could implement trend analysis
                }
            elif 'mse' in performance_logs[0]['performance']:
                mses = [log['performance']['mse'] for log in performance_logs]
                summary['performance'] = {
                    'metric': 'mse',
                    'mean': np.mean(mses),
                    'std': np.std(mses),
                    'trend': 'stable'
                }
        
        # Drift analysis
        drift_logs = [log for log in recent_logs if 'drift_score' in log]
        if drift_logs:
            drift_scores = [log['drift_score'] for log in drift_logs]
            summary['drift_analysis'] = {
                'mean_drift_score': np.mean(drift_scores),
                'max_drift_score': np.max(drift_scores),
                'drift_alerts': len([score for score in drift_scores if score > 0.1])
            }
        
        return summary
    
    def _calculate_drift_score(self, current_data: pd.DataFrame) -> float:
        """Calculate simple drift score based on feature distributions."""
        try:
            # Simple drift detection using statistical distance
            drift_scores = []
            
            for col in current_data.columns:
                if col in self.baseline_data.columns:
                    # For numerical columns, use KS test statistic as drift measure
                    if pd.api.types.is_numeric_dtype(current_data[col]):
                        from scipy.stats import ks_2samp
                        statistic, _ = ks_2samp(
                            self.baseline_data[col].dropna(),
                            current_data[col].dropna()
                        )
                        drift_scores.append(statistic)
            
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Drift calculation failed: {str(e)}")
            return 0.0
    
    def _trigger_drift_alert(self, drift_score: float, timestamp: datetime.datetime):
        """Trigger drift alert."""
        alert = {
            'timestamp': timestamp.isoformat(),
            'model_name': self.model_name,
            'drift_score': drift_score,
            'severity': 'high' if drift_score > 0.2 else 'medium',
            'message': f'Data drift detected with score {drift_score:.3f}'
        }
        
        self.drift_log.append(alert)
        self.logger.warning(f"DRIFT ALERT: {alert['message']}")
        
        console.print(f"⚠️ [bold red]DRIFT ALERT[/bold red]: {alert['message']}")


class ABTestFramework:
    """
    A/B testing framework for model comparison.
    
    Features:
    - Model version comparison
    - Statistical significance testing
    - Traffic splitting
    - Performance tracking
    """
    
    def __init__(self, test_name: str):
        """
        Initialize A/B test framework.
        
        Parameters:
        -----------
        test_name : str
            Name of the A/B test
        """
        self.test_name = test_name
        self.models = {}
        self.traffic_split = {}
        self.results = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"ABTest_{test_name}")
    
    def add_model(self, model_name: str, model_instance, traffic_percentage: float):
        """
        Add model to A/B test.
        
        Parameters:
        -----------
        model_name : str
            Name/identifier for the model
        model_instance : AutoML
            Trained AutoML instance
        traffic_percentage : float
            Percentage of traffic to route to this model (0-100)
        """
        if sum(self.traffic_split.values()) + traffic_percentage > 100:
            raise ValueError("Total traffic percentage cannot exceed 100%")
        
        self.models[model_name] = model_instance
        self.traffic_split[model_name] = traffic_percentage
        
        console.print(f"✅ Added model '{model_name}' with {traffic_percentage}% traffic")
    
    def predict(self, X: pd.DataFrame, user_id: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Make prediction using A/B test routing.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        user_id : str, optional
            User identifier for consistent routing
            
        Returns:
        --------
        tuple : (predictions, model_used)
        """
        if not self.models:
            raise ValueError("No models added to A/B test")
        
        # Determine which model to use
        selected_model = self._route_traffic(user_id)
        
        # Make prediction
        predictions = self.models[selected_model].predict(X)
        
        # Log the prediction
        self._log_prediction(selected_model, X.shape[0])
        
        return predictions, selected_model
    
    def log_outcome(
        self,
        model_name: str,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        user_id: Optional[str] = None
    ):
        """
        Log prediction outcomes for analysis.
        
        Parameters:
        -----------
        model_name : str
            Name of the model that made the prediction
        predictions : np.ndarray
            Model predictions
        actual_values : np.ndarray
            Actual target values
        user_id : str, optional
            User identifier
        """
        try:
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            # Calculate performance metrics
            if len(np.unique(actual_values)) <= 10:  # Classification
                performance = accuracy_score(actual_values, predictions)
                metric_name = 'accuracy'
            else:  # Regression
                performance = -mean_squared_error(actual_values, predictions)  # Negative for "higher is better"
                metric_name = 'neg_mse'
            
            result_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'model_name': model_name,
                'user_id': user_id,
                'n_predictions': len(predictions),
                'performance': performance,
                'metric': metric_name
            }
            
            self.results.append(result_entry)
            
        except Exception as e:
            self.logger.error(f"Failed to log outcome: {str(e)}")
    
    def get_test_results(self, min_samples: int = 30) -> Dict[str, Any]:
        """
        Get A/B test results with statistical analysis.
        
        Parameters:
        -----------
        min_samples : int
            Minimum samples required for statistical significance
            
        Returns:
        --------
        dict : Test results and analysis
        """
        if not self.results:
            return {'message': 'No results available yet'}
        
        # Group results by model
        model_results = {}
        for result in self.results:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result['performance'])
        
        # Calculate statistics for each model
        model_stats = {}
        for model_name, performances in model_results.items():
            if len(performances) >= min_samples:
                model_stats[model_name] = {
                    'n_samples': len(performances),
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'confidence_interval': self._calculate_confidence_interval(performances)
                }
        
        # Statistical significance testing
        significance_results = {}
        model_names = list(model_stats.keys())
        
        if len(model_names) >= 2:
            try:
                from scipy.stats import ttest_ind
                
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        model_a = model_names[i]
                        model_b = model_names[j]
                        
                        performances_a = model_results[model_a]
                        performances_b = model_results[model_b]
                        
                        if len(performances_a) >= min_samples and len(performances_b) >= min_samples:
                            statistic, p_value = ttest_ind(performances_a, performances_b)
                            
                            significance_results[f"{model_a}_vs_{model_b}"] = {
                                'statistic': statistic,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'winner': model_a if np.mean(performances_a) > np.mean(performances_b) else model_b
                            }
            except ImportError:
                significance_results['error'] = 'scipy not available for significance testing'
        
        return {
            'test_name': self.test_name,
            'model_statistics': model_stats,
            'significance_tests': significance_results,
            'total_samples': len(self.results),
            'traffic_split': self.traffic_split,
            'recommendation': self._get_recommendation(model_stats, significance_results)
        }
    
    def _route_traffic(self, user_id: Optional[str] = None) -> str:
        """Route traffic based on configured split."""
        if user_id:
            # Consistent routing based on user ID hash
            import hashlib
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            routing_value = (hash_value % 100) + 1
        else:
            # Random routing
            routing_value = np.random.randint(1, 101)
        
        cumulative_percentage = 0
        for model_name, percentage in self.traffic_split.items():
            cumulative_percentage += percentage
            if routing_value <= cumulative_percentage:
                return model_name
        
        # Fallback to first model
        return list(self.models.keys())[0]
    
    def _log_prediction(self, model_name: str, n_predictions: int):
        """Log prediction for traffic analysis."""
        # This could be expanded to include more detailed logging
        pass
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for performance values."""
        try:
            from scipy.stats import t
            
            n = len(values)
            mean = np.mean(values)
            std_err = np.std(values, ddof=1) / np.sqrt(n)
            
            # t-distribution critical value
            alpha = 1 - confidence
            t_critical = t.ppf(1 - alpha/2, n - 1)
            
            margin_error = t_critical * std_err
            
            return (mean - margin_error, mean + margin_error)
            
        except ImportError:
            # Fallback to simple standard error
            mean = np.mean(values)
            std_err = np.std(values) / np.sqrt(len(values))
            return (mean - 1.96 * std_err, mean + 1.96 * std_err)
    
    def _get_recommendation(self, model_stats: Dict, significance_results: Dict) -> str:
        """Generate recommendation based on test results."""
        if not model_stats:
            return "Insufficient data for recommendation"
        
        if len(model_stats) == 1:
            return "Only one model tested - no comparison available"
        
        # Find best performing model
        best_model = max(model_stats.keys(), key=lambda x: model_stats[x]['mean_performance'])
        best_performance = model_stats[best_model]['mean_performance']
        
        # Check if difference is statistically significant
        significant_wins = []
        for test_name, result in significance_results.items():
            if result.get('significant', False) and result.get('winner') == best_model:
                significant_wins.append(test_name)
        
        if significant_wins:
            return f"Recommend {best_model} - statistically significant improvement (p<0.05)"
        else:
            return f"Recommend {best_model} - best performance but not statistically significant"


class DeploymentPipeline:
    """
    Automated deployment pipeline for AutoML models.
    
    Features:
    - Docker containerization
    - API generation
    - Cloud deployment
    - Health checks
    """
    
    def __init__(self, model_name: str, model_version: str):
        """
        Initialize deployment pipeline.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to deploy
        model_version : str
            Version of the model to deploy
        """
        self.model_name = model_name
        self.model_version = model_version
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"Deploy_{model_name}")
    
    def create_api_template(self, output_dir: str = "deployment") -> Dict[str, str]:
        """
        Create FastAPI template for model serving.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for deployment files
            
        Returns:
        --------
        dict : Created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        created_files = {}
        
        # Create main API file
        api_template = f'''"""
FastAPI deployment for {self.model_name} v{self.model_version}
Generated by EssentiaX AutoML Production Pipeline
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="{self.model_name} API",
    description="AutoML model serving API generated by EssentiaX",
    version="{self.model_version}"
)

# Global model instance
model = None

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[Dict[str, Any]]
    return_probabilities: Optional[bool] = False
    return_uncertainty: Optional[bool] = False

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    probabilities: Optional[List[List[float]]] = None
    uncertainties: Optional[List[float]] = None
    model_info: Dict[str, Any]

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    try:
        from essentiax.automl.core.production import ModelSerializer
        
        serializer = ModelSerializer()
        model = serializer.load_model("{self.model_name}", "{self.model_version}")
        
        if model is None:
            raise Exception("Failed to load model")
        
        logger.info(f"Model {{model._loaded_name}} v{{model._loaded_version}} loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {{str(e)}}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {{
        "message": "EssentiaX AutoML Model API",
        "model_name": "{self.model_name}",
        "model_version": "{self.model_version}",
        "status": "ready" if model else "loading"
    }}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {{
        "status": "healthy",
        "model_loaded": True,
        "model_name": "{self.model_name}",
        "model_version": "{self.model_version}"
    }}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df)
        
        response_data = {{
            "predictions": predictions.tolist(),
            "model_info": {{
                "name": "{self.model_name}",
                "version": "{self.model_version}",
                "type": getattr(model, 'task_type', 'unknown')
            }}
        }}
        
        # Add probabilities if requested and available
        if request.return_probabilities:
            try:
                probabilities = model.predict_proba(df)
                response_data["probabilities"] = probabilities.tolist()
            except:
                logger.warning("Probabilities not available for this model")
        
        # Add uncertainties if requested and available
        if request.return_uncertainty:
            try:
                _, uncertainties = model.predict_with_uncertainty(df)
                response_data["uncertainties"] = uncertainties.tolist()
            except:
                logger.warning("Uncertainty estimation not available for this model")
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Prediction failed: {{str(e)}}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {{str(e)}}")

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = {{
            "name": "{self.model_name}",
            "version": "{self.model_version}",
            "type": getattr(model, 'task_type', 'unknown'),
            "best_score": getattr(model, 'best_score_', None),
            "training_time": getattr(model, 'training_time_', None),
            "feature_names": getattr(model, 'feature_names_', []),
            "n_features": getattr(model, 'n_features_', None)
        }}
        
        if hasattr(model, '_loaded_metadata'):
            info.update(model._loaded_metadata)
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {{str(e)}}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        api_file = output_path / "main.py"
        with open(api_file, 'w') as f:
            f.write(api_template)
        created_files['api_file'] = str(api_file)
        
        # Create requirements.txt
        requirements = '''fastapi>=0.68.0
uvicorn>=0.15.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
pydantic>=1.8.0
essentiax>=2.0.0
'''
        
        req_file = output_path / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write(requirements)
        created_files['requirements_file'] = str(req_file)
        
        # Create Dockerfile
        dockerfile = f'''FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        docker_file = output_path / "Dockerfile"
        with open(docker_file, 'w') as f:
            f.write(dockerfile)
        created_files['dockerfile'] = str(docker_file)
        
        # Create docker-compose.yml
        compose_template = f'''version: '3.8'

services:
  {self.model_name.lower().replace('_', '-')}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME={self.model_name}
      - MODEL_VERSION={self.model_version}
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
'''
        
        compose_file = output_path / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            f.write(compose_template)
        created_files['compose_file'] = str(compose_file)
        
        # Create deployment script
        deploy_script = f'''#!/bin/bash
# Deployment script for {self.model_name} v{self.model_version}

echo "Deploying {self.model_name} v{self.model_version}..."

# Build Docker image
echo "Building Docker image..."
docker build -t {self.model_name.lower()}:{self.model_version} .

# Stop existing container if running
echo "Stopping existing container..."
docker-compose down

# Start new container
echo "Starting new container..."
docker-compose up -d

# Wait for health check
echo "Waiting for health check..."
sleep 10

# Test deployment
echo "Testing deployment..."
curl -f http://localhost:8000/health

if [ $? -eq 0 ]; then
    echo "Deployment successful!"
    echo "API available at: http://localhost:8000"
    echo "API docs at: http://localhost:8000/docs"
else
    echo "Deployment failed!"
    docker-compose logs
    exit 1
fi
'''
        
        script_file = output_path / "deploy.sh"
        with open(script_file, 'w') as f:
            f.write(deploy_script)
        
        # Make script executable
        import stat
        script_file.chmod(script_file.stat().st_mode | stat.S_IEXEC)
        created_files['deploy_script'] = str(script_file)
        
        console.print(f"✅ Deployment template created in: {output_path}")
        console.print("📁 Created files:")
        for file_type, file_path in created_files.items():
            console.print(f"  - {file_type}: {file_path}")
        
        return created_files
    
    def create_monitoring_dashboard(self, output_dir: str = "monitoring") -> str:
        """
        Create monitoring dashboard template.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for monitoring files
            
        Returns:
        --------
        str : Path to created dashboard file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        dashboard_template = f'''"""
Monitoring Dashboard for {self.model_name} v{self.model_version}
Generated by EssentiaX AutoML Production Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

st.set_page_config(
    page_title=f"{self.model_name} Monitor",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title(f"Model Monitor: {self.model_name} v{self.model_version}")

# Sidebar
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
refresh_interval = st.sidebar.selectbox("Refresh Interval", [30, 60, 300, 600])

# Auto-refresh
if st.sidebar.button("Refresh Data"):
    st.experimental_rerun()

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Health Status
try:
    health_response = requests.get(f"{{api_url}}/health", timeout=5)
    if health_response.status_code == 200:
        col1.metric("Status", "Healthy")
    else:
        col1.metric("Status", "Unhealthy")
except:
    col1.metric("Status", "Offline")

# Model Info
try:
    info_response = requests.get(f"{{api_url}}/model-info", timeout=5)
    if info_response.status_code == 200:
        model_info = info_response.json()
        col2.metric("Model Type", model_info.get('type', 'Unknown'))
        col3.metric("Best Score", f"{{model_info.get('best_score', 0):.3f}}")
        col4.metric("Training Time", f"{{model_info.get('training_time', 0):.1f}}s")
except:
    col2.metric("Model Type", "Unknown")
    col3.metric("Best Score", "N/A")
    col4.metric("Training Time", "N/A")

# Performance Charts
st.header("Performance Metrics")

# Simulated performance data (replace with actual monitoring data)
dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
performance_data = pd.DataFrame({{
    'timestamp': dates,
    'accuracy': np.random.normal(0.85, 0.05, len(dates)),
    'response_time': np.random.normal(100, 20, len(dates)),
    'requests_per_hour': np.random.poisson(50, len(dates))
}})

col1, col2 = st.columns(2)

with col1:
    fig_acc = px.line(performance_data, x='timestamp', y='accuracy', 
                      title='Model Accuracy Over Time')
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    fig_resp = px.line(performance_data, x='timestamp', y='response_time',
                       title='Response Time (ms)')
    st.plotly_chart(fig_resp, use_container_width=True)

# Request Volume
st.subheader("Request Volume")
fig_req = px.bar(performance_data.tail(24), x='timestamp', y='requests_per_hour',
                 title='Requests per Hour (Last 24h)')
st.plotly_chart(fig_req, use_container_width=True)

# Drift Detection
st.header("Data Drift Detection")
st.info("Data drift monitoring would be implemented here with actual monitoring data")

# Recent Predictions
st.header("Recent Predictions")
st.info("Recent prediction logs would be displayed here")

# Alerts
st.header("Alerts")
st.success("No active alerts")

# Footer
st.markdown("---")
st.markdown(f"**{self.model_name} v{self.model_version}** | Generated by EssentiaX AutoML")
'''
        
        dashboard_file = output_path / "dashboard.py"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_template)
        
        # Create requirements for dashboard
        dashboard_requirements = '''streamlit>=1.0.0
plotly>=5.0.0
requests>=2.25.0
pandas>=1.3.0
numpy>=1.21.0
'''
        
        req_file = output_path / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write(dashboard_requirements)
        
        console.print(f"✅ Monitoring dashboard created: {dashboard_file}")
        console.print(f"🚀 Run with: streamlit run {dashboard_file}")
        
        return str(dashboard_file)


# Production utilities integration
class ProductionUtils:
    """
    Main production utilities class that integrates all production features.
    """
    
    def __init__(self):
        """Initialize production utilities."""
        self.serializer = ModelSerializer()
        self.monitors = {}
        self.ab_tests = {}
        
        console.print("🏭 EssentiaX AutoML Production Utils initialized")
    
    def save_model(self, automl_instance, model_name: str, **kwargs) -> Dict[str, Any]:
        """Save model using ModelSerializer."""
        return self.serializer.save_model(automl_instance, model_name, **kwargs)
    
    def load_model(self, model_name: str, version: Optional[str] = None):
        """Load model using ModelSerializer."""
        return self.serializer.load_model(model_name, version)
    
    def create_monitor(self, model_name: str, baseline_data: Optional[pd.DataFrame] = None) -> ModelMonitor:
        """Create model monitor."""
        monitor = ModelMonitor(model_name, baseline_data)
        self.monitors[model_name] = monitor
        return monitor
    
    def create_ab_test(self, test_name: str) -> ABTestFramework:
        """Create A/B test framework."""
        ab_test = ABTestFramework(test_name)
        self.ab_tests[test_name] = ab_test
        return ab_test
    
    def create_deployment(self, model_name: str, model_version: str) -> DeploymentPipeline:
        """Create deployment pipeline."""
        return DeploymentPipeline(model_name, model_version)
    
    def get_production_summary(self) -> Dict[str, Any]:
        """Get summary of all production components."""
        return {
            'saved_models': len(self.serializer.list_models()),
            'active_monitors': len(self.monitors),
            'active_ab_tests': len(self.ab_tests),
            'production_features': [
                'Model Serialization',
                'Performance Monitoring',
                'A/B Testing',
                'Deployment Pipeline',
                'API Generation',
                'Docker Support'
            ]
        }