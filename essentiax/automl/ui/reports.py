"""
EssentiaX AutoML - Advanced Report Generation
============================================

Comprehensive report generation capabilities for AutoML results and analysis.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import os
from datetime import datetime
from jinja2 import Template
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from .visualizations import AutoMLVisualizer

class ReportGenerator:
    """
    📋 Advanced Report Generator for AutoML
    
    Generates comprehensive reports including:
    - Executive summaries with key insights
    - Detailed technical analysis
    - Model performance comparisons
    - Feature importance analysis
    - Hyperparameter optimization results
    - Recommendations and next steps
    - Export to multiple formats (HTML, PDF, JSON)
    """
    
    def __init__(self, visualizer: Optional[AutoMLVisualizer] = None):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        visualizer : AutoMLVisualizer, optional
            Visualizer instance for generating plots
        """
        self.visualizer = visualizer or AutoMLVisualizer()
        self.report_data = {}
        self.generated_plots = {}
        
    def generate_comprehensive_report(self, automl_results: Dict[str, Any], 
                                    dataset_info: Optional[Dict[str, Any]] = None,
                                    output_path: str = "./automl_report.html") -> str:
        """
        Generate a comprehensive AutoML report.
        
        Parameters:
        -----------
        automl_results : dict
            Dictionary containing AutoML results and metrics
        dataset_info : dict, optional
            Information about the dataset used
        output_path : str, default="./automl_report.html"
            Path to save the report
            
        Returns:
        --------
        str
            Path to the generated report
        """
        # Prepare report data
        self.report_data = self._prepare_report_data(automl_results, dataset_info)
        
        # Generate visualizations
        self._generate_report_visualizations(automl_results)
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 Comprehensive report generated: {output_path}")
        return output_path
    
    def generate_executive_summary(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an executive summary of AutoML results.
        
        Parameters:
        -----------
        automl_results : dict
            Dictionary containing AutoML results
            
        Returns:
        --------
        dict
            Executive summary data
        """
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'overview': {},
            'key_findings': [],
            'recommendations': [],
            'performance_metrics': {},
            'model_insights': {}
        }
        
        # Overview
        if 'model_results' in automl_results and automl_results['model_results']:
            best_model = max(automl_results['model_results'], 
                           key=lambda x: x.get('score', 0))
            summary['overview'] = {
                'best_model': best_model['name'],
                'best_score': best_model.get('score', 0),
                'total_models_trained': len(automl_results['model_results']),
                'training_time': sum(r.get('training_time', 0) 
                                   for r in automl_results['model_results'])
            }
        
        # Key findings
        summary['key_findings'] = self._extract_key_findings(automl_results)
        
        # Recommendations
        summary['recommendations'] = self._generate_recommendations(automl_results)
        
        # Performance metrics
        summary['performance_metrics'] = self._calculate_performance_metrics(automl_results)
        
        # Model insights
        summary['model_insights'] = self._extract_model_insights(automl_results)
        
        return summary
    
    def export_to_json(self, automl_results: Dict[str, Any], 
                      output_path: str = "./automl_results.json") -> str:
        """
        Export AutoML results to JSON format.
        
        Parameters:
        -----------
        automl_results : dict
            Dictionary containing AutoML results
        output_path : str, default="./automl_results.json"
            Path to save the JSON file
            
        Returns:
        --------
        str
            Path to the exported JSON file
        """
        # Prepare data for JSON export
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'essentiax_version': '2.0.0',
                'report_type': 'automl_results'
            },
            'executive_summary': self.generate_executive_summary(automl_results),
            'detailed_results': automl_results,
            'analysis': {
                'performance_analysis': self._analyze_performance(automl_results),
                'feature_analysis': self._analyze_features(automl_results),
                'optimization_analysis': self._analyze_optimization(automl_results)
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = self._convert_numpy_to_json(export_data)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results exported to JSON: {output_path}")
        return output_path
    
    def generate_model_comparison_report(self, model_results: List[Dict[str, Any]], 
                                       output_path: str = "./model_comparison.html") -> str:
        """
        Generate a detailed model comparison report.
        
        Parameters:
        -----------
        model_results : list
            List of model result dictionaries
        output_path : str, default="./model_comparison.html"
            Path to save the report
            
        Returns:
        --------
        str
            Path to the generated report
        """
        if not model_results:
            return self._generate_empty_report(output_path, "No models to compare")
        
        # Sort models by performance
        sorted_models = sorted(model_results, key=lambda x: x.get('score', 0), reverse=True)
        
        # Generate comparison visualizations
        comparison_plot = self.visualizer.plot_model_comparison(model_results)
        comparison_plot_b64 = self._plot_to_base64(comparison_plot)
        
        # Prepare comparison data
        comparison_data = {
            'models': sorted_models,
            'best_model': sorted_models[0] if sorted_models else None,
            'performance_gap': self._calculate_performance_gap(sorted_models),
            'training_efficiency': self._analyze_training_efficiency(sorted_models),
            'model_diversity': self._analyze_model_diversity(sorted_models)
        }
        
        # Generate HTML report
        html_content = self._generate_model_comparison_html(comparison_data, comparison_plot_b64)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Model comparison report generated: {output_path}")
        return output_path
    
    def _prepare_report_data(self, automl_results: Dict[str, Any], 
                           dataset_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare structured data for report generation."""
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': dataset_info or {},
            'automl_results': automl_results,
            'executive_summary': self.generate_executive_summary(automl_results),
            'analysis': {
                'performance': self._analyze_performance(automl_results),
                'features': self._analyze_features(automl_results),
                'optimization': self._analyze_optimization(automl_results),
                'efficiency': self._analyze_efficiency(automl_results)
            }
        }
    
    def _generate_report_visualizations(self, automl_results: Dict[str, Any]):
        """Generate all visualizations needed for the report."""
        self.generated_plots = {}
        
        # Model comparison plot
        if 'model_results' in automl_results:
            fig = self.visualizer.plot_model_comparison(automl_results['model_results'])
            self.generated_plots['model_comparison'] = self._plot_to_base64(fig)
            plt.close(fig)
        
        # Training progress plot
        if 'progress_data' in automl_results:
            fig = self.visualizer.plot_training_progress(automl_results['progress_data'])
            self.generated_plots['training_progress'] = self._plot_to_base64(fig)
            plt.close(fig)
        
        # Hyperparameter optimization plot
        if 'hyperopt_history' in automl_results:
            fig = self.visualizer.plot_hyperparameter_optimization(automl_results['hyperopt_history'])
            self.generated_plots['hyperopt_progress'] = self._plot_to_base64(fig)
            plt.close(fig)
        
        # Feature importance plot
        if 'feature_importance' in automl_results:
            fig = self.visualizer.plot_feature_importance(automl_results['feature_importance'])
            self.generated_plots['feature_importance'] = self._plot_to_base64(fig)
            plt.close(fig)
    
    def _generate_html_report(self) -> str:
        """Generate the complete HTML report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EssentiaX AutoML Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #333;
            border-left: 4px solid #007acc;
            padding-left: 15px;
            margin-bottom: 20px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.1em;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .findings-list {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }
        .findings-list ul {
            margin: 0;
            padding-left: 20px;
        }
        .findings-list li {
            margin-bottom: 10px;
            color: #333;
        }
        .recommendations {
            background-color: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
        .model-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .model-table th,
        .model-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .model-table th {
            background-color: #007acc;
            color: white;
        }
        .model-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 EssentiaX AutoML Report</h1>
            <div class="subtitle">Comprehensive Analysis & Results</div>
            <div class="subtitle">Generated on {{ timestamp }}</div>
        </div>

        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Best Model</h3>
                    <div class="value">{{ executive_summary.overview.best_model or 'N/A' }}</div>
                </div>
                <div class="metric-card">
                    <h3>Best Score</h3>
                    <div class="value">{{ "%.4f"|format(executive_summary.overview.best_score or 0) }}</div>
                </div>
                <div class="metric-card">
                    <h3>Models Trained</h3>
                    <div class="value">{{ executive_summary.overview.total_models_trained or 0 }}</div>
                </div>
                <div class="metric-card">
                    <h3>Training Time</h3>
                    <div class="value">{{ "%.1f"|format((executive_summary.overview.training_time or 0)/60) }}min</div>
                </div>
            </div>
        </div>

        {% if generated_plots.model_comparison %}
        <div class="section">
            <h2>🏆 Model Performance Comparison</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{{ generated_plots.model_comparison }}" alt="Model Comparison">
            </div>
        </div>
        {% endif %}

        {% if generated_plots.training_progress %}
        <div class="section">
            <h2>⏱️ Training Progress</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{{ generated_plots.training_progress }}" alt="Training Progress">
            </div>
        </div>
        {% endif %}

        {% if generated_plots.hyperopt_progress %}
        <div class="section">
            <h2>🔧 Hyperparameter Optimization</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{{ generated_plots.hyperopt_progress }}" alt="Hyperparameter Optimization">
            </div>
        </div>
        {% endif %}

        {% if generated_plots.feature_importance %}
        <div class="section">
            <h2>🎯 Feature Importance Analysis</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{{ generated_plots.feature_importance }}" alt="Feature Importance">
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>🔍 Key Findings</h2>
            <div class="findings-list">
                <ul>
                    {% for finding in executive_summary.key_findings %}
                    <li>{{ finding }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {% for recommendation in executive_summary.recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        {% if automl_results.model_results %}
        <div class="section">
            <h2>📋 Detailed Model Results</h2>
            <table class="model-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Training Time (s)</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in automl_results.model_results[:10] %}
                    <tr>
                        <td>{{ model.name }}</td>
                        <td>{{ "%.4f"|format(model.score or 0) }}</td>
                        <td>{{ "%.2f"|format(model.training_time or 0) }}</td>
                        <td>{{ model.params|length if model.params else 0 }} params</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="footer">
            <p>Generated by EssentiaX AutoML v2.0 | © 2024 EssentiaX</p>
            <p>For more information, visit our documentation</p>
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            timestamp=self.report_data['timestamp'],
            executive_summary=self.report_data['executive_summary'],
            automl_results=self.report_data['automl_results'],
            generated_plots=self.generated_plots
        )
    
    def _extract_key_findings(self, automl_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from AutoML results."""
        findings = []
        
        if 'model_results' in automl_results and automl_results['model_results']:
            model_results = automl_results['model_results']
            best_model = max(model_results, key=lambda x: x.get('score', 0))
            
            findings.append(f"Best performing model: {best_model['name']} with score {best_model.get('score', 0):.4f}")
            
            # Performance distribution
            scores = [r.get('score', 0) for r in model_results]
            if len(scores) > 1:
                score_std = np.std(scores)
                if score_std < 0.01:
                    findings.append("Model performances are very consistent across different algorithms")
                elif score_std > 0.1:
                    findings.append("Significant performance variation between models suggests algorithm choice is critical")
            
            # Training efficiency
            training_times = [r.get('training_time', 0) for r in model_results]
            if training_times:
                fastest_model = min(model_results, key=lambda x: x.get('training_time', float('inf')))
                findings.append(f"Most efficient model: {fastest_model['name']} ({fastest_model.get('training_time', 0):.2f}s)")
        
        if 'feature_importance' in automl_results:
            feature_importance = automl_results['feature_importance']
            if feature_importance:
                top_feature = max(feature_importance.items(), key=lambda x: x[1])
                findings.append(f"Most important feature: {top_feature[0]} (importance: {top_feature[1]:.4f})")
                
                # Feature concentration
                sorted_importance = sorted(feature_importance.values(), reverse=True)
                top_5_sum = sum(sorted_importance[:5]) / sum(sorted_importance) * 100
                if top_5_sum > 80:
                    findings.append("Feature importance is highly concentrated in top 5 features")
        
        if 'hyperopt_history' in automl_results:
            hyperopt_history = automl_results['hyperopt_history']
            if len(hyperopt_history) > 10:
                scores = [r.get('score', 0) for r in hyperopt_history]
                improvement = (max(scores) - min(scores)) / min(scores) * 100 if min(scores) > 0 else 0
                findings.append(f"Hyperparameter optimization achieved {improvement:.1f}% improvement")
        
        return findings
    
    def _generate_recommendations(self, automl_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if 'model_results' in automl_results and automl_results['model_results']:
            model_results = automl_results['model_results']
            
            # Model diversity recommendation
            model_types = set(r['name'].split('_')[0] for r in model_results)
            if len(model_types) < 3:
                recommendations.append("Consider trying more diverse model types for potentially better performance")
            
            # Training time vs performance
            scores = [r.get('score', 0) for r in model_results]
            times = [r.get('training_time', 0) for r in model_results]
            if times and scores:
                efficiency_scores = [s/t if t > 0 else 0 for s, t in zip(scores, times)]
                if max(efficiency_scores) > 0:
                    best_efficient = model_results[np.argmax(efficiency_scores)]
                    recommendations.append(f"For production deployment, consider {best_efficient['name']} for optimal speed-accuracy balance")
        
        if 'feature_importance' in automl_results:
            feature_importance = automl_results['feature_importance']
            if len(feature_importance) > 50:
                recommendations.append("Consider feature selection to reduce model complexity and improve interpretability")
        
        if 'hyperopt_history' in automl_results:
            hyperopt_history = automl_results['hyperopt_history']
            if len(hyperopt_history) < 20:
                recommendations.append("Increase hyperparameter optimization budget for potentially better results")
        
        # General recommendations
        recommendations.extend([
            "Validate model performance on a separate test set before deployment",
            "Monitor model performance in production for potential drift",
            "Consider ensemble methods to combine multiple models for better robustness"
        ])
        
        return recommendations
    
    def _calculate_performance_metrics(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        if 'model_results' in automl_results and automl_results['model_results']:
            scores = [r.get('score', 0) for r in automl_results['model_results']]
            training_times = [r.get('training_time', 0) for r in automl_results['model_results']]
            
            metrics.update({
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'total_training_time': np.sum(training_times),
                'mean_training_time': np.mean(training_times),
                'models_count': len(scores)
            })
        
        return metrics
    
    def _extract_model_insights(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights about model behavior and characteristics."""
        insights = {}
        
        if 'model_results' in automl_results and automl_results['model_results']:
            model_results = automl_results['model_results']
            
            # Model type analysis
            model_types = {}
            for result in model_results:
                model_type = result['name'].split('_')[0]
                if model_type not in model_types:
                    model_types[model_type] = []
                model_types[model_type].append(result.get('score', 0))
            
            # Best performing model type
            type_averages = {k: np.mean(v) for k, v in model_types.items()}
            best_type = max(type_averages.items(), key=lambda x: x[1])
            
            insights.update({
                'model_types_tested': list(model_types.keys()),
                'best_model_type': best_type[0],
                'best_type_avg_score': best_type[1],
                'model_type_performance': type_averages
            })
        
        return insights
    
    def _analyze_performance(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall performance characteristics."""
        analysis = {}
        
        if 'model_results' in automl_results:
            model_results = automl_results['model_results']
            if model_results:
                scores = [r.get('score', 0) for r in model_results]
                analysis.update({
                    'performance_consistency': 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0,
                    'performance_range': np.max(scores) - np.min(scores),
                    'top_quartile_threshold': np.percentile(scores, 75),
                    'performance_distribution': {
                        'q1': np.percentile(scores, 25),
                        'median': np.percentile(scores, 50),
                        'q3': np.percentile(scores, 75)
                    }
                })
        
        return analysis
    
    def _analyze_features(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance and characteristics."""
        analysis = {}
        
        if 'feature_importance' in automl_results:
            feature_importance = automl_results['feature_importance']
            if feature_importance:
                importance_values = list(feature_importance.values())
                analysis.update({
                    'total_features': len(feature_importance),
                    'importance_concentration': sum(sorted(importance_values, reverse=True)[:5]) / sum(importance_values),
                    'feature_diversity': len([v for v in importance_values if v > 0.01]),
                    'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
                })
        
        return analysis
    
    def _analyze_optimization(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hyperparameter optimization effectiveness."""
        analysis = {}
        
        if 'hyperopt_history' in automl_results:
            hyperopt_history = automl_results['hyperopt_history']
            if hyperopt_history:
                scores = [r.get('score', 0) for r in hyperopt_history]
                analysis.update({
                    'optimization_iterations': len(hyperopt_history),
                    'improvement_rate': (max(scores) - min(scores)) / len(scores) if len(scores) > 1 else 0,
                    'convergence_point': self._find_convergence_point(scores),
                    'optimization_efficiency': self._calculate_optimization_efficiency(scores)
                })
        
        return analysis
    
    def _analyze_efficiency(self, automl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training efficiency and resource utilization."""
        analysis = {}
        
        if 'model_results' in automl_results:
            model_results = automl_results['model_results']
            if model_results:
                scores = [r.get('score', 0) for r in model_results]
                times = [r.get('training_time', 0) for r in model_results]
                
                if times and scores:
                    efficiency_scores = [s/t if t > 0 else 0 for s, t in zip(scores, times)]
                    analysis.update({
                        'time_efficiency': np.mean(efficiency_scores),
                        'fastest_model_score': scores[np.argmin(times)] if times else 0,
                        'slowest_model_score': scores[np.argmax(times)] if times else 0,
                        'time_performance_correlation': np.corrcoef(times, scores)[0, 1] if len(times) > 1 else 0
                    })
        
        return analysis
    
    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    
    def _convert_numpy_to_json(self, obj):
        """Convert numpy arrays and types to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        else:
            return obj
    
    def _find_convergence_point(self, scores: List[float]) -> int:
        """Find the point where optimization converged."""
        if len(scores) < 5:
            return len(scores)
        
        # Look for point where improvement becomes minimal
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        threshold = max(improvements) * 0.1  # 10% of max improvement
        
        for i, improvement in enumerate(improvements):
            if improvement < threshold:
                return i + 1
        
        return len(scores)
    
    def _calculate_optimization_efficiency(self, scores: List[float]) -> float:
        """Calculate how efficiently the optimization process improved scores."""
        if len(scores) < 2:
            return 0.0
        
        total_improvement = max(scores) - min(scores)
        iterations = len(scores)
        
        return total_improvement / iterations if iterations > 0 else 0.0
    
    def _generate_empty_report(self, output_path: str, message: str) -> str:
        """Generate a simple report when no data is available."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>AutoML Report</title></head>
        <body>
            <h1>AutoML Report</h1>
            <p>{message}</p>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_model_comparison_html(self, comparison_data: Dict[str, Any], 
                                      plot_b64: str) -> str:
        """Generate HTML for model comparison report."""
        # This would be a detailed model comparison template
        # For brevity, returning a simple template
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .plot {{ text-align: center; margin: 30px 0; }}
                .plot img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
            <div class="plot">
                <img src="data:image/png;base64,{plot_b64}" alt="Model Comparison">
            </div>
            <h2>Best Model: {comparison_data.get('best_model', {}).get('name', 'N/A')}</h2>
            <p>Score: {comparison_data.get('best_model', {}).get('score', 0):.4f}</p>
        </body>
        </html>
        """
    
    def _calculate_performance_gap(self, sorted_models: List[Dict[str, Any]]) -> float:
        """Calculate performance gap between best and worst models."""
        if len(sorted_models) < 2:
            return 0.0
        
        best_score = sorted_models[0].get('score', 0)
        worst_score = sorted_models[-1].get('score', 0)
        
        return best_score - worst_score
    
    def _analyze_training_efficiency(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze training efficiency across models."""
        if not model_results:
            return {}
        
        times = [r.get('training_time', 0) for r in model_results]
        scores = [r.get('score', 0) for r in model_results]
        
        return {
            'avg_training_time': np.mean(times),
            'time_std': np.std(times),
            'efficiency_correlation': np.corrcoef(times, scores)[0, 1] if len(times) > 1 else 0
        }
    
    def _analyze_model_diversity(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diversity of models tested."""
        model_types = [r['name'].split('_')[0] for r in model_results]
        unique_types = set(model_types)
        
        return {
            'total_types': len(unique_types),
            'type_distribution': {t: model_types.count(t) for t in unique_types},
            'diversity_score': len(unique_types) / len(model_results) if model_results else 0
        }