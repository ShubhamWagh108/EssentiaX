"""
EssentiaX AutoML - Advanced Interactive Dashboard
===============================================

Interactive training dashboard with real-time updates and comprehensive visualization.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import time
import threading
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import sys

class AdvancedDashboard:
    """
    🎛️ Advanced Interactive Dashboard for AutoML Training
    
    Provides real-time visualization of:
    - Training progress with detailed metrics
    - Model performance comparisons
    - Hyperparameter optimization progress
    - Resource usage monitoring
    - Interactive controls and status updates
    """
    
    def __init__(self, console: Optional[Console] = None, update_interval: float = 0.5, test_mode: bool = False):
        """
        Initialize the advanced dashboard.
        
        Parameters:
        -----------
        console : Console, optional
            Rich console instance for output
        update_interval : float, default=0.5
            Update interval in seconds for real-time updates
        test_mode : bool, default=False
            Whether to run in test mode (disables threading)
        """
        self.console = console or Console()
        self.update_interval = update_interval
        self._test_mode = test_mode
        
        # Dashboard state
        self.is_active = False
        self.start_time = None
        self.current_phase = "Initializing"
        self.progress_data = {}
        self.model_results = []
        self.hyperopt_history = []
        self.resource_usage = []
        self.status_messages = []
        
        # Layout components
        self.layout = None
        self.live_display = None
        self._update_thread = None
        
        # Performance tracking
        self.best_score = None
        self.best_model = None
        self.total_models_trained = 0
        self.current_model = None
        
    def start(self, total_time_budget: int = 3600, task_type: str = "classification"):
        """
        Start the interactive dashboard.
        
        Parameters:
        -----------
        total_time_budget : int
            Total time budget for training in seconds
        task_type : str
            Type of ML task (classification, regression, etc.)
        """
        self.is_active = True
        self.start_time = time.time()
        self.total_time_budget = total_time_budget
        self.task_type = task_type
        
        # Initialize layout
        self._create_layout()
        
        # Start live display (but don't start update thread in test mode)
        try:
            self.live_display = Live(
                self.layout,
                console=self.console,
                refresh_per_second=1/self.update_interval,
                screen=False
            )
            self.live_display.start()
            
            # Only start update thread if not in test mode
            if not hasattr(self, '_test_mode'):
                self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
                self._update_thread.start()
        except Exception as e:
            # Fallback for environments where Live display doesn't work
            pass
        
        self.add_status_message("🚀 Dashboard started - AutoML training initiated")
    
    def stop(self):
        """Stop the dashboard and cleanup resources."""
        self.is_active = False
        
        if self.live_display:
            try:
                self.live_display.stop()
            except Exception:
                pass  # Ignore errors during cleanup
        
        if self._update_thread and self._update_thread.is_alive():
            try:
                self._update_thread.join(timeout=1.0)
            except Exception:
                pass  # Ignore errors during cleanup
        
        self.add_status_message("✅ Dashboard stopped - Training completed")
    
    def _create_layout(self):
        """Create the dashboard layout structure."""
        self.layout = Layout()
        
        # Main layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Split main area
        self.layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left area
        self.layout["left"].split_column(
            Layout(name="progress", size=8),
            Layout(name="models", size=12),
            Layout(name="hyperopt")
        )
        
        # Split right area
        self.layout["right"].split_column(
            Layout(name="status", size=10),
            Layout(name="resources")
        )
    
    def _update_loop(self):
        """Main update loop for real-time dashboard updates."""
        while self.is_active:
            try:
                self._update_display()
                time.sleep(self.update_interval)
            except Exception as e:
                self.add_status_message(f"⚠️ Dashboard update error: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _update_display(self):
        """Update all dashboard components."""
        if not self.layout:
            return
        
        # Update header
        self.layout["header"].update(self._create_header())
        
        # Update progress section
        self.layout["progress"].update(self._create_progress_panel())
        
        # Update models section
        self.layout["models"].update(self._create_models_panel())
        
        # Update hyperparameter optimization
        self.layout["hyperopt"].update(self._create_hyperopt_panel())
        
        # Update status
        self.layout["status"].update(self._create_status_panel())
        
        # Update resources
        self.layout["resources"].update(self._create_resources_panel())
        
        # Update footer
        self.layout["footer"].update(self._create_footer())
    
    def _create_header(self) -> Panel:
        """Create the header panel with title and basic info."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        remaining_time = max(0, self.total_time_budget - elapsed_time)
        
        progress_pct = (elapsed_time / self.total_time_budget * 100) if self.total_time_budget > 0 else 0
        
        header_text = Text()
        header_text.append("🤖 EssentiaX AutoML Dashboard", style="bold blue")
        header_text.append(f" | Task: {self.task_type.title()}", style="cyan")
        header_text.append(f" | Progress: {progress_pct:.1f}%", style="green")
        header_text.append(f" | Remaining: {remaining_time/60:.1f}min", style="yellow")
        
        return Panel(
            Align.center(header_text),
            style="blue",
            title="AutoML Training Dashboard"
        )
    
    def _create_progress_panel(self) -> Panel:
        """Create the progress tracking panel."""
        if not self.progress_data:
            return Panel("⏳ Waiting for training to begin...", title="Training Progress")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="cyan", width=20)
        table.add_column("Status", style="green", width=15)
        table.add_column("Progress", style="yellow", width=20)
        table.add_column("Time", style="blue", width=10)
        
        for phase, data in self.progress_data.items():
            status = "✅ Complete" if data.get('completed', False) else "🔄 Running"
            progress_bar = self._create_progress_bar(data.get('progress', 0))
            elapsed = f"{data.get('elapsed_time', 0):.1f}s"
            
            table.add_row(phase, status, progress_bar, elapsed)
        
        return Panel(table, title="🎯 Training Progress", border_style="green")
    
    def _create_models_panel(self) -> Panel:
        """Create the models comparison panel."""
        if not self.model_results:
            return Panel("📊 No models trained yet...", title="Model Performance")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", width=15)
        table.add_column("Score", style="green", width=10)
        table.add_column("Training Time", style="yellow", width=12)
        table.add_column("Status", style="blue", width=10)
        
        # Sort models by score (descending)
        sorted_models = sorted(self.model_results, key=lambda x: x.get('score', 0), reverse=True)
        
        for i, model in enumerate(sorted_models[:10]):  # Show top 10
            model_name = model.get('name', 'Unknown')
            score = f"{model.get('score', 0):.4f}"
            train_time = f"{model.get('training_time', 0):.2f}s"
            status = "🏆 Best" if i == 0 else "✅ Complete"
            
            table.add_row(model_name, score, train_time, status)
        
        return Panel(table, title="🏆 Model Performance Leaderboard", border_style="yellow")
    
    def _create_hyperopt_panel(self) -> Panel:
        """Create the hyperparameter optimization panel."""
        if not self.hyperopt_history:
            return Panel("🔧 Hyperparameter optimization not started...", title="Hyperparameter Optimization")
        
        # Show recent optimization attempts
        recent_attempts = self.hyperopt_history[-5:]  # Last 5 attempts
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Iteration", style="cyan", width=8)
        table.add_column("Score", style="green", width=10)
        table.add_column("Parameters", style="yellow", width=30)
        table.add_column("Improvement", style="blue", width=12)
        
        for i, attempt in enumerate(recent_attempts):
            iteration = str(attempt.get('iteration', i))
            score = f"{attempt.get('score', 0):.4f}"
            params = str(attempt.get('best_params', {}))[:28] + "..."
            improvement = "🔥 New Best!" if attempt.get('is_best', False) else "📈 Progress"
            
            table.add_row(iteration, score, params, improvement)
        
        return Panel(table, title="🔧 Hyperparameter Optimization", border_style="cyan")
    
    def _create_status_panel(self) -> Panel:
        """Create the status messages panel."""
        if not self.status_messages:
            return Panel("📝 No status messages yet...", title="Status Log")
        
        # Show recent messages
        recent_messages = self.status_messages[-8:]  # Last 8 messages
        
        status_text = Text()
        for msg in recent_messages:
            timestamp = msg.get('timestamp', '')
            message = msg.get('message', '')
            status_text.append(f"[{timestamp}] {message}\n", style="white")
        
        return Panel(status_text, title="📝 Status Log", border_style="white")
    
    def _create_resources_panel(self) -> Panel:
        """Create the resource usage panel."""
        if not self.resource_usage:
            return Panel("💻 Resource monitoring not available...", title="Resource Usage")
        
        latest = self.resource_usage[-1] if self.resource_usage else {}
        
        resource_text = Text()
        resource_text.append(f"CPU Usage: {latest.get('cpu_percent', 0):.1f}%\n", style="cyan")
        resource_text.append(f"Memory: {latest.get('memory_mb', 0):.1f} MB\n", style="green")
        resource_text.append(f"Models Trained: {self.total_models_trained}\n", style="yellow")
        resource_text.append(f"Best Score: {self.best_score:.4f}\n" if self.best_score else "Best Score: N/A\n", style="blue")
        
        return Panel(resource_text, title="💻 Resource Usage", border_style="magenta")
    
    def _create_footer(self) -> Panel:
        """Create the footer with controls and summary."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        footer_text = Text()
        footer_text.append("Controls: ", style="bold white")
        footer_text.append("[Ctrl+C] Stop Training ", style="red")
        footer_text.append("| ", style="white")
        footer_text.append(f"Elapsed: {elapsed_time/60:.1f}min ", style="green")
        footer_text.append("| ", style="white")
        footer_text.append(f"Current Phase: {self.current_phase}", style="cyan")
        
        return Panel(
            Align.center(footer_text),
            style="white",
            title="Dashboard Controls"
        )
    
    def _create_progress_bar(self, progress: float) -> str:
        """Create a text-based progress bar."""
        bar_length = 15
        filled_length = int(bar_length * progress / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        return f"{bar} {progress:.1f}%"
    
    # Public methods for updating dashboard state
    
    def update_phase(self, phase_name: str, progress: float = 0, completed: bool = False):
        """Update the current training phase."""
        self.current_phase = phase_name
        
        if phase_name not in self.progress_data:
            self.progress_data[phase_name] = {
                'start_time': time.time(),
                'progress': 0,
                'completed': False,
                'elapsed_time': 0
            }
        
        self.progress_data[phase_name]['progress'] = progress
        self.progress_data[phase_name]['completed'] = completed
        self.progress_data[phase_name]['elapsed_time'] = time.time() - self.progress_data[phase_name]['start_time']
    
    def add_model_result(self, model_name: str, score: float, training_time: float, 
                        params: Optional[Dict] = None):
        """Add a model training result."""
        self.total_models_trained += 1
        self.current_model = model_name
        
        result = {
            'name': model_name,
            'score': score,
            'training_time': training_time,
            'params': params or {},
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        self.model_results.append(result)
        
        # Update best score
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_model = model_name
            self.add_status_message(f"🏆 New best model: {model_name} (Score: {score:.4f})")
    
    def add_hyperopt_result(self, iteration: int, score: float, params: Dict, is_best: bool = False):
        """Add a hyperparameter optimization result."""
        result = {
            'iteration': iteration,
            'score': score,
            'best_params': params,
            'is_best': is_best,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        self.hyperopt_history.append(result)
    
    def add_status_message(self, message: str):
        """Add a status message to the log."""
        status_entry = {
            'message': message,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        self.status_messages.append(status_entry)
        
        # Keep only recent messages
        if len(self.status_messages) > 50:
            self.status_messages = self.status_messages[-50:]
    
    def update_resource_usage(self, cpu_percent: float, memory_mb: float):
        """Update resource usage statistics."""
        usage_entry = {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'timestamp': time.time()
        }
        
        self.resource_usage.append(usage_entry)
        
        # Keep only recent data
        if len(self.resource_usage) > 100:
            self.resource_usage = self.resource_usage[-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the training session."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'elapsed_time': elapsed_time,
            'total_models_trained': self.total_models_trained,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'current_phase': self.current_phase,
            'hyperopt_iterations': len(self.hyperopt_history),
            'status_messages_count': len(self.status_messages)
        }
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export all session data for analysis."""
        return {
            'progress_data': self.progress_data,
            'model_results': self.model_results,
            'hyperopt_history': self.hyperopt_history,
            'resource_usage': self.resource_usage,
            'status_messages': self.status_messages,
            'summary': self.get_summary()
        }

class DashboardIntegration:
    """
    🔗 Dashboard Integration Helper
    
    Provides easy integration between AutoML and the Advanced Dashboard.
    """
    
    def __init__(self, automl_instance, dashboard: Optional[AdvancedDashboard] = None):
        """
        Initialize dashboard integration.
        
        Parameters:
        -----------
        automl_instance : AutoML
            The AutoML instance to monitor
        dashboard : AdvancedDashboard, optional
            Dashboard instance to use
        """
        self.automl = automl_instance
        self.dashboard = dashboard or AdvancedDashboard()
        self.monitoring_active = False
    
    def start_monitoring(self, time_budget: int = 3600, task_type: str = "classification"):
        """Start dashboard monitoring for AutoML training."""
        self.dashboard.start(time_budget, task_type)
        self.monitoring_active = True
        
        # Hook into AutoML events (if supported)
        self._setup_automl_hooks()
    
    def stop_monitoring(self):
        """Stop dashboard monitoring."""
        self.monitoring_active = False
        self.dashboard.stop()
    
    def _setup_automl_hooks(self):
        """Setup hooks to capture AutoML events."""
        # This would integrate with AutoML's internal progress tracking
        # For now, we'll provide manual update methods
        pass
    
    def update_training_progress(self, phase: str, progress: float, completed: bool = False):
        """Update training progress in dashboard."""
        if self.monitoring_active:
            self.dashboard.update_phase(phase, progress, completed)
    
    def log_model_result(self, model_name: str, score: float, training_time: float):
        """Log a model training result."""
        if self.monitoring_active:
            self.dashboard.add_model_result(model_name, score, training_time)
    
    def log_hyperopt_result(self, iteration: int, score: float, params: Dict, is_best: bool = False):
        """Log a hyperparameter optimization result."""
        if self.monitoring_active:
            self.dashboard.add_hyperopt_result(iteration, score, params, is_best)
    
    def log_status(self, message: str):
        """Log a status message."""
        if self.monitoring_active:
            self.dashboard.add_status_message(message)