"""
EssentiaX AutoML - Advanced UI & Reporting
==========================================

Advanced user interface and reporting capabilities for AutoML.
"""

try:
    from .dashboard import AdvancedDashboard, DashboardIntegration
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    AdvancedDashboard = None
    DashboardIntegration = None
    DASHBOARD_AVAILABLE = False

try:
    from .visualizations import AutoMLVisualizer
    VISUALIZATIONS_AVAILABLE = True
except ImportError as e:
    AutoMLVisualizer = None
    VISUALIZATIONS_AVAILABLE = False

try:
    from .reports import ReportGenerator
    REPORTS_AVAILABLE = True
except ImportError as e:
    ReportGenerator = None
    REPORTS_AVAILABLE = False

__all__ = ['AdvancedDashboard', 'DashboardIntegration', 'AutoMLVisualizer', 'ReportGenerator']