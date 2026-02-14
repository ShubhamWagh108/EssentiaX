"""
Pydantic schemas for EssentiaX Backend
"""
from .user import (
    User, UserCreate, UserUpdate, UserLogin, 
    Token, PasswordReset, PasswordResetRequest
)
from .report import (
    Dataset, DatasetCreate, DatasetUpdate,
    Report, ReportCreate, ReportUpdate,
    AnalysisRequest, AnalysisResponse,
    FileUploadResponse, ShareReportRequest, ShareReportResponse,
    DashboardStats, ReportSummary,
    ReportType, AnalysisMode, ReportStatus, DatasetStatus
)

__all__ = [
    # User schemas
    "User", "UserCreate", "UserUpdate", "UserLogin",
    "Token", "PasswordReset", "PasswordResetRequest",
    
    # Report and Dataset schemas
    "Dataset", "DatasetCreate", "DatasetUpdate",
    "Report", "ReportCreate", "ReportUpdate",
    "AnalysisRequest", "AnalysisResponse",
    "FileUploadResponse", "ShareReportRequest", "ShareReportResponse",
    "DashboardStats", "ReportSummary",
    
    # Enums
    "ReportType", "AnalysisMode", "ReportStatus", "DatasetStatus"
]