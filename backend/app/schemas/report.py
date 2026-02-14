"""
Pydantic schemas for report and dataset operations
"""
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ReportType(str, Enum):
    EDA = "eda"
    PROBLEM_CARD = "problem_card"
    VISUALIZATION = "visualization"


class AnalysisMode(str, Enum):
    CONSOLE = "console"
    HTML = "html"
    INTERACTIVE = "interactive"
    ALL = "all"


class ReportStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


# Dataset schemas
class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None


class Dataset(DatasetBase):
    id: int
    filename: str
    file_size: int
    rows_count: Optional[int] = None
    columns_count: Optional[int] = None
    columns_info: Optional[Dict[str, Any]] = None
    status: DatasetStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    owner_id: int

    class Config:
        from_attributes = True


# Report schemas
class ReportBase(BaseModel):
    title: str
    description: Optional[str] = None
    report_type: ReportType
    analysis_mode: AnalysisMode
    target_column: Optional[str] = None
    sample_size: Optional[int] = None
    is_public: bool = False


class ReportCreate(ReportBase):
    dataset_id: Optional[int] = None


class ReportUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None


class Report(ReportBase):
    id: int
    status: ReportStatus
    html_report_path: Optional[str] = None
    json_results: Optional[Dict[str, Any]] = None
    data_quality_score: Optional[int] = None
    problem_type: Optional[str] = None
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    share_token: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    owner_id: int
    dataset_id: Optional[int] = None

    class Config:
        from_attributes = True


# Analysis request schemas
class AnalysisRequest(BaseModel):
    dataset_id: int
    report_title: str
    report_description: Optional[str] = None
    target_column: Optional[str] = None
    analysis_mode: AnalysisMode = AnalysisMode.ALL
    sample_size: Optional[int] = None
    max_plots: int = 8
    show_visualizations: bool = True

    @validator('sample_size')
    def validate_sample_size(cls, v):
        if v is not None and v < 100:
            raise ValueError('Sample size must be at least 100')
        return v


class AnalysisResponse(BaseModel):
    report_id: int
    status: ReportStatus
    message: str


# File upload schemas
class FileUploadResponse(BaseModel):
    dataset_id: int
    filename: str
    file_size: int
    status: DatasetStatus
    message: str


# Sharing schemas
class ShareReportRequest(BaseModel):
    is_public: bool = True


class ShareReportResponse(BaseModel):
    share_token: str
    share_url: str
    is_public: bool


# Dashboard schemas
class DashboardStats(BaseModel):
    total_reports: int
    total_datasets: int
    recent_reports: List[Report]
    recent_datasets: List[Dataset]
    storage_used: int  # in bytes


class ReportSummary(BaseModel):
    id: int
    title: str
    report_type: ReportType
    status: ReportStatus
    data_quality_score: Optional[int] = None
    problem_type: Optional[str] = None
    created_at: datetime