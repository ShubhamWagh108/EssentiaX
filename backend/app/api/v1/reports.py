"""
Report management and analysis endpoints
"""
import os
import json
import pandas as pd
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ...db.database import get_db
from ...api.deps import get_current_user, get_optional_current_user
from ...crud import report as crud_report
from ...schemas.report import (
    Report, ReportCreate, ReportUpdate, AnalysisRequest, AnalysisResponse,
    ShareReportRequest, ShareReportResponse, DashboardStats
)
from ...models.user import User as UserModel
from ...core.config import settings

# Import the unified EDA function
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from essentiax.eda import smart_eda

router = APIRouter()


def run_eda_analysis(
    dataset_path: str,
    report_id: int,
    target_column: str = None,
    analysis_mode: str = "all",
    sample_size: int = None,
    max_plots: int = 8
):
    """
    Background task to run EDA analysis
    """
    from ...db.database import SessionLocal
    
    db = SessionLocal()
    try:
        # Load dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Create reports directory
        reports_dir = settings.REPORTS_DIR
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate HTML report path
        html_report_path = os.path.join(reports_dir, f"report_{report_id}.html")
        
        # Run EDA analysis
        results = smart_eda(
            df=df,
            target=target_column,
            mode=analysis_mode,
            sample_size=sample_size,
            report_path=html_report_path,
            max_plots=max_plots,
            show_visualizations=False  # Don't show in background task
        )
        
        # Update report with results
        crud_report.update_report_results(
            db=db,
            report_id=report_id,
            html_report_path=html_report_path,
            json_results=results,
            data_quality_score=results.get("data_quality_score"),
            problem_type=results.get("problem_type"),
            status="completed"
        )
        
    except Exception as e:
        # Update report with error status
        crud_report.update_report_results(
            db=db,
            report_id=report_id,
            status="failed"
        )
        # Log error (in production, use proper logging)
        print(f"Analysis failed for report {report_id}: {str(e)}")
    
    finally:
        db.close()


@router.post("/analyze", response_model=AnalysisResponse)
def create_analysis(
    *,
    db: Session = Depends(get_db),
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Create new EDA analysis report
    """
    # Get dataset
    dataset = crud_report.get_dataset(db, dataset_id=analysis_request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check permissions
    if dataset.owner_id != current_user.id and not dataset.is_public:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if dataset.status != "ready":
        raise HTTPException(status_code=400, detail="Dataset is not ready for analysis")
    
    # Validate target column if provided
    if analysis_request.target_column:
        if not dataset.columns_info or analysis_request.target_column not in dataset.columns_info:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{analysis_request.target_column}' not found in dataset"
            )
    
    # Create report record
    report_create = ReportCreate(
        title=analysis_request.report_title,
        description=analysis_request.report_description,
        report_type="eda",
        analysis_mode=analysis_request.analysis_mode,
        target_column=analysis_request.target_column,
        sample_size=analysis_request.sample_size,
        dataset_id=analysis_request.dataset_id,
        is_public=False
    )
    
    report = crud_report.create_report(
        db=db,
        report=report_create,
        owner_id=current_user.id
    )
    
    # Start background analysis
    background_tasks.add_task(
        run_eda_analysis,
        dataset.file_path,
        report.id,
        analysis_request.target_column,
        analysis_request.analysis_mode.value,
        analysis_request.sample_size,
        analysis_request.max_plots
    )
    
    return AnalysisResponse(
        report_id=report.id,
        status="processing",
        message="Analysis started. You will be notified when it's complete."
    )


@router.get("/", response_model=List[Report])
def read_reports(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Retrieve user's reports
    """
    reports = crud_report.get_user_reports(
        db, user_id=current_user.id, skip=skip, limit=limit
    )
    return reports


@router.get("/public", response_model=List[Report])
def read_public_reports(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve public reports
    """
    reports = crud_report.get_public_reports(db, skip=skip, limit=limit)
    return reports


@router.get("/{report_id}", response_model=Report)
def read_report(
    *,
    db: Session = Depends(get_db),
    report_id: int,
    current_user: UserModel = Depends(get_optional_current_user),
) -> Any:
    """
    Get report by ID
    """
    report = crud_report.get_report(db, report_id=report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if not report.is_public:
        if not current_user or report.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return report


@router.get("/shared/{share_token}", response_model=Report)
def read_shared_report(
    *,
    db: Session = Depends(get_db),
    share_token: str,
) -> Any:
    """
    Get report by share token
    """
    report = crud_report.get_report_by_share_token(db, share_token=share_token)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return report


@router.put("/{report_id}", response_model=Report)
def update_report(
    *,
    db: Session = Depends(get_db),
    report_id: int,
    report_in: ReportUpdate,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Update report
    """
    report = crud_report.get_report(db, report_id=report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    report = crud_report.update_report(db, report_id=report_id, report_update=report_in)
    return report


@router.post("/{report_id}/share", response_model=ShareReportResponse)
def share_report(
    *,
    db: Session = Depends(get_db),
    report_id: int,
    share_request: ShareReportRequest,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Share report and generate share token
    """
    report = crud_report.get_report(db, report_id=report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Generate share token
    share_token = crud_report.generate_share_token(db, report_id=report_id)
    if not share_token:
        raise HTTPException(status_code=500, detail="Failed to generate share token")
    
    # Construct share URL (in production, use proper domain)
    share_url = f"/shared/{share_token}"
    
    return ShareReportResponse(
        share_token=share_token,
        share_url=share_url,
        is_public=True
    )


@router.get("/{report_id}/download")
def download_report(
    *,
    db: Session = Depends(get_db),
    report_id: int,
    current_user: UserModel = Depends(get_optional_current_user),
) -> Any:
    """
    Download HTML report
    """
    report = crud_report.get_report(db, report_id=report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if not report.is_public:
        if not current_user or report.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if not report.html_report_path or not os.path.exists(report.html_report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=report.html_report_path,
        filename=f"{report.title.replace(' ', '_')}_report.html",
        media_type="text/html"
    )


@router.get("/{report_id}/results")
def get_report_results(
    *,
    db: Session = Depends(get_db),
    report_id: int,
    current_user: UserModel = Depends(get_optional_current_user),
) -> Any:
    """
    Get report analysis results as JSON
    """
    report = crud_report.get_report(db, report_id=report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if not report.is_public:
        if not current_user or report.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if not report.json_results:
        raise HTTPException(status_code=404, detail="Report results not available")
    
    return report.json_results


@router.delete("/{report_id}")
def delete_report(
    *,
    db: Session = Depends(get_db),
    report_id: int,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Delete report
    """
    report = crud_report.get_report(db, report_id=report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check permissions
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Delete HTML report file
    if report.html_report_path and os.path.exists(report.html_report_path):
        os.remove(report.html_report_path)
    
    # Delete database record
    crud_report.delete_report(db, report_id=report_id)
    
    return {"message": "Report deleted successfully"}


@router.get("/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(
    *,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get dashboard statistics for current user
    """
    stats = crud_report.get_dashboard_stats(db, user_id=current_user.id)
    return DashboardStats(**stats)