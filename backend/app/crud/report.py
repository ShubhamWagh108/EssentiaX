"""
CRUD operations for Report and Dataset models
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..models.report import Report, Dataset
from ..schemas.report import ReportCreate, ReportUpdate, DatasetCreate, DatasetUpdate
import uuid


def get_dataset(db: Session, dataset_id: int) -> Optional[Dataset]:
    """Get dataset by ID"""
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def get_dataset_by_hash(db: Session, file_hash: str) -> Optional[Dataset]:
    """Get dataset by file hash"""
    return db.query(Dataset).filter(Dataset.file_hash == file_hash).first()


def get_user_datasets(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Dataset]:
    """Get datasets owned by user"""
    return (
        db.query(Dataset)
        .filter(Dataset.owner_id == user_id)
        .order_by(desc(Dataset.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_dataset(db: Session, dataset: DatasetCreate, owner_id: int, 
                  filename: str, file_path: str, file_size: int, file_hash: str) -> Dataset:
    """Create new dataset"""
    db_dataset = Dataset(
        name=dataset.name,
        description=dataset.description,
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        file_hash=file_hash,
        is_public=dataset.is_public,
        owner_id=owner_id,
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def update_dataset(db: Session, dataset_id: int, dataset_update: DatasetUpdate) -> Optional[Dataset]:
    """Update dataset"""
    db_dataset = get_dataset(db, dataset_id)
    if not db_dataset:
        return None
    
    update_data = dataset_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_dataset, field, value)
    
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def update_dataset_metadata(db: Session, dataset_id: int, rows_count: int, 
                           columns_count: int, columns_info: dict) -> Optional[Dataset]:
    """Update dataset metadata after processing"""
    db_dataset = get_dataset(db, dataset_id)
    if not db_dataset:
        return None
    
    db_dataset.rows_count = rows_count
    db_dataset.columns_count = columns_count
    db_dataset.columns_info = columns_info
    db_dataset.status = "ready"
    
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def delete_dataset(db: Session, dataset_id: int) -> bool:
    """Delete dataset"""
    db_dataset = get_dataset(db, dataset_id)
    if not db_dataset:
        return False
    
    db.delete(db_dataset)
    db.commit()
    return True


# Report CRUD operations
def get_report(db: Session, report_id: int) -> Optional[Report]:
    """Get report by ID"""
    return db.query(Report).filter(Report.id == report_id).first()


def get_report_by_share_token(db: Session, share_token: str) -> Optional[Report]:
    """Get report by share token"""
    return db.query(Report).filter(Report.share_token == share_token).first()


def get_user_reports(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Report]:
    """Get reports owned by user"""
    return (
        db.query(Report)
        .filter(Report.owner_id == user_id)
        .order_by(desc(Report.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_public_reports(db: Session, skip: int = 0, limit: int = 100) -> List[Report]:
    """Get public reports"""
    return (
        db.query(Report)
        .filter(Report.is_public == True)
        .order_by(desc(Report.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_report(db: Session, report: ReportCreate, owner_id: int) -> Report:
    """Create new report"""
    db_report = Report(
        title=report.title,
        description=report.description,
        report_type=report.report_type,
        analysis_mode=report.analysis_mode,
        target_column=report.target_column,
        sample_size=report.sample_size,
        is_public=report.is_public,
        owner_id=owner_id,
        dataset_id=report.dataset_id,
        status="pending"
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report


def update_report(db: Session, report_id: int, report_update: ReportUpdate) -> Optional[Report]:
    """Update report"""
    db_report = get_report(db, report_id)
    if not db_report:
        return None
    
    update_data = report_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_report, field, value)
    
    db.commit()
    db.refresh(db_report)
    return db_report


def update_report_results(db: Session, report_id: int, html_report_path: str = None,
                         json_results: dict = None, data_quality_score: int = None,
                         problem_type: str = None, status: str = "completed") -> Optional[Report]:
    """Update report with analysis results"""
    db_report = get_report(db, report_id)
    if not db_report:
        return None
    
    if html_report_path:
        db_report.html_report_path = html_report_path
    if json_results:
        db_report.json_results = json_results
    if data_quality_score is not None:
        db_report.data_quality_score = data_quality_score
    if problem_type:
        db_report.problem_type = problem_type
    
    db_report.status = status
    
    db.commit()
    db.refresh(db_report)
    return db_report


def generate_share_token(db: Session, report_id: int) -> Optional[str]:
    """Generate share token for report"""
    db_report = get_report(db, report_id)
    if not db_report:
        return None
    
    share_token = str(uuid.uuid4())
    db_report.share_token = share_token
    db_report.is_public = True
    
    db.commit()
    return share_token


def delete_report(db: Session, report_id: int) -> bool:
    """Delete report"""
    db_report = get_report(db, report_id)
    if not db_report:
        return False
    
    db.delete(db_report)
    db.commit()
    return True


def get_dashboard_stats(db: Session, user_id: int) -> dict:
    """Get dashboard statistics for user"""
    total_reports = db.query(Report).filter(Report.owner_id == user_id).count()
    total_datasets = db.query(Dataset).filter(Dataset.owner_id == user_id).count()
    
    recent_reports = (
        db.query(Report)
        .filter(Report.owner_id == user_id)
        .order_by(desc(Report.created_at))
        .limit(5)
        .all()
    )
    
    recent_datasets = (
        db.query(Dataset)
        .filter(Dataset.owner_id == user_id)
        .order_by(desc(Dataset.created_at))
        .limit(5)
        .all()
    )
    
    # Calculate storage used
    storage_used = (
        db.query(Dataset)
        .filter(Dataset.owner_id == user_id)
        .with_entities(Dataset.file_size)
        .all()
    )
    total_storage = sum(size[0] for size in storage_used if size[0])
    
    return {
        "total_reports": total_reports,
        "total_datasets": total_datasets,
        "recent_reports": recent_reports,
        "recent_datasets": recent_datasets,
        "storage_used": total_storage
    }