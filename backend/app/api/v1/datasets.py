"""
Dataset management endpoints
"""
import os
import hashlib
import pandas as pd
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from ...db.database import get_db
from ...api.deps import get_current_user
from ...crud import report as crud_report
from ...schemas.report import Dataset, DatasetCreate, DatasetUpdate, FileUploadResponse
from ...models.user import User as UserModel
from ...core.config import settings

router = APIRouter()


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def process_dataset_file(file_path: str) -> dict:
    """Process uploaded dataset file and extract metadata"""
    try:
        # Try to read as CSV first
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Extract metadata
        rows_count = len(df)
        columns_count = len(df.columns)
        
        # Column information
        columns_info = {}
        for col in df.columns:
            columns_info[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
        
        return {
            "rows_count": rows_count,
            "columns_count": columns_count,
            "columns_info": columns_info,
            "status": "ready"
        }
    
    except Exception as e:
        return {
            "rows_count": None,
            "columns_count": None,
            "columns_info": None,
            "status": "error",
            "error_message": str(e)
        }


@router.post("/upload", response_model=FileUploadResponse)
async def upload_dataset(
    *,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    is_public: bool = Form(False)
) -> Any:
    """
    Upload a new dataset file
    """
    # Validate file type
    allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Create upload directory if it doesn't exist
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(upload_dir, f"{current_user.id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        # Check if file already exists
        existing_dataset = crud_report.get_dataset_by_hash(db, file_hash)
        if existing_dataset:
            # Remove the duplicate file
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="This file has already been uploaded"
            )
        
        # Create dataset record
        dataset_create = DatasetCreate(
            name=name,
            description=description,
            is_public=is_public
        )
        
        dataset = crud_report.create_dataset(
            db=db,
            dataset=dataset_create,
            owner_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            file_size=file.size,
            file_hash=file_hash
        )
        
        # Process file in background (for now, do it synchronously)
        metadata = process_dataset_file(file_path)
        
        if metadata["status"] == "ready":
            crud_report.update_dataset_metadata(
                db=db,
                dataset_id=dataset.id,
                rows_count=metadata["rows_count"],
                columns_count=metadata["columns_count"],
                columns_info=metadata["columns_info"]
            )
        else:
            # Update with error status
            dataset.status = "error"
            dataset.error_message = metadata.get("error_message")
            db.commit()
        
        return FileUploadResponse(
            dataset_id=dataset.id,
            filename=file.filename,
            file_size=file.size,
            status=metadata["status"],
            message="Dataset uploaded and processed successfully" if metadata["status"] == "ready" 
                   else f"Dataset uploaded but processing failed: {metadata.get('error_message', 'Unknown error')}"
        )
    
    except Exception as e:
        # Clean up file if something went wrong
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[Dataset])
def read_datasets(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Retrieve user's datasets
    """
    datasets = crud_report.get_user_datasets(
        db, user_id=current_user.id, skip=skip, limit=limit
    )
    return datasets


@router.get("/{dataset_id}", response_model=Dataset)
def read_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get dataset by ID
    """
    dataset = crud_report.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check permissions
    if dataset.owner_id != current_user.id and not dataset.is_public:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return dataset


@router.put("/{dataset_id}", response_model=Dataset)
def update_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    dataset_in: DatasetUpdate,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Update dataset
    """
    dataset = crud_report.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check permissions
    if dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    dataset = crud_report.update_dataset(db, dataset_id=dataset_id, dataset_update=dataset_in)
    return dataset


@router.delete("/{dataset_id}")
def delete_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Delete dataset
    """
    dataset = crud_report.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check permissions
    if dataset.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Delete file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    
    # Delete database record
    crud_report.delete_dataset(db, dataset_id=dataset_id)
    
    return {"message": "Dataset deleted successfully"}


@router.get("/{dataset_id}/preview")
def preview_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    rows: int = 10,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Preview dataset content
    """
    dataset = crud_report.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check permissions
    if dataset.owner_id != current_user.id and not dataset.is_public:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if dataset.status != "ready":
        raise HTTPException(status_code=400, detail="Dataset is not ready for preview")
    
    try:
        # Read file
        if dataset.filename.endswith('.csv'):
            df = pd.read_csv(dataset.file_path, nrows=rows)
        elif dataset.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(dataset.file_path, nrows=rows)
        elif dataset.filename.endswith('.json'):
            df = pd.read_json(dataset.file_path)
            df = df.head(rows)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Convert to dict for JSON response
        preview_data = {
            "columns": df.columns.tolist(),
            "data": df.to_dict('records'),
            "total_rows": dataset.rows_count,
            "preview_rows": len(df)
        }
        
        return preview_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")