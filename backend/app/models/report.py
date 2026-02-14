"""
Report model for storing EDA analysis reports
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..db.database import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Report metadata
    report_type = Column(String, nullable=False)  # "eda", "problem_card", "visualization"
    status = Column(String, default="completed")  # "pending", "processing", "completed", "failed"
    
    # Analysis configuration
    target_column = Column(String, nullable=True)
    analysis_mode = Column(String, nullable=False)  # "console", "html", "interactive", "all"
    sample_size = Column(Integer, nullable=True)
    
    # Results and files
    html_report_path = Column(String, nullable=True)
    json_results = Column(JSON, nullable=True)
    data_quality_score = Column(Integer, nullable=True)
    problem_type = Column(String, nullable=True)
    
    # File information
    original_filename = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    file_hash = Column(String, nullable=True)
    
    # Sharing and visibility
    is_public = Column(Boolean, default=False)
    share_token = Column(String, unique=True, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="reports")
    dataset = relationship("Dataset", back_populates="reports")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # File information
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String, nullable=False, unique=True)
    mime_type = Column(String, nullable=True)
    
    # Dataset metadata
    rows_count = Column(Integer, nullable=True)
    columns_count = Column(Integer, nullable=True)
    columns_info = Column(JSON, nullable=True)  # Column names and types
    
    # Processing status
    status = Column(String, default="uploaded")  # "uploaded", "processing", "ready", "error"
    error_message = Column(Text, nullable=True)
    
    # Sharing
    is_public = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    reports = relationship("Report", back_populates="dataset")