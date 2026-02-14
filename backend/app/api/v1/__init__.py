"""
API v1 endpoints
"""
from fastapi import APIRouter
from .auth import router as auth_router
from .users import router as users_router
from .datasets import router as datasets_router
from .reports import router as reports_router

api_router = APIRouter()

api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(users_router, prefix="/users", tags=["users"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(reports_router, prefix="/reports", tags=["reports"])