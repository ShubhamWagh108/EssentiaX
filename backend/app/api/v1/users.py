"""
User management endpoints
"""
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...db.database import get_db
from ...api.deps import get_current_user, get_current_active_superuser
from ...crud import user as crud_user
from ...schemas.user import User, UserUpdate
from ...models.user import User as UserModel

router = APIRouter()


@router.get("/me", response_model=User)
def read_user_me(
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get current user profile
    """
    return current_user


@router.put("/me", response_model=User)
def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Update current user profile
    """
    # Check if email is being changed and already exists
    if user_in.email and user_in.email != current_user.email:
        existing_user = crud_user.get_user_by_email(db, email=user_in.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )
    
    # Check if username is being changed and already exists
    if user_in.username and user_in.username != current_user.username:
        existing_user = crud_user.get_user_by_username(db, username=user_in.username)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this username already exists"
            )
    
    user = crud_user.update_user(db, user_id=current_user.id, user_update=user_in)
    return user


@router.get("/{user_id}", response_model=User)
def read_user_by_id(
    user_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Get a specific user by id (only for superusers or the user themselves)
    """
    user = crud_user.get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    # Users can only see their own profile unless they're superuser
    if user.id != current_user.id and not crud_user.is_superuser(current_user):
        raise HTTPException(
            status_code=403,
            detail="Not enough permissions"
        )
    
    return user


@router.get("/", response_model=List[User])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: UserModel = Depends(get_current_active_superuser),
) -> Any:
    """
    Retrieve users (superuser only)
    """
    users = db.query(UserModel).offset(skip).limit(limit).all()
    return users


@router.put("/{user_id}", response_model=User)
def update_user(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    user_in: UserUpdate,
    current_user: UserModel = Depends(get_current_active_superuser),
) -> Any:
    """
    Update a user (superuser only)
    """
    user = crud_user.get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    user = crud_user.update_user(db, user_id=user_id, user_update=user_in)
    return user