from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
# from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from backend.security.app.utils.security import create_access_token, verify_password, create_refresh_token, get_password_hash
from backend.security.app.schemas.user import Token,LoginRequest, UserCreate, UserResponse, RefreshRequest, TokenResponse, get_login_form, LogoutAllRequest, LogoutRequest
from backend.security.app.crud.user import get_user, create_user, get_user_by_username, get_user_by_email
from backend.security.app.utils.dependencies import get_db, get_current_user
from backend.security.app.utils.config import settings
import logging
from datetime import datetime
from backend.security.app.crud.security import store_refresh_token, verify_refresh_token, revoke_refresh_token, revoke_all_user_tokens
from backend.security.app.models import User

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/login", response_model=Token)
def login_for_access_token(
    login_data: LoginRequest,
    # request: Request,
    db: Session = Depends(get_db)
):
    logger.debug(f"Login attempt for user {login_data.email}")

    user = get_user_by_email(db, login_data.email)
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        logger.warning("Authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token with user data
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        user_id=user.id,
        email=user.email,
        username=user.username,
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(
        user_id=user.id,
        email=user.email,
        expires_delta=refresh_token_expires
    )
    
    # Store refresh token in database (hashed)
    res = store_refresh_token(
        db=db,
        user_id=user.id,
        refresh_token=refresh_token,
        expires_at=datetime.utcnow() + refresh_token_expires,
        # device_info=request.headers.get("User-Agent", ""),
        # ip_address=request.client.host if request.client else None
    )
    
    logger.info(f"User {user.email} logged in successfully")
    
    access_expires_at = datetime.utcnow() + access_token_expires
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_at": int(access_expires_at.timestamp()),
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "fullname": user.full_name,
            "roles":  user.roles,
            
            "created_at": user.created_at,
        }
    }


@router.post("/create_user/", response_model=UserResponse)
def create_user_endpoint(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    new_user = create_user(db=db, user=user)
    return UserResponse.model_validate(new_user)

# Pydantic model for refresh request

@router.post("/refresh", response_model=TokenResponse)
def refresh_access_token(
    refresh_request: RefreshRequest,
    db: Session = Depends(get_db)
):
    """
    Exchange a valid refresh token for a new access token
    """
    logger.debug("Refresh token request received")
    
    # # 1. Validate refresh token format
    # if not is_valid_token_format(refresh_request.refresh_token):
    #     logger.warning("Invalid refresh token format")
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Invalid token format",
    #     )
    
    # 2. Verify refresh token
    try:
        user = verify_refresh_token(db, refresh_request.refresh_token)
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user:
        logger.warning("Invalid refresh token attempt")
        # Log for security monitoring
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
   
    # 4. Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        user_id=user.id,
        email=user.email,
        username=user.username,
        expires_delta=access_token_expires
    )
    
    # 5. Always rotate refresh token (security best practice)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    new_refresh_token = create_refresh_token(
        user_id=user.id,
        email=user.email,
        expires_delta=refresh_token_expires
    )
    
    # 6. Store new refresh token with transaction
    try:
        # Use transaction for atomicity
        res = store_refresh_token(
            db=db,
            user_id=user.id,
            refresh_token=new_refresh_token,
            expires_at=datetime.utcnow() + refresh_token_expires,
        )
        
        # Revoke old refresh token
        revoke_refresh_token(db, refresh_request.refresh_token)
        
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to rotate refresh token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token",
        )
    
    # 7. Calculate expiration
    access_expires_at = datetime.utcnow() + access_token_expires
    
    # 8. Prepare response
    response_data = {
        "access_token":access_token,
        "refresh_token":new_refresh_token,
        "token_type":"bearer",
        "expires_at":int(access_expires_at.timestamp()),
        "user":{
            "id":user.id,
            "email":user.email,
            "username":user.username,
            "created_at":user.created_at,
        }
    }
        
    logger.info(f"Token refreshed for user {user.email}")
    
    return response_data



# Logout (revoke specific refresh token)
@router.post("/logout")
def logout(
    logout_request: LogoutRequest,
    # current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout by revoking a specific refresh token
    """
    if not logout_request.refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token required"
        )
    
    # Revoke the specific refresh token
    success = revoke_refresh_token(db, logout_request.refresh_token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Refresh token not found or already revoked"
        )
    
    logger.info(f"User logged out (specific token revoked)")
    
    return {
        "message": "Successfully logged out",
        "action": "single_token_revoked"
    }

# Logout from all devices
@router.post("/logout/all")
def logout_all(
    logout_all_request: LogoutAllRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout from all devices by revoking all refresh tokens
    """
    if not logout_all_request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required"
        )
    
    # Revoke all user's refresh tokens
    revoked_count = revoke_all_user_tokens(db, current_user.id)
    
    logger.info(f"User {current_user.email} logged out from all devices. {revoked_count} tokens revoked.")
    
    return {
        "message": "Successfully logged out from all devices",
        "user_id": current_user.id,
        "tokens_revoked": revoked_count,
        "action": "all_tokens_revoked"
    }