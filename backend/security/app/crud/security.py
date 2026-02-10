from datetime import datetime
from typing import Optional
from backend.security.app.models.user import RefreshToken, User
from backend.security.app.utils.security import hash_token
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import logging
import jwt
from backend.security.app.utils.config import settings
import uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def store_refresh_token(
    db: Session,
    user_id: any,
    refresh_token: str,
    expires_at: datetime,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None
) -> RefreshToken:
    """
    Store hashed refresh token in database
    """
    # Create token hash
    token_hash = hash_token(refresh_token)
    # Create database entry
    db_token = RefreshToken(
        token_hash=token_hash,
        user_id=user_id,
        is_revoked = False,
        expires_at=expires_at,
        device_info=device_info,
        ip_address=ip_address
    )
    
    db.add(db_token)
    db.commit()
    db.refresh(db_token)
    
    return db_token


def verify_refresh_token(db: Session, refresh_token: str) -> Optional[User]:
    """
    Verify refresh token validity and return user if valid
    """
    try:
        # Decode JWT
        secret_key = getattr(settings, 'REFRESH_SECRET_KEY', settings.REFRESH_SECRET_KEY)
        payload = jwt.decode(
            refresh_token,
            secret_key,
            algorithms=[settings.ALGORITHM]
        )
        
        # Check token type
        if payload.get("type") != "refresh":
            return None
        
        # Get user_id from payload
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Hash the token to check in database
        token_hash = hash_token(refresh_token)


        
        # Find token in database
        db_token = db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash,
            RefreshToken.user_id == uuid.UUID(user_id)
        ).first()


        # Validate token
        if not db_token:
            return None  # Token not found in database
        
        if db_token.is_revoked:
            return None  # Token revoked
        
        if db_token.expires_at < datetime.utcnow():
            return None  # Token expired
            
        # Update last used time
        db_token.last_used_at = datetime.utcnow()
        db.commit()
        
        # Get user
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()

        return user
        
    except jwt.PyJWTError:
        return None  # Invalid JWT
    except Exception as e:
        logger.error(f"Error verifying refresh token: {e}")
        return None
    

# Revoke refresh token
def revoke_refresh_token(db: Session, refresh_token: str) -> bool:
    """
    Revoke a specific refresh token
    """
    try:
        token_hash = hash_token(refresh_token)
        print("$$$$$$$$$$$$$$$$$$$$$$$"*20)
        db_token = db.query(RefreshToken).filter(
            RefreshToken.token_hash == token_hash
        ).first()
        
        if db_token:
            db_token.is_revoked = True
            db.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Error revoking token: {e}")
        return False
    
# Revoke all user tokens
def revoke_all_user_tokens(db: Session, user_id: any) -> int:
    """
    Revoke all refresh tokens for a user
    Returns number of tokens revoked
    """
    result = db.query(RefreshToken).filter(
        RefreshToken.user_id == uuid.UUID(user_id),
        RefreshToken.is_revoked == False
    ).update({"is_revoked": True})
    
    db.commit()
    return result
