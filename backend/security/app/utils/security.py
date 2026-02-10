from datetime import datetime, timedelta
from typing import Optional
import jwt
from backend.security.app.utils.config import settings
from passlib.context import CryptContext
import hashlib
import secrets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password):
#     return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Hash the password first with SHA256, then use bcrypt
    # This avoids the 72-byte limitation
    password_bytes = plain_password.encode('utf-8')
    
    # If password is too long, hash it first
    if len(password_bytes) > 72:
        # Use SHA256 to reduce the size
        sha256_hash = hashlib.sha256(password_bytes).hexdigest()
        return pwd_context.verify(sha256_hash, hashed_password)
    
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    password_bytes = password.encode('utf-8')
    
    # If password is too long, hash it first
    if len(password_bytes) > 72:
        # Use SHA256 to reduce the size
        password = hashlib.sha256(password_bytes).hexdigest()
    
    return pwd_context.hash(password)

def create_access_token(
    user_id: str,
    email: str,
    username: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token with user information
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    payload = {
        "sub": str(user_id),      # User ID
        "email": email,
        "username": username,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
        "jti": generate_secure_token(32)
    }
    
    encoded_jwt = jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt

def create_refresh_token(
    user_id: str,
    email: Optional[str] = None,
    user_name: Optional[str] = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token with user data
    """
    to_encode = {
        "sub": str(user_id),  # User ID
        "type": "refresh",    # Token type
        "jti": generate_secure_token(32)  # Unique token ID
    }
    
    # Add email if provided
    if email:
        to_encode["email"] = email
    if user_name:
        to_encode["username"] = user_name
    
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire})
    
    # Use separate secret for refresh tokens (optional but recommended)
    secret_key = getattr(settings, 'REFRESH_SECRET_KEY', settings.SECRET_KEY)
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=settings.ALGORITHM)
    return encoded_jwt


# Generate a secure random token
def generate_secure_token(length: int = 64) -> str:
    """Generate cryptographically secure random token"""
    return secrets.token_urlsafe(length)

# Hash token for storage
def hash_token(token: str) -> str:
    """Create SHA256 hash of token for secure storage"""
    return hashlib.sha256(token.encode()).hexdigest()


def test_password_hashing():
    plain_password = "superduper"
    hashed_password = get_password_hash(plain_password)
    assert verify_password(plain_password, hashed_password) == True
    assert verify_password("wrongpassword", hashed_password) == False

# test_password_hashing()