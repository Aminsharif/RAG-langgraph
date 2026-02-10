# src/security/auth.py
import os
from typing import Any, Dict, List

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError, PyJWTError
from langgraph_sdk import Auth

# Use your project's sync DB factory or dependency helper
from backend.security.app.utils.dependencies import get_db  # generator-style dependency
from backend.security.app.crud.user import get_user_by_username

JWT_SECRET = os.environ.get("SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.environ.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

AUTH_EXCEPTION = Auth.exceptions.HTTPException(
    status_code=401,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

AUTH_EXPIRED_EXCEPTION = Auth.exceptions.HTTPException(
    status_code=401,
    detail="Access token expired",
    headers={
        "WWW-Authenticate": 'Bearer error="invalid_token", error_description="The access token expired"'
    },
)

auth = Auth()


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Authenticate JWT and look up user using the DB without FastAPI Depends."""
    print(authorization)
    if not authorization:
        raise AUTH_EXCEPTION

    try:
        scheme, token = authorization.strip().split(" ", 1)
        if scheme.lower() != "bearer":
            raise AUTH_EXCEPTION

        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[ALGORITHM],
            options={"verify_aud": False},
            leeway=ACCESS_TOKEN_EXPIRE_MINUTES,
        )
        username = payload.get("username")
        if not username:
            raise AUTH_EXCEPTION

        # Get a DB session from the dependency generator directly (no Depends)
        db_gen = get_db()           # returns a generator that yields a Session
        db = next(db_gen)           # get the Session instance
        try:
            user = get_user_by_username(db, username)
        finally:
            # close the generator to run its cleanup and close the session
            db_gen.close()

      

        if not user:
            raise AUTH_EXCEPTION

        role = payload.get("role")
        permissions: List[str] = [role] if isinstance(role, str) else (list(role) if isinstance(role, list) else ["user"])

        return {
            "identity": str(user.id),
            "display_name": getattr(user, "username", None),
            "is_authenticated": True,
            "permissions": permissions,
        }

    except ExpiredSignatureError as e:
        print("Token expired:", e)
        raise AUTH_EXPIRED_EXCEPTION from e
    except (InvalidTokenError, PyJWTError, ValueError) as e:
        print("Invalid token:", e)
        raise AUTH_EXCEPTION from e
    except Exception as e:
        print("Unexpected auth error:", e)
        raise AUTH_EXCEPTION