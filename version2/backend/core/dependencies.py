# backend/core/dependencies.py
# Reusable FastAPI dependencies for authentication and role-based access control.

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.security import decode_token
from core.database import get_db

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    Dependency: Decode Bearer token → return user payload.
    Raises 401 if token is missing or invalid.
    """
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload  # {"sub": user_id, "email": ..., "role": ..., "name": ..., "org_id": ...}


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Dependency: Same as get_current_user but also checks role == 'admin'.
    Raises 403 if the user is not an admin.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user
