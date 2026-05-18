# backend/routers/auth.py
# Handles user registration and login.
# POST /api/auth/register  →  create account
# POST /api/auth/login     →  return JWT

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from core.database import get_db
from core.security import hash_password, verify_password, create_access_token
from models.schemas import RegisterRequest, LoginRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    """
    Create a new user account.
    The very first registered user is automatically made admin.
    """
    db = get_db()
    users_col = db["users"]

    # Check email uniqueness
    if await users_col.find_one({"email": body.email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # First user ever → admin; everyone else → user
    is_first_user = (await users_col.count_documents({})) == 0
    role = "admin" if is_first_user else "user"

    # Build user document
    user_doc = {
        "name": body.name,
        "email": body.email,
        "password_hash": hash_password(body.password),
        "role": role,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }

    result = await users_col.insert_one(user_doc)
    user_id = str(result.inserted_id)

    logger.info("New user registered: %s (%s) role=%s", body.email, user_id, role)

    # Return token immediately so the user is logged in after registration
    token = create_access_token(
        {"sub": user_id, "email": body.email, "role": role, "name": body.name}
    )
    return TokenResponse(access_token=token, role=role, name=body.name, user_id=user_id)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Verify credentials and return a JWT access token."""
    db = get_db()
    user = await db["users"].find_one({"email": body.email})

    # Intentionally same error message for wrong email OR password (security)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact an administrator.",
        )

    user_id = str(user["_id"])
    token = create_access_token(
        {
            "sub": user_id,
            "email": user["email"],
            "role": user["role"],
            "name": user["name"],
        }
    )
    logger.info("User logged in: %s", body.email)
    return TokenResponse(
        access_token=token,
        role=user["role"],
        name=user["name"],
        user_id=user_id,
    )
