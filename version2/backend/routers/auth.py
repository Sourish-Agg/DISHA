# backend/routers/auth.py
# Handles user registration and login.
# POST /api/auth/register  →  create account
# POST /api/auth/login     →  return JWT

import logging
import secrets
import string
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status

from core.database import get_db
from core.security import hash_password, verify_password, create_access_token
from models.schemas import RegisterRequest, LoginRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)


def _generate_org_code() -> str:
    """Short, human-friendly invite code, e.g. 'K7QP2M'. Avoids ambiguous chars."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I, O, 0, 1
    return "".join(secrets.choice(alphabet) for _ in range(6))


async def _create_unique_org(db, org_name: str) -> dict:
    """Create a new organization with a guaranteed-unique invite code."""
    for _ in range(10):
        code = _generate_org_code()
        if not await db["organizations"].find_one({"org_code": code}):
            doc = {
                "name": org_name,
                "org_code": code,
                "created_at": datetime.now(timezone.utc),
            }
            result = await db["organizations"].insert_one(doc)
            doc["_id"] = result.inserted_id
            return doc
    raise HTTPException(status_code=500, detail="Could not allocate an organization code. Try again.")


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    """
    Create a new user account.

    Tenancy rules:
      • If org_code is provided → the user JOINS that organization as a regular
        user.
      • If org_code is omitted → a NEW organization is created and this user
        becomes its admin (the first member is always the org admin).
    """
    db = get_db()
    users_col = db["users"]

    # ── Resolve organization (join existing or create new) ───────────────────
    if body.org_code:
        org = await db["organizations"].find_one({"org_code": body.org_code})
        if not org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid organization code. Check the code and try again.",
            )
        # Joining members are regular users; the org already has an admin.
        role = "user"
    else:
        # New organization — name it after the user for now; admin can rename it.
        org = await _create_unique_org(db, f"{body.name}'s Organization")
        role = "admin"

    org_id = str(org["_id"])

    # ── Email must be unique WITHIN this organization ────────────────────────
    if await users_col.find_one({"org_id": org_id, "email": body.email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists in this organization.",
        )

    user_doc = {
        "org_id": org_id,
        "name": body.name,
        "email": body.email,
        "password_hash": hash_password(body.password),
        "role": role,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    result = await users_col.insert_one(user_doc)
    user_id = str(result.inserted_id)

    logger.info("Registered %s (%s) in org %s as %s", body.email, user_id, org_id, role)

    token = create_access_token(
        {"sub": user_id, "email": body.email, "role": role, "name": body.name, "org_id": org_id}
    )
    return TokenResponse(
        access_token=token,
        role=role,
        name=body.name,
        user_id=user_id,
        org_id=org_id,
        org_name=org["name"],
        org_code=org["org_code"],
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """
    Verify credentials and return a JWT access token.

    Because an email can exist in more than one organization, we look up all
    accounts with this email and select the one whose password verifies.
    """
    db = get_db()
    candidates = await db["users"].find({"email": body.email}).to_list(20)

    user = None
    for c in candidates:
        if verify_password(body.password, c["password_hash"]):
            user = c
            break

    # Intentionally same error message for wrong email OR password (security)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact an administrator.",
        )

    org_id = user["org_id"]
    org = await db["organizations"].find_one({"_id": ObjectId(org_id)})
    if not org:
        raise HTTPException(status_code=500, detail="Organization not found for this account.")

    user_id = str(user["_id"])
    token = create_access_token(
        {
            "sub": user_id,
            "email": user["email"],
            "role": user["role"],
            "name": user["name"],
            "org_id": org_id,
        }
    )
    logger.info("User logged in: %s (org %s)", body.email, org_id)
    return TokenResponse(
        access_token=token,
        role=user["role"],
        name=user["name"],
        user_id=user_id,
        org_id=org_id,
        org_name=org["name"],
        org_code=org["org_code"],
    )