# backend/routers/users.py
# Authenticated user profile endpoints:
#   GET   /api/users/me           → get own profile
#   PATCH /api/users/me/password  → change password

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.database import get_db
from core.dependencies import get_current_user
from core.security import hash_password, verify_password
from models.schemas import UserOut
from bson import ObjectId

router = APIRouter(prefix="/api/users", tags=["users"])
logger = logging.getLogger(__name__)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)


@router.get("/me", response_model=UserOut)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    db = get_db()
    doc = await db["users"].find_one({"_id": ObjectId(current_user["sub"])})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserOut(
        id=str(doc["_id"]),
        name=doc["name"],
        email=doc["email"],
        role=doc["role"],
        org_id=doc["org_id"],
        created_at=doc["created_at"],
        is_active=doc.get("is_active", True),
    )


@router.patch("/me/password", status_code=204)
async def change_password(
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
):
    """Allow the current user to change their own password."""
    db = get_db()
    doc = await db["users"].find_one({"_id": ObjectId(current_user["sub"])})
    if not doc or not verify_password(body.current_password, doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    await db["users"].update_one(
        {"_id": ObjectId(current_user["sub"])},
        {"$set": {"password_hash": hash_password(body.new_password)}},
    )
    logger.info("User %s changed password.", current_user["email"])