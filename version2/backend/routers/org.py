# backend/routers/org.py
# Organization (tenant) endpoints:
#   GET   /api/org           → current org info (name, invite code, member count)
#   PATCH /api/org           → rename the organization (admin only)

import logging
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from core.database import get_db
from core.dependencies import get_current_user, require_admin
from models.schemas import OrganizationOut, OrgUpdateRequest

router = APIRouter(prefix="/api/org", tags=["organization"])
logger = logging.getLogger(__name__)


async def _org_out(db, org_id: str) -> OrganizationOut:
    org = await db["organizations"].find_one({"_id": ObjectId(org_id)})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found.")
    member_count = await db["users"].count_documents({"org_id": org_id})
    return OrganizationOut(
        id=str(org["_id"]),
        name=org["name"],
        org_code=org["org_code"],
        created_at=org["created_at"],
        member_count=member_count,
    )


@router.get("", response_model=OrganizationOut)
async def get_my_org(current_user: dict = Depends(get_current_user)):
    """Return the caller's organization (any member may view it)."""
    db = get_db()
    return await _org_out(db, current_user["org_id"])


@router.patch("", response_model=OrganizationOut)
async def rename_org(body: OrgUpdateRequest, admin=Depends(require_admin)):
    """Rename the organization. Admin only; scoped to the admin's own org."""
    db = get_db()
    org_id = admin["org_id"]
    await db["organizations"].update_one(
        {"_id": ObjectId(org_id)},
        {"$set": {"name": body.name.strip()}},
    )
    logger.info("Admin %s renamed org %s to %r", admin["email"], org_id, body.name)
    return await _org_out(db, org_id)