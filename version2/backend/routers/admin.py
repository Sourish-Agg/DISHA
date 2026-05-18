# backend/routers/admin.py
# Admin-only endpoints:
#   GET    /api/admin/stats          → platform overview stats
#   GET    /api/admin/users          → list all users
#   PATCH  /api/admin/users/{id}     → update user role / active status
#   DELETE /api/admin/users/{id}     → delete a user and all their data

import logging
from typing import List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from core.database import get_db
from core.dependencies import require_admin
from models.schemas import AdminStats, UserOut, UserUpdateRequest

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)


def _format_user(doc: dict) -> UserOut:
    return UserOut(
        id=str(doc["_id"]),
        name=doc["name"],
        email=doc["email"],
        role=doc["role"],
        created_at=doc["created_at"],
        is_active=doc.get("is_active", True),
    )


@router.get("/stats", response_model=AdminStats)
async def get_stats(_admin=Depends(require_admin)):
    """Platform-wide statistics for the admin dashboard."""
    db = get_db()

    total_users = await db["users"].count_documents({})
    total_sessions = await db["sessions"].count_documents({})
    total_events = await db["events"].count_documents({})

    # Events grouped by type
    pipeline = [
        {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    type_agg = await db["events"].aggregate(pipeline).to_list(20)
    events_by_type = {item["_id"]: item["count"] for item in type_agg}

    # 10 most recent sessions with user names
    recent_pipeline = [
        {"$sort": {"started_at": -1}},
        {"$limit": 10},
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "user",
            }
        },
        {"$unwind": {"path": "$user", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "session_id": {"$toString": "$_id"},
                "user_name": "$user.name",
                "user_email": "$user.email",
                "driver_name": 1,
                "started_at": 1,
                "ended_at": 1,
                "total_alerts": 1,
                "max_risk_score": 1,
            }
        },
    ]
    recent_sessions = await db["sessions"].aggregate(recent_pipeline).to_list(10)

    # MongoDB ObjectIds are not JSON-serialisable — convert them
    for s in recent_sessions:
        s.pop("_id", None)

    return AdminStats(
        total_users=total_users,
        total_sessions=total_sessions,
        total_events=total_events,
        events_by_type=events_by_type,
        recent_sessions=recent_sessions,
    )


@router.get("/users", response_model=List[UserOut])
async def list_users(_admin=Depends(require_admin)):
    """List every registered user."""
    db = get_db()
    docs = await db["users"].find({}).sort("created_at", -1).to_list(500)
    return [_format_user(d) for d in docs]


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: str,
    body: UserUpdateRequest,
    admin=Depends(require_admin),
):
    """Update a user's name, role, or active status."""
    db = get_db()
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")

    # Prevent admin from accidentally demoting themselves
    if user_id == admin["sub"] and body.role == "user":
        raise HTTPException(status_code=400, detail="Cannot demote your own admin account.")

    update_fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update.")

    result = await db["users"].find_one_and_update(
        {"_id": oid},
        {"$set": update_fields},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="User not found.")

    logger.info("Admin %s updated user %s: %s", admin["email"], user_id, update_fields)
    return _format_user(result)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, admin=Depends(require_admin)):
    """Delete a user and all their sessions and events."""
    db = get_db()
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")

    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")

    user = await db["users"].find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Cascade delete: events → sessions → user
    sessions = await db["sessions"].find({"user_id": oid}).to_list(1000)
    session_ids = [str(s["_id"]) for s in sessions]

    if session_ids:
        await db["events"].delete_many({"session_id": {"$in": session_ids}})
    await db["sessions"].delete_many({"user_id": oid})
    await db["users"].delete_one({"_id": oid})

    logger.info("Admin %s deleted user %s and %d sessions", admin["email"], user_id, len(session_ids))

@router.post("/cleanup-sessions")
async def cleanup_stuck_sessions(_admin=Depends(require_admin)):
    """
    Mark all sessions with no ended_at as ended.
    Useful for fixing sessions that got stuck due to browser close / network error.
    Sessions are ended with duration calculated from started_at to now.
    """
    from datetime import datetime, timezone
    db  = get_db()
    now = datetime.now(timezone.utc)

    stuck = await db["sessions"].find({"ended_at": None}).to_list(1000)
    fixed = 0
    for s in stuck:
        sid      = str(s["_id"])
        duration = (now - s["started_at"]).total_seconds()
        total_alerts = await db["events"].count_documents({"session_id": sid})
        agg = await db["events"].aggregate([
            {"$match": {"session_id": sid}},
            {"$group": {"_id": None, "max_risk": {"$max": "$risk_score"}}},
        ]).to_list(1)
        max_risk = agg[0]["max_risk"] if agg else 0.0

        await db["sessions"].update_one(
            {"_id": s["_id"]},
            {"$set": {
                "ended_at":        now,
                "duration_seconds": duration,
                "total_alerts":    total_alerts,
                "max_risk_score":  max_risk,
                "notes":           "Auto-closed by admin cleanup",
            }}
        )
        fixed += 1

    logger.info("Admin cleanup: fixed %d stuck sessions", fixed)
    return {"fixed": fixed, "message": f"Closed {fixed} stuck session(s)."}