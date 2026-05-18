# backend/routers/sessions.py
# Monitoring session lifecycle:
#   POST   /api/sessions/           → start session
#   PATCH  /api/sessions/{id}/end   → end session
#   GET    /api/sessions/           → list own sessions (user) or all (admin)
#   GET    /api/sessions/{id}       → get single session

import logging
from datetime import datetime, timezone
from typing import List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from core.database import get_db
from core.dependencies import get_current_user, require_admin
from models.schemas import SessionCreateRequest, SessionEndRequest, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


def _format_session(doc: dict) -> SessionOut:
    """Convert a MongoDB session document to the SessionOut schema."""
    return SessionOut(
        id=str(doc["_id"]),
        user_id=str(doc["user_id"]),
        driver_name=doc.get("driver_name"),
        started_at=doc["started_at"],
        ended_at=doc.get("ended_at"),
        duration_seconds=doc.get("duration_seconds"),
        total_alerts=doc.get("total_alerts", 0),
        max_risk_score=doc.get("max_risk_score", 0.0),
        notes=doc.get("notes"),
    )


@router.post("/", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def start_session(
    body: SessionCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new monitoring session for the logged-in user."""
    db = get_db()
    doc = {
        "user_id": ObjectId(current_user["sub"]),
        "driver_name": body.driver_name or current_user.get("name", "Driver"),
        "started_at": datetime.now(timezone.utc),
        "ended_at": None,
        "duration_seconds": None,
        "total_alerts": 0,
        "max_risk_score": 0.0,
    }
    result = await db["sessions"].insert_one(doc)
    doc["_id"] = result.inserted_id
    logger.info("Session started: %s by user %s", result.inserted_id, current_user["sub"])
    return _format_session(doc)


@router.patch("/{session_id}/end", response_model=SessionOut)
async def end_session(
    session_id: str,
    body: Optional[SessionEndRequest] = None,
    current_user: dict = Depends(get_current_user),
):
    """Mark a session as ended and calculate its duration."""
    db = get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    session = await db["sessions"].find_one({"_id": oid})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Users can only end their own sessions; admins can end any
    if (
        current_user["role"] != "admin"
        and str(session["user_id"]) != current_user["sub"]
    ):
        raise HTTPException(status_code=403, detail="Not authorised.")

    if session.get("ended_at"):
        raise HTTPException(status_code=409, detail="Session already ended.")

    ended_at = datetime.now(timezone.utc)
    duration = (ended_at - session["started_at"]).total_seconds()

    # Count total alerts for this session
    total_alerts = await db["events"].count_documents({"session_id": session_id})

    # Find highest risk score
    pipeline = [
        {"$match": {"session_id": session_id}},
        {"$group": {"_id": None, "max_risk": {"$max": "$risk_score"}}},
    ]
    agg = await db["events"].aggregate(pipeline).to_list(1)
    max_risk = agg[0]["max_risk"] if agg else 0.0

    await db["sessions"].update_one(
        {"_id": oid},
        {"$set": {
            "ended_at": ended_at,
            "duration_seconds": duration,
            "total_alerts": total_alerts,
            "max_risk_score": max_risk,
            "notes": (body.notes if body and body.notes else None),
        }},
    )
    session.update(
        ended_at=ended_at,
        duration_seconds=duration,
        total_alerts=total_alerts,
        max_risk_score=max_risk,
    )
    return _format_session(session)


@router.get("/", response_model=List[SessionOut])
async def list_sessions(
    current_user: dict = Depends(get_current_user),
    limit: int = 50,
):
    """
    Users see only their own sessions.
    Admins see all sessions.
    """
    db = get_db()
    query = {} if current_user["role"] == "admin" else {"user_id": ObjectId(current_user["sub"])}
    cursor = db["sessions"].find(query).sort("started_at", -1).limit(limit)
    docs = await cursor.to_list(limit)
    return [_format_session(d) for d in docs]


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get a single session by ID."""
    db = get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    session = await db["sessions"].find_one({"_id": oid})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if (
        current_user["role"] != "admin"
        and str(session["user_id"]) != current_user["sub"]
    ):
        raise HTTPException(status_code=403, detail="Not authorised.")

    return _format_session(session)