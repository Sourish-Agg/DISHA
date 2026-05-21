# backend/routers/events.py
# Detection event logging and retrieval.
#   POST  /api/events/              → log a new detection event
#   GET   /api/events/{session_id}  → get all events for a session

import logging
from datetime import datetime, timezone
from typing import List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from core.database import get_db
from core.dependencies import get_current_user
from models.schemas import EventCreateRequest, EventOut

router = APIRouter(prefix="/api/events", tags=["events"])
logger = logging.getLogger(__name__)


def _format_event(doc: dict) -> EventOut:
    return EventOut(
        id=str(doc["_id"]),
        session_id=doc["session_id"],
        event_type=doc["event_type"],
        ear=doc.get("ear"),
        mar=doc.get("mar"),
        yaw=doc.get("yaw"),
        pitch=doc.get("pitch"),
        perclos=doc.get("perclos"),
        risk_score=doc.get("risk_score", 0.0),
        timestamp=doc["timestamp"],
    )


@router.post("/", response_model=EventOut, status_code=201)
async def log_event(
    body: EventCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Log a single detection alert event.
    The frontend calls this whenever a threshold is crossed.
    """
    db = get_db()

    # Verify session exists and belongs to this user (or user is admin)
    try:
        session_oid = ObjectId(body.session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    session = await db["sessions"].find_one({"_id": session_oid})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Tenant guard: session must be in the caller's org; users only their own.
    if session.get("org_id") != current_user["org_id"]:
        raise HTTPException(status_code=403, detail="Not authorised.")
    if (
        current_user["role"] != "admin"
        and str(session["user_id"]) != current_user["sub"]
    ):
        raise HTTPException(status_code=403, detail="Not authorised.")

    doc = {
        "org_id": current_user["org_id"],
        "session_id": body.session_id,
        "event_type": body.event_type,
        "ear": body.ear,
        "mar": body.mar,
        "yaw": body.yaw,
        "pitch": body.pitch,
        "perclos": body.perclos,
        "risk_score": body.risk_score,
        # Use provided timestamp or default to now
        "timestamp": body.timestamp or datetime.now(timezone.utc),
    }

    result = await db["events"].insert_one(doc)
    doc["_id"] = result.inserted_id
    logger.debug(
        "Event logged: type=%s session=%s risk=%.1f",
        body.event_type, body.session_id, body.risk_score,
    )
    return _format_event(doc)


@router.get("/{session_id}", response_model=List[EventOut])
async def get_events_for_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    limit: int = 200,
):
    """Return all detection events for a specific session."""
    db = get_db()

    # Ownership check
    try:
        session_oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    session = await db["sessions"].find_one({"_id": session_oid})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.get("org_id") != current_user["org_id"]:
        raise HTTPException(status_code=403, detail="Not authorised.")
    if (
        current_user["role"] != "admin"
        and str(session["user_id"]) != current_user["sub"]
    ):
        raise HTTPException(status_code=403, detail="Not authorised.")

    cursor = db["events"].find({"session_id": session_id}).sort("timestamp", 1).limit(limit)
    docs = await cursor.to_list(limit)
    return [_format_event(d) for d in docs]