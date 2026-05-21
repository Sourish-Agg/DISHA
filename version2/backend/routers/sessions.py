# backend/routers/sessions.py
import logging
from datetime import datetime, timezone
from typing import List, Optional
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from core.database import get_db
from core.dependencies import get_current_user
from models.schemas import SessionCreateRequest, SessionEndRequest, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


def _authorize_session(session: dict, user: dict) -> None:
    """
    Tenant + ownership guard.
      • The session must belong to the caller's organization.
      • A regular user may only touch their own sessions; an admin may touch
        any session WITHIN their own organization (never another tenant's).
    Raises 403 on any violation. (404 is handled by the caller.)
    """
    if session.get("org_id") != user["org_id"]:
        raise HTTPException(403, "Not authorised.")
    if user["role"] != "admin" and str(session["user_id"]) != user["sub"]:
        raise HTTPException(403, "Not authorised.")


def _fmt(doc: dict) -> SessionOut:
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


@router.post("/", response_model=SessionOut, status_code=201)
async def start_session(
    body: SessionCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    doc = {
        "org_id":          current_user["org_id"],
        "user_id":         ObjectId(current_user["sub"]),
        "driver_name":     body.driver_name or current_user.get("name", "Driver"),
        "started_at":      datetime.now(timezone.utc),
        "ended_at":        None,
        "duration_seconds": None,
        "total_alerts":    0,
        "max_risk_score":  0.0,
        "notes":           None,
    }
    result     = await db["sessions"].insert_one(doc)
    doc["_id"] = result.inserted_id
    logger.info("Session started: %s by %s", result.inserted_id, current_user["sub"])
    return _fmt(doc)


@router.patch("/{session_id}/end", response_model=SessionOut)
async def end_session(
    session_id: str,
    body: Optional[SessionEndRequest] = None,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID.")

    session = await db["sessions"].find_one({"_id": oid})
    if not session:
        raise HTTPException(404, "Session not found.")
    _authorize_session(session, current_user)

    # If already ended, just return current state instead of erroring —
    # this prevents the frontend from getting a 409 and losing the summary
    if session.get("ended_at"):
        logger.info("Session %s already ended — returning existing record.", session_id)
        return _fmt(session)

    ended_at = datetime.now(timezone.utc)
    # MongoDB may return started_at as naive (no tzinfo). Treat it as UTC
    # so the subtraction doesn't raise "can't subtract offset-naive and
    # offset-aware datetimes".
    started_at = session["started_at"]
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    duration = (ended_at - started_at).total_seconds()

    total_alerts = await db["events"].count_documents({"session_id": session_id})
    agg = await db["events"].aggregate([
        {"$match":  {"session_id": session_id}},
        {"$group":  {"_id": None, "max_risk": {"$max": "$risk_score"}}},
    ]).to_list(1)
    max_risk = agg[0]["max_risk"] if agg else 0.0

    notes = (body.notes if body and body.notes else None)

    await db["sessions"].update_one(
        {"_id": oid},
        {"$set": {
            "ended_at":        ended_at,
            "duration_seconds": duration,
            "total_alerts":    total_alerts,
            "max_risk_score":  max_risk,
            "notes":           notes,
        }},
    )

    # Build updated doc for response
    session["ended_at"]         = ended_at
    session["duration_seconds"] = duration
    session["total_alerts"]     = total_alerts
    session["max_risk_score"]   = max_risk
    session["notes"]            = notes

    logger.info("Session ended: %s duration=%.0fs alerts=%d", session_id, duration, total_alerts)
    return _fmt(session)


@router.get("/", response_model=List[SessionOut])
async def list_sessions(
    current_user: dict = Depends(get_current_user),
    limit: int = 50,
):
    db    = get_db()
    # Admin → all sessions in their organization. User → only their own.
    if current_user["role"] == "admin":
        query = {"org_id": current_user["org_id"]}
    else:
        query = {"org_id": current_user["org_id"], "user_id": ObjectId(current_user["sub"])}
    docs  = await db["sessions"].find(query).sort("started_at", -1).limit(limit).to_list(limit)
    return [_fmt(d) for d in docs]


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session ID.")
    session = await db["sessions"].find_one({"_id": oid})
    if not session:
        raise HTTPException(404, "Session not found.")
    _authorize_session(session, current_user)
    return _fmt(session)