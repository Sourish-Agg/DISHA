# backend/routers/analytics.py
# Analytics endpoints:
#   GET /api/analytics/trend          → alerts per day (last 7 days) — admin
#   GET /api/analytics/export/csv     → download session events as CSV — admin
#   GET /api/analytics/summary/{sid}  → session summary for post-session card

import csv
import io
from datetime import datetime, timezone, timedelta
from typing import List

from bson import ObjectId
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.database import get_db
from core.dependencies import get_current_user, require_admin

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ── Alert trend — past 7 days ─────────────────────────────────────────────────
@router.get("/trend")
async def alert_trend(admin=Depends(require_admin)):
    """Returns alert counts per day for the last 7 days, grouped by type (this org)."""
    db  = get_db()
    org_id = admin["org_id"]
    now = datetime.now(timezone.utc)

    result = {}
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        label     = day_start.strftime("%d %b")

        pipeline = [
            {"$match": {"org_id": org_id, "timestamp": {"$gte": day_start, "$lt": day_end}}},
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
        ]
        agg = await db["events"].aggregate(pipeline).to_list(20)
        result[label] = {item["_id"]: item["count"] for item in agg}

    return result


# ── CSV export ────────────────────────────────────────────────────────────────
@router.get("/export/csv")
async def export_csv(admin=Depends(require_admin)):
    """
    Download this organization's events as a CSV file.
    Streams the response so large datasets don't blow memory.
    """
    db     = get_db()
    events = await db["events"].find({"org_id": admin["org_id"]}).sort("timestamp", 1).to_list(50000)

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "session_id", "event_type",
                         "ear", "mar", "yaw", "pitch", "perclos", "risk_score"])
        for e in events:
            writer.writerow([
                e.get("timestamp", "").isoformat() if hasattr(e.get("timestamp", ""), "isoformat") else e.get("timestamp", ""),
                e.get("session_id", ""),
                e.get("event_type", ""),
                e.get("ear", ""),
                e.get("mar", ""),
                e.get("yaw", ""),
                e.get("pitch", ""),
                e.get("perclos", ""),
                e.get("risk_score", ""),
            ])
        yield buf.getvalue()

    filename = f"disha_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Session summary ───────────────────────────────────────────────────────────
@router.get("/summary/{session_id}")
async def session_summary(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Full summary for the post-session card:
    - Duration, total alerts, max risk, alert breakdown
    - Risk scores over time (sampled every 10 events for the chart)
    - Peak risk moment timestamp
    """
    db = get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(400, "Invalid session ID")

    session = await db["sessions"].find_one({"_id": oid})
    if not session:
        from fastapi import HTTPException
        raise HTTPException(404, "Session not found")

    # Auth check: same org, and (admin OR owner)
    from fastapi import HTTPException
    if session.get("org_id") != current_user["org_id"]:
        raise HTTPException(403, "Not authorised")
    if (current_user["role"] != "admin"
            and str(session["user_id"]) != current_user["sub"]):
        raise HTTPException(403, "Not authorised")

    events = await db["events"].find(
        {"session_id": session_id}
    ).sort("timestamp", 1).to_list(5000)

    # Alert type breakdown
    breakdown = {}
    for e in events:
        t = e.get("event_type", "unknown")
        breakdown[t] = breakdown.get(t, 0) + 1

    # Risk timeline — sample every N events for chart (max 60 points)
    step = max(1, len(events) // 60)
    timeline = [
        {
            "t": e["timestamp"].isoformat() if hasattr(e.get("timestamp"), "isoformat") else str(e.get("timestamp", "")),
            "r": round(e.get("risk_score", 0)),
        }
        for i, e in enumerate(events) if i % step == 0
    ]

    # Peak risk moment
    peak_event = max(events, key=lambda e: e.get("risk_score", 0), default=None)
    peak = {
        "risk":      round(peak_event.get("risk_score", 0)),
        "type":      peak_event.get("event_type", ""),
        "timestamp": peak_event["timestamp"].isoformat() if peak_event and hasattr(peak_event.get("timestamp"), "isoformat") else None,
    } if peak_event else None

    ended_at = session.get("ended_at")
    return {
        "session_id":       session_id,
        "driver_name":      session.get("driver_name"),
        "started_at":       session["started_at"].isoformat(),
        "ended_at":         ended_at.isoformat() if ended_at else None,
        "duration_seconds": session.get("duration_seconds"),
        "total_alerts":     len(events),
        "max_risk_score":   session.get("max_risk_score", 0),
        "notes":            session.get("notes", ""),
        "breakdown":        breakdown,
        "timeline":         timeline,
        "peak":             peak,
    }   