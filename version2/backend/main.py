# backend/main.py
# FastAPI application entrypoint.
# Run from inside the backend/ directory:
#     uvicorn main:app --reload --port 8000
#
# Wires together all routers, manages the MongoDB connection lifecycle,
# applies CORS, and pre-warms the YOLOv8 phone-detection model on startup.

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import connect_db, close_db

# Routers
from routers import auth, users, sessions, events, analytics, admin, phone_detect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("disha")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    await connect_db()

    # Pre-warm the YOLOv8 model so the first /api/phone/detect call isn't slow.
    # Non-fatal: if ultralytics isn't installed, phone detection just reports
    # available=False and the app keeps running.
    try:
        phone_detect._load_yolo()
    except Exception as e:  # pragma: no cover
        logger.warning("YOLO warmup skipped: %s", e)

    logger.info("DISHA backend started (env=%s).", settings.APP_ENV)
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await close_db()
    logger.info("DISHA backend shut down.")


app = FastAPI(
    title="D.I.S.H.A. API",
    description="Driver Insight & Safety Heuristics Assistant — backend API.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(events.router)
app.include_router(analytics.router)
app.include_router(admin.router)
app.include_router(phone_detect.router)


@app.get("/", tags=["health"])
async def root():
    """Simple health check."""
    return {"status": "ok", "service": "disha-api", "env": settings.APP_ENV}