# backend/main.py
# FastAPI application entry point.
# Run with: uvicorn main:app --reload  (from the backend/ directory)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import connect_db, close_db
from routers import auth, sessions, events, admin, users, analytics

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.APP_ENV == "development" else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("disha")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_db()
    logger.info("D.I.S.H.A. API started  (env=%s)", settings.APP_ENV)
    yield
    # Shutdown
    await close_db()
    logger.info("D.I.S.H.A. API stopped.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="D.I.S.H.A. API",
    description="Driver Insight & Safety Heuristics Assistant — Backend API",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production
    docs_url="/docs" if settings.APP_ENV == "development" else None,
    redoc_url=None,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Development: allow all origins so VS Code Live Server (any port) never causes 400s.
# Production:  restrict to the explicit list in .env CORS_ORIGINS.
# NOTE: allow_credentials must be False when allow_origins=["*"] (browser security rule).
_cors_origins = ["*"] if settings.APP_ENV == "development" else settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False if settings.APP_ENV == "development" else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(events.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(analytics.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "env": settings.APP_ENV}