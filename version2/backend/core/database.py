# backend/core/database.py
# Async MongoDB connection using Motor.
# Call connect_db() on startup and close_db() on shutdown.

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import settings

logger = logging.getLogger(__name__)

# Module-level client — shared across all requests
_client: AsyncIOMotorClient | None = None


async def connect_db() -> None:
    """Open MongoDB connection and create indexes on startup."""
    global _client
    _client = AsyncIOMotorClient(settings.MONGO_URL)

    db = _client[settings.MONGO_DB_NAME]

    # ── Indexes ──────────────────────────────────────────────────────────────
    # users: unique email
    await db["users"].create_index("email", unique=True)

    # sessions: fast lookup by user
    await db["sessions"].create_index("user_id")
    await db["sessions"].create_index("started_at")

    # events: fast lookup by session and timestamp
    await db["events"].create_index("session_id")
    await db["events"].create_index("timestamp")

    logger.info("MongoDB connected → %s / %s", settings.MONGO_URL, settings.MONGO_DB_NAME)


async def close_db() -> None:
    """Close MongoDB connection on shutdown."""
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    """Return the active database. Call after connect_db()."""
    if _client is None:
        raise RuntimeError("Database not initialised. Call connect_db() first.")
    return _client[settings.MONGO_DB_NAME]
