"""
D.I.S.H.A. Logging Backend — main.py
FastAPI + PostgreSQL (asyncpg)

Install:
    pip install fastapi uvicorn asyncpg passlib[bcrypt] python-jose[cryptography] python-dotenv

Setup PostgreSQL:
    createdb disha_db
    (or set DATABASE_URL in .env)

Run:
    uvicorn main:app --reload --port 8000

API Docs (auto-generated): http://localhost:8000/docs
"""

import os, json
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
DATABASE_URL = "postgresql://admin:secret@localhost:5433/disha_db"
SECRET_KEY="wdnef2e9239e03ej3irn239dfb"
ALGORITHM    = "HS256"
TOKEN_EXPIRE_DAYS = 7

pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ── DB Pool ──────────────────────────────────────────────────────────
pool: asyncpg.Pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    await init_db()
    yield
    await pool.close()

app = FastAPI(title="D.I.S.H.A. Logging API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ───────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    username    TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'user',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_login  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS sessions (
    id              SERIAL PRIMARY KEY,
    user_id         INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    duration_sec    INT,
    total_frames    INT DEFAULT 0,
    alert_count     INT DEFAULT 0,
    yawn_count      INT DEFAULT 0,
    drowsy_events   INT DEFAULT 0,
    phone_events    INT DEFAULT 0,
    distract_events INT DEFAULT 0,
    max_risk        INT DEFAULT 0,
    avg_risk        FLOAT DEFAULT 0,
    avg_ear         FLOAT DEFAULT 0,
    avg_perclos     FLOAT DEFAULT 0,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    session_id  INT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id     INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type  TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ear         FLOAT,
    mar         FLOAT,
    risk_score  INT,
    perclos     FLOAT,
    yaw         FLOAT,
    pitch       FLOAT,
    details     JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user   ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start  ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_user     ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(timestamp DESC);
"""

async def init_db():
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA)
        # Default admin
        exists = await conn.fetchval("SELECT id FROM users WHERE role='admin' LIMIT 1")
        if not exists:
            hashed = pwd_ctx.hash("admin123")
            await conn.execute(
                "INSERT INTO users (username, email, password, role) VALUES ($1,$2,$3,$4)"
                " ON CONFLICT DO NOTHING",
                "admin", "admin@disha.local", hashed, "admin"
            )
            print("[DISHA] Default admin created: admin / admin123")

# ── Auth helpers ──────────────────────────────────────────────────────
def make_token(user_id: int, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "role": role, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(creds.credentials)
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id, username, email, role FROM users WHERE id=$1",
            int(payload["sub"])
        )
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

async def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def now_utc():
    return datetime.now(timezone.utc)

# ── Pydantic models ───────────────────────────────────────────────────
class RegisterIn(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def min_length(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

class LoginIn(BaseModel):
    username: str
    password: str

class SessionEndIn(BaseModel):
    total_frames:    int   = 0
    alert_count:     int   = 0
    yawn_count:      int   = 0
    drowsy_events:   int   = 0
    phone_events:    int   = 0
    distract_events: int   = 0
    max_risk:        int   = 0
    avg_risk:        float = 0.0
    avg_ear:         float = 0.0
    avg_perclos:     float = 0.0
    notes:           str   = ""

class EventIn(BaseModel):
    session_id:  int
    event_type:  str
    severity:    str = "info"
    timestamp:   Optional[datetime] = None
    ear:         Optional[float] = None
    mar:         Optional[float] = None
    risk_score:  Optional[int]   = None
    perclos:     Optional[float] = None
    yaw:         Optional[float] = None
    pitch:       Optional[float] = None
    details:     dict = {}

class BatchEventsIn(BaseModel):
    events: List[EventIn]

class RoleIn(BaseModel):
    role: str

# ════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════════════

@app.post("/api/auth/register", status_code=201)
async def register(data: RegisterIn):
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT id FROM users WHERE username=$1 OR email=$2",
            data.username, data.email.lower()
        )
        if exists:
            raise HTTPException(409, "Username or email already taken")
        hashed = pwd_ctx.hash(data.password)
        user = await conn.fetchrow(
            "INSERT INTO users (username, email, password) VALUES ($1,$2,$3) RETURNING id,username,email,role,created_at",
            data.username, data.email.lower(), hashed
        )
    u = dict(user)
    return {"user": u, "token": make_token(u["id"], u["role"])}

@app.post("/api/auth/login")
async def login(data: LoginIn):
    async with pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE username=$1", data.username)
        if not user or not pwd_ctx.verify(data.password, user["password"]):
            raise HTTPException(401, "Invalid username or password")
        await conn.execute("UPDATE users SET last_login=NOW() WHERE id=$1", user["id"])
    u = dict(user)
    return {
        "token": make_token(u["id"], u["role"]),
        "user": {k: u[k] for k in ("id","username","email","role","created_at","last_login")}
    }

@app.get("/api/auth/me")
async def me(user=Depends(get_current_user)):
    return user

# ════════════════════════════════════════════════════════════════════
#  SESSION ROUTES
# ════════════════════════════════════════════════════════════════════

@app.post("/api/sessions/start", status_code=201)
async def start_session(user=Depends(get_current_user)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO sessions (user_id, started_at) VALUES ($1, NOW()) RETURNING *",
            user["id"]
        )
    return dict(row)

@app.post("/api/sessions/{sid}/end")
async def end_session(sid: int, data: SessionEndIn, user=Depends(get_current_user)):
    async with pool.acquire() as conn:
        sess = await conn.fetchrow("SELECT * FROM sessions WHERE id=$1", sid)
        if not sess:
            raise HTTPException(404, "Session not found")
        if sess["user_id"] != user["id"] and user["role"] != "admin":
            raise HTTPException(403, "Forbidden")
        dur = int((now_utc() - sess["started_at"]).total_seconds())
        row = await conn.fetchrow("""
            UPDATE sessions SET
                ended_at=NOW(), duration_sec=$1,
                total_frames=$2, alert_count=$3, yawn_count=$4,
                drowsy_events=$5, phone_events=$6, distract_events=$7,
                max_risk=$8, avg_risk=$9, avg_ear=$10, avg_perclos=$11, notes=$12
            WHERE id=$13 RETURNING *
        """, dur, data.total_frames, data.alert_count, data.yawn_count,
            data.drowsy_events, data.phone_events, data.distract_events,
            data.max_risk, data.avg_risk, data.avg_ear, data.avg_perclos,
            data.notes, sid)
    return dict(row)

@app.get("/api/sessions")
async def list_sessions(page: int=1, limit: int=20, user=Depends(get_current_user)):
    limit = min(limit, 50)
    offset = (page - 1) * limit
    async with pool.acquire() as conn:
        if user["role"] == "admin":
            rows = await conn.fetch("""
                SELECT s.*, u.username FROM sessions s
                JOIN users u ON u.id=s.user_id
                ORDER BY s.started_at DESC LIMIT $1 OFFSET $2
            """, limit, offset)
            total = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        else:
            rows = await conn.fetch("""
                SELECT * FROM sessions WHERE user_id=$1
                ORDER BY started_at DESC LIMIT $2 OFFSET $3
            """, user["id"], limit, offset)
            total = await conn.fetchval("SELECT COUNT(*) FROM sessions WHERE user_id=$1", user["id"])
    return {"sessions": [dict(r) for r in rows], "total": total, "page": page, "limit": limit}

@app.get("/api/sessions/{sid}")
async def get_session(sid: int, user=Depends(get_current_user)):
    async with pool.acquire() as conn:
        sess = await conn.fetchrow(
            "SELECT s.*, u.username FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.id=$1", sid
        )
        if not sess:
            raise HTTPException(404, "Session not found")
        s = dict(sess)
        if s["user_id"] != user["id"] and user["role"] != "admin":
            raise HTTPException(403, "Forbidden")
        events = await conn.fetch("SELECT * FROM events WHERE session_id=$1 ORDER BY timestamp ASC", sid)
    return {"session": s, "events": [dict(e) for e in events]}

@app.delete("/api/sessions/{sid}")
async def delete_session(sid: int, _=Depends(require_admin)):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM sessions WHERE id=$1", sid)
    return {"deleted": sid}

# ════════════════════════════════════════════════════════════════════
#  EVENT ROUTES
# ════════════════════════════════════════════════════════════════════

@app.post("/api/events", status_code=201)
async def log_event(data: EventIn, user=Depends(get_current_user)):
    ts = data.timestamp or now_utc()
    async with pool.acquire() as conn:
        sess = await conn.fetchrow("SELECT user_id FROM sessions WHERE id=$1", data.session_id)
        if not sess or (sess["user_id"] != user["id"] and user["role"] != "admin"):
            raise HTTPException(403, "Forbidden or session not found")
        row = await conn.fetchrow("""
            INSERT INTO events
                (session_id, user_id, event_type, severity, timestamp,
                 ear, mar, risk_score, perclos, yaw, pitch, details)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING id
        """, data.session_id, user["id"], data.event_type, data.severity, ts,
            data.ear, data.mar, data.risk_score, data.perclos,
            data.yaw, data.pitch, json.dumps(data.details))
    return {"id": row["id"]}

@app.post("/api/events/batch", status_code=201)
async def log_events_batch(payload: BatchEventsIn, user=Depends(get_current_user)):
    async with pool.acquire() as conn:
        rows = [
            (e.session_id, user["id"], e.event_type, e.severity,
             e.timestamp or now_utc(),
             e.ear, e.mar, e.risk_score, e.perclos, e.yaw, e.pitch,
             json.dumps(e.details))
            for e in payload.events[:500]
        ]
        await conn.executemany("""
            INSERT INTO events
                (session_id, user_id, event_type, severity, timestamp,
                 ear, mar, risk_score, perclos, yaw, pitch, details)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        """, rows)
    return {"inserted": len(rows)}

# ════════════════════════════════════════════════════════════════════
#  USER ROUTES  (admin)
# ════════════════════════════════════════════════════════════════════

@app.get("/api/users")
async def list_users(_=Depends(require_admin)):
    async with pool.acquire() as conn:
        users = await conn.fetch(
            "SELECT id,username,email,role,created_at,last_login FROM users ORDER BY created_at DESC"
        )
        result = []
        for u in users:
            d = dict(u)
            d["session_count"] = await conn.fetchval(
                "SELECT COUNT(*) FROM sessions WHERE user_id=$1", d["id"]
            )
            result.append(d)
    return result

@app.get("/api/users/{uid}")
async def get_user(uid: int, _=Depends(require_admin)):
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT id,username,email,role,created_at,last_login FROM users WHERE id=$1", uid
        )
        if not user:
            raise HTTPException(404, "User not found")
        sessions = await conn.fetch(
            "SELECT * FROM sessions WHERE user_id=$1 ORDER BY started_at DESC", uid
        )
    u = dict(user)
    u["sessions"] = [dict(s) for s in sessions]
    return u

@app.patch("/api/users/{uid}/role")
async def set_role(uid: int, data: RoleIn, current=Depends(require_admin)):
    if data.role not in ("user", "admin"):
        raise HTTPException(400, "role must be 'user' or 'admin'")
    if uid == current["id"]:
        raise HTTPException(400, "Cannot change your own role")
    async with pool.acquire() as conn:
        await conn.execute("UPDATE users SET role=$1 WHERE id=$2", data.role, uid)
    return {"id": uid, "role": data.role}

@app.delete("/api/users/{uid}")
async def delete_user(uid: int, current=Depends(require_admin)):
    if uid == current["id"]:
        raise HTTPException(400, "Cannot delete yourself")
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM users WHERE id=$1", uid)
    return {"deleted": uid}

# ════════════════════════════════════════════════════════════════════
#  STATS
# ════════════════════════════════════════════════════════════════════

@app.get("/api/stats")
async def stats(user=Depends(get_current_user)):
    async with pool.acquire() as conn:
        if user["role"] == "admin":
            s = await conn.fetchrow("""
                SELECT
                    COUNT(DISTINCT u.id)                AS total_users,
                    COUNT(DISTINCT s.id)                AS total_sessions,
                    COUNT(e.id)                         AS total_events,
                    COALESCE(SUM(s.duration_sec),0)     AS total_drive_sec,
                    COALESCE(SUM(s.alert_count),0)      AS total_alerts,
                    COALESCE(SUM(s.yawn_count),0)       AS total_yawns,
                    COALESCE(AVG(s.avg_risk),0)         AS global_avg_risk,
                    COALESCE(AVG(s.avg_ear),0)          AS global_avg_ear
                FROM users u
                LEFT JOIN sessions s ON s.user_id=u.id
                LEFT JOIN events   e ON e.session_id=s.id
            """)
            recent = await conn.fetch("""
                SELECT s.id, s.started_at, s.ended_at, s.duration_sec,
                       s.alert_count, s.avg_risk, s.max_risk, u.username
                FROM sessions s JOIN users u ON u.id=s.user_id
                ORDER BY s.started_at DESC LIMIT 10
            """)
        else:
            s = await conn.fetchrow("""
                SELECT
                    COUNT(DISTINCT s.id)            AS total_sessions,
                    COUNT(e.id)                     AS total_events,
                    COALESCE(SUM(s.duration_sec),0) AS total_drive_sec,
                    COALESCE(SUM(s.alert_count),0)  AS total_alerts,
                    COALESCE(SUM(s.yawn_count),0)   AS total_yawns,
                    COALESCE(AVG(s.avg_risk),0)     AS avg_risk,
                    COALESCE(AVG(s.avg_ear),0)      AS avg_ear,
                    COALESCE(MAX(s.max_risk),0)     AS worst_risk
                FROM sessions s
                LEFT JOIN events e ON e.session_id=s.id
                WHERE s.user_id=$1
            """, user["id"])
            recent = await conn.fetch("""
                SELECT id, started_at, ended_at, duration_sec,
                       alert_count, yawn_count, avg_risk, max_risk
                FROM sessions WHERE user_id=$1
                ORDER BY started_at DESC LIMIT 10
            """, user["id"])
    result = dict(s)
    result["recent_sessions"] = [dict(r) for r in recent]
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "service": "D.I.S.H.A. Logging API v1.0"}