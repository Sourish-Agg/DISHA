# backend/models/schemas.py
# Pydantic v2 models that define the shape of request bodies, responses,
# and MongoDB documents throughout the app.

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import re

# Simple email regex — accepts test@test, user@domain.com etc.
# Intentionally lenient for development/prototype use.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+$")


# ── Helper ────────────────────────────────────────────────────────────────────

class PyObjectId(str):
    """String alias for MongoDB ObjectId so Pydantic serialises it cleanly."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    email: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=6, max_length=128)
    # Optional invite code. If provided, the user joins that organization as a
    # regular user. If omitted, a brand-new organization is created and this
    # user becomes its admin.
    org_code: Optional[str] = Field(None, min_length=4, max_length=12)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("Enter a valid email address (e.g. user@example.com)")
        return v

    @field_validator("org_code")
    @classmethod
    def normalize_org_code(cls, v):
        return v.strip().upper() if v else v


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=120)
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str
    user_id: str
    org_id: str
    org_name: str
    org_code: str


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANIZATION (tenant)
# ═══════════════════════════════════════════════════════════════════════════════

class OrganizationOut(BaseModel):
    id: str
    name: str
    org_code: str
    created_at: datetime
    member_count: int = 0


class OrgUpdateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)


# ═══════════════════════════════════════════════════════════════════════════════
# USER
# ═══════════════════════════════════════════════════════════════════════════════

class UserOut(BaseModel):
    """Safe user representation (no password)."""
    id: str
    name: str
    email: str
    role: Literal["user", "admin"]
    org_id: str
    created_at: datetime
    is_active: bool


class UserUpdateRequest(BaseModel):
    """Admin can update name, role, or active status."""
    name: Optional[str] = Field(None, min_length=2, max_length=60)
    role: Optional[Literal["user", "admin"]] = None
    is_active: Optional[bool] = None


# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING SESSION
# ═══════════════════════════════════════════════════════════════════════════════

class SessionCreateRequest(BaseModel):
    """Frontend sends this to start a monitoring session."""
    driver_name: Optional[str] = None     # optional label for the session

class SessionEndRequest(BaseModel):
    """Optionally send notes when ending a session."""
    notes: Optional[str] = Field(None, max_length=300)


class SessionOut(BaseModel):
    id: str
    user_id: str
    driver_name: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: Optional[float]
    total_alerts: int
    max_risk_score: float
    notes: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION EVENT
# ═══════════════════════════════════════════════════════════════════════════════

class EventCreateRequest(BaseModel):
    """
    Posted by the frontend whenever a detection alert fires.
    All numeric values come from the JS detection engine.
    """
    session_id: str
    event_type: Literal[
        "drowsy_eyes",      # Eyes closed too long (EAR threshold)
        "yawning",          # MAR threshold crossed
        "phone_detected",   # COCO-SSD phone detection
        "head_distraction", # Yaw/pitch outside safe range
        "high_risk",        # Fused risk score > 70%
    ]
    # Raw sensor values at the moment of the event
    ear: Optional[float] = None
    mar: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    perclos: Optional[float] = None
    risk_score: float = 0.0
    timestamp: Optional[datetime] = None  # filled server-side if absent


class EventOut(BaseModel):
    id: str
    session_id: str
    event_type: str
    ear: Optional[float]
    mar: Optional[float]
    yaw: Optional[float]
    pitch: Optional[float]
    perclos: Optional[float]
    risk_score: float
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN STATS
# ═══════════════════════════════════════════════════════════════════════════════

class AdminStats(BaseModel):
    total_users: int
    total_sessions: int
    total_events: int
    events_by_type: dict
    recent_sessions: list