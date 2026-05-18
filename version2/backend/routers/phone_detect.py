# backend/routers/phone_detect.py
# Server-side YOLOv8 phone detection endpoint.
# The frontend sends a base64 JPEG frame every ~5th frame.
# This runs YOLOv8n.pt locally (same approach as version1/app.py)
# and returns whether a phone was detected and its confidence.
#
# Install: pip install ultralytics
# Model:   yolov8n.pt is auto-downloaded by ultralytics on first run (~6 MB)

import base64
import logging
import io
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from core.dependencies import get_current_user

router = APIRouter(prefix="/api/phone", tags=["phone_detection"])
logger = logging.getLogger(__name__)

# COCO class 67 = cell phone
PHONE_CLASS_ID = 67
CONFIDENCE_THRESHOLD = 0.40

# Load YOLOv8n once at module import — non-fatal if ultralytics not installed
_yolo = None

def _load_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo
    try:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8n.pt")   # auto-downloads on first run
        logger.info("YOLOv8n loaded for server-side phone detection.")
    except Exception as e:
        logger.warning("YOLOv8 not available — phone detection disabled. (%s)", e)
        _yolo = None
    return _yolo


class PhoneDetectRequest(BaseModel):
    # Base64-encoded JPEG frame from the browser
    frame_b64: str


class PhoneDetectResponse(BaseModel):
    detected:   bool
    confidence: float
    available:  bool   # False if YOLOv8 not installed


@router.on_event("startup")  # pre-warm the model
async def _warmup():
    _load_yolo()


@router.post("/detect", response_model=PhoneDetectResponse)
async def detect_phone(
    body: PhoneDetectRequest,
    _user: dict = Depends(get_current_user),
):
    """
    Accepts a base64-encoded JPEG frame, runs YOLOv8n inference,
    returns phone detection result.
    Called by the frontend every 5th frame during monitoring.
    """
    model = _load_yolo()
    if model is None:
        return PhoneDetectResponse(detected=False, confidence=0.0, available=False)

    try:
        # Decode base64 → numpy image
        img_bytes = base64.b64decode(body.frame_b64)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)

        import cv2
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")

        # Run YOLOv8 — only detect phone class for speed
        results    = model(frame, verbose=False, classes=[PHONE_CLASS_ID],
                           conf=CONFIDENCE_THRESHOLD)
        best_conf  = 0.0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == PHONE_CLASS_ID:
                    best_conf = max(best_conf, float(box.conf[0]))

        return PhoneDetectResponse(
            detected   = best_conf >= CONFIDENCE_THRESHOLD,
            confidence = round(best_conf, 3),
            available  = True,
        )

    except Exception as e:
        logger.warning("Phone detection inference error: %s", e)
        return PhoneDetectResponse(detected=False, confidence=0.0, available=True)