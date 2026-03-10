"""
D.I.S.H.A. Backend — app.py  v3.0
Python 3.9 | macOS M2 | MediaPipe 0.10.x Tasks API

KEY CHANGE from v2: Uses MediaPipe BLENDSHAPES (eyeBlinkLeft/Right, jawOpen)
instead of manually computing EAR/MAR from raw landmarks.
Blendshapes are ML-predicted scores — far more accurate and stable than
geometric ratios which break under head rotation, lighting, or glasses.

EAR/MAR are still computed as backup and shown in dashboard.
Head pose uses the facial_transformation_matrixes (4x4) from MediaPipe
instead of solvePnP — more accurate because MediaPipe already solved it.

Install:
    pip install flask flask-sock opencv-python mediapipe numpy

Run:
    python3 app.py
    Open: http://localhost:5001
"""

import cv2
import numpy as np
import json, base64, math, time, os, urllib.request

from flask import Flask, send_from_directory
from flask_sock import Sock

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app  = Flask(__name__)
sock = Sock(app)

# ── Model download ────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("[DISHA] Downloading face landmarker model (~30 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[DISHA] Model ready.")

# ── FaceLandmarker — VIDEO mode + blendshapes + transform matrix ──────
#
# output_face_blendshapes=True  → gives us eyeBlinkLeft, jawOpen, etc.
# output_facial_transformation_matrixes=True → gives accurate head pose
#
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,               # ← critical for accuracy
    output_facial_transformation_matrixes=True, # ← accurate head pose
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)

# ── Blendshape name → index lookup ────────────────────────────────────
# MediaPipe returns 52 blendshapes in a fixed order.
# We build a name→index map once at startup.
BLENDSHAPE_NAMES = [
    "neutral","browDownLeft","browDownRight","browInnerUp",
    "browOuterUpLeft","browOuterUpRight","cheekPuff","cheekSquintLeft",
    "cheekSquintRight","eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft",
    "eyeLookDownRight","eyeLookInLeft","eyeLookInRight","eyeLookOutLeft",
    "eyeLookOutRight","eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft",
    "eyeSquintRight","eyeWideLeft","eyeWideRight","jawForward","jawLeft",
    "jawOpen","jawRight","mouthClose","mouthDimpleLeft","mouthDimpleRight",
    "mouthFrownLeft","mouthFrownRight","mouthFunnel","mouthLeft",
    "mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft",
    "mouthPressRight","mouthPucker","mouthRight","mouthRollLower",
    "mouthRollUpper","mouthShrugLower","mouthShrugUpper","mouthSmileLeft",
    "mouthSmileRight","mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight","noseSneerLeft","noseSneerRight",
]
BS = {name: idx for idx, name in enumerate(BLENDSHAPE_NAMES)}

# ── EAR landmark indices (backup / display) ───────────────────────────
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_POINTS  = [61, 291, 82, 312, 13, 87, 317, 14]

def dist2d(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x*w, lm[i].y*h) for i in indices]
    A = dist2d(pts[1], pts[5])
    B = dist2d(pts[2], pts[4])
    C = dist2d(pts[0], pts[3])
    return (A+B)/(2.0*C) if C > 1e-6 else 0.0

def calc_mar(lm, w, h):
    pts = [(lm[i].x*w, lm[i].y*h) for i in MOUTH_POINTS]
    A = dist2d(pts[2], pts[5])
    B = dist2d(pts[3], pts[6])
    C = dist2d(pts[4], pts[7])
    H = dist2d(pts[0], pts[1])
    return (A+B+C)/(3.0*H) if H > 1e-6 else 0.0

def get_blendshape(blendshapes, name):
    """Safely get a blendshape score by name. Returns 0.0 if unavailable."""
    if not blendshapes:
        return 0.0
    idx = BS.get(name)
    if idx is None or idx >= len(blendshapes):
        return 0.0
    return float(blendshapes[idx].score)

def get_head_pose_from_matrix(transform_matrix):
    """
    Extract pitch, yaw, roll from MediaPipe's 4x4 facial transformation matrix.
    This is MORE accurate than solvePnP because MediaPipe already solved it
    internally using its own calibrated 3D model.
    Returns (pitch, yaw, roll) in degrees.
    """
    try:
        mat = np.array(transform_matrix.data).reshape(4, 4)
        # Extract 3x3 rotation submatrix
        R = mat[:3, :3]
        # Decompose to Euler angles (ZYX convention)
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2( R[2,1], R[2,2]))
            yaw   = math.degrees(math.atan2(-R[2,0], sy))
            roll  = math.degrees(math.atan2( R[1,0], R[0,0]))
        else:
            pitch = math.degrees(math.atan2(-R[1,2], R[1,1]))
            yaw   = math.degrees(math.atan2(-R[2,0], sy))
            roll  = 0.0
        return (
            round(max(-90.0, min(90.0, pitch)), 1),
            round(max(-90.0, min(90.0, yaw)),   1),
            round(max(-90.0, min(90.0, roll)),  1),
        )
    except Exception:
        return 0.0, 0.0, 0.0


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'disha-app.html')

@app.route('/health')
def health():
    return {'status': 'ok', 'mediapipe': mp.__version__}


# ── WebSocket ─────────────────────────────────────────────────────────
@sock.route('/ws')
def websocket(ws):
    """
    Streams JSON payloads to browser at ~25-30fps.

    Payload fields:
        faces       int     0 or 1
        blink_l     float   eyeBlinkLeft  blendshape  (0=open, 1=closed) ← PRIMARY
        blink_r     float   eyeBlinkRight blendshape
        blink_avg   float   avg of both
        jaw_open    float   jawOpen blendshape (0=closed, 1=fully open)  ← PRIMARY
        ear         float   geometric EAR avg (backup/display)
        ear_l       float   left eye EAR
        ear_r       float   right eye EAR
        mar         float   geometric MAR (backup/display)
        pitch       float   degrees, from transform matrix
        yaw         float   degrees
        roll        float   degrees
        conf        float   face presence confidence
        width/height int
        frame       str     base64 JPEG
        ts          float   unix timestamp
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        ws.send(json.dumps({"error": "Camera could not be opened"}))
        return

    frame_ts_ms = 0
    # Thresholds for drawing overlay (visual only — actual detection in frontend)
    BLINK_DRAW_THRESH = 0.4
    JAW_DRAW_THRESH   = 0.5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_ts_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            payload = {
                "faces": 0, "blink_l": 0.0, "blink_r": 0.0, "blink_avg": 0.0,
                "jaw_open": 0.0, "ear": 0.0, "ear_l": 0.0, "ear_r": 0.0,
                "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "conf": 0.0, "width": w, "height": h, "ts": time.time(),
            }

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                bs = result.face_blendshapes[0] if result.face_blendshapes else []

                # ── PRIMARY: Blendshape scores ────────────────────────────
                blink_l   = get_blendshape(bs, "eyeBlinkLeft")
                blink_r   = get_blendshape(bs, "eyeBlinkRight")
                blink_avg = (blink_l + blink_r) / 2.0
                jaw_open  = get_blendshape(bs, "jawOpen")

                # ── BACKUP: Geometric EAR / MAR ───────────────────────────
                ear_l = calc_ear(lm, LEFT_EYE_EAR,  w, h)
                ear_r = calc_ear(lm, RIGHT_EYE_EAR, w, h)
                ear   = (ear_l + ear_r) / 2.0
                mar   = calc_mar(lm, w, h)

                # ── Head pose from transform matrix ───────────────────────
                pitch, yaw, roll = 0.0, 0.0, 0.0
                if result.facial_transformation_matrixes:
                    pitch, yaw, roll = get_head_pose_from_matrix(
                        result.facial_transformation_matrixes[0]
                    )

                # ── Confidence ────────────────────────────────────────────
                conf_raw = getattr(lm[1], 'visibility', None)
                conf = float(conf_raw) if conf_raw is not None else 0.9

                payload.update({
                    "faces":     1,
                    "blink_l":   round(blink_l,   3),
                    "blink_r":   round(blink_r,   3),
                    "blink_avg": round(blink_avg, 3),
                    "jaw_open":  round(jaw_open,  3),
                    "ear":       round(ear,   4),
                    "ear_l":     round(ear_l, 4),
                    "ear_r":     round(ear_r, 4),
                    "mar":       round(mar,   4),
                    "pitch":     pitch,
                    "yaw":       yaw,
                    "roll":      roll,
                    "conf":      round(conf,  3),
                })

                # ── Draw overlay on frame ─────────────────────────────────
                eyes_closed = blink_avg > BLINK_DRAW_THRESH
                mouth_open  = jaw_open  > JAW_DRAW_THRESH

                eye_color   = (0, 60, 255)  if eyes_closed else (0, 255, 140)
                mouth_color = (0, 140, 255) if mouth_open  else (255, 200, 0)

                for idx_set, col in [(LEFT_EYE_EAR, eye_color), (RIGHT_EYE_EAR, eye_color)]:
                    pts = np.array(
                        [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx_set], np.int32
                    )
                    cv2.polylines(frame, [pts], True, col, 1, cv2.LINE_AA)

                mouth_pts = np.array(
                    [(int(lm[i].x*w), int(lm[i].y*h)) for i in MOUTH_POINTS], np.int32
                )
                cv2.polylines(frame, [mouth_pts], True, mouth_color, 1, cv2.LINE_AA)

                # Bounding box
                xs = [lm[i].x for i in range(min(468, len(lm)))]
                ys = [lm[i].y for i in range(min(468, len(lm)))]
                x1 = max(0, int(min(xs)*w)-10)
                y1 = max(0, int(min(ys)*h)-10)
                x2 = min(w, int(max(xs)*w)+10)
                y2 = min(h, int(max(ys)*h)+10)
                box_col = (0, 60, 255) if eyes_closed else (0, 212, 255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), box_col, 1)
                cv2.putText(
                    frame,
                    f"BL:{blink_avg:.2f} JW:{jaw_open:.2f} Y:{yaw:.0f} P:{pitch:.0f}",
                    (x1, max(y1-8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1, cv2.LINE_AA
                )

            # ── Encode & send ─────────────────────────────────────────
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            payload["frame"] = base64.b64encode(buf).decode('utf-8')

            try:
                ws.send(json.dumps(payload))
            except Exception:
                break

    finally:
        cap.release()
        print("[DISHA] Camera released.")


if __name__ == '__main__':
    print("=" * 55)
    print(f"  D.I.S.H.A. v3.0  —  MediaPipe {mp.__version__}")
    print(f"  Detection: BLENDSHAPES (eyeBlink + jawOpen)")
    print(f"  Head pose: facial_transformation_matrixes")
    print(f"  Server: http://localhost:5001")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)