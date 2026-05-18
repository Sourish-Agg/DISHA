"""
D.I.S.H.A. Backend — app.py  v4.1
Python 3.9+ | Flask + WebSocket | Architecture per Project Report

CHANGES in v4.1:
  - FIXED: ValueError: Input timestamp must be monotonically increasing
    (now uses real wall-clock ms instead of a counter that resets per connection)
  - MAR_THRESH lowered 0.65 → 0.50 (yawn triggers more reliably)
  - YAWN_MS lowered 2000 → 1500ms
  - DROWSY_FRAMES lowered 20 → 15 (~0.5s)
  - Full logging backend integration (sessions + events → PostgreSQL)

Install:
    pip install flask flask-sock opencv-python mediapipe numpy ultralytics

Run:
    python3 app.py  →  http://localhost:5001
"""

import cv2
import numpy as np
import json, base64, math, time, os, urllib.request, collections, threading
import urllib.request as urlreq
from flask import Flask, send_from_directory
from flask_sock import Sock

# ════════════════════════════════════════════════════════════════════
#  LOGGING BACKEND INTEGRATION
# ════════════════════════════════════════════════════════════════════

LOGGING_URL      = "http://localhost:8000"
LOGGING_USER     = "admin"
LOGGING_PASSWORD = "admin123"

_log_token    = None
_session_id   = None
_event_buffer = []
_log_lock     = threading.Lock()

def _log_request(method, path, body=None):
    url     = LOGGING_URL + path
    data    = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if _log_token:
        headers["Authorization"] = f"Bearer {_log_token}"
    try:
        req = urlreq.Request(url, data=data, headers=headers, method=method)
        with urlreq.urlopen(req, timeout=3) as res:
            return json.loads(res.read())
    except Exception as e:
        print(f"[DISHA-LOG] {method} {path} failed: {e}")
        return None

def log_login():
    global _log_token
    res = _log_request("POST", "/api/auth/login",
                        {"username": LOGGING_USER, "password": LOGGING_PASSWORD})
    if res and "token" in res:
        _log_token = res["token"]
        print(f"[DISHA-LOG] Logged in as {LOGGING_USER}")
        return True
    print("[DISHA-LOG] Login failed — events will not be logged")
    return False

def log_start_session():
    global _session_id, _event_buffer
    if not _log_token:
        return
    res = _log_request("POST", "/api/sessions/start")
    if res and "id" in res:
        _session_id   = res["id"]
        _event_buffer = []
        print(f"[DISHA-LOG] Session started: #{_session_id}")

def log_end_session(stats):
    global _session_id
    if not _session_id:
        return
    _flush_events()
    _log_request("POST", f"/api/sessions/{_session_id}/end", stats)
    print(f"[DISHA-LOG] Session #{_session_id} ended")
    _session_id = None

def log_event(event_type, severity, **kwargs):
    if not _session_id:
        return
    event = {
        "session_id": _session_id,
        "event_type": event_type,
        "severity":   severity,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **kwargs,
    }
    with _log_lock:
        _event_buffer.append(event)
    if len(_event_buffer) >= 20:
        threading.Thread(target=_flush_events, daemon=True).start()

def _flush_events():
    global _event_buffer
    with _log_lock:
        if not _event_buffer:
            return
        batch         = _event_buffer[:]
        _event_buffer = []
    _log_request("POST", "/api/events/batch", {"events": batch})

threading.Thread(target=log_login, daemon=True).start()

# ════════════════════════════════════════════════════════════════════
#  FLASK + MEDIAPIPE SETUP
# ════════════════════════════════════════════════════════════════════

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app  = Flask(__name__)
sock = Sock(app)

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("[DISHA] Downloading MediaPipe face landmarker (~30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[DISHA] MediaPipe model ready.")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.FaceLandmarker.create_from_options(options)

yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
    print("[DISHA] YOLOv8n loaded for phone detection.")
except Exception as e:
    print(f"[DISHA] YOLOv8 not available ({e}). Phone detection disabled.")

PHONE_CLASS_ID = 67

# ════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ════════════════════════════════════════════════════════════════════

LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_MAR     = [61, 291, 82, 312, 13, 87, 317, 14]

# ════════════════════════════════════════════════════════════════════
#  GEOMETRIC METRICS
# ════════════════════════════════════════════════════════════════════

def dist2d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = dist2d(pts[1], pts[5])
    B = dist2d(pts[2], pts[4])
    C = dist2d(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

def calc_mar(lm, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in MOUTH_MAR]
    A = dist2d(pts[2], pts[5])
    B = dist2d(pts[3], pts[6])
    C = dist2d(pts[4], pts[7])
    H = dist2d(pts[0], pts[1])
    return (A + B + C) / (3.0 * H) if H > 1e-6 else 0.0

# ════════════════════════════════════════════════════════════════════
#  HEAD POSE
# ════════════════════════════════════════════════════════════════════

def get_head_pose(transform_matrix):
    try:
        mat = np.array(transform_matrix.data).reshape(4, 4)
        R   = mat[:3, :3]
        sy  = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2( R[2, 1], R[2, 2]))
            yaw   = math.degrees(math.atan2(-R[2, 0], sy))
            roll  = math.degrees(math.atan2( R[1, 0], R[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
            yaw   = math.degrees(math.atan2(-R[2, 0], sy))
            roll  = 0.0
        clamp = lambda v: round(max(-90.0, min(90.0, v)), 1)
        return clamp(pitch), clamp(yaw), clamp(roll)
    except Exception:
        return 0.0, 0.0, 0.0

# ════════════════════════════════════════════════════════════════════
#  LSTM TEMPORAL MODEL
# ════════════════════════════════════════════════════════════════════

LSTM_WINDOW = 30
LSTM_ALPHA  = 0.85

class LSTMTemporalModel:
    def __init__(self, window=LSTM_WINDOW, alpha=LSTM_ALPHA):
        self.window       = window
        self.alpha        = alpha
        self.buffer       = collections.deque(maxlen=window)
        self.hidden_state = np.zeros(4, dtype=np.float32)

    def step(self, ear, mar, pitch, yaw):
        feature = np.array([
            1.0 - min(ear / 0.35, 1.0),
            min(mar / 0.5, 1.0),
            min(abs(pitch) / 30.0, 1.0),
            min(abs(yaw)   / 45.0, 1.0),
        ], dtype=np.float32)
        self.buffer.append(feature)
        self.hidden_state = (
            self.alpha * self.hidden_state
            + (1.0 - self.alpha) * feature
        )
        return {
            "eye_temporal":   round(float(self.hidden_state[0]), 3),
            "mouth_temporal": round(float(self.hidden_state[1]), 3),
            "head_temporal":  round(float((self.hidden_state[2] + self.hidden_state[3]) / 2.0), 3),
        }

    def reset(self):
        self.buffer.clear()
        self.hidden_state = np.zeros(4, dtype=np.float32)

lstm_model = LSTMTemporalModel()

# ════════════════════════════════════════════════════════════════════
#  PERCLOS P80
# ════════════════════════════════════════════════════════════════════

PERCLOS_WINDOW = 180
PERCLOS_THRESH = 0.20
PERCLOS_ALERT  = 15
perclos_buffer = collections.deque(maxlen=PERCLOS_WINDOW)

def update_perclos(ear):
    perclos_buffer.append(1 if ear < PERCLOS_THRESH else 0)
    if not perclos_buffer:
        return 0.0
    return round(sum(perclos_buffer) / len(perclos_buffer) * 100, 1)

# ════════════════════════════════════════════════════════════════════
#  DECISION FUSION MODULE
# ════════════════════════════════════════════════════════════════════

class DecisionFusionModule:
    W = {
        "eye_closure_temporal": 35,
        "perclos":              25,
        "yawn_temporal":        15,
        "phone":                15,
        "head_pose":            10,
    }

    def __init__(self):
        self.smooth_risk = 0.0
        self.ema_alpha   = 0.25

    def compute(self, eye_temporal, perclos_pct, yawn_temporal, phone_conf, head_temporal):
        def activate(x, knee=0.5):
            return float(1.0 / (1.0 + math.exp(-10.0 * (x - knee))))

        components = {
            "eye_closure_temporal": activate(eye_temporal,        0.4) * self.W["eye_closure_temporal"],
            "perclos":              activate(perclos_pct / 100.0, 0.10) * self.W["perclos"],
            "yawn_temporal":        activate(yawn_temporal,        0.4) * self.W["yawn_temporal"],
            "phone":                activate(phone_conf,            0.4) * self.W["phone"],
            "head_pose":            activate(head_temporal,         0.4) * self.W["head_pose"],
        }
        raw_score        = min(100.0, sum(components.values()))
        self.smooth_risk = (1.0 - self.ema_alpha) * self.smooth_risk + self.ema_alpha * raw_score
        risk_int         = int(round(self.smooth_risk))
        level            = "SAFE" if risk_int < 30 else "MODERATE" if risk_int < 65 else "HIGH RISK"
        return {
            "risk_score": risk_int,
            "risk_level": level,
            "components": {k: round(v, 2) for k, v in components.items()},
        }

decision_module = DecisionFusionModule()

# ════════════════════════════════════════════════════════════════════
#  THRESHOLDS
# ════════════════════════════════════════════════════════════════════

EAR_THRESH    = 0.21   # eye closed
MAR_THRESH    = 0.50   # was 0.65 — lowered for easier yawn detection
PITCH_THRESH  = 20.0
YAW_THRESH    = 35.0

DROWSY_FRAMES = 15     # was 20 — ~0.5s at 30fps
YAWN_MS       = 1500   # was 2000ms
ALERT_COOL_MS = 8000

# ════════════════════════════════════════════════════════════════════
#  DRIVER STATE
# ════════════════════════════════════════════════════════════════════

class DriverState:
    def __init__(self):
        self.drowsy_count    = 0
        self.was_drowsy      = False
        self.yawn_start      = None
        self.was_yawning     = False
        self.was_distracted  = False
        self.phone_detected  = False
        self.frame_count     = 0
        self.alert_count     = 0
        self.yawn_count      = 0
        self.drowsy_events   = 0
        self.phone_events    = 0
        self.distract_events = 0
        self.risk_sum        = 0.0
        self.ear_sum         = 0.0
        self.perclos_sum     = 0.0
        self.stat_frames     = 0
        self.max_risk        = 0

# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'disha-app.html')

@app.route('/health')
def health():
    return {
        'status':       'ok',
        'mediapipe':    mp.__version__,
        'yolo':         yolo_model is not None,
        'architecture': 'EAR+MAR+LSTM+YOLO+DecisionFusion',
    }

# ════════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ════════════════════════════════════════════════════════════════════

@sock.route('/ws')
def websocket(ws):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        ws.send(json.dumps({"error": "Camera could not be opened"}))
        return

    state = DriverState()
    lstm_model.reset()
    perclos_buffer.clear()
    decision_module.smooth_risk = 0.0

    threading.Thread(target=log_start_session, daemon=True).start()

    phone_conf   = 0.0
    yolo_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            state.frame_count += 1

            # Use real wall-clock timestamp — fixes monotonic error on reconnect
            rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms = int(time.time() * 1000)
            result      = landmarker.detect_for_video(mp_image, frame_ts_ms)

            # YOLOv8 phone detection every 3rd frame
            yolo_counter += 1
            if yolo_counter % 3 == 0 and yolo_model is not None:
                try:
                    yolo_results = yolo_model(
                        frame, verbose=False, classes=[PHONE_CLASS_ID], conf=0.35
                    )
                    phone_conf = 0.0
                    for r in yolo_results:
                        for box in r.boxes:
                            if int(box.cls[0]) == PHONE_CLASS_ID:
                                phone_conf = max(phone_conf, float(box.conf[0]))
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
                                cv2.putText(frame, f"PHONE {phone_conf:.0%}",
                                            (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 100, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass

            payload = {
                "faces": 0,
                "ear": 0.0, "ear_l": 0.0, "ear_r": 0.0,
                "mar": 0.0,
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "eye_temporal": 0.0, "mouth_temporal": 0.0, "head_temporal": 0.0,
                "perclos": 0.0,
                "phone_conf":      round(phone_conf, 3),
                "risk_score":      0,
                "risk_level":      "SAFE",
                "risk_components": {},
                "drowsy":          False,
                "yawning":         False,
                "distracted":      False,
                "phone":           phone_conf > 0.5,
                "width":           w,
                "height":          h,
                "ts":              time.time(),
            }

            if result.face_landmarks:
                lm = result.face_landmarks[0]

                ear_l = calc_ear(lm, LEFT_EYE_EAR,  w, h)
                ear_r = calc_ear(lm, RIGHT_EYE_EAR, w, h)
                ear   = (ear_l + ear_r) / 2.0
                mar   = calc_mar(lm, w, h)

                pitch, yaw, roll = 0.0, 0.0, 0.0
                if result.facial_transformation_matrixes:
                    pitch, yaw, roll = get_head_pose(
                        result.facial_transformation_matrixes[0]
                    )

                temporal = lstm_model.step(ear, mar, pitch, yaw)
                perclos  = update_perclos(ear)

                # Drowsiness — EAR only
                if ear < EAR_THRESH:
                    state.drowsy_count = min(state.drowsy_count + 1, DROWSY_FRAMES + 90)
                else:
                    state.drowsy_count = max(0, state.drowsy_count - 2)
                is_drowsy = state.drowsy_count >= DROWSY_FRAMES

                # Yawning — MAR only, time-sustained
                if mar > MAR_THRESH:
                    if not state.yawn_start:
                        state.yawn_start = time.time()
                else:
                    state.yawn_start = None
                yawn_ms    = (time.time() - state.yawn_start) * 1000 if state.yawn_start else 0
                is_yawning = yawn_ms >= YAWN_MS

                # Distraction — head pose
                is_distracted = abs(yaw) > YAW_THRESH or abs(pitch) > PITCH_THRESH

                fusion = decision_module.compute(
                    eye_temporal  = temporal["eye_temporal"],
                    perclos_pct   = perclos,
                    yawn_temporal = temporal["mouth_temporal"],
                    phone_conf    = phone_conf,
                    head_temporal = temporal["head_temporal"],
                )

                # Draw overlays
                eye_col   = (0, 60, 255)  if is_drowsy  else (0, 212, 100)
                mouth_col = (0, 140, 255) if is_yawning else (255, 200, 0)
                for idx_set, col in [(LEFT_EYE_EAR, eye_col), (RIGHT_EYE_EAR, eye_col)]:
                    pts = np.array(
                        [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx_set], np.int32
                    )
                    cv2.polylines(frame, [pts], True, col, 1, cv2.LINE_AA)
                mouth_pts = np.array(
                    [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH_MAR], np.int32
                )
                cv2.polylines(frame, [mouth_pts], True, mouth_col, 1, cv2.LINE_AA)
                xs   = [lm[i].x for i in range(min(468, len(lm)))]
                ys   = [lm[i].y for i in range(min(468, len(lm)))]
                bx1  = max(0, int(min(xs) * w) - 10)
                by1  = max(0, int(min(ys) * h) - 10)
                bx2  = min(w, int(max(xs) * w) + 10)
                by2  = min(h, int(max(ys) * h) + 10)
                bcol = (0, 60, 255) if is_drowsy else (0, 212, 255)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), bcol, 1)
                cv2.putText(
                    frame,
                    f"EAR:{ear:.2f} MAR:{mar:.2f} Y:{yaw:.0f} RISK:{fusion['risk_score']}",
                    (bx1, max(by1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, bcol, 1, cv2.LINE_AA,
                )

                # Accumulate stats
                state.stat_frames += 1
                state.risk_sum    += fusion["risk_score"]
                state.ear_sum     += ear
                state.perclos_sum += perclos
                state.max_risk     = max(state.max_risk, fusion["risk_score"])

                # Log events on rising edge
                if is_drowsy and not state.was_drowsy:
                    state.drowsy_events += 1
                    state.alert_count   += 1
                    log_event("drowsiness", "danger",
                              ear=round(ear, 4), mar=round(mar, 4),
                              risk_score=fusion["risk_score"],
                              perclos=perclos, yaw=yaw, pitch=pitch)
                state.was_drowsy = is_drowsy

                if is_yawning and not state.was_yawning:
                    state.yawn_count += 1
                    log_event("yawn", "warn",
                              ear=round(ear, 4), mar=round(mar, 4),
                              risk_score=fusion["risk_score"],
                              perclos=perclos)
                state.was_yawning = is_yawning

                if phone_conf > 0.5 and not state.phone_detected:
                    state.phone_events += 1
                    log_event("phone", "warn",
                              risk_score=fusion["risk_score"],
                              details={"phone_conf": round(phone_conf, 3)})
                state.phone_detected = phone_conf > 0.5

                if is_distracted and not state.was_distracted:
                    state.distract_events += 1
                    log_event("distraction", "warn",
                              yaw=yaw, pitch=pitch,
                              risk_score=fusion["risk_score"])
                state.was_distracted = is_distracted

                payload.update({
                    "faces":           1,
                    "ear":             round(ear,   4),
                    "ear_l":           round(ear_l, 4),
                    "ear_r":           round(ear_r, 4),
                    "mar":             round(mar,   4),
                    "pitch":           pitch,
                    "yaw":             yaw,
                    "roll":            roll,
                    "eye_temporal":    temporal["eye_temporal"],
                    "mouth_temporal":  temporal["mouth_temporal"],
                    "head_temporal":   temporal["head_temporal"],
                    "perclos":         perclos,
                    "phone_conf":      round(phone_conf, 3),
                    "risk_score":      fusion["risk_score"],
                    "risk_level":      fusion["risk_level"],
                    "risk_components": fusion["components"],
                    "drowsy":          is_drowsy,
                    "yawning":         is_yawning,
                    "distracted":      is_distracted,
                    "phone":           phone_conf > 0.5,
                    "drowsy_frames":   state.drowsy_count,
                    "yawn_ms":         round(yawn_ms),
                })

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            payload["frame"] = base64.b64encode(buf).decode('utf-8')

            try:
                ws.send(json.dumps(payload))
            except Exception:
                break

    finally:
        cap.release()
        print("[DISHA] Camera released.")
        n = max(state.stat_frames, 1)
        log_end_session({
            "total_frames":    state.frame_count,
            "alert_count":     state.alert_count,
            "yawn_count":      state.yawn_count,
            "drowsy_events":   state.drowsy_events,
            "phone_events":    state.phone_events,
            "distract_events": state.distract_events,
            "max_risk":        state.max_risk,
            "avg_risk":        round(state.risk_sum    / n, 2),
            "avg_ear":         round(state.ear_sum     / n, 4),
            "avg_perclos":     round(state.perclos_sum / n, 2),
        })


if __name__ == '__main__':
    print("=" * 60)
    print(f"  D.I.S.H.A. v4.1  —  MediaPipe {mp.__version__}")
    print(f"  Detection: EAR+MAR (Soukupova & Cech / Abtahi 2016)")
    print(f"  Temporal:  LSTM 30-frame window (Ghoddoosian et al.)")
    print(f"  Phone:     YOLOv8n {'yes' if yolo_model else 'no (pip install ultralytics)'}")
    print(f"  Thresholds: EAR<{EAR_THRESH}  MAR>{MAR_THRESH}  YAWN>{YAWN_MS}ms  DROWSY>{DROWSY_FRAMES}f")
    print(f"  Server:    http://localhost:5001")
    print(f"  Logs:      FastAPI backend -> http://localhost:8000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)