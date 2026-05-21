// frontend/js/detection.js — v6
// ═══════════════════════════════════════════════════════════════════════════════
// DETECTION ENGINE — rebuilt from version1/app.py proven logic
//
// ROOT CAUSE OF BLINK FALSE POSITIVES (now fixed):
//   v5 used CONSEC_CLOSED_FRAMES=48 — so high that PERCLOS accumulated
//   during the wait, causing cue2 (PERCLOS) to fire even on normal blinking.
//   The fix: use version1's DROWSY_FRAMES=15 (~0.5s) with RISING EDGE logic.
//
// HOW VERSION1 PREVENTED BLINK FALSE POSITIVES:
//   A blink is ~6 frames. drowsy_count increments by 1 per closed frame,
//   decrements by 2 per open frame. After a 6-frame blink:
//     - drowsy_count peaks at 6 (well below threshold of 15)
//     - 3 open frames later it's back to 0
//   Eyes genuinely closed for 0.5s (15 frames) → alert fires ONCE on rising edge.
//   8-second cooldown prevents re-triggering.
//
// ARCHITECTURE (matches version1 exactly):
//   EAR threshold:    0.21   (Soukupová & Čech 2016)
//   DROWSY_FRAMES:    15     (~0.5s at 30fps — version1 value)
//   PERCLOS window:   180    (6s — version1 value)
//   PERCLOS thresh:   20%    (version1 P80 standard)
//   PERCLOS alert:    15%    (NHTSA 1994)
//   LSTM alpha:       0.85   (version1 value)
//   Yawn threshold:   0.50   (version1 value)
//   Yawn duration:    1500ms (version1 YAWN_MS)
//   Phone confidence: 0.65   (raised from version1's 0.35 — filters remotes)
//   Alert cooldown:   8000ms (version1 ALERT_COOL_MS)
//
// RISING EDGE RULE (key to no false positives):
//   Alert only fires when state transitions FALSE→TRUE, not while TRUE.
//   Same as version1's: if is_drowsy and not state.was_drowsy → alert.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Thresholds (version1 values) ─────────────────────────────────────────────
const EAR_THRESH      = 0.21;   // version1 exact value
const MAR_THRESH      = 0.50;   // version1 exact value
const DROWSY_FRAMES   = 15;     // ~0.5s at 30fps — version1 exact value
const PERCLOS_WINDOW  = 180;    // 6s rolling window — version1 exact value
const PERCLOS_THRESH  = 0.20;   // EAR < this = eye >80% closed — version1
const PERCLOS_ALERT   = 0.15;   // 15% triggers drowsy via PERCLOS path
const YAWN_MS         = 1500;   // version1 exact value
const LSTM_ALPHA      = 0.85;   // version1 exact value
const PHONE_CONF      = 0.65;   // raised — filters AC remotes, glasses cases
const YAW_THRESH      = 35;
const PITCH_THRESH    = 20;
const HEAD_DIST_FRAMES= 50;
const ALERT_COOL_MS   = 8000;   // version1 exact value

// Calibration (personalised EAR offset on top of base threshold)
const CALIBRATION_FRAMES = 90;

// ── Landmark indices ──────────────────────────────────────────────────────────
const LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]; // version1 LEFT_EYE_EAR
const RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]; // version1 RIGHT_EYE_EAR
const MOUTH_IDX     = [61, 291, 82, 312, 13, 87, 317, 14]; // version1 MOUTH_MAR

// ── State ─────────────────────────────────────────────────────────────────────
// Per-session state — reset on resetDetectionState()
let drowsyCount    = 0;         // version1: state.drowsy_count
let wasDrowsy      = false;     // version1: state.was_drowsy — RISING EDGE
let yawnStart      = null;      // version1: state.yawn_start (timestamp ms)
let wasYawning     = false;
let wasDistracted  = false;
let perclosBuffer  = [];        // version1: perclos_buffer (deque maxlen=180)
let headDistrCount = 0;

// LSTM hidden state — version1: lstm_model.hidden_state (4 components)
let hiddenState = [0, 0, 0, 0]; // [eye, mouth, pitch_norm, yaw_norm]

// Smoothed sensor values
let smoothEAR   = 0.30;
let smoothMAR   = 0.00;
let smoothYaw   = 0;
let smoothPitch = 0;

// Alert cooldowns (separate per type — version1 uses one global ALERT_COOL_MS)
let alertCooldowns = {};

// Personalised EAR offset from calibration
let calibSamples     = [];
let calibDone        = false;
let earOffset        = 0;       // added to EAR_THRESH after calibration

// Luminance
let lowLightMode     = false;
let _luxCanvas = null, _luxCtx = null;

// ── Distance helper ───────────────────────────────────────────────────────────
function dist2d(a, b) {
  // Use x/y only (version1 uses pixel coords via x*w, y*h — same ratio in normalised)
  return Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2);
}

// ── EAR — version1 calc_ear logic ────────────────────────────────────────────
// Formula: EAR = (A + B) / (2*C)
// A = dist(p[1], p[5]), B = dist(p[2], p[4]), C = dist(p[0], p[3])
function calcEAR(lm, indices) {
  const p = indices.map(i => lm[i]);
  const A = dist2d(p[1], p[5]);
  const B = dist2d(p[2], p[4]);
  const C = dist2d(p[0], p[3]);
  return C > 0.0001 ? (A + B) / (2.0 * C) : 0;
}

// ── MAR — version1 calc_mar logic ────────────────────────────────────────────
// pts[0]=61 pts[1]=291 pts[2]=82 pts[3]=312 pts[4]=13 pts[5]=87 pts[6]=317 pts[7]=14
// MAR = (A + B + C) / (3 * H)
// A = dist(pts[2], pts[5]), B = dist(pts[3], pts[6]), C = dist(pts[4], pts[7])
// H = dist(pts[0], pts[1])
function calcMAR(lm) {
  const pts = MOUTH_IDX.map(i => lm[i]);
  const A = dist2d(pts[2], pts[5]);
  const B = dist2d(pts[3], pts[6]);
  const C = dist2d(pts[4], pts[7]);
  const H = dist2d(pts[0], pts[1]);
  return H > 0.0001 ? (A + B + C) / (3.0 * H) : 0;
}

// ── Head pose ─────────────────────────────────────────────────────────────────
function calcHeadPose(lm) {
  const nose     = lm[1];
  const leftEar  = lm[234];
  const rightEar = lm[454];
  const chin     = lm[152];
  const forehead = lm[10];
  const earMidX  = (leftEar.x + rightEar.x) / 2;
  const earSpan  = Math.abs(leftEar.x - rightEar.x);
  const yaw      = earSpan > 0.001 ? ((nose.x - earMidX) / earSpan) * 90 : 0;
  const vertMid  = (forehead.y + chin.y) / 2;
  const vertSpan = Math.abs(forehead.y - chin.y);
  const pitch    = vertSpan > 0.001 ? -((nose.y - vertMid) / vertSpan) * 60 : 0;
  return { yaw, pitch };
}

// ── PERCLOS — version1 update_perclos logic ───────────────────────────────────
// Appends 1 if ear < PERCLOS_THRESH else 0. Window = 180 frames.
// Returns percentage (0-100) like version1.
function updatePerclos(ear) {
  perclosBuffer.push(ear < PERCLOS_THRESH ? 1 : 0);
  if (perclosBuffer.length > PERCLOS_WINDOW) perclosBuffer.shift();
  if (!perclosBuffer.length) return 0;
  return (perclosBuffer.reduce((a,b)=>a+b,0) / perclosBuffer.length) * 100;
}

// ── LSTM temporal — version1 LSTMTemporalModel.step() ────────────────────────
// feature = [1-ear/0.35, mar/0.5, |pitch|/30, |yaw|/45]
// hidden  = alpha * hidden + (1-alpha) * feature
function updateLSTM(ear, mar, pitch, yaw) {
  const feature = [
    1.0 - Math.min(ear / 0.35, 1.0),
    Math.min(mar / 0.5, 1.0),
    Math.min(Math.abs(pitch) / 30.0, 1.0),
    Math.min(Math.abs(yaw)   / 45.0, 1.0),
  ];
  hiddenState = hiddenState.map((h, i) =>
    LSTM_ALPHA * h + (1 - LSTM_ALPHA) * feature[i]
  );
  return {
    eyeTemporal:   +hiddenState[0].toFixed(3),
    mouthTemporal: +hiddenState[1].toFixed(3),
    headTemporal:  +((hiddenState[2] + hiddenState[3]) / 2).toFixed(3),
  };
}

// ── Alert cooldown ────────────────────────────────────────────────────────────
function canAlert(type) {
  const now = Date.now();
  if (!alertCooldowns[type] || now - alertCooldowns[type] > ALERT_COOL_MS) {
    alertCooldowns[type] = now;
    return true;
  }
  return false;
}

// ── Risk fusion — version1 DecisionFusionModule.compute() ────────────────────
// Weights: eye 35%, perclos 25%, yawn 15%, phone 15%, head 10%
// Uses sigmoid activation: 1 / (1 + exp(-10*(x-knee)))
function sigmoid(x, knee = 0.5) {
  return 1.0 / (1.0 + Math.exp(-10.0 * (x - knee)));
}

function computeRisk({ eyeTemporal, perclosPct, mouthTemporal, phoneConf, headTemporal }) {
  const score =
    sigmoid(eyeTemporal,        0.4) * 35 +
    sigmoid(perclosPct / 100.0, 0.10) * 25 +
    sigmoid(mouthTemporal,      0.4) * 15 +
    sigmoid(phoneConf,          0.4) * 15 +
    sigmoid(headTemporal,       0.4) * 10;
  return Math.min(100, Math.round(score));
}

// ── Luminance (awareness badge only) ─────────────────────────────────────────
function measureLuminance(videoEl) {
  if (!_luxCanvas) {
    _luxCanvas = document.createElement("canvas");
    _luxCanvas.width = 40; _luxCanvas.height = 30;
    _luxCtx = _luxCanvas.getContext("2d", { willReadFrequently: true });
  }
  try {
    _luxCtx.drawImage(videoEl, 0, 0, 40, 30);
    const px = _luxCtx.getImageData(0, 0, 40, 30).data;
    let s = 0;
    for (let i = 0; i < px.length; i += 4)
      s += 0.2126*px[i] + 0.7152*px[i+1] + 0.0722*px[i+2];
    return s / (40 * 30);
  } catch(_) { return 128; }
}

// ── Calibration ───────────────────────────────────────────────────────────────
// Measures personal resting EAR to set an offset.
// e.g. if your resting EAR is 0.26 but default thresh is 0.21,
// offset = 0 (threshold stays — you have normal eyes).
// If resting EAR is 0.19 (small eyes), offset = -0.02 (threshold lowers).
function calibrationStep(lm) {
  if (calibDone) return { done: true, progress: 100 };
  if (!lm || lm.length < 478) return { done: false, progress: 0 };

  const earL = calcEAR(lm, LEFT_EYE_IDX);
  const earR = calcEAR(lm, RIGHT_EYE_IDX);
  calibSamples.push((earL + earR) / 2);

  const progress = Math.round(calibSamples.length / CALIBRATION_FRAMES * 100);

  if (calibSamples.length >= CALIBRATION_FRAMES) {
    const sorted   = [...calibSamples].sort((a,b) => a-b);
    const medianEAR = sorted[Math.floor(sorted.length * 0.5)];
    // Compute offset: if user's median open EAR is below 0.26, lower threshold
    // Never raise threshold (don't punish users with wide eyes)
    const naturalThresh = medianEAR * 0.78;  // 78% of median open = closed
    earOffset = Math.min(0, naturalThresh - EAR_THRESH);
    earOffset = Math.max(-0.06, earOffset);   // cap adjustment at -0.06
    calibDone = true;
    console.log(`[DISHA] Calibration: medianEAR=${medianEAR.toFixed(3)}, earOffset=${earOffset.toFixed(3)}, effectiveThresh=${(EAR_THRESH+earOffset).toFixed(3)}`);
  }
  return { done: calibDone, progress };
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetDetectionState() {
  drowsyCount    = 0;
  wasDrowsy      = false;
  yawnStart      = null;
  wasYawning     = false;
  wasDistracted  = false;
  perclosBuffer  = [];
  headDistrCount = 0;
  hiddenState    = [0, 0, 0, 0];
  smoothEAR      = 0.30;
  smoothMAR      = 0.00;
  smoothYaw      = 0;
  smoothPitch    = 0;
  alertCooldowns = {};
}

function resetCalibration() {
  calibSamples = [];
  calibDone    = false;
  earOffset    = 0;
  resetDetectionState();
}

// ── Main per-frame processing — mirrors version1 websocket loop ───────────────
/**
 * @param {Array}       lm          — 478 MediaPipe landmarks
 * @param {number}      phoneConf   — 0.0-1.0 confidence from YOLOv8 backend
 * @param {HTMLElement} videoEl     — for luminance sampling
 * @param {number}      frameCount
 */
function processFrame(lm, phoneConf = 0, videoEl = null, frameCount = 0) {
  if (!lm || lm.length < 478) return { faceDetected: false, lowLightMode };

  // Luminance every 30 frames (awareness badge only)
  if (videoEl && frameCount % 30 === 0) {
    const lux = measureLuminance(videoEl);
    lowLightMode = lux < 60;
  }

  // ── Raw calculations (version1 exact formulas) ────────────────────────────
  const earL   = calcEAR(lm, LEFT_EYE_IDX);
  const earR   = calcEAR(lm, RIGHT_EYE_IDX);
  const ear    = (earL + earR) / 2.0;
  const mar    = calcMAR(lm);
  const { yaw, pitch } = calcHeadPose(lm);

  // Smooth to reduce jitter (small alpha = heavy smoothing)
  smoothEAR   = 0.25 * smoothEAR   + 0.75 * ear;
  smoothMAR   = 0.25 * smoothMAR   + 0.75 * mar;
  smoothYaw   = 0.25 * smoothYaw   + 0.75 * yaw;
  smoothPitch = 0.25 * smoothPitch + 0.75 * pitch;

  // ── LSTM temporal scores (version1 uses raw, not smoothed) ───────────────
  const temporal = updateLSTM(ear, mar, pitch, yaw);

  // ── PERCLOS (version1 uses raw EAR, not smoothed) ────────────────────────
  const perclosPct = updatePerclos(ear);

  // ── Effective EAR threshold (calibrated) ─────────────────────────────────
  const effectiveEARThresh = EAR_THRESH + earOffset;

  // ── Drowsiness (version1 exact logic) ────────────────────────────────────
  // drowsy_count increments when ear < thresh, decrements by 2 when open
  // A 6-frame blink peaks at 6 — never reaches DROWSY_FRAMES=15
  if (ear < effectiveEARThresh) {
    drowsyCount = Math.min(drowsyCount + 1, DROWSY_FRAMES + 90);
  } else {
    drowsyCount = Math.max(0, drowsyCount - 2);
  }
  const isDrowsy = drowsyCount >= DROWSY_FRAMES;

  // ── Yawning (version1 time-based, not frame-based) ───────────────────────
  const now = Date.now();
  if (mar > MAR_THRESH) {
    if (!yawnStart) yawnStart = now;
  } else {
    yawnStart = null;
  }
  const yawnMs   = yawnStart ? (now - yawnStart) : 0;
  const isYawning = yawnMs >= YAWN_MS;

  // ── Head distraction (sustained) ─────────────────────────────────────────
  const headOff = Math.abs(smoothYaw) > YAW_THRESH || Math.abs(smoothPitch) > PITCH_THRESH;
  if (headOff) { headDistrCount++; }
  else         { headDistrCount = Math.max(0, headDistrCount - 3); }
  const isDistracted = headDistrCount >= HEAD_DIST_FRAMES;

  // ── Phone (confidence gate) ───────────────────────────────────────────────
  const phoneDetected = phoneConf >= PHONE_CONF;

  // ── Risk score (version1 Decision Fusion) ─────────────────────────────────
  const riskScore = computeRisk({
    eyeTemporal:   temporal.eyeTemporal,
    perclosPct,
    mouthTemporal: temporal.mouthTemporal,
    phoneConf:     phoneDetected ? phoneConf : 0,
    headTemporal:  temporal.headTemporal,
  });

  // ── RISING EDGE alerts (version1 key pattern) ─────────────────────────────
  // Alert ONLY fires on the transition from safe→detected, not while detected.
  // This + 8s cooldown = no repeated firing.
  const alerts = [];

  // Drowsy: EAR-based OR PERCLOS path (version1 has both)
  const isDrowsyViaPERCLOS = perclosPct >= PERCLOS_ALERT * 100; // 15%
  const drowsyAlert = (isDrowsy || isDrowsyViaPERCLOS) &&
                      (!wasDrowsy) && canAlert("drowsy_eyes");
  if (drowsyAlert) {
    alerts.push({ type:"drowsy_eyes", message:"⚠️ Drowsiness detected — eyes closing" });
  }

  // Yawn: rising edge
  if (isYawning && !wasYawning && canAlert("yawning")) {
    alerts.push({ type:"yawning", message:"⚠️ Yawning detected — take a break" });
  }

  // Head distraction: rising edge
  if (isDistracted && !wasDistracted && canAlert("head_distraction")) {
    alerts.push({ type:"head_distraction", message:"⚠️ Eyes off road too long" });
  }

  // Phone: rising edge
  if (phoneDetected && canAlert("phone_detected")) {
    alerts.push({ type:"phone_detected", message:"🚫 Phone detected — put it down!" });
  }

  // High risk
  if (riskScore >= 70 && canAlert("high_risk")) {
    alerts.push({ type:"high_risk", message:"🔴 HIGH RISK — Pull over safely!" });
  }

  // ── Update previous-state flags (version1 rising edge tracking) ──────────
  wasDrowsy    = isDrowsy || isDrowsyViaPERCLOS;
  wasYawning   = isYawning;
  wasDistracted = isDistracted;

  return {
    faceDetected:   true,
    // Raw values (for admin/debug logging)
    _ear:           +ear.toFixed(3),
    _earL:          +earL.toFixed(3),
    _earR:          +earR.toFixed(3),
    _mar:           +mar.toFixed(3),
    _effectiveEARThresh: +effectiveEARThresh.toFixed(3),
    // Temporal scores (for UI sparklines)
    eyeTemporal:    temporal.eyeTemporal,
    mouthTemporal:  temporal.mouthTemporal,
    headTemporal:   temporal.headTemporal,
    // Metrics
    perclosPct:     +perclosPct.toFixed(1),
    yawnMs,
    yaw:            +smoothYaw.toFixed(1),
    pitch:          +smoothPitch.toFixed(1),
    drowsyCount,
    headDistrCount,
    isDrowsy,
    isYawning,
    isDistracted,
    phoneDetected,
    phoneConf,
    riskScore,
    alerts,
    lowLightMode,
    calibDone,
    // Human-readable statuses for UI
    eyeStatus:  isDrowsy ? "Drowsy" : ear < effectiveEARThresh ? "Closing" : "Open",
    yawnStatus: isYawning ? "Yawning" : mar > MAR_THRESH ? `${(yawnMs/1000).toFixed(1)}s…` : "Normal",
  };
}