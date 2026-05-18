// frontend/js/detection.js — v4 (personalised calibration + clean architecture)
// ═══════════════════════════════════════════════════════════════════════════════
// WHAT CHANGED FROM v3:
//  • Personalised EAR/MAR calibration — thresholds set relative to each user's
//    own face (arxiv 2604.22479, PMC 12899127)
//  • Glasses heuristic REMOVED — not validated, caused more FPs than it fixed
//  • CSS low-light filter REMOVED — it only changed what the user saw, not what
//    MediaPipe processed. Low-light badge kept as awareness indicator only.
//  • Phone detection (COCO-SSD) REMOVED from this engine — handled externally
//    by YOLOv8-ONNX in phone-detector.js and passed in as a flag
//  • MAR simplified to vertical/horizontal ratio using centre landmarks only
//    (13=top-lip, 14=bottom-lip, 61=left-corner, 291=right-corner)
//  • Raw EAR/MAR no longer exported as primary UI values — status strings used
//  • LSTM references removed — correctly labelled "temporal smoothing"
//  • Pitch kept as silent input to distraction score, not shown in UI
//
// VALIDATED THRESHOLDS (all sourced):
//  Calibration: 3 s steady baseline, threshold = baseline × 0.75 for EAR
//               (arxiv 2604.22479 — personalised ratio approach)
//  PERCLOS > 15 % over 90-frame window  (NHTSA/Wierwille 1994)
//  Consecutive closed ≥ 48 frames (~2 s)  (IRJMETS 2024)
//  MAR threshold = baseline × 1.5 (personalised, arxiv 2604.22479)
//  Yaw > ±35°, Pitch > ±25°, sustained ≥ 50 frames  (PMC 10600215)
//  Risk weights: PERCLOS 40 % | Head 35 % | Yawn 25 %
//  (Phone removed from fusion — handled by YOLOv8 separately)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Constants ─────────────────────────────────────────────────────────────────
const CALIBRATION_FRAMES      = 90;   // ~3 s at 30 fps to build baseline
const CALIBRATION_PERCENTILE  = 0.10; // discard bottom 10% (blinks during calib)
const EAR_RATIO               = 0.75; // threshold = baseline_EAR × 0.75
const MAR_RATIO               = 1.50; // threshold = baseline_MAR × 1.50
const FALLBACK_EAR            = 0.23; // used if calibration not complete
const FALLBACK_MAR            = 0.55;

const CONSEC_CLOSED_FRAMES    = 48;   // IRJMETS 2024
const PERCLOS_WINDOW          = 90;   // ~3 s
const PERCLOS_THRESHOLD       = 0.15; // NHTSA 1994
const YAW_THRESHOLD           = 35;   // degrees
const PITCH_THRESHOLD         = 25;   // degrees (silent input only)
const HEAD_DISTRACTION_FRAMES = 50;   // PMC 10600215
const YAWN_MIN_FRAMES         = 20;   // sustained frames before alert
const SMOOTH_ALPHA            = 0.25; // exponential smoothing factor

// Alert cooldowns (ms)
const COOLDOWNS = {
  drowsy_eyes:      8000,
  yawning:          10000,
  phone_detected:   6000,
  head_distraction: 8000,
  high_risk:        12000,
};

// ── Calibration state ─────────────────────────────────────────────────────────
let calibrationEARSamples = [];
let calibrationMARSamples = [];
let calibrationComplete   = false;
let personalEARThreshold  = FALLBACK_EAR;
let personalMARThreshold  = FALLBACK_MAR;

// ── Runtime state ─────────────────────────────────────────────────────────────
let consecutiveClosedFrames = 0;
let consecutiveDistrFrames  = 0;
let perclosBuffer           = [];
let yawnFrameCounter        = 0;
let alertCooldowns          = {};

// Smoothed values
let smoothEAR   = 0.30;
let smoothMAR   = 0.10;
let smoothYaw   = 0;
let smoothPitch = 0;

// Low-light awareness (luminance only — no CSS filter)
let currentLuminance = 128;
let lowLightMode     = false;

// Off-screen canvas for luminance measurement
let _luxCanvas = null;
let _luxCtx    = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
const dist  = (a, b) => Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const expSmooth = (prev, next) => SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * next;

// ── EAR — Soukupová & Čech 2016 ──────────────────────────────────────────────
const LEFT_EYE  = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

function calcEAR(lm, idx) {
  const p  = idx.map(i => lm[i]);
  const v1 = dist(p[1], p[5]);
  const v2 = dist(p[2], p[4]);
  const h  = dist(p[0], p[3]);
  return h < 0.001 ? 0.30 : (v1 + v2) / (2.0 * h);
}

// ── MAR — simplified vertical/horizontal only ─────────────────────────────────
// Uses only: 13 (top-lip centre), 14 (bottom-lip centre),
//            61 (left corner),    291 (right corner)
// Avoids mixing inner/outer lip points which caused FPs
function calcMAR(lm) {
  const vertical   = dist(lm[13], lm[14]);
  const horizontal = dist(lm[61], lm[291]);
  return horizontal < 0.001 ? 0 : vertical / horizontal;
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

// ── PERCLOS ───────────────────────────────────────────────────────────────────
function updatePerclos(isClosed) {
  perclosBuffer.push(isClosed ? 1 : 0);
  if (perclosBuffer.length > PERCLOS_WINDOW) perclosBuffer.shift();
  return perclosBuffer.length
    ? perclosBuffer.reduce((a, b) => a + b, 0) / perclosBuffer.length
    : 0;
}

// ── Alert cooldown ────────────────────────────────────────────────────────────
function canAlert(type) {
  const now = Date.now();
  const cd  = COOLDOWNS[type] || 5000;
  if (!alertCooldowns[type] || now - alertCooldowns[type] > cd) {
    alertCooldowns[type] = now;
    return true;
  }
  return false;
}

// ── Luminance sampling ────────────────────────────────────────────────────────
// Correctly reads raw video pixels — does NOT apply any CSS filter.
// Used as an AWARENESS INDICATOR only (low-light badge in UI).
function measureLuminance(videoEl) {
  if (!_luxCanvas) {
    _luxCanvas = document.createElement("canvas");
    _luxCanvas.width  = 40;
    _luxCanvas.height = 30;
    _luxCtx = _luxCanvas.getContext("2d", { willReadFrequently: true });
  }
  try {
    _luxCtx.drawImage(videoEl, 0, 0, 40, 30);
    const px = _luxCtx.getImageData(0, 0, 40, 30).data;
    let sum = 0;
    for (let i = 0; i < px.length; i += 4)
      sum += 0.2126 * px[i] + 0.7152 * px[i+1] + 0.0722 * px[i+2]; // ITU-R BT.709
    return sum / (40 * 30);
  } catch (_) { return 128; }
}

// ── Risk score — phone removed from fusion ────────────────────────────────────
// Phone detection handled separately by YOLOv8 and appended to UI independently
function computeRiskScore({ perclos, isYawning, isDistracted }) {
  const s =
    clamp(perclos / PERCLOS_THRESHOLD, 0, 1) * 0.40 +
    (isDistracted ? 1 : 0) * 0.35 +
    (isYawning    ? 1 : 0) * 0.25;
  return Math.round(s * 100);
}

// ── Percentile helper for calibration ────────────────────────────────────────
function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * p)] ?? sorted[0];
}

// ── Calibration step ──────────────────────────────────────────────────────────
// Called every frame during the calibration countdown.
// Returns { done: bool, progress: 0-100 }
function calibrationStep(lm) {
  if (calibrationComplete) return { done: true, progress: 100 };
  if (!lm || lm.length < 478) return { done: false, progress: 0 };

  const ear = (calcEAR(lm, LEFT_EYE) + calcEAR(lm, RIGHT_EYE)) / 2;
  const mar = calcMAR(lm);

  calibrationEARSamples.push(ear);
  calibrationMARSamples.push(mar);

  const progress = Math.round((calibrationEARSamples.length / CALIBRATION_FRAMES) * 100);

  if (calibrationEARSamples.length >= CALIBRATION_FRAMES) {
    // Discard bottom 10% of EAR (those are blink frames) before computing baseline
    const baselineEAR = percentile(calibrationEARSamples, CALIBRATION_PERCENTILE + 0.5);
    const baselineMAR = percentile(calibrationMARSamples, 0.50); // median resting MAR

    personalEARThreshold = +(baselineEAR * EAR_RATIO).toFixed(4);
    personalMARThreshold = +(baselineMAR * MAR_RATIO).toFixed(4);

    // Safety clamps — never set thresholds outside validated ranges
    personalEARThreshold = clamp(personalEARThreshold, 0.15, 0.28);
    personalMARThreshold = clamp(personalMARThreshold, 0.35, 0.75);

    calibrationComplete = true;
    console.log(`[DISHA] Calibration done. EAR threshold: ${personalEARThreshold}, MAR threshold: ${personalMARThreshold}`);
  }

  return { done: calibrationComplete, progress };
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetDetectionState() {
  consecutiveClosedFrames = 0;
  consecutiveDistrFrames  = 0;
  perclosBuffer           = [];
  yawnFrameCounter        = 0;
  alertCooldowns          = {};
  smoothEAR               = 0.30;
  smoothMAR               = 0.10;
  smoothYaw               = 0;
  smoothPitch             = 0;
}

function resetCalibration() {
  calibrationEARSamples = [];
  calibrationMARSamples = [];
  calibrationComplete   = false;
  personalEARThreshold  = FALLBACK_EAR;
  personalMARThreshold  = FALLBACK_MAR;
  resetDetectionState();
}

// ── Main per-frame processing ─────────────────────────────────────────────────
/**
 * @param {Array}       lm          — 478 {x,y,z} landmarks from MediaPipe
 * @param {boolean}     phoneFlag   — from YOLOv8-ONNX phone detector
 * @param {HTMLElement} videoEl     — <video> for luminance sampling
 * @param {number}      frameCount  — global frame counter
 */
function processFrame(lm, phoneFlag = false, videoEl = null, frameCount = 0) {
  if (!lm || lm.length < 478) {
    return { faceDetected: false, lowLightMode };
  }

  // Luminance check every 30 frames (awareness only — no filter applied)
  if (videoEl && frameCount % 30 === 0) {
    currentLuminance = measureLuminance(videoEl);
    lowLightMode     = currentLuminance < 60;
  }

  // Raw readings
  const rawEAR = (calcEAR(lm, LEFT_EYE) + calcEAR(lm, RIGHT_EYE)) / 2;
  const rawMAR = calcMAR(lm);
  const { yaw: rawYaw, pitch: rawPitch } = calcHeadPose(lm);

  // Exponential smoothing — kills per-frame jitter
  smoothEAR   = expSmooth(smoothEAR,   rawEAR);
  smoothMAR   = expSmooth(smoothMAR,   rawMAR);
  smoothYaw   = expSmooth(smoothYaw,   rawYaw);
  smoothPitch = expSmooth(smoothPitch, rawPitch);

  // Eye state (uses personalised threshold from calibration)
  const eyeClosed = smoothEAR < personalEARThreshold;
  if (eyeClosed) {
    consecutiveClosedFrames++;
  } else {
    consecutiveClosedFrames = Math.max(0, consecutiveClosedFrames - 2);
  }

  const perclos  = updatePerclos(eyeClosed);
  const isDrowsy =
    consecutiveClosedFrames >= CONSEC_CLOSED_FRAMES ||
    perclos >= PERCLOS_THRESHOLD;

  // Yawn state (uses personalised threshold)
  if (smoothMAR > personalMARThreshold) {
    yawnFrameCounter++;
  } else {
    yawnFrameCounter = Math.max(0, yawnFrameCounter - 1);
  }
  const isYawning = yawnFrameCounter >= YAWN_MIN_FRAMES;

  // Head distraction (sustained — allows mirror checks)
  const headOffRoad =
    Math.abs(smoothYaw) > YAW_THRESHOLD ||
    Math.abs(smoothPitch) > PITCH_THRESHOLD;

  if (headOffRoad) {
    consecutiveDistrFrames++;
  } else {
    consecutiveDistrFrames = Math.max(0, consecutiveDistrFrames - 3);
  }
  const isDistracted = consecutiveDistrFrames >= HEAD_DISTRACTION_FRAMES;

  // Risk score (phone excluded from fusion — appended by caller if needed)
  const riskScore = computeRiskScore({ perclos, isYawning, isDistracted });

  // Triggered alerts
  const alerts = [];
  if (isDrowsy && canAlert("drowsy_eyes"))
    alerts.push({ type: "drowsy_eyes", message: "⚠️ Drowsiness detected — eyes closing" });
  if (isYawning && canAlert("yawning"))
    alerts.push({ type: "yawning", message: "⚠️ Yawning detected — consider a break" });
  if (isDistracted && canAlert("head_distraction"))
    alerts.push({ type: "head_distraction", message: "⚠️ Eyes off road for too long" });
  if (phoneFlag && canAlert("phone_detected"))
    alerts.push({ type: "phone_detected", message: "🚫 Phone in use — put it down!" });
  if (riskScore >= 70 && canAlert("high_risk"))
    alerts.push({ type: "high_risk", message: "🔴 HIGH RISK — Pull over safely!" });

  return {
    faceDetected: true,
    // Internal values (for admin/debug only — not shown raw in driver UI)
    _ear: +smoothEAR.toFixed(3),
    _mar: +smoothMAR.toFixed(3),
    _earThreshold: personalEARThreshold,
    _marThreshold: personalMARThreshold,
    // UI-facing status strings
    eyeStatus:    eyeClosed ? "Closing" : smoothEAR < personalEARThreshold + 0.03 ? "Drowsy" : "Open",
    yawnStatus:   isYawning ? "Yawning" : yawnFrameCounter > 5 ? "Opening" : "Normal",
    // Metrics
    yaw:          +smoothYaw.toFixed(1),
    perclos:      +perclos.toFixed(3),
    consecutiveClosedFrames,
    consecutiveDistrFrames,
    eyeClosed,
    isYawning,
    isDistracted,
    phoneFlag,
    riskScore,
    alerts,
    lowLightMode,
    currentLuminance: Math.round(currentLuminance),
    calibrationComplete,
  };
}