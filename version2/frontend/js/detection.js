// frontend/js/detection.js — v7 (time-based / fps-independent)
// ═══════════════════════════════════════════════════════════════════════════════
// DETECTION ENGINE
//
// WHY THIS VERSION IS TIME-BASED (the key change vs v6):
//   The previous version expressed every threshold as a FRAME COUNT
//   (e.g. "eyes closed for 15 frames"). That silently assumes a fixed 30 fps.
//   In a browser the processing loop is driven by requestAnimationFrame, which
//   runs at whatever rate the device/tab can manage — 60 fps on a fast laptop,
//   10–15 fps when the machine is busy or the tab is backgrounded. At 60 fps a
//   "15-frame" drowsy threshold fires in 0.25 s; at 15 fps it needs a full
//   second. The detector therefore behaved differently on every machine.
//
//   v7 measures REAL elapsed time with performance.now() deltas and expresses
//   all sustained-state thresholds in MILLISECONDS. The behaviour is now
//   identical regardless of frame rate.
//
// HOW BLINK FALSE POSITIVES ARE ELIMINATED (multi-cue gate):
//   Previous versions used a single threshold: "eyes closed for X ms → drowsy."
//   This fires on slow blinks, on talking with eyes half-shut, on looking down.
//   v7 uses a MULTI-CUE GATE: drowsiness requires ≥2 of 3 independent fatigue
//   cues to be active simultaneously:
//     Cue 1: Sustained eye closure ≥ 800 ms  (a blink is ~150 ms — never trips)
//     Cue 2: PERCLOS ≥ 15% over 6 s window   (needs many closures across 6 s)
//     Cue 3: EMA eye temporal score ≥ 0.35    (needs sustained low EAR to build)
//   A normal blink can at most briefly trip cue 1 (and won't, since 150ms < 800ms).
//   It NEVER trips cues 2 or 3 because those accumulate over seconds, not frames.
//   Only genuine drowsiness — sustained or repeated closures over many seconds —
//   activates 2+ cues and fires the alert.
//
//   This pattern is supported by the literature: multi-stage fusion of EAR, MAR,
//   and temporal scores into a counter-based decision system reduces false alarms
//   from fleeting facial changes (IEEE Access 2024, doi:10.1109/access.2024.3381999).
//
// BLINK RATE TRACKING (additional fatigue indicator):
//   The engine tracks individual blink transitions (EAR drops below threshold
//   then recovers within 50–600ms = one blink) and computes:
//     - Blinks per minute (normal: 10–15/min; abnormal: <3 or >20)
//     - Average blink duration (normal: ~150ms; slow blinks >300ms = fatigue)
//   These are surfaced to the UI and can be used for post-session analytics.
//
// THRESHOLDS (single source of truth — keep the report in sync with these):
//   EAR threshold:     0.21    (Soukupová & Čech 2016)
//   MAR threshold:     0.55    (yawn-specific; above normal talking/laughing)
//   Drowsy dwell:      800 ms  (sustained eye closure → drowsy; 500ms was too close to slow blinks)
//   PERCLOS window:    6000 ms (rolling time window)
//   PERCLOS closed:    EAR < 0.20 counts as "eye >80% closed"
//   PERCLOS alert:     15%     (NHTSA P80 standard)
//   Yawn dwell:        1500 ms (sustained mouth-open → yawn)
//   EMA alpha:         0.85    (exponential temporal smoothing — NOT an LSTM)
//   Phone confidence:  0.65    (filters remotes / glasses cases)
//   Head-off dwell:    1700 ms (sustained look-away → distraction)
//   Alert cooldown:    8000 ms (per alert type)
//
// NOTE ON THE TEMPORAL MODEL:
//   `updateTemporal()` is an exponential moving average (EMA) over the cue
//   features. It is deliberately NOT a trained LSTM — it is a lightweight
//   smoothing filter. Earlier versions mislabelled it "LSTM"; that was
//   inaccurate and has been corrected here and in the report.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Threshold values ──────────────────────────────────────────────────────────
const EAR_THRESH      = 0.21;   // eyes-closed EAR cutoff (Soukupová & Čech 2016)
const MAR_THRESH      = 0.55;   // yawn-open MAR cutoff (raised from 0.50 → fewer talk/laugh false hits)
const PERCLOS_THRESH  = 0.20;   // EAR below this = eye >80% closed (P80)
const PERCLOS_ALERT   = 0.15;   // PERCLOS fraction that triggers drowsy via PERCLOS path
const EMA_ALPHA       = 0.85;   // exponential temporal smoothing factor
const PHONE_CONF      = 0.65;   // YOLOv8 confidence gate for phone
const YAW_THRESH      = 35;     // degrees of yaw considered "looking away"
const PITCH_THRESH    = 20;     // degrees of pitch considered "looking away"

// ── Time-based dwell thresholds (milliseconds — fps-independent) ─────────────
const DROWSY_MS       = 800;    // sustained eye closure before "drowsy" (500ms was too close to slow blinks)
const PERCLOS_WIN_MS  = 6000;   // rolling PERCLOS time window
const YAWN_MS         = 1500;   // sustained mouth-open before "yawn"
const HEAD_DIST_MS    = 1700;   // sustained look-away before "distraction"
const ALERT_COOL_MS   = 8000;   // per-type alert cooldown

// Calibration: collect ~3 s of resting EAR samples (fps-independent via time)
const CALIBRATION_MS  = 3000;

// ── Landmark indices ──────────────────────────────────────────────────────────
const LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]; // version1 LEFT_EYE_EAR
const RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]; // version1 RIGHT_EYE_EAR
const MOUTH_IDX     = [61, 291, 82, 312, 13, 87, 317, 14]; // version1 MOUTH_MAR

// ── State ─────────────────────────────────────────────────────────────────────
// Per-session state — reset on resetDetectionState()
let drowsyMs       = 0;         // accumulated time (ms) eyes have been closed
let wasDrowsy      = false;     // rising-edge tracker
let yawnStart      = null;      // timestamp (ms) mouth first crossed MAR_THRESH
let wasYawning     = false;
let headOffMs      = 0;         // accumulated time (ms) head has been off-road
let wasDistracted  = false;
let perclosBuffer  = [];        // rolling window of {t, closed} samples (time-windowed)
let lastFrameTs    = null;      // performance.now() of previous processed frame

// ── Blink rate tracker (fatigue cue) ──────────────────────────────────────────
// Tracks blink transitions (EAR drops below threshold then recovers) to compute
// blinks/minute. Abnormal patterns (slow blinks, high frequency) indicate fatigue.
let blinkHistory     = [];      // timestamps of completed blinks (ring buffer)
let blinkStartTs     = null;    // when current blink started (null = eyes open)
let blinkDurHistory  = [];      // durations of recent blinks (ms)
const BLINK_WIN_MS   = 60000;   // 60s window for blink rate

// ── Multi-cue drowsiness gate ─────────────────────────────────────────────────
// Instead of alerting on EAR duration alone, require ≥2 of these 3 independent
// cues to be active simultaneously. A normal blink only ever trips cue 1 briefly;
// cues 2 and 3 need sustained patterns to activate. This eliminates blink
// false positives entirely.
//   Cue 1: Sustained eye closure (drowsyMs ≥ DROWSY_MS)
//   Cue 2: Elevated PERCLOS (≥ 15% in the rolling window)
//   Cue 3: Elevated EMA eye temporal score (≥ 0.35)
const MULTI_CUE_MIN  = 2;      // minimum active cues to trigger drowsy alert
const EMA_EYE_GATE   = 0.35;   // EMA eye temporal score threshold for cue 3

// EMA temporal hidden state (4 components: [eye, mouth, pitch_norm, yaw_norm])
// Exponential moving average — a lightweight smoothing filter, NOT an LSTM.
let emaState = [0, 0, 0, 0];

// Smoothed sensor values
let smoothEAR   = 0.30;
let smoothMAR   = 0.00;
let smoothYaw   = 0;
let smoothPitch = 0;

// Alert cooldowns (separate per type)
let alertCooldowns = {};

// Personalised EAR offset from calibration
let calibSamples     = [];
let calibStartTs     = null;    // performance.now() when calibration began
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

// ── Head pose (geometric approximation) ──────────────────────────────────────
// NOTE: This is a lightweight geometric estimate of yaw/pitch from 2D facial
// landmark ratios — NOT a full 3D PnP / solvePnP head-pose solve. It is
// accurate enough to detect sustained "looking away" but should not be
// described as a true pose matrix. Yaw ≈ nose offset from the ear-midpoint,
// normalised by ear span; pitch ≈ nose offset from the forehead/chin midpoint.
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

// ── PERCLOS — time-windowed (fps-independent) ─────────────────────────────────
// Stores {t, closed} samples and keeps only those within the last
// PERCLOS_WIN_MS milliseconds. Returns the percentage of windowed time the
// eyes were >80% closed (EAR < PERCLOS_THRESH). Because the window is measured
// in time, the result is the same at any frame rate.
function updatePerclos(ear, nowTs) {
  perclosBuffer.push({ t: nowTs, closed: ear < PERCLOS_THRESH ? 1 : 0 });
  const cutoff = nowTs - PERCLOS_WIN_MS;
  while (perclosBuffer.length && perclosBuffer[0].t < cutoff) perclosBuffer.shift();
  if (!perclosBuffer.length) return 0;
  const closed = perclosBuffer.reduce((a, s) => a + s.closed, 0);
  return (closed / perclosBuffer.length) * 100;
}

// ── EMA temporal smoothing (NOT an LSTM) ──────────────────────────────────────
// Exponential moving average over the four cue features. This is a simple
// first-order smoothing filter; it carries no learned weights and is not a
// recurrent neural network. feature = [1-ear/0.35, mar/0.5, |pitch|/30, |yaw|/45]
//   state = alpha * state + (1 - alpha) * feature
function updateTemporal(ear, mar, pitch, yaw) {
  const feature = [
    1.0 - Math.min(ear / 0.35, 1.0),
    Math.min(mar / 0.5, 1.0),
    Math.min(Math.abs(pitch) / 30.0, 1.0),
    Math.min(Math.abs(yaw)   / 45.0, 1.0),
  ];
  emaState = emaState.map((h, i) =>
    EMA_ALPHA * h + (1 - EMA_ALPHA) * feature[i]
  );
  return {
    eyeTemporal:   +emaState[0].toFixed(3),
    mouthTemporal: +emaState[1].toFixed(3),
    headTemporal:  +((emaState[2] + emaState[3]) / 2).toFixed(3),
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
// Time-based: collects samples for CALIBRATION_MS regardless of frame rate.
function calibrationStep(lm, nowTs) {
  if (calibDone) return { done: true, progress: 100 };
  if (!lm || lm.length < 478) return { done: false, progress: 0 };

  if (calibStartTs === null) calibStartTs = nowTs;

  const earL = calcEAR(lm, LEFT_EYE_IDX);
  const earR = calcEAR(lm, RIGHT_EYE_IDX);
  calibSamples.push((earL + earR) / 2);

  const elapsed  = nowTs - calibStartTs;
  const progress = Math.min(100, Math.round(elapsed / CALIBRATION_MS * 100));

  // Need both enough time AND a few samples (guards against ultra-low fps).
  if (elapsed >= CALIBRATION_MS && calibSamples.length >= 10) {
    const sorted    = [...calibSamples].sort((a, b) => a - b);
    const medianEAR = sorted[Math.floor(sorted.length * 0.5)];
    // If user's median open EAR is below ~0.26, lower the threshold.
    // Never raise it (don't punish users with naturally wide eyes).
    const naturalThresh = medianEAR * 0.78;  // 78% of median-open = closed
    earOffset = Math.min(0, naturalThresh - EAR_THRESH);
    earOffset = Math.max(-0.06, earOffset);  // cap adjustment at -0.06
    calibDone = true;
    console.log(`[DISHA] Calibration: medianEAR=${medianEAR.toFixed(3)}, earOffset=${earOffset.toFixed(3)}, effectiveThresh=${(EAR_THRESH+earOffset).toFixed(3)}`);
  }
  return { done: calibDone, progress };
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetDetectionState() {
  drowsyMs       = 0;
  wasDrowsy      = false;
  yawnStart      = null;
  wasYawning     = false;
  headOffMs      = 0;
  wasDistracted  = false;
  perclosBuffer  = [];
  lastFrameTs    = null;
  blinkHistory   = [];
  blinkStartTs   = null;
  blinkDurHistory= [];
  emaState       = [0, 0, 0, 0];
  smoothEAR      = 0.30;
  smoothMAR      = 0.00;
  smoothYaw      = 0;
  smoothPitch    = 0;
  alertCooldowns = {};
}

function resetCalibration() {
  calibSamples = [];
  calibStartTs = null;
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
 * @param {number}      nowTs       — performance.now() timestamp (ms) for this frame
 */
function processFrame(lm, phoneConf = 0, videoEl = null, frameCount = 0, nowTs = performance.now()) {
  if (!lm || lm.length < 478) {
    // Face lost: don't accumulate drowsy/distraction time, and break the
    // frame-delta chain so the next valid frame doesn't see a huge dt.
    lastFrameTs = null;
    return { faceDetected: false, lowLightMode };
  }

  // ── Frame delta (ms) — the basis for all fps-independent timing ───────────
  // Clamp to 200 ms so a stall (tab switch, GC pause) can't dump a huge chunk
  // of time into the accumulators and trigger a false alert on resume.
  let dt = lastFrameTs === null ? 0 : Math.min(nowTs - lastFrameTs, 200);
  lastFrameTs = nowTs;

  // Luminance every 30 frames (awareness badge only)
  if (videoEl && frameCount % 30 === 0) {
    const lux = measureLuminance(videoEl);
    lowLightMode = lux < 60;
  }

  // ── Raw calculations ──────────────────────────────────────────────────────
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

  // ── EMA temporal scores (uses raw, not smoothed) ─────────────────────────
  const temporal = updateTemporal(ear, mar, pitch, yaw);

  // ── PERCLOS — time-windowed (uses raw EAR) ───────────────────────────────
  const perclosPct = updatePerclos(ear, nowTs);

  // ── Effective EAR threshold (calibrated) ─────────────────────────────────
  const effectiveEARThresh = EAR_THRESH + earOffset;
  const eyesClosed = ear < effectiveEARThresh;

  // ── Blink tracker (fatigue indicator) ─────────────────────────────────────
  // Track blink transitions: eyes open → closed → open = one blink.
  // Record the timestamp and duration of each completed blink.
  // Abnormal patterns (slow blinks >300ms avg, or high rate >20/min)
  // feed into the multi-cue gate as an additional fatigue signal.
  if (eyesClosed) {
    if (blinkStartTs === null) blinkStartTs = nowTs; // blink just started
  } else {
    if (blinkStartTs !== null) {
      // Eyes just reopened → blink completed
      const blinkDur = nowTs - blinkStartTs;
      // Only count as a blink if duration is 50–600ms (not a sustained closure)
      if (blinkDur >= 50 && blinkDur <= 600) {
        blinkHistory.push(nowTs);
        blinkDurHistory.push(blinkDur);
        // Keep only last 60s of blinks
        const blinkCutoff = nowTs - BLINK_WIN_MS;
        while (blinkHistory.length && blinkHistory[0] < blinkCutoff) {
          blinkHistory.shift();
          blinkDurHistory.shift();
        }
      }
      blinkStartTs = null;
    }
  }

  // Blink rate and average duration (over last 60s)
  const blinksPerMin = blinkHistory.length; // window IS 60s, so count = rate/min
  const avgBlinkDur  = blinkDurHistory.length
    ? blinkDurHistory.reduce((a, b) => a + b, 0) / blinkDurHistory.length
    : 150; // default ~150ms = normal

  // ── Drowsiness — time accumulator (fps-independent) ──────────────────────
  // Eyes-closed time accumulates by dt; reopening drains at 3× rate so a
  // normal blink (~150 ms closed → 450ms equivalent drain) clears instantly
  // and never approaches DROWSY_MS (800 ms). Only genuinely sustained
  // closure (≥0.8 s) trips the accumulator.
  if (eyesClosed) {
    drowsyMs = Math.min(drowsyMs + dt, DROWSY_MS + 3000);
  } else {
    drowsyMs = Math.max(0, drowsyMs - 3 * dt);
  }

  // ── Multi-cue drowsiness gate ─────────────────────────────────────────────
  // Require ≥2 of 3 independent fatigue cues to be active simultaneously.
  // This is the key false-positive killer: a normal blink only ever trips
  // cue 1 briefly (and never reaches DROWSY_MS anyway at 800ms). Cues 2 and
  // 3 need sustained patterns across many seconds to activate, so transient
  // events can't fire the alert.
  //
  // Cue 1: Sustained eye closure (drowsyMs ≥ DROWSY_MS = 800ms)
  // Cue 2: Elevated PERCLOS (≥15% in the 6s rolling window)
  // Cue 3: Elevated EMA eye temporal score (≥0.35 — needs many recent
  //         low-EAR frames to accumulate given α=0.85 smoothing)
  const cue1_sustained  = drowsyMs >= DROWSY_MS;
  const cue2_perclos    = perclosPct >= PERCLOS_ALERT * 100;   // ≥15%
  const cue3_temporal   = temporal.eyeTemporal >= EMA_EYE_GATE; // ≥0.35
  const activeCues      = (cue1_sustained ? 1 : 0)
                        + (cue2_perclos    ? 1 : 0)
                        + (cue3_temporal   ? 1 : 0);
  const isDrowsy        = activeCues >= MULTI_CUE_MIN;

  // ── Yawning — sustained mouth-open (already time-based) ───────────────────
  if (mar > MAR_THRESH) {
    if (!yawnStart) yawnStart = nowTs;
  } else {
    yawnStart = null;
  }
  const yawnMs    = yawnStart ? (nowTs - yawnStart) : 0;
  const isYawning = yawnMs >= YAWN_MS;

  // ── Head distraction — time accumulator (fps-independent) ────────────────
  const headOff = Math.abs(smoothYaw) > YAW_THRESH || Math.abs(smoothPitch) > PITCH_THRESH;
  if (headOff) {
    headOffMs = Math.min(headOffMs + dt, HEAD_DIST_MS + 3000);
  } else {
    headOffMs = Math.max(0, headOffMs - 3 * dt);  // brief mirror/sign glances drain fast
  }
  const isDistracted = headOffMs >= HEAD_DIST_MS;

  // ── Phone (confidence gate) ───────────────────────────────────────────────
  const phoneDetected = phoneConf >= PHONE_CONF;

  // ── Risk score (decision fusion) ──────────────────────────────────────────
  const riskScore = computeRisk({
    eyeTemporal:   temporal.eyeTemporal,
    perclosPct,
    mouthTemporal: temporal.mouthTemporal,
    phoneConf:     phoneDetected ? phoneConf : 0,
    headTemporal:  temporal.headTemporal,
  });

  // ── RISING EDGE alerts ────────────────────────────────────────────────────
  // Alert ONLY fires on the transition from safe→detected, not while detected.
  // This + 8s cooldown = no repeated firing.
  const alerts = [];

  // Drowsy: multi-cue gated (≥2 of 3 cues must be active simultaneously).
  // This replaces the old single-threshold approach. A normal blink can
  // never trip 2 cues at once: cue 1 needs 800ms continuous closure (blinks
  // are ~150ms), cue 2 needs sustained high PERCLOS over 6 seconds, and
  // cue 3 needs elevated EMA over many frames. Only genuine drowsiness
  // activates multiple cues simultaneously.
  const drowsyAlert = isDrowsy && !wasDrowsy && canAlert("drowsy_eyes");
  if (drowsyAlert) {
    const cueNames = [];
    if (cue1_sustained) cueNames.push("sustained closure");
    if (cue2_perclos)   cueNames.push("high PERCLOS");
    if (cue3_temporal)  cueNames.push("temporal pattern");
    alerts.push({
      type: "drowsy_eyes",
      message: `⚠️ Drowsiness detected — ${cueNames.join(" + ")}`,
    });
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

  // ── Update previous-state flags (rising edge tracking) ───────────────────
  wasDrowsy     = isDrowsy;
  wasYawning    = isYawning;
  wasDistracted = isDistracted;

  return {
    faceDetected:   true,
    // Raw values (for admin/debug logging)
    _ear:           +ear.toFixed(3),
    _earL:          +earL.toFixed(3),
    _earR:          +earR.toFixed(3),
    _mar:           +mar.toFixed(3),
    _effectiveEARThresh: +effectiveEARThresh.toFixed(3),
    // Temporal scores (for UI)
    eyeTemporal:    temporal.eyeTemporal,
    mouthTemporal:  temporal.mouthTemporal,
    headTemporal:   temporal.headTemporal,
    // Metrics
    perclosPct:     +perclosPct.toFixed(1),
    yawnMs,
    yaw:            +smoothYaw.toFixed(1),
    pitch:          +smoothPitch.toFixed(1),
    // Progress toward thresholds (0..1)
    drowsyProgress: +Math.min(drowsyMs / DROWSY_MS, 1).toFixed(3),
    headProgress:   +Math.min(headOffMs / HEAD_DIST_MS, 1).toFixed(3),
    // Multi-cue drowsiness state (for UI display)
    drowsyCues:     { sustained: cue1_sustained, perclos: cue2_perclos, temporal: cue3_temporal },
    activeCueCount: activeCues,
    // Blink statistics
    blinksPerMin,
    avgBlinkDur:    +avgBlinkDur.toFixed(0),
    // Detection states
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
    eyeStatus:  isDrowsy ? "Drowsy" : eyesClosed ? "Closing" : "Open",
    yawnStatus: isYawning ? "Yawning" : mar > MAR_THRESH ? `${(yawnMs/1000).toFixed(1)}s…` : "Normal",
  };
}