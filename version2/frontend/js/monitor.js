// frontend/js/monitor.js
// Orchestrates the monitoring page:
//   1. Camera access via getUserMedia
//   2. MediaPipe FaceLandmarker (478 points) for EAR / MAR / head pose
//   3. TensorFlow.js COCO-SSD for phone detection (every 3rd frame)
//   4. Calls detection.js processFrame() each frame
//   5. Updates all UI gauges and badges
//   6. Logs alerts to the backend via api.js

requireUser();  // no token → login.html, admin → admin.html

// ── DOM references ────────────────────────────────────────────────────────────
const video          = document.getElementById("video");
const canvas         = document.getElementById("canvas");
const ctx            = canvas.getContext("2d");
const placeholder    = document.getElementById("placeholder");
const badgeStatus    = document.getElementById("badgeStatus");
const badgeLive      = document.getElementById("badgeLive");
const btnStart       = document.getElementById("btnStart");
const btnStop        = document.getElementById("btnStop");
const sessionLabel   = document.getElementById("sessionLabel");
const riskCircle     = document.getElementById("riskCircle");
const riskValue      = document.getElementById("riskValue");
const earValue       = document.getElementById("earValue");
const earGauge       = document.getElementById("earGauge");
const marValue       = document.getElementById("marValue");
const marGauge       = document.getElementById("marGauge");
const perclosValue   = document.getElementById("perclosValue");
const perclosGauge   = document.getElementById("perclosGauge");
const yawValue       = document.getElementById("yawValue");
const pitchValue     = document.getElementById("pitchValue");
const poseStatus     = document.getElementById("poseStatus");
const phoneStatus    = document.getElementById("phoneStatus");
const alertLog       = document.getElementById("alertLog");

// ── Runtime state ─────────────────────────────────────────────────────────────
let isMonitoring      = false;
let faceLandmarker    = null;   // MediaPipe FaceLandmarker instance
let cocoModel         = null;   // TF.js COCO-SSD model
let frameCount        = 0;      // total frames processed
let currentSessionId  = null;   // backend session ID
let animFrameId       = null;   // requestAnimationFrame handle
let phoneDetected     = false;  // result from last COCO-SSD pass
let stream            = null;   // MediaStream

// ── Initialise ML models ──────────────────────────────────────────────────────
async function initModels() {
  // Load MediaPipe FaceLandmarker
  // Using CDN task-vision package
  const { FaceLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"
  );

  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",  // falls back to CPU automatically if GPU unavailable
    },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
    numFaces: 1,  // single driver
  });

  // Load TF.js COCO-SSD for phone detection
  // cocoSsd is loaded via <script> tag (see monitor.html)
  cocoModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });

  console.log("[DISHA] Models loaded.");
}

// ── Start monitoring ──────────────────────────────────────────────────────────
async function startMonitoring() {
  btnStart.disabled = true;
  btnStart.textContent = "Starting…";

  try {
    // 1. Request camera access
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // 2. Load models if not already loaded
    if (!faceLandmarker || !cocoModel) {
      await initModels();
    }

    // 3. Start a backend session
    const user = getAuthUser();
    const session = await apiFetch("/api/sessions/", {
      method: "POST",
      body: JSON.stringify({ driver_name: user.name }),
    });
    currentSessionId = session.id;

    // 4. Update UI
    placeholder.classList.add("hidden");
    badgeLive.classList.remove("hidden");
    btnStart.classList.add("hidden");
    btnStop.disabled = false;
    btnStop.classList.remove("hidden");
    sessionLabel.textContent = `Session ID: ${currentSessionId.slice(-8)}`;

    isMonitoring = true;
    resetDetectionState();

    // 5. Kick off the frame loop
    processLoop();

  } catch (err) {
    console.error("[DISHA] Start error:", err);
    showToast(err.message || "Could not start monitoring.", "error");
    btnStart.disabled = false;
    btnStart.textContent = "Start Monitoring";
  }
}

// ── Stop monitoring ───────────────────────────────────────────────────────────
async function stopMonitoring() {
  isMonitoring = false;

  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  // Stop camera stream
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  // End backend session
  if (currentSessionId) {
    try {
      await apiFetch(`/api/sessions/${currentSessionId}/end`, { method: "PATCH" });
    } catch (e) {
      console.warn("[DISHA] Could not end session:", e);
    }
    currentSessionId = null;
  }

  // Reset UI
  placeholder.classList.remove("hidden");
  badgeLive.classList.add("hidden");
  btnStop.classList.add("hidden");
  btnStart.classList.remove("hidden");
  btnStart.disabled = false;
  btnStart.textContent = "Start Monitoring";
  sessionLabel.textContent = "";
  updateStatusBadge(0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  showToast("Monitoring stopped.", "info");
}

// ── Main frame loop ───────────────────────────────────────────────────────────
function processLoop() {
  if (!isMonitoring) return;

  const now = performance.now();

  // Resize canvas to match video
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  let result = { faceDetected: false };

  if (
    video.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA &&
    faceLandmarker
  ) {
    // Run FaceLandmarker
    const faceResult = faceLandmarker.detectForVideo(video, now);

    // Phone detection: run on every 3rd frame to save CPU
    if (frameCount % 3 === 0 && cocoModel) {
      cocoModel.detect(video).then(predictions => {
        // Check if any "cell phone" class is detected with reasonable confidence
        phoneDetected = predictions.some(
          p => p.class === "cell phone" && p.score > 0.45
        );
      });
    }

    if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
      const landmarks = faceResult.faceLandmarks[0];

      // Run our detection logic
      result = processFrame(landmarks, phoneDetected);

      // Draw landmarks overlay on canvas
      drawOverlay(ctx, landmarks, result);
    }

    frameCount++;
  }

  // Update UI with latest result
  updateUI(result);

  // Handle any triggered alerts
  if (result.alerts && result.alerts.length > 0) {
    result.alerts.forEach(alert => {
      handleAlert(alert, result);
    });
  }

  // Schedule next frame
  animFrameId = requestAnimationFrame(processLoop);
}

// ── Draw landmarks overlay ────────────────────────────────────────────────────
function drawOverlay(ctx, landmarks, result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const w = canvas.width;
  const h = canvas.height;

  // Draw face mesh dots (subtle)
  ctx.fillStyle = "rgba(79, 122, 255, 0.4)";
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Highlight eyes — colour depends on state
  const eyeColor = result.eyeClosed ? "#ef4444" : "#22c55e";
  const eyeAlpha = result.eyeClosed ? "0.9" : "0.6";

  for (const idx of [...[33,160,158,133,153,144], ...[362,385,387,263,373,380]]) {
    const lm = landmarks[idx];
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = eyeColor;
    ctx.fill();
  }

  // Highlight mouth — colour depends on yawn state
  const mouthColor = result.isYawning ? "#f59e0b" : "rgba(255,255,255,0.3)";
  for (const idx of [61, 291, 13, 14, 78, 308, 82, 312]) {
    const lm = landmarks[idx];
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = mouthColor;
    ctx.fill();
  }

  // Draw head pose direction arrow from nose tip
  const nose = landmarks[1];
  const arrowColor = result.isDistracted ? "#ef4444" : "#22c55e";
  const arrowLen = 40;
  const nx = nose.x * w;
  const ny = nose.y * h;
  const dx = Math.sin((result.yaw  || 0) * Math.PI / 180) * arrowLen;
  const dy = Math.sin((result.pitch || 0) * Math.PI / 180) * arrowLen;

  ctx.strokeStyle = arrowColor;
  ctx.lineWidth   = 2.5;
  ctx.beginPath();
  ctx.moveTo(nx, ny);
  ctx.lineTo(nx + dx, ny - dy);
  ctx.stroke();

  // Draw nose dot
  ctx.beginPath();
  ctx.arc(nx, ny, 4, 0, Math.PI * 2);
  ctx.fillStyle = arrowColor;
  ctx.fill();
}

// ── Update UI metrics ─────────────────────────────────────────────────────────
function updateUI(result) {
  const risk = result.riskScore ?? 0;
  updateStatusBadge(risk);

  if (!result.faceDetected) {
    earValue.textContent    = "--";
    marValue.textContent    = "--";
    perclosValue.textContent = "--";
    yawValue.textContent    = "--";
    pitchValue.textContent  = "--";
    setGauge(earGauge,     0, "ok");
    setGauge(marGauge,     0, "ok");
    setGauge(perclosGauge, 0, "ok");
    poseStatus.textContent  = "No face";
    phoneStatus.textContent = "---";
    return;
  }

  // EAR
  earValue.textContent = result.ear.toFixed(2);
  const earPct = Math.max(0, Math.min(100, ((0.35 - result.ear) / 0.35) * 100));
  const earClass = result.ear < EAR_THRESHOLD ? "danger" : result.ear < 0.23 ? "warn" : "ok";
  setGauge(earGauge, earPct, earClass);
  earValue.className = `metric-value ${earClass}`;

  // MAR
  marValue.textContent = result.mar.toFixed(2);
  const marPct = Math.min(100, (result.mar / 0.8) * 100);
  const marClass = result.mar > MAR_THRESHOLD ? "warn" : "ok";
  setGauge(marGauge, marPct, marClass);
  marValue.className = `metric-value ${marClass}`;

  // PERCLOS
  const perclosPct = Math.min(100, (result.perclos / 0.30) * 100);
  const perclosClass = result.perclos >= 0.15 ? "danger" : result.perclos >= 0.08 ? "warn" : "ok";
  perclosValue.textContent = `${(result.perclos * 100).toFixed(0)}%`;
  setGauge(perclosGauge, perclosPct, perclosClass);
  perclosValue.className = `metric-value ${perclosClass}`;

  // Head pose
  yawValue.textContent   = `${result.yaw > 0 ? "R" : "L"} ${Math.abs(result.yaw).toFixed(0)}°`;
  pitchValue.textContent = `${result.pitch > 0 ? "Up" : "Dn"} ${Math.abs(result.pitch).toFixed(0)}°`;
  poseStatus.textContent  = result.isDistracted ? "Distracted" : "Forward";
  poseStatus.className    = `metric-value ${result.isDistracted ? "danger" : "ok"}`;

  // Phone
  phoneStatus.textContent = result.phoneDetected ? "Detected!" : "None";
  phoneStatus.className   = `metric-value ${result.phoneDetected ? "danger" : "ok"}`;
}

// ── Gauge helper ──────────────────────────────────────────────────────────────
function setGauge(el, pct, cls) {
  el.style.width = `${Math.min(100, pct)}%`;
  el.className   = `gauge-fill ${cls}`;
}

// ── Status badge ──────────────────────────────────────────────────────────────
function updateStatusBadge(risk) {
  riskValue.textContent  = `${risk}%`;
  riskCircle.textContent = `${risk}%`;

  if (risk < 30) {
    badgeStatus.textContent  = "SAFE";
    badgeStatus.className    = "video-badge badge-status badge-safe";
    riskCircle.className     = "risk-circle safe";
  } else if (risk < 70) {
    badgeStatus.textContent  = "WARNING";
    badgeStatus.className    = "video-badge badge-status badge-warn";
    riskCircle.className     = "risk-circle warn";
  } else {
    badgeStatus.textContent  = "DANGER";
    badgeStatus.className    = "video-badge badge-status badge-danger";
    riskCircle.className     = "risk-circle danger";
  }
}

// ── Handle alert ──────────────────────────────────────────────────────────────
async function handleAlert(alert, metrics) {
  // 1. Add to on-screen log
  addAlertToLog(alert);

  // 2. Play audio beep
  playBeep(alert.type);

  // 3. Log to backend if session is active
  if (currentSessionId) {
    try {
      await apiFetch("/api/events/", {
        method: "POST",
        body: JSON.stringify({
          session_id:  currentSessionId,
          event_type:  alert.type,
          ear:         metrics.ear,
          mar:         metrics.mar,
          yaw:         metrics.yaw,
          pitch:       metrics.pitch,
          perclos:     metrics.perclos,
          risk_score:  metrics.riskScore,
        }),
      });
    } catch (e) {
      console.warn("[DISHA] Failed to log event:", e);
    }
  }
}

// ── On-screen alert log ───────────────────────────────────────────────────────
function addAlertToLog(alert) {
  // Remove "no alerts" placeholder if present
  const empty = alertLog.querySelector(".empty-state");
  if (empty) empty.remove();

  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const entry = document.createElement("div");
  entry.className = `alert-entry ${alert.type}`;
  entry.innerHTML = `
    <span class="alert-time">${time}</span>
    <span class="alert-msg">${alert.message}</span>
  `;

  // Prepend so newest is at top
  alertLog.insertBefore(entry, alertLog.firstChild);

  // Keep log max 20 entries
  while (alertLog.children.length > 20) {
    alertLog.removeChild(alertLog.lastChild);
  }
}

// ── Audio alert ───────────────────────────────────────────────────────────────
// Uses Web Audio API to synthesise a beep — no file needed.
function playBeep(alertType) {
  try {
    const audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode   = audioCtx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // Different tones for different alert types
    const freqMap = {
      drowsy_eyes:     880,   // A5 — urgent
      yawning:         660,   // E5 — moderate
      phone_detected:  1100,  // C#6 — sharp
      head_distraction: 770,  // G5
      high_risk:       1200,  // High — critical
    };
    oscillator.frequency.value = freqMap[alertType] || 880;
    oscillator.type            = "sine";
    gainNode.gain.value        = 0.3;

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + (alertType === "high_risk" ? 0.8 : 0.4));
  } catch (e) {
    // Audio context may be blocked before user interaction — fail silently
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────
btnStart.addEventListener("click", startMonitoring);
btnStop.addEventListener("click",  stopMonitoring);

// Populate user name in sidebar
const user = getAuthUser();
document.getElementById("sidebarUserName").textContent = user.name || "Driver";
document.getElementById("sidebarUserRole").textContent = user.role || "user";
document.getElementById("sidebarAvatar").textContent   = (user.name || "D")[0].toUpperCase();

// Hide admin link for non-admins
if (user.role !== "admin") {
  const adminLinks = document.querySelectorAll(".admin-only");
  adminLinks.forEach(el => el.classList.add("hidden"));
}

// Logout button
document.getElementById("btnLogout").addEventListener("click", () => {
  if (isMonitoring) stopMonitoring();
  authLogout();
});

// Pre-load models in background after page load so Start is faster
window.addEventListener("load", () => {
  setTimeout(initModels, 800);
});