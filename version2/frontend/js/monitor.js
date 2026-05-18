// frontend/js/monitor.js — v4
// Full monitor page orchestrator:
//  • 3-phase flow: CALIBRATION → MONITORING → SESSION SUMMARY
//  • Confidence gating: face-lost state pauses PERCLOS
//  • YOLOv8-ONNX phone detection (every 5th frame)
//  • Fullscreen toggle
//  • Session notes on stop
//  • Correct local timestamps

requireUser();

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video           = document.getElementById("video");
const canvas          = document.getElementById("canvas");
const ctx             = canvas.getContext("2d");
const placeholder     = document.getElementById("placeholder");
const calibOverlay    = document.getElementById("calibOverlay");
const calibProgress   = document.getElementById("calibProgress");
const calibMsg        = document.getElementById("calibMsg");
const badgeStatus     = document.getElementById("badgeStatus");
const badgeLive       = document.getElementById("badgeLive");
const badgeLowLight   = document.getElementById("badgeLowLight");
const btnStart        = document.getElementById("btnStart");
const btnStop         = document.getElementById("btnStop");
const btnFullscreen   = document.getElementById("btnFullscreen");
const sessionLabel    = document.getElementById("sessionLabel");
const riskCircle      = document.getElementById("riskCircle");
const riskValue       = document.getElementById("riskValue");
const eyeStatusEl     = document.getElementById("eyeStatus");
const eyeGauge        = document.getElementById("eyeGauge");
const perclosValue    = document.getElementById("perclosValue");
const perclosGauge    = document.getElementById("perclosGauge");
const yawnStatusEl    = document.getElementById("yawnStatus");
const yawnGauge       = document.getElementById("yawnGauge");
const yawValue        = document.getElementById("yawValue");
const poseStatus      = document.getElementById("poseStatus");
const phoneStatus     = document.getElementById("phoneStatus");
const faceStatus      = document.getElementById("faceStatus");
const alertLog        = document.getElementById("alertLog");
const summaryModal    = document.getElementById("summaryModal");
const notesModal      = document.getElementById("notesModal");
const notesInput      = document.getElementById("notesInput");
const btnConfirmStop  = document.getElementById("btnConfirmStop");
const btnCancelStop   = document.getElementById("btnCancelStop");

// ── State ─────────────────────────────────────────────────────────────────────
let isMonitoring      = false;
let isCalibrating     = false;
let faceLandmarker    = null;
let yoloSession       = null;   // ONNX Runtime session for YOLOv8
let frameCount        = 0;
let currentSessionId  = null;
let animFrameId       = null;
let stream            = null;
let modelsReady       = false;
let phoneDetected     = false;
let faceLostFrames    = 0;      // consecutive frames with no face
const FACE_LOST_THRESHOLD = 15; // frames before "face lost" state

// ── Model loading ─────────────────────────────────────────────────────────────
async function initModels() {
  if (modelsReady) return;

  // 1. MediaPipe FaceLandmarker
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
      delegate: "GPU",
    },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  });

  // 2. YOLOv8n via ONNX Runtime Web (WebAssembly)
  // Uses the public yolov8n model from CDN — ~13 MB, 80-class COCO
  try {
    // ort is loaded via script tag in index.html
    yoloSession = await ort.InferenceSession.create(
      "https://huggingface.co/onnx-community/yolov8n/resolve/main/onnx/model.onnx",
      { executionProviders: ["wasm"] }
    );
    console.log("[DISHA] YOLOv8n ONNX loaded.");
  } catch (e) {
    // YOLOv8 load failure is non-fatal — phone detection just won't run
    console.warn("[DISHA] YOLOv8 load failed (phone detection disabled):", e.message);
    yoloSession = null;
  }

  modelsReady = true;
  console.log("[DISHA] All models ready.");
}

// ── YOLOv8 phone detection ────────────────────────────────────────────────────
// Runs on every 5th frame. COCO class 67 = "cell phone"
async function detectPhone() {
  if (!yoloSession) return;
  try {
    // Draw video frame to an off-screen 640×640 canvas for YOLO input
    const sz   = 640;
    const offsc = document.createElement("canvas");
    offsc.width = offsc.height = sz;
    const offCtx = offsc.getContext("2d");
    offCtx.drawImage(video, 0, 0, sz, sz);
    const imageData = offCtx.getImageData(0, 0, sz, sz).data;

    // Convert RGBA → normalised float32 RGB tensor [1,3,640,640]
    const tensor = new Float32Array(3 * sz * sz);
    for (let i = 0; i < sz * sz; i++) {
      tensor[i]             = imageData[i * 4]     / 255; // R
      tensor[i + sz * sz]   = imageData[i * 4 + 1] / 255; // G
      tensor[i + 2*sz * sz] = imageData[i * 4 + 2] / 255; // B
    }

    const input  = new ort.Tensor("float32", tensor, [1, 3, sz, sz]);
    const output = await yoloSession.run({ images: input });
    const data   = output[Object.keys(output)[0]].data;

    // YOLOv8 output shape: [1, 84, 8400]
    // cols 0-3: cx,cy,w,h  cols 4-83: class scores
    // Class 67 = cell phone in COCO
    const numDetections = 8400;
    const numClasses    = 80;
    let found = false;
    for (let i = 0; i < numDetections; i++) {
      const scores = Array.from({length: numClasses}, (_, c) =>
        data[(4 + c) * numDetections + i]
      );
      const maxScore = Math.max(...scores);
      const classId  = scores.indexOf(maxScore);
      if (classId === 67 && maxScore > 0.50) { found = true; break; }
    }
    phoneDetected = found;
  } catch (_) {
    // Inference errors are non-fatal
  }
}

// ── Start flow ────────────────────────────────────────────────────────────────
async function startMonitoring() {
  btnStart.disabled    = true;
  btnStart.textContent = "Starting…";

  try {
    // Camera
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // Models
    if (!modelsReady) {
      btnStart.textContent = "Loading AI models…";
      await initModels();
    }

    // Start backend session
    const user    = getAuthUser();
    const session = await apiFetch("/api/sessions/", {
      method: "POST",
      body: JSON.stringify({ driver_name: user.name }),
    });
    currentSessionId = session.id;

    // Reset state
    resetCalibration();
    frameCount    = 0;
    faceLostFrames = 0;
    phoneDetected  = false;

    // Show UI
    placeholder.classList.add("hidden");
    badgeLive.classList.remove("hidden");
    btnStart.classList.add("hidden");
    btnStop.disabled = false;
    btnStop.classList.remove("hidden");
    sessionLabel.textContent = `Session: ${currentSessionId.slice(-8).toUpperCase()}`;

    // Start calibration phase
    isCalibrating = true;
    isMonitoring  = true;
    showCalibOverlay(0);
    processLoop();

  } catch (err) {
    console.error("[DISHA] Start error:", err);
    showToast(err.message || "Could not start monitoring.", "error");
    btnStart.disabled    = false;
    btnStart.textContent = "▶ Start Monitoring";
  }
}

// ── Calibration overlay ───────────────────────────────────────────────────────
function showCalibOverlay(pct) {
  calibOverlay.classList.remove("hidden");
  calibProgress.style.width = `${pct}%`;
  if (pct < 30)
    calibMsg.textContent = "Look straight at the camera, relax your face…";
  else if (pct < 70)
    calibMsg.textContent = "Keep still — measuring your face baseline…";
  else
    calibMsg.textContent = "Almost done…";
}

function hideCalibOverlay() {
  calibOverlay.classList.add("hidden");
  showToast("Calibration complete — monitoring started.", "success");
}

// ── Stop → notes modal ────────────────────────────────────────────────────────
function requestStop() {
  // Show notes modal before actually stopping
  notesInput.value = "";
  notesModal.classList.remove("hidden");
  notesInput.focus();
}

async function doStop(notes) {
  notesModal.classList.add("hidden");
  isMonitoring  = false;
  isCalibrating = false;
  cancelAnimationFrame(animFrameId);
  stream && stream.getTracks().forEach(t => t.stop());
  stream = null;

  // End backend session with optional notes
  if (currentSessionId) {
    try {
      await apiFetch(`/api/sessions/${currentSessionId}/end`, {
        method: "PATCH",
        body: JSON.stringify({ notes: notes || null }),
      });
      // Load and show session summary
      await showSessionSummary(currentSessionId);
    } catch (e) {
      console.warn("[DISHA] Session end failed:", e);
    }
    currentSessionId = null;
  }

  // Reset UI
  placeholder.classList.remove("hidden");
  calibOverlay.classList.add("hidden");
  badgeLive.classList.add("hidden");
  badgeLowLight.classList.add("hidden");
  btnStop.classList.add("hidden");
  btnStart.classList.remove("hidden");
  btnStart.disabled    = false;
  btnStart.textContent = "▶ Start Monitoring";
  sessionLabel.textContent = "";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  updateStatusBadge(0);
}

btnStop.addEventListener("click", requestStop);
btnConfirmStop.addEventListener("click", () => doStop(notesInput.value.trim()));
btnCancelStop.addEventListener("click", () => {
  notesModal.classList.add("hidden");
  // Resume monitoring — user changed their mind
});

// ── Session summary modal ─────────────────────────────────────────────────────
async function showSessionSummary(sessionId) {
  try {
    const s = await apiFetch(`/api/analytics/summary/${sessionId}`);
    renderSummary(s);
    summaryModal.classList.remove("hidden");
  } catch (e) {
    console.warn("[DISHA] Summary load failed:", e);
  }
}

function renderSummary(s) {
  const dur = s.duration_seconds
    ? (s.duration_seconds >= 60
        ? `${Math.floor(s.duration_seconds/60)}m ${Math.floor(s.duration_seconds%60)}s`
        : `${Math.floor(s.duration_seconds)}s`)
    : "—";

  const risk     = Math.round(s.max_risk_score || 0);
  const riskCls  = risk < 30 ? "safe" : risk < 70 ? "warn" : "danger";

  const typeLabels = {
    drowsy_eyes:"😴 Drowsy Eyes", yawning:"🥱 Yawning",
    phone_detected:"📱 Phone", head_distraction:"↩️ Distraction", high_risk:"🔴 High Risk"
  };

  const breakdownHtml = Object.entries(s.breakdown || {}).map(([t, c]) =>
    `<div class="alert-entry ${t}" style="justify-content:space-between">
      <span class="alert-msg">${typeLabels[t]||t}</span>
      <strong>${c}</strong>
    </div>`
  ).join("") || '<p style="color:var(--muted);font-size:.82rem">No alerts.</p>';

  // Risk timeline sparkline (simple SVG)
  const sparkSvg = buildSparkline(s.timeline || []);

  const peakHtml = s.peak
    ? `<p style="font-size:.8rem;color:var(--muted);margin-top:.5rem">
         Peak: <strong style="color:var(--danger)">${s.peak.risk}%</strong> risk
         at ${new Date(s.peak.timestamp).toLocaleTimeString()}
         (${typeLabels[s.peak.type]||s.peak.type})
       </p>`
    : "";

  document.getElementById("summaryContent").innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:1.2rem">
      <div class="kpi-mini"><div class="kpi-lbl">Duration</div><div class="kpi-big">${dur}</div></div>
      <div class="kpi-mini"><div class="kpi-lbl">Total Alerts</div><div class="kpi-big">${s.total_alerts}</div></div>
      <div class="kpi-mini"><div class="kpi-lbl">Max Risk</div>
        <div class="kpi-big" style="color:var(--${riskCls})">${risk}%</div></div>
    </div>
    ${s.notes ? `<p style="font-size:.82rem;color:var(--muted);margin-bottom:.8rem">📝 ${s.notes}</p>` : ""}
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem">
      <div>
        <div class="card-title">Alert Breakdown</div>
        ${breakdownHtml}
      </div>
      <div>
        <div class="card-title">Risk Over Time</div>
        ${sparkSvg}
        ${peakHtml}
      </div>
    </div>`;
}

function buildSparkline(timeline) {
  if (!timeline.length) return '<p style="color:var(--muted);font-size:.8rem">No data.</p>';
  const W = 220, H = 80, pad = 6;
  const max = Math.max(...timeline.map(p => p.r), 1);
  const pts = timeline.map((p, i) => {
    const x = pad + (i / Math.max(timeline.length - 1, 1)) * (W - pad*2);
    const y = H - pad - (p.r / max) * (H - pad*2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:80px">
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    <!-- 70% danger line -->
    <line x1="${pad}" y1="${(H - pad - (70/max)*(H-pad*2)).toFixed(1)}"
          x2="${W-pad}" y2="${(H - pad - (70/max)*(H-pad*2)).toFixed(1)}"
          stroke="var(--danger)" stroke-dasharray="3,3" stroke-width="1"/>
  </svg>`;
}

document.getElementById("btnCloseSummary")?.addEventListener("click", () => {
  summaryModal.classList.add("hidden");
});

// ── Frame loop ────────────────────────────────────────────────────────────────
function processLoop() {
  if (!isMonitoring) return;

  const now = performance.now();
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  if (video.readyState === HTMLMediaElement.HAVE_ENOUGH_DATA && faceLandmarker) {
    const faceResult = faceLandmarker.detectForVideo(video, now);
    const landmarks  = faceResult.faceLandmarks?.[0] ?? null;

    // ── Confidence gating ──────────────────────────────────────────────────
    if (!landmarks) {
      faceLostFrames++;
      if (faceLostFrames >= FACE_LOST_THRESHOLD) {
        updateFaceStatus(false);
        // Don't call processFrame — PERCLOS is NOT updated during face-lost
      }
    } else {
      faceLostFrames = 0;
      updateFaceStatus(true);

      // ── Calibration phase ────────────────────────────────────────────────
      if (isCalibrating) {
        const { done, progress } = calibrationStep(landmarks);
        showCalibOverlay(progress);
        if (done) {
          isCalibrating = false;
          hideCalibOverlay();
        }
        // Still draw overlay during calibration
        drawCalibOverlay(ctx, landmarks);
      } else {
        // ── Phone detection every 5th frame ─────────────────────────────
        if (frameCount % 5 === 0) detectPhone();

        // ── Main detection ───────────────────────────────────────────────
        const result = processFrame(landmarks, phoneDetected, video, frameCount);
        drawOverlay(ctx, landmarks, result);
        updateUI(result);

        if (result.alerts?.length) {
          result.alerts.forEach(a => handleAlert(a, result));
        }
      }
    }

    frameCount++;
  }

  animFrameId = requestAnimationFrame(processLoop);
}

// ── Canvas overlays ───────────────────────────────────────────────────────────
function drawCalibOverlay(ctx, lm) {
  // Just show face mesh in blue — no metrics yet
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "rgba(79,122,255,0.5)";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x*W, p.y*H, 1.5, 0, Math.PI*2);
    ctx.fill();
  }
}

function drawOverlay(ctx, lm, result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const W = canvas.width, H = canvas.height;

  // Face mesh dots
  ctx.fillStyle = "rgba(79,122,255,0.3)";
  for (const p of lm) {
    ctx.beginPath(); ctx.arc(p.x*W, p.y*H, 1.0, 0, Math.PI*2); ctx.fill();
  }

  // Eyes
  const eyeCol = result.eyeClosed ? "#ef4444" : "#22c55e";
  for (const i of [33,160,158,133,153,144, 362,385,387,263,373,380]) {
    ctx.beginPath(); ctx.arc(lm[i].x*W, lm[i].y*H, 2.8, 0, Math.PI*2);
    ctx.fillStyle = eyeCol; ctx.fill();
  }

  // Mouth
  const mouthCol = result.isYawning ? "#f59e0b" : "rgba(255,255,255,0.2)";
  for (const i of [61,291,13,14]) {
    ctx.beginPath(); ctx.arc(lm[i].x*W, lm[i].y*H, 2.5, 0, Math.PI*2);
    ctx.fillStyle = mouthCol; ctx.fill();
  }

  // Yaw arrow from nose
  const nose = lm[1];
  const col  = result.isDistracted ? "#ef4444" : "#22c55e";
  const nx = nose.x*W, ny = nose.y*H;
  const dx = Math.sin((result.yaw||0)*Math.PI/180)*45;
  ctx.strokeStyle = col; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(nx+dx, ny); ctx.stroke();
  ctx.beginPath(); ctx.arc(nx, ny, 4, 0, Math.PI*2);
  ctx.fillStyle = col; ctx.fill();

  // Distraction progress bar (bottom)
  if (result.consecutiveDistrFrames > 5) {
    const pct = Math.min(result.consecutiveDistrFrames/50, 1);
    ctx.fillStyle = `rgba(239,68,68,${0.3+pct*0.5})`;
    ctx.fillRect(0, H-5, W*pct, 5);
  }

  // Eye closure progress bar (top)
  if (result.consecutiveClosedFrames > 5) {
    const pct = Math.min(result.consecutiveClosedFrames/48, 1);
    ctx.fillStyle = `rgba(245,158,11,${0.3+pct*0.5})`;
    ctx.fillRect(0, 0, W*pct, 4);
  }
}

// ── UI updates ────────────────────────────────────────────────────────────────
function updateFaceStatus(detected) {
  if (!faceStatus) return;
  if (detected) {
    faceStatus.textContent = "Face ✓";
    faceStatus.style.color = "var(--safe)";
  } else {
    faceStatus.textContent = "⚠ Face lost";
    faceStatus.style.color = "var(--danger)";
  }
}

function updateUI(result) {
  badgeLowLight?.classList.toggle("hidden", !result.lowLightMode);
  updateStatusBadge(result.riskScore ?? 0);

  // Eye status — human readable, not raw EAR number
  if (eyeStatusEl) {
    const s = result.eyeStatus || "--";
    eyeStatusEl.textContent = s;
    eyeStatusEl.className = `metric-value ${s==="Open"?"ok":s==="Closing"?"warn":"danger"}`;
  }
  // Eye gauge: shows how close to threshold (inverted — lower EAR = fuller bar)
  const earPct = result._ear != null
    ? Math.max(0, Math.min(100, ((result._earThreshold+0.15-result._ear)/0.20)*100))
    : 0;
  const earCls = result.eyeClosed ? "danger" : earPct > 60 ? "warn" : "ok";
  setGauge(eyeGauge, earPct, earCls);

  // PERCLOS
  if (perclosValue) {
    const pPct = Math.round((result.perclos||0)*100);
    const cls  = pPct >= 15 ? "danger" : pPct >= 8 ? "warn" : "ok";
    perclosValue.textContent = `${pPct}%`;
    perclosValue.className   = `metric-value ${cls}`;
    setGauge(perclosGauge, Math.min(100,(result.perclos||0)/0.30*100), cls);
  }

  // Yawn — human readable
  if (yawnStatusEl) {
    const s = result.yawnStatus || "--";
    yawnStatusEl.textContent = s;
    yawnStatusEl.className = `metric-value ${s==="Yawning"?"warn":"ok"}`;
  }
  const marPct = result._mar != null
    ? Math.min(100, (result._mar / (result._marThreshold||0.55) * 0.8) * 100)
    : 0;
  setGauge(yawnGauge, marPct, result.isYawning ? "warn" : "ok");

  // Head pose — yaw only (pitch hidden per design decision)
  if (yawValue) {
    yawValue.textContent = result.yaw != null
      ? `${result.yaw>0?"R":"L"} ${Math.abs(result.yaw).toFixed(0)}°`
      : "--";
  }
  if (poseStatus) {
    const distrPct = Math.min(100, Math.round((result.consecutiveDistrFrames||0)/50*100));
    poseStatus.textContent = result.isDistracted
      ? "⚠ Distracted"
      : distrPct > 10 ? `Off-road ${distrPct}%` : "Forward ✓";
    poseStatus.className = `metric-value ${result.isDistracted?"danger":distrPct>10?"warn":"ok"}`;
  }

  // Phone
  if (phoneStatus) {
    phoneStatus.textContent = result.phoneFlag ? "Detected!" : "None";
    phoneStatus.className   = `metric-value ${result.phoneFlag?"danger":"ok"}`;
  }
}

function setGauge(el, pct, cls) {
  if (!el) return;
  el.style.width  = `${Math.min(100, pct)}%`;
  el.className    = `gauge-fill ${cls}`;
}

function updateStatusBadge(risk) {
  if (riskCircle) riskCircle.textContent = `${risk}%`;
  const level = risk < 30 ? "safe" : risk < 70 ? "warn" : "danger";
  if (riskCircle) riskCircle.className = `risk-circle ${level}`;
  if (riskValue)  riskValue.textContent = level.charAt(0).toUpperCase()+level.slice(1);
  if (badgeStatus) {
    badgeStatus.textContent = level.toUpperCase();
    badgeStatus.className   = `video-badge badge-status badge-${level}`;
  }
}

// ── Alert handling ────────────────────────────────────────────────────────────
async function handleAlert(alert, metrics) {
  addAlertToLog(alert);
  playBeep(alert.type);
  if (currentSessionId) {
    try {
      await apiFetch("/api/events/", {
        method: "POST",
        body: JSON.stringify({
          session_id: currentSessionId,
          event_type: alert.type,
          ear:        metrics._ear,
          mar:        metrics._mar,
          yaw:        metrics.yaw,
          pitch:      null, // pitch not stored — unreliable
          perclos:    metrics.perclos,
          risk_score: metrics.riskScore,
        }),
      });
    } catch (_) {}
  }
}

function addAlertToLog(alert) {
  const empty = alertLog.querySelector(".empty-state");
  if (empty) empty.remove();
  const time  = new Date().toLocaleTimeString("en-US",
    { hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:true });
  const entry = document.createElement("div");
  entry.className = `alert-entry ${alert.type}`;
  entry.innerHTML = `<span class="alert-time">${time}</span>
                     <span class="alert-msg">${alert.message}</span>`;
  alertLog.insertBefore(entry, alertLog.firstChild);
  while (alertLog.children.length > 30) alertLog.removeChild(alertLog.lastChild);
}

function playBeep(type) {
  try {
    const ac   = new (window.AudioContext||window.webkitAudioContext)();
    const osc  = ac.createOscillator();
    const gain = ac.createGain();
    osc.connect(gain); gain.connect(ac.destination);
    osc.frequency.value = {drowsy_eyes:880,yawning:660,phone_detected:1100,
                            head_distraction:770,high_risk:1200}[type]||880;
    osc.type = "sine";
    gain.gain.setValueAtTime(0.3, ac.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ac.currentTime+(type==="high_risk"?0.9:0.45));
    osc.start(); osc.stop(ac.currentTime+1);
  } catch(_) {}
}

// ── Fullscreen ────────────────────────────────────────────────────────────────
btnFullscreen?.addEventListener("click", () => {
  const wrap = document.querySelector(".video-wrap");
  if (!document.fullscreenElement) {
    wrap.requestFullscreen?.();
    btnFullscreen.textContent = "⛶ Exit Fullscreen";
  } else {
    document.exitFullscreen?.();
    btnFullscreen.textContent = "⛶ Fullscreen";
  }
});

// ── Sidebar + init ────────────────────────────────────────────────────────────
const user = getAuthUser();
document.getElementById("sidebarUserName").textContent = user.name || "Driver";
document.getElementById("sidebarUserRole").textContent = user.role || "user";
document.getElementById("sidebarAvatar").textContent   = (user.name||"D")[0].toUpperCase();
document.getElementById("btnLogout").addEventListener("click", () => {
  if (isMonitoring) doStop("");
  authLogout();
});
btnStart.addEventListener("click", startMonitoring);

// Keyboard shortcut: Space = start/stop
document.addEventListener("keydown", e => {
  if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA") {
    e.preventDefault();
    if (!isMonitoring) startMonitoring();
    else requestStop();
  }
});

window.addEventListener("load", () => setTimeout(initModels, 800));