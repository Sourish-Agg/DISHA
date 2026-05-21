// frontend/js/monitor.js — v4
// Full monitor page orchestrator:
//  • 3-phase flow: CALIBRATION → MONITORING → SESSION SUMMARY
//  • Confidence gating: face-lost state pauses PERCLOS
//  • YOLOv8 phone detection (server-side, every 5th frame)
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
// Phone detection is server-side — no local model state needed
let frameCount        = 0;
let currentSessionId  = null;
let animFrameId       = null;
let stream            = null;
let modelsReady       = false;
let phoneDetected     = false;
let phoneConf         = 0;      // raw YOLOv8 confidence (0 when no phone) — passed to processFrame
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

  // 2. Phone detection is handled server-side via /api/phone/detect
  // No browser model loading needed — backend runs YOLOv8n.pt (same as version1)
  console.log("[DISHA] Phone detection: server-side YOLOv8n via /api/phone/detect");

  modelsReady = true;
  console.log("[DISHA] All models ready.");
}

// ── Server-side YOLOv8 phone detection ───────────────────────────────────────
// Captures a JPEG frame and sends it to /api/phone/detect (backend runs YOLOv8n.pt)
// Same approach as version1/app.py — reliable, no browser ONNX needed.
async function detectPhone() {
  try {
    // Capture current video frame to a small canvas
    const offsc  = document.createElement("canvas");
    offsc.width  = 320;   // downscale for faster transfer
    offsc.height = 240;
    const offCtx = offsc.getContext("2d");
    offCtx.drawImage(video, 0, 0, 320, 240);

    // Get base64 JPEG
    const dataUrl   = offsc.toDataURL("image/jpeg", 0.7);
    const frame_b64 = dataUrl.split(",")[1];

    const result = await apiFetch("/api/phone/detect", {
      method: "POST",
      body:   JSON.stringify({ frame_b64 }),
    });

    // Always sync local confidence to the latest result. The backend returns
    // confidence 0.0 when no phone is found (or available=false), so this
    // naturally decays phoneConf back to 0 instead of leaving a stale value.
    phoneConf = result.available ? (result.confidence || 0) : 0;
  } catch (_) {
    // Non-fatal — treat a failed/timed-out detect as "no phone this cycle".
    phoneConf = 0;
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
    phoneConf      = 0;

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
    const sid = currentSessionId;
    currentSessionId = null;  // clear immediately so logout/reload doesn't double-end

    try {
      // Only send body if there are actually notes — avoids body parsing edge cases
      const patchOptions = { method: "PATCH" };
      if (notes && notes.trim()) {
        patchOptions.body = JSON.stringify({ notes: notes.trim() });
      }
      await apiFetch(`/api/sessions/${sid}/end`, patchOptions);
      // Load and show session summary
      await showSessionSummary(sid);
    } catch (e) {
      console.error("[DISHA] Session end failed:", e);
      showToast("Session saved but summary unavailable: " + e.message, "warning");
    }
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
  // Use the canvas element's displayed size for drawing coordinates
  // This ensures landmarks (normalised 0-1) map correctly onto the visible area
  const rect = canvas.getBoundingClientRect();
  if (rect.width > 0) {
    canvas.width  = rect.width;
    canvas.height = rect.height;
  } else {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
  }

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
        const { done, progress } = calibrationStep(landmarks, now);
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
        try {
          const result = processFrame(landmarks, phoneConf, video, frameCount, now);
          drawOverlay(ctx, landmarks, result);
          updateUI(result);
          if (result.alerts?.length) {
            result.alerts.forEach(a => handleAlert(a, result));
          }
        } catch(err) {
          // Surface errors visibly instead of silently failing
          console.error("[DISHA] processFrame error:", err);
          if (faceStatus) {
            faceStatus.textContent = "⚠ Error: " + err.message;
            faceStatus.style.color = "var(--danger)";
          }
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

  // ── Face mesh (subtle dots, skip every other for performance) ──────────
  ctx.fillStyle = "rgba(79,122,255,0.15)";
  for (let i = 0; i < lm.length; i += 2) {
    ctx.beginPath(); ctx.arc(lm[i].x*W, lm[i].y*H, 0.8, 0, Math.PI*2); ctx.fill();
  }

  // ── Eye contours (polyline — like version1) ──────────────────────────
  const eyeCol = (result.eyeStatus==="Drowsy"||result.eyeStatus==="Closing") ? "#ef4444" : "#22c55e";
  ctx.strokeStyle = eyeCol;
  ctx.lineWidth = 1.5;

  // Left eye contour
  const leftEyeContour = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
  ctx.beginPath();
  leftEyeContour.forEach((idx, i) => {
    const p = lm[idx];
    i === 0 ? ctx.moveTo(p.x*W, p.y*H) : ctx.lineTo(p.x*W, p.y*H);
  });
  ctx.closePath(); ctx.stroke();

  // Right eye contour
  const rightEyeContour = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
  ctx.beginPath();
  rightEyeContour.forEach((idx, i) => {
    const p = lm[idx];
    i === 0 ? ctx.moveTo(p.x*W, p.y*H) : ctx.lineTo(p.x*W, p.y*H);
  });
  ctx.closePath(); ctx.stroke();

  // Iris dots (version1 draws pupil indicators)
  const irisCol = eyeCol;
  for (const idx of [468, 473]) { // left iris center, right iris center
    if (lm[idx]) {
      ctx.beginPath(); ctx.arc(lm[idx].x*W, lm[idx].y*H, 3, 0, Math.PI*2);
      ctx.fillStyle = irisCol; ctx.globalAlpha = 0.6; ctx.fill(); ctx.globalAlpha = 1.0;
    }
  }

  // ── Mouth contour (outer lip — like version1) ────────────────────────
  const mouthCol = result.isYawning ? "#f59e0b" : "rgba(255,200,0,0.5)";
  const outerLip = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185];
  ctx.strokeStyle = mouthCol;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  outerLip.forEach((idx, i) => {
    const p = lm[idx];
    i === 0 ? ctx.moveTo(p.x*W, p.y*H) : ctx.lineTo(p.x*W, p.y*H);
  });
  ctx.closePath(); ctx.stroke();

  // ── Face bounding box (like version1's cv2.rectangle) ────────────────
  const xs = [], ys = [];
  for (let i = 0; i < Math.min(468, lm.length); i++) { xs.push(lm[i].x); ys.push(lm[i].y); }
  const bx1 = Math.max(0, Math.min(...xs) * W - 10);
  const by1 = Math.max(0, Math.min(...ys) * H - 10);
  const bx2 = Math.min(W, Math.max(...xs) * W + 10);
  const by2 = Math.min(H, Math.max(...ys) * H + 10);
  const bboxCol = result.isDrowsy ? "#ef4444" : result.isDistracted ? "#f59e0b" : "rgba(0,200,240,0.5)";

  ctx.strokeStyle = bboxCol;
  ctx.lineWidth = 1;
  ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);

  // ── HUD text on canvas (like version1's cv2.putText) ─────────────────
  ctx.font = "11px 'Share Tech Mono', monospace";
  ctx.fillStyle = bboxCol;
  const hudText = `EAR:${(result._ear||0).toFixed(2)} MAR:${(result._mar||0).toFixed(2)} Y:${(result.yaw||0).toFixed(0)}° RISK:${result.riskScore||0}`;
  ctx.fillText(hudText, bx1 + 2, Math.max(by1 - 6, 12));

  // ── Yaw direction arrow from nose ────────────────────────────────────
  const nose = lm[1];
  const yawCol = result.isDistracted ? "#ef4444" : "#22c55e";
  const nx = nose.x*W, ny = nose.y*H;
  const dx = Math.sin((result.yaw||0)*Math.PI/180) * 40;
  ctx.strokeStyle = yawCol; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(nx+dx, ny); ctx.stroke();
  ctx.beginPath(); ctx.arc(nx, ny, 3, 0, Math.PI*2);
  ctx.fillStyle = yawCol; ctx.fill();

  // ── Progress bars ────────────────────────────────────────────────────
  // Distraction progress bar (bottom)
  if ((result.headProgress || 0) > 0.1) {
    const pct = result.headProgress;
    ctx.fillStyle = `rgba(239,68,68,${0.3+pct*0.5})`;
    ctx.fillRect(0, H-5, W*pct, 5);
  }

  // Eye closure progress bar (top)
  if ((result.drowsyProgress || 0) > 0.1) {
    const pct = result.drowsyProgress;
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

  // Eye status — show multi-cue count for transparency
  if (eyeStatusEl) {
    const s = result.eyeStatus || "--";
    const cueCount = result.activeCueCount || 0;
    // Show cue count when eyes are closing or drowsy so user understands
    // why/why-not an alert fires (e.g. "Closing (1/2)" means 1 cue active, need 2)
    const label = s === "Open" ? s
      : s === "Drowsy" ? `Drowsy (${cueCount}/2 cues)`
      : `${s} (${cueCount}/2)`;
    eyeStatusEl.textContent = label;
    eyeStatusEl.className   = `metric-value ${s==="Open"?"ok":s==="Closing"?"warn":"danger"}`;
  }
  // Eye gauge: shows how close to threshold (inverted — lower EAR = fuller bar)
  const earPct = result._ear != null
    ? Math.max(0, Math.min(100, ((result._effectiveEARThresh+0.12 - result._ear)/0.18)*100))
    : 0;
  const earCls = result.eyeStatus==="Drowsy" ? "danger" : earPct > 60 ? "warn" : "ok";
  setGauge(eyeGauge, earPct, earCls);

  // PERCLOS
  if (perclosValue) {
    const pPct = Math.round(result.perclosPct || (result.perclos||0)*100);
    const cls  = pPct >= 15 ? "danger" : pPct >= 8 ? "warn" : "ok";  // pPct is 0-100
    perclosValue.textContent = `${pPct}%`;
    perclosValue.className   = `metric-value ${cls}`;
    setGauge(perclosGauge, Math.min(100, (pPct/15)*100), cls);
  }

  // Yawn — human readable
  if (yawnStatusEl) {
    const s = result.yawnStatus || "--";
    yawnStatusEl.textContent = s;
    yawnStatusEl.className = `metric-value ${s==="Yawning"?"warn":"ok"}`;
  }
  const marPct = result._mar != null ? Math.min(100,(result._mar/0.50)*100) : 0;
  setGauge(yawnGauge, marPct, result.isYawning ? "warn" : "ok");

  // Head pose — yaw only (pitch hidden per design decision)
  if (yawValue) {
    yawValue.textContent = result.yaw != null
      ? `${result.yaw>0?"R":"L"} ${Math.abs(result.yaw).toFixed(0)}°`
      : "--";
  }
  if (poseStatus) {
    const distrPct = Math.round((result.headProgress || 0) * 100);
    poseStatus.textContent = result.isDistracted
      ? "⚠ Distracted"
      : distrPct > 10 ? `Off-road ${distrPct}%` : "Forward ✓";
    poseStatus.className = `metric-value ${result.isDistracted?"danger":distrPct>10?"warn":"ok"}`;
  }

  // Phone
  if (phoneStatus) {
    phoneStatus.textContent = result.phoneDetected ? `Detected (${(result.phoneConf*100).toFixed(0)}%)` : "None";
    phoneStatus.className   = `metric-value ${result.phoneDetected?"danger":"ok"}`;
  }

  // Temporal scores with multi-cue indicators (if elements exist)
  const etEl = document.getElementById("eyeTemporal");
  const mtEl = document.getElementById("mouthTemporal");
  const htEl = document.getElementById("headTemporal");
  if (etEl && result.eyeTemporal != null) {
    const cue3 = result.drowsyCues?.temporal;
    etEl.textContent = result.eyeTemporal.toFixed(2) + (cue3 ? " ●" : "");
    etEl.style.color = cue3 ? "var(--danger)" : result.eyeTemporal > 0.2 ? "var(--warn)" : "var(--safe)";
  }
  if (mtEl && result.mouthTemporal != null) {
    mtEl.textContent = result.mouthTemporal.toFixed(2);
    mtEl.style.color = result.mouthTemporal > 0.4 ? "var(--warn)" : "var(--safe)";
  }
  if (htEl && result.headTemporal != null) {
    htEl.textContent = result.headTemporal.toFixed(2);
    htEl.style.color = result.headTemporal > 0.4 ? "var(--warn)" : "var(--safe)";
  }

  // Blink rate (shown below PERCLOS if element exists — user can add
  // <span id="blinkRate"> to index.html's metrics panel)
  const brEl = document.getElementById("blinkRate");
  if (brEl && result.blinksPerMin != null) {
    const bpm = result.blinksPerMin;
    const dur = result.avgBlinkDur || 150;
    brEl.textContent = `${bpm}/min · ${dur}ms avg`;
    brEl.style.color = bpm > 20 || dur > 300 ? "var(--warn)" : bpm < 3 ? "var(--danger)" : "var(--safe)";
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
          perclos:    (metrics.perclosPct||0)/100,
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