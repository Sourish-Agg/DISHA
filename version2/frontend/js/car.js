// frontend/js/car.js — D.I.S.H.A. ESP32 Car Control Module
// ═══════════════════════════════════════════════════════════════
// Reads detection results from processFrame() and sends HTTP
// commands to the ESP32 over WiFi. Runs independently — no
// changes needed to detection.js.
// ═══════════════════════════════════════════════════════════════

const CAR_ESP32_IP = "10.27.91.9";  // ← Update if IP changes
const CAR_BASE_URL = `http://${CAR_ESP32_IP}`;

// ── Throttle: don't spam the ESP32 with identical commands ───
let _lastCarCmd  = null;
let _lastCmdTime = 0;
const CMD_MIN_INTERVAL_MS = 300;  // minimum ms between commands

// ── Ping interval to keep watchdog alive during "normal" ─────
let _pingInterval = null;
const PING_INTERVAL_MS = 3000;  // ping every 3s to prevent watchdog timeout (5s)

// ── Send command to ESP32 (fire-and-forget) ──────────────────
async function sendCarCommand(cmd) {
  const now = Date.now();

  // Skip if same command sent recently (avoid flooding)
  if (cmd === _lastCarCmd && (now - _lastCmdTime) < CMD_MIN_INTERVAL_MS) {
    return;
  }

  _lastCarCmd  = cmd;
  _lastCmdTime = now;

  try {
    await fetch(`${CAR_BASE_URL}/${cmd}`, {
      method: "GET",
      mode: "cors",
      signal: AbortSignal.timeout(2000),  // 2s timeout
    });
    console.log(`[DISHA-CAR] → /${cmd}`);
  } catch (err) {
    // Non-fatal — ESP32 might be briefly unreachable
    console.warn(`[DISHA-CAR] Failed /${cmd}:`, err.message);
  }
}

// ── Decision logic: detection result → car command ───────────
// Priority order (highest to lowest):
//   1. No face detected     → stop
//   2. Drowsy (multi-cue)   → stop
//   3. Risk ≥ 70%           → stop
//   4. Yawning              → warn (slow down)
//   5. Distracted (head)    → warn (slow down)
//   6. Phone detected       → beep (alert only, don't stop)
//   7. Everything OK        → normal (full speed)
function decideCarCommand(result) {
  if (!result.faceDetected)               return "stop";
  if (result.isDrowsy)                    return "stop";
  if (result.riskScore >= 70)             return "stop";
  if (result.isYawning || result.isDistracted) return "warn";
  if (result.phoneDetected)               return "beep";
  return "normal";
}

// ── Public API (called from monitor.js) ──────────────────────
window.DishaCar = {

  // Called after every processFrame() result
  updateCar(result) {
    const cmd = decideCarCommand(result);
    sendCarCommand(cmd);

    // If car is in "normal" state, keep pinging to prevent watchdog
    if (cmd === "normal") {
      if (!_pingInterval) {
        _pingInterval = setInterval(() => {
          if (_lastCarCmd === "normal") {
            sendCarCommand("ping");
            // Reset _lastCarCmd so the next real command isn't skipped
            _lastCarCmd = "ping";
          }
        }, PING_INTERVAL_MS);
      }
    } else {
      // Non-normal command — clear ping interval
      if (_pingInterval) {
        clearInterval(_pingInterval);
        _pingInterval = null;
      }
    }
  },

  // Emergency stop (face lost, session ended, etc.)
  stopCar() {
    if (_pingInterval) {
      clearInterval(_pingInterval);
      _pingInterval = null;
    }
    _lastCarCmd = null;  // force send even if last cmd was "stop"
    sendCarCommand("stop");
  },
};

console.log(`[DISHA-CAR] Car control loaded — target: ${CAR_BASE_URL}`);
