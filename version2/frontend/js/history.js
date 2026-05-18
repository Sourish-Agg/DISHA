// frontend/js/history.js
// Renders the session history page.

requireUser();

const user = getAuthUser();
document.getElementById("sidebarUserName").textContent = user.name || "Driver";
document.getElementById("sidebarUserRole").textContent = user.role || "user";
document.getElementById("sidebarAvatar").textContent   = (user.name || "D")[0].toUpperCase();
if (user.role !== "admin") {
  document.querySelectorAll(".admin-only").forEach(el => el.classList.add("hidden"));
}
document.getElementById("btnLogout").addEventListener("click", authLogout);

const tableBody   = document.getElementById("sessionsBody");
const emptyState  = document.getElementById("emptyState");
const modalOverlay = document.getElementById("modalOverlay");
const modalTitle   = document.getElementById("modalTitle");
const eventsList   = document.getElementById("eventsList");
const btnCloseModal = document.getElementById("btnCloseModal");

// ── Load sessions ─────────────────────────────────────────────────────────────
async function loadSessions() {
  try {
    const sessions = await apiFetch("/api/sessions/");
    renderSessions(sessions);
  } catch (e) {
    showToast(e.message, "error");
  }
}

function renderSessions(sessions) {
  tableBody.innerHTML = "";

  if (!sessions.length) {
    emptyState.classList.remove("hidden");
    return;
  }
  emptyState.classList.add("hidden");

  sessions.forEach(s => {
    const risk = Math.round(s.max_risk_score);
    const riskClass = risk < 30 ? "safe" : risk < 70 ? "warn" : "danger";
    const duration  = s.duration_seconds
      ? formatDuration(s.duration_seconds)
      : "In progress";
    const started   = new Date(s.started_at).toLocaleString();

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${started}</td>
      <td>${s.driver_name || "—"}</td>
      <td>${duration}</td>
      <td>${s.total_alerts}</td>
      <td><span class="risk-pill ${riskClass}">${risk}%</span></td>
      <td>
        <button class="btn btn-ghost" style="padding:.3rem .8rem;font-size:.78rem"
          onclick="viewEvents('${s.id}', '${started}')">
          View Events
        </button>
      </td>
    `;
    tableBody.appendChild(tr);
  });
}

// ── View events modal ─────────────────────────────────────────────────────────
async function viewEvents(sessionId, label) {
  modalTitle.textContent = `Events — ${label}`;
  eventsList.innerHTML   = '<p style="color:var(--muted);font-size:.85rem">Loading…</p>';
  modalOverlay.classList.remove("hidden");

  try {
    const events = await apiFetch(`/api/events/${sessionId}`);
    renderEvents(events);
  } catch (e) {
    eventsList.innerHTML = `<p style="color:var(--danger)">${e.message}</p>`;
  }
}

function renderEvents(events) {
  if (!events.length) {
    eventsList.innerHTML = '<p class="empty-state"><span class="empty-icon">✅</span>No alerts in this session.</p>';
    return;
  }

  const typeLabels = {
    drowsy_eyes:      "😴 Drowsy Eyes",
    yawning:          "🥱 Yawning",
    phone_detected:   "📱 Phone Detected",
    head_distraction: "↩️ Head Distraction",
    high_risk:        "🔴 High Risk",
  };

  eventsList.innerHTML = events.map(e => `
    <div class="alert-entry ${e.event_type}" style="margin-bottom:.3rem">
      <span class="alert-time">${new Date(e.timestamp).toLocaleTimeString()}</span>
      <span class="alert-msg">
        ${typeLabels[e.event_type] || e.event_type}
        &nbsp;·&nbsp; risk ${Math.round(e.risk_score)}%
        ${e.ear != null ? `&nbsp;·&nbsp; EAR ${e.ear.toFixed(2)}` : ""}
        ${e.mar != null ? `&nbsp;·&nbsp; MAR ${e.mar.toFixed(2)}` : ""}
      </span>
    </div>
  `).join("");
}

function formatDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

btnCloseModal.addEventListener("click", () => modalOverlay.classList.add("hidden"));
modalOverlay.addEventListener("click", e => {
  if (e.target === modalOverlay) modalOverlay.classList.add("hidden");
});

loadSessions();