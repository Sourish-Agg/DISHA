// frontend/js/admin.js — full admin dashboard
requireAdmin();

const u = getAuthUser();
document.getElementById("sidebarUserName").textContent = u.name || "Admin";
document.getElementById("sidebarUserRole").textContent = "admin";
document.getElementById("sidebarAvatar").textContent   = (u.name || "A")[0].toUpperCase();
document.getElementById("btnLogout").addEventListener("click", authLogout);

// ── Tab system ────────────────────────────────────────────────────────────────
document.querySelectorAll(".admin-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".admin-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.panel).classList.remove("hidden");
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
const riskClass = r => r < 30 ? "safe" : r < 70 ? "warn" : "danger";

function formatDuration(sec) {
  if (!sec && sec !== 0) return "In progress";
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = Math.floor(sec%60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const TYPE_LABELS = {
  drowsy_eyes:      "😴 Drowsy Eyes",
  yawning:          "🥱 Yawning",
  phone_detected:   "📱 Phone",
  head_distraction: "↩️ Head Distraction",
  high_risk:        "🔴 High Risk",
};

// ── Stats ─────────────────────────────────────────────────────────────────────
let statsData = null;

async function loadStats() {
  try {
    statsData = await apiFetch("/api/admin/stats");
    renderStats(statsData);
  } catch(e) { showToast(e.message, "error"); }
}

function renderStats(s) {
  document.getElementById("statUsers").textContent    = s.total_users;
  document.getElementById("statSessions").textContent = s.total_sessions;
  document.getElementById("statEvents").textContent   = s.total_events;

  // Alert type breakdown with mini bar chart
  const total = Object.values(s.events_by_type).reduce((a,b)=>a+b,0) || 1;
  document.getElementById("eventBreakdown").innerHTML =
    Object.entries(s.events_by_type).map(([type, count]) => {
      const pct = Math.round((count/total)*100);
      return `
        <div style="margin-bottom:.7rem">
          <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:.3rem">
            <span>${TYPE_LABELS[type]||type}</span>
            <span style="color:var(--muted)">${count} (${pct}%)</span>
          </div>
          <div class="gauge-track">
            <div class="gauge-fill ${type==="high_risk"||type==="phone_detected"?"danger":type==="drowsy_eyes"?"warn":"ok"}"
              style="width:${pct}%;transition:width .6s ease"></div>
          </div>
        </div>`;
    }).join("") || '<p style="color:var(--muted);font-size:.82rem">No events yet.</p>';

  // Recent sessions
  document.getElementById("recentSessionsBody").innerHTML =
    (s.recent_sessions || []).map(sess => `
      <tr>
        <td style="font-size:.82rem">${sess.user_name||"—"}</td>
        <td>${sess.driver_name||"—"}</td>
        <td style="font-size:.8rem;color:var(--muted)">${new Date(sess.started_at).toLocaleString()}</td>
        <td>${formatDuration(sess.duration_seconds)}</td>
        <td>${sess.total_alerts||0}</td>
        <td><span class="risk-pill ${riskClass(Math.round(sess.max_risk_score||0))}">${Math.round(sess.max_risk_score||0)}%</span></td>
      </tr>`).join("") || '<tr><td colspan="6" style="color:var(--muted);text-align:center">No sessions yet.</td></tr>';
}

// ── Users ─────────────────────────────────────────────────────────────────────
let allUsers = [];
let userSearch = "";

async function loadUsers() {
  try {
    allUsers = await apiFetch("/api/admin/users");
    renderUsers(allUsers);
  } catch(e) { showToast(e.message, "error"); }
}

function renderUsers(users) {
  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(userSearch) ||
    u.email.toLowerCase().includes(userSearch)
  );

  document.getElementById("userCount").textContent = `${filtered.length} user${filtered.length!==1?"s":""}`;

  document.getElementById("usersBody").innerHTML = filtered.map(u => `
    <tr>
      <td>
        <div style="display:flex;align-items:center;gap:.6rem">
          <div class="user-avatar" style="width:28px;height:28px;font-size:.72rem;flex-shrink:0">
            ${u.name[0].toUpperCase()}
          </div>
          <div>
            <div style="font-size:.85rem">${u.name}</div>
            <div style="font-size:.73rem;color:var(--muted)">${u.email}</div>
          </div>
        </div>
      </td>
      <td><span class="role-badge ${u.role}">${u.role}</span></td>
      <td><span class="risk-pill ${u.is_active?"safe":"danger"}">${u.is_active?"Active":"Disabled"}</span></td>
      <td style="font-size:.8rem;color:var(--muted)">${new Date(u.created_at).toLocaleDateString()}</td>
      <td>
        <div style="display:flex;gap:.35rem;flex-wrap:wrap">
          ${u.role!=="admin"
            ?`<button class="btn btn-ghost" style="padding:.22rem .6rem;font-size:.73rem"
                onclick="toggleRole('${u.id}','${u.role}')">
                ${u.role==="user"?"→ Admin":"→ User"}
              </button>`:""}
          <button class="btn btn-ghost" style="padding:.22rem .6rem;font-size:.73rem"
            onclick="toggleActive('${u.id}',${u.is_active})">
            ${u.is_active?"Disable":"Enable"}
          </button>
          <button class="btn btn-danger" style="padding:.22rem .6rem;font-size:.73rem"
            onclick="confirmDeleteUser('${u.id}','${u.name}')">
            Delete
          </button>
        </div>
      </td>
    </tr>`).join("") ||
    '<tr><td colspan="5" class="empty-state">No users match your search.</td></tr>';
}

// Search
document.getElementById("userSearchInput")?.addEventListener("input", e => {
  userSearch = e.target.value.toLowerCase();
  renderUsers(allUsers);
});

async function toggleRole(uid, cur) {
  try {
    await apiFetch(`/api/admin/users/${uid}`,{method:"PATCH",body:JSON.stringify({role:cur==="user"?"admin":"user"})});
    showToast("Role updated.", "success");
    loadUsers(); loadStats();
  } catch(e){ showToast(e.message,"error"); }
}

async function toggleActive(uid, active) {
  try {
    await apiFetch(`/api/admin/users/${uid}`,{method:"PATCH",body:JSON.stringify({is_active:!active})});
    showToast(`User ${active?"disabled":"enabled"}.`, "success");
    loadUsers();
  } catch(e){ showToast(e.message,"error"); }
}

// Delete with modal confirmation
function confirmDeleteUser(uid, name) {
  const modal = document.getElementById("confirmModal");
  document.getElementById("confirmMsg").textContent =
    `Delete "${name}" and ALL their sessions and events? This cannot be undone.`;
  modal.classList.remove("hidden");

  document.getElementById("confirmYes").onclick = async () => {
    modal.classList.add("hidden");
    try {
      await apiFetch(`/api/admin/users/${uid}`,{method:"DELETE"});
      showToast(`"${name}" deleted.`, "success");
      loadUsers(); loadStats();
    } catch(e){ showToast(e.message,"error"); }
  };
  document.getElementById("confirmNo").onclick = () => modal.classList.add("hidden");
}

// ── Events modal ──────────────────────────────────────────────────────────────
const modalOverlay = document.getElementById("modalOverlay");
document.getElementById("btnCloseModal")?.addEventListener("click", ()=>modalOverlay.classList.add("hidden"));
modalOverlay?.addEventListener("click", e=>{ if(e.target===modalOverlay) modalOverlay.classList.add("hidden"); });

async function viewEvents(sessionId, label) {
  document.getElementById("modalTitle").textContent = `Events — ${label}`;
  document.getElementById("eventsList").innerHTML = '<p style="color:var(--muted)">Loading…</p>';
  modalOverlay.classList.remove("hidden");
  try {
    const events = await apiFetch(`/api/events/${sessionId}`);
    if (!events.length) {
      document.getElementById("eventsList").innerHTML =
        '<p class="empty-state"><span class="empty-icon">✅</span>No alerts in this session.</p>';
      return;
    }
    document.getElementById("eventsList").innerHTML = events.map(e=>`
      <div class="alert-entry ${e.event_type}" style="margin-bottom:.3rem">
        <span class="alert-time">${new Date(e.timestamp).toLocaleTimeString()}</span>
        <span class="alert-msg">
          ${TYPE_LABELS[e.event_type]||e.event_type}
          &nbsp;·&nbsp; risk ${Math.round(e.risk_score)}%
          ${e.ear!=null?`&nbsp;·&nbsp; EAR ${e.ear.toFixed(2)}`:""}
          ${e.mar!=null?`&nbsp;·&nbsp; MAR ${e.mar.toFixed(2)}`:""}
          ${e.yaw!=null?`&nbsp;·&nbsp; Yaw ${e.yaw.toFixed(0)}°`:""}
        </span>
      </div>`).join("");
  } catch(e){
    document.getElementById("eventsList").innerHTML = `<p style="color:var(--danger)">${e.message}</p>`;
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadStats();
loadUsers();

// ── Alert trend chart (past 7 days) ──────────────────────────────────────────
async function loadTrend() {
  try {
    const trend = await apiFetch("/api/analytics/trend");
    renderTrendChart(trend);
  } catch(e) { console.warn("Trend load failed:", e); }
}

function renderTrendChart(trend) {
  const el = document.getElementById("trendChart");
  if (!el) return;

  const days   = Object.keys(trend);
  const types  = ["drowsy_eyes","yawning","head_distraction","phone_detected","high_risk"];
  const colors = { drowsy_eyes:"#f59e0b", yawning:"#3b82f6",
                   head_distraction:"#8b5cf6", phone_detected:"#ef4444", high_risk:"#dc2626" };
  const labels = { drowsy_eyes:"Drowsy", yawning:"Yawning",
                   head_distraction:"Distraction", phone_detected:"Phone", high_risk:"High Risk" };

  if (!days.length) { el.innerHTML = '<p style="color:var(--muted);font-size:.82rem">No data yet.</p>'; return; }

  const maxVal = Math.max(...days.flatMap(d => types.map(t => trend[d]?.[t]||0)), 1);
  const W = 480, H = 140, padL = 28, padB = 24, padT = 10, padR = 10;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const barGroupW = chartW / days.length;
  const barW = Math.min(10, barGroupW / (types.length + 1));

  let svgLines = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:${H}px;overflow:visible">`;

  // Y-axis grid lines
  for (let g = 0; g <= 4; g++) {
    const y = padT + chartH - (g/4)*chartH;
    const val = Math.round((g/4)*maxVal);
    svgLines += `<line x1="${padL}" y1="${y.toFixed(0)}" x2="${W-padR}" y2="${y.toFixed(0)}"
      stroke="var(--border)" stroke-width="1"/>
      <text x="${padL-3}" y="${(y+4).toFixed(0)}" font-size="8" fill="var(--muted)"
        text-anchor="end">${val}</text>`;
  }

  // Bars
  days.forEach((day, di) => {
    const groupX = padL + di * barGroupW + barGroupW * 0.1;
    types.forEach((type, ti) => {
      const val  = trend[day]?.[type] || 0;
      const barH = (val / maxVal) * chartH;
      const x    = groupX + ti * (barW + 1);
      const y    = padT + chartH - barH;
      if (val > 0)
        svgLines += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}"
          width="${barW}" height="${barH.toFixed(1)}"
          fill="${colors[type]}" opacity="0.85" rx="1">
          <title>${labels[type]}: ${val}</title></rect>`;
    });
    // X-axis label
    const labelX = padL + di * barGroupW + barGroupW/2;
    svgLines += `<text x="${labelX.toFixed(0)}" y="${H-6}" font-size="8"
      fill="var(--muted)" text-anchor="middle">${day}</text>`;
  });

  // Legend
  let legendX = padL;
  types.forEach(type => {
    svgLines += `<rect x="${legendX}" y="${H+8}" width="8" height="8" fill="${colors[type]}" rx="1"/>
      <text x="${legendX+10}" y="${H+16}" font-size="8" fill="var(--muted)">${labels[type]}</text>`;
    legendX += 80;
  });

  svgLines += `</svg>`;
  el.innerHTML = svgLines;
}

// ── CSV export ────────────────────────────────────────────────────────────────
document.getElementById("btnExportCSV")?.addEventListener("click", async () => {
  try {
    const token = localStorage.getItem("disha_token");
    const res   = await fetch("http://127.0.0.1:8000/api/analytics/export/csv", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) throw new Error("Export failed.");
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `disha_export_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("CSV downloaded.", "success");
  } catch(e) { showToast(e.message, "error"); }
});

// Load trend alongside stats
loadTrend();