// frontend/js/admin.js — v3
// User-centric admin dashboard.
// Overview: KPIs + alert trend chart
// Users tab: each user row expands to show THEIR sessions + events

requireAdmin();

const u = getAuthUser();
document.getElementById("sidebarUserName").textContent = u.name || "Admin";
document.getElementById("sidebarUserRole").textContent = "admin";
document.getElementById("sidebarAvatar").textContent   = (u.name||"A")[0].toUpperCase();
document.getElementById("btnLogout").addEventListener("click", authLogout);

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".admin-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".admin-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.panel).classList.remove("hidden");
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
const riskCls = r => r<30?"safe":r<70?"warn":"danger";
function fmtDur(sec) {
  if(!sec && sec!==0) return "—";
  const m=Math.floor(sec/60), s=Math.floor(sec%60);
  return m>0?`${m}m ${s}s`:`${s}s`;
}
function fmtTime(iso) { return new Date(iso).toLocaleString(undefined,{dateStyle:"short",timeStyle:"short"}); }

const TYPE_LABELS = {
  drowsy_eyes:"😴 Drowsy",yawning:"🥱 Yawn",
  phone_detected:"📱 Phone",head_distraction:"↩️ Distraction",high_risk:"🔴 High Risk"
};

// ── Fix stuck sessions ────────────────────────────────────────────────────────
document.getElementById("btnCleanup")?.addEventListener("click", async () => {
  if (!confirm("Close all sessions currently showing as \"In progress\"?")) return;
  try {
    const r = await apiFetch("/api/admin/cleanup-sessions", { method:"POST" });
    showToast(r.message, "success", 5000);
    loadStats(); loadUsers();
  } catch(e) { showToast(e.message,"error"); }
});

// ── Stats (Overview tab) ──────────────────────────────────────────────────────
async function loadStats() {
  try {
    const s = await apiFetch("/api/admin/stats");
    document.getElementById("statUsers").textContent    = s.total_users;
    document.getElementById("statSessions").textContent = s.total_sessions;
    document.getElementById("statEvents").textContent   = s.total_events;

    // Alert breakdown bars
    const total = Object.values(s.events_by_type).reduce((a,b)=>a+b,0)||1;
    document.getElementById("eventBreakdown").innerHTML =
      Object.entries(s.events_by_type).map(([t,c])=>{
        const pct=Math.round(c/total*100);
        return `<div style="margin-bottom:.6rem">
          <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:.25rem">
            <span>${TYPE_LABELS[t]||t}</span>
            <span style="color:var(--muted)">${c} (${pct}%)</span>
          </div>
          <div class="gauge-track">
            <div class="gauge-fill ${t==="high_risk"||t==="phone_detected"?"danger":t==="drowsy_eyes"?"warn":"ok"}"
              style="width:${pct}%;transition:width .5s"></div>
          </div>
        </div>`;
      }).join("") || '<p style="color:var(--muted);font-size:.82rem">No events yet.</p>';
  } catch(e) { showToast(e.message,"error"); }
}

// ── Alert trend chart (last 7 days) ──────────────────────────────────────────
async function loadTrend() {
  try {
    const trend = await apiFetch("/api/analytics/trend");
    renderTrendChart(trend);
  } catch(e) { console.warn("Trend failed:", e); }
}

function renderTrendChart(trend) {
  const el = document.getElementById("trendChart");
  if (!el) return;
  const days   = Object.keys(trend);
  const types  = ["drowsy_eyes","yawning","head_distraction","phone_detected","high_risk"];
  const colors = {drowsy_eyes:"#f59e0b",yawning:"#3b82f6",head_distraction:"#8b5cf6",phone_detected:"#ef4444",high_risk:"#dc2626"};
  const labels = {drowsy_eyes:"Drowsy",yawning:"Yawn",head_distraction:"Distraction",phone_detected:"Phone",high_risk:"High Risk"};

  if (!days.length) { el.innerHTML='<p style="color:var(--muted);font-size:.82rem">No trend data yet.</p>'; return; }

  const maxVal = Math.max(...days.flatMap(d=>types.map(t=>trend[d]?.[t]||0)),1);
  const W=480,H=140,pL=28,pB=24,pT=10,pR=10;
  const cW=W-pL-pR, cH=H-pT-pB;

  let svg = `<svg viewBox="0 0 ${W} ${H+22}" style="width:100%;overflow:visible">`;
  // Grid lines
  for(let g=0;g<=4;g++){
    const y=pT+cH-(g/4)*cH;
    svg+=`<line x1="${pL}" y1="${y.toFixed(0)}" x2="${W-pR}" y2="${y.toFixed(0)}" stroke="var(--border)" stroke-width="1"/>
      <text x="${pL-3}" y="${(y+4).toFixed(0)}" font-size="8" fill="var(--muted)" text-anchor="end">${Math.round(g/4*maxVal)}</text>`;
  }
  // Bars
  const bGrpW=cW/days.length;
  const bW=Math.min(9,bGrpW/(types.length+1));
  days.forEach((day,di)=>{
    const gx=pL+di*bGrpW+bGrpW*0.1;
    types.forEach((type,ti)=>{
      const val=trend[day]?.[type]||0;
      const bH=(val/maxVal)*cH;
      const x=gx+ti*(bW+1);
      const y=pT+cH-bH;
      if(val>0) svg+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bW}" height="${bH.toFixed(1)}" fill="${colors[type]}" opacity="0.85" rx="1"><title>${labels[type]}: ${val}</title></rect>`;
    });
    const lx=pL+di*bGrpW+bGrpW/2;
    svg+=`<text x="${lx.toFixed(0)}" y="${H-6}" font-size="8" fill="var(--muted)" text-anchor="middle">${day}</text>`;
  });
  // Legend
  let lx=pL;
  types.forEach(t=>{
    svg+=`<rect x="${lx}" y="${H+6}" width="8" height="8" fill="${colors[t]}" rx="1"/>
      <text x="${lx+10}" y="${H+14}" font-size="8" fill="var(--muted)">${labels[t]}</text>`;
    lx+=80;
  });
  svg+="</svg>";
  el.innerHTML=svg;
}

// ── CSV export ────────────────────────────────────────────────────────────────
document.getElementById("btnExportCSV")?.addEventListener("click", async () => {
  try {
    const token = localStorage.getItem("disha_token");
    const res   = await fetch("http://127.0.0.1:8000/api/analytics/export/csv",
      { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw new Error("Export failed.");
    const blob = await res.blob();
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `disha_export_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    showToast("CSV downloaded.", "success");
  } catch(e) { showToast(e.message,"error"); }
});

// ── Users tab — user-centric with drill-down ──────────────────────────────────
let allUsers = [], userSearch = "";

async function loadUsers() {
  try {
    allUsers = await apiFetch("/api/admin/users");
    renderUsers(allUsers);
  } catch(e) { showToast(e.message,"error"); }
}

function renderUsers(users) {
  const filtered = users.filter(u=>
    u.name.toLowerCase().includes(userSearch)||
    u.email.toLowerCase().includes(userSearch)
  );
  document.getElementById("userCount").textContent = `${filtered.length} user${filtered.length!==1?"s":""}`;

  document.getElementById("usersBody").innerHTML = filtered.map(u=>`
    <tr class="user-row" data-uid="${u.id}">
      <td>
        <div style="display:flex;align-items:center;gap:.6rem">
          <div class="user-avatar" style="width:30px;height:30px;font-size:.75rem;flex-shrink:0">
            ${u.name[0].toUpperCase()}
          </div>
          <div>
            <div style="font-size:.87rem;font-weight:500">${u.name}</div>
            <div style="font-size:.73rem;color:var(--muted)">${u.email}</div>
          </div>
        </div>
      </td>
      <td><span class="role-badge ${u.role}">${u.role}</span></td>
      <td><span class="risk-pill ${u.is_active?"safe":"danger"}">${u.is_active?"Active":"Disabled"}</span></td>
      <td style="font-size:.8rem;color:var(--muted)">${new Date(u.created_at).toLocaleDateString()}</td>
      <td>
        <div style="display:flex;gap:.35rem;flex-wrap:wrap;align-items:center">
          <button class="btn btn-ghost" style="padding:.22rem .6rem;font-size:.73rem"
            onclick="viewUserDetail('${u.id}','${u.name}')">
            📊 Details
          </button>
          ${u.role!=="admin"?`<button class="btn btn-ghost" style="padding:.22rem .6rem;font-size:.73rem"
            onclick="toggleRole('${u.id}','${u.role}')">→ Admin</button>`:""}
          <button class="btn btn-ghost" style="padding:.22rem .6rem;font-size:.73rem"
            onclick="toggleActive('${u.id}',${u.is_active})">${u.is_active?"Disable":"Enable"}</button>
          <button class="btn btn-danger" style="padding:.22rem .6rem;font-size:.73rem"
            onclick="confirmDelete('${u.id}','${u.name}')">Delete</button>
        </div>
      </td>
    </tr>`).join("") || '<tr><td colspan="5" class="empty-state">No users match your search.</td></tr>';
}

document.getElementById("userSearchInput")?.addEventListener("input", e=>{
  userSearch=e.target.value.toLowerCase();
  renderUsers(allUsers);
});

// ── User detail modal (sessions + events for one user) ────────────────────────
async function viewUserDetail(userId, userName) {
  const modal = document.getElementById("userDetailModal");
  document.getElementById("userDetailName").textContent = userName;
  document.getElementById("userDetailBody").innerHTML = '<p style="color:var(--muted)">Loading…</p>';
  modal.classList.remove("hidden");

  try {
    // Load this user's sessions (admin sees all, filter by user_id in UI)
    const allSessions = await apiFetch("/api/sessions/?limit=200");
    const sessions = allSessions.filter(s => s.user_id === userId);

    if (!sessions.length) {
      document.getElementById("userDetailBody").innerHTML =
        '<p class="empty-state"><span class="empty-icon">📋</span>No sessions yet.</p>';
      return;
    }

    // Stats for this user
    const totalAlerts   = sessions.reduce((a,s)=>a+(s.total_alerts||0),0);
    const maxRisk       = Math.max(...sessions.map(s=>s.max_risk_score||0));
    const completedSess = sessions.filter(s=>s.ended_at).length;
    const totalDur      = sessions.reduce((a,s)=>a+(s.duration_seconds||0),0);

    document.getElementById("userDetailBody").innerHTML = `
      <!-- User summary KPIs -->
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:1.2rem">
        ${[
          ["Sessions",completedSess],
          ["Total Alerts",totalAlerts],
          ["Max Risk",Math.round(maxRisk)+"%"],
          ["Total Drive",fmtDur(totalDur)],
        ].map(([l,v])=>`
          <div style="background:var(--surface2);border-radius:7px;padding:.7rem;text-align:center">
            <div style="font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em">${l}</div>
            <div style="font-size:1.4rem;font-weight:700;margin-top:.1rem">${v}</div>
          </div>`).join("")}
      </div>

      <!-- Sessions table -->
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">
        Sessions (${sessions.length})
      </div>
      <div style="overflow-x:auto;max-height:320px;overflow-y:auto">
        <table class="sessions-table" style="font-size:.82rem">
          <thead><tr>
            <th>Date</th><th>Duration</th><th>Alerts</th><th>Max Risk</th><th>Notes</th><th>Events</th>
          </tr></thead>
          <tbody>
            ${sessions.map(s=>{
              const risk=Math.round(s.max_risk_score||0);
              return `<tr>
                <td>${fmtTime(s.started_at)}</td>
                <td>${fmtDur(s.duration_seconds)}</td>
                <td>${s.total_alerts||0}</td>
                <td><span class="risk-pill ${riskCls(risk)}">${risk}%</span></td>
                <td style="color:var(--muted);font-size:.75rem">${s.notes||"—"}</td>
                <td>
                  <button class="btn btn-ghost" style="padding:.2rem .55rem;font-size:.72rem"
                    onclick="viewSessionEvents('${s.id}','${fmtTime(s.started_at)}')">
                    View
                  </button>
                </td>
              </tr>`;
            }).join("")}
          </tbody>
        </table>
      </div>`;
  } catch(e) {
    document.getElementById("userDetailBody").innerHTML = `<p style="color:var(--danger)">${e.message}</p>`;
  }
}

// ── Session events sub-modal ──────────────────────────────────────────────────
async function viewSessionEvents(sessionId, label) {
  const modal = document.getElementById("eventsModal");
  document.getElementById("eventsModalTitle").textContent = `Events — ${label}`;
  document.getElementById("eventsModalBody").innerHTML = '<p style="color:var(--muted)">Loading…</p>';
  modal.classList.remove("hidden");
  try {
    const events = await apiFetch(`/api/events/${sessionId}`);
    if (!events.length) {
      document.getElementById("eventsModalBody").innerHTML = '<p class="empty-state"><span class="empty-icon">✅</span>No alerts in this session.</p>';
      return;
    }
    document.getElementById("eventsModalBody").innerHTML = events.map(e=>`
      <div class="alert-entry ${e.event_type}" style="margin-bottom:.3rem">
        <span class="alert-time">${new Date(e.timestamp).toLocaleTimeString()}</span>
        <span class="alert-msg">
          ${TYPE_LABELS[e.event_type]||e.event_type}
          &nbsp;·&nbsp; risk ${Math.round(e.risk_score)}%
          ${e.ear!=null?`&nbsp;·&nbsp; EAR ${e.ear.toFixed(2)}`:""}
          ${e.yaw!=null?`&nbsp;·&nbsp; Yaw ${e.yaw.toFixed(0)}°`:""}
        </span>
      </div>`).join("");
  } catch(e) {
    document.getElementById("eventsModalBody").innerHTML = `<p style="color:var(--danger)">${e.message}</p>`;
  }
}

// ── Role / active / delete ────────────────────────────────────────────────────
async function toggleRole(uid,cur){
  try{await apiFetch(`/api/admin/users/${uid}`,{method:"PATCH",body:JSON.stringify({role:cur==="user"?"admin":"user"})});showToast("Role updated.","success");loadUsers();}catch(e){showToast(e.message,"error");}
}
async function toggleActive(uid,active){
  try{await apiFetch(`/api/admin/users/${uid}`,{method:"PATCH",body:JSON.stringify({is_active:!active})});showToast(`User ${active?"disabled":"enabled"}.`,"success");loadUsers();}catch(e){showToast(e.message,"error");}
}
function confirmDelete(uid,name){
  const modal=document.getElementById("confirmModal");
  document.getElementById("confirmMsg").textContent=`Delete "${name}" and ALL their sessions? This cannot be undone.`;
  modal.classList.remove("hidden");
  document.getElementById("confirmYes").onclick=async()=>{
    modal.classList.add("hidden");
    try{await apiFetch(`/api/admin/users/${uid}`,{method:"DELETE"});showToast(`"${name}" deleted.`,"success");loadStats();loadUsers();}catch(e){showToast(e.message,"error");}
  };
  document.getElementById("confirmNo").onclick=()=>modal.classList.add("hidden");
}

// ── Modal close helpers ───────────────────────────────────────────────────────
["modalOverlay","userDetailModal","eventsModal","confirmModal"].forEach(id=>{
  const el=document.getElementById(id);
  if(!el) return;
  el.addEventListener("click",e=>{if(e.target===el)el.classList.add("hidden");});
});
document.getElementById("btnCloseModal")?.addEventListener("click",()=>document.getElementById("modalOverlay").classList.add("hidden"));
document.getElementById("btnCloseUserDetail")?.addEventListener("click",()=>document.getElementById("userDetailModal").classList.add("hidden"));
document.getElementById("btnCloseEvents")?.addEventListener("click",()=>document.getElementById("eventsModal").classList.add("hidden"));

// ── Init ──────────────────────────────────────────────────────────────────────
loadStats(); loadUsers(); loadTrend();