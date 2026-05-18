// frontend/js/admin.js
// Admin dashboard: stats overview + user management table.

requireAdmin();  // only admins can view this page

const user = getAuthUser();
document.getElementById("sidebarUserName").textContent = user.name || "Admin";
document.getElementById("sidebarUserRole").textContent = "admin";
document.getElementById("sidebarAvatar").textContent   = (user.name || "A")[0].toUpperCase();
document.getElementById("btnLogout").addEventListener("click", authLogout);

// ── Stat cards ────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const stats = await apiFetch("/api/admin/stats");

    document.getElementById("statUsers").textContent    = stats.total_users;
    document.getElementById("statSessions").textContent = stats.total_sessions;
    document.getElementById("statEvents").textContent   = stats.total_events;

    // Events breakdown
    const typeLabels = {
      drowsy_eyes:      "Drowsy Eyes",
      yawning:          "Yawning",
      phone_detected:   "Phone",
      head_distraction: "Head Distraction",
      high_risk:        "High Risk",
    };
    const breakdownEl = document.getElementById("eventBreakdown");
    breakdownEl.innerHTML = Object.entries(stats.events_by_type).map(([type, count]) => `
      <div class="alert-entry ${type}" style="justify-content:space-between;align-items:center">
        <span class="alert-msg">${typeLabels[type] || type}</span>
        <strong>${count}</strong>
      </div>
    `).join("") || '<p class="empty-state" style="padding:.5rem">No events yet.</p>';

    // Recent sessions table
    const tbody = document.getElementById("recentSessionsBody");
    tbody.innerHTML = stats.recent_sessions.map(s => {
      const risk = Math.round(s.max_risk_score || 0);
      const riskClass = risk < 30 ? "safe" : risk < 70 ? "warn" : "danger";
      return `
        <tr>
          <td>${s.user_name || "—"}</td>
          <td>${s.driver_name || "—"}</td>
          <td>${new Date(s.started_at).toLocaleString()}</td>
          <td>${s.total_alerts || 0}</td>
          <td><span class="risk-pill ${riskClass}">${risk}%</span></td>
        </tr>
      `;
    }).join("") || '<tr><td colspan="5" class="empty-state">No sessions yet.</td></tr>';

  } catch (e) {
    showToast(e.message, "error");
  }
}

// ── User management ───────────────────────────────────────────────────────────
async function loadUsers() {
  try {
    const users = await apiFetch("/api/admin/users");
    renderUsers(users);
  } catch (e) {
    showToast(e.message, "error");
  }
}

function renderUsers(users) {
  const tbody = document.getElementById("usersBody");
  tbody.innerHTML = users.map(u => `
    <tr>
      <td>${u.name}</td>
      <td>${u.email}</td>
      <td><span class="role-badge ${u.role}">${u.role}</span></td>
      <td>
        <span class="risk-pill ${u.is_active ? 'safe' : 'danger'}">
          ${u.is_active ? "Active" : "Disabled"}
        </span>
      </td>
      <td>${new Date(u.created_at).toLocaleDateString()}</td>
      <td style="display:flex;gap:.4rem;flex-wrap:wrap">
        ${u.role !== "admin"
          ? `<button class="btn btn-ghost" style="padding:.25rem .65rem;font-size:.75rem"
               onclick="toggleRole('${u.id}', '${u.role}')">
               ${u.role === "user" ? "Make Admin" : "Make User"}
             </button>`
          : ""}
        <button class="btn btn-ghost" style="padding:.25rem .65rem;font-size:.75rem"
          onclick="toggleActive('${u.id}', ${u.is_active})">
          ${u.is_active ? "Disable" : "Enable"}
        </button>
        <button class="btn btn-danger" style="padding:.25rem .65rem;font-size:.75rem"
          onclick="deleteUser('${u.id}', '${u.name}')">
          Delete
        </button>
      </td>
    </tr>
  `).join("");
}

async function toggleRole(userId, currentRole) {
  const newRole = currentRole === "user" ? "admin" : "user";
  try {
    await apiFetch(`/api/admin/users/${userId}`, {
      method: "PATCH",
      body: JSON.stringify({ role: newRole }),
    });
    showToast(`Role updated to ${newRole}.`, "success");
    loadUsers();
  } catch (e) {
    showToast(e.message, "error");
  }
}

async function toggleActive(userId, isActive) {
  try {
    await apiFetch(`/api/admin/users/${userId}`, {
      method: "PATCH",
      body: JSON.stringify({ is_active: !isActive }),
    });
    showToast(`User ${isActive ? "disabled" : "enabled"}.`, "success");
    loadUsers();
  } catch (e) {
    showToast(e.message, "error");
  }
}

async function deleteUser(userId, name) {
  if (!confirm(`Delete "${name}" and all their data? This cannot be undone.`)) return;
  try {
    await apiFetch(`/api/admin/users/${userId}`, { method: "DELETE" });
    showToast(`User "${name}" deleted.`, "success");
    loadStats();
    loadUsers();
  } catch (e) {
    showToast(e.message, "error");
  }
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".admin-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".admin-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.panel).classList.remove("hidden");
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadStats();
loadUsers();
