// frontend/js/api.js
// Centralised API client.
// All fetch calls go through `apiFetch` which:
//   - Prepends the base URL
//   - Attaches the Bearer token automatically
//   - Handles 401 by logging the user out
//   - Parses FastAPI error detail correctly (string OR array)

const API_BASE = "http://127.0.0.1:8000";

/**
 * Parse FastAPI error detail into a readable string.
 * FastAPI 422 returns detail as an array of objects like:
 *   [{ loc: ["body","email"], msg: "value is not a valid email address", type: "..." }]
 * Any other error returns detail as a plain string.
 */
function parseErrorDetail(detail) {
  if (!detail) return "Request failed.";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // Extract the human-readable msg from each validation error
    return detail.map(e => {
      const field = e.loc ? e.loc[e.loc.length - 1] : "";
      return field ? `${field}: ${e.msg}` : e.msg;
    }).join(" | ");
  }
  return String(detail);
}

/**
 * Make an authenticated API request.
 * @param {string} path    - e.g. "/api/sessions/"
 * @param {object} options - standard fetch options (method, body, etc.)
 * @returns {Promise<any>} - parsed JSON response
 */
async function apiFetch(path, options = {}) {
  const token = localStorage.getItem("disha_token");

  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  // Token expired or invalid → force logout
  if (res.status === 401) {
    authLogout();
    throw new Error("Session expired. Please log in again.");
  }

  // 204 No Content — no body to parse
  if (res.status === 204) return null;

  const data = await res.json();
  if (!res.ok) throw new Error(parseErrorDetail(data.detail));
  return data;
}

// ── Auth helpers ──────────────────────────────────────────────────────────────

function authSave(tokenData) {
  // tokenData = { access_token, role, name, user_id }
  localStorage.setItem("disha_token",   tokenData.access_token);
  localStorage.setItem("disha_role",    tokenData.role);
  localStorage.setItem("disha_name",    tokenData.name);
  localStorage.setItem("disha_user_id", tokenData.user_id);
}

function authLogout() {
  localStorage.removeItem("disha_token");
  localStorage.removeItem("disha_role");
  localStorage.removeItem("disha_name");
  localStorage.removeItem("disha_user_id");
  window.location.replace("login.html");
}

function getAuthUser() {
  return {
    token:  localStorage.getItem("disha_token"),
    role:   localStorage.getItem("disha_role"),
    name:   localStorage.getItem("disha_name"),
    userId: localStorage.getItem("disha_user_id"),
  };
}

function requireAuth() {
  if (!localStorage.getItem("disha_token")) {
    window.location.replace("login.html");
    throw new Error("Not authenticated");
  }
}

function requireUser() {
  // Regular users only — admins redirected to their dashboard
  requireAuth();
  if (localStorage.getItem("disha_role") === "admin") {
    window.location.replace("admin.html");
    throw new Error("Admins use the admin dashboard");
  }
}

function requireAdmin() {
  // Admins only — regular users redirected to monitor
  requireAuth();
  if (localStorage.getItem("disha_role") !== "admin") {
    window.location.replace("index.html");
    throw new Error("Admin access required");
  }
}

// ── Toast helper ──────────────────────────────────────────────────────────────

function showToast(message, type = "info", duration = 4000) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    document.body.appendChild(container);
  }
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toast.onclick = () => toast.remove();
  container.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

// ── Theme toggle (persisted in localStorage) ──────────────────────────────────
(function initTheme() {
  if (localStorage.getItem("disha_theme") === "light") {
    document.body.classList.add("light");
  }
})();

function setupThemeToggle() {
  const btn = document.getElementById("btnTheme");
  if (!btn) return;
  const isLight = document.body.classList.contains("light");
  btn.textContent = isLight ? "☀️" : "🌙";
  btn.addEventListener("click", () => {
    document.body.classList.toggle("light");
    const light = document.body.classList.contains("light");
    localStorage.setItem("disha_theme", light ? "light" : "dark");
    btn.textContent = light ? "☀️" : "🌙";
  });
}
document.addEventListener("DOMContentLoaded", setupThemeToggle);