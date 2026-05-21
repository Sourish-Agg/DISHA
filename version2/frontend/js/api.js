// frontend/js/api.js — v6 (multi-tenant aware)
// Centralised API client + auth helpers + theme toggle

const API_BASE = "http://127.0.0.1:8000";

// ── Error parser ──────────────────────────────────────────────────────────────
// FastAPI 422 returns detail as an array; other errors return a string.
function parseErrorDetail(detail) {
  if (!detail) return "Request failed.";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map(e => {
      const field = e.loc ? e.loc[e.loc.length - 1] : "";
      return field ? `${field}: ${e.msg}` : e.msg;
    }).join(" | ");
  }
  return String(detail);
}

// ── Core fetch ────────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const token = localStorage.getItem("disha_token");
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (res.status === 401) {
    // Token expired or invalid — clear and send to login
    _clearAuth();
    window.location.replace("login.html");
    throw new Error("Session expired. Please log in again.");
  }

  if (res.status === 204) return null;

  const data = await res.json();
  if (!res.ok) throw new Error(parseErrorDetail(data.detail));
  return data;
}

// ── Auth helpers ──────────────────────────────────────────────────────────────
function authSave(tokenData) {
  localStorage.setItem("disha_token",    tokenData.access_token);
  localStorage.setItem("disha_role",     tokenData.role);
  localStorage.setItem("disha_name",     tokenData.name);
  localStorage.setItem("disha_user_id",  tokenData.user_id);
  // Tenant info (multi-tenancy)
  localStorage.setItem("disha_org_id",   tokenData.org_id   || "");
  localStorage.setItem("disha_org_name", tokenData.org_name || "");
  localStorage.setItem("disha_org_code", tokenData.org_code || "");
}

function _clearAuth() {
  ["disha_token","disha_role","disha_name","disha_user_id",
   "disha_org_id","disha_org_name","disha_org_code"].forEach(k => localStorage.removeItem(k));
}

function authLogout() {
  _clearAuth();
  window.location.replace("login.html");
}

function getAuthUser() {
  return {
    token:   localStorage.getItem("disha_token"),
    role:    localStorage.getItem("disha_role"),
    name:    localStorage.getItem("disha_name"),
    userId:  localStorage.getItem("disha_user_id"),
    orgId:   localStorage.getItem("disha_org_id"),
    orgName: localStorage.getItem("disha_org_name"),
    orgCode: localStorage.getItem("disha_org_code"),
  };
}

// ── Auth guards ───────────────────────────────────────────────────────────────
function requireAuth() {
  if (!localStorage.getItem("disha_token")) {
    window.location.replace("login.html");
    throw new Error("Not authenticated");
  }
}

function requireUser() {
  if (!localStorage.getItem("disha_token")) {
    window.location.replace("login.html");
    throw new Error("Not authenticated");
  }
  if (localStorage.getItem("disha_role") === "admin") {
    window.location.replace("admin.html");
    throw new Error("Admins use the admin dashboard");
  }
}

function requireAdmin() {
  if (!localStorage.getItem("disha_token")) {
    window.location.replace("login.html");
    throw new Error("Not authenticated");
  }
  if (localStorage.getItem("disha_role") !== "admin") {
    window.location.replace("index.html");
    throw new Error("Admin access required");
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
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

// ── Theme toggle ──────────────────────────────────────────────────────────────
(function initTheme() {
  if (localStorage.getItem("disha_theme") === "light") {
    document.body.classList.add("light");
  }
})();

function setupThemeToggle() {
  const btn = document.getElementById("btnTheme");
  if (!btn) return;
  btn.textContent = document.body.classList.contains("light") ? "☀️" : "🌙";
  btn.addEventListener("click", () => {
    document.body.classList.toggle("light");
    const light = document.body.classList.contains("light");
    localStorage.setItem("disha_theme", light ? "light" : "dark");
    btn.textContent = light ? "☀️" : "🌙";
  });
}
document.addEventListener("DOMContentLoaded", setupThemeToggle);