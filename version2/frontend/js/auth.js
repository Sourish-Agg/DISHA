// frontend/js/auth.js — v3 (multi-tenant: optional org code on register)
// Login + register. Does NOT use apiFetch() — uses raw fetch() directly
// to avoid the 401→authLogout redirect loop on the login page itself.

(async function checkExistingSession() {
  const token = localStorage.getItem("disha_token");
  if (!token) return;
  try {
    const res = await fetch("http://127.0.0.1:8000/api/users/me", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) {
      const user = await res.json();
      window.location.replace(user.role === "admin" ? "admin.html" : "index.html");
    } else {
      ["disha_token","disha_role","disha_name","disha_user_id",
       "disha_org_id","disha_org_name","disha_org_code"].forEach(k => localStorage.removeItem(k));
    }
  } catch (_) {
    ["disha_token","disha_role","disha_name","disha_user_id",
     "disha_org_id","disha_org_name","disha_org_code"].forEach(k => localStorage.removeItem(k));
  }
})();

function redirectAfterAuth(role) {
  window.location.replace(role === "admin" ? "admin.html" : "index.html");
}

// ── Shared error display ──────────────────────────────────────────────────────
function showFormError(message) {
  let el = document.getElementById("formError");
  if (!el) {
    el = document.createElement("div");
    el.id = "formError";
    el.style.cssText = `
      background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.4);
      color:#ef4444;padding:.6rem .9rem;border-radius:6px;
      font-size:.83rem;margin-bottom:.8rem;text-align:center;`;
    const form = document.querySelector("form");
    if (form) form.insertBefore(el, form.firstChild);
  }
  el.textContent = message;
  el.style.display = "block";
}

function clearFormError() {
  const el = document.getElementById("formError");
  if (el) el.style.display = "none";
}

// ── Raw API call (bypasses apiFetch to avoid redirect loops) ──────────────────
async function authFetch(path, body) {
  const res = await fetch(`http://127.0.0.1:8000${path}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = data.detail;
    if (Array.isArray(detail)) {
      throw new Error(detail.map(e => {
        const field = e.loc?.[e.loc.length-1] ?? "";
        return field ? `${field}: ${e.msg}` : e.msg;
      }).join(" | "));
    }
    throw new Error(typeof detail === "string" ? detail : "Request failed.");
  }
  return data;
}

// ── Login ─────────────────────────────────────────────────────────────────────
const loginForm = document.getElementById("loginForm");
if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormError();
    const btn = loginForm.querySelector("button[type=submit]");
    btn.disabled = true;
    btn.textContent = "Signing in…";

    try {
      const data = await authFetch("/api/auth/login", {
        email:    document.getElementById("email").value.trim(),
        password: document.getElementById("password").value,
      });
      authSave(data);
      redirectAfterAuth(data.role);
    } catch (err) {
      showFormError(err.message);
      btn.disabled = false;
      btn.textContent = "Sign In";
    }
  });

  loginForm.querySelectorAll("input").forEach(inp =>
    inp.addEventListener("input", clearFormError)
  );
}

// ── Register ──────────────────────────────────────────────────────────────────
const registerForm = document.getElementById("registerForm");
if (registerForm) {
  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormError();
    const btn = registerForm.querySelector("button[type=submit]");

    const password = document.getElementById("password").value;
    const confirm  = document.getElementById("confirmPassword").value;
    if (password !== confirm) {
      showFormError("Passwords do not match.");
      return;
    }

    // Optional organization invite code. Empty → backend creates a new org
    // and makes this user its admin.
    const orgCodeEl = document.getElementById("orgCode");
    const orgCode   = orgCodeEl ? orgCodeEl.value.trim() : "";

    const payload = {
      name:     document.getElementById("name").value.trim(),
      email:    document.getElementById("email").value.trim(),
      password,
    };
    if (orgCode) payload.org_code = orgCode;

    btn.disabled = true;
    btn.textContent = "Creating account…";

    try {
      const data = await authFetch("/api/auth/register", payload);
      authSave(data);
      // Show the invite code to brand-new admins so they can share it.
      if (data.role === "admin") {
        try { sessionStorage.setItem("disha_show_orgcode", data.org_code); } catch (_) {}
      }
      redirectAfterAuth(data.role);
    } catch (err) {
      showFormError(err.message);
      btn.disabled = false;
      btn.textContent = "Create Account";
    }
  });

  registerForm.querySelectorAll("input").forEach(inp =>
    inp.addEventListener("input", clearFormError)
  );
}