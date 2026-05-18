// frontend/js/auth.js
// Login and register page logic.
// 
// IMPORTANT: We do NOT redirect based on localStorage alone.
// If a token exists, we VERIFY it with the backend first.
// This prevents stale/expired tokens from bypassing the login screen.

(async function checkExistingSession() {
  const token = localStorage.getItem("disha_token");
  if (!token) return; // no token — stay on login page

  // Verify token is still valid by calling /api/users/me
  try {
    const res = await fetch("http://127.0.0.1:8000/api/users/me", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) {
      const user = await res.json();
      // Token is valid — redirect based on actual role from backend
      window.location.replace(user.role === "admin" ? "admin.html" : "index.html");
    } else {
      // Token invalid/expired — clear it and stay on login
      localStorage.removeItem("disha_token");
      localStorage.removeItem("disha_role");
      localStorage.removeItem("disha_name");
      localStorage.removeItem("disha_user_id");
    }
  } catch (_) {
    // Backend unreachable — clear token, stay on login
    localStorage.removeItem("disha_token");
    localStorage.removeItem("disha_role");
    localStorage.removeItem("disha_name");
    localStorage.removeItem("disha_user_id");
  }
})();

function redirectAfterAuth(role) {
  window.location.replace(role === "admin" ? "admin.html" : "index.html");
}

// ── Login ─────────────────────────────────────────────────────────────────────
const loginForm = document.getElementById("loginForm");
if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = loginForm.querySelector("button[type=submit]");
    btn.disabled = true;
    btn.textContent = "Signing in…";

    try {
      const data = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({
          email:    document.getElementById("email").value.trim(),
          password: document.getElementById("password").value,
        }),
      });
      authSave(data);
      redirectAfterAuth(data.role);
    } catch (err) {
      showToast(err.message, "error");
      btn.disabled = false;
      btn.textContent = "Sign In";
    }
  });
}

// ── Register ──────────────────────────────────────────────────────────────────
const registerForm = document.getElementById("registerForm");
if (registerForm) {
  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = registerForm.querySelector("button[type=submit]");

    const password = document.getElementById("password").value;
    const confirm  = document.getElementById("confirmPassword").value;
    if (password !== confirm) {
      showToast("Passwords do not match.", "error");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Creating account…";

    try {
      const data = await apiFetch("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({
          name:     document.getElementById("name").value.trim(),
          email:    document.getElementById("email").value.trim(),
          password,
        }),
      });
      authSave(data);
      showToast("Account created! Welcome to D.I.S.H.A.", "success");
      setTimeout(() => redirectAfterAuth(data.role), 800);
    } catch (err) {
      showToast(err.message, "error");
      btn.disabled = false;
      btn.textContent = "Create Account";
    }
  });
}