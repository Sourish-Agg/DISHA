// frontend/js/auth.js
// Handles login and register form submissions.

// If already logged in, redirect based on role immediately
(function () {
  const token = localStorage.getItem("disha_token");
  const role  = localStorage.getItem("disha_role");
  if (token) {
    window.location.replace(role === "admin" ? "admin.html" : "index.html");
  }
})();

// Helper — redirect after successful auth based on role
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
    } catch (e) {
      showToast(e.message, "error");
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
    } catch (e) {
      showToast(e.message, "error");
      btn.disabled = false;
      btn.textContent = "Create Account";
    }
  });
}