# D.I.S.H.A. — Driver Insight & Safety Heuristics Assistant

Real-time driver drowsiness and distraction monitoring system.

---

## What it does

| Feature | Detail |
|---|---|
| 👁 Eye monitoring | EAR < 0.19 → eye closed (MDPI 2022) |
| 😮 Yawn detection | MAR > 0.50 → yawning (PMC meta-review) |
| 🔄 Head pose | Yaw > ±20° or Pitch > ±15° → distracted (PMC 2023) |
| 📱 Phone detection | TF.js COCO-SSD, confidence > 0.45 |
| 📊 PERCLOS | > 15% closure over 3-second window → drowsy (NHTSA 1994) |
| 🔴 Risk score | Fused: PERCLOS 35% + Yawn 25% + Head 25% + Phone 15% |
| 🗃 Backend | FastAPI + MongoDB, full session and event logging |
| 🔐 Auth | JWT-based, RBAC (admin / user), bcrypt passwords |
| ⚙️ Admin panel | User management, stats, recent sessions |

---

## Project structure

```
disha/
├── frontend/               ← Pure HTML/CSS/JS (no build step needed)
│   ├── index.html          ← Live monitor page
│   ├── login.html
│   ├── register.html
│   ├── history.html        ← Session history
│   ├── admin.html          ← Admin dashboard
│   ├── css/style.css
│   └── js/
│       ├── api.js          ← API client + auth helpers
│       ├── detection.js    ← EAR / MAR / PERCLOS / head pose / risk engine
│       ├── monitor.js      ← Camera + MediaPipe + COCO-SSD + UI
│       ├── history.js
│       ├── admin.js
│       └── auth.js
└── backend/
    ├── main.py             ← FastAPI app entry point
    ├── requirements.txt
    ├── .env.example        ← Copy to .env and fill in values
    ├── core/
    │   ├── config.py       ← Pydantic Settings (reads .env)
    │   ├── database.py     ← Motor async MongoDB client
    │   ├── security.py     ← JWT + bcrypt
    │   └── dependencies.py ← FastAPI dependency injection
    ├── models/
    │   └── schemas.py      ← Pydantic request/response models
    └── routers/
        ├── auth.py         ← POST /api/auth/register, /login
        ├── sessions.py     ← CRUD for monitoring sessions
        ├── events.py       ← Detection event logging
        ├── admin.py        ← Admin-only endpoints
        └── users.py        ← User profile endpoints
```

---

## Prerequisites

Install these before starting:

1. **Python 3.10+** — https://python.org/downloads
2. **MongoDB Community** — https://www.mongodb.com/try/download/community
   - Install and start it. Default: `mongodb://localhost:27017`
3. **VS Code** with the **Live Server** extension
   - Install Live Server: Extensions panel → search "Live Server" → Install
4. **Node.js** is NOT required.

---

## Setup — step by step

### Step 1 — Open the project in VS Code

```
File → Open Folder → select the `disha` folder
```

---

### Step 2 — Set up the backend

Open a **new terminal** in VS Code (`Ctrl+`` ` or Terminal → New Terminal).

```bash
# 1. Go into the backend folder
cd backend

# 2. Create a Python virtual environment
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create your .env file
cp .env.example .env
```

Open `.env` and check the settings. The defaults work for a local MongoDB:

```env
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=disha_db
JWT_SECRET=change_this_to_a_long_random_secret_key_in_production
```

To generate a strong JWT secret, run:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Paste the output as your `JWT_SECRET`.

---

### Step 3 — Start MongoDB

**Windows:** MongoDB runs as a Windows Service after installation. If it's not running:
```
Start → Services → MongoDB → Start
```
Or via terminal:
```bash
net start MongoDB
```

**macOS (Homebrew):**
```bash
brew services start mongodb-community
```

**Linux:**
```bash
sudo systemctl start mongod
```

Verify MongoDB is running:
```bash
mongosh --eval "db.runCommand({ping:1})"
# Should print: { ok: 1 }
```

---

### Step 4 — Start the backend server

From the `backend/` directory (with venv activated):

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

You should see:
```
INFO  MongoDB connected → mongodb://localhost:27017 / disha_db
INFO  D.I.S.H.A. API started  (env=development)
INFO  Uvicorn running on http://127.0.0.1:8000
```

Test it's alive: open http://127.0.0.1:8000/health in your browser.
It should return: `{"status":"ok","env":"development"}`

Interactive API docs: http://127.0.0.1:8000/docs

---

### Step 5 — Serve the frontend

Open a **second terminal** in VS Code.

**Option A — VS Code Live Server (recommended):**
1. Right-click on `frontend/index.html` in the Explorer panel
2. Click **"Open with Live Server"**
3. Browser opens at `http://127.0.0.1:5500/frontend/index.html`

**Option B — Python simple server:**
```bash
cd frontend
python -m http.server 5500
```
Then open: http://localhost:5500

---

### Step 6 — Create your account

1. Browser opens on `index.html` → you'll be redirected to `login.html`
2. Click **"Create one"** to go to the register page
3. Fill in your name, email, and password
4. **The very first registered user is automatically made Admin**
5. You'll be logged in and taken to the Live Monitor page

---

### Step 7 — Start monitoring

1. Click **"▶ Start Monitoring"**
2. Allow browser camera access when prompted
3. The system loads MediaPipe (face detection) and COCO-SSD (phone detection) — takes ~5–10 seconds on first run
4. Face the camera — you'll see:
   - Live EAR / MAR / PERCLOS / head pose metrics updating in real time
   - Green landmarks drawn on your face
   - Status badge showing SAFE / WARNING / DANGER
5. Try closing your eyes for 2+ seconds → drowsy alert fires
6. Try turning your head sideways → head distraction alert
7. Hold a phone in frame → phone detected alert
8. Click **"■ Stop"** to end the session

---

## Admin features

Log in as the admin user and click **Admin** in the sidebar:

- **Overview tab** — total users, sessions, events, alerts by type, recent sessions
- **User Management tab** — promote users to admin, disable accounts, delete users

---

## API overview

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/register` | None | Create account |
| POST | `/api/auth/login` | None | Get JWT token |
| GET | `/api/users/me` | User | Get own profile |
| PATCH | `/api/users/me/password` | User | Change password |
| POST | `/api/sessions/` | User | Start session |
| PATCH | `/api/sessions/{id}/end` | User | End session |
| GET | `/api/sessions/` | User | List sessions |
| POST | `/api/events/` | User | Log detection event |
| GET | `/api/events/{session_id}` | User | Get session events |
| GET | `/api/admin/stats` | Admin | Platform stats |
| GET | `/api/admin/users` | Admin | List all users |
| PATCH | `/api/admin/users/{id}` | Admin | Update user |
| DELETE | `/api/admin/users/{id}` | Admin | Delete user |

---

## Detection thresholds — sources

| Metric | Threshold | Source |
|--------|-----------|--------|
| EAR (eye closed) | < 0.19 | MDPI Electronics 2022, Dewi et al. |
| Consecutive closed frames | ≥ 15 (~0.5 s) | PERCLOS standard |
| PERCLOS (3-second window) | > 15% | NHTSA/Wierwille 1994 |
| MAR (yawning) | > 0.50 | PMC meta-review 2023 |
| Yaw (lateral distraction) | > ±20° | PMC 12899127, 2023 |
| Pitch (vertical distraction) | > ±15° | PMC 12899127, 2023 |
| Phone confidence (COCO-SSD) | > 0.45 | Empirically tuned |

---

## Troubleshooting

**"Camera not found"**
→ Check no other app is using the webcam. Try refreshing.

**"Could not start monitoring"**
→ Open browser console (F12). If you see CORS errors, make sure the backend is running on port 8000 and the frontend is on port 5500.

**Models take long to load**
→ Normal on first use — MediaPipe (~30 MB) and COCO-SSD (~6 MB) are downloaded from CDN. They cache in the browser after the first load.

**MongoDB connection error**
→ Make sure MongoDB is running (`mongosh --eval "db.runCommand({ping:1})"`)

**"Invalid or expired token"**
→ JWT secret was changed. Clear localStorage in the browser (F12 → Application → Local Storage → Clear All) and log in again.
