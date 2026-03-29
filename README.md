<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>D.I.S.H.A. README</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      background-color: #0f172a;
      color: #e2e8f0;
      padding: 20px;
    }
    h1, h2, h3 {
      color: #38bdf8;
    }
    code {
      background-color: #1e293b;
      padding: 4px 6px;
      border-radius: 4px;
      color: #facc15;
    }
    pre {
      background-color: #1e293b;
      padding: 10px;
      border-radius: 6px;
      overflow-x: auto;
    }
    ul {
      margin-left: 20px;
    }
    .section {
      margin-bottom: 30px;
    }
  </style>
</head>
<body>

  <h1>🚘 D.I.S.H.A.</h1>
  <h3>Driver Information & State Hazard Analytics</h3>

  <p>
    <strong>D.I.S.H.A.</strong> is an advanced, real-time driver state monitoring system. It leverages 
    computer vision and temporal machine learning models to detect drowsiness, yawning, and distracted driving, 
    effectively fusing multiple risk factors into a unified risk score.
  </p>

  <p>
    The project features a real-time computer vision processing server, a robust logging API, 
    and a sleek log review dashboard.
  </p>

  <div class="section">
    <h2>✨ Key Features</h2>

    <ul>
      <li>
        <strong>👁️ Drowsiness Detection:</strong> Tracks Eye Aspect Ratio (EAR) and utilizes 
        PERCLOS (Percentage of Eyelid Closure) to identify driver fatigue.
      </li>

      <li>
        <strong>🥱 Yawn Detection:</strong> Monitors Mouth Aspect Ratio (MAR) to detect sustained yawning events.
      </li>

      <li>
        <strong>📱 Phone & Distraction Detection:</strong> Uses YOLOv8n for mobile detection and calculates 
        head pose (pitch, yaw, roll) to detect distraction.
      </li>

      <li>
        <strong>🧠 Temporal LSTM Modeling:</strong> Uses a 30-frame LSTM window to analyze temporal changes 
        and reduce false positives.
      </li>

      <li>
        <strong>📊 Decision Fusion:</strong> Combines all signals into a unified 
        <code>risk_score</code> (SAFE, MODERATE, HIGH RISK).
      </li>

      <li>
        <strong>🔒 Secure Logging API:</strong> FastAPI backend with PostgreSQL, supporting role-based access.
      </li>

      <li>
        <strong>📈 Log Review Dashboard:</strong> Frontend for session tracking and analytics.
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>🛠️ Architecture & Tech Stack</h2>

    <h3>Computer Vision Server (app.py)</h3>
    <ul>
      <li><strong>Framework:</strong> Flask + Flask-Sock (WebSockets)</li>
      <li><strong>ML:</strong> MediaPipe, Ultralytics (YOLOv8n)</li>
      <li><strong>Libraries:</strong> OpenCV, NumPy</li>
    </ul>

    <h3>Logging Backend (main.py)</h3>
    <ul>
      <li><strong>API:</strong> FastAPI, Uvicorn</li>
      <li><strong>Database:</strong> PostgreSQL (asyncpg)</li>
      <li><strong>Auth:</strong> JWT (python-jose), Passlib (bcrypt)</li>
    </ul>

    <h3>Frontend (disha_frontend.html)</h3>
    <ul>
      <li><strong>Stack:</strong> HTML5, CSS3, JavaScript</li>
      <li><strong>Design:</strong> Dark mode UI, responsive layout</li>
    </ul>
  </div>

  <div class="section">
    <h2>🚀 Installation & Setup</h2>

    <h3>1. Database Setup</h3>
    <pre><code>createdb disha_db</code></pre>

    <h3>2. Start Logging API</h3>
    <pre><code>pip install fastapi uvicorn asyncpg passlib[bcrypt] python-jose[cryptography] python-dotenv pydantic[email]
uvicorn main:app --reload --port 8000</code></pre>

    <p>Access API docs at: <code>http://localhost:8000/docs</code></p>

    <h3>3. Start Computer Vision Server</h3>
    <pre><code>pip install flask flask-sock opencv-python mediapipe numpy ultralytics
python3 app.py</code></pre>

    <p>Runs on: <code>http://localhost:5001</code></p>
  </div>

  <div class="section">
    <h2>💻 Usage & Dashboard Access</h2>

    <ul>
      <li>Ensure both servers are running (Ports 5001 & 8000).</li>
      <li>Open: <code>http://localhost:5001</code> for live monitoring.</li>
      <li>Open <code>disha_frontend.html</code> for analytics dashboard.</li>
    </ul>

    <h3>Default Admin Credentials</h3>
    <ul>
      <li><strong>Username:</strong> admin</li>
      <li><strong>Password:</strong> admin123</li>
    </ul>
  </div>

  <div class="section">
    <h2>⚙️ Configuration Thresholds</h2>

    <ul>
      <li><strong>EAR:</strong> &lt; 0.21 → Eye closure</li>
      <li><strong>MAR:</strong> &gt; 0.50 → Yawn detection</li>
      <li><strong>Head Pose:</strong> Pitch &gt; 20° or Yaw &gt; 35° → Distraction</li>
    </ul>
  </div>

</body>
</html>
