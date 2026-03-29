🚘 D.I.S.H.A.
Driver Information & State Hazard Analytics

D.I.S.H.A. is an advanced, real-time driver state monitoring system. It leverages computer vision and temporal machine learning models to detect drowsiness, yawning, and distracted driving, effectively fusing multiple risk factors into a unified risk score. The project features a real-time computer vision processing server, a robust logging API, and a sleek log review dashboard.

✨ Key Features
👁️ Drowsiness Detection: Tracks Eye Aspect Ratio (EAR) and utilizes PERCLOS (Percentage of Eyelid Closure) to accurately identify driver fatigue.

🥱 Yawn Detection: Monitors Mouth Aspect Ratio (MAR) to detect yawning events that are sustained over time.

📱 Phone & Distraction Detection: Integrates YOLOv8n for real-time mobile phone detection. It also calculates precise head poses (pitch, yaw, roll) to determine if the driver's eyes are off the road.

🧠 Temporal LSTM Modeling: Uses an LSTM model with a 30-frame window to analyze temporal changes in driver state, greatly minimizing false positives.

📊 Decision Fusion: Intelligently combines eye closure, yawning, head pose, and phone presence into a single, smooth risk_score (SAFE, MODERATE, HIGH RISK).

🔒 Secure Logging API: A FastAPI backend connected to PostgreSQL logs sessions, events, and metrics. It supports role-based access control (Admin & User).

📈 Log Review Dashboard: A custom frontend for reviewing driving sessions, event logs, and analyzing overall performance metrics.

🛠️ Architecture & Tech Stack
Computer Vision Server (app.py)

Python Framework: Flask + Flask-Sock (WebSockets)

Machine Learning: MediaPipe (Face Landmarks), Ultralytics (YOLOv8n)

Math & Vision: OpenCV, NumPy

Logging Backend (main.py)

API Framework: FastAPI, Uvicorn

Database: PostgreSQL (asyncpg)

Authentication: JWT (python-jose), Passlib (bcrypt)

Frontend (disha_frontend.html)

Stack: Pure HTML5, CSS3, JavaScript

Design: Custom dark-mode UI with space-grotesk typography and responsive stat grids.

🚀 Installation & Setup
1. Database Setup
Ensure PostgreSQL is installed and running, then create the database:

Bash
createdb disha_db
2. Start the Logging API (FastAPI)
Navigate to the disha_backend directory, install requirements, and spin up the server:

Bash
pip install fastapi uvicorn asyncpg passlib[bcrypt] python-jose[cryptography] python-dotenv pydantic[email]
uvicorn main:app --reload --port 8000
Note: The API documentation will be automatically generated at http://localhost:8000/docs.

3. Start the Computer Vision Server
Open a new terminal, navigate to the main directory, install the computer vision dependencies, and start the app:

Bash
pip install flask flask-sock opencv-python mediapipe numpy ultralytics
python3 app.py
Note: This server runs on http://localhost:5001. It will automatically download the MediaPipe face landmarker model on its first run.

💻 Usage & Dashboard Access
Ensure both your Vision Server (Port 5001) and Logging API (Port 8000) are running.

Open your web browser and navigate to http://localhost:5001 to view the active WebSockets monitoring app.

For Historical Analytics & Admin controls, launch the frontend page (disha_frontend.html) in your browser.

Default Admin Credentials:

Username: admin

Password: admin123

⚙️ Configuration Thresholds
D.I.S.H.A. comes pre-configured with optimally tuned thresholds:

EAR (Eye Aspect Ratio): < 0.21 triggers closure detection.

MAR (Mouth Aspect Ratio): > 0.50 triggers yawn detection.

Head Pose: Pitch > 20.0° or Yaw > 35.0° flags driver distraction.
