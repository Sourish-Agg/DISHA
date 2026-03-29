
**D.I.S.H.A.** is an advanced, real-time driver monitoring system that uses **Computer Vision + Temporal Machine Learning** to detect:

- 😴 Drowsiness  
- 🥱 Yawning  
- 📱 Phone usage  
- 👀 Driver distraction  

All signals are fused into a **single intelligent risk score**.

---

## ✨ Key Features

### 👁️ Drowsiness Detection
- Uses **Eye Aspect Ratio (EAR)**  
- Implements **PERCLOS** for fatigue detection  

### 🥱 Yawn Detection
- Tracks **Mouth Aspect Ratio (MAR)**  
- Detects sustained yawning events  

### 📱 Phone & Distraction Detection
- Powered by **YOLOv8n**
- Head pose tracking:
  - Pitch
  - Yaw
  - Roll  

### 🧠 Temporal LSTM Modeling
- 30-frame sequence analysis  
- Reduces false positives significantly  

### 📊 Decision Fusion
- Combines all signals into:
  ```bash
  SAFE | MODERATE | HIGH RISK
````

### 🔒 Secure Logging API

* Built with **FastAPI**
* JWT authentication
* Role-based access (Admin/User)

### 📈 Dashboard

* Session analytics
* Event tracking
* Clean UI for monitoring

---

## 🏗️ Architecture

```
Camera Input
     ↓
Computer Vision Server (Flask + WebSockets)
     ↓
Feature Extraction (EAR, MAR, Head Pose, YOLO)
     ↓
LSTM Model (Temporal Analysis)
     ↓
Decision Fusion Engine
     ↓
FastAPI Backend → PostgreSQL
     ↓
Frontend Dashboard
```

---

## 🛠️ Tech Stack

### 🔹 Computer Vision Server (`app.py`)

* Flask + Flask-Sock
* OpenCV, NumPy
* MediaPipe (Face Landmarks)
* Ultralytics YOLOv8

### 🔹 Backend (`main.py`)

* FastAPI + Uvicorn
* PostgreSQL (asyncpg)
* JWT Auth (python-jose)
* Passlib (bcrypt)

### 🔹 Frontend

* HTML5, CSS3, JavaScript
* Dark UI + responsive design

---

## ⚙️ Installation & Setup

### 1️⃣ Database Setup

```bash
createdb disha_db
```

---

### 2️⃣ Start Backend API

```bash
pip install fastapi uvicorn asyncpg passlib[bcrypt] python-jose[cryptography] python-dotenv pydantic[email]

uvicorn main:app --reload --port 8000
```

📌 API Docs:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 3️⃣ Start Vision Server

```bash
pip install flask flask-sock opencv-python mediapipe numpy ultralytics

python3 app.py
```

📌 Runs on:
👉 [http://localhost:5001](http://localhost:5001)

---

## 💻 Usage

1. Start both servers
2. Open:

   * Live Monitoring → `http://localhost:5001`
   * Dashboard → `disha_frontend.html`

---

## 🔐 Default Credentials

```bash
Username: admin
Password: admin123
```

---

## 🎯 Configuration Thresholds

| Metric | Threshold | Purpose     |
| ------ | --------- | ----------- |
| EAR    | < 0.21    | Eye closure |
| MAR    | > 0.50    | Yawning     |
| Pitch  | > 20°     | Distraction |
| Yaw    | > 35°     | Distraction |

---

## 📸 Demo (Optional)

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=D.I.S.H.A+Demo" />
</p>

---

## 🚧 Future Improvements

* 🔊 Audio alerts
* ☁️ Cloud deployment
* 📱 Mobile app integration
* 🤖 Better ML model tuning

---


