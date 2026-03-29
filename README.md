Got it — the issue is **GitHub README doesn’t support full HTML pages (`<html>`, `<head>`, `<style>`)**, so it’s rendering your raw tags as text (as seen in your screenshot).

What you want instead is a **Markdown + inline HTML hybrid** (GitHub-compatible, clean, professional, and styled via badges + layout — not CSS).

---

# ✅ **FINAL: Professional GitHub README (Drop-in Ready)**

Just copy-paste this into your `README.md` 👇

---

````markdown
<h1 align="center">🚘 D.I.S.H.A.</h1>
<h3 align="center">Driver Information & State Hazard Analytics</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Build-Passing-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-LSTM-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Database-PostgreSQL-blue?style=for-the-badge&logo=postgresql" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

---

## 🚀 Overview

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


