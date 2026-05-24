/*
 * ============================================================
 *  D.I.S.H.A. — Driver Insight & Safety Heuristics Assistant
 *  ESP32 Actuation Layer Firmware
 * ============================================================
 *  Board:  ESP32 Dev Module
 *  Upload: 921600 baud
 *  Serial: 115200 baud
 * ============================================================
 *
 *  ⚠️  BEFORE UPLOADING:
 *      1. Replace WIFI_SSID and WIFI_PASS below
 *      2. Verify GPIO wiring matches pin definitions
 *      3. Confirm ENA/ENB jumper caps are REMOVED from L298N
 *      4. Confirm battery polarity (RED = +, BLACK = −)
 *
 * ============================================================
 */

#include <WiFi.h>
#include <WebServer.h>

// ─────────────────────────────────────────────
//  🔧 CONFIGURATION — EDIT THESE TWO LINES
// ─────────────────────────────────────────────
const char* WIFI_SSID = "Sourish's Nothing 3";     // ← Replace with your WiFi name
const char* WIFI_PASS = "sourish10000";  // ← Replace with your WiFi password

// ─────────────────────────────────────────────
//  📌 GPIO PIN DEFINITIONS
// ─────────────────────────────────────────────

// Motor Control (ESP32 → L298N)
#define LEFT_IN1   5     // Left motors direction pin 1
#define LEFT_IN2   18    // Left motors direction pin 2
#define LEFT_ENA   19    // Left motors PWM speed

#define RIGHT_IN3  25    // Right motors direction pin 1
#define RIGHT_IN4  26    // Right motors direction pin 2
#define RIGHT_ENB  27    // Right motors PWM speed

// Indicators
#define RED_LED    21    // High-risk indicator
#define GREEN_LED  22    // Safe state indicator
#define BUZZER     23    // Audio alert

// Ultrasonic Sensor (HC-SR04)
#define TRIG_PIN   12    // Trigger
#define ECHO_PIN   14    // Echo

// ─────────────────────────────────────────────
//  ⚙️ PWM CONFIGURATION
// ─────────────────────────────────────────────
#define PWM_FREQ       5000   // 5 kHz (inaudible)
#define PWM_RESOLUTION 8      // 8-bit → 0-255 range


// ─────────────────────────────────────────────
//  🛡️ SAFETY PARAMETERS
// ─────────────────────────────────────────────
#define WATCHDOG_TIMEOUT_MS   5000   // Auto-stop if no command for 5s
#define OBSTACLE_THRESHOLD_CM 5      // Emergency stop distance (lowered — sensor picks up chassis/ground at 15)
#define ULTRASONIC_INTERVAL   500    // Check obstacle every 500ms (less CPU load)

// ─────────────────────────────────────────────
//  📊 SPEED DEFINITIONS
// ─────────────────────────────────────────────
#define SPEED_FULL  255   // /normal — 100%
#define SPEED_WARN  80    // /warn   — ~31%
#define SPEED_SLOW  50    // /slow   — ~20%
#define SPEED_STOP  0     // /stop   — 0%

// ─────────────────────────────────────────────
//  🌐 WEB SERVER
// ─────────────────────────────────────────────
WebServer server(80);

// ─────────────────────────────────────────────
//  🕐 TIMING VARIABLES
// ─────────────────────────────────────────────
unsigned long lastCommandTime = 0;
unsigned long lastUltrasonicCheck = 0;
bool obstacleOverride = false;     // True when ultrasonic detects obstacle
String currentCommand = "stop";    // Track current state for Serial logging

// ═══════════════════════════════════════════════
//  MOTOR CONTROL FUNCTIONS
// ═══════════════════════════════════════════════

void setMotors(int speed) {
  if (speed > 0) {
    // Forward direction
    digitalWrite(LEFT_IN1, HIGH);
    digitalWrite(LEFT_IN2, LOW);
    digitalWrite(RIGHT_IN3, HIGH);
    digitalWrite(RIGHT_IN4, LOW);
  } else {
    // Stop — set all direction pins LOW
    digitalWrite(LEFT_IN1, LOW);
    digitalWrite(LEFT_IN2, LOW);
    digitalWrite(RIGHT_IN3, LOW);
    digitalWrite(RIGHT_IN4, LOW);
  }
  // Apply PWM speed
  ledcWrite(LEFT_ENA, speed);
  ledcWrite(RIGHT_ENB, speed);
}

// ═══════════════════════════════════════════════
//  INDICATOR CONTROL
// ═══════════════════════════════════════════════

void setIndicators(bool redOn, bool greenOn) {
  digitalWrite(RED_LED, redOn ? HIGH : LOW);
  digitalWrite(GREEN_LED, greenOn ? HIGH : LOW);
}

void buzzerBeep(int count, int durationMs) {
  for (int i = 0; i < count; i++) {
    digitalWrite(BUZZER, HIGH);
    delay(durationMs);
    digitalWrite(BUZZER, LOW);
    if (i < count - 1) delay(150);  // Gap between beeps
  }
}

// ═══════════════════════════════════════════════
//  ULTRASONIC SENSOR
// ═══════════════════════════════════════════════

float getDistanceCm() {
  // Send 10µs trigger pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Measure echo duration (timeout 30ms ≈ ~5m max)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return 999.0;  // No echo = no obstacle (sensor timeout)
  }

  // Speed of sound ≈ 343 m/s → distance = duration * 0.0343 / 2
  return (duration * 0.0343) / 2.0;
}

// ═══════════════════════════════════════════════
//  CORS HEADERS (required for browser access)
// ═══════════════════════════════════════════════

void sendCORSHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "*");
}

void handleOptions() {
  sendCORSHeaders();
  server.send(204);
}

// ═══════════════════════════════════════════════
//  HTTP COMMAND HANDLERS
// ═══════════════════════════════════════════════

void handleNormal() {
  sendCORSHeaders();
  lastCommandTime = millis();
  obstacleOverride = false;

  setMotors(SPEED_FULL);
  setIndicators(false, true);  // Green ON, Red OFF

  currentCommand = "NORMAL";
  Serial.println("► COMMAND: NORMAL — Full speed, green LED");
  server.send(200, "application/json", "{\"status\":\"normal\",\"speed\":255}");
}

void handleWarn() {
  sendCORSHeaders();
  lastCommandTime = millis();

  setMotors(SPEED_WARN);
  setIndicators(true, false);  // Red ON, Green OFF
  buzzerBeep(2, 150);          // 2 short beeps

  currentCommand = "WARN";
  Serial.println("► COMMAND: WARN — Reduced speed, 2 beeps");
  server.send(200, "application/json", "{\"status\":\"warn\",\"speed\":80}");
}

void handleSlow() {
  sendCORSHeaders();
  lastCommandTime = millis();

  setMotors(SPEED_SLOW);
  setIndicators(true, false);  // Red ON

  currentCommand = "SLOW";
  Serial.println("► COMMAND: SLOW — Crawl speed, red LED");
  server.send(200, "application/json", "{\"status\":\"slow\",\"speed\":50}");
}

void handleStop() {
  sendCORSHeaders();
  lastCommandTime = millis();

  setMotors(SPEED_STOP);
  setIndicators(true, false);  // Red ON
  buzzerBeep(1, 500);          // 1 long beep

  currentCommand = "STOP";
  Serial.println("► COMMAND: STOP — Motors off, buzzer alert");
  server.send(200, "application/json", "{\"status\":\"stop\",\"speed\":0}");
}

void handleBeep() {
  sendCORSHeaders();
  lastCommandTime = millis();

  buzzerBeep(1, 300);  // Single short beep (phone detected)

  currentCommand = "BEEP";
  Serial.println("► COMMAND: BEEP — Phone detection alert");
  server.send(200, "application/json", "{\"status\":\"beep\"}");
}

void handlePing() {
  sendCORSHeaders();
  lastCommandTime = millis();

  Serial.println("► PING received (heartbeat)");
  server.send(200, "application/json", "{\"status\":\"ok\",\"uptime\":" + String(millis() / 1000) + "}");
}

void handleRoot() {
  sendCORSHeaders();
  String html = "<html><head><title>D.I.S.H.A. ESP32</title></head><body>";
  html += "<h1>D.I.S.H.A. Actuation Layer</h1>";
  html += "<p>Status: Online</p>";
  html += "<p>Current: " + currentCommand + "</p>";
  html += "<p>Uptime: " + String(millis() / 1000) + "s</p>";
  html += "<p>Endpoints: /normal /warn /slow /stop /beep /ping</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleNotFound() {
  sendCORSHeaders();
  server.send(404, "application/json", "{\"error\":\"Unknown command\"}");
}

// ═══════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("════════════════════════════════════════");
  Serial.println("  D.I.S.H.A. — ESP32 Actuation Layer");
  Serial.println("════════════════════════════════════════");

  // ── Configure GPIO Pins ──
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(RIGHT_IN3, OUTPUT);
  pinMode(RIGHT_IN4, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // ── Configure PWM Channels ──
  ledcAttach(LEFT_ENA, PWM_FREQ, PWM_RESOLUTION);

  ledcAttach(RIGHT_ENB, PWM_FREQ, PWM_RESOLUTION);

  // ── Start in STOP state ──
  setMotors(SPEED_STOP);
  setIndicators(true, false);  // Red ON at boot (not yet connected)

  // ── Connect to WiFi (with persistence + auto-reconnect) ──
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);   // Auto-reconnect if WiFi drops
  WiFi.persistent(true);         // Remember credentials across reboots
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 60) {  // 30 second timeout (longer for hotspot)
      Serial.println("\n✗ WiFi connection FAILED. Check SSID/password.");
      Serial.println("  Restarting in 3 seconds...");
      delay(3000);
      ESP.restart();
    }
  }

  Serial.println();
  Serial.println("✓ WiFi connected!");
  Serial.print("✓ IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println();

  // ── Flash green LED to indicate WiFi success ──
  for (int i = 0; i < 3; i++) {
    setIndicators(false, true);
    delay(200);
    setIndicators(false, false);
    delay(200);
  }

  // ── Register HTTP Endpoints ──
  server.on("/",       HTTP_GET,     handleRoot);
  server.on("/normal", HTTP_GET,     handleNormal);
  server.on("/warn",   HTTP_GET,     handleWarn);
  server.on("/slow",   HTTP_GET,     handleSlow);
  server.on("/stop",   HTTP_GET,     handleStop);
  server.on("/beep",   HTTP_GET,     handleBeep);
  server.on("/ping",   HTTP_GET,     handlePing);

  // CORS preflight for all endpoints
  server.on("/normal", HTTP_OPTIONS, handleOptions);
  server.on("/warn",   HTTP_OPTIONS, handleOptions);
  server.on("/slow",   HTTP_OPTIONS, handleOptions);
  server.on("/stop",   HTTP_OPTIONS, handleOptions);
  server.on("/beep",   HTTP_OPTIONS, handleOptions);
  server.on("/ping",   HTTP_OPTIONS, handleOptions);

  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println("════════════════════════════════════════");
  Serial.println("  ✓ HTTP server started on port 80");
  Serial.println("  ✓ Waiting for commands...");
  Serial.println("════════════════════════════════════════");
  Serial.println();

  // Set initial watchdog time
  lastCommandTime = millis();

  // Ready state — green LED solid
  setIndicators(false, true);
}

// ═══════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════

void loop() {
  // Handle incoming HTTP requests
  server.handleClient();

  unsigned long now = millis();

  // ── Ultrasonic Obstacle Check ──
  if (now - lastUltrasonicCheck >= ULTRASONIC_INTERVAL) {
    lastUltrasonicCheck = now;

    float distance = getDistanceCm();

    if (distance > 0 && distance < OBSTACLE_THRESHOLD_CM) {
      if (!obstacleOverride) {
        obstacleOverride = true;
        setMotors(SPEED_STOP);
        setIndicators(true, false);
        buzzerBeep(3, 100);  // 3 rapid beeps
        Serial.print("⚠ OBSTACLE DETECTED at ");
        Serial.print(distance, 1);
        Serial.println(" cm — Emergency stop!");
      }
    } else {
      // Clear obstacle override only if a new command has been received
      if (obstacleOverride && distance >= OBSTACLE_THRESHOLD_CM + 5) {
        obstacleOverride = false;
        Serial.println("✓ Obstacle cleared — resuming command control");
      }
    }
  }

  // ── Watchdog Timer ──
  if (now - lastCommandTime > WATCHDOG_TIMEOUT_MS) {
    if (currentCommand != "WATCHDOG") {
      setMotors(SPEED_STOP);
      setIndicators(true, false);
      currentCommand = "WATCHDOG";
      Serial.println("⚠ WATCHDOG: No command for 5s — auto-stop activated");
    }
  }

  // ── WiFi Reconnection (non-blocking, retries before restart) ──
  static unsigned long lastWifiCheck = 0;
  static int wifiRetries = 0;

  if (now - lastWifiCheck >= 2000) {  // Check every 2 seconds
    lastWifiCheck = now;

    if (WiFi.status() != WL_CONNECTED) {
      wifiRetries++;
      Serial.print("⚠ WiFi disconnected — retry ");
      Serial.print(wifiRetries);
      Serial.println("/10...");

      setMotors(SPEED_STOP);
      setIndicators(true, false);

      if (wifiRetries == 1) {
        // First attempt: simple reconnect
        WiFi.reconnect();
      } else if (wifiRetries == 5) {
        // After 5 failures: full disconnect and reconnect
        Serial.println("  Trying full WiFi reset...");
        WiFi.disconnect();
        delay(1000);
        WiFi.begin(WIFI_SSID, WIFI_PASS);
      } else if (wifiRetries >= 10) {
        // After 10 failures: reboot ESP32
        Serial.println("✗ WiFi recovery failed. Restarting...");
        buzzerBeep(5, 100);
        delay(1000);
        ESP.restart();
      }
    } else {
      if (wifiRetries > 0) {
        Serial.println("✓ WiFi reconnected: " + WiFi.localIP().toString());
        setIndicators(false, true);
        wifiRetries = 0;
      }
    }
  }
}
