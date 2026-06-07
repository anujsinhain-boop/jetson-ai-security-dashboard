from flask import Flask, Response, jsonify, request
from ultralytics import YOLO
import cv2
import time
import os
import re
import threading
import subprocess
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
last_unknown_save_time = 0
unknown_save_cooldown_sec = 20

MODEL_PATH = "yolov8n.engine"
KNOWN_FACES_DIR = Path("known_faces")
SNAPSHOT_DIR = Path("alerts")
SNAPSHOT_DIR.mkdir(exist_ok=True)
KNOWN_FACES_DIR.mkdir(exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

monitoring_enabled = False
last_alert_time = 0
alert_cooldown_sec = 15
alerts = []
latest_stats = {}
last_unknown_face_path = None

COCO_NAMES = {0: "person"}

model = YOLO(MODEL_PATH, task="detect")
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

recognizer = None
label_to_name = {}


def train_face_recognizer():
    global recognizer, label_to_name

    if not hasattr(cv2, "face"):
        print("ERROR: cv2.face missing. Install opencv-contrib-python.")
        recognizer = None
        return

    faces = []
    labels = []
    label_to_name = {}
    label_id = 0

    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        label_to_name[label_id] = name

        for image_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(detected) == 0:
                continue

            x, y, w, h = detected[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            labels.append(label_id)

        label_id += 1

    if not faces:
        print("No known faces found.")
        recognizer = None
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    print(f"Trained face recognizer with {len(faces)} images:")
    print(label_to_name)


def recognize_person(person_crop):
    if recognizer is None or face_cascade.empty():
        return "Unknown", None

    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(detected) == 0:
        return "Unknown", None

    best_name = "Unknown"
    best_conf = 999
    best_face = None

    for (x, y, w, h) in detected:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face_resized)

        if confidence < best_conf:
            best_conf = confidence
            best_name = label_to_name.get(label, "Unknown")
            best_face = person_crop[y:y+h, x:x+w]

    if best_conf < 75:
        return best_name, best_face

    return "Unknown", best_face

def save_unknown_face(face_img, fallback_img=None):
    global last_unknown_face_path

    img = face_img if face_img is not None else fallback_img

    if img is None:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SNAPSHOT_DIR / f"unknown_face_{ts}.jpg"
    cv2.imwrite(str(path), img)
    last_unknown_face_path = str(path)

def send_telegram_alert(message, image_path=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        print("TELEGRAM_BOT_TOKEN empty:", not bool(TELEGRAM_BOT_TOKEN))
        print("TELEGRAM_CHAT_ID empty:", not bool(TELEGRAM_CHAT_ID))
        return

    try:
        if image_path and Path(image_path).exists():
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as img:
                r = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
                    files={"photo": img},
                    timeout=10,
                )
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            r = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=10,
            )

        print("Telegram status:", r.status_code)
        print("Telegram response:", r.text)

    except Exception as e:
        print("Telegram error:", e)

def cleanup_alert_images():
    for pattern in ["person_*.jpg", "unknown_person_*.jpg", "unknown_face_*.jpg"]:
        for f in SNAPSHOT_DIR.glob(pattern):
            try:
                f.unlink()
            except Exception:
                pass

def add_alert(label, identity, image_path):
    global alerts
    alerts.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "identity": identity,
        "image": str(image_path),
    })
    alerts = alerts[:20]


def stats_loop():
    global latest_stats

    while True:
        try:
            p = subprocess.Popen(
                ["tegrastats"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            line = p.stdout.readline().strip()
            p.terminate()

            ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            swap_match = re.search(r"SWAP\s+(\d+)/(\d+)MB", line)
            gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            cpu_match = re.search(r"CPU\s+\[([^\]]+)\]", line)
            gpu_temp_match = re.search(r"gpu@([\d.]+)C", line)
            cpu_temp_match = re.search(r"cpu@([\d.]+)C", line)

            latest_stats = {
                "ram": f"{ram_match.group(1)} / {ram_match.group(2)} MB" if ram_match else "unknown",
                "swap": f"{swap_match.group(1)} / {swap_match.group(2)} MB" if swap_match else "unknown",
                "gpu": f"{gpu_match.group(1)}%" if gpu_match else "unknown",
                "cpu": cpu_match.group(1) if cpu_match else "unknown",
                "gpu_temp": f"{gpu_temp_match.group(1)} C" if gpu_temp_match else "unknown",
                "cpu_temp": f"{cpu_temp_match.group(1)} C" if cpu_temp_match else "unknown",
                "raw": line,
            }

        except Exception as e:
            latest_stats = {"raw": str(e)}

        time.sleep(2)


def gen_frames():
    global last_alert_time, last_unknown_face_path, last_unknown_save_time

    while True:
        success, frame = cap.read()
        if not success:
            continue

        if monitoring_enabled:
            results = model(frame, imgsz=320, verbose=False)
            person_detected = False
            identity = "Unknown"

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != 0 or conf < 0.45:
                    continue

                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = frame[max(0, y1):y2, max(0, x1):x2]

                identity, face_img = recognize_person(person_crop)

                if identity == "Unknown":
                    now_unknown = time.time()

                if now_unknown - last_unknown_save_time > unknown_save_cooldown_sec:
                    last_unknown_save_time = now_unknown

                    # Prefer face crop. If no face is found, save person crop.
                    img_to_save = face_img if face_img is not None else person_crop

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unknown_path = SNAPSHOT_DIR / "latest_unknown.jpg"

                    cv2.imwrite(str(unknown_path), img_to_save)
                    last_unknown_face_path = str(unknown_path)

                    print("Saved unknown person:", last_unknown_face_path)
                    label = f"{identity} {conf:.2f}" if identity != "Unknown" else f"Unknown person {conf:.2f}"
                    color = (0, 255, 0) if identity != "Unknown" else (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            now = time.time()
            if person_detected and now - last_alert_time > alert_cooldown_sec:
                last_alert_time = now
                image_path = SNAPSHOT_DIR / "latest_person.jpg"
                cv2.imwrite(str(image_path), frame)
                cleanup_alert_images()
                add_alert("Person detected", identity, image_path)

                msg = (
                    f"Person detected\n"
                    f"Identity: {identity}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                threading.Thread(
                    target=send_telegram_alert,
                    args=(msg, image_path),
                    daemon=True,
                ).start()
        else:
            cv2.putText(
                frame,
                "Monitoring stopped",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Jetson AI Security Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .card { background: #1c1c1c; border-radius: 12px; padding: 16px; margin-bottom: 20px; }
        img { width: 100%; border-radius: 8px; }
        button { padding: 12px 18px; margin: 5px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
        input { padding: 10px; width: 80%; border-radius: 6px; border: none; margin: 5px; }
        .start { background: #1f9d55; color: white; }
        .stop { background: #d64545; color: white; }
        .save { background: #2878ff; color: white; }
        .alert { border-bottom: 1px solid #333; padding: 8px 0; }
        small { color: #aaa; word-break: break-word; }
    </style>
</head>
<body>
    <h1>Jetson AI Security Dashboard</h1>

    <div class="grid">
        <div class="card">
            <h2>Live Camera</h2>
            <img src="/video">
        </div>

        <div>
            <div class="card">
                <h2>Monitoring</h2>
                <button class="start" onclick="startMonitoring()">Start</button>
                <button class="stop" onclick="stopMonitoring()">Stop</button>
                <div id="status">Loading...</div>
            </div>

            <div class="card">
                <h2>System Stats</h2>
                <div id="stats">Loading...</div>
            </div>

            <div class="card">
                <h2>Teach Unknown Person</h2>
                <p id="unknownFaceText">Checking...</p>
                <input id="personName" placeholder="Enter person name">
                <button class="save" onclick="labelFace()">Save Name</button>
            </div>

            <div class="card">
                <h2>Alerts</h2>
                <div id="alerts">No alerts yet</div>
            </div>
        </div>
    </div>

<script>
async function startMonitoring() {
    await fetch('/start', {method: 'POST'});
    refreshStatus();
}

async function stopMonitoring() {
    await fetch('/stop', {method: 'POST'});
    refreshStatus();
}

async function refreshStatus() {
    const r = await fetch('/status');
    const data = await r.json();
    document.getElementById('status').innerText =
        data.monitoring ? 'Monitoring: ON' : 'Monitoring: OFF';
}

async function refreshStats() {
    const r = await fetch('/stats');
    const s = await r.json();

    document.getElementById('stats').innerHTML = `
        <p><b>RAM:</b> ${s.ram || 'unknown'}</p>
        <p><b>SWAP:</b> ${s.swap || 'unknown'}</p>
        <p><b>GPU:</b> ${s.gpu || 'unknown'}</p>
        <p><b>CPU:</b> ${s.cpu || 'unknown'}</p>
        <p><b>GPU Temp:</b> ${s.gpu_temp || 'unknown'}</p>
        <p><b>CPU Temp:</b> ${s.cpu_temp || 'unknown'}</p>
        <small>${s.raw || ''}</small>
    `;
}

async function refreshAlerts() {
    const r = await fetch('/alerts');
    const data = await r.json();

    if (data.length === 0) {
        document.getElementById('alerts').innerHTML = 'No alerts yet';
        return;
    }

    document.getElementById('alerts').innerHTML = data.map(a => `
        <div class="alert">
            <b>${a.label}</b><br>
            Identity: ${a.identity}<br>
            Time: ${a.time}
        </div>
    `).join('');
}

async function refreshUnknownFace() {
    const r = await fetch('/unknown-face');
    const data = await r.json();

    if (data.image) {
        document.getElementById('unknownFaceText').innerText =
            'Unknown person detected. Enter their name to remember them.';
    } else {
        document.getElementById('unknownFaceText').innerText =
            'No unknown person waiting.';
    }
}

async function labelFace() {
    const name = document.getElementById('personName').value.trim();

    if (!name) {
        alert('Please enter a name.');
        return;
    }

    const r = await fetch('/label-face', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name})
    });

    const data = await r.json();

    if (data.ok) {
        alert(`Saved ${data.name}. Future detections can recognize this person.`);
        document.getElementById('personName').value = '';
    } else {
        alert(data.error);
    }

    refreshUnknownFace();
}

setInterval(refreshStatus, 2000);
setInterval(refreshStats, 2000);
setInterval(refreshAlerts, 3000);
setInterval(refreshUnknownFace, 3000);

refreshStatus();
refreshStats();
refreshAlerts();
refreshUnknownFace();
</script>
</body>
</html>
"""


@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    global monitoring_enabled
    monitoring_enabled = True
    return jsonify({"monitoring": monitoring_enabled})


@app.route("/stop", methods=["POST"])
def stop():
    global monitoring_enabled
    monitoring_enabled = False
    return jsonify({"monitoring": monitoring_enabled})


@app.route("/status")
def status():
    return jsonify({"monitoring": monitoring_enabled})


@app.route("/stats")
def stats():
    return jsonify(latest_stats)


@app.route("/alerts")
def get_alerts():
    return jsonify(alerts)


@app.route("/unknown-face")
def unknown_face():
    return jsonify({"image": last_unknown_face_path})


@app.route("/label-face", methods=["POST"])
def label_face():
    global last_unknown_face_path

    data = request.get_json()
    name = data.get("name", "").strip() if data else ""

    if not name:
        return jsonify({"ok": False, "error": "Missing name"}), 400

    if not last_unknown_face_path or not Path(last_unknown_face_path).exists():
        return jsonify({"ok": False, "error": "No unknown face available"}), 400

    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()

    if not safe_name:
        return jsonify({"ok": False, "error": "Invalid name"}), 400

    person_dir = KNOWN_FACES_DIR / safe_name
    person_dir.mkdir(parents=True, exist_ok=True)

    count = len(list(person_dir.glob("*.jpg"))) + 1
    dest = person_dir / f"{count}.jpg"

    img = cv2.imread(last_unknown_face_path)
    cv2.imwrite(str(dest), img)

    train_face_recognizer()

    last_unknown_face_path = None

    return jsonify({"ok": True, "saved": str(dest), "name": safe_name})


if __name__ == "__main__":
    train_face_recognizer()
    threading.Thread(target=stats_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
