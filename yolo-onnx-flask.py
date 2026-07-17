import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import threading
from flask import Flask, Response, render_template_string, jsonify, request

# === Konfigurasi model ===
MODEL_OPTIONS = {
    128: "best128.onnx",
    224: "best224.onnx",
    320: "best320.onnx",
    640: "best640.onnx",
}
DEFAULT_SIZE = 128   # paling ringan, untuk deteksi tiap frame

# === Kamera ===
DEFAULT_CAMERA = 1
MAX_CAMERA_SCAN = 5
CAMERA_WIDTH = 320    # lebih kecil agar lebih cepat
CAMERA_HEIGHT = 240
CAMERA_BUFFERSIZE = 1

JPEG_QUALITY = 60      # lebih rendah untuk mempercepat encoding

coco_names_path = "coco.names"
conf_threshold = 0.25
nms_threshold = 0.4

tosca = (208, 224, 64)

# === Load class labels ===
class_names = None
if os.path.exists(coco_names_path):
    with open(coco_names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

def get_label_text(class_id):
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"Class {class_id}"

def letterbox(img, new_shape, color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, ratio, (dw, dh)

def run_yolo_detection(frame, session, input_name, input_size):
    """Melakukan inferensi YOLO dan mengembalikan list deteksi (label, score, x, y, w, h)."""
    try:
        img_input, ratio, (dw, dh) = letterbox(frame, new_shape=(input_size, input_size))
        input_tensor = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        output = session.run(None, {input_name: input_tensor})[0]
        output = np.squeeze(output).T

        class_scores = output[:, 4:]
        scores_all = class_scores.max(axis=1)
        class_ids_all = class_scores.argmax(axis=1)

        conf_mask = scores_all > conf_threshold
        output = output[conf_mask]
        scores = scores_all[conf_mask]
        class_ids = class_ids_all[conf_mask]

        results = []
        if output.shape[0] > 0:
            cx, cy, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
            cx = (cx - dw) / ratio
            cy = (cy - dh) / ratio
            w = w / ratio
            h = h / ratio

            x = (cx - w / 2).astype(int)
            y = (cy - h / 2).astype(int)
            w = w.astype(int)
            h = h.astype(int)

            boxes = np.stack([x, y, w, h], axis=1)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)

            if len(indices) > 0:
                for i in np.array(indices).flatten():
                    bx, by, bw, bh = boxes[i]
                    class_id = int(class_ids[i])
                    label = get_label_text(class_id)
                    results.append((label, float(scores[i]), int(bx), int(by), int(bw), int(bh)))
        return results
    except Exception as e:
        print(f"[YOLO Error] {e}")
        return []

def scan_available_cameras(max_index=MAX_CAMERA_SCAN):
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(idx)
        cap.release()
    return available

def configure_capture(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFERSIZE)
    except Exception:
        pass

# ============================================================
# Kelas DetectorState (deteksi tiap frame)
# ============================================================
class DetectorState:
    def __init__(self, default_size, default_camera):
        self.lock = threading.Lock()
        self.latest_frame = None      # bytes hasil JPEG
        self.latest_frame_id = 0
        self.fps = 0.0
        self.status = "DETECTING"
        self.object_count = 0

        # raw frame dari capture thread
        self.raw_lock = threading.Lock()
        self.raw_frame = None
        self.raw_frame_id = 0
        self.capture_thread = None
        self.capture_running = False

        # inisialisasi kamera
        self.available_cameras = scan_available_cameras()
        if default_camera not in self.available_cameras:
            if self.available_cameras:
                default_camera = self.available_cameras[0]
            else:
                self.available_cameras = [default_camera]

        self.cap = None
        self.current_camera = None
        self._open_camera(default_camera)

        # load model
        self.session = None
        self.input_name = None
        self.input_size = None
        self.current_size = None
        self._load_model(default_size)

        # jalankan thread deteksi
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Thread pembaca kamera, terus update raw_frame."""
        while self.capture_running:
            with self.raw_lock:
                cap = self.cap
            if cap is None:
                time.sleep(0.02)
                continue
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            with self.raw_lock:
                self.raw_frame = frame
                self.raw_frame_id += 1

    def _open_camera(self, index):
        new_cap = cv2.VideoCapture(index)
        if not new_cap.isOpened():
            new_cap.release()
            raise RuntimeError(f"Gagal membuka kamera index: {index}")
        configure_capture(new_cap)

        # hentikan capture thread lama
        if self.capture_running:
            self.capture_running = False
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=2.0)

        with self.raw_lock:
            old_cap = self.cap
            self.cap = new_cap
            self.raw_frame = None
            self.raw_frame_id = 0

        with self.lock:
            self.current_camera = index

        if old_cap is not None:
            old_cap.release()

        # start capture thread baru
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print(f"[Kamera] Ganti ke index {index}")

    def switch_camera(self, index):
        self._open_camera(index)

    def _load_model(self, size):
        if size not in MODEL_OPTIONS:
            raise ValueError(f"Ukuran model tidak dikenal: {size}")
        model_path = MODEL_OPTIONS[size]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        with self.lock:
            self.session = session
            self.input_name = session.get_inputs()[0].name
            self.input_size = size
            self.current_size = size
        print(f"[Model] Ganti ke ukuran {size} -> {model_path}")

    def switch_model(self, size):
        self._load_model(size)

    def _loop(self):
        """Loop utama deteksi - setiap frame."""
        last_seen_id = -1
        while self.running:
            try:
                # ambil frame terbaru
                with self.raw_lock:
                    frame_id = self.raw_frame_id
                    frame = self.raw_frame.copy() if self.raw_frame is not None else None

                if frame is None or frame_id == last_seen_id:
                    time.sleep(0.005)
                    continue
                last_seen_id = frame_id

                with self.lock:
                    session = self.session
                    input_name = self.input_name
                    input_size = self.input_size

                start_time = time.time()

                # Deteksi setiap frame
                detections = run_yolo_detection(frame, session, input_name, input_size)

                # Gambar bounding box
                for label, score, bx, by, bw, bh in detections:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), tosca, 2)
                    text = f"{label}: {score:.2f}"
                    cv2.putText(frame, text, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tosca, 2)

                elapsed = time.time() - start_time
                fps = 1 / elapsed if elapsed > 0 else 0

                with self.lock:
                    current_size = self.current_size
                    current_camera = self.current_camera

                cv2.putText(frame, f"FPS: {fps:.2f}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(frame, f"DETECTING | {current_size}px | Cam {current_camera}", (40, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # encode ke JPEG
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self.lock:
                        self.latest_frame = buf.tobytes()
                        self.latest_frame_id += 1
                        self.fps = fps
                        self.status = "DETECTING"
                        self.object_count = len(detections)
            except Exception as e:
                print(f"[Loop Error] {e}")
                time.sleep(0.1)

    def get_jpeg_if_new(self, last_id):
        with self.lock:
            if self.latest_frame_id != last_id and self.latest_frame is not None:
                return self.latest_frame, self.latest_frame_id
            return None, last_id

    def get_status(self):
        with self.lock:
            return {
                "status": self.status,
                "fps": round(self.fps, 2),
                "objects": self.object_count,
                "model_size": self.current_size,
                "camera": self.current_camera,
                "available_cameras": self.available_cameras,
            }

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
detector = DetectorState(DEFAULT_SIZE, DEFAULT_CAMERA)

PAGE = """
<!doctype html>
<html>
<head>
  <title>YOLOv8 ONNX - Deteksi Tiap Frame</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; text-align: center; }
    img { border: 2px solid #444; margin-top: 10px; max-width: 90%; }
    #status { margin-top: 10px; font-size: 1.1em; }
    .controls { margin-top: 15px; }
    button { margin: 4px; padding: 8px 16px; font-size: 1em; cursor: pointer; }
    button.active { background: #2b7; color: white; }
  </style>
</head>
<body>
  <h2>YOLOv8 ONNX - Deteksi Setiap Frame</h2>
  <img src="{{ url_for('video_feed') }}">
  <div id="status">Status: - | FPS: - | Objek: - | Model: - | Kamera: -</div>

  <div class="controls">
    <b>Ukuran model:</b><br>
    {% for size in sizes %}
      <button id="btn-model-{{ size }}" onclick="switchModel({{ size }})">{{ size }}px</button>
    {% endfor %}
  </div>

  <div class="controls">
    <b>Kamera:</b><br>
    {% for cam in cameras %}
      <button id="btn-cam-{{ cam }}" onclick="switchCamera({{ cam }})">Cam {{ cam }}</button>
    {% endfor %}
  </div>

  <script>
    async function updateStatus() {
      const res = await fetch('/status');
      const data = await res.json();
      document.getElementById('status').innerText =
        `Status: ${data.status} | FPS: ${data.fps} | Objek: ${data.objects} | Model: ${data.model_size}px | Kamera: ${data.camera}`;

      document.querySelectorAll('[id^="btn-model-"]').forEach(b => b.classList.remove('active'));
      const activeModelBtn = document.getElementById('btn-model-' + data.model_size);
      if (activeModelBtn) activeModelBtn.classList.add('active');

      document.querySelectorAll('[id^="btn-cam-"]').forEach(b => b.classList.remove('active'));
      const activeCamBtn = document.getElementById('btn-cam-' + data.camera);
      if (activeCamBtn) activeCamBtn.classList.add('active');
    }
    async function switchModel(size) {
      await fetch('/switch_model?size=' + size, { method: 'POST' });
    }
    async function switchCamera(index) {
      await fetch('/switch_camera?index=' + index, { method: 'POST' });
    }
    setInterval(updateStatus, 500);
    updateStatus();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        PAGE,
        sizes=sorted(MODEL_OPTIONS.keys()),
        cameras=detector.available_cameras,
    )

def mjpeg_generator():
    last_id = -1
    while True:
        frame, last_id = detector.get_jpeg_if_new(last_id)
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.005)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify(detector.get_status())

@app.route("/switch_model", methods=["POST"])
def switch_model():
    size = request.args.get("size", type=int)
    if size not in MODEL_OPTIONS:
        return jsonify({"ok": False, "error": "Ukuran tidak valid"}), 400
    try:
        detector.switch_model(size)
        return jsonify({"ok": True, "model_size": size})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 404

@app.route("/switch_camera", methods=["POST"])
def switch_camera():
    index = request.args.get("index", type=int)
    if index is None:
        return jsonify({"ok": False, "error": "Parameter 'index' wajib diisi"}), 400
    try:
        detector.switch_camera(index)
        return jsonify({"ok": True, "camera": index})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 404

@app.route("/rescan_cameras", methods=["POST"])
def rescan_cameras():
    found = scan_available_cameras()
    detector.available_cameras = found
    return jsonify({"ok": True, "available_cameras": found})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)