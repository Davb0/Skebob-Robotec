import cv2
import numpy as np
import threading
from flask import Flask, Response, render_template
from picamera2 import Picamera2

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_CENTER_X = FRAME_WIDTH // 2
IMAGE_CENTER_Y = FRAME_HEIGHT // 2

# HSV color range for tracking target (tune these for your object)
HSV_LOWER = np.array([100, 150, 50])   # blue-ish by default
HSV_UPPER = np.array([140, 255, 255])

app = Flask(__name__)

# Shared state between capture thread and Flask stream
_frame_lock = threading.Lock()
_latest_frame = None


# ---------------------------------------------------------------------------
# Arduino communication stub
# Replace the print() calls below with pyserial writes when ready, e.g.:
#   ser.write(f"PAN:{pan_error},TILT:{tilt_error}\n".encode())
# ---------------------------------------------------------------------------
def send_to_arduino(pan_error: int, tilt_error: int) -> None:
    """Send pan/tilt error values to the Arduino motor driver."""
    print(f"[ARDUINO] pan_error={pan_error:+d}  tilt_error={tilt_error:+d}")


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
def find_target_hsv(frame: np.ndarray):
    """Return (cx, cy) of the largest HSV-matched blob, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:   # ignore tiny blobs
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def annotate_frame(frame: np.ndarray, target_pos, pan_error: int, tilt_error: int) -> np.ndarray:
    """Draw crosshair, target marker, and error text onto a copy of frame."""
    out = frame.copy()
    # Image centre crosshair
    cv2.line(out, (IMAGE_CENTER_X - 20, IMAGE_CENTER_Y), (IMAGE_CENTER_X + 20, IMAGE_CENTER_Y), (0, 255, 0), 1)
    cv2.line(out, (IMAGE_CENTER_X, IMAGE_CENTER_Y - 20), (IMAGE_CENTER_X, IMAGE_CENTER_Y + 20), (0, 255, 0), 1)

    if target_pos:
        cx, cy = target_pos
        cv2.circle(out, (cx, cy), 10, (0, 0, 255), 2)
        cv2.line(out, (IMAGE_CENTER_X, IMAGE_CENTER_Y), (cx, cy), (255, 0, 0), 1)
        cv2.putText(out, f"err ({pan_error:+d}, {tilt_error:+d})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(out, "No target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out


# ---------------------------------------------------------------------------
# Capture loop (runs in a background thread)
# ---------------------------------------------------------------------------
def capture_loop() -> None:
    global _latest_frame

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    ))
    cam.start()
    print("Camera started. Starting tracking loop...")

    while True:
        frame = cam.capture_array()

        target_pos = find_target_hsv(frame)

        if target_pos:
            cx, cy = target_pos
            pan_error = cx - IMAGE_CENTER_X
            tilt_error = IMAGE_CENTER_Y - cy
            send_to_arduino(pan_error, tilt_error)
        else:
            pan_error = tilt_error = 0

        annotated = annotate_frame(frame, target_pos, pan_error, tilt_error)

        with _frame_lock:
            _latest_frame = annotated


# ---------------------------------------------------------------------------
# Flask MJPEG stream
# ---------------------------------------------------------------------------
def generate_mjpeg():
    import time
    while True:
        time.sleep(0.033)  # ~30 fps cap

        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            continue

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    print("Dashboard → http://<raspberry-pi-ip>:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
