import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template, jsonify
from picamera2 import Picamera2

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_CENTER_X = FRAME_WIDTH // 2
IMAGE_CENTER_Y = FRAME_HEIGHT // 2

MIN_BLOB_AREA = 200
MAX_JUMP = 200
MAX_AREA_RATIO = 4.0
GRACE_FRAMES = 10
REACQUIRE_FRAMES = 45   # frames to search before giving up (~1.5 s at 30 fps)
REACQUIRE_RADIUS = 180  # px — search radius around last known position

# Tracker states
IDLE      = "idle"
LOCKED    = "locked"
SEARCHING = "searching"

app = Flask(__name__)
_frame_lock = threading.Lock()
_latest_frame = None
_latest_raw = None

# Shared tracker state (read by Flask routes)
_state_lock = threading.Lock()
_shared_state = IDLE
_snapshot_request = False
_reset_request = False
_shared_telemetry = {"pan": 0, "tilt": 0, "tx": 0, "ty": 0, "tw": 0, "th": 0}


# ---------------------------------------------------------------------------
# Arduino stub
# ---------------------------------------------------------------------------
def send_to_arduino(pan_error: int, tilt_error: int) -> None:
    print(f"[ARDUINO] pan_error={pan_error:+d}  tilt_error={tilt_error:+d}")


# ---------------------------------------------------------------------------
# MOG2 motion detection
# ---------------------------------------------------------------------------
def make_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=40, detectShadows=False
    )


def find_target_motion(fg_mask: np.ndarray, last_pos=None, last_area=None,
                       max_jump=MAX_JUMP):
    """Return (cx, cy, x, y, w, h, area) of the best moving blob, or None."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.dilate(clean, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        candidates.append((cx, cy, x, y, w, h, area))

    if not candidates:
        return None

    if last_pos:
        lx, ly = last_pos
        valid = [b for b in candidates
                 if np.hypot(b[0] - lx, b[1] - ly) <= max_jump
                 and (last_area is None
                      or 1 / MAX_AREA_RATIO <= b[6] / last_area <= MAX_AREA_RATIO)]
        if not valid:
            return None
        return min(valid, key=lambda b: np.hypot(b[0] - lx, b[1] - ly))

    return max(candidates, key=lambda b: b[6])


def init_csrt(frame, x, y, w, h):
    t = cv2.TrackerCSRT_create()
    t.init(frame, (x, y, w, h))
    return t


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
_STATE_STYLE = {
    IDLE:      {"color": (0, 0, 255),   "label": "TRACKING"},
    LOCKED:    {"color": (0, 165, 255), "label": "LOCKED"},
    SEARCHING: {"color": (0, 200, 255), "label": "SEARCHING"},
}

def annotate_frame(frame, box, cx, cy, pan_error, tilt_error, state):
    out = frame.copy()
    cx0, cy0 = IMAGE_CENTER_X, IMAGE_CENTER_Y
    cv2.line(out, (cx0 - 20, cy0), (cx0 + 20, cy0), (0, 255, 0), 1)
    cv2.line(out, (cx0, cy0 - 20), (cx0, cy0 + 20), (0, 255, 0), 1)

    style = _STATE_STYLE[state]
    if box:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), style["color"], 2)
        cv2.line(out, (cx0, cy0), (cx, cy), (255, 0, 0), 1)
        cv2.putText(out, f"{style['label']}  err ({pan_error:+d}, {tilt_error:+d})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif state == SEARCHING:
        cv2.putText(out, "SEARCHING...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, style["color"], 2)
    else:
        cv2.putText(out, "No target", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return out


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
def capture_loop() -> None:
    global _latest_frame, _latest_raw, _shared_state, _snapshot_request, _reset_request, _shared_telemetry

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
    ))
    cam.start()
    print("Camera started.")

    bg_sub = make_bg_subtractor()
    state = IDLE
    csrt = None
    last_pos = None
    last_area = None
    last_known_pos = None
    last_known_area = None
    lost_count = 0
    search_count = 0

    while True:
        frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        with _frame_lock:
            _latest_raw = frame

        # Check for lock request from UI
        with _state_lock:
            do_snap = _snapshot_request
            _snapshot_request = False
            do_reset = _reset_request
            _reset_request = False

        if do_reset:
            state = IDLE
            csrt = None
            last_pos = last_area = None
            last_known_pos = last_known_area = None
            lost_count = search_count = 0
            print("[RESET] Tracker reset to IDLE")

        if do_snap:
            mot = find_target_motion(fg, last_pos, last_area)
            if mot:
                _, _, x, y, w, h, area = mot
                csrt = init_csrt(frame, x, y, w, h)
                state = LOCKED
                last_pos = (x + w // 2, y + h // 2)
                last_area = area
                search_count = 0
                print(f"[LOCK] CSRT init at ({x},{y},{w},{h})")
            else:
                print("[LOCK] No target visible to lock onto")

        # Always update MOG2 so the background model stays fresh
        fg = bg_sub.apply(frame)

        box = None
        cx = cy = 0

        # ---- State machine ----
        if state == LOCKED:
            ok, rect = csrt.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in rect)
                cx, cy = x + w // 2, y + h // 2
                box = (x, y, w, h)
                last_pos = (cx, cy)
                last_known_pos = (cx, cy)
                last_area = w * h
                last_known_area = w * h
                lost_count = 0
            else:
                print("[LOCK] Target lost — searching...")
                state = SEARCHING
                search_count = 0
                csrt = None

        elif state == SEARCHING:
            mot = find_target_motion(fg, last_known_pos, last_known_area,
                                     max_jump=REACQUIRE_RADIUS)
            if mot:
                _, _, x, y, w, h, area = mot
                csrt = init_csrt(frame, x, y, w, h)
                state = LOCKED
                last_pos = (x + w // 2, y + h // 2)
                last_area = area
                search_count = 0
                print("[LOCK] Re-acquired target.")
            else:
                search_count += 1
                if search_count > REACQUIRE_FRAMES:
                    print("[LOCK] Re-acquisition failed — resetting.")
                    state = IDLE
                    csrt = None
                    last_pos = None
                    last_area = None
                    last_known_pos = None
                    last_known_area = None

        else:  # IDLE
            mot = find_target_motion(fg, last_pos, last_area)
            if mot:
                cx, cy, x, y, w, h, area = mot
                box = (x, y, w, h)
                last_pos = (cx, cy)
                last_area = area
                lost_count = 0
            else:
                lost_count += 1
                if lost_count > GRACE_FRAMES:
                    last_pos = None
                    last_area = None

        with _state_lock:
            _shared_state = state

        if box:
            pan_error = cx - IMAGE_CENTER_X
            tilt_error = IMAGE_CENTER_Y - cy
            send_to_arduino(pan_error, tilt_error)
        else:
            pan_error = tilt_error = 0

        x, y, w, h = box if box else (0, 0, 0, 0)
        with _state_lock:
            _shared_telemetry = {"pan": pan_error, "tilt": tilt_error,
                                 "tx": cx, "ty": cy, "tw": w, "th": h}

        annotated = annotate_frame(frame, box, cx, cy, pan_error, tilt_error, state)
        with _frame_lock:
            _latest_frame = annotated


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
def generate_mjpeg():
    while True:
        time.sleep(0.033)
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/lock", methods=["POST"])
def lock_target():
    global _snapshot_request
    with _state_lock:
        _snapshot_request = True
    return jsonify({"status": "lock requested"})

@app.route("/reset", methods=["POST"])
def reset_tracker():
    global _reset_request
    with _state_lock:
        _reset_request = True
    return jsonify({"status": "reset"})

@app.route("/status")
def tracker_status():
    with _state_lock:
        return jsonify({"state": _shared_state, **_shared_telemetry})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    print("Dashboard → http://<raspberry-pi-ip>:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
