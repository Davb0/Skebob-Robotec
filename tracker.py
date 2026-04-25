import os
import sys
import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo
from motor import Motor
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_CENTER_X = FRAME_WIDTH // 2
IMAGE_CENTER_Y = FRAME_HEIGHT // 2

MIN_BLOB_AREA = 200
MAX_JUMP = 200
MAX_AREA_RATIO = 4.0
GRACE_FRAMES = 10
REACQUIRE_FRAMES = 45
REACQUIRE_RADIUS = 180
AUTO_LOCK_FRAMES = 8    # consecutive MOG2 detections before auto-locking

PAN_CHANNEL  = 0
TILT_CHANNEL = 1
SERVO_DEADZONE  = 15    # px — ignore error smaller than this
SERVO_STEP_GAIN = 0.03  # px of error → degrees of step
SERVO_STEP_MAX  = 2.0   # degrees — hard cap per frame (prevents oscillation)
SERVO_SEARCH_MAX = 4.5  # degrees — faster sweep when target left FOV
EMA_ALPHA       = 0.20  # smoothing factor (lower = smoother but laggier)
CSRT_PADDING = 0.4      # fractional padding added around blob before CSRT init
CLICK_BOX_SIZE = 40     # px — CSRT box half-size when locking via click
TILT_IDLE_ANGLE = 60.0  # degrees — tilt rests here until a target is locked
MOUSE_EMA = 0.12        # how fast camera follows mouse position (higher = faster)

# Tracker states
IDLE      = "idle"
LOCKED    = "locked"
SEARCHING = "searching"

# ---------------------------------------------------------------------------
# Servo init
# ---------------------------------------------------------------------------
_i2c = board.I2C()
_pca = PCA9685(_i2c)
_pca.frequency = 50
_pan_servo  = adafruit_servo.Servo(_pca.channels[PAN_CHANNEL])
_tilt_servo = adafruit_servo.Servo(_pca.channels[TILT_CHANNEL])
_pan_angle  = 90.0
_tilt_angle = TILT_IDLE_ANGLE
_pan_servo.angle  = _pan_angle
_tilt_servo.angle = _tilt_angle
_servo_lock = threading.Lock()


def _clamp(val, lo=0.0, hi=180.0):
    return max(lo, min(hi, val))


def _snap_to_home() -> None:
    global _pan_angle, _tilt_angle
    with _servo_lock:
        try:
            _pan_angle  = 90.0
            _tilt_angle = TILT_IDLE_ANGLE
            _pan_servo.angle  = _pan_angle
            _tilt_servo.angle = _tilt_angle
        except OSError as e:
            print(f"[SERVO] I2C error on home: {e}")


def move_servos(pan_error: int, tilt_error: int, step_max: float = SERVO_STEP_MAX) -> None:
    global _pan_angle, _tilt_angle
    with _servo_lock:
        try:
            if abs(pan_error) > SERVO_DEADZONE:
                step = min(abs(pan_error) * SERVO_STEP_GAIN, step_max)
                _pan_angle = _clamp(_pan_angle - (step if pan_error > 0 else -step))
                _pan_servo.angle = _pan_angle
            if abs(tilt_error) > SERVO_DEADZONE:
                step = min(abs(tilt_error) * SERVO_STEP_GAIN, step_max)
                _tilt_angle = _clamp(_tilt_angle - (step if tilt_error > 0 else -step))
                _tilt_servo.angle = _tilt_angle
        except OSError as e:
            print(f"[SERVO] I2C error: {e}")


# ---------------------------------------------------------------------------
# Motor init
# ---------------------------------------------------------------------------
try:
    left_motor  = Motor(19, 18, 12)   # IN2, IN1, ENA
    right_motor = Motor(20, 21, 13)   # IN3, IN4, ENB
    print("Motors initialized.")
except Exception as e:
    print(f"[MOTOR] Init failed: {e} — driving disabled.")
    left_motor = right_motor = None


def drive_motor(motor, speed: float) -> None:
    if motor is None:
        return
    speed = max(-1.0, min(1.0, speed))
    if speed > 0:    motor.forward(speed)
    elif speed < 0:  motor.backward(-speed)
    else:            motor.stop()


# ---------------------------------------------------------------------------
# Flask / shared state
# ---------------------------------------------------------------------------
app = Flask(__name__)
_frame_lock = threading.Lock()
_latest_frame = None

_state_lock = threading.Lock()
_shared_state       = IDLE
_click_lock_request = None   # (norm_x, norm_y) from browser click, or None
_reset_request      = False
_mouse_control      = False
_mouse_nx           = 0.5
_mouse_ny           = 0.5
_shared_telemetry = {"pan_err": 0, "tilt_err": 0,
                     "tx": 0, "ty": 0, "tw": 0, "th": 0,
                     "pan_angle": 90, "tilt_angle": 90,
                     "mouse_control": False}


# ---------------------------------------------------------------------------
# MOG2 motion detection
# ---------------------------------------------------------------------------
def make_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=40, detectShadows=False
    )


def find_target_motion(fg_mask, last_pos=None, last_area=None, max_jump=MAX_JUMP):
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
    # Make box square (use the larger dimension) then add padding.
    # This gives CSRT equal context in all directions so it won't
    # lose the target when it moves perpendicular to its detected shape.
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half = int(size * (1 + CSRT_PADDING) / 2)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(FRAME_WIDTH,  cx + half)
    y2 = min(FRAME_HEIGHT, cy + half)
    t = cv2.TrackerCSRT_create()
    t.init(frame, (x1, y1, x2 - x1, y2 - y1))
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
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
    global _latest_frame, _shared_state, _click_lock_request, _reset_request, _shared_telemetry
    global _mouse_control, _mouse_nx, _mouse_ny, _pan_angle, _tilt_angle

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"},
        controls={"FrameRate": 15}
    ))
    cam.start()
    print("Camera started.")

    bg_sub = make_bg_subtractor()
    state         = IDLE
    csrt          = None
    last_pos      = None
    last_area     = None
    last_pan_error  = 0
    last_tilt_error = 0
    smooth_pan  = 0.0
    smooth_tilt = 0.0
    last_known_pos  = None
    last_known_area = None
    lost_count    = 0
    search_count  = 0

    while True:
        frame = cv2.rotate(cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR), cv2.ROTATE_180)

        with _state_lock:
            do_click  = _click_lock_request; _click_lock_request = None
            do_reset  = _reset_request;      _reset_request      = False
            mouse_on  = _mouse_control
            mouse_nx  = _mouse_nx
            mouse_ny  = _mouse_ny

        if do_reset:
            state = IDLE; csrt = None
            last_pos = last_area = None
            last_known_pos = last_known_area = None
            lost_count = search_count = 0
            _snap_to_home()
            print("[RESET] Tracker reset to IDLE")

        # Always keep MOG2 background model fresh
        fg = bg_sub.apply(frame)

        # Click-to-lock: init CSRT centered on the clicked point
        if do_click is not None:
            nx, ny = do_click
            fx = int(nx * FRAME_WIDTH)
            fy = int(ny * FRAME_HEIGHT)
            half = CLICK_BOX_SIZE
            x1 = max(0, fx - half); y1 = max(0, fy - half)
            x2 = min(FRAME_WIDTH, fx + half); y2 = min(FRAME_HEIGHT, fy + half)
            csrt  = cv2.TrackerCSRT_create()
            csrt.init(frame, (x1, y1, x2 - x1, y2 - y1))
            state = LOCKED
            last_pos = last_known_pos = (fx, fy)
            last_area = last_known_area = (x2 - x1) * (y2 - y1)
            print(f"[CLICK-LOCK] CSRT init at frame ({fx},{fy})")

        box = None; cx = cy = 0

        # ── State machine ──────────────────────────────────────────────────
        if state == LOCKED:
            ok, rect = csrt.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in rect)
                cx, cy = x + w // 2, y + h // 2
                box = (x, y, w, h)
                last_pos = last_known_pos = (cx, cy)
                last_area = last_known_area = w * h
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
                csrt  = init_csrt(frame, x, y, w, h)
                state = LOCKED
                last_pos  = (x + w // 2, y + h // 2)
                last_area = area
                search_count = 0
                print("[LOCK] Re-acquired.")
            else:
                search_count += 1
                if search_count > REACQUIRE_FRAMES:
                    print("[LOCK] Re-acquisition failed — IDLE.")
                    state = IDLE; csrt = None
                    last_pos = last_area = None
                    last_known_pos = last_known_area = None
                    search_count = 0
                    _snap_to_home()

        else:  # IDLE
            mot = find_target_motion(fg, last_pos, last_area)
            if mot:
                cx, cy, x, y, w, h, area = mot
                box = (x, y, w, h)
                last_pos  = (cx, cy)
                last_area = area
                lost_count = 0
            else:
                lost_count += 1
                if lost_count > GRACE_FRAMES:
                    last_pos = last_area = None

        # ── Servo control ──────────────────────────────────────────────────
        pan_error = tilt_error = 0
        if mouse_on:
            # Position control: mouse maps directly to a target servo angle
            target_pan  = 30.0 + mouse_nx * 120.0   # 0-1 → 30-150°
            target_tilt = 30.0 + mouse_ny * 120.0
            with _servo_lock:
                try:
                    _pan_angle  += MOUSE_EMA * (target_pan  - _pan_angle)
                    _tilt_angle += MOUSE_EMA * (target_tilt - _tilt_angle)
                    _pan_servo.angle  = _pan_angle
                    _tilt_servo.angle = _tilt_angle
                except OSError as e:
                    print(f"[SERVO] I2C error: {e}")
        elif box:
            smooth_pan  = EMA_ALPHA * (cx - IMAGE_CENTER_X)  + (1 - EMA_ALPHA) * smooth_pan
            smooth_tilt = EMA_ALPHA * (IMAGE_CENTER_Y - cy)  + (1 - EMA_ALPHA) * smooth_tilt
            pan_error  = int(smooth_pan)
            tilt_error = int(smooth_tilt)
            move_servos(pan_error, tilt_error)
            last_pan_error  = pan_error
            last_tilt_error = tilt_error
        elif state == SEARCHING:
            move_servos(last_pan_error, last_tilt_error, step_max=SERVO_SEARCH_MAX)

        with _servo_lock:
            pa, ta = _pan_angle, _tilt_angle

        with _state_lock:
            _shared_state = state
            _shared_telemetry = {
                "pan_err": pan_error, "tilt_err": tilt_error,
                "tx": cx, "ty": cy,
                "tw": box[2] if box else 0, "th": box[3] if box else 0,
                "pan_angle": round(pa, 1), "tilt_angle": round(ta, 1),
                "mouse_control": mouse_on,
            }

        annotated = annotate_frame(frame, box, cx, cy, pan_error, tilt_error, state)
        with _frame_lock:
            _latest_frame = annotated


# ---------------------------------------------------------------------------
# Joystick — drives motors + manual servo buttons + dashboard telemetry
# ---------------------------------------------------------------------------
_joy_lock  = threading.Lock()
_joy_state = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0,
               "lt": 0.0, "rt": 0.0, "connected": False, "name": ""}
JOY_DEADZONE = 0.15

# Face-button → (servo axis, step degrees)
_SERVO_BUTTONS = {
    4: ("tilt", -5),   # Y — tilt up
    0: ("tilt", +5),   # A — tilt down
    1: ("pan",  -5),   # B — pan right
    3: ("pan",  +5),   # X — pan left
}

def _apply_servo_button(button: int) -> None:
    global _pan_angle, _tilt_angle
    if button not in _SERVO_BUTTONS:
        return
    axis, step = _SERVO_BUTTONS[button]
    with _servo_lock:
        try:
            if axis == "pan":
                _pan_angle = _clamp(_pan_angle + step)
                _pan_servo.angle = _pan_angle
            else:
                _tilt_angle = _clamp(_tilt_angle + step)
                _tilt_servo.angle = _tilt_angle
        except OSError as e:
            print(f"[SERVO] I2C error: {e}")


def joystick_loop():
    pygame.init()
    pygame.joystick.init()
    joy = None
    axes = {}

    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axes[event.axis] = event.value
                # Drive motors on every stick update
                ly = -axes.get(1, 0.0)
                ry = -axes.get(3, 0.0)
                if abs(ly) < JOY_DEADZONE: ly = 0.0
                if abs(ry) < JOY_DEADZONE: ry = 0.0
                drive_motor(left_motor,  ly)
                drive_motor(right_motor, ry)
            elif event.type == pygame.JOYBUTTONDOWN:
                _apply_servo_button(event.button)

        # Hot-plug detection
        count = pygame.joystick.get_count()
        if count > 0 and joy is None:
            joy = pygame.joystick.Joystick(0)
            joy.init()
            axes = {i: 0.0 for i in range(joy.get_numaxes())}
            print(f"[JOY] Connected: {joy.get_name()}")
        elif count == 0 and joy is not None:
            print("[JOY] Disconnected — stopping motors.")
            drive_motor(left_motor, 0)
            drive_motor(right_motor, 0)
            joy = None
            axes = {}

        # Update dashboard state
        if joy:
            def dz(v): return 0.0 if abs(v) < JOY_DEADZONE else round(v, 3)
            with _joy_lock:
                _joy_state.update({
                    "lx":  dz(axes.get(0, 0.0)),
                    "ly": -dz(axes.get(1, 0.0)),
                    "rx":  dz(axes.get(2, 0.0)),
                    "ry": -dz(axes.get(3, 0.0)),
                    "lt":  max(0.0, axes.get(4, 0.0)),
                    "rt":  max(0.0, axes.get(5, 0.0)),
                    "connected": True, "name": joy.get_name(),
                })
        else:
            with _joy_lock:
                _joy_state.update({"lx":0,"ly":0,"rx":0,"ry":0,
                                   "lt":0,"rt":0,"connected":False,"name":""})
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
def generate_mjpeg():
    while True:
        time.sleep(0.066)
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

@app.route("/click-lock", methods=["POST"])
def click_lock():
    global _click_lock_request
    data = request.get_json(force=True)
    with _state_lock:
        _click_lock_request = (float(data["x"]), float(data["y"]))
    return jsonify({"ok": True})

@app.route("/reset", methods=["POST"])
def reset_tracker():
    global _reset_request
    with _state_lock:
        _reset_request = True
    return jsonify({"ok": True})

@app.route("/mouse-toggle", methods=["POST"])
def mouse_toggle():
    global _mouse_control, _mouse_nx, _mouse_ny
    with _state_lock:
        _mouse_control = not _mouse_control
        if not _mouse_control:
            _mouse_nx = _mouse_ny = 0.5
        val = _mouse_control
    return jsonify({"mouse_control": val})

@app.route("/mouse-move", methods=["POST"])
def mouse_move_route():
    global _mouse_nx, _mouse_ny
    data = request.get_json(force=True)
    _mouse_nx = max(0.0, min(1.0, float(data.get("x", 0.5))))
    _mouse_ny = max(0.0, min(1.0, float(data.get("y", 0.5))))
    return "", 204


@app.route("/status")
def tracker_status():
    with _state_lock:
        return jsonify({"state": _shared_state, **_shared_telemetry})

@app.route("/joystick")
def joystick_status():
    with _joy_lock:
        return jsonify(_joy_state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=capture_loop,  daemon=True).start()
    threading.Thread(target=joystick_loop, daemon=True).start()
    print("Dashboard → http://<raspberry-pi-ip>:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
