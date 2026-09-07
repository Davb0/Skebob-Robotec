import os


class Cfg:
    # ── Cameras ──────────────────────────────────────────────────────────────
    CAM0_INDEX = 0          # RGB webcam  (video0)
    CAM1_INDEX = 2          # IR camera   (video2 — grayscale Windows-Hello sensor)
    FRAME_W = 640
    FRAME_H = 360
    FRAME_RATE = 20
    JPEG_QUALITY = 65

    # ── Vision / Tracker ─────────────────────────────────────────────────────
    CLICK_HALF = 45         # px half-size of initial click ROI
    MOG2_HISTORY = 100
    MOG2_THRESH = 60
    MIN_BLOB_AREA = 250
    MAX_JUMP_PX = 220
    MAX_AREA_RATIO = 4.5
    REACQUIRE_FRAMES = 50
    REACQUIRE_RADIUS = 200
    TRACKER_PADDING = 0.35
    EMA_ALPHA = 0.35
    DEADZONE = 12           # px — ignore error smaller than this

    # ── PID / Servo ──────────────────────────────────────────────────────────
    PAN_HOME = 90.0
    TILT_HOME = 90.0
    PAN_MIN, PAN_MAX = 0.0, 180.0
    TILT_MIN, TILT_MAX = 0.0, 180.0
    PAN_KP, PAN_KI, PAN_KD = 0.045, 0.0005, 0.012
    TILT_KP, TILT_KI, TILT_KD = 0.040, 0.0004, 0.010
    PID_INTEGRAL_CLAMP = 30.0
    SERVO_MAX_STEP = 3.5
    JOY_SERVO_STEP = 5.0

    # ── Color memory ─────────────────────────────────────────────────────────
    COLOR_H_BINS = 36
    COLOR_S_BINS = 32
    COLOR_DRIFT_THRESH = 0.45
    COLOR_REACQ_THRESH = 0.38
    COLOR_HIST_UPDATE = 0.08
    COLOR_SAT_MIN = 30
    COLOR_BP_MIN_SCORE = 0.18

    # ── Chassis ──────────────────────────────────────────────────────────────
    CHASSIS_PAN_TRIGGER = 30.0
    CHASSIS_PAN_TARGET = 90.0
    CHASSIS_TURN_SPEED = 0.40
    CHASSIS_LOCK_TILT_MAX = 80

    # ── Joystick ─────────────────────────────────────────────────────────────
    JOY_DEADZONE = 0.15
    JOY_AXIS_LEFT_Y = 1
    JOY_AXIS_RIGHT_Y = 3

    # ── LiDAR ────────────────────────────────────────────────────────────────
    LIDAR_PORT = os.environ.get("LIDAR_PORT", "/dev/ttyUSB0")
    LIDAR_SIMULATED = True          # False → connect real RPLIDAR

    # ── Server ───────────────────────────────────────────────────────────────
    HOST = "0.0.0.0"
    PORT = 8000
    STATUS_HZ = 10          # WebSocket status broadcast rate
    LIDAR_HZ = 5            # WebSocket lidar broadcast rate
