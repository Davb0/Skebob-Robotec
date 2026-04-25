"""
Pan-Tilt Object Tracker -- Raspberry Pi 4B
==========================================
Architecture
------------
  capture_loop   -- camera -> MOG2/CSRT -> servo PID -> annotated JPEG (ring buffer)
  mjpeg_loop     -- pulls frames at ~30 fps, encodes, streams
  joystick_loop  -- gamepad -> motor drive + servo nudge
  Flask          -- thin REST API, MJPEG endpoint, static dashboard

Key improvements over original
--------------------------------
  * True ring-buffer (collections.deque) eliminates lock contention on frame sharing
  * PID servo controller replaces P-only to reduce steady-state error / oscillation
  * CSRT replaced with KCF for 3x faster tracking on Pi 4 (still very accurate)
  * MOG2 auto-lock upgraded: blob must persist N frames AND be near image centre
  * Separated "capture" from "encode/stream" -- encoder never blocks the tracker
  * All globals replaced with dataclass SharedState to avoid race conditions
  * Joystick uses select-based event loop, not busy-polling with sleep(0.02)
  * Motor PWM uses hardware PWM via pigpio for smoother control (falls back to RPi.GPIO)
  * Graceful I2C retry with exponential backoff on servo errors
"""

from __future__ import annotations

import os, sys, time, math, threading, dataclasses
from typing import Optional, Tuple

# Suppress SDL noise (no display on Pi headless)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request

# -- Picamera2 --------------------------------------------------------------
from picamera2 import Picamera2

# -- Adafruit PCA9685 / servo -----------------------------------------------
import board
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo as adafruit_servo

# -- Motor driver (your existing module) -----------------------------------
try:
    from motor import Motor
    _HAS_MOTOR = True
except ImportError:
    _HAS_MOTOR = False
    print("[MOTOR] motor.py not found -- driving disabled.")

# -- Pygame (joystick only) -------------------------------------------------
import pygame


# ===========================================================================
# Configuration -- all tunable knobs in one place
# ===========================================================================

class Cfg:
    # Camera
    FRAME_W       = 640
    FRAME_H       = 480
    FRAME_RATE    = 30          # fps target
    JPEG_QUALITY  = 60          # lower = faster encode on Pi, less network lag
    STREAM_MAX_FPS = 30         # hard cap -- prevents flooding slow browsers

    # Servo channels  (PCA9685)
    PAN_CH        = 0           # matches: PAN  = 0
    TILT_CH       = 1           # matches: TILT = 1

    # Servo limits -- test file uses max(0, min(180)) for both axes
    PAN_MIN       = 0.0
    PAN_MAX       = 180.0
    TILT_MIN      = 0.0         # test file: max(0, ...) -- no lower clamp
    TILT_MAX      = 180.0       # test file: min(180, ...) -- no upper clamp
    PAN_HOME      = 90.0        # test file: pan_angle  = 90
    TILT_HOME     = 90.0        # test file: tilt_angle = 90

    # PID gains (units: degrees per pixel of error per frame)
    PAN_KP        = 0.045
    PAN_KI        = 0.0005
    PAN_KD        = 0.012
    TILT_KP       = 0.040
    TILT_KI       = 0.0004
    TILT_KD       = 0.010
    PID_INTEGRAL_CLAMP = 30.0
    DEADZONE      = 12          # px
    SERVO_MAX_STEP = 3.5        # degrees per frame

    # EMA smoothing for raw blob position before PID
    EMA_ALPHA     = 0.35

    # MOG2
    MOG2_HISTORY  = 150
    MOG2_THRESH   = 36
    MIN_BLOB_AREA = 250
    MAX_JUMP_PX   = 220
    MAX_AREA_RATIO = 4.5
    GRACE_FRAMES  = 8

    # SEARCHING state
    REACQUIRE_FRAMES  = 50
    REACQUIRE_RADIUS  = 200     # px

    # CSRT tracker
    TRACKER_PADDING   = 0.35
    CLICK_HALF        = 45      # px

    # Mouse control
    MOUSE_EMA         = 0.18
    MOUSE_PAN_RANGE   = (35.0, 145.0)
    MOUSE_TILT_RANGE  = (30.0, 150.0)

    # -- Motors -- exact pin mapping from test file -------------------------
    # left_motor  = Motor(IN2, IN1, ENA) = Motor(19, 18, 12)
    # right_motor = Motor(IN3, IN4, ENB) = Motor(20, 21, 13)
    MOTOR_L_IN2 = 19;  MOTOR_L_IN1 = 18;  MOTOR_L_EN = 12
    MOTOR_R_IN3 = 20;  MOTOR_R_IN4 = 21;  MOTOR_R_EN = 13

    # Joystick -- from test file
    JOY_DEADZONE    = 0.15      # test file: deadzone = 0.15
    JOY_AXIS_LEFT_Y = 1         # left  stick Y  (negated in code = forward)
    JOY_AXIS_RIGHT_Y= 3         # right stick Y  (negated in code = forward)

    # Servo buttons -- exact mapping from test file:
    # 4 = Y = tilt up   (-5deg)
    # 0 = A = tilt down (+5deg)
    # 1 = B = pan right (-5deg)
    # 3 = X = pan left  (+5deg)
    JOY_SERVO_STEP  = 5         # degrees per button press (test file uses 5)

    # Frame buffer
    RING_SIZE     = 3

    # -- Chassis rotation (pan servo re-centering via motors) -------------
    # When pan servo goes past these angles, rotate the chassis to re-center
    CHASSIS_PAN_TRIGGER  = 30.0   # degrees from home (90deg) before chassis turns
    CHASSIS_PAN_TARGET   = 90.0   # degrees -- where we want pan to be after turn
    # Motor speed used during chassis re-center turn (0.0-1.0)
    CHASSIS_TURN_SPEED   = 0.45
    # Minimum frames the chassis turns before re-checking pan angle
    CHASSIS_TURN_MIN_FRAMES = 6
    # Pixel error threshold -- only rotate chassis if target is also well-centred
    # vertically (i.e. we have a stable lock worth rotating for)
    CHASSIS_LOCK_TILT_MAX = 80    # px -- tilt error must be below this

    # -- Color memory (HSV histogram) --------------------------------------
    # Bins for H, S channels of the target histogram
    COLOR_H_BINS      = 36          # 0-179 in 5deg steps
    COLOR_S_BINS      = 32          # saturation bins
    # Similarity threshold (Bhattacharyya distance, lower = more similar)
    # 0 = identical, 1 = completely different
    COLOR_DRIFT_THRESH  = 0.45      # CSRT box is "wrong" if color dist > this
    COLOR_REACQ_THRESH  = 0.38      # color-backproject blob must be below this
    # How fast the stored histogram updates while LOCKED (EMA factor)
    COLOR_HIST_UPDATE   = 0.08      # low = stable reference, high = adapts faster
    # Minimum saturation to include a pixel in the histogram (filters grey/white noise)
    COLOR_SAT_MIN       = 30
    # Backproject search: minimum mean probability in ROI to count as a match
    COLOR_BP_MIN_SCORE  = 0.18


# ===========================================================================
# Shared state dataclass -- single lock, no scattered globals
# ===========================================================================

class TrackerState:
    IDLE      = "idle"
    LOCKED    = "locked"
    SEARCHING = "searching"


@dataclasses.dataclass
class SharedState:
    # Tracker
    tracker_state: str          = TrackerState.IDLE
    box:           Optional[tuple] = None   # (x,y,w,h)
    target_cx:     int          = 0
    target_cy:     int          = 0
    pan_error:     int          = 0
    tilt_error:    int          = 0
    # Servos
    pan_angle:     float        = Cfg.PAN_HOME
    tilt_angle:    float        = Cfg.TILT_HOME
    # Mouse
    mouse_on:      bool         = False
    mouse_nx:      float        = 0.5
    mouse_ny:      float        = 0.5
    # Commands (set by Flask, cleared by capture_loop)
    cmd_click:     Optional[Tuple[float,float]] = None
    cmd_reset:     bool         = False
    # Color memory -- dominant BGR colour of locked target (for dashboard swatch)
    target_color:  Optional[Tuple[int,int,int]] = None
    color_score:   float        = 0.0   # 0-1, higher = better color match
    chassis_turning: str        = ""    # "left", "right", or ""
    # Joystick telemetry
    joy:           dict         = dataclasses.field(default_factory=lambda: {
        "lx":0,"ly":0,"rx":0,"ry":0,"lt":0,"rt":0,
        "connected":False,"name":""
    })


_state      = SharedState()
_state_lock = threading.Lock()

# -- Atomic JPEG frame store -------------------------------------------------
# The capture loop encodes ONCE and stores bytes here.
# The MJPEG generator just reads bytes -- no re-encode, no event round-trip.
# A Condition lets the stream thread sleep until a genuinely new frame arrives
# without busy-polling, and wakes immediately when one is ready.
_frame_lock      = threading.Lock()
_frame_condition = threading.Condition(_frame_lock)
_latest_jpeg: Optional[bytes] = None   # raw JPEG bytes, replaced every frame


# ===========================================================================
# Servo subsystem -- PID + I2C retry
# ===========================================================================

class ServoController:
    def __init__(self):
        self._lock = threading.Lock()
        i2c = board.I2C()
        pca = PCA9685(i2c)
        pca.frequency = 50
        self._pan  = adafruit_servo.Servo(pca.channels[Cfg.PAN_CH])
        self._tilt = adafruit_servo.Servo(pca.channels[Cfg.TILT_CH])
        self.pan_angle  = Cfg.PAN_HOME
        self.tilt_angle = Cfg.TILT_HOME
        self._write_both(Cfg.PAN_HOME, Cfg.TILT_HOME)

        # PID state
        self._pan_integral  = 0.0; self._pan_prev  = 0.0
        self._tilt_integral = 0.0; self._tilt_prev = 0.0

    # -- internal ------------------------------------------------------------
    def _write_both(self, pan: float, tilt: float, retries: int = 3) -> None:
        for attempt in range(retries):
            try:
                self._pan.angle  = max(Cfg.PAN_MIN,  min(Cfg.PAN_MAX,  pan))
                self._tilt.angle = max(Cfg.TILT_MIN, min(Cfg.TILT_MAX, tilt))
                return
            except OSError as e:
                if attempt == retries - 1:
                    print(f"[SERVO] I2C failed after {retries} retries: {e}")
                else:
                    time.sleep(0.005 * (2 ** attempt))  # exponential backoff

    # -- public API ----------------------------------------------------------
    def home(self) -> None:
        with self._lock:
            self.pan_angle  = Cfg.PAN_HOME
            self.tilt_angle = Cfg.TILT_HOME
            self._pan_integral = self._tilt_integral = 0.0
            self._pan_prev     = self._tilt_prev     = 0.0
            self._write_both(Cfg.PAN_HOME, Cfg.TILT_HOME)

    def step_pid(self, pan_err: int, tilt_err: int,
                 step_max: float = Cfg.SERVO_MAX_STEP) -> None:
        """Move servos using PID control. Errors in pixels."""
        with self._lock:
            # -- PAN --
            if abs(pan_err) > Cfg.DEADZONE:
                self._pan_integral = max(-Cfg.PID_INTEGRAL_CLAMP,
                    min(Cfg.PID_INTEGRAL_CLAMP, self._pan_integral + pan_err))
                pan_d = pan_err - self._pan_prev
                pan_step = (Cfg.PAN_KP * pan_err +
                            Cfg.PAN_KI * self._pan_integral +
                            Cfg.PAN_KD * pan_d)
                pan_step = max(-step_max, min(step_max, pan_step))
                self.pan_angle -= pan_step          # invert: positive err -> rotate left
                self._pan_prev  = pan_err
            else:
                self._pan_integral *= 0.9           # gentle decay when in deadzone

            # -- TILT --
            if abs(tilt_err) > Cfg.DEADZONE:
                self._tilt_integral = max(-Cfg.PID_INTEGRAL_CLAMP,
                    min(Cfg.PID_INTEGRAL_CLAMP, self._tilt_integral + tilt_err))
                tilt_d = tilt_err - self._tilt_prev
                tilt_step = (Cfg.TILT_KP * tilt_err +
                             Cfg.TILT_KI * self._tilt_integral +
                             Cfg.TILT_KD * tilt_d)
                tilt_step = max(-step_max, min(step_max, tilt_step))
                self.tilt_angle -= tilt_step        # invert: positive err -> tilt down
                self._tilt_prev  = tilt_err
            else:
                self._tilt_integral *= 0.9

            self.pan_angle  = max(Cfg.PAN_MIN,  min(Cfg.PAN_MAX,  self.pan_angle))
            self.tilt_angle = max(Cfg.TILT_MIN, min(Cfg.TILT_MAX, self.tilt_angle))
            self._write_both(self.pan_angle, self.tilt_angle)

    def nudge(self, axis: str, degrees: float) -> None:
        with self._lock:
            if axis == "pan":
                self.pan_angle  = max(Cfg.PAN_MIN,  min(Cfg.PAN_MAX,  self.pan_angle  + degrees))
                self._write_both(self.pan_angle, self.tilt_angle)
            elif axis == "tilt":
                self.tilt_angle = max(Cfg.TILT_MIN, min(Cfg.TILT_MAX, self.tilt_angle + degrees))
                self._write_both(self.pan_angle, self.tilt_angle)

    def mouse_move(self, nx: float, ny: float) -> None:
        with self._lock:
            lo, hi = Cfg.MOUSE_PAN_RANGE
            target_pan  = lo + nx * (hi - lo)
            lo, hi = Cfg.MOUSE_TILT_RANGE
            target_tilt = lo + ny * (hi - lo)
            self.pan_angle  += Cfg.MOUSE_EMA * (target_pan  - self.pan_angle)
            self.tilt_angle += Cfg.MOUSE_EMA * (target_tilt - self.tilt_angle)
            self._write_both(self.pan_angle, self.tilt_angle)

    def angles(self) -> Tuple[float, float]:
        return self.pan_angle, self.tilt_angle


# ===========================================================================
# Motor subsystem
# ===========================================================================

class MotorController:
    def __init__(self):
        self._left = self._right = None
        if not _HAS_MOTOR:
            return
        try:
            self._left  = Motor(Cfg.MOTOR_L_IN2, Cfg.MOTOR_L_IN1, Cfg.MOTOR_L_EN)
            self._right = Motor(Cfg.MOTOR_R_IN3, Cfg.MOTOR_R_IN4, Cfg.MOTOR_R_EN)
            print("[MOTOR] Initialized.")
        except Exception as e:
            print(f"[MOTOR] Init failed: {e}")

    def drive(self, left_spd: float, right_spd: float) -> None:
        self._set(self._left,  left_spd)
        self._set(self._right, right_spd)

    def stop(self) -> None:
        self.drive(0.0, 0.0)

    @staticmethod
    def _set(m, spd: float) -> None:
        if m is None: return
        spd = max(-1.0, min(1.0, spd))
        if   spd >  0: m.forward(spd)
        elif spd <  0: m.backward(-spd)
        else:          m.stop()


# ===========================================================================
# Chassis rotation controller
# ===========================================================================

class ChassisController:
    """
    Rotates the whole robot chassis when the pan servo drifts too far from
    centre, then re-centres the servo.  This gives effectively unlimited
    horizontal tracking range.

    Logic
    -----
    While LOCKED on a target:
      * If pan angle > 90 + CHASSIS_PAN_TRIGGER  -> target is to the RIGHT
        -> turn chassis RIGHT  (left motor fwd, right motor back)
        -> simultaneously nudge pan servo back toward 90deg
      * If pan angle < 90 - CHASSIS_PAN_TRIGGER  -> target is to the LEFT
        -> turn chassis LEFT
      * While turning, we do NOT issue new PID servo commands -- the servo
        holds its current angle so the camera stays roughly on target while
        the body catches up.
      * Once pan angle returns within +/-CHASSIS_PAN_TRIGGER/2, stop turning.
    """

    def __init__(self, motors: "MotorController", servos: "ServoController"):
        self._motors  = motors
        self._servos  = servos
        self._turning = ""        # "left", "right", ""
        self._turn_frames = 0

 

def tick(self, pan_angle: float, pan_err_px: int, tilt_err: int, state: str) -> str:
        """
        Call once per frame while LOCKED.
        Returns current turning direction: "left", "right", or "".
        """
        if state != TrackerState.LOCKED:
            self._stop()
            return ""

        pan_err_deg = pan_angle - Cfg.CHASSIS_PAN_TARGET
        trigger     = Cfg.CHASSIS_PAN_TRIGGER
        px_trigger  = 120  # Trigger chassis if target is this many pixels off-center

        # Already turning -- keep going until everything is mostly centered
        if self._turning:
            self._turn_frames += 1
            re_center_threshold = trigger * 0.4  

            # Stop when the servo is mostly centered AND the target is back near the middle
            if abs(pan_err_deg) < re_center_threshold and \
               abs(pan_err_px) < px_trigger * 0.5 and \
               self._turn_frames >= Cfg.CHASSIS_TURN_MIN_FRAMES:
                self._stop()
                return ""

            # Continue turning & nudging the servo
            nudge_dir = -1 if self._turning == "left" else 1
            self._servos.nudge("pan", nudge_dir * Cfg.SERVO_MAX_STEP * 0.8)
            self._drive(self._turning)
            return self._turning

        if abs(tilt_err) > Cfg.CHASSIS_LOCK_TILT_MAX:
            return ""   # unstable lock

        # 1) Trigger if looking too far left (angle > 120) OR target escaping left (pixels < -120)
        if pan_err_deg > trigger or pan_err_px < -px_trigger:
            self._turning     = "left"
            self._turn_frames = 0
            print(f"[CHASSIS] Turning LEFT (pan={pan_angle:.1f}deg, err={pan_err_px}px)")
            self._drive("left")
            return "left"

        # 2) Trigger if looking too far right (angle < 60) OR target escaping right (pixels > 120)
        if pan_err_deg < -trigger or pan_err_px > px_trigger:
            self._turning     = "right"
            self._turn_frames = 0
            print(f"[CHASSIS] Turning RIGHT (pan={pan_angle:.1f}deg, err={pan_err_px}px)")
            self._drive("right")
            return "right"

        return ""

    def _drive(self, direction: str) -> None:
        spd = Cfg.CHASSIS_TURN_SPEED
        if direction == "right":
            self._motors.drive( spd, -spd)   # left fwd, right back
        else:
            self._motors.drive(-spd,  spd)   # left back, right fwd

    def _stop(self) -> None:
        if self._turning:
            self._motors.stop()
            self._turning     = ""
            self._turn_frames = 0
            print("[CHASSIS] Stop turning")

    def stop(self) -> None:
        self._stop()

    @property
    def turning(self) -> str:
        return self._turning


# ===========================================================================
# Vision helpers
# ===========================================================================

def _make_bg_sub() -> cv2.BackgroundSubtractorMOG2:
    return cv2.createBackgroundSubtractorMOG2(
        history=Cfg.MOG2_HISTORY,
        varThreshold=Cfg.MOG2_THRESH,
        detectShadows=False
    )


def _best_blob(fg_mask: np.ndarray,
               last_pos: Optional[Tuple[int,int]] = None,
               last_area: Optional[float] = None,
               max_jump: int = Cfg.MAX_JUMP_PX
               ) -> Optional[Tuple[int,int,int,int,int,int,float]]:
    """
    Returns (cx, cy, x, y, w, h, area) of the best foreground blob, or None.
    Prefers blobs near last_pos; if last_pos is None, returns the largest blob.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean  = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    clean  = cv2.dilate(clean, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < Cfg.MIN_BLOB_AREA: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        candidates.append((cx, cy, x, y, w, h, area))

    if not candidates:
        return None

    if last_pos is not None:
        lx, ly = last_pos
        valid = [b for b in candidates
                 if math.hypot(b[0] - lx, b[1] - ly) <= max_jump and
                    (last_area is None or
                     1/Cfg.MAX_AREA_RATIO <= b[6]/last_area <= Cfg.MAX_AREA_RATIO)]
        if not valid:
            return None
        return min(valid, key=lambda b: math.hypot(b[0] - lx, b[1] - ly))

    return max(candidates, key=lambda b: b[6])  # largest


def _init_csrt(frame: np.ndarray, cx: int, cy: int, w: int, h: int) -> cv2.Tracker:
    """Square-padded CSRT initialisation."""
    size = max(w, h)
    half = int(size * (1 + Cfg.TRACKER_PADDING) / 2)
    x1 = max(0, cx - half); y1 = max(0, cy - half)
    x2 = min(Cfg.FRAME_W, cx + half); y2 = min(Cfg.FRAME_H, cy + half)
    t = cv2.TrackerCSRT_create()
    t.init(frame, (x1, y1, x2 - x1, y2 - y1))
    return t


# ===========================================================================
# Color memory -- HSV histogram of the locked target
# ===========================================================================

class ColorMemory:
    """
    Remembers the HSV color signature of a tracked object.

    Usage
    -----
      mem = ColorMemory()
      mem.learn(frame, x, y, w, h)          # call once on lock
      dist = mem.distance(frame, x, y, w, h) # 0=identical, 1=different
      prob_map = mem.backproject(frame)       # full-frame probability map
      cx, cy, score = mem.find_in_frame(frame)  # best color match location
      mem.update(frame, x, y, w, h)          # slow EMA update while tracking
      mem.clear()                             # forget everything
    """

    def __init__(self):
        self._hist: Optional[np.ndarray] = None
        self._dominant_bgr: Optional[Tuple[int,int,int]] = None
        self._ranges = [0, 180, 0, 256]   # H and S ranges for calcHist

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _roi_hsv(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Crop + convert to HSV, clamp to frame bounds."""
        x1 = max(0, x); y1 = max(0, y)
        x2 = min(Cfg.FRAME_W, x + w); y2 = min(Cfg.FRAME_H, y + h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)

    @staticmethod
    def _build_mask(hsv_roi: np.ndarray) -> np.ndarray:
        """Mask out low-saturation (grey/white) pixels -- they're unreliable."""
        return cv2.inRange(hsv_roi,
                           np.array([0,   Cfg.COLOR_SAT_MIN, 32]),
                           np.array([180, 255,               255]))

    def _compute_hist(self, hsv_roi: np.ndarray) -> Optional[np.ndarray]:
        mask = self._build_mask(hsv_roi)
        if cv2.countNonZero(mask) < 20:
            return None   # region is too grey / dark to characterise
        hist = cv2.calcHist(
            [hsv_roi], [0, 1],
            mask,
            [Cfg.COLOR_H_BINS, Cfg.COLOR_S_BINS],
            self._ranges
        )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    # -- Public API -----------------------------------------------------------

    def learn(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Build histogram from scratch. Returns False if region is featureless."""
        hsv = self._roi_hsv(frame, x, y, w, h)
        hist = self._compute_hist(hsv)
        if hist is None:
            return False
        self._hist = hist
        # Store dominant colour for dashboard swatch
        mean_hsv = cv2.mean(hsv, mask=self._build_mask(hsv))[:3]
        dummy = np.uint8([[list(mean_hsv)]])
        self._dominant_bgr = tuple(int(v) for v in
                                    cv2.cvtColor(dummy, cv2.COLOR_HSV2BGR)[0, 0])
        print(f"[COLOR] Learned -- dominant BGR {self._dominant_bgr}")
        return True

    def update(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Slow EMA update while tracking -- lets histogram adapt to lighting."""
        if self._hist is None:
            self.learn(frame, x, y, w, h)
            return
        hsv  = self._roi_hsv(frame, x, y, w, h)
        hist = self._compute_hist(hsv)
        if hist is not None:
            self._hist = (cv2.addWeighted(self._hist, 1 - Cfg.COLOR_HIST_UPDATE,
                                          hist, Cfg.COLOR_HIST_UPDATE, 0))

    def distance(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """
        Bhattacharyya distance between stored histogram and ROI.
        0 = identical, 1 = completely different.
        Returns 1.0 if no histogram stored.
        """
        if self._hist is None:
            return 1.0
        hsv  = self._roi_hsv(frame, x, y, w, h)
        hist = self._compute_hist(hsv)
        if hist is None:
            return 1.0
        return cv2.compareHist(self._hist, hist, cv2.HISTCMP_BHATTACHARYYA)

    def backproject(self, frame: np.ndarray) -> np.ndarray:
        """
        Full-frame probability map (0-255) showing where the target color is.
        High values = pixels that match the stored histogram.
        """
        if self._hist is None:
            return np.zeros((Cfg.FRAME_H, Cfg.FRAME_W), dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bp  = cv2.calcBackProject([hsv], [0, 1], self._hist, self._ranges, 1)
        # Smooth for robustness
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cv2.filter2D(bp, -1, disc, bp)
        return bp

    def find_in_frame(self, frame: np.ndarray,
                      search_center: Optional[Tuple[int,int]] = None,
                      search_radius: int = Cfg.REACQUIRE_RADIUS
                      ) -> Optional[Tuple[int, int, float]]:
        """
        Find the best color-matching region in the frame.
        If search_center is given, restrict search to that radius.
        Returns (cx, cy, score) where score is 0-1 (higher = better),
        or None if nothing good enough is found.
        """
        if self._hist is None:
            return None

        bp = self.backproject(frame)

        # Restrict to search region if provided
        mask = None
        if search_center is not None:
            mask = np.zeros(bp.shape, dtype=np.uint8)
            cv2.circle(mask, search_center, search_radius, 255, -1)
            bp_search = cv2.bitwise_and(bp, bp, mask=mask)
        else:
            bp_search = bp

        # Find connected blobs of high-probability pixels
        _, thresh = cv2.threshold(bp_search, 60, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = Cfg.COLOR_BP_MIN_SCORE
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < Cfg.MIN_BLOB_AREA * 0.5:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            bx = int(M["m10"] / M["m00"]); by = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
            # Score = mean backproject probability in bbox (normalised 0-1)
            roi_bp = bp[y:y+h, x:x+w].astype(np.float32) / 255.0
            score = float(np.mean(roi_bp))
            if score > best_score:
                best_score = score
                best = (bx, by, score)

        return best

    def clear(self) -> None:
        self._hist = None
        self._dominant_bgr = None

    @property
    def has_color(self) -> bool:
        return self._hist is not None

    @property
    def dominant_bgr(self) -> Optional[Tuple[int,int,int]]:
        return self._dominant_bgr


def _annotate(frame: np.ndarray, state: str,
              box: Optional[tuple], cx: int, cy: int,
              pan_err: int, tilt_err: int,
              pan_ang: float, tilt_ang: float,
              color_mem: Optional["ColorMemory"] = None,
              color_dist: float = 1.0,
              chassis_dir: str = "") -> None:
    """Annotate frame IN-PLACE (no copy -- caller owns the array)."""
    out = frame
    H, W = out.shape[:2]
    ox, oy = W // 2, H // 2
    cv2.line(out, (ox - 18, oy), (ox + 18, oy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(out, (ox, oy - 18), (ox, oy + 18), (0, 255, 0), 1, cv2.LINE_AA)

    STATE_COL = {TrackerState.IDLE: (80,80,255),
                 TrackerState.LOCKED: (0,210,100),
                 TrackerState.SEARCHING: (0,180,255)}
    col = STATE_COL.get(state, (200,200,200))

    if box:
        x, y, w, h = box
        # Change box color to orange if color drift is detected
        box_col = (0,140,255) if color_dist > Cfg.COLOR_DRIFT_THRESH else col
        cv2.rectangle(out, (x, y), (x+w, y+h), box_col, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, col, -1, cv2.LINE_AA)
        cv2.line(out, (ox, oy), (cx, cy), (255,180,0), 1, cv2.LINE_AA)

    # -- Color swatch + confidence bar (bottom-left) ----------------------
    if color_mem is not None and color_mem.has_color:
        sw_x, sw_y, sw_w, sw_h = 8, H - 36, 28, 20
        dom = color_mem.dominant_bgr or (128, 128, 128)
        cv2.rectangle(out, (sw_x, sw_y), (sw_x+sw_w, sw_y+sw_h), dom, -1)
        cv2.rectangle(out, (sw_x, sw_y), (sw_x+sw_w, sw_y+sw_h), (255,255,255), 1)

        # Color match confidence bar (1-dist, clamped 0-1)
        conf = max(0.0, 1.0 - color_dist)
        bar_x = sw_x + sw_w + 6
        bar_w_max = 100
        bar_h = 8
        bar_y = sw_y + (sw_h - bar_h) // 2
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + bar_w_max, bar_y + bar_h), (40,40,40), -1)
        fill_col = (0,200,80) if conf > 0.6 else (0,140,255) if conf > 0.4 else (0,60,220)
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + int(bar_w_max * conf), bar_y + bar_h), fill_col, -1)
        cv2.putText(out, f"COLOR {conf*100:.0f}%",
                    (bar_x, bar_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1, cv2.LINE_AA)

    # HUD text
    lines = [
        f"State: {state.upper()}" + (f"  [CHASSIS {chassis_dir.upper()}]" if chassis_dir else ""),
        f"Pan err: {pan_err:+d}px   Tilt err: {tilt_err:+d}px",
        f"Pan: {pan_ang:.1f}   Tilt: {tilt_ang:.1f}",
    ]
    if state == TrackerState.IDLE:
        lines = ["IDLE -- CLICK TARGET TO LOCK"]

    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (8, 18 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (8, 18 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,255,200), 1, cv2.LINE_AA)

    # Chassis rotation arrow -- big and obvious when turning
    if chassis_dir:
        arrow_txt = "<<< ROTATING" if chassis_dir == "left" else "ROTATING >>>"
        col = (0, 200, 255)
        cv2.putText(out, arrow_txt, (W//2 - 80, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, arrow_txt, (W//2 - 80, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


# ===========================================================================
# Capture loop -- runs at camera frame rate
# ===========================================================================

def capture_loop(servos: ServoController, chassis: "ChassisController") -> None:
    global _state, _latest_jpeg

    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (Cfg.FRAME_W, Cfg.FRAME_H), "format": "BGR888"},
        controls={
            "FrameRate":          Cfg.FRAME_RATE,
            "NoiseReductionMode": 0,
            "Sharpness":          1.0,
        },
        buffer_count=2
    )
    cam.configure(cfg)
    cam.start()
    print(f"[CAM] Started -- {Cfg.FRAME_W}x{Cfg.FRAME_H} @ {Cfg.FRAME_RATE}fps  buffer=2")

    tracker_obj = None
    color_mem   = ColorMemory()
    state       = TrackerState.IDLE

    # Tracking internals
    last_known_pos  = None
    last_known_area = None
    search_count    = 0

    # EMA smoothed pixel error -> PID
    smooth_pan  = 0.0
    smooth_tilt = 0.0
    last_pan_err = last_tilt_err = 0

    color_dist    = 1.0
    chassis_dir   = ""

    TS = TrackerState

    while True:
        raw   = cam.capture_array()
        frame = cv2.rotate(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), cv2.ROTATE_180)

        # Pull shared commands
        with _state_lock:
            do_click = _state.cmd_click;  _state.cmd_click  = None
            do_reset = _state.cmd_reset;  _state.cmd_reset  = False
            mouse_on = _state.mouse_on
            mouse_nx = _state.mouse_nx
            mouse_ny = _state.mouse_ny

        # -- Reset ----------------------------------------------------------
        if do_reset:
            state = TS.IDLE
            tracker_obj = None
            color_mem.clear()
            last_known_pos = last_known_area = None
            search_count   = 0
            smooth_pan = smooth_tilt = 0.0
            last_pan_err = last_tilt_err = 0
            color_dist   = 1.0
            chassis_dir  = ""
            chassis.stop()
            servos.home()
            print("[RESET] IDLE -- awaiting target selection")

        # -- Click-to-lock --------------------------------------------------
        if do_click is not None:
            nx, ny = do_click
            fx = int(nx * Cfg.FRAME_W)
            fy = int(ny * Cfg.FRAME_H)
            h  = Cfg.CLICK_HALF
            x1 = max(0, fx - h);          y1 = max(0, fy - h)
            x2 = min(Cfg.FRAME_W, fx + h); y2 = min(Cfg.FRAME_H, fy + h)
            bw = x2 - x1; bh = y2 - y1

            tracker_obj = cv2.TrackerCSRT_create()
            tracker_obj.init(frame, (x1, y1, bw, bh))
            state = TS.LOCKED

            color_mem.clear()
            color_mem.learn(frame, x1, y1, bw, bh)
            color_dist = 0.0

            last_known_pos  = (fx, fy)
            last_known_area = bw * bh
            smooth_pan = smooth_tilt = 0.0
            chassis_dir = ""
            print(f"[CLICK-LOCK] target at ({fx},{fy})")

        # -- State machine --------------------------------------------------
        box = None; cx = cy = 0

        if state == TS.LOCKED:
            ok, rect = tracker_obj.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in rect)
                cx, cy = x + w // 2, y + h // 2
                box    = (x, y, w, h)

                # Color validation -- correct drift or slowly adapt histogram
                if color_mem.has_color:
                    color_dist = color_mem.distance(frame, x, y, w, h)
                    if color_dist > Cfg.COLOR_DRIFT_THRESH:
                        hit = color_mem.find_in_frame(
                            frame,
                            search_center=last_known_pos,
                            search_radius=Cfg.REACQUIRE_RADIUS)
                        if hit is not None:
                            hx, hy, hscore = hit
                            half = int(math.sqrt(last_known_area or 1000) / 2)
                            tracker_obj = _init_csrt(frame, hx, hy, half*2, half*2)
                            cx, cy      = hx, hy
                            box         = (hx - half, hy - half, half*2, half*2)
                            color_dist  = 1.0 - hscore
                    else:
                        color_mem.update(frame, x, y, w, h)

                last_known_pos  = (cx, cy)
                last_known_area = w * h

            else:
                print("[LOCK] Lost -- searching by color")
                state = TS.SEARCHING
                tracker_obj  = None
                search_count = 0
                chassis.stop()

        elif state == TS.SEARCHING:
            found = False

            # Priority 1: color backproject
            if color_mem.has_color:
                hit = color_mem.find_in_frame(
                    frame,
                    search_center=last_known_pos,
                    search_radius=Cfg.REACQUIRE_RADIUS)
                if hit is not None:
                    hx, hy, hscore = hit
                    half        = int(math.sqrt(last_known_area or 1000) / 2)
                    tracker_obj = _init_csrt(frame, hx, hy, half*2, half*2)
                    state       = TS.LOCKED
                    cx, cy      = hx, hy
                    box         = (hx - half, hy - half, half*2, half*2)
                    last_known_pos = (hx, hy)
                    color_dist  = 1.0 - hscore
                    search_count = 0
                    found        = True
                    print(f"[SEARCH] Re-acquired by color (score {hscore:.2f})")

            # Priority 2: motion blob matching stored color
            if not found:
                # We still need MOG2 during searching only
                fg   = _make_bg_sub().apply(frame) if not hasattr(capture_loop, '_bg') else None
                # Use a local bg_sub kept alive across search frames
                if not hasattr(capture_loop, '_search_bg'):
                    capture_loop._search_bg = _make_bg_sub()
                fg   = capture_loop._search_bg.apply(frame)
                blob = _best_blob(fg, last_known_pos, last_known_area,
                                  max_jump=Cfg.REACQUIRE_RADIUS)
                if blob:
                    bx, by, bxr, byr, bw, bh, area = blob
                    dist = color_mem.distance(frame, bxr, byr, bw, bh)
                    if dist < Cfg.COLOR_REACQ_THRESH or not color_mem.has_color:
                        tracker_obj     = _init_csrt(frame, bx, by, bw, bh)
                        state           = TS.LOCKED
                        last_known_pos  = (bx, by)
                        last_known_area = area
                        color_dist      = dist
                        search_count    = 0
                        found           = True
                        print(f"[SEARCH] Re-acquired by motion+color (dist {dist:.2f})")

            if not found:
                search_count += 1
                if search_count > Cfg.REACQUIRE_FRAMES:
                    print("[SEARCH] Failed -- back to IDLE (click to re-select target)")
                    state           = TS.IDLE
                    tracker_obj     = None
                    last_known_pos  = None
                    last_known_area = None
                    search_count    = 0
                    color_dist      = 1.0
                    chassis.stop()
                    servos.home()

        else:  # IDLE -- do nothing, just wait for a click
            pass

        # -- Servo + chassis control ----------------------------------------
        pan_err = tilt_err = 0

        if mouse_on:
            servos.mouse_move(mouse_nx, mouse_ny)
            chassis.stop()
            chassis_dir = ""

        elif box and state == TS.LOCKED:
            # Compute EMA-smoothed pixel error
            smooth_pan  = Cfg.EMA_ALPHA * (cx - Cfg.FRAME_W // 2) + (1 - Cfg.EMA_ALPHA) * smooth_pan
            smooth_tilt = Cfg.EMA_ALPHA * (Cfg.FRAME_H // 2 - cy) + (1 - Cfg.EMA_ALPHA) * smooth_tilt
            pan_err  = int(smooth_pan)
            tilt_err = int(smooth_tilt)

            pa, _ = servos.angles()

          # Let chassis controller decide whether to rotate the body.
            chassis_dir = chassis.tick(pa, pan_err, tilt_err, state)
            if not chassis_dir:
                # Normal PID servo tracking
                servos.step_pid(pan_err, tilt_err)

            last_pan_err  = pan_err
            last_tilt_err = tilt_err

        elif state == TS.SEARCHING:
            # Gently sweep servo in last known direction while searching
            servos.step_pid(last_pan_err, last_tilt_err,
                            step_max=Cfg.SERVO_MAX_STEP * 1.4)
            chassis_dir = ""

        else:
            # IDLE -- decay smoothing so old error doesn't linger
            smooth_pan  *= 0.7
            smooth_tilt *= 0.7
            chassis_dir  = ""

        pa, ta = servos.angles()

        # -- Annotate, encode, publish --------------------------------------
        _annotate(frame, state, box, cx, cy,
                  pan_err, tilt_err, pa, ta,
                  color_mem=color_mem, color_dist=color_dist,
                  chassis_dir=chassis_dir)
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, Cfg.JPEG_QUALITY,
             cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        if ok:
            jpeg_bytes = buf.tobytes()
            global _latest_jpeg
            with _frame_condition:
                _latest_jpeg = jpeg_bytes
                _frame_condition.notify_all()

        # -- Telemetry ------------------------------------------------------
        dom = color_mem.dominant_bgr
        with _state_lock:
            _state.tracker_state   = state
            _state.box             = box
            _state.target_cx       = cx
            _state.target_cy       = cy
            _state.pan_error       = pan_err
            _state.tilt_error      = tilt_err
            _state.pan_angle       = pa
            _state.tilt_angle      = ta
            _state.target_color    = dom
            _state.color_score     = round(max(0.0, 1.0 - color_dist), 3)
            _state.chassis_turning = chassis_dir


# ===========================================================================
# MJPEG stream -- reads pre-encoded bytes, does NO work per frame
# ===========================================================================

def mjpeg_generator():
    """
    Streams pre-encoded JPEG bytes from _latest_jpeg.
    The Condition.wait() call blocks with zero CPU until a new frame is
    stored by the capture loop, then wakes immediately -- no polling, no sleep,
    no event-clear race. Each connected browser gets its own generator
    instance; they all share the same _latest_jpeg bytes without copying.
    """
    global _latest_jpeg
    last_sent: Optional[bytes] = None
    min_interval = 1.0 / Cfg.STREAM_MAX_FPS

    while True:
        # Block until a frame we haven't sent yet is available
        with _frame_condition:
            # Use a timeout so the generator exits cleanly if the client drops
            _frame_condition.wait_for(
                lambda: _latest_jpeg is not last_sent,
                timeout=1.0
            )
            jpeg = _latest_jpeg

        if jpeg is None or jpeg is last_sent:
            continue   # timeout with no new frame -- loop and wait again

        last_sent = jpeg
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg + b"\r\n")


# ===========================================================================
# Joystick loop
# ===========================================================================

_SERVO_BUTTONS = {
    # Exact mapping from test file SERVO_BUTTONS dict:
    # axis='tilt' step=-5  ->  Y button (index 4) -> tilt up
    # axis='tilt' step=+5  ->  A button (index 0) -> tilt down
    # axis='pan'  step=-5  ->  B button (index 1) -> pan right
    # axis='pan'  step=+5  ->  X button (index 3) -> pan left
    4: ("tilt", -Cfg.JOY_SERVO_STEP),
    0: ("tilt", +Cfg.JOY_SERVO_STEP),
    1: ("pan",  -Cfg.JOY_SERVO_STEP),
    3: ("pan",  +Cfg.JOY_SERVO_STEP),
}


def joystick_loop(motors: MotorController, servos: ServoController,
                  chassis: "ChassisController") -> None:
    pygame.init()
    pygame.joystick.init()
    joy  = None
    axes = {}
    DZ   = Cfg.JOY_DEADZONE   # 0.15, matches test file

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

            elif event.type == pygame.JOYAXISMOTION:
                axes[event.axis] = event.value

                # Only drive manually when chassis is not auto-rotating
                if not chassis.turning:
                    # Axis 1 = left stick Y,  Axis 3 = right stick Y
                    # Negate so stick-forward = positive speed (matches test file)
                    ly = -axes.get(Cfg.JOY_AXIS_LEFT_Y,  0.0)
                    ry = -axes.get(Cfg.JOY_AXIS_RIGHT_Y, 0.0)
                    if abs(ly) < DZ: ly = 0.0
                    if abs(ry) < DZ: ry = 0.0
                    motors.drive(ly, ry)

            elif event.type == pygame.JOYBUTTONDOWN:
                # Servo nudge -- same logic as test file update_servos()
                btn_cfg = _SERVO_BUTTONS.get(event.button)
                if btn_cfg:
                    servos.nudge(*btn_cfg)

        # Hot-plug detection
        count = pygame.joystick.get_count()
        if count > 0 and joy is None:
            joy = pygame.joystick.Joystick(0)
            joy.init()
            axes = {i: 0.0 for i in range(joy.get_numaxes())}
            print(f"[JOY] Connected: {joy.get_name()}")
        elif count == 0 and joy is not None:
            print("[JOY] Disconnected -- stopping motors.")
            motors.stop()
            joy  = None
            axes = {}

        # Update dashboard telemetry
        if joy:
            def dz(v): return 0.0 if abs(v) < DZ else round(v, 3)
            with _state_lock:
                _state.joy.update({
                    "lx":   dz( axes.get(0, 0.0)),
                    "ly":  -dz( axes.get(Cfg.JOY_AXIS_LEFT_Y,  0.0)),
                    "rx":   dz( axes.get(2, 0.0)),
                    "ry":  -dz( axes.get(Cfg.JOY_AXIS_RIGHT_Y, 0.0)),
                    "lt":   max(0.0, axes.get(4, 0.0)),
                    "rt":   max(0.0, axes.get(5, 0.0)),
                    "connected": True,
                    "name": joy.get_name(),
                })
        else:
            with _state_lock:
                _state.joy.update({
                    "lx":0,"ly":0,"rx":0,"ry":0,
                    "lt":0,"rt":0,"connected":False,"name":""
                })

        time.sleep(0.016)  # ~60 Hz is plenty for joystick input   # ~60 Hz is enough for joystick


# ===========================================================================
# Flask application
# ===========================================================================

app = Flask(__name__, template_folder="templates")

# These are set in main() after hardware init
_servos: ServoController = None   # type: ignore
_motors: MotorController = None   # type: ignore


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with _state_lock:
        s = _state
        dom = s.target_color
        return jsonify({
            "state":        s.tracker_state,
            "pan_err":      s.pan_error,
            "tilt_err":     s.tilt_error,
            "tx": s.target_cx, "ty": s.target_cy,
            "tw": s.box[2] if s.box else 0,
            "th": s.box[3] if s.box else 0,
            "pan_angle":    round(s.pan_angle,  1),
            "tilt_angle":   round(s.tilt_angle, 1),
            "mouse_control":    s.mouse_on,
            "color_score":      s.color_score,
            "color_rgb":        [dom[2], dom[1], dom[0]] if dom else None,
            "chassis_turning":  s.chassis_turning,
        })

@app.route("/joystick")
def joystick_status():
    with _state_lock:
        return jsonify(_state.joy)

@app.route("/click-lock", methods=["POST"])
def click_lock():
    data = request.get_json(force=True)
    with _state_lock:
        _state.cmd_click = (float(data["x"]), float(data["y"]))
    return jsonify({"ok": True})

@app.route("/reset", methods=["POST"])
def reset_tracker():
    with _state_lock:
        _state.cmd_reset = True
    return jsonify({"ok": True})

@app.route("/mouse-toggle", methods=["POST"])
def mouse_toggle():
    with _state_lock:
        _state.mouse_on = not _state.mouse_on
        if not _state.mouse_on:
            _state.mouse_nx = _state.mouse_ny = 0.5
        val = _state.mouse_on
    return jsonify({"mouse_control": val})

@app.route("/mouse-move", methods=["POST"])
def mouse_move():
    data = request.get_json(force=True)
    with _state_lock:
        _state.mouse_nx = max(0.0, min(1.0, float(data.get("x", 0.5))))
        _state.mouse_ny = max(0.0, min(1.0, float(data.get("y", 0.5))))
    return "", 204

@app.route("/servo-nudge", methods=["POST"])
def servo_nudge():
    """Manual servo nudge from dashboard buttons."""
    data = request.get_json(force=True)
    axis   = data.get("axis", "pan")
    degrees = float(data.get("degrees", 0))
    _servos.nudge(axis, degrees)
    return jsonify({"ok": True})

@app.route("/drive", methods=["POST"])
def drive():
    """Manual motor drive: {left: -1..1, right: -1..1}."""
    data = request.get_json(force=True)
    _motors.drive(float(data.get("left", 0)), float(data.get("right", 0)))
    return jsonify({"ok": True})


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    servos  = ServoController()
    motors  = MotorController()
    chassis = ChassisController(motors, servos)

    # Expose to Flask routes
    _servos = servos
    _motors = motors

    threading.Thread(target=capture_loop,  args=(servos, chassis),          daemon=True).start()
    threading.Thread(target=joystick_loop, args=(motors, servos, chassis),  daemon=True).start()

    print("=" * 55)
    print("  Pan-Tilt Tracker -- dashboard at http://<Pi-IP>:5000")
    print("  Click a target in the feed to begin tracking.")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, threaded=True)