"""
Vision subsystem: KCF object tracking + MOG2 background subtraction
                  + HSV color memory for re-acquisition.

Ported from tracker.py (Raspberry Pi / Picamera2) — now hardware-agnostic.
Call VisionTracker.update(frame) once per frame from any capture source.
"""
from __future__ import annotations

import math
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from config import Cfg

_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ── Color memory ──────────────────────────────────────────────────────────────

class ColorMemory:
    def __init__(self):
        self._hist: Optional[np.ndarray] = None
        self._dominant_bgr: Optional[Tuple[int, int, int]] = None
        self._ranges = [0, 180, 0, 256]

    @staticmethod
    def _roi_hsv(frame, x, y, w, h) -> np.ndarray:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(Cfg.FRAME_W, x + w), min(Cfg.FRAME_H, y + h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)

    @staticmethod
    def _mask(hsv) -> np.ndarray:
        return cv2.inRange(hsv,
                           np.array([0, Cfg.COLOR_SAT_MIN, 32]),
                           np.array([180, 255, 255]))

    def _hist_from_roi(self, frame, x, y, w, h) -> Optional[np.ndarray]:
        hsv = self._roi_hsv(frame, x, y, w, h)
        mask = self._mask(hsv)
        if cv2.countNonZero(mask) < 20:
            return None
        hist = cv2.calcHist([hsv], [0, 1], mask,
                             [Cfg.COLOR_H_BINS, Cfg.COLOR_S_BINS], self._ranges)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def learn(self, frame, x, y, w, h) -> bool:
        hist = self._hist_from_roi(frame, x, y, w, h)
        if hist is None:
            return False
        self._hist = hist
        hsv = self._roi_hsv(frame, x, y, w, h)
        mean_hsv = cv2.mean(hsv, mask=self._mask(hsv))[:3]
        dummy = np.uint8([[list(mean_hsv)]])
        self._dominant_bgr = tuple(
            int(v) for v in cv2.cvtColor(dummy, cv2.COLOR_HSV2BGR)[0, 0]
        )
        return True

    def update(self, frame, x, y, w, h) -> None:
        if self._hist is None:
            self.learn(frame, x, y, w, h)
            return
        hist = self._hist_from_roi(frame, x, y, w, h)
        if hist is not None:
            self._hist = cv2.addWeighted(
                self._hist, 1 - Cfg.COLOR_HIST_UPDATE, hist, Cfg.COLOR_HIST_UPDATE, 0
            )

    def distance(self, frame, x, y, w, h) -> float:
        if self._hist is None:
            return 1.0
        hist = self._hist_from_roi(frame, x, y, w, h)
        if hist is None:
            return 1.0
        return cv2.compareHist(self._hist, hist, cv2.HISTCMP_BHATTACHARYYA)

    def backproject(self, frame) -> np.ndarray:
        if self._hist is None:
            return np.zeros((Cfg.FRAME_H, Cfg.FRAME_W), dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0, 1], self._hist, self._ranges, 1)
        cv2.filter2D(bp, -1, _K5, bp)
        return bp

    def find_in_frame(self, frame, search_center=None,
                      search_radius=Cfg.REACQUIRE_RADIUS):
        if self._hist is None:
            return None
        bp = self.backproject(frame)
        if search_center is not None:
            mask = np.zeros(bp.shape, dtype=np.uint8)
            cv2.circle(mask, search_center, search_radius, 255, -1)
            bp = cv2.bitwise_and(bp, bp, mask=mask)

        _, thresh = cv2.threshold(bp, 60, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, _K7)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        best_score, best = Cfg.COLOR_BP_MIN_SCORE, None
        for c in contours:
            area = cv2.contourArea(c)
            if area < Cfg.MIN_BLOB_AREA * 0.5:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            bx, by = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(c)
            score = float(np.mean(bp[y:y + h, x:x + w].astype(np.float32) / 255.0))
            if score > best_score:
                best_score, best = score, (bx, by, score)
        return best

    def clear(self) -> None:
        self._hist = None
        self._dominant_bgr = None

    @property
    def has_color(self) -> bool:
        return self._hist is not None

    @property
    def dominant_bgr(self):
        return self._dominant_bgr


# ── helpers ───────────────────────────────────────────────────────────────────

def _best_blob(fg, last_pos=None, last_area=None, max_jump=Cfg.MAX_JUMP_PX):
    clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _K5, iterations=1)
    clean = cv2.dilate(clean, _K5, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < Cfg.MIN_BLOB_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        candidates.append((cx, cy, x, y, w, h, area))
    if not candidates:
        return None
    if last_pos is not None:
        lx, ly = last_pos
        valid = [
            b for b in candidates
            if math.hypot(b[0] - lx, b[1] - ly) <= max_jump
            and (last_area is None
                 or 1 / Cfg.MAX_AREA_RATIO <= b[6] / last_area <= Cfg.MAX_AREA_RATIO)
        ]
        if not valid:
            return None
        return min(valid, key=lambda b: math.hypot(b[0] - lx, b[1] - ly))
    return max(candidates, key=lambda b: b[6])


def _init_kcf(frame, cx, cy, w, h) -> cv2.Tracker:
    size = max(w, h)
    half = int(size * (1 + Cfg.TRACKER_PADDING) / 2)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(Cfg.FRAME_W, cx + half)
    y2 = min(Cfg.FRAME_H, cy + half)
    t = cv2.TrackerKCF_create()
    t.init(frame, (x1, y1, x2 - x1, y2 - y1))
    return t


# ── tracker state machine ─────────────────────────────────────────────────────

class VisionTracker:
    IDLE = "idle"
    LOCKED = "locked"
    SEARCHING = "searching"

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_internal()
        self._frame_n = 0
        self._bg_sub = self._make_bg_sub()

        # Commands injected from the server thread
        self._cmd_click: Optional[Tuple[float, float]] = None
        self._cmd_reset = False

        # Public output — read by server, protected by _lock
        self.state = self.IDLE
        self.box: Optional[tuple] = None
        self.target_cx = 0
        self.target_cy = 0
        self.pan_err = 0
        self.tilt_err = 0
        self.color_score = 0.0
        self.target_color: Optional[Tuple[int, int, int]] = None

    # ── commands (called from server / WS handler thread) ─────────────────────

    def cmd_click(self, nx: float, ny: float) -> None:
        with self._lock:
            self._cmd_click = (nx, ny)

    def cmd_reset(self) -> None:
        with self._lock:
            self._cmd_reset = True

    # ── per-frame update (called from vision thread) ───────────────────────────

    def update(self, frame: np.ndarray) -> None:
        self._frame_n += 1

        with self._lock:
            do_click = self._cmd_click
            self._cmd_click = None
            do_reset = self._cmd_reset
            self._cmd_reset = False

        if do_reset:
            self._reset_internal()

        if do_click is not None:
            self._do_click(frame, *do_click)

        box, cx, cy, color_dist = self._step(frame)

        pan_err = cx - Cfg.FRAME_W // 2 if box else 0
        tilt_err = Cfg.FRAME_H // 2 - cy if box else 0

        with self._lock:
            self.state = self._state
            self.box = box
            self.target_cx = cx
            self.target_cy = cy
            self.pan_err = pan_err
            self.tilt_err = tilt_err
            self.color_score = round(max(0.0, 1.0 - color_dist), 3)
            self.target_color = self._color_mem.dominant_bgr

    # ── snapshot for server ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "box": self.box,
                "target_cx": self.target_cx,
                "target_cy": self.target_cy,
                "pan_err": self.pan_err,
                "tilt_err": self.tilt_err,
                "color_score": self.color_score,
                "target_color": self.target_color,
            }

    # ── internals ─────────────────────────────────────────────────────────────

    def _reset_internal(self):
        self._state = self.IDLE
        self._tracker_obj = None
        self._color_mem = ColorMemory()
        self._last_pos = None
        self._last_area = None
        self._search_count = 0
        self._color_dist = 1.0

    @staticmethod
    def _make_bg_sub():
        return cv2.createBackgroundSubtractorMOG2(
            history=Cfg.MOG2_HISTORY, varThreshold=Cfg.MOG2_THRESH,
            detectShadows=False
        )

    def _do_click(self, frame, nx, ny):
        fx = int(nx * Cfg.FRAME_W)
        fy = int(ny * Cfg.FRAME_H)
        h = Cfg.CLICK_HALF
        x1, y1 = max(0, fx - h), max(0, fy - h)
        x2, y2 = min(Cfg.FRAME_W, fx + h), min(Cfg.FRAME_H, fy + h)
        bw, bh = x2 - x1, y2 - y1
        self._tracker_obj = cv2.TrackerKCF_create()
        self._tracker_obj.init(frame, (x1, y1, bw, bh))
        self._state = self.LOCKED
        self._color_mem.clear()
        self._color_mem.learn(frame, x1, y1, bw, bh)
        self._color_dist = 0.0
        self._last_pos = (fx, fy)
        self._last_area = bw * bh
        self._search_count = 0

    def _step(self, frame) -> tuple:
        box = None
        cx = cy = 0
        color_dist = self._color_dist

        if self._state == self.LOCKED:
            ok, rect = self._tracker_obj.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in rect)
                cx, cy = x + w // 2, y + h // 2
                box = (x, y, w, h)
                if self._color_mem.has_color and self._frame_n % 2 == 0:
                    color_dist = self._color_mem.distance(frame, x, y, w, h)
                    self._color_dist = color_dist
                    if color_dist > Cfg.COLOR_DRIFT_THRESH:
                        hit = self._color_mem.find_in_frame(
                            frame, self._last_pos, Cfg.REACQUIRE_RADIUS)
                        if hit is not None:
                            hx, hy, hs = hit
                            half = int(math.sqrt(self._last_area or 1000) / 2)
                            self._tracker_obj = _init_kcf(frame, hx, hy, half*2, half*2)
                            cx, cy = hx, hy
                            box = (hx - half, hy - half, half * 2, half * 2)
                            self._color_dist = 1.0 - hs
                    else:
                        self._color_mem.update(frame, x, y, w, h)
                self._last_pos = (cx, cy)
                self._last_area = w * h
            else:
                self._state = self.SEARCHING
                self._tracker_obj = None
                self._search_count = 0

        elif self._state == self.SEARCHING:
            found = False
            if self._color_mem.has_color:
                hit = self._color_mem.find_in_frame(
                    frame, self._last_pos, Cfg.REACQUIRE_RADIUS)
                if hit is not None:
                    hx, hy, hs = hit
                    half = int(math.sqrt(self._last_area or 1000) / 2)
                    self._tracker_obj = _init_kcf(frame, hx, hy, half*2, half*2)
                    self._state = self.LOCKED
                    cx, cy = hx, hy
                    box = (hx - half, hy - half, half * 2, half * 2)
                    self._last_pos = (hx, hy)
                    self._color_dist = 1.0 - hs
                    color_dist = self._color_dist
                    self._search_count = 0
                    found = True

            if not found:
                fg = self._bg_sub.apply(frame)
                blob = _best_blob(fg, self._last_pos, self._last_area,
                                  Cfg.REACQUIRE_RADIUS)
                if blob:
                    bx, by, bxr, byr, bw, bh, area = blob
                    dist = self._color_mem.distance(frame, bxr, byr, bw, bh)
                    if dist < Cfg.COLOR_REACQ_THRESH or not self._color_mem.has_color:
                        self._tracker_obj = _init_kcf(frame, bx, by, bw, bh)
                        self._state = self.LOCKED
                        self._last_pos = (bx, by)
                        self._last_area = area
                        self._color_dist = dist
                        color_dist = dist
                        self._search_count = 0
                        found = True

            if not found:
                self._search_count += 1
                if self._search_count > Cfg.REACQUIRE_FRAMES:
                    self._reset_internal()

        return box, cx, cy, color_dist


# ── frame annotation ──────────────────────────────────────────────────────────

_STATE_COLORS = {
    VisionTracker.IDLE:      (80,  80, 255),
    VisionTracker.LOCKED:    (0,  210, 100),
    VisionTracker.SEARCHING: (0,  180, 255),
}


def annotate(frame: np.ndarray, snap: dict,
             pan_ang: float, tilt_ang: float,
             chassis_dir: str = "") -> None:
    state = snap["state"]
    box = snap["box"]
    cx, cy = snap["target_cx"], snap["target_cy"]
    pan_err, tilt_err = snap["pan_err"], snap["tilt_err"]
    color_dist = 1.0 - snap["color_score"]
    target_color = snap["target_color"]

    H, W = frame.shape[:2]
    ox, oy = W // 2, H // 2
    col = _STATE_COLORS.get(state, (200, 200, 200))

    cv2.line(frame, (ox - 18, oy), (ox + 18, oy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (ox, oy - 18), (ox, oy + 18), (0, 255, 0), 1, cv2.LINE_AA)

    if box:
        x, y, w, h = box
        bc = (0, 140, 255) if color_dist > Cfg.COLOR_DRIFT_THRESH else col
        cv2.rectangle(frame, (x, y), (x + w, y + h), bc, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, col, -1, cv2.LINE_AA)
        cv2.line(frame, (ox, oy), (cx, cy), (255, 180, 0), 1, cv2.LINE_AA)

    if target_color is not None:
        sw_x, sw_y, sw_w, sw_h = 8, H - 36, 28, 20
        cv2.rectangle(frame, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h),
                      target_color, -1)
        cv2.rectangle(frame, (sw_x, sw_y), (sw_x + sw_w, sw_y + sw_h),
                      (255, 255, 255), 1)
        conf = max(0.0, 1.0 - color_dist)
        bar_x = sw_x + sw_w + 6
        bar_w_max = 100
        bar_y = sw_y + 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 8),
                      (40, 40, 40), -1)
        fill = ((0, 200, 80) if conf > 0.6
                else (0, 140, 255) if conf > 0.4 else (0, 60, 220))
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w_max * conf), bar_y + 8), fill, -1)

    lines = (["IDLE — CLICK TARGET TO LOCK"] if state == VisionTracker.IDLE else [
        f"State: {state.upper()}" + (f"  [{chassis_dir.upper()}]" if chassis_dir else ""),
        f"Err  pan:{pan_err:+d}px  tilt:{tilt_err:+d}px",
        f"Ang  pan:{pan_ang:.1f}°  tilt:{tilt_ang:.1f}°",
    ])
    for i, txt in enumerate(lines):
        y_pos = 18 + i * 18
        cv2.putText(frame, txt, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 255, 200), 1, cv2.LINE_AA)

    if chassis_dir:
        txt = "<<< ROTATING" if chassis_dir == "left" else "ROTATING >>>"
        cv2.putText(frame, txt, (W // 2 - 80, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (W // 2 - 80, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
