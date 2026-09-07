"""
Dual-camera subsystem.

PC prototype  : OpenCV VideoCapture with automatic test-pattern fallback.
Jetson (cam0) : replace VideoCapture with a GStreamer pipeline string, e.g.
                  "nvarguscamerasrc sensor-id=0 ! ... ! appsink"
Jetson (cam1) : thermal USB cam → VideoCapture(1), or FLIR Lepton via SPI.
"""
import queue
import threading
import time

import cv2
import numpy as np

from config import Cfg


class DualCamera:
    def __init__(self):
        # cam0: RGB / IMX678 — used by vision tracker + annotation
        # cam1: thermal — display-only
        self.cam0_q: queue.Queue = queue.Queue(maxsize=2)
        self.cam1_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop_ev = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run_cam0, daemon=True, name="cam0").start()
        threading.Thread(target=self._run_cam1, daemon=True, name="cam1").start()
        print("[CAM] Dual camera threads started")

    def stop(self) -> None:
        self._stop_ev.set()

    # ── cam0: RGB ─────────────────────────────────────────────────────────────

    def _run_cam0(self) -> None:
        cap = self._open(Cfg.CAM0_INDEX, "CAM0-RGB")
        interval = 1.0 / Cfg.FRAME_RATE
        while not self._stop_ev.is_set():
            t0 = time.monotonic()
            frame = self._read(cap, 0)
            self._push(self.cam0_q, frame)
            self._sleep(t0, interval)
        if cap:
            cap.release()

    # ── cam1: thermal ─────────────────────────────────────────────────────────

    def _run_cam1(self) -> None:
        # IR sensors (e.g. Windows Hello depth cam on video2) reject cap.set() calls,
        # so open raw without forcing resolution or FPS.
        cap = None
        label = "CAM1-IR"
        if Cfg.CAM1_INDEX >= 0:
            cap = cv2.VideoCapture(Cfg.CAM1_INDEX, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"[{label}] Opened at index {Cfg.CAM1_INDEX} (raw mode)")
            else:
                cap.release()
                cap = None
                print(f"[{label}] Not found — animated test pattern active")

        interval = 1.0 / Cfg.FRAME_RATE
        while not self._stop_ev.is_set():
            t0 = time.monotonic()
            if cap is not None:
                ret, raw = cap.read()
                if ret:
                    # IR sensor returns single-channel grayscale
                    gray = raw if raw.ndim == 2 else (
                        raw[:, :, 0] if raw.shape[2] == 1
                        else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                    )
                    gray = cv2.resize(gray, (Cfg.FRAME_W, Cfg.FRAME_H))
                    frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                else:
                    frame = self._test_pattern(1)
            else:
                frame = self._fake_thermal()
            self._push(self.cam1_q, frame)
            self._sleep(t0, interval)
        if cap:
            cap.release()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _open(index: int, label: str):
        if index < 0:
            print(f"[{label}] Disabled (index={index}) — test pattern active")
            return None
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Cfg.FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Cfg.FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, Cfg.FRAME_RATE)
            print(f"[{label}] Opened at index {index}")
            return cap
        cap.release()
        print(f"[{label}] Not found — test pattern active")
        return None

    @staticmethod
    def _read(cap, cam_id: int) -> np.ndarray:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                return cv2.resize(frame, (Cfg.FRAME_W, Cfg.FRAME_H))
        return DualCamera._test_pattern(cam_id)

    @staticmethod
    def _test_pattern(cam_id: int) -> np.ndarray:
        h, w = Cfg.FRAME_H, Cfg.FRAME_W
        frame = np.full((h, w, 3),
                        (30, 20, 20) if cam_id == 0 else (20, 20, 40),
                        dtype=np.uint8)
        for x in range(0, w, 80):
            cv2.line(frame, (x, 0), (x, h), (50, 50, 70), 1)
        for y in range(0, h, 60):
            cv2.line(frame, (0, y), (w, y), (50, 50, 70), 1)
        label = "RGB CAM — NO SIGNAL" if cam_id == 0 else "IR CAM — NO SIGNAL"
        cv2.putText(frame, label, (20, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "Check USB / camera index in config.py", (20, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 140), 1, cv2.LINE_AA)
        return frame

    @staticmethod
    def _fake_thermal() -> np.ndarray:
        """Animated false-colour thermal test pattern."""
        t = time.monotonic()
        h, w = Cfg.FRAME_H, Cfg.FRAME_W
        xs = np.linspace(0, 1, w)
        ys = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(xs, ys)
        import math
        hot_x = 0.5 + 0.3 * math.sin(t * 0.4)
        hot_y = 0.5 + 0.2 * math.cos(t * 0.3)
        heat = np.exp(-((xv - hot_x) ** 2 + (yv - hot_y) ** 2) * 18)
        heat2 = np.exp(-((xv - 0.25) ** 2 + (yv - 0.75) ** 2) * 25) * 0.6
        raw = np.clip((heat + heat2) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(raw, cv2.COLORMAP_INFERNO)

    @staticmethod
    def _push(q: queue.Queue, frame: np.ndarray) -> None:
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(frame)

    @staticmethod
    def _sleep(t0: float, interval: float) -> None:
        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
