"""
LiDAR subsystem (RPLIDAR A1/A2/S1).

LIDAR_SIMULATED=True  → animated fake scan (no hardware needed).
LIDAR_SIMULATED=False → reads from real RPLIDAR via rplidar-robotics library.
"""
import math
import threading
import time

import numpy as np

from config import Cfg


class LidarScanner:
    def __init__(self):
        self._scan: list[tuple[float, float]] = []   # [(angle_deg, dist_mm)]
        self._lock = threading.Lock()
        self._stop_ev = threading.Event()

    def start(self) -> None:
        target = self._simulate if Cfg.LIDAR_SIMULATED else self._read_real
        threading.Thread(target=target, daemon=True, name="lidar").start()
        print(f"[LIDAR] {'Simulated' if Cfg.LIDAR_SIMULATED else Cfg.LIDAR_PORT} started")

    def stop(self) -> None:
        self._stop_ev.set()

    def get_scan(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._scan)

    def nearest_mm(self, front_half_deg: float = 30.0) -> float | None:
        """Closest reading within ±front_half_deg of forward (0°/360°)."""
        scan = self.get_scan()
        hi = front_half_deg
        lo = 360.0 - front_half_deg
        hits = [d for a, d in scan if d > 0 and (a <= hi or a >= lo)]
        return min(hits) if hits else None

    # ── simulated scan ────────────────────────────────────────────────────────

    def _simulate(self) -> None:
        rng = np.random.default_rng(7)
        t = 0.0
        while not self._stop_ev.is_set():
            pts = []
            for deg in range(0, 360, 2):
                rad = math.radians(deg)
                # Irregular room boundary
                wall = 3200 + 700 * math.sin(rad * 1.4 + 0.8) \
                             + 400 * math.cos(rad * 2.1)

                # One moving obstacle in the front-right quadrant
                obs_ang = (20 + 25 * math.sin(t * 0.25)) % 360
                obs_dist = 550 + 180 * math.sin(t * 0.5)
                diff = min(abs(deg - obs_ang), 360 - abs(deg - obs_ang))
                d = obs_dist + rng.normal(0, 15) if diff < 8 \
                    else wall + rng.normal(0, 35)

                pts.append((float(deg), max(50.0, d)))

            with self._lock:
                self._scan = pts
            t += 0.1
            time.sleep(0.1)

    # ── real RPLIDAR ──────────────────────────────────────────────────────────

    def _read_real(self) -> None:
        try:
            from rplidar import RPLidar
            lidar = RPLidar(Cfg.LIDAR_PORT)
            for scan in lidar.iter_scans():
                if self._stop_ev.is_set():
                    break
                with self._lock:
                    self._scan = [(float(a), float(d)) for _, a, d in scan]
            lidar.stop()
            lidar.disconnect()
        except Exception as exc:
            print(f"[LIDAR] Real device error: {exc} — falling back to simulation")
            Cfg.LIDAR_SIMULATED = True
            self._simulate()
