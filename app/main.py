"""
FPVDefense — Desktop Control App (PyQt6)
=========================================
Connects to the backend at http://localhost:8000 (or set ROBOT_HOST env var).

Layout
------
┌─────────────────────────────────────────────────────────────────┐
│  ● FPVDefense      [RGB] [IR] [BOTH]   Connected  localhost:8000 │
├──────────────────────────────┬──────────────────────────────────┤
│  CAM 0 — RGB  (click=lock)   │  CAM 1 — INFRARED               │
│                              │                                   │
├────────────┬─────────────────┴──────────────────────────────────┤
│  LiDAR     │  Status   │  Drive pad  │  Log                     │
└────────────┴───────────┴─────────────┴──────────────────────────┘

Keyboard shortcuts
------------------
W / ↑   forward        R / Space   reset tracker
S / ↓   backward       Tab         cycle cam layout
A / ←   turn left      Esc         stop driving
D / →   turn right
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import requests

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QPoint, QPointF, QTimer, QRectF,
)
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QImage, QKeyEvent, QPainter,
    QPainterPath, QPen, QPixmap, QPolygon, QBrush, QPalette,
    QLinearGradient,
)
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget, QStackedWidget, QScrollArea,
)

HOST = os.environ.get("ROBOT_HOST", "localhost")
PORT = int(os.environ.get("ROBOT_PORT", "8000"))
BASE_URL = f"http://{HOST}:{PORT}"
WS_URL   = f"ws://{HOST}:{PORT}/ws"

DRIVE_SPEED = 0.65
DRIVE_TURN  = 0.50

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
BG2     = "#161b22"
BG3     = "#21262d"
BG4     = "#2d333b"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
RED     = "#f85149"
ORANGE  = "#e3702a"
TEXT    = "#c9d1d9"
TEXT_DIM = "#6e7681"
BORDER  = "#30363d"

def _c(hex_: str) -> QColor:
    return QColor(hex_)

def _style(color: str) -> str:
    return f"color: {color};"


# ── MJPEG thread ──────────────────────────────────────────────────────────────

class MJPEGThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, url: str):
        super().__init__()
        self._url = url

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                r = requests.get(self._url, stream=True, timeout=5)
                buf = b""
                for chunk in r.iter_content(chunk_size=8192):
                    if self.isInterruptionRequested():
                        return
                    buf += chunk
                    while True:
                        s = buf.find(b"\xff\xd8")
                        e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
                        if s < 0 or e < 0:
                            break
                        jpg = buf[s:e + 2]
                        buf = buf[e + 2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            h, w = img.shape[:2]
                            qimg = QImage(img.data, w, h, w * 3,
                                          QImage.Format.Format_BGR888)
                            self.frame_ready.emit(qimg.copy())
            except Exception:
                time.sleep(1)


# ── WebSocket thread ──────────────────────────────────────────────────────────

class WSThread(QThread):
    status_received = pyqtSignal(dict)
    lidar_received  = pyqtSignal(list)
    conn_changed    = pyqtSignal(bool)

    def __init__(self, url: str):
        super().__init__()
        self._url  = url
        self._loop = None
        self._ws   = None

    def send(self, cmd: dict) -> None:
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps(cmd)), self._loop)

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        while not self.isInterruptionRequested():
            try:
                self._loop.run_until_complete(self._connect())
            except Exception:
                self.conn_changed.emit(False)
                time.sleep(2)

    async def _connect(self) -> None:
        import websockets
        async with websockets.connect(self._url) as ws:
            self._ws = ws
            self.conn_changed.emit(True)
            async for raw in ws:
                data = json.loads(raw)
                t = data.get("type")
                if t == "status":
                    self.status_received.emit(data)
                elif t == "lidar":
                    self.lidar_received.emit(data.get("points", []))
            self._ws = None
            self.conn_changed.emit(False)


# ── Camera widget ─────────────────────────────────────────────────────────────

class CameraWidget(QWidget):
    click_at = pyqtSignal(float, float)

    def __init__(self, label: str, clickable: bool = False):
        super().__init__()
        self._label     = label
        self._clickable = clickable
        self._pixmap    = None
        self._state     = "idle"
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if clickable:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def update_frame(self, qimg: QImage) -> None:
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(self.rect(), _c(BG2))

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            ox = (w - scaled.width())  // 2
            oy = (h - scaled.height()) // 2
            p.drawPixmap(ox, oy, scaled)
        else:
            # Placeholder grid
            p.setPen(QPen(_c(BG4), 1))
            for x in range(0, w, 60):
                p.drawLine(x, 0, x, h)
            for y in range(0, h, 60):
                p.drawLine(0, y, w, y)
            p.setPen(_c(TEXT_DIM))
            p.setFont(QFont("monospace", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       f"{self._label}\nConnecting…")

        # State badge (top-left)
        state_colors = {"idle": TEXT_DIM, "locked": GREEN,
                        "searching": YELLOW}
        sc = state_colors.get(self._state, TEXT_DIM)
        badge_txt = self._state.upper()
        p.setFont(QFont("monospace", 8, QFont.Weight.Bold))
        fm = QFontMetrics(p.font())
        bw = fm.horizontalAdvance(badge_txt) + 12
        bh = 18
        p.setBrush(_c(BG3))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(6, 6, bw, bh, 4, 4)
        p.setPen(_c(sc))
        p.drawText(6, 6, bw, bh, Qt.AlignmentFlag.AlignCenter, badge_txt)

        # Label (bottom-left)
        p.setFont(QFont("monospace", 8))
        p.setPen(_c(TEXT_DIM))
        p.drawText(8, h - 6, self._label)

        # Crosshair hint when clickable and idle
        if self._clickable and self._state == "idle" and self._pixmap:
            cx, cy = w // 2, h // 2
            p.setPen(QPen(_c(ACCENT), 1, Qt.PenStyle.DotLine))
            p.drawLine(cx - 20, cy, cx + 20, cy)
            p.drawLine(cx, cy - 20, cx, cy + 20)

        # Border colour by state
        border_col = {"locked": GREEN, "searching": YELLOW}.get(self._state, BORDER)
        p.setPen(QPen(_c(border_col), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(1, 1, w - 2, h - 2, 6, 6)

    def mousePressEvent(self, ev) -> None:
        if not self._clickable or self._pixmap is None:
            return
        w, h = self.width(), self.height()
        sw = self._pixmap.width()
        sh = self._pixmap.height()
        # recompute scaled size
        scaled_w = sw
        scaled_h = sh
        if sw > 0 and sh > 0:
            scale = min(w / sw, h / sh)
            scaled_w = int(sw * scale)
            scaled_h = int(sh * scale)
        ox = (w - scaled_w) // 2
        oy = (h - scaled_h) // 2
        x = ev.position().x() - ox
        y = ev.position().y() - oy
        if 0 <= x <= scaled_w and 0 <= y <= scaled_h:
            self.click_at.emit(x / scaled_w, y / scaled_h)


# ── LiDAR polar widget ────────────────────────────────────────────────────────

class LidarWidget(QWidget):
    MAX_MM = 5000

    def __init__(self):
        super().__init__()
        self._points: list  = []
        self._nearest: float | None = None
        self._sweep_angle   = 0.0      # animated sweep line
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)          # 20fps sweep animation
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _tick(self) -> None:
        self._sweep_angle = (self._sweep_angle + 3) % 360
        self.update()

    def update_scan(self, points: list, nearest_m: float | None) -> None:
        self._points  = points
        self._nearest = nearest_m

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 14

        p.fillRect(self.rect(), _c(BG2))

        # Sweep gradient (radar glow)
        sweep_rad = math.radians(self._sweep_angle - 90)
        grad = QLinearGradient(QPointF(cx, cy),
                               QPointF(cx + r * math.cos(sweep_rad),
                                       cy + r * math.sin(sweep_rad)))
        grad.setColorAt(0.0, QColor(0, 255, 80, 60))
        grad.setColorAt(1.0, QColor(0, 255, 80, 0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        # Draw a sector (pie slice) for the sweep
        span = 40  # degrees
        p.drawPie(cx - r, cy - r, r * 2, r * 2,
                  int(-(self._sweep_angle - 90 - span) * 16),
                  int(-span * 16))

        # Grid rings
        for i in range(1, 5):
            fr = r * i // 4
            p.setPen(QPen(_c("#1e3a1e"), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPoint(cx, cy), fr, fr)
            dist = self.MAX_MM / 1000 * i / 4
            p.setPen(_c(TEXT_DIM))
            p.setFont(QFont("monospace", 6))
            p.drawText(cx + fr + 2, cy - 2, f"{dist:.1f}m")

        # Spoke lines every 30°
        p.setPen(QPen(_c("#1e3a1e"), 1))
        for deg in range(0, 360, 30):
            rad = math.radians(deg - 90)
            p.drawLine(cx, cy,
                       int(cx + r * math.cos(rad)),
                       int(cy + r * math.sin(rad)))

        # Cardinal labels
        p.setPen(_c(TEXT_DIM))
        p.setFont(QFont("monospace", 7, QFont.Weight.Bold))
        for txt, dx, dy in [("N", -4, -r - 4), ("S", -3, r + 10),
                             ("W", -r - 12, 4), ("E", r + 3, 4)]:
            p.drawText(cx + dx, cy + dy, txt)

        # Danger zone ring (< 600mm) fill
        danger_r = int(r * 600 / self.MAX_MM)
        p.setBrush(QBrush(QColor(248, 81, 73, 18)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPoint(cx, cy), danger_r, danger_r)

        # Scan points
        p.setPen(Qt.PenStyle.NoPen)
        for angle, dist in self._points:
            if dist <= 0 or dist > self.MAX_MM:
                continue
            rad  = math.radians(angle - 90)
            pr   = r * dist / self.MAX_MM
            px_  = int(cx + pr * math.cos(rad))
            py_  = int(cy + pr * math.sin(rad))
            col  = (_c(RED) if dist < 600
                    else _c(YELLOW) if dist < 1500
                    else _c(GREEN))
            p.setBrush(QBrush(col))
            dot_r = 3 if dist < 600 else 2
            p.drawEllipse(QPoint(px_, py_), dot_r, dot_r)

        # Sweep line
        p.setPen(QPen(QColor(0, 255, 80, 200), 1))
        p.drawLine(cx, cy,
                   int(cx + r * math.cos(sweep_rad)),
                   int(cy + r * math.sin(sweep_rad)))

        # Robot icon
        p.setBrush(QBrush(_c(ACCENT)))
        p.setPen(QPen(_c("#a0c8ff"), 1))
        p.drawPolygon(QPolygon([QPoint(cx, cy - 10),
                                QPoint(cx - 6, cy + 7),
                                QPoint(cx + 6, cy + 7)]))

        # Nearest obstacle readout
        if self._nearest is not None:
            col = RED if self._nearest < 0.6 else YELLOW if self._nearest < 1.5 else GREEN
            p.setPen(_c(col))
            p.setFont(QFont("monospace", 8, QFont.Weight.Bold))
            txt = f"⚠ {self._nearest:.2f} m" if self._nearest < 0.6 else f"{self._nearest:.2f} m"
            p.drawText(6, h - 6, txt)

        # Border
        p.setPen(QPen(_c(BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)


# ── Gauge bar ─────────────────────────────────────────────────────────────────

class GaugeBar(QWidget):
    """Horizontal bar from -1 to +1 with centre zero line."""

    def __init__(self, label: str, col_pos: str = GREEN, col_neg: str = ORANGE):
        super().__init__()
        self._label   = label
        self._value   = 0.0
        self._col_pos = col_pos
        self._col_neg = col_neg
        self.setFixedHeight(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v: float) -> None:
        self._value = max(-1.0, min(1.0, v))
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        mid = w // 2

        p.fillRect(self.rect(), _c(BG3))

        bar_col = _c(self._col_pos if self._value >= 0 else self._col_neg)
        bar_w   = int(abs(self._value) * (w // 2))
        if self._value >= 0:
            p.fillRect(mid, 2, bar_w, h - 4, bar_col)
        else:
            p.fillRect(mid - bar_w, 2, bar_w, h - 4, bar_col)

        # Centre line
        p.setPen(QPen(_c(BORDER), 1))
        p.drawLine(mid, 0, mid, h)

        # Label + value
        p.setPen(_c(TEXT_DIM))
        p.setFont(QFont("monospace", 7))
        p.drawText(4, h - 4, self._label)
        p.setPen(_c(TEXT))
        p.drawText(w - 40, h - 4, f"{self._value:+.2f}")


# ── Drive pad widget ──────────────────────────────────────────────────────────

class DrivePad(QWidget):
    """Visual D-pad for on-screen driving — also shows current motor vectors."""

    drive_cmd = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self._left  = 0.0
        self._right = 0.0
        self.setFixedSize(140, 140)

    def set_speeds(self, left: float, right: float) -> None:
        self._left  = left
        self._right = right
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2 - 8

        p.fillRect(self.rect(), _c(BG2))
        p.setPen(QPen(_c(BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(cx, cy), r, r)

        # Motor vector arrows
        def draw_arrow(x, spd, label):
            bar_h = int(abs(spd) * r * 0.7)
            col = _c(GREEN if spd > 0 else RED if spd < 0 else BORDER)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            if spd >= 0:
                p.drawRect(x - 6, cy - bar_h, 12, bar_h)
            else:
                p.drawRect(x - 6, cy, 12, -bar_h)
            p.setPen(_c(TEXT_DIM))
            p.setFont(QFont("monospace", 7))
            p.drawText(x - 8, cy + r - 2, label)

        draw_arrow(cx - 28, self._left,  "L")
        draw_arrow(cx + 28, self._right, "R")

        # Centre dot
        p.setBrush(QBrush(_c(ACCENT)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPoint(cx, cy), 4, 4)

        # D-pad arrow buttons (just decorative labels)
        p.setPen(_c(TEXT_DIM))
        p.setFont(QFont("monospace", 14))
        p.drawText(cx - 7, cy - r + 18, "↑")
        p.drawText(cx - 7, cy + r - 4,  "↓")
        p.drawText(cx - r + 4, cy + 5,  "←")
        p.drawText(cx + r - 16, cy + 5, "→")


# ── Event log ─────────────────────────────────────────────────────────────────

class EventLog(QWidget):
    MAX = 60

    def __init__(self):
        super().__init__()
        self._entries: deque = deque(maxlen=self.MAX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("  EVENT LOG")
        header.setFont(QFont("monospace", 8, QFont.Weight.Bold))
        header.setStyleSheet(f"background:{BG3}; color:{TEXT_DIM}; padding:3px;")
        header.setFixedHeight(20)
        layout.addWidget(header)

        self._area = QScrollArea()
        self._area.setWidgetResizable(True)
        self._area.setStyleSheet(
            f"background:{BG2}; border:none;")
        self._area.verticalScrollBar().setStyleSheet(
            f"QScrollBar{{background:{BG3}; width:6px;}}"
            f"QScrollBar::handle{{background:{BG4}; border-radius:3px;}}")

        self._inner = QWidget()
        self._inner.setStyleSheet(f"background:{BG2};")
        self._vbox = QVBoxLayout(self._inner)
        self._vbox.setContentsMargins(4, 2, 4, 2)
        self._vbox.setSpacing(1)
        self._vbox.addStretch()
        self._area.setWidget(self._inner)
        layout.addWidget(self._area)

    def log(self, msg: str, color: str = TEXT_DIM) -> None:
        ts = time.strftime("%H:%M:%S")
        lbl = QLabel(f"{ts}  {msg}")
        lbl.setFont(QFont("monospace", 7))
        lbl.setStyleSheet(f"color:{color}; background:{BG2};")
        lbl.setWordWrap(False)
        self._vbox.insertWidget(self._vbox.count() - 1, lbl)
        self._entries.append(lbl)
        if len(self._entries) == self.MAX:
            old = self._entries[0]
            self._vbox.removeWidget(old)
            old.deleteLater()
        # Scroll to bottom
        sb = self._area.verticalScrollBar()
        QTimer.singleShot(10, lambda: sb.setValue(sb.maximum()))


# ── Status panel ──────────────────────────────────────────────────────────────

class StatusPanel(QWidget):
    reset_clicked   = pyqtSignal()
    drive_cmd       = pyqtSignal(float, float)
    nudge_cmd       = pyqtSignal(str, float)

    STATE_COLORS = {"idle": TEXT_DIM, "locked": GREEN, "searching": YELLOW}

    def __init__(self):
        super().__init__()
        self._keys: set[str] = set()
        self._drive_timer = QTimer(self)
        self._drive_timer.setInterval(50)
        self._drive_timer.timeout.connect(self._emit_drive)
        self._drive_timer.start()

        self.setStyleSheet(
            f"background:{BG2}; border:1px solid {BORDER}; border-radius:6px;")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── title ─────────────────────────────────────────────────────────────
        title = QLabel("FPVDEFENSE")
        title.setFont(QFont("monospace", 13, QFont.Weight.Bold))
        title.setStyleSheet(_style(ACCENT))
        root.addWidget(title)
        root.addWidget(self._hline())

        # ── tracker state badge ───────────────────────────────────────────────
        self._state_badge = QLabel("● IDLE")
        self._state_badge.setFont(QFont("monospace", 10, QFont.Weight.Bold))
        self._state_badge.setStyleSheet(_style(TEXT_DIM))
        root.addWidget(self._state_badge)

        root.addWidget(self._hline())

        # ── servo angles ──────────────────────────────────────────────────────
        self._lbl_pan  = self._row(root, "PAN",  "90.0°")
        self._lbl_tilt = self._row(root, "TILT", "90.0°")

        # Pan/tilt error bars
        self._bar_pan  = GaugeBar("PAN ERR",  YELLOW, YELLOW)
        self._bar_tilt = GaugeBar("TILT ERR", YELLOW, YELLOW)
        root.addWidget(self._bar_pan)
        root.addWidget(self._bar_tilt)

        root.addWidget(self._hline())

        # ── target info ───────────────────────────────────────────────────────
        self._lbl_color   = self._row(root, "COLOR MATCH", "—")
        self._lbl_chassis = self._row(root, "CHASSIS",     "—")

        root.addWidget(self._hline())

        # ── motors ────────────────────────────────────────────────────────────
        self._bar_motor_l = GaugeBar("MOTOR L", GREEN, ORANGE)
        self._bar_motor_r = GaugeBar("MOTOR R", GREEN, ORANGE)
        root.addWidget(self._bar_motor_l)
        root.addWidget(self._bar_motor_r)

        root.addWidget(self._hline())

        # ── obstacle ──────────────────────────────────────────────────────────
        self._lbl_obstacle = QLabel("OBSTACLE  —")
        self._lbl_obstacle.setFont(QFont("monospace", 9, QFont.Weight.Bold))
        self._lbl_obstacle.setStyleSheet(_style(TEXT_DIM))
        root.addWidget(self._lbl_obstacle)

        root.addWidget(self._hline())

        # ── servo nudge buttons ───────────────────────────────────────────────
        nudge_lbl = QLabel("SERVO NUDGE")
        nudge_lbl.setFont(QFont("monospace", 7))
        nudge_lbl.setStyleSheet(_style(TEXT_DIM))
        root.addWidget(nudge_lbl)

        nudge_row = QHBoxLayout()
        for label, axis, deg in [("◀", "pan", -5), ("▶", "pan", +5),
                                   ("▲", "tilt", -5), ("▼", "tilt", +5)]:
            btn = self._small_btn(label)
            btn.clicked.connect(lambda _, a=axis, d=deg: self.nudge_cmd.emit(a, d))
            nudge_row.addWidget(btn)
        root.addLayout(nudge_row)

        root.addStretch()

        # ── reset button ──────────────────────────────────────────────────────
        btn_reset = QPushButton("⟳  RESET TRACKER")
        btn_reset.setFont(QFont("monospace", 9, QFont.Weight.Bold))
        btn_reset.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{ACCENT};"
            f"border:1px solid {ACCENT};border-radius:4px;padding:7px;}}"
            f"QPushButton:hover{{background:{ACCENT};color:{BG};}}")
        btn_reset.clicked.connect(self.reset_clicked)
        root.addWidget(btn_reset)

        # ── key hint ──────────────────────────────────────────────────────────
        hint = QLabel("WASD/↑↓←→ drive   R reset")
        hint.setFont(QFont("monospace", 7))
        hint.setStyleSheet(_style(TEXT_DIM))
        root.addWidget(hint)

    # ── public update ─────────────────────────────────────────────────────────

    def update_status(self, d: dict) -> None:
        state = d.get("state", "idle")
        col   = self.STATE_COLORS.get(state, TEXT)
        self._state_badge.setText(f"● {state.upper()}")
        self._state_badge.setStyleSheet(_style(col))

        self._lbl_pan.setText(f"{d.get('pan', 0):.1f}°")
        self._lbl_tilt.setText(f"{d.get('tilt', 0):.1f}°")

        fw = 640  # nominal frame width
        fh = 360
        self._bar_pan.set_value(d.get("pan_err", 0) / (fw / 2))
        self._bar_tilt.set_value(d.get("tilt_err", 0) / (fh / 2))

        score = d.get("color_score", 0)
        self._lbl_color.setText(f"{score * 100:.0f}%")

        chassis = d.get("chassis", "")
        self._lbl_chassis.setText(chassis.upper() if chassis else "—")

        self._bar_motor_l.set_value(d.get("motor_l", 0))
        self._bar_motor_r.set_value(d.get("motor_r", 0))

        obs = d.get("obstacle_m")
        if obs is not None:
            col_o = RED if obs < 0.6 else YELLOW if obs < 1.5 else GREEN
            self._lbl_obstacle.setText(f"OBSTACLE  {obs:.2f} m")
            self._lbl_obstacle.setStyleSheet(_style(col_o))
        else:
            self._lbl_obstacle.setText("OBSTACLE  —")
            self._lbl_obstacle.setStyleSheet(_style(TEXT_DIM))

    # ── keyboard drive ────────────────────────────────────────────────────────

    def key_down(self, key: str) -> None:
        self._keys.add(key)

    def key_up(self, key: str) -> None:
        self._keys.discard(key)

    def _emit_drive(self) -> None:
        fwd = "w" in self._keys or "up"    in self._keys
        bwd = "s" in self._keys or "down"  in self._keys
        lft = "a" in self._keys or "left"  in self._keys
        rgt = "d" in self._keys or "right" in self._keys

        l = r = 0.0
        if fwd:  l, r = DRIVE_SPEED, DRIVE_SPEED
        elif bwd: l, r = -DRIVE_SPEED, -DRIVE_SPEED
        if lft:  l -= DRIVE_TURN; r += DRIVE_TURN
        if rgt:  l += DRIVE_TURN; r -= DRIVE_TURN

        self.drive_cmd.emit(max(-1.0, min(1.0, l)),
                            max(-1.0, min(1.0, r)))

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row(layout, key: str, val: str) -> QLabel:
        row = QHBoxLayout()
        k = QLabel(key)
        k.setFont(QFont("monospace", 8))
        k.setStyleSheet(_style(TEXT_DIM))
        k.setFixedWidth(90)
        v = QLabel(val)
        v.setFont(QFont("monospace", 8, QFont.Weight.Bold))
        v.setStyleSheet(_style(TEXT))
        row.addWidget(k)
        row.addWidget(v)
        row.addStretch()
        layout.addLayout(row)
        return v

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color:{BORDER};")
        return line

    @staticmethod
    def _small_btn(label: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFont(QFont("monospace", 10))
        btn.setFixedSize(32, 28)
        btn.setStyleSheet(
            f"QPushButton{{background:{BG3};color:{TEXT};"
            f"border:1px solid {BORDER};border-radius:3px;}}"
            f"QPushButton:hover{{background:{BG4};color:{ACCENT};}}")
        return btn


# ── Header bar ────────────────────────────────────────────────────────────────

class HeaderBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(38)
        self.setStyleSheet(
            f"background:{BG3}; border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(10)

        self._dot = QLabel("●")
        self._dot.setFont(QFont("monospace", 11))
        self._dot.setStyleSheet(_style(RED))
        lay.addWidget(self._dot)

        title = QLabel("FPVDefense")
        title.setFont(QFont("monospace", 12, QFont.Weight.Bold))
        title.setStyleSheet(_style(ACCENT))
        lay.addWidget(title)

        lay.addStretch()

        self._conn_lbl = QLabel("Connecting…")
        self._conn_lbl.setFont(QFont("monospace", 8))
        self._conn_lbl.setStyleSheet(_style(TEXT_DIM))
        lay.addWidget(self._conn_lbl)

        sep = QLabel("|")
        sep.setStyleSheet(_style(BORDER))
        lay.addWidget(sep)

        host_lbl = QLabel(f"{HOST}:{PORT}")
        host_lbl.setFont(QFont("monospace", 8))
        host_lbl.setStyleSheet(_style(TEXT_DIM))
        lay.addWidget(host_lbl)

    def set_connected(self, ok: bool) -> None:
        if ok:
            self._dot.setStyleSheet(_style(GREEN))
            self._conn_lbl.setText("Connected")
            self._conn_lbl.setStyleSheet(_style(GREEN))
        else:
            self._dot.setStyleSheet(_style(RED))
            self._conn_lbl.setText("Disconnected")
            self._conn_lbl.setStyleSheet(_style(RED))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPVDefense")
        self.resize(1400, 820)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, _c(BG))
        self.setPalette(pal)
        self.setStyleSheet(f"QMainWindow{{background:{BG};}}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── header ────────────────────────────────────────────────────────────
        self._header = HeaderBar()
        root.addWidget(self._header)

        # ── main grid ─────────────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(5)

        # Row 0: two camera feeds
        self._cam0 = CameraWidget("CAM 0 — RGB  (click to lock)", clickable=True)
        self._cam1 = CameraWidget("CAM 1 — INFRARED")
        grid.addWidget(self._cam0, 0, 0)
        grid.addWidget(self._cam1, 0, 1)

        # Row 1: lidar | status | drive pad + log
        self._lidar  = LidarWidget()
        self._status = StatusPanel()
        self._pad    = DrivePad()
        self._log    = EventLog()

        # Right bottom cell: drive pad on top, log below
        right_bot = QWidget()
        rb_lay = QVBoxLayout(right_bot)
        rb_lay.setContentsMargins(0, 0, 0, 0)
        rb_lay.setSpacing(4)
        rb_lay.addWidget(self._pad, 0, Qt.AlignmentFlag.AlignHCenter)
        rb_lay.addWidget(self._log, 1)

        grid.addWidget(self._lidar,   1, 0)
        grid.addWidget(self._status,  1, 1)
        grid.addWidget(right_bot,     1, 2)

        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 5)
        grid.setColumnStretch(2, 3)
        grid.setRowStretch(0, 4)
        grid.setRowStretch(1, 3)

        root.addLayout(grid)

        # ── network ───────────────────────────────────────────────────────────
        self._mjpeg0 = MJPEGThread(f"{BASE_URL}/cam0")
        self._mjpeg1 = MJPEGThread(f"{BASE_URL}/cam1")
        self._ws     = WSThread(WS_URL)

        self._mjpeg0.frame_ready.connect(self._cam0.update_frame)
        self._mjpeg1.frame_ready.connect(self._cam1.update_frame)
        self._ws.status_received.connect(self._on_status)
        self._ws.lidar_received.connect(self._on_lidar)
        self._ws.conn_changed.connect(self._on_conn)

        self._mjpeg0.start()
        self._mjpeg1.start()
        self._ws.start()

        # ── signals ───────────────────────────────────────────────────────────
        self._cam0.click_at.connect(self._on_cam_click)
        self._status.reset_clicked.connect(self._on_reset)
        self._status.drive_cmd.connect(self._on_drive)
        self._status.nudge_cmd.connect(self._on_nudge)

        self._last_state    = "idle"
        self._last_obstacle = None

        self._log.log("FPVDefense started", ACCENT)
        self._log.log(f"Backend → {HOST}:{PORT}", TEXT_DIM)

    # ── handlers ──────────────────────────────────────────────────────────────

    def _on_conn(self, ok: bool) -> None:
        self._header.set_connected(ok)
        msg = "Connected to backend" if ok else "Lost connection — retrying…"
        self._log.log(msg, GREEN if ok else RED)

    def _on_cam_click(self, nx: float, ny: float) -> None:
        self._ws.send({"type": "click", "x": nx, "y": ny})
        self._log.log(f"Lock target at ({nx:.2f}, {ny:.2f})", YELLOW)

    def _on_reset(self) -> None:
        self._ws.send({"type": "reset"})
        self._log.log("Tracker reset", ORANGE)

    def _on_drive(self, left: float, right: float) -> None:
        self._ws.send({"type": "drive", "left": left, "right": right})
        self._pad.set_speeds(left, right)

    def _on_nudge(self, axis: str, deg: float) -> None:
        self._ws.send({"type": "nudge", "axis": axis, "degrees": deg})

    def _on_status(self, d: dict) -> None:
        self._status.update_status(d)
        self._last_obstacle = d.get("obstacle_m")

        # State-change log events
        state = d.get("state", "idle")
        if state != self._last_state:
            colors = {"locked": GREEN, "searching": YELLOW, "idle": TEXT_DIM}
            self._log.log(f"Tracker → {state.upper()}", colors.get(state, TEXT))
            self._last_state = state
            self._cam0.set_state(state)

        # Obstacle alert
        obs = d.get("obstacle_m")
        if obs is not None and obs < 0.6:
            self._log.log(f"⚠ OBSTACLE {obs:.2f} m", RED)

    def _on_lidar(self, points: list) -> None:
        self._lidar.update_scan(points, self._last_obstacle)

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        k = self._key_name(ev.key())
        if k in ("r", "space"):
            self._on_reset()
        elif k:
            self._status.key_down(k)

    def keyReleaseEvent(self, ev: QKeyEvent) -> None:
        k = self._key_name(ev.key())
        if k:
            self._status.key_up(k)

    @staticmethod
    def _key_name(key: int) -> str | None:
        return {
            Qt.Key.Key_W: "w",    Qt.Key.Key_A: "a",
            Qt.Key.Key_S: "s",    Qt.Key.Key_D: "d",
            Qt.Key.Key_Up: "up",  Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right",
            Qt.Key.Key_R: "r",    Qt.Key.Key_Space: "space",
        }.get(key)

    def closeEvent(self, ev) -> None:
        for t in (self._mjpeg0, self._mjpeg1, self._ws):
            t.requestInterruption()
            t.quit()
        ev.accept()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("FPVDefense")
    app.setStyle("Fusion")

    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window,          _c(BG))
    pal.setColor(QPalette.ColorRole.WindowText,      _c(TEXT))
    pal.setColor(QPalette.ColorRole.Base,            _c(BG2))
    pal.setColor(QPalette.ColorRole.AlternateBase,   _c(BG3))
    pal.setColor(QPalette.ColorRole.Button,          _c(BG3))
    pal.setColor(QPalette.ColorRole.ButtonText,      _c(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight,       _c(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, _c(BG))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
