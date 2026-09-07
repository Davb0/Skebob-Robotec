"""
FPVDefense — Backend Server
================================
Runs on the robot (PC for prototype, Jetson Orin Nano for production).

Endpoints
---------
GET  /cam0        MJPEG stream — RGB / IMX678  (annotated)
GET  /cam1        MJPEG stream — Thermal
WS   /ws          Bidirectional JSON channel

WebSocket messages (server → client)
-------------------------------------
{"type":"status",  "state":…, "pan":…, "tilt":…, "pan_err":…, "tilt_err":…,
 "color_score":…, "color_rgb":…, "chassis":…, "motor_l":…, "motor_r":…,
 "obstacle_m": float|null}

{"type":"lidar", "points":[[angle_deg, dist_mm], …]}

WebSocket messages (client → server)
--------------------------------------
{"type":"click",  "x":0.0–1.0, "y":0.0–1.0}
{"type":"reset"}
{"type":"nudge",  "axis":"pan"|"tilt", "degrees":±float}
{"type":"drive",  "left":−1..1, "right":−1..1}
"""

import asyncio
import json
import queue
import threading
import time
from contextlib import asynccontextmanager

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config import Cfg
from camera import DualCamera
from lidar import LidarScanner
from motor import MotorController
from servo import ServoController
from chassis import ChassisController
from vision import VisionTracker, annotate

# ── subsystems ────────────────────────────────────────────────────────────────

cameras = DualCamera()
lidar = LidarScanner()
motors = MotorController()
servos = ServoController()
chassis = ChassisController(motors, servos)
tracker = VisionTracker()

# Annotated cam0 frames ready for MJPEG streaming
_cam0_out_q: queue.Queue = queue.Queue(maxsize=2)

# Commands from WebSocket → processed by vision thread
_cmd_q: queue.Queue = queue.Queue()

# Connected WebSocket clients (asyncio-side, protected by asyncio.Lock)
_ws_clients: list[WebSocket] = []
_ws_lock: asyncio.Lock | None = None   # created inside event loop


# ── background threads ────────────────────────────────────────────────────────

def _vision_thread() -> None:
    """Reads cam0 frames, runs tracker + PID + chassis, annotates output."""
    smooth_pan = smooth_tilt = 0.0
    last_pan_err = last_tilt_err = 0
    print("[VISION] thread started")

    while True:
        try:
            frame = cameras.cam0_q.get()
            annotate_frame = frame.copy()

            # Drain command queue
            while True:
                try:
                    cmd = _cmd_q.get_nowait()
                except queue.Empty:
                    break
                t = cmd.get("type")
                if t == "click":
                    tracker.cmd_click(cmd["x"], cmd["y"])
                elif t == "reset":
                    tracker.cmd_reset()
                    servos.home()
                    motors.stop()
                    chassis.stop()
                    smooth_pan = smooth_tilt = 0.0
                elif t == "nudge":
                    servos.nudge(cmd.get("axis", "pan"), float(cmd.get("degrees", 0)))
                elif t == "drive":
                    motors.drive(float(cmd.get("left", 0)), float(cmd.get("right", 0)))

            tracker.update(frame)

            snap = tracker.snapshot()
            state = snap["state"]
            box = snap["box"]
            pan_err, tilt_err = snap["pan_err"], snap["tilt_err"]

            chassis_dir = ""
            if box and state == VisionTracker.LOCKED:
                smooth_pan = Cfg.EMA_ALPHA * pan_err + (1 - Cfg.EMA_ALPHA) * smooth_pan
                smooth_tilt = Cfg.EMA_ALPHA * tilt_err + (1 - Cfg.EMA_ALPHA) * smooth_tilt
                pan_err_s = int(smooth_pan)
                tilt_err_s = int(smooth_tilt)
                pa, _ = servos.angles()
                chassis_dir = chassis.tick(pa, pan_err_s, tilt_err_s, state)
                servos.step_pid(pan_err_s, tilt_err_s)
                last_pan_err, last_tilt_err = pan_err_s, tilt_err_s
            elif state == VisionTracker.SEARCHING:
                servos.step_pid(last_pan_err, last_tilt_err,
                                step_max=Cfg.SERVO_MAX_STEP * 1.4)
                chassis.stop()
            else:
                smooth_pan *= 0.7
                smooth_tilt *= 0.7
                chassis.stop()

            pa, ta = servos.angles()
            annotate(annotate_frame, snap, pa, ta, chassis_dir)

            if _cam0_out_q.full():
                try:
                    _cam0_out_q.get_nowait()
                except queue.Empty:
                    pass
            _cam0_out_q.put(annotate_frame)

        except Exception as exc:
            import traceback
            print(f"[VISION] ERROR: {exc}")
            traceback.print_exc()


def _joystick_thread() -> None:
    """Optional gamepad support via pygame (Xbox / PS controller)."""
    try:
        import pygame
    except ImportError:
        print("[JOY] pygame not installed — joystick disabled")
        return

    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.joystick.init()
    joy = None
    axes: dict = {}
    DZ = Cfg.JOY_DEADZONE

    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axes[event.axis] = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                step = Cfg.JOY_SERVO_STEP
                mapping = {4: ("tilt", -step), 0: ("tilt", step),
                           1: ("pan", -step), 3: ("pan", step)}
                if event.button in mapping:
                    axis, deg = mapping[event.button]
                    servos.nudge(axis, deg)

        count = pygame.joystick.get_count()
        if count > 0 and joy is None:
            joy = pygame.joystick.Joystick(0)
            joy.init()
            axes = {i: 0.0 for i in range(joy.get_numaxes())}
            print(f"[JOY] Connected: {joy.get_name()}")
        elif count == 0 and joy is not None:
            motors.stop()
            joy = None
            axes = {}

        if joy and not chassis.turning:
            def dz(v):
                return 0.0 if abs(v) < DZ else v
            ly = -dz(axes.get(Cfg.JOY_AXIS_LEFT_Y, 0.0))
            ry = -dz(axes.get(Cfg.JOY_AXIS_RIGHT_Y, 0.0))
            motors.drive(ly, ry)

        time.sleep(0.016)


# ── MJPEG helpers ─────────────────────────────────────────────────────────────

def _encode_jpeg(frame) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, Cfg.JPEG_QUALITY])
    return buf.tobytes() if ok else None


async def _mjpeg_gen(src_q: queue.Queue):
    loop = asyncio.get_running_loop()
    while True:
        frame = await loop.run_in_executor(None, src_q.get)
        data = await loop.run_in_executor(None, _encode_jpeg, frame)
        if data:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")


# ── WebSocket broadcast tasks ─────────────────────────────────────────────────

async def _broadcast_status() -> None:
    interval = 1.0 / Cfg.STATUS_HZ
    while True:
        await asyncio.sleep(interval)
        snap = tracker.snapshot()
        pa, ta = servos.angles()
        ml, mr = motors.speeds()
        dom = snap["target_color"]
        obstacle = lidar.nearest_mm()

        msg = json.dumps({
            "type": "status",
            "state": snap["state"],
            "pan": round(pa, 1),
            "tilt": round(ta, 1),
            "pan_err": snap["pan_err"],
            "tilt_err": snap["tilt_err"],
            "color_score": snap["color_score"],
            "color_rgb": [dom[2], dom[1], dom[0]] if dom else None,
            "chassis": chassis.turning,
            "motor_l": round(ml, 3),
            "motor_r": round(mr, 3),
            "obstacle_m": round(obstacle / 1000, 2) if obstacle else None,
        })
        await _send_all(msg)


async def _broadcast_lidar() -> None:
    interval = 1.0 / Cfg.LIDAR_HZ
    while True:
        await asyncio.sleep(interval)
        points = lidar.get_scan()
        msg = json.dumps({"type": "lidar", "points": points})
        await _send_all(msg)


async def _send_all(msg: str) -> None:
    dead = []
    async with _ws_lock:
        for ws in _ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ws_lock
    _ws_lock = asyncio.Lock()

    cameras.start()
    lidar.start()
    threading.Thread(target=_vision_thread, daemon=True, name="vision").start()
    threading.Thread(target=_joystick_thread, daemon=True, name="joystick").start()

    asyncio.create_task(_broadcast_status())
    asyncio.create_task(_broadcast_lidar())

    print("=" * 55)
    print("  FPVDefense Backend")
    print(f"  http://localhost:{Cfg.PORT}")
    print(f"  Cameras  : {Cfg.CAM0_INDEX} (RGB)  {Cfg.CAM1_INDEX} (Thermal)")
    print(f"  LiDAR    : {'simulated' if Cfg.LIDAR_SIMULATED else Cfg.LIDAR_PORT}")
    print("=" * 55)
    yield
    cameras.stop()
    lidar.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/cam0")
async def cam0_feed():
    return StreamingResponse(
        _mjpeg_gen(_cam0_out_q),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/cam1")
async def cam1_feed():
    return StreamingResponse(
        _mjpeg_gen(cameras.cam1_q),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    print(f"[WS] Client connected — {len(_ws_clients)} total")
    try:
        async for text in websocket.iter_text():
            try:
                _cmd_q.put_nowait(json.loads(text))
            except (json.JSONDecodeError, queue.Full):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)
        print(f"[WS] Client disconnected — {len(_ws_clients)} remaining")


if __name__ == "__main__":
    uvicorn.run("main:app", host=Cfg.HOST, port=Cfg.PORT, reload=False)
