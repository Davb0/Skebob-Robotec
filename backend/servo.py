"""
Servo / pan-tilt subsystem.

PC prototype: software-only PID — angles live in RAM, nothing moves.
Jetson migration: add PCA9685 init in __init__ and write to
                  adafruit_servo.Servo in _write_angles().
"""
import threading

from config import Cfg


class ServoController:
    def __init__(self):
        self._lock = threading.Lock()
        self.pan_angle = Cfg.PAN_HOME
        self.tilt_angle = Cfg.TILT_HOME
        self._pan_i = self._pan_prev = 0.0
        self._tilt_i = self._tilt_prev = 0.0

    # ── public interface ──────────────────────────────────────────────────────

    def home(self) -> None:
        with self._lock:
            self.pan_angle = Cfg.PAN_HOME
            self.tilt_angle = Cfg.TILT_HOME
            self._pan_i = self._pan_prev = 0.0
            self._tilt_i = self._tilt_prev = 0.0
            self._write_angles()

    def step_pid(self, pan_err: int, tilt_err: int,
                 step_max: float = Cfg.SERVO_MAX_STEP) -> None:
        with self._lock:
            self.pan_angle -= self._pid_step(
                pan_err, self._pan_i, self._pan_prev,
                Cfg.PAN_KP, Cfg.PAN_KI, Cfg.PAN_KD, step_max
            )
            self.tilt_angle -= self._pid_step(
                tilt_err, self._tilt_i, self._tilt_prev,
                Cfg.TILT_KP, Cfg.TILT_KI, Cfg.TILT_KD, step_max
            )
            self.pan_angle = max(Cfg.PAN_MIN, min(Cfg.PAN_MAX, self.pan_angle))
            self.tilt_angle = max(Cfg.TILT_MIN, min(Cfg.TILT_MAX, self.tilt_angle))
            self._write_angles()

    def nudge(self, axis: str, degrees: float) -> None:
        with self._lock:
            if axis == "pan":
                self.pan_angle = max(Cfg.PAN_MIN,
                                     min(Cfg.PAN_MAX, self.pan_angle + degrees))
            elif axis == "tilt":
                self.tilt_angle = max(Cfg.TILT_MIN,
                                      min(Cfg.TILT_MAX, self.tilt_angle + degrees))
            self._write_angles()

    def angles(self) -> tuple[float, float]:
        return self.pan_angle, self.tilt_angle

    # ── internals ────────────────────────────────────────────────────────────

    def _pid_step(self, err, integral, prev, kp, ki, kd, step_max):
        if abs(err) <= Cfg.DEADZONE:
            # decay integral when inside deadzone
            if err == 0:
                integral *= 0.9
            return 0.0
        integral = max(-Cfg.PID_INTEGRAL_CLAMP,
                       min(Cfg.PID_INTEGRAL_CLAMP, integral + err))
        step = kp * err + ki * integral + kd * (err - prev)
        prev = err
        return max(-step_max, min(step_max, step))

    def _write_angles(self) -> None:
        # No-op on PC.  On Jetson: write self.pan_angle / self.tilt_angle
        # to adafruit_servo.Servo instances via PCA9685.
        pass
