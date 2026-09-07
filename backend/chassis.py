from config import Cfg


class ChassisController:
    def __init__(self, motors, servos):
        self._motors = motors
        self._servos = servos
        self._turning = ""
        self._turn_frames = 0

    def tick(self, pan_angle: float, pan_err_px: int,
             tilt_err: int, state: str) -> str:
        if state != "locked":
            self._stop()
            return ""

        pan_err_deg = pan_angle - Cfg.CHASSIS_PAN_TARGET
        trigger = Cfg.CHASSIS_PAN_TRIGGER
        px_trigger = int(Cfg.FRAME_W * 0.2)

        if self._turning:
            self._turn_frames += 1
            relaxed = abs(pan_err_deg) < trigger * 0.8
            safe = abs(pan_err_px) < px_trigger * 0.8
            timeout = self._turn_frames > 15
            if (relaxed or safe or timeout) and self._turn_frames >= 4:
                self._stop()
                return ""
            self._drive(self._turning)
            return self._turning

        if abs(tilt_err) > Cfg.CHASSIS_LOCK_TILT_MAX:
            return ""

        if pan_err_deg > trigger or pan_err_px < -px_trigger:
            self._start("left")
            return "left"
        if pan_err_deg < -trigger or pan_err_px > px_trigger:
            self._start("right")
            return "right"

        return ""

    def stop(self) -> None:
        self._stop()

    @property
    def turning(self) -> str:
        return self._turning

    def _start(self, direction: str) -> None:
        self._turning = direction
        self._turn_frames = 0
        self._drive(direction)

    def _drive(self, direction: str) -> None:
        spd = Cfg.CHASSIS_TURN_SPEED
        self._motors.drive(-spd if direction == "left" else spd,
                           spd if direction == "left" else -spd)

    def _stop(self) -> None:
        if self._turning:
            self._motors.stop()
            self._turning = ""
            self._turn_frames = 0
