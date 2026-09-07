"""
Motor subsystem.

PC prototype: pure-software stub, tracks speed only.
Jetson migration: replace Motor.__init__ and forward/backward/stop
                  with Jetson.GPIO calls (L298N or similar driver).
"""


class Motor:
    def __init__(self, in1: int, in2: int, en: int):
        # in1, in2, en are GPIO pin numbers — ignored on PC
        self._speed = 0.0

    def forward(self, speed: float = 1.0) -> None:
        self._speed = max(0.0, min(1.0, speed))

    def backward(self, speed: float = 1.0) -> None:
        self._speed = -max(0.0, min(1.0, speed))

    def stop(self) -> None:
        self._speed = 0.0

    @property
    def speed(self) -> float:
        return self._speed


class MotorController:
    def __init__(self):
        self._left = Motor(0, 0, 0)
        self._right = Motor(0, 0, 0)

    def drive(self, left_spd: float, right_spd: float) -> None:
        self._apply(self._left, left_spd)
        self._apply(self._right, right_spd)

    def stop(self) -> None:
        self._left.stop()
        self._right.stop()

    def speeds(self) -> tuple[float, float]:
        return self._left.speed, self._right.speed

    @staticmethod
    def _apply(m: Motor, spd: float) -> None:
        spd = max(-1.0, min(1.0, spd))
        if spd > 0:
            m.forward(spd)
        elif spd < 0:
            m.backward(-spd)
        else:
            m.stop()
