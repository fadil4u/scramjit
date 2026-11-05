"""A temperature controller as a plain dataclass, with @scram on its method. Each step depends on the one before, so numpy
can't vectorize it. `self.integral` and `self.previous` stay on the object, between calls as they would in Python, and
`name` can be anything, since only the attributes the method uses matter to scramjit."""

import dataclasses
import time

import numpy as np

from scramjit import scram


@dataclasses.dataclass
class Controller:
    kp: float
    ki: float
    kd: float
    dt: float
    name: str = 'oven'
    integral: float = 0.0
    previous: float = 0.0
    steps: int = 0
    saturated: int = 0

    @scram
    def run(self, target, temperature, heat_loss):
        """PID control of a room losing heat, one sample of the target per step. Returns the final temperature."""
        for i in range(len(target)):
            error = target[i] - temperature
            self.integral += error * self.dt
            derivative = (error - self.previous) / self.dt
            self.previous = error
            power = self.kp * error + self.ki * self.integral + self.kd * derivative
            if power > 100.0:
                power = 100.0
                self.saturated += 1
            elif power < 0.0:
                power = 0.0
            temperature += (power - heat_loss * temperature) * self.dt
            self.steps += 1
        return temperature


def settings():
    return Controller(kp=8.0, ki=0.4, kd=1.0, dt=0.01)


def main():
    target = 20.0 + 5.0 * np.sin(np.arange(2_000_000) * 1e-4)
    first = settings()
    plain = Controller.run.__wrapped__

    t = time.perf_counter()
    expected = plain(first, target, 15.0, 0.1)
    python_s = time.perf_counter() - t

    second = settings()
    t = time.perf_counter()
    second.run(target[:1], 15.0, 0.1)      # the first call compiles
    compile_s = time.perf_counter() - t

    second = settings()
    t = time.perf_counter()
    result = second.run(target, 15.0, 0.1)
    scramjit_s = time.perf_counter() - t

    assert result == expected and second == first
    assert Controller.run.signatures, 'it should have compiled'
    print(f'{second.name}: temperature {result:.4f} after {second.steps} steps, power at its limit for {second.saturated}')
    print(f'Python {python_s:.3f} s, scramjit {scramjit_s * 1e3:.1f} ms ({python_s / scramjit_s:.0f}x), compile {compile_s:.2f} s')


if __name__ == '__main__':
    main()
