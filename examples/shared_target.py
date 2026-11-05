"""A bouncing ball followed by a camera and a spotlight, all dataclasses: the camera's target and the light's target are the
ball itself, so moving the ball moves what both aim at. One object reached three ways is part of the signature: in the
compiled function it's one set of variables, and a write through any name is seen through the others, as in Python."""

import dataclasses
import math
import time

from scramjit import scram


@dataclasses.dataclass
class Vec:
    x: float
    y: float
    z: float


@dataclasses.dataclass
class Camera:
    eye: Vec
    target: Vec

    def distance(self):
        dx, dy, dz = self.target.x - self.eye.x, self.target.y - self.eye.y, self.target.z - self.eye.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclasses.dataclass
class Spot:
    pos: Vec
    target: Vec

    def follow(self, k):
        self.pos.x += (self.target.x - self.pos.x) * k      # the light drifts after what it aims at
        self.pos.y += (self.target.y - self.pos.y) * k


@dataclasses.dataclass
class Scene:
    ball: Vec
    vel: Vec
    camera: Camera
    light: Spot


@scram
def simulate(scene, steps, dt):
    """Steps the ball, bouncing on the floor; returns the average distance the camera keeps to its target."""
    total = 0.0
    for _ in range(steps):
        scene.vel.z -= 9.81 * dt
        scene.ball.x += scene.vel.x * dt
        scene.ball.y += scene.vel.y * dt
        scene.ball.z += scene.vel.z * dt
        if scene.ball.z < 0.0:
            scene.ball.z = -scene.ball.z
            scene.vel.z = -0.9 * scene.vel.z
        scene.light.follow(0.01)
        total += scene.camera.distance()
    return total / steps


def make():
    ball = Vec(0.0, 0.0, 5.0)
    return Scene(ball, Vec(0.3, 0.2, 0.0), Camera(Vec(-4.0, 0.0, 2.0), ball), Spot(Vec(0.0, 0.0, 8.0), ball))


def main():
    steps, dt = 400_000, 0.001
    plain, fast = make(), make()

    t = time.perf_counter()
    expected = simulate.__wrapped__(plain, steps, dt)
    python_s = time.perf_counter() - t

    t = time.perf_counter()
    simulate(make(), 1, dt)                       # the first call compiles
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    result = simulate(fast, steps, dt)
    scramjit_s = time.perf_counter() - t

    assert result == expected and fast == plain
    assert fast.camera.target is fast.ball is fast.light.target
    assert simulate.signatures, 'it should have compiled'
    print(f'ball at ({fast.ball.x:.2f}, {fast.ball.y:.2f}, {fast.ball.z:.2f}), light at ({fast.light.pos.x:.2f}, {fast.light.pos.y:.2f}), '
          f'camera {result:.3f} from its target on average')
    print(f'Python {python_s:.3f} s, scramjit {scramjit_s * 1e3:.1f} ms ({python_s / scramjit_s:.0f}x), compile {compile_s:.2f} s')


if __name__ == '__main__':
    main()
