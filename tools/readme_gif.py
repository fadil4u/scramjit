"""The README's race: the site's Mandelbrot kernel as plain Python and through scramjit, side by
side on one clock. Every frame of the GIF gives each side the same real time; nothing is staged.
scramjit shows at most one new fractal frame per GIF frame, as the site's demo does.
Needs Pillow (not a dependency of scramjit). Run from the repository root:
    python3 tools/readme_gif.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(HERE, '..', 'docs', 'lab'), os.path.join(HERE, '..', '..', 'cimo', 'python')]

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import fractal  # noqa: E402
import lab  # noqa: E402
from scramjit import scram  # noqa: E402

N = lab.FRACTAL_N
FPS, SECONDS = 15, 8
BUDGET = 1 / FPS
BAR, GAP = 34, 6
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'
OUT = os.path.join(HERE, '..', 'docs', 'readme', 'race.gif')


class Side:
    def __init__(self, fn, cap):
        self.fn, self.cap = fn, cap
        self.img = np.zeros((N, N, 4), np.uint8)
        self.img[..., 3] = 255
        self.scale, self.row, self.frames, self.its, self.spent = 3.2 / N, 0, 0, 0, 0.0

    def run(self, seconds, pal, centre):
        """Rows of the current frame for this much real time."""
        t0 = time.perf_counter()
        done = 0
        while time.perf_counter() - t0 < seconds and not (self.cap and done):
            lab.S['scale'] = self.scale
            self.its += int(self.fn(self.img, pal, *centre, self.scale, lab.fractal_maxit(), self.row, self.row + 1))
            self.row += 1
            if self.row == N:
                self.row, self.frames = 0, self.frames + 1
                self.scale = self.scale * 0.955 if self.scale > 1e-9 else 3.2 / N
                done += 1
        self.spent += time.perf_counter() - t0


def main():
    pal = lab.palette()
    centre = lab.DIVES[0]
    compiled = scram(fractal.mandelbrot, fallback=False)
    compiled(np.zeros((N, N, 4), np.uint8), pal, *centre, 3.2 / N, 100, 0, 1)      # compile first
    py, sj = Side(fractal.mandelbrot, False), Side(compiled, True)
    big, small = ImageFont.truetype(FONT, 14), ImageFont.truetype(FONT, 12)
    W, H = 2 * N + GAP, N + BAR
    frames = []
    for f in range(FPS * SECONDS):
        py.run(BUDGET, pal, centre)
        sj.run(BUDGET, pal, centre)
        im = Image.new('RGB', (W, H), (11, 13, 22))
        im.paste(Image.fromarray(py.img[..., :3]), (0, BAR))
        im.paste(Image.fromarray(sj.img[..., :3]), (N + GAP, BAR))
        d = ImageDraw.Draw(im)
        d.text((8, 9), 'Python', font=big, fill=(147, 197, 253))
        d.text((N + GAP + 8, 9), '@scram', font=big, fill=(253, 186, 116))
        d.text((N - 8, 10), f'{py.frames} frames', font=small, fill=(170, 179, 200), anchor='ra')
        d.text((W - 8, 10), f'{sj.frames} frames', font=small, fill=(170, 179, 200), anchor='ra')
        frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=round(1000 / FPS), loop=0, optimize=True)
    rate = lambda s: s.its / s.spent
    print(f'python {py.frames} frames, {rate(py) / 1e6:.1f}M it/s; scramjit {sj.frames} frames, {rate(sj) / 1e6:.0f}M it/s; '
          f'{rate(sj) / rate(py):.0f}x; {os.path.getsize(OUT) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
