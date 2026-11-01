import math


def mandelbrot(img, palette, cx: float, cy: float, scale: float, maxit: int, r0: int, r1: int):
    """Rows r0 to r1 of the Mandelbrot set around (cx, cy). Each pixel iterates z = z*z + c until
    it escapes, and how long that takes depends on the pixel. Returns the iterations done."""
    rows, cols = img.shape[0], img.shape[1]
    n = palette.shape[0]
    total = 0
    for y in range(r0, r1):
        ci = cy + (y - rows / 2) * scale
        for x in range(cols):
            cr = cx + (x - cols / 2) * scale
            zr = 0.0
            zi = 0.0
            i = 0
            while i < maxit and zr * zr + zi * zi < 256.0:
                t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                i += 1
            total += i
            if i == maxit:
                img[y, x, 0] = 0
                img[y, x, 1] = 0
                img[y, x, 2] = 0
            else:
                # smooth colouring: the fractional escape count, from how far past the radius z got
                nu = i + 1 - math.log(math.log(zr * zr + zi * zi) * 0.5) / math.log(2.0)
                k = int(nu * 3.0) % n
                img[y, x, 0] = palette[k, 0]
                img[y, x, 1] = palette[k, 1]
                img[y, x, 2] = palette[k, 2]
    return total
