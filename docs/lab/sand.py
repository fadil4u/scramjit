EMPTY, WALL, SAND, WATER, DRAIN = 0, 1, 2, 3, 4


def sand(cells, rng, steps: int):
    """Falling sand: grains fall and slide, water falls and spreads, drains swallow both, and
    emitters along the top keep pouring. Cells update in place, bottom up, so each one moves
    into a world the ones before it already changed."""
    rows, cols = cells.shape
    for s in range(steps):
        for x in range(4, cols - 4, 24):                   # the emitters
            rng[0] = (rng[0] * 1103515245 + 12345) % 2147483648
            if rng[0] % 3 == 0:
                for e in range(x, x + 2):
                    if cells[1, e] == 0:
                        cells[1, e] = 2 if (x // 24) % 2 == 0 else 3
        for y in range(rows - 2, -1, -1):
            right = (s + y) % 2 == 0                        # alternate the sweep, so nothing drifts
            for k in range(cols):
                x = k if right else cols - 1 - k
                c = cells[y, x]
                if c != 2 and c != 3:
                    continue
                rng[0] = (rng[0] * 1103515245 + 12345) % 2147483648
                d = 1 if rng[0] % 2 == 0 else -1
                below = cells[y + 1, x]
                if below == 4:                              # a drain takes it
                    cells[y, x] = 0
                elif below == 0 or (c == 2 and below == 3):
                    cells[y + 1, x] = c
                    cells[y, x] = below
                elif 0 <= x + d < cols and (cells[y + 1, x + d] == 0 or (c == 2 and cells[y + 1, x + d] == 3)):
                    cells[y, x] = cells[y + 1, x + d]
                    cells[y + 1, x + d] = c
                elif c == 3:                                # water spreads sideways
                    if 0 <= x + d < cols and cells[y, x + d] == 0:
                        cells[y, x + d] = 3
                        cells[y, x] = 0
                    elif 0 <= x - d < cols and cells[y, x - d] == 0:
                        cells[y, x - d] = 3
                        cells[y, x] = 0
