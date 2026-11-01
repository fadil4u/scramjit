def slide(h, talus: float, rate: float, r0: int, r1: int):
    """Thermal erosion: where a cell is steeper than the talus angle above its lowest
    neighbour, material slides down. Updated in place, so each cell sees the ones before it."""
    rows, cols = h.shape
    for i in range(max(r0, 1), min(r1, rows - 1)):
        for j in range(1, cols - 1):
            c = h[i, j]
            drop = 0.0
            bi = i
            bj = j
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    d = c - h[i + di, j + dj]
                    if d > drop:
                        drop = d
                        bi = i + di
                        bj = j + dj
            if drop > talus:
                move = (drop - talus) * rate
                h[i, j] -= move
                h[bi, bj] += move
