"""Flood fill: replace the 4-connected region of equal values around (y0, x0) with `new`.

Data-dependent and sequential: which cells are visited depends on what earlier steps found.
The stack is a preallocated int64 array of at least grid.size entries, not a list, because a
list that grows keeps the loop in Python (see flood_fill_list). Each cell is pushed at most once,
since it is recolored when pushed. Returns the number of cells filled.
"""


def flood_fill(grid, stack, y0: int, x0: int, new: int):
    h, w = grid.shape
    old = grid[y0, x0]
    if old == new:
        return 0
    if y0 < 0:
        y0 += h
    if x0 < 0:
        x0 += w
    grid[y0, x0] = new
    stack[0] = y0 * w + x0
    top = 1
    n = 0
    while top > 0:
        top -= 1
        p = stack[top]
        y = p // w
        x = p % w
        n += 1
        if y > 0 and grid[y - 1, x] == old:
            grid[y - 1, x] = new
            stack[top] = p - w
            top += 1
        if y < h - 1 and grid[y + 1, x] == old:
            grid[y + 1, x] = new
            stack[top] = p + w
            top += 1
        if x > 0 and grid[y, x - 1] == old:
            grid[y, x - 1] = new
            stack[top] = p - 1
            top += 1
        if x < w - 1 and grid[y, x + 1] == old:
            grid[y, x + 1] = new
            stack[top] = p + 1
            top += 1
    return n


def flood_fill_list(grid, y0: int, x0: int, new: int):
    """The same fill written the usual way, with a list as the stack.

    The list grows (append) and shrinks (pop): from 0.0.4 it is held by reference, as Python holds
    it, and the whole function compiles.
    """
    h, w = grid.shape
    old = grid[y0, x0]
    if old == new:
        return 0
    grid[y0, x0] = new
    todo = [(y0 % h, x0 % w)]
    n = 0
    while todo:
        y, x = todo.pop()
        n += 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == old:
                grid[ny, nx] = new
                todo.append((ny, nx))
    return n
