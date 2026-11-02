NAME = "keydoor"
DESCRIPTION = "Pick up the key, open the door in the wall, reach the goal. Layout changes each episode."
W = 9
H = 9
ACTIONS = 4
MAX_STEPS = 200
OBS = 7
GLYPHS = {0: ".", 1: "#", 2: "k", 3: "D", 4: "G"}
AGENT_X = 0
AGENT_Y = 1

def reset(s, grid, obs, rnd, w, h):
    for y in range(h):
        for x in range(w):
            grid[y, x] = 0
    wall = 3 + rnd[0] % 3
    door = 1 + rnd[1] % (h - 2)
    for y in range(h):
        grid[y, wall] = 1
    grid[door, wall] = 3
    s[0] = 0
    s[1] = rnd[2] % h
    if grid[s[1], s[0]] != 0:
        s[1] = 0
    ky = rnd[3] % h
    kx = 1 + rnd[4] % (wall - 1)
    grid[ky, kx] = 2
    gy = rnd[5] % h
    gx = wall + 1 + rnd[6] % (w - wall - 1)
    grid[gy, gx] = 4
    s[2] = 0
    s[3] = kx
    s[4] = ky
    s[5] = wall
    s[6] = door
    s[7] = gx
    s[8] = gy

def step(s, grid, obs, rnd, w, h, action):
    x = s[0]
    y = s[1]
    if action == 0:
        y -= 1
    elif action == 1:
        y += 1
    elif action == 2:
        x -= 1
    else:
        x += 1
    reward = -0.01
    if x >= 0 and x < w and y >= 0 and y < h:
        c = grid[y, x]
        if c == 0:
            s[0] = x
            s[1] = y
        elif c == 2:
            s[0] = x
            s[1] = y
            s[2] = 1
            grid[y, x] = 0
            reward = 0.5
        elif c == 3:
            if s[2] == 1:
                grid[y, x] = 0
                s[0] = x
                s[1] = y
                reward = 0.5
        elif c == 4:
            s[0] = x
            s[1] = y
            reward = 1.0
            done = True
            win = True

def observe(s, grid, obs, w, h):
    # the four neighbours, the direction to what it needs next, and whether it has the key
    x = s[0]
    y = s[1]
    i = 0
    for k in range(4):
        dx = 0
        dy = 0
        if k == 0:
            dy = -1
        elif k == 1:
            dy = 1
        elif k == 2:
            dx = -1
        else:
            dx = 1
        v = 1
        if y + dy >= 0 and y + dy < h and x + dx >= 0 and x + dx < w:
            v = grid[y + dy, x + dx]
        obs[i] = v
        i += 1
    tx = s[7]
    ty = s[8]
    if s[2] == 0:
        tx = s[3]
        ty = s[4]
    elif grid[s[6], s[5]] == 3:
        tx = s[5]
        ty = s[6]
    obs[4] = 1
    if tx > x:
        obs[4] = 2
    elif tx < x:
        obs[4] = 0
    obs[5] = 1
    if ty > y:
        obs[5] = 2
    elif ty < y:
        obs[5] = 0
    obs[6] = s[2]
