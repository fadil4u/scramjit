"""Lookups in a binary search tree of dataclass nodes, each walking node = node.left or node.right until the key is found
or the walk reaches None, counting the visits on the nodes it passes. Which node comes next depends on the data, so the
nodes are a pool in the compiled function and a reference is an index into it; the visits go back onto the nodes."""

import dataclasses
import random
import time

from scramjit import scram


@dataclasses.dataclass
class Node:
    key: int
    left: object = None
    right: object = None
    visits: int = 0


def insert(root, key):
    if root is None:
        return Node(key)
    node = root
    while True:
        side = 'left' if key < node.key else 'right'
        child = getattr(node, side)
        if child is None:
            setattr(node, side, Node(key))
            return root
        node = child


@scram
def lookups(root, keys):
    """How many of keys are in the tree, and the depth of the deepest walk."""
    found = 0
    deepest = 0
    for key in keys:
        node = root
        depth = 0
        while node is not None:
            node.visits += 1
            if key == node.key:
                found += 1
                break
            node = node.left if key < node.key else node.right
            depth += 1
        deepest = max(deepest, depth)
    return found, deepest


def tree(n, seed):
    rng = random.Random(seed)
    root = None
    for k in rng.sample(range(4 * n), n):
        root = insert(root, k)
    return root


def main():
    keys = random.Random(2).choices(range(8000), k=200_000)
    plain, fast = tree(2000, 1), tree(2000, 1)

    t = time.perf_counter()
    expected = lookups.__wrapped__(plain, keys)
    python_s = time.perf_counter() - t

    t = time.perf_counter()
    lookups(tree(10, 1), keys[:10])               # the first call compiles
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    result = lookups(fast, keys)
    scramjit_s = time.perf_counter() - t

    assert result == expected and fast == plain   # the visits went back onto the nodes too
    assert lookups.signatures, 'it should have compiled'
    print(f'{result[0]:,} of {len(keys):,} keys found, deepest walk {result[1]}, root visited {fast.visits:,} times')
    print(f'Python {python_s:.3f} s, scramjit {scramjit_s * 1e3:.1f} ms ({python_s / scramjit_s:.0f}x), compile {compile_s:.2f} s')


if __name__ == '__main__':
    main()
