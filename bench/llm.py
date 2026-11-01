"""A llama2.c forward pass as one function of plain Python loops, on stories15M: tokens per second
undecorated, through scramjit with default options, and with boundscheck=False, parity=False.
Weights are float64 because scramjit doesn't compile float32 arithmetic yet.

Needs two files in one directory (not in the repository):
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
  https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

Native, from the repository root:  python3 bench/llm.py DIR
Pyodide:                           MOUNT=DIR node tools/run_pyodide.mjs bench/llm.py
"""

import math
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'cimo', 'python'))

import numpy as np  # noqa: E402

from scramjit import scram  # noqa: E402

STEPS = 100     # tokens per run, compiled
PLAIN_STEPS = 2  # undecorated, about 3 to 9 s each
ROUNDS = 3


def forward(token, pos, dim, hidden, n_layers, n_heads, seq_len, vocab,
            emb, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final,
            key_cache, val_cache, logits, x, xb, xb2, hb, hb2, q, k, v, att):
    head = dim // n_heads
    root = math.sqrt(head)
    for i in range(dim):
        x[i] = emb[token * dim + i]

    for l in range(n_layers):
        # attention norm
        ss = 0.0
        for j in range(dim):
            ss += x[j] * x[j]
        ss = 1.0 / math.sqrt(ss / dim + 1e-5)
        for j in range(dim):
            xb[j] = rms_att[l * dim + j] * (ss * x[j])

        # query, key, value
        w0 = l * dim * dim
        for i in range(dim):
            a = 0.0
            b = 0.0
            c = 0.0
            r = w0 + i * dim
            for j in range(dim):
                a += wq[r + j] * xb[j]
                b += wk[r + j] * xb[j]
                c += wv[r + j] * xb[j]
            q[i] = a
            k[i] = b
            v[i] = c

        # RoPE
        for i in range(0, dim, 2):
            f = pos / 10000.0 ** ((i % head) / head)
            fr = math.cos(f)
            fi = math.sin(f)
            a = q[i]
            b = q[i + 1]
            q[i] = a * fr - b * fi
            q[i + 1] = a * fi + b * fr
            a = k[i]
            b = k[i + 1]
            k[i] = a * fr - b * fi
            k[i + 1] = a * fi + b * fr

        off = l * seq_len * dim + pos * dim
        for i in range(dim):
            key_cache[off + i] = k[i]
            val_cache[off + i] = v[i]

        # attention, every head
        loff = l * seq_len * dim
        for h in range(n_heads):
            hoff = h * head
            top = -1e30
            for t in range(pos + 1):
                r = loff + t * dim + hoff
                s = 0.0
                for i in range(head):
                    s += q[hoff + i] * key_cache[r + i]
                s /= root
                att[t] = s
                if s > top:
                    top = s
            total = 0.0
            for t in range(pos + 1):
                att[t] = math.exp(att[t] - top)
                total += att[t]
            for i in range(head):
                xb[hoff + i] = 0.0
            for t in range(pos + 1):
                r = loff + t * dim + hoff
                a = att[t] / total
                for i in range(head):
                    xb[hoff + i] += a * val_cache[r + i]

        # attention output
        r = l * dim * dim
        for i in range(dim):
            a = 0.0
            for j in range(dim):
                a += wo[r + i * dim + j] * xb[j]
            x[i] += a

        # feed-forward norm
        ss = 0.0
        for j in range(dim):
            ss += x[j] * x[j]
        ss = 1.0 / math.sqrt(ss / dim + 1e-5)
        for j in range(dim):
            xb[j] = rms_ffn[l * dim + j] * (ss * x[j])

        # SwiGLU
        r = l * dim * hidden
        for i in range(hidden):
            a = 0.0
            b = 0.0
            for j in range(dim):
                a += w1[r + i * dim + j] * xb[j]
                b += w3[r + i * dim + j] * xb[j]
            hb[i] = a * (1.0 / (1.0 + math.exp(-a))) * b

        for i in range(dim):
            a = 0.0
            for j in range(hidden):
                a += w2[r + i * hidden + j] * hb[j]
            x[i] += a

    # final norm and logits
    ss = 0.0
    for j in range(dim):
        ss += x[j] * x[j]
    ss = 1.0 / math.sqrt(ss / dim + 1e-5)
    for j in range(dim):
        x[j] = rms_final[j] * (ss * x[j])
    for i in range(vocab):
        a = 0.0
        for j in range(dim):
            a += emb[i * dim + j] * x[j]
        logits[i] = a


def load(path):
    raw = np.fromfile(path, dtype=np.float32).astype(np.float64)
    with open(path, 'rb') as f:
        dim, hidden, layers, heads, _, vocab, seq_len = struct.unpack('7i', f.read(28))
    vocab = abs(vocab)
    sizes = [vocab * dim, layers * dim] + [layers * dim * dim] * 4 + [layers * dim] + \
            [layers * dim * hidden] * 3 + [dim]
    weights = []
    p = 7
    for n in sizes:
        weights.append(raw[p:p + n])
        p += n
    return (dim, hidden, layers, heads, seq_len, vocab), weights


def load_tokenizer(path, vocab):
    with open(path, 'rb') as f:
        data = f.read()
    pieces = []
    p = 4
    for _ in range(vocab):
        n = struct.unpack_from('i', data, p + 4)[0]
        pieces.append(data[p + 8:p + 8 + n].decode('utf8', 'replace'))
        p += 8 + n
    return pieces


class Model:
    def __init__(self, directory):
        self.shape, self.weights = load(os.path.join(directory, 'stories15M.bin'))
        dim, hidden, layers, _, seq_len, vocab = self.shape
        self.pieces = load_tokenizer(os.path.join(directory, 'tokenizer.bin'), vocab)
        self.vocab = vocab
        self.caches = [np.zeros(layers * seq_len * dim) for _ in range(2)]
        self.logits = np.zeros(vocab)
        self.work = [np.zeros(n) for n in (dim, dim, dim, hidden, hidden, dim, dim, dim, seq_len)]

    def args(self, token, pos):
        return (token, pos, *self.shape, *self.weights, *self.caches, self.logits, *self.work)

    def run(self, fn, steps, seed=42):
        """Generates from BOS; only the forward passes are timed. Returns tok/s and the text."""
        rng = np.random.default_rng(seed)
        token = 1
        text = []
        elapsed = 0.0
        for pos in range(steps):
            args = self.args(token, pos)
            t = time.perf_counter()
            fn(*args)
            elapsed += time.perf_counter() - t
            p = self.logits / 0.8
            p = np.exp(p - p.max())
            token = int(rng.choice(self.vocab, p=p / p.sum()))
            text.append(self.pieces[token])
        return steps / elapsed, ''.join(text).replace('\u2581', ' ')


def main():
    directory = '/mnt' if sys.platform == 'emscripten' else sys.argv[1]
    model = Model(directory)
    default = scram(fallback=False, cache=False)(forward)
    fast = scram(fallback=False, cache=False, boundscheck=False, parity=False)(forward)
    for name, fn in (('default', default), ('fast', fast)):
        t = time.perf_counter()
        fn(*model.args(1, 0))
        print(f'compile {name}: {time.perf_counter() - t:.2f} s', flush=True)
    best = {'default': 0.0, 'fast': 0.0}
    for _ in range(ROUNDS):      # interleaved, best of each
        for name, fn in (('default', default), ('fast', fast)):
            rate, text = model.run(fn, STEPS)
            best[name] = max(best[name], rate)
            print(f'  {name}: {rate:.1f} tok/s', flush=True)
    print(f'{sys.platform}, {STEPS} tokens, best of {ROUNDS}')
    print(f'undecorated: {model.run(forward, PLAIN_STEPS)[0]:.3f} tok/s')
    print(f'default:     {best["default"]:.1f} tok/s')
    print(f'fast:        {best["fast"]:.1f} tok/s')
    print(text[:160])


if __name__ == '__main__':
    main()
