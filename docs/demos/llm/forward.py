# One forward pass of Llama 2, as plain loops.
# The page samples the next token from logits.
# Ported from llama2.c by Andrej Karpathy (MIT).
import math


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

    # temperature
    for i in range(vocab):
        logits[i] /= 0.8
