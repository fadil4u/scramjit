"""preflight --live: the report while the program runs."""

import json
import linecache
import os
import sys
import threading
import time

from .. import _dispatch
from . import report


def snapshot(elapsed):
    """The state of the run so far, adding the calls still under way, which may be most of it."""
    rows = report.run_rows()
    now = time.perf_counter()
    running = {}
    native_now = 0.0
    for f, native, started in _dispatch.running():
        for r in rows:
            if r['name'] == f._owner:
                r['native' if native else 'python'] += now - started
        native_now += (now - started) if native else 0.0
        running[f._owner] = native
    native = sum(r['native'] for r in rows)
    return {'type': 'snapshot', 'seconds': elapsed, 'native_share': min(native / elapsed, 1.0) if elapsed else 0.0,
            'running': sorted(running), 'functions': [{k: v for k, v in r.items() if k != 'function'} for r in rows],
            'hottest': [{'times': v, 'function': name, 'file': path, 'line': n} for v, name, path, n in report.hottest(5)]}


class _Tee:
    """sys.stdout or sys.stderr on a terminal: clears the block before the program writes through it."""

    def __init__(self, stream, live):
        self._stream = stream
        self._live = live

    def write(self, text):
        with self._live.lock:
            self._live.clear()
            return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class Live:
    """A thread that shows snapshots: redrawn in place on a terminal, a line at a time when stderr is piped,
    one JSON object per line with --json. Reads no more than a few bytes per function, a few times a second."""

    def __init__(self, term, output, as_json, counting):
        self.term = term
        self.counting = counting
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self._loop, name='scramjit-live', daemon=True)
        self.drawn = 0
        self.saved = None
        self.owned = None
        self.json_stream = None
        if as_json:
            self.json_stream = self.owned = open(output, 'w', encoding='utf-8') if output else None
            self.json_stream = self.json_stream or sys.stderr
        stream = term.stream
        self.redraw = not as_json and stream.isatty() and os.environ.get('TERM') != 'dumb'
        self.every = 1.0 if as_json else 0.25 if self.redraw else 5.0

    def start(self, t0):
        self.t0 = t0
        if self.redraw:
            self.saved = sys.stdout, sys.stderr
            sys.stderr = _Tee(sys.stderr, self)
            if sys.stdout.isatty():
                sys.stdout = _Tee(sys.stdout, self)
        self.thread.start()

    def stop(self):
        self.halt.set()
        self.thread.join()
        if self.saved:
            sys.stdout, sys.stderr = self.saved
        with self.lock:
            self.clear()

    def close(self):
        if self.owned:
            self.owned.close()

    def clear(self):
        if self.drawn:
            self.term.stream.write(f'\r\x1b[{self.drawn}A\x1b[J')
            self.term.stream.flush()
            self.drawn = 0

    def _loop(self):
        while not self.halt.wait(self.every):
            try:
                snap = snapshot(time.perf_counter() - self.t0)
            except RuntimeError:         # a dict changed size as it was read: the next tick gets it
                continue
            with self.lock:
                if self.json_stream:
                    json.dump(snap, self.json_stream)
                    self.json_stream.write('\n')
                    self.json_stream.flush()
                elif self.redraw:
                    self.clear()
                    block = self._block(snap)
                    self.term.stream.write(''.join(line + '\n' for line in block))
                    self.term.stream.flush()
                    self.drawn = len(block)
                else:
                    self.term.write(self._summary(snap))

    @staticmethod
    def _summary(snap):
        busiest = max(snap['functions'], key=lambda r: r['native'] + r['python'], default=None)
        most = f', busiest {busiest["name"]} ({busiest["calls"]:,} calls)' if busiest else ''
        return f'scramjit {snap["seconds"]:.1f} s: {snap["native_share"] * 100:.0f}% native{most}'

    def _block(self, snap):
        t = self.term
        wide = max(t.width - 1, 20)
        block = [t.style('scramjit', 'command') + t.style(f' {snap["seconds"]:.1f} s', 'muted') +
                 f'  {snap["native_share"] * 100:.0f}% native']
        rows = sorted(snap['functions'], key=lambda r: -(r['native'] + r['python']))[:5]
        if rows:
            across = max(len(r['name']) for r in rows)
            for r in rows:
                now = '*' if r['name'] in snap['running'] else ' '
                text = f'{now} {r["name"]:<{across}}  {r["calls"]:>9,} call{"" if r["calls"] == 1 else "s"}  {r["native"]:>7.2f} s native  {r["python"]:>7.2f} s python'
                block.append(text[:wide])
        for h in snap['hottest']:
            source = linecache.getline(h['file'], h['line']).strip()
            text = f'  {h["times"]:>14,}  {os.path.basename(h["file"])}:{h["line"]}  {source}'
            block.append(t.style(text[:wide], 'yellow'))
        return block
