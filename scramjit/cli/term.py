"""Terminal output: colors, styles, symbols, wrapping and a spinner. Standard library only.

Color follows the usual conventions: `--color always|never|auto`, then NO_COLOR (off),
FORCE_COLOR / CLICOLOR_FORCE (on), TERM=dumb (off), else on when the stream is a terminal.
"""

import os
import re
import shutil
import sys
import threading
import time

_CODES = {
    'bold': '1', 'dim': '2', 'italic': '3', 'underline': '4',
    'red': '31', 'green': '32', 'yellow': '33', 'blue': '34', 'magenta': '35', 'cyan': '36',
    'gray': '90', 'bright_red': '91', 'bright_green': '92', 'bright_yellow': '93',
    'bright_blue': '94', 'bright_magenta': '95', 'bright_cyan': '96',
}

# Roles used across the CLI, so the palette lives in one place.
THEME = {
    'heading': ('bold',),
    'command': ('bold', 'cyan'),
    'option': ('cyan',),
    'metavar': ('dim', 'italic'),
    'code': ('cyan',),
    'prompt': ('dim',),
    'error': ('bold', 'red'),
    'warning': ('bold', 'yellow'),
    'note': ('bold', 'blue'),
    'hint': ('bold', 'green'),
    'ok': ('bold', 'green'),
    'path': ('underline',),
    'muted': ('dim',),
    'number': ('bold',),
    'caret': ('bold', 'green'),
}

_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def _wants_color(stream, mode):
    if mode == 'always':
        return True
    if mode == 'never':
        return False
    env = os.environ
    if env.get('NO_COLOR'):
        return False
    if env.get('FORCE_COLOR') or env.get('CLICOLOR_FORCE', '0') != '0':
        return True
    if env.get('TERM') == 'dumb':
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _enable_windows_vt():
    if sys.platform != 'win32':
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        ok = True
        for n in (-11, -12):  # stdout, stderr
            h = k.GetStdHandle(n)
            mode = ctypes.c_uint32()
            if not k.GetConsoleMode(h, ctypes.byref(mode)):
                ok = False
                continue
            k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return ok
    except Exception:
        return False


class Terminal:
    """Styled output to one stream (stdout or stderr)."""

    def __init__(self, stream, mode='auto'):
        self.stream = stream
        # --color=always colors even where Windows VT mode can't be enabled (a pipe)
        self.color = _wants_color(stream, mode) and (_enable_windows_vt() or mode == 'always')
        enc = getattr(stream, 'encoding', None) or 'ascii'
        try:
            '✓✗→•│─╭╰⠋'.encode(enc)
            self.unicode = True
        except (UnicodeEncodeError, LookupError):
            self.unicode = False

    # styles

    def style(self, text, *styles):
        """text in the given styles or theme roles ('bold', 'red', 'error', ...)."""
        if not self.color or not styles:
            return text
        codes = []
        for s in styles:
            for c in THEME.get(s, (s,)):
                codes.append(_CODES[c])
        return '\x1b[%sm%s\x1b[0m' % (';'.join(codes), text)

    def code(self, text):
        """Help text with `backticks` shown as code (the backticks are dropped when colored)."""
        if not self.color:
            return text
        return re.sub(r'`([^`]+)`', lambda m: self.style(m.group(1), 'code'), text)

    def symbol(self, name):
        u = {'ok': '✓', 'fail': '✗', 'arrow': '→', 'bullet': '•', 'bar': '│', 'dash': '─'}
        a = {'ok': '+', 'fail': 'x', 'arrow': '->', 'bullet': '*', 'bar': '|', 'dash': '-'}
        return (u if self.unicode else a)[name]

    # layout

    @property
    def width(self):
        return max(40, min(100, shutil.get_terminal_size((88, 24)).columns))

    def write(self, text='', end='\n'):
        try:
            self.stream.write(text + end)
        except UnicodeEncodeError:              # source lines and symbols on a terminal that can't show them
            enc = getattr(self.stream, 'encoding', None) or 'ascii'
            self.stream.write((text + end).encode(enc, 'replace').decode(enc))

    def flush(self):
        try:
            self.stream.flush()
        except (AttributeError, ValueError):
            pass


def visible_len(s):
    return len(_ANSI.sub('', s))


def wrap(text, width, indent=0, first=None):
    """Word-wrap text that may contain ANSI codes. first: indent of the first line (default
    indent). Returns lines without trailing newlines."""
    first = indent if first is None else first
    lines, cur, cur_len = [], ' ' * first, first
    start = True
    for word in text.split():
        n = visible_len(word)
        if not start and cur_len + 1 + n > width:
            lines.append(cur)
            cur, cur_len = ' ' * indent + word, indent + n
        else:
            if start:
                cur, cur_len = cur + word, cur_len + n
            else:
                cur, cur_len = cur + ' ' + word, cur_len + 1 + n
        start = False
    lines.append(cur)
    return lines


class Spinner:
    """A status line with a spinner on a terminal, a plain line otherwise.

        with Spinner(err, 'Building the runtime library') as sp:
            ...
            sp.update('still building')
    """

    FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    def __init__(self, term, text):
        self.term, self.text = term, text
        self.live = term.color and term.unicode and term.stream.isatty()
        self._stop = threading.Event()
        self._thread = None
        self.t0 = time.time()

    def __enter__(self):
        if self.live:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            self.term.write(self.term.style(self.text + '...', 'muted'))
        return self

    def update(self, text):
        self.text = text

    def _spin(self):
        i = 0
        while not self._stop.wait(0.08):
            frame = self.term.style(self.FRAMES[i % len(self.FRAMES)], 'cyan')
            self.term.stream.write('\r\x1b[2K%s %s' % (frame, self.text))
            self.term.flush()
            i += 1

    def __exit__(self, *exc):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self.term.stream.write('\r\x1b[2K')
            self.term.flush()
        return False

    @property
    def elapsed(self):
        return time.time() - self.t0
