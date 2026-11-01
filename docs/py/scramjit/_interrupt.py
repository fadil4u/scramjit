"""Ctrl-C during a compiled call.

Python runs signal handlers between bytecodes, and a kernel runs none, so on its own KeyboardInterrupt
waits until the kernel returns. Here a thread waits on signal.set_wakeup_fd, which Python's C handler
writes to, and sets a flag that kernels check in their loops. A kernel that sees it returns INTERRUPTED,
and KeyboardInterrupt is raised as usual (see _runtime.exception).

It only ever stops a kernel on the main thread, where Python would raise, and only under Python's default
handler: a custom handler might not raise, and a stopped kernel can't resume. So with one, or if something
else owns the wakeup fd (asyncio's add_signal_handler), a call simply waits as it did before.
"""

import ctypes
import os
import signal
import threading

_flag = ctypes.c_int(0)
_never = ctypes.c_int(0)         # what other threads' kernels get: nothing stops them
FLAG = ctypes.addressof(_flag)
NEVER = ctypes.addressof(_never)
_main = threading.main_thread().ident
_lock = threading.Lock()
_socks = None


def address(get_ident=threading.get_ident):
    return FLAG if get_ident() == _main else NEVER


def _watch(reader):
    number = signal.SIGINT.value
    while True:
        try:
            data = reader.recv(4096)
        except OSError:
            return
        if not data:
            return
        if number in data and signal.getsignal(signal.SIGINT) is signal.default_int_handler:
            _flag.value = 1


def ensure():
    """Start watching, once. Only the main thread can, and only if nothing else has the wakeup fd."""
    global _socks
    if _socks is not None or threading.current_thread() is not threading.main_thread():
        return
    import socket
    with _lock:
        if _socks is not None:
            return
        reader, writer = socket.socketpair()
        writer.setblocking(False)
        try:
            previous = signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False)
        except ValueError:              # not really the main thread of the main interpreter
            reader.close()
            writer.close()
            return
        if previous != -1:              # someone else's: put it back
            signal.set_wakeup_fd(previous)
            reader.close()
            writer.close()
            _socks = ()
            return
        _socks = (reader, writer)
        threading.Thread(target=_watch, args=(reader,), name='scramjit-interrupt', daemon=True).start()


def _after_fork_in_child():
    """A forked child shares the parent's sockets: its Ctrl-C must not stop the parent's kernel."""
    global _socks, _main
    _main = threading.get_ident()
    _flag.value = 0
    if _socks:
        signal.set_wakeup_fd(-1)
        for s in _socks:
            s.close()
        _socks = None
        ensure()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_in_child)
