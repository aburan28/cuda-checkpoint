"""Structured line logging, stdlib only.

One JSON object per line so the agent's output is greppable out of a container
log without a sidecar. Human mode is available for local runs.
"""

import json
import os
import sys
import threading
import time

_LOCK = threading.Lock()
_HUMAN = os.environ.get("MNCR_LOG_HUMAN", "") not in ("", "0", "false")
_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_MIN = _LEVELS.get(os.environ.get("MNCR_LOG_LEVEL", "info").lower(), 20)


def _emit(level, component, msg, fields):
    if _LEVELS[level] < _MIN:
        return
    rec = {"ts": round(time.time(), 6), "level": level, "component": component, "msg": msg}
    rec.update(fields)
    with _LOCK:
        if _HUMAN:
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            sys.stderr.write(f"{level:>5} [{component}] {msg} {extra}\n".rstrip() + "\n")
        else:
            sys.stderr.write(json.dumps(rec, default=str) + "\n")
        sys.stderr.flush()


class Logger:
    __slots__ = ("component", "_bound")

    def __init__(self, component, **bound):
        self.component = component
        self._bound = bound

    def bind(self, **fields):
        merged = dict(self._bound)
        merged.update(fields)
        return Logger(self.component, **merged)

    def debug(self, msg, **f):
        _emit("debug", self.component, msg, {**self._bound, **f})

    def info(self, msg, **f):
        _emit("info", self.component, msg, {**self._bound, **f})

    def warn(self, msg, **f):
        _emit("warn", self.component, msg, {**self._bound, **f})

    def error(self, msg, **f):
        _emit("error", self.component, msg, {**self._bound, **f})


def get(component, **bound):
    return Logger(component, **bound)
