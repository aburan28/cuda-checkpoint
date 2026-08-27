"""Rank-side state machine.

Deliberately smaller than the coordinator's phase model: a rank only knows
whether it is running, cleaning up, waiting to hear what happened, or coming
back. It never learns whether the commit point was crossed - that is the
coordinator's concern, and telling the rank would only invite it to act on the
information.
"""

import enum
import threading
import time


class RankState(enum.Enum):
    RUNNING = "running"
    AGREEING = "agreeing"      # negotiating the target step with peers
    QUIESCING = "quiescing"    # running on_quiesce hooks
    VOTED = "voted"            # vote posted, waiting for the agent
    WAITING = "waiting"        # blocked on a resume or abort token
    RESUMING = "resuming"      # running on_resume hooks
    FAILED = "failed"


class RankStatus:
    """Thread-safe current state plus a short history, for diagnostics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = RankState.RUNNING
        self._epoch = None
        self._history = []
        self._step = 0

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def epoch(self):
        with self._lock:
            return self._epoch

    @property
    def step(self):
        with self._lock:
            return self._step

    def advance_step(self):
        with self._lock:
            self._step += 1
            return self._step

    def set_step(self, value):
        with self._lock:
            self._step = int(value)

    def enter(self, state, epoch=None, note=None):
        with self._lock:
            prev = self._state
            self._state = state
            if epoch is not None:
                self._epoch = epoch
            self._history.append(
                {
                    "at": time.time(),
                    "from": prev.value,
                    "to": state.value,
                    "epoch": self._epoch,
                    "note": note,
                }
            )
            del self._history[:-64]
        return self

    def snapshot(self):
        with self._lock:
            return {
                "state": self._state.value,
                "epoch": self._epoch,
                "step": self._step,
                "history": list(self._history[-8:]),
            }
