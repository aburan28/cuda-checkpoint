"""Driver backends: the four calls this whole system is built around.

CliBackend shells out to cuda-checkpoint. It is the production path, because
the utility is a ~6 KB shim that resolves everything through cuGetExportTable -
capability tracks the installed driver, so there is nothing to gain from binding
libcuda ourselves except a dependency.

FakeBackend implements the same contract in memory, with the same illegal-
transition rules and a fault injection hook. Every protocol test and the whole
chaos matrix run against it.
"""

import subprocess
import threading
import time

from mncr import log
from mncr.errors import DriverError

_LOG = log.get("agent.driver")

STATE_RUNNING = "running"
STATE_LOCKED = "locked"
STATE_CHECKPOINTED = "checkpointed"


class DriverBackend:
    def get_state(self, pid):
        raise NotImplementedError

    def lock(self, pid, timeout_ms):
        raise NotImplementedError

    def checkpoint(self, pid):
        raise NotImplementedError

    def restore(self, pid, device_map=None):
        raise NotImplementedError

    def unlock(self, pid):
        raise NotImplementedError


class CliBackend(DriverBackend):
    def __init__(self, binary="cuda-checkpoint", timeout=900.0):
        self.binary = binary
        self.timeout = timeout

    def _run(self, args, timeout=None):
        cmd = [self.binary] + args
        _LOG.debug("exec", cmd=" ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout or self.timeout
            )
        except FileNotFoundError as exc:
            raise DriverError(f"{self.binary} not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise DriverError(f"{' '.join(cmd)} timed out") from exc
        if proc.returncode != 0:
            raise DriverError(
                f"{' '.join(cmd)} failed rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[:400]}"
            )
        return (proc.stdout or "").strip()

    def get_state(self, pid):
        return self._run(["--get-state", "--pid", str(pid)]).lower()

    def lock(self, pid, timeout_ms):
        # The timeout is the distributed-deadlock control: a rank still draining
        # work fails the lock rather than hanging the epoch.
        return self._run(
            ["--action", "lock", "--pid", str(pid), "--timeout", str(int(timeout_ms))],
            timeout=(timeout_ms / 1000.0) + 30,
        )

    def checkpoint(self, pid):
        return self._run(["--action", "checkpoint", "--pid", str(pid)])

    def restore(self, pid, device_map=None):
        args = ["--action", "restore", "--pid", str(pid)]
        if device_map:
            args += ["--device-map", device_map]
        return self._run(args)

    def unlock(self, pid):
        return self._run(["--action", "unlock", "--pid", str(pid)])

    def get_restore_tid(self, pid):
        return self._run(["--get-restore-tid", "--pid", str(pid)])


_LEGAL = {
    ("lock", STATE_RUNNING): STATE_LOCKED,
    ("checkpoint", STATE_LOCKED): STATE_CHECKPOINTED,
    ("restore", STATE_CHECKPOINTED): STATE_LOCKED,
    ("unlock", STATE_LOCKED): STATE_RUNNING,
}


class FakeBackend(DriverBackend):
    """In-memory driver with the real ordering rules and injectable faults.

    Enforcing the transition table is the point: a coordinator bug that
    checkpoints before locking, or unlocks twice, fails here rather than on a
    node where the failure is unrecoverable.
    """

    def __init__(self, call_latency=0.0):
        self._lock = threading.Lock()
        self._state = {}
        self._faults = {}
        self.calls = []
        # Real lock and checkpoint calls take real time, and they run one
        # process at a time within a node. Without a latency the sequential
        # term is invisible and the scale harness cannot measure the property
        # it exists to measure.
        self.call_latency = float(call_latency)

    # ------------------------------------------------------------ test hooks
    def add_pid(self, pid, state=STATE_RUNNING):
        with self._lock:
            self._state[int(pid)] = state
        return self

    def fail_next(self, action, pid=None, message="injected fault"):
        """Make the next matching call raise. pid=None matches any pid."""
        self._faults.setdefault((action, pid), []).append(message)
        return self

    def state_of(self, pid):
        with self._lock:
            return self._state.get(int(pid))

    def _maybe_fail(self, action, pid):
        for key in ((action, int(pid)), (action, None)):
            queue = self._faults.get(key)
            if queue:
                raise DriverError(queue.pop(0))

    def _transition(self, action, pid):
        pid = int(pid)
        self.calls.append((action, pid))
        self._maybe_fail(action, pid)
        if self.call_latency:
            time.sleep(self.call_latency)
        with self._lock:
            current = self._state.get(pid)
            if current is None:
                raise DriverError(f"pid {pid} is not a CUDA process")
            nxt = _LEGAL.get((action, current))
            if nxt is None:
                raise DriverError(
                    f"illegal driver transition: {action} while {current} (pid {pid})"
                )
            self._state[pid] = nxt
            return nxt

    # -------------------------------------------------------------- contract
    def get_state(self, pid):
        with self._lock:
            state = self._state.get(int(pid))
        if state is None:
            raise DriverError(f"pid {pid} is not a CUDA process")
        return state

    def lock(self, pid, timeout_ms):
        return self._transition("lock", pid)

    def checkpoint(self, pid):
        return self._transition("checkpoint", pid)

    def restore(self, pid, device_map=None):
        self._device_map = device_map
        return self._transition("restore", pid)

    def unlock(self, pid):
        return self._transition("unlock", pid)


#: What the driver says when asked to do something already done to the process.
ALREADY = "cannot be performed in the present state"


def resume(backend, pid, device_map=None, log=None):
    """Bring a process back, whatever state it was left in.

    Measured on driver 595 with CRIU 4.2.1: when the CUDA plugin is installed,
    `criu restore` performs the CUDA restore itself. The process comes back on
    the GPU with its device memory intact, and a subsequent
    cuda-checkpoint --action restore fails because there is nothing left to
    restore. Without the plugin - or resuming a process that never went through
    criu at all - the explicit calls are exactly what is needed.

    So ask rather than assume, and treat "already done" as success. Reporting a
    working restore as a failure, past the commit point, would cost the job.

    Returns "already-running" or "restored".
    """
    state = None
    try:
        state = backend.get_state(pid)
    except DriverError as exc:
        if log:
            log.debug("state query failed; probing instead", pid=pid, error=str(exc))

    if state == STATE_RUNNING:
        if log:
            log.info("already restored by the criu plugin", pid=pid)
        return "already-running"

    if state in (STATE_CHECKPOINTED, None):
        try:
            backend.restore(pid, device_map=device_map)
        except DriverError as exc:
            if ALREADY not in str(exc):
                raise
            if log:
                log.info("restore already performed", pid=pid)
            return "already-running"

    try:
        backend.unlock(pid)
    except DriverError as exc:
        if ALREADY not in str(exc):
            raise
    return "restored"


def make(cfg):
    if cfg.fake:
        return FakeBackend(call_latency=getattr(cfg, "fake_call_latency", 0.0))
    return CliBackend(cfg.cuda_checkpoint)
