"""Public API of the rank library.

    import torchckpt

    torchckpt.init(job_id="train-7", rank=rank, world_size=world)

    for step in range(steps):
        with torchckpt.safe_point():
            train_step()

A checkpoint is only ever taken at a safe point. Everything else in this module
exists to make that sentence true: agreeing on which safe point, tearing down
what the driver cannot handle before reaching it, proving the teardown worked,
and rebuilding afterwards whether the epoch committed or aborted.
"""

import contextlib
import dataclasses
import os
import time

from mncr import config, log
from mncr.errors import PreconditionError, TimeoutError_
from mncr.proto import Vote

from . import cleanroom, graphs, torch_backend
from .channel import AgentClient, ControlDir
from .state import RankState, RankStatus

_LOG = log.get("torchckpt")


@dataclasses.dataclass
class ResumeContext:
    """Everything a rank needs to come back, handed to on_resume hooks."""

    epoch_id: str
    rank: int
    world_size: int
    restored: bool           # True if the process was dumped and restored
    aborted: bool            # True if the epoch was abandoned before committing
    init_method: str = None  # fresh rendezvous; peers may have moved
    backend: str = "nccl"
    device_map: dict = dataclasses.field(default_factory=dict)
    extra: dict = dataclasses.field(default_factory=dict)


class _Runtime:
    def __init__(self):
        self.cfg = config.load()
        self.job_id = None
        self.rank = None
        self.world_size = None
        self.control = None
        self.agent = None
        self.status = RankStatus()
        self.quiesce_hooks = []
        self.resume_hooks = []
        self.auto_teardown = True
        self.strict_clean = True
        self.proc_root = "/proc"
        self.pending = None          # {"epoch_id":..., "target_step":...}
        self.initialized = False
        self.last_report = {}
        self.had_process_group = False

    # ------------------------------------------------------------- lifecycle
    def init(
        self,
        job_id=None,
        rank=None,
        world_size=None,
        agent_addr=None,
        control_root=None,
        auto_teardown=True,
        strict_clean=None,
        proc_root="/proc",
        register=True,
    ):
        self.job_id = job_id or os.environ.get("MNCR_JOB_ID") or "job"
        self.rank = int(
            rank if rank is not None else os.environ.get("RANK", os.environ.get("MNCR_RANK", 0))
        )
        self.world_size = int(
            world_size
            if world_size is not None
            else os.environ.get("WORLD_SIZE", os.environ.get("MNCR_WORLD_SIZE", 1))
        )
        self.control = ControlDir(
            control_root or os.environ.get("MNCR_CONTROL_ROOT", "/run/mncr"),
            self.job_id,
        )
        self.agent = AgentClient(agent_addr or self.cfg.rank_addr)
        self.auto_teardown = auto_teardown
        self.strict_clean = (
            self.cfg.strict_clean if strict_clean is None else bool(strict_clean)
        )
        self.proc_root = proc_root
        self.initialized = True

        if register:
            try:
                self.agent.register(
                    self.job_id,
                    self.rank,
                    os.getpid(),
                    self.world_size,
                    torch_backend.local_gpu_uuids(),
                )
            except Exception as exc:
                # Registration is best-effort at startup: the agent may come up
                # after the job. It is retried on every resume.
                _LOG.warn("agent registration failed", error=str(exc))

        _LOG.info(
            "initialized",
            job=self.job_id,
            rank=self.rank,
            world_size=self.world_size,
            control=self.control.job_dir,
        )
        return self

    def _require_init(self):
        if not self.initialized:
            raise PreconditionError("torchckpt.init() has not been called")

    # ------------------------------------------------------------ safe point
    def safe_point(self):
        """Service any pending epoch, then return so the caller can do a step."""
        self._require_init()
        step = self.status.step

        if self.pending is None:
            request = self.control.poll_request()
            if request and request.get("action") == "checkpoint":
                self._begin(request)

        if self.pending is not None and step >= self.pending["target_step"]:
            self._service()

        self.status.advance_step()

    def _begin(self, request):
        epoch_id = request.get("epoch_id")
        self.status.enter(RankState.AGREEING, epoch=epoch_id)
        target = torch_backend.agree_on_target_step(
            self.status.step, lookahead=int(request.get("lookahead", 1))
        )
        self.pending = {"epoch_id": epoch_id, "target_step": target, "request": request}
        _LOG.info(
            "epoch requested", epoch=epoch_id, target_step=target, rank=self.rank
        )

    # --------------------------------------------------------------- service
    def _service(self):
        epoch_id = self.pending["epoch_id"]
        request = self.pending["request"]
        started = time.time()

        vote, findings, error = Vote.CLEAN, [], None
        try:
            self.status.enter(RankState.QUIESCING, epoch=epoch_id)
            self.last_report = self._quiesce()
            cleanroom.assert_clean(self.proc_root, strict=self.strict_clean)
        except Exception as exc:
            vote, error = Vote.DIRTY, str(exc)
            findings = cleanroom.findings_as_dicts(
                getattr(exc, "findings", []) or []
            )
            _LOG.error("quiesce failed", epoch=epoch_id, error=error)

        self.status.enter(RankState.VOTED, epoch=epoch_id)
        try:
            self.agent.post_vote(
                self.job_id,
                self.rank,
                epoch_id,
                vote.value,
                os.getpid(),
                findings=findings,
                error=error,
            )
        except Exception as exc:
            # If the vote cannot be delivered the coordinator will time us out
            # and abort. Keep waiting for the token rather than resuming into a
            # job whose other ranks are quiesced.
            _LOG.error("vote undeliverable", epoch=epoch_id, error=str(exc))

        self.status.enter(RankState.WAITING, epoch=epoch_id)
        timeout = float(request.get("wait_timeout", self.cfg.checkpoint_timeout))
        token = self.control.wait_for_token(self.rank, epoch_id, timeout)
        if token is None:
            self.status.enter(RankState.FAILED, epoch=epoch_id)
            raise TimeoutError_(
                f"no resume token for epoch {epoch_id} after {timeout}s"
            )

        self._resume(token, epoch_id)
        self.pending = None
        _LOG.info(
            "epoch complete",
            epoch=epoch_id,
            rank=self.rank,
            seconds=round(time.time() - started, 3),
            restored=bool(token.get("restored")),
            aborted=bool(token.get("aborted")),
        )

    def _quiesce(self):
        report = {}
        for name, hook in self.quiesce_hooks:
            hook()
            report[name] = "ok"
        if self.auto_teardown:
            report["default_teardown"] = torch_backend.default_teardown()
            # Only rebuild what was actually torn down. A rank with no process
            # group must not acquire one on resume just because the default
            # rebuild ran.
            self.had_process_group = bool(
                report["default_teardown"].get("process_group_destroyed")
            )
        return report

    def _resume(self, token, epoch_id):
        self.status.enter(RankState.RESUMING, epoch=epoch_id)
        ctx = ResumeContext(
            epoch_id=epoch_id,
            rank=int(token.get("rank", self.rank)),
            world_size=int(token.get("world_size", self.world_size)),
            restored=bool(token.get("restored")),
            aborted=bool(token.get("aborted")),
            init_method=token.get("init_method"),
            backend=token.get("backend", "nccl"),
            device_map=token.get("device_map") or {},
            extra=token.get("extra") or {},
        )
        # The rank may have moved: rank id and world size can both change across
        # a restore onto different hardware.
        self.rank, self.world_size = ctx.rank, ctx.world_size

        if self.auto_teardown and self.had_process_group:
            torch_backend.default_rebuild(ctx)
        for name, hook in self.resume_hooks:
            hook(ctx)
        if graphs.REGISTRY.names():
            graphs.REGISTRY.recapture_all()

        if ctx.restored:
            # A restored process may have a different host pid and a different
            # agent; re-announce before anything else can go wrong.
            try:
                self.agent.register(
                    self.job_id,
                    self.rank,
                    os.getpid(),
                    self.world_size,
                    torch_backend.local_gpu_uuids(),
                )
            except Exception as exc:
                _LOG.warn("re-registration failed", error=str(exc))

        self.status.enter(RankState.RUNNING, epoch=epoch_id)


_RT = _Runtime()


# --------------------------------------------------------------- public API
def init(**kwargs):
    return _RT.init(**kwargs)


def on_quiesce(fn):
    """Register a teardown hook. Hooks run in registration order."""
    _RT.quiesce_hooks.append((fn.__name__, fn))
    return fn


def on_resume(fn):
    """Register a rebuild hook, called with a ResumeContext."""
    _RT.resume_hooks.append((fn.__name__, fn))
    return fn


@contextlib.contextmanager
def safe_point():
    """Mark a point at which a checkpoint may be taken."""
    _RT.safe_point()
    yield


def checkpoint_barrier():
    """Explicit form of safe_point() for loops that cannot use a with block."""
    _RT.safe_point()


def assert_clean(strict=None):
    return cleanroom.assert_clean(
        _RT.proc_root, strict=_RT.strict_clean if strict is None else strict
    )


def status():
    snap = _RT.status.snapshot()
    snap.update(
        {"job_id": _RT.job_id, "rank": _RT.rank, "world_size": _RT.world_size}
    )
    return snap


def runtime():
    """Escape hatch for tests and for the verification harness."""
    return _RT
