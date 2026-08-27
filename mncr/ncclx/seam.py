"""Strategy seam: destroy-and-rebuild versus suspend-and-resume.

Destroy-and-rebuild is correct everywhere and costs a full communicator
re-initialisation on resume - seconds at TP=8. Suspend-and-resume is much
cheaper but, as shipped in NCCL 2.29.7, only releases dynamic GPU memory
(NCCL_SUSPEND_MEM). It does not release NVLS multicast state or network
resources, which are exactly what the checkpoint path cannot handle.

So the shipped suspend is not sufficient on its own, and this module refuses to
select it unless the runtime advertises flags that cover the rest. That refusal
is the point: a strategy that looks cheaper and silently produces
unrestorable processes is worse than a slow one.
"""

import os
import time

from mncr import log

_LOG = log.get("ncclx.seam")

# Flag values mirror the NCCL header. MEM is upstream today; NET and NVLS are
# what issue #2337 proposes and what a vendored build would add.
NCCL_SUSPEND_MEM = 0x01
NCCL_SUSPEND_NET = 0x02
NCCL_SUSPEND_NVLS = 0x04

REQUIRED_FOR_CHECKPOINT = NCCL_SUSPEND_MEM | NCCL_SUSPEND_NET | NCCL_SUSPEND_NVLS


def nccl_version():
    try:
        import torch

        raw = torch.cuda.nccl.version()
        if isinstance(raw, tuple):
            return raw
        return (raw // 10000, (raw // 100) % 100, raw % 100)
    except Exception:
        return None


def supported_suspend_flags():
    """Flags this NCCL build advertises.

    A vendored build carrying the NVLS/net work is expected to export
    MNCR_NCCL_SUSPEND_FLAGS; without it we assume only the upstream MEM flag,
    which is not enough to checkpoint.
    """
    override = os.environ.get("MNCR_NCCL_SUSPEND_FLAGS")
    if override:
        try:
            return int(override, 0)
        except ValueError:
            _LOG.warn("unparseable MNCR_NCCL_SUSPEND_FLAGS", value=override)
    version = nccl_version()
    if version and version >= (2, 29, 7):
        return NCCL_SUSPEND_MEM
    return 0


def nvls_in_use():
    """Whether NVLS is plausibly active. Conservative: unknown counts as yes.

    Being wrong in the safe direction costs a slower resume. Being wrong in the
    other direction costs the job.
    """
    disabled = os.environ.get("NCCL_NVLS_ENABLE", "").strip()
    if disabled in ("0", "false", "False"):
        return False
    return True


class Strategy:
    name = "abstract"

    def teardown(self):
        raise NotImplementedError

    def rebuild(self, ctx):
        raise NotImplementedError

    def describe(self):
        return {"strategy": self.name}


class DestroyRebuild(Strategy):
    """Correct everywhere. The default, and the fallback for everything else."""

    name = "destroy-rebuild"

    def teardown(self):
        from torchckpt import torch_backend

        return torch_backend.default_teardown()

    def rebuild(self, ctx):
        from torchckpt import torch_backend

        return torch_backend.default_rebuild(ctx)


class SuspendResume(Strategy):
    """Only selected when the runtime covers net and NVLS state as well as memory."""

    name = "suspend-resume"

    def __init__(self, flags=REQUIRED_FOR_CHECKPOINT):
        self.flags = flags
        self._suspended = False

    def teardown(self):
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return {"skipped": "no process group"}
        torch.cuda.synchronize()
        group = dist.distributed_c10d._get_default_group()
        suspend = _find_binding(group, "suspend")
        if suspend is None:
            raise RuntimeError(
                "ncclCommSuspend is not reachable from this torch build; "
                "select destroy-rebuild"
            )
        suspend(self.flags)
        self._suspended = True
        torch.cuda.empty_cache()
        return {"suspended": True, "flags": hex(self.flags)}

    def rebuild(self, ctx):
        import torch.distributed as dist

        if not self._suspended:
            from torchckpt import torch_backend

            return torch_backend.default_rebuild(ctx)
        group = dist.distributed_c10d._get_default_group()
        resume = _find_binding(group, "resume")
        if resume is None:
            raise RuntimeError("ncclCommResume is not reachable; job cannot resume")
        resume()
        self._suspended = False
        return {"resumed": True}


def _find_binding(group, verb):
    """Locate the suspend/resume entry point on whatever torch exposes.

    Torch has moved these around; probing rather than hard-coding one path keeps
    the seam working across versions and makes the failure explicit when it is
    genuinely absent.
    """
    backend = getattr(group, "_get_backend", lambda *_a: None)(None) or group
    for attr in (f"comm_{verb}", verb, f"_{verb}", f"nccl_comm_{verb}"):
        fn = getattr(backend, attr, None)
        if callable(fn):
            return fn
    return None


def select(prefer=None, allow_suspend=None):
    """Choose a strategy, and say why.

    prefer="suspend" asks for the fast path; it is granted only if the runtime
    advertises the flags that make it safe.
    """
    prefer = prefer or os.environ.get("MNCR_NCCL_STRATEGY", "auto")
    allow = (
        allow_suspend
        if allow_suspend is not None
        else os.environ.get("MNCR_ALLOW_SUSPEND", "").lower() in ("1", "true", "yes")
    )
    flags = supported_suspend_flags()

    if prefer in ("suspend", "auto") and (allow or prefer == "suspend"):
        missing = REQUIRED_FOR_CHECKPOINT & ~flags
        if missing == 0:
            _LOG.info("selected suspend-resume", flags=hex(flags))
            return SuspendResume(REQUIRED_FOR_CHECKPOINT)
        reason = []
        if missing & NCCL_SUSPEND_NET:
            reason.append("network state not covered")
        if missing & NCCL_SUSPEND_NVLS:
            reason.append("NVLS state not covered (see NCCL #2337)")
        _LOG.warn(
            "suspend-resume unavailable, falling back",
            reason="; ".join(reason),
            flags=hex(flags),
            nvls_in_use=nvls_in_use(),
        )
    return DestroyRebuild()


def timed(strategy, ctx, iterations=1):
    """Measure one strategy's teardown+rebuild cost."""
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        strategy.teardown()
        t1 = time.perf_counter()
        strategy.rebuild(ctx)
        t2 = time.perf_counter()
        samples.append({"teardown_s": t1 - t0, "rebuild_s": t2 - t1, "total_s": t2 - t0})
    return {
        "strategy": strategy.name,
        "iterations": iterations,
        "samples": samples,
        "mean_total_s": sum(s["total_s"] for s in samples) / max(len(samples), 1),
    }
