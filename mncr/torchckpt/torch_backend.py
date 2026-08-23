"""PyTorch-specific teardown and rebuild.

Imports of torch are deferred and guarded so this package can be imported, and
most of it tested, on a host with no torch and no GPU.

The interesting function here is agree_on_target_step. Ranks are coupled by
collectives but not perfectly aligned: when a checkpoint request lands, one rank
may have finished step 100 while another is midway through it. Resuming a job
whose ranks are on different steps is a correctness bug, not a performance one.
So before tearing anything down, the ranks use the communicator that is about to
be destroyed to agree on a step to stop at, then keep training until they all
reach it.
"""

from mncr import log

_LOG = log.get("torchckpt.torch")


def _torch():
    import torch

    return torch


def available():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_importable():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def distributed_ready():
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def agree_on_target_step(local_step, lookahead=1):
    """Return the step every rank will stop at.

    max(local_step) + lookahead, agreed over the live communicator. With no
    communicator there is nobody to disagree with, so the local answer stands.
    """
    if not distributed_ready():
        return local_step + lookahead
    try:
        import torch
        import torch.distributed as dist

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor([int(local_step)], dtype=torch.int64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        if device == "cuda":
            torch.cuda.synchronize()
        target = int(tensor.item()) + lookahead
        _LOG.info("agreed target step", local=local_step, target=target)
        return target
    except Exception as exc:
        # Agreement failed, which usually means the communicator is already
        # broken. Fall back to the local step rather than hanging: the
        # coordinator's vote timeout is the real backstop.
        _LOG.warn("step agreement failed", error=str(exc))
        return local_step + lookahead


def default_teardown():
    """Release everything the checkpoint path cannot handle.

    Order matters. Synchronise first so no kernel is still writing into memory
    that is about to be freed; destroy communicators next, which is what
    actually releases the cuMem allocations, the NVLS multicast groups and the
    verbs fds; only then return cached blocks to the driver.
    """
    if not torch_importable():
        return {"skipped": "torch not importable"}
    torch = _torch()
    report = {}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        report["synchronized"] = True

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            report["process_group_destroyed"] = True
    except Exception as exc:
        report["process_group_error"] = str(exc)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            report["cache_emptied"] = True
            stats = torch.cuda.memory_stats()
            report["reserved_bytes_after"] = stats.get("reserved_bytes.all.current", 0)
        except Exception as exc:
            report["empty_cache_error"] = str(exc)

    _LOG.info("teardown complete", **report)
    return report


def default_rebuild(ctx):
    """Re-establish the process group from a fresh rendezvous.

    The store is rebuilt rather than reused because on a cross-node restore the
    peers are at different addresses than they were at checkpoint time.
    """
    if not torch_importable():
        return {"skipped": "torch not importable"}
    try:
        import torch.distributed as dist
    except Exception as exc:
        return {"error": str(exc)}

    if dist.is_initialized():
        return {"already_initialized": True}

    kwargs = {
        "backend": ctx.backend,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
    }
    if ctx.init_method:
        kwargs["init_method"] = ctx.init_method
    dist.init_process_group(**kwargs)
    _LOG.info(
        "process group rebuilt",
        rank=ctx.rank,
        world_size=ctx.world_size,
        init_method=ctx.init_method,
    )
    return {"process_group": "rebuilt"}


def local_gpu_uuids():
    """UUIDs of the devices this rank can see, for the restore device map."""
    if not available():
        return []
    torch = _torch()
    out = []
    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)
            uuid = getattr(props, "uuid", None)
            out.append(f"GPU-{uuid}" if uuid else f"index-{index}")
        except Exception:
            out.append(f"index-{index}")
    return out


def memory_snapshot():
    if not available():
        return {}
    torch = _torch()
    stats = torch.cuda.memory_stats()
    return {
        "allocated": stats.get("allocated_bytes.all.current", 0),
        "reserved": stats.get("reserved_bytes.all.current", 0),
    }
