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

import gc
import os

from mncr import log

_LOG = log.get("torchckpt.torch")


def _torch():
    import torch

    return torch


def nccl_version():
    """The NCCL torch was built against, as "2.29.7", or None without CUDA."""
    try:
        import torch

        version = torch.cuda.nccl.version()
    except Exception:
        return None
    if isinstance(version, (tuple, list)):
        return ".".join(str(part) for part in version)
    return str(version)


def active_devices(torch):
    """Device ordinals whose primary context this process has actually created.

    Asked of the driver rather than of torch, because every torch query that
    could answer it - synchronize(i), memory_stats(i) - creates the context
    it is asking about. A rank on an 8-GPU node that touched one device must
    not leave the teardown holding all eight.
    """
    if not torch.cuda.is_initialized():
        return []
    count = torch.cuda.device_count()
    try:
        import ctypes

        lib = ctypes.CDLL("libcuda.so.1")
        active = []
        for index in range(count):
            dev = ctypes.c_int()
            if lib.cuDeviceGet(ctypes.byref(dev), index) != 0:
                continue
            flags, state = ctypes.c_uint(), ctypes.c_int()
            if lib.cuDevicePrimaryCtxGetState(dev, ctypes.byref(flags), ctypes.byref(state)) != 0:
                continue
            if state.value:
                active.append(index)
        return active
    except (OSError, AttributeError):
        # No driver library to ask (or not Linux): the current device is the
        # one we know about.
        return [torch.cuda.current_device()]


_TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
    "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
    "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING",
}


def _tcp_table():
    """inode -> (local, remote, state) for this process's network namespace."""
    table = {}
    for name in ("tcp", "tcp6"):
        try:
            with open(f"/proc/self/net/{name}") as fh:
                next(fh)
                for line in fh:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    table[parts[9]] = (
                        _addr(parts[1]), _addr(parts[2]), _TCP_STATES.get(parts[3], parts[3])
                    )
        except OSError:
            continue
    return table


def _addr(hex_addr):
    host, _, port = hex_addr.partition(":")
    if len(host) == 8:
        octets = [str(int(host[i:i + 2], 16)) for i in (6, 4, 2, 0)]
        return ".".join(octets) + f":{int(port, 16)}"
    return f"[{host}]:{int(port, 16)}"


def socket_inventory():
    """The sockets this process holds, described. Linux only; None elsewhere.

    After teardown a rank should hold none: the communicator, the store and
    the bootstrap connections are all gone. Anything left is something the
    teardown did not reach, and CRIU will have to bind its address again on
    restore - which fails when that port is taken, and past the commit point.
    """
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return None
    table = _tcp_table()
    out = []
    for name in names:
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if not target.startswith("socket:"):
            continue
        inode = target[8:-1]
        local, remote, state = table.get(inode, ("?", "?", "non-tcp"))
        out.append({"fd": int(name), "local": local, "remote": remote, "state": state})
    return sorted(out, key=lambda s: s["fd"])


def open_sockets():
    inventory = socket_inventory()
    return None if inventory is None else len(inventory)


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


def _collective_device():
    import torch
    import torch.distributed as dist

    try:
        backend = str(dist.get_backend())
    except Exception:
        backend = ""
    return "cuda" if backend == "nccl" and torch.cuda.is_available() else "cpu"


def agree_on_request(epoch_int, lookahead):
    """What any rank has seen, shared with every rank. (0, 0) if nobody has.

    One tiny all_reduce, MAX over [epoch, lookahead], once per safe point while
    a communicator is live. The epoch travels in the collective rather than as
    a flag so a rank whose own request file is still on its way can act on
    exactly what its peers are acting on, at the same step. On NCCL this is a
    device sync per step, which is the price of a consistent cut; a loop that
    cannot afford it can set coupled_polling=False on the runtime and accept
    that ranks may enter an epoch on different steps.
    """
    if not distributed_ready():
        return int(epoch_int), int(lookahead)
    try:
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([int(epoch_int), int(lookahead)], dtype=torch.int64,
                              device=_collective_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        seen, ahead = (int(v) for v in tensor.tolist())
        return seen, ahead
    except Exception as exc:
        _LOG.warn("coupled poll failed", error=str(exc))
        return int(epoch_int), int(lookahead)


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

        device = _collective_device()
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
    # is_initialized, not is_available: a rank that never used CUDA has
    # nothing to synchronise, and asking is_available would initialise the
    # driver in a process that was clean without it.
    cuda = torch.cuda.is_initialized()

    if cuda:
        # Every device this process touched, not only the current one: a rank
        # driving several GPUs has work in flight on all of them.
        devices = active_devices(torch)
        for index in devices:
            torch.cuda.synchronize(index)
        report["synchronized"] = devices

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            report["backend"] = str(dist.get_backend())
            if report["backend"] == "nccl":
                report["nccl_version"] = nccl_version()
            # No group argument, deliberately: this destroys every group,
            # sub-groups included. A tensor-parallel sub-group left alive
            # keeps its communicator, and with it the peer memory it imported,
            # which is the one thing the driver cannot restore.
            dist.destroy_process_group()
            report["process_group_destroyed"] = True
    except Exception as exc:
        report["process_group_error"] = str(exc)

    # destroy_process_group drops the group's own references. Whatever Python
    # still holds - the TCPStore behind the default group, a communicator
    # wrapper reachable from a cycle - is released here. NCCL frees its buffers
    # when the last reference to the communicator goes, so this is part of the
    # teardown rather than housekeeping after it.
    gc.collect()

    if cuda:
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            report["cache_emptied"] = True
            stats = torch.cuda.memory_stats()
            report["reserved_bytes_after"] = stats.get("reserved_bytes.all.current", 0)
        except Exception as exc:
            report["empty_cache_error"] = str(exc)

    sockets = socket_inventory()
    if sockets is not None:
        report["open_sockets"] = len(sockets)
        report["sockets"] = sockets

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
    else:
        # env:// - MASTER_ADDR as the launcher set it. Right on the node the
        # job started on, wrong after a migration; the coordinator issues a
        # fresh address per epoch precisely so this branch is not taken.
        _LOG.warn(
            "no init_method in the resume token; falling back to env://",
            master_addr=os.environ.get("MASTER_ADDR"),
        )
    dist.init_process_group(**kwargs)
    report = {"process_group": "rebuilt"}

    # Build the communicator now rather than at the next training collective.
    # A rebuild that cannot reach its peers should fail here, inside the resume
    # path where it is reported against the epoch, not three lines into the
    # user's training step.
    if (ctx.extra or {}).get("warm_up", True):
        report["warm_up"] = warm_up(ctx.backend)

    _LOG.info(
        "process group rebuilt",
        rank=ctx.rank,
        world_size=ctx.world_size,
        init_method=ctx.init_method,
        **report,
    )
    return report


def warm_up(backend):
    """One all_reduce over the fresh group. True if every rank was counted."""
    torch = _torch()
    import torch.distributed as dist

    device = "cuda" if backend == "nccl" and torch.cuda.is_available() else "cpu"
    probe = torch.ones(1, device=device)
    dist.all_reduce(probe)
    if device == "cuda":
        torch.cuda.synchronize()
    return int(probe.item()) == dist.get_world_size()


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
