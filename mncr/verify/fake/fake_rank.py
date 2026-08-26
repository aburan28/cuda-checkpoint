#!/usr/bin/env python3
"""A rank that exercises the real torchckpt code path.

This is not a mock of the library - it imports and drives the same api.py the
production rank uses. What is faked is only the work: a "step" is a sleep, and
the resources torn down are recorded rather than real. That keeps the state
machine, the control channel, the vote path and the resume path under test.

With --backend it stops being fake in the way that matters. gloo gives real
collectives on CPU with no GPU needed; nccl gives the production communicator
on real devices, so what is torn down before the checkpoint and rebuilt after
it is exactly what a training job holds.
"""

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import torchckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--agent-addr", required=True)
    ap.add_argument("--control-root", required=True)
    ap.add_argument("--progress", required=True, help="where to write rank state")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--step-seconds", type=float, default=0.01)
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--dirty", action="store_true", help="fail the clean check")
    ap.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default=None,
        help="join a real process group and run a collective every step, so "
             "step agreement and communicator teardown/rebuild are exercised "
             "for real rather than falling through the no-process-group path",
    )
    ap.add_argument("--gloo", action="store_true", help="alias for --backend gloo")
    ap.add_argument("--init-method", default=None, help="rendezvous for --backend")
    ap.add_argument(
        "--device", type=int, default=None,
        help="CUDA device for this rank; default rank modulo the visible count",
    )
    ap.add_argument(
        "--cuda",
        action="store_true",
        help="hold real device memory and verify it after every resume; "
             "required when running against the real driver, which will not "
             "checkpoint a process that has no CUDA state",
    )
    args = ap.parse_args()
    backend = args.backend or ("gloo" if args.gloo else None)

    events = []
    device_state = {}
    collective = {}
    facts = {"rank": args.rank, "pid": os.getpid(), "backend": backend}

    def record(name, **fields):
        events.append({"at": time.time(), "event": name, **fields})
        payload = dict(facts)
        payload.update(
            {
                "pid": os.getpid(),
                "step": torchckpt.status().get("step"),
                "state": torchckpt.status().get("state"),
                "events": events,
            }
        )
        tmp = f"{args.progress}.tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, default=str)
        os.replace(tmp, args.progress)

    try:
        return run(args, backend, events, device_state, collective, facts, record)
    except BaseException as exc:  # noqa: BLE001 - the harness reads this, not stderr
        record("crashed", error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc()[-3000:])
        raise


def run(args, backend, events, device_state, collective, facts, record):
    device_index = None
    torch_cuda = False
    if backend == "nccl" or args.cuda:
        try:
            import torch

            torch_cuda = bool(torch.cuda.is_available())
        except ImportError:
            torch_cuda = False
        if backend == "nccl" and not torch_cuda:
            raise RuntimeError("--backend nccl needs torch with a visible CUDA device")
        if torch_cuda:
            device_index = (
                args.device if args.device is not None
                else args.rank % max(1, torch.cuda.device_count())
            )
            torch.cuda.set_device(device_index)
            props = torch.cuda.get_device_properties(device_index)
            facts["device"] = device_index
            facts["gpu"] = props.name
            facts["gpu_uuid"] = f"GPU-{props.uuid}" if getattr(props, "uuid", None) else None

    if backend:
        import torch
        import torch.distributed as dist

        dist.init_process_group(
            backend,
            init_method=args.init_method,
            rank=args.rank,
            world_size=args.world_size,
        )
        collective["torch"] = torch
        collective["dist"] = dist
        collective["device"] = "cuda" if backend == "nccl" else "cpu"
        if backend == "nccl":
            from torchckpt import torch_backend

            facts["nccl_version"] = torch_backend.nccl_version()

    if args.cuda:
        # Torch when it is there, the driver API when it is not. Either way the
        # rank is a genuine CUDA process, which is what cuda-checkpoint needs.
        if torch_cuda:
            import torch

            device_state["kind"] = "torch"
            device_state["expected"] = torch.arange(
                1 << 20, dtype=torch.int64, device="cuda"
            )
            device_state["buffer"] = device_state["expected"].clone()
            torch.cuda.synchronize()
        else:
            from cuda_ctypes import DeviceBuffer

            device_state["kind"] = "driver-api"
            device_state["buffer"] = DeviceBuffer(8 << 20)

    torchckpt.init(
        job_id=args.job_id,
        rank=args.rank,
        world_size=args.world_size,
        agent_addr=args.agent_addr,
        control_root=args.control_root,
        auto_teardown=True,
        strict_clean=True,
        proc_root=args.proc_root,
    )
    record("initialized")

    @torchckpt.on_quiesce
    def teardown():
        if args.dirty:
            raise RuntimeError("simulated teardown failure: communicator still live")
        # The step at which this rank stopped. Every rank must report the same
        # one, or the restored job has ranks on different steps.
        record("quiesced", step=torchckpt.status()["step"])

    @torchckpt.on_resume
    def rebuild(ctx):
        intact = None
        if device_state.get("kind") == "torch":
            import torch

            intact = bool(
                (device_state["buffer"] == device_state["expected"]).all().item()
            )
            torch.cuda.synchronize()
        elif device_state.get("kind") == "driver-api":
            intact = bool(device_state["buffer"].intact())
        rejoined = None
        if collective:
            dist = collective["dist"]
            torch = collective["torch"]
            if dist.is_initialized():
                probe = torch.ones(4, device=collective["device"]) * (ctx.rank + 1)
                dist.all_reduce(probe)
                expected = ctx.world_size * (ctx.world_size + 1) / 2
                rejoined = bool(
                    torch.allclose(probe.cpu(), torch.full((4,), expected))
                )
        runtime = torchckpt.runtime()
        record(
            "resumed",
            rejoined=rejoined,
            epoch=ctx.epoch_id,
            restored=ctx.restored,
            aborted=ctx.aborted,
            world_size=ctx.world_size,
            rank=ctx.rank,
            init_method=ctx.init_method,
            device_memory_intact=intact,
            teardown=runtime.last_report.get("default_teardown"),
            rebuild=runtime.last_rebuild,
            hostname=os.uname().nodename,
        )

    torchckpt.graphs.register("decode-graph", lambda: record("graph_recaptured"))

    for _ in range(args.steps):
        with torchckpt.safe_point():
            if collective:
                dist = collective["dist"]
                torch = collective["torch"]
                # A collective every step is what couples the ranks: a rank that
                # stops issuing them blocks every peer at the next one.
                dist.all_reduce(torch.ones(4, device=collective["device"]))
            time.sleep(args.step_seconds)
        if torchckpt.status()["step"] % 25 == 0:
            record("step")

    record("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
