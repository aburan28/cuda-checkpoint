#!/usr/bin/env python3
"""A rank that exercises the real torchckpt code path without a GPU.

This is not a mock of the library - it imports and drives the same api.py the
production rank uses. What is faked is only the work: a "step" is a sleep, and
the resources torn down are recorded rather than real. That keeps the state
machine, the control channel, the vote path and the resume path under test.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        "--cuda",
        action="store_true",
        help="hold real device memory and verify it after every resume; "
             "required when running against the real driver, which will not "
             "checkpoint a process that has no CUDA state",
    )
    args = ap.parse_args()

    events = []
    device_state = {}

    def record(name, **fields):
        events.append({"at": time.time(), "event": name, **fields})
        with open(args.progress, "w") as fh:
            json.dump(
                {
                    "rank": args.rank,
                    "pid": os.getpid(),
                    "step": torchckpt.status().get("step"),
                    "state": torchckpt.status().get("state"),
                    "events": events,
                },
                fh,
            )

    if args.cuda:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--cuda requested but no CUDA device is available")
        device_state["expected"] = torch.arange(
            1 << 20, dtype=torch.int64, device="cuda"
        )
        device_state["buffer"] = device_state["expected"].clone()
        torch.cuda.synchronize()

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
        record("quiesced")

    @torchckpt.on_resume
    def rebuild(ctx):
        intact = None
        if device_state:
            import torch

            intact = bool(
                (device_state["buffer"] == device_state["expected"]).all().item()
            )
            torch.cuda.synchronize()
        record(
            "resumed",
            epoch=ctx.epoch_id,
            restored=ctx.restored,
            aborted=ctx.aborted,
            world_size=ctx.world_size,
            device_memory_intact=intact,
        )

    torchckpt.graphs.register("decode-graph", lambda: record("graph_recaptured"))

    for _ in range(args.steps):
        with torchckpt.safe_point():
            time.sleep(args.step_seconds)
        if torchckpt.status()["step"] % 25 == 0:
            record("step")

    record("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
