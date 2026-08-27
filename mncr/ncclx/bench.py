#!/usr/bin/env python3
"""Benchmark: is a vendored NCCL worth carrying?

Answers one question with a number - how much resume latency does
suspend/resume save over destroy/rebuild at this TP width. If the answer is
small, the patch is not worth maintaining out of tree; if it is seconds, it is
the difference between viable and not for the cold-start use case.

Requires a GPU and an initialised process group. Reports honestly when it
cannot run.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncclx import seam  # noqa: E402
from torchckpt.api import ResumeContext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--tensor-mib", type=int, default=256)
    args = ap.parse_args()

    report = {
        "nccl_version": seam.nccl_version(),
        "suspend_flags": hex(seam.supported_suspend_flags()),
        "nvls_in_use": seam.nvls_in_use(),
    }

    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        report["skipped"] = "torch not importable"
        print(json.dumps(report, indent=2))
        return 0

    if not torch.cuda.is_available():
        report["skipped"] = "no CUDA device"
        print(json.dumps(report, indent=2))
        return 0

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29515")
        dist.init_process_group(
            "nccl",
            rank=int(os.environ.get("RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
        )

    rank = dist.get_rank()
    world = dist.get_world_size()
    payload = torch.ones(args.tensor_mib << 18, device="cuda")
    dist.all_reduce(payload)
    torch.cuda.synchronize()

    ctx = ResumeContext(
        epoch_id="bench",
        rank=rank,
        world_size=world,
        restored=False,
        aborted=False,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )

    results = []
    for strategy in (seam.DestroyRebuild(), seam.select(prefer="suspend", allow_suspend=True)):
        if any(r["strategy"] == strategy.name for r in results):
            continue  # select() fell back to the one already measured
        try:
            results.append(seam.timed(strategy, ctx, args.iterations))
        except Exception as exc:
            results.append({"strategy": strategy.name, "error": str(exc)})

    report["world_size"] = world
    report["results"] = results
    if len(results) == 2 and all("mean_total_s" in r for r in results):
        base, fast = results[0]["mean_total_s"], results[1]["mean_total_s"]
        report["speedup"] = round(base / max(fast, 1e-9), 2)
        report["saved_seconds_per_epoch"] = round(base - fast, 4)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
