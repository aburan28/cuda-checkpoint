#!/usr/bin/env python3
"""Call agree_on_target_step from inside a real process group.

Each rank arrives with a deliberately different local step, which is the
situation the function exists for: ranks are coupled by collectives but not
aligned, and a request can land while one rank has finished step 100 and
another is midway through it. All ranks must leave with the same answer.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torchckpt import torch_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--init-method", required=True)
    ap.add_argument("--local-step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookahead", type=int, default=1)
    args = ap.parse_args()

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=args.init_method, rank=args.rank, world_size=args.world_size
    )
    ready = torch_backend.distributed_ready()
    target = torch_backend.agree_on_target_step(args.local_step, lookahead=args.lookahead)

    # And prove the teardown the rank library performs actually releases it.
    report = torch_backend.default_teardown()

    with open(args.out, "w") as fh:
        json.dump(
            {
                "rank": args.rank,
                "local_step": args.local_step,
                "target": target,
                "distributed_ready": ready,
                "teardown": report,
            },
            fh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
