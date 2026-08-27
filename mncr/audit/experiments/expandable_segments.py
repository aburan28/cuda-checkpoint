#!/usr/bin/env python3
"""P0's decisive experiment: does the checkpoint path tolerate VMM allocations?

PyTorch's expandable segments allocate through cuMemCreate/cuMemMap - the same
VMM family that cuMemExportToShareableHandle belongs to. The documented
limitation names the *export*, not the allocation. Which of these is true
decides a fleet-wide policy:

  (a) the driver rejects any process holding VMM allocations
      -> expandable_segments must be off everywhere, and P1 must prove the
         caching allocator holds none at the lock

  (b) only exported handles are rejected
      -> expandable_segments can stay on, and P1 only has to destroy
         communicators

Do not design around either answer until this has been run on real hardware.

Each cell runs in a fresh subprocess because the allocator config must be set
before CUDA initialises. Results are JSON on stdout.

    python3 expandable_segments.py                 # run the whole matrix
    python3 expandable_segments.py --cell 3        # run one cell (internal)
"""

import argparse
import json
import os
import subprocess
import sys

CELLS = [
    # name,                      expandable, nccl_cumem, build_pg, teardown
    ("baseline",                 False, None,  False, False),
    ("expandable",               True,  None,  False, False),
    ("expandable+drained",       True,  None,  False, True),
    ("nccl-default",             False, None,  True,  False),
    ("nccl-default+teardown",    False, None,  True,  True),
    ("nccl-cumem-off",           False, "0",   True,  False),
    ("expandable+nccl",          True,  None,  True,  False),
    ("expandable+nccl+teardown", True,  None,  True,  True),
]


def run_cell(index):
    """Executed in the child. Sets up one condition and checkpoints itself."""
    name, expandable, nccl_cumem, build_pg, teardown = CELLS[index]
    result = {"cell": name, "index": index}

    try:
        import torch  # noqa: F401
    except ImportError:
        result["skipped"] = "torch not importable"
        return result
    if not torch.cuda.is_available():
        result["skipped"] = "no CUDA device"
        return result

    try:
        buf = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")  # 256 MiB
        buf.fill_(7)
        result["alloc_ok"] = True
    except Exception as exc:
        result["alloc_ok"] = False
        result["alloc_error"] = str(exc)
        return result

    pg_built = False
    if build_pg:
        try:
            import torch.distributed as dist

            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29513")
            dist.init_process_group("nccl", rank=0, world_size=1)
            dist.all_reduce(torch.ones(16, device="cuda"))
            torch.cuda.synchronize()
            pg_built = True
        except Exception as exc:
            result["pg_error"] = str(exc)
    result["pg_built"] = pg_built

    if teardown:
        try:
            if pg_built:
                import torch.distributed as dist

                dist.destroy_process_group()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            result["teardown_ok"] = True
        except Exception as exc:
            result["teardown_ok"] = False
            result["teardown_error"] = str(exc)

    # Checkpoint this process from inside it. cuda-checkpoint does not suspend
    # CPU threads, so the call returns to us; this is the same shape as the
    # r580-migration-cli demo.
    pid = str(os.getpid())
    for action in ("lock", "checkpoint", "restore", "unlock"):
        proc = subprocess.run(
            ["cuda-checkpoint", "--action", action, "--pid", pid]
            + (["--timeout", "60000"] if action == "lock" else []),
            capture_output=True,
            text=True,
        )
        result[f"{action}_rc"] = proc.returncode
        if proc.returncode != 0:
            result[f"{action}_stderr"] = (proc.stderr or proc.stdout).strip()[:400]
            result["verdict"] = f"failed at {action}"
            return result

    try:
        result["data_intact"] = bool((buf == 7).all().item())
    except Exception as exc:
        result["data_intact"] = False
        result["verify_error"] = str(exc)

    result["verdict"] = "ok"
    return result


def child_env(index):
    _, expandable, nccl_cumem, _, _ = CELLS[index]
    env = dict(os.environ)
    if expandable:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if nccl_cumem is not None:
        env["NCCL_CUMEM_ENABLE"] = nccl_cumem
    else:
        env.pop("NCCL_CUMEM_ENABLE", None)
    # Capture what the process actually allocated, if the interposer is built.
    interposer = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "interpose",
        "cuda_audit.so",
    )
    if os.path.exists(interposer):
        env["LD_PRELOAD"] = interposer
        env["MNCR_AUDIT_OUT"] = f"/tmp/mncr-audit-cell{index}"
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--only", default=None, help="substring filter on cell name")
    args = ap.parse_args()

    if args.cell is not None:
        print(json.dumps(run_cell(args.cell)))
        return 0

    results = []
    for i, cell in enumerate(CELLS):
        if args.only and args.only not in cell[0]:
            continue
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cell", str(i)],
            capture_output=True,
            text=True,
            env=child_env(i),
        )
        line = (proc.stdout or "").strip().splitlines()
        try:
            results.append(json.loads(line[-1]) if line else {"cell": cell[0]})
        except json.JSONDecodeError:
            results.append(
                {"cell": cell[0], "crashed": True, "stderr": proc.stderr[-400:]}
            )

    print(json.dumps({"matrix": results}, indent=2))

    verdicts = {r["cell"]: r.get("verdict") or r.get("skipped") for r in results}
    sys.stderr.write("\nSUMMARY\n")
    for name, verdict in verdicts.items():
        sys.stderr.write(f"  {name:<28} {verdict}\n")
    sys.stderr.write(
        "\nRead: if 'expandable' fails but 'expandable+drained' passes, the "
        "allocator must be drained before the lock.\nIf both fail, "
        "expandable_segments must be disabled fleet-wide.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
