#!/usr/bin/env python3
"""PyTorch variant of the smoke target, for nodes without nvcc.

Same file protocol as smoke_target.cu. Uses torch because that is what the
production ranks use, at the cost of bringing an allocator into a test whose
subject is the driver - prefer the CUDA C target when nvcc is available.
"""

import argparse
import os
import time

SEED = 0x5A5A0001


def write_state(path, text):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def read_command(path):
    cmd_path = f"{path}.cmd"
    try:
        with open(cmd_path) as fh:
            command = fh.read().strip()
        os.unlink(cmd_path)
        return command
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--elements", type=int, default=16 << 20)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        write_state(args.state, "error no CUDA device\n")
        return 1

    index = torch.arange(args.elements, dtype=torch.int64, device="cuda")
    expected = (SEED + index * 2654435761) % (1 << 32)
    buffer = expected.to(torch.int64).clone()
    torch.cuda.synchronize()

    write_state(args.state, f"ready {os.getpid()} {args.elements}\n")

    while True:
        command = read_command(args.state)
        if command == "exit":
            break
        if command == "verify":
            bad = int((buffer != expected).sum().item())
            total = int(buffer.sum().item())
            write_state(args.state, f"sum {total} bad {bad} pid {os.getpid()}\n")
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
