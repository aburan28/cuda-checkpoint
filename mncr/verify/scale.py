#!/usr/bin/env python3
"""Scale ladder: does checkpoint wall clock track ranks-per-node or job size?

The design claims node-level parallelism makes job size free - the driver
serialises within a node, nothing serialises across them. This measures it.

Two series:
    width   fixed ranks-per-node, growing node count. Time should stay flat.
    depth   fixed node count, growing ranks-per-node. Time should grow.

Fake driver and fake CRIU, so the numbers are protocol overhead only. Real
device-memory copies and image writes dominate on hardware; what this validates
is that the coordination layer does not add its own scaling term.
"""

import argparse
import json
import statistics
import sys
import time

from .sim import SimCluster

PHASES = ("preparing", "prepared", "locked", "checkpointed", "dumped")


def phase_durations(epoch):
    """Seconds spent in each phase, from the epoch's own history."""
    history = epoch.get("history", [])
    marks = [(h["to"], h["at"]) for h in history]
    out = {}
    for index, (phase, at) in enumerate(marks[:-1]):
        out[phase] = round(marks[index + 1][1] - at, 4)
    return out


def run_point(nodes, ranks_per_node, repeats=3, call_latency=0.0):
    sim = SimCluster(
        nodes=nodes, ranks_per_node=ranks_per_node, step_seconds=0.002,
        call_latency=call_latency,
    ).start()
    try:
        sim.launch_ranks()
        sim.wait_registered()
        samples, breakdown = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            result = sim.coord.checkpoint(sim.job_id, mode="continue")
            samples.append(time.perf_counter() - started)
            epoch = sim.coord.store.get(result["epoch_id"])
            breakdown.append(phase_durations(epoch))
            if not sim.wait_for_event("resumed", timeout=20):
                raise RuntimeError("ranks did not resume between samples")
        mean_phases = {
            phase: round(
                statistics.mean(b.get(phase, 0.0) for b in breakdown), 4
            )
            for phase in PHASES
        }
        return {
            "nodes": nodes,
            "ranks_per_node": ranks_per_node,
            "total_ranks": nodes * ranks_per_node,
            "mean_s": round(statistics.mean(samples), 4),
            "min_s": round(min(samples), 4),
            "max_s": round(max(samples), 4),
            "phases": mean_phases,
        }
    finally:
        sim.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=4)
    ap.add_argument("--max-ranks", type=int, default=4)
    ap.add_argument(
        "--call-latency",
        type=float,
        default=0.05,
        help="simulated seconds per driver call; stands in for real lock and "
             "checkpoint cost so the sequential term is measurable",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    width, depth = [], []
    nodes = 1
    while nodes <= args.max_nodes:
        width.append(run_point(nodes, 1, args.repeats, args.call_latency))
        nodes *= 2
    ranks = 1
    while ranks <= args.max_ranks:
        depth.append(run_point(1, ranks, args.repeats, args.call_latency))
        ranks *= 2

    report = {"width": width, "depth": depth}
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    def table(title, rows, expectation):
        print(f"\n{title}   ({expectation})")
        print(f"{'nodes':>6} {'r/node':>7} {'ranks':>6} {'mean s':>9} {'min':>8} {'max':>8}")
        print("-" * 50)
        for row in rows:
            print(
                f"{row['nodes']:>6} {row['ranks_per_node']:>7} {row['total_ranks']:>6} "
                f"{row['mean_s']:>9.4f} {row['min_s']:>8.4f} {row['max_s']:>8.4f}"
            )

    table("WIDTH - more nodes, 1 rank each", width, "expect roughly flat")
    table("DEPTH - one node, more ranks", depth, "expect growth")

    if len(width) > 1:
        growth = width[-1]["mean_s"] / max(width[0]["mean_s"], 1e-9)
        node_growth = width[-1]["nodes"] / width[0]["nodes"]
        print(
            f"\nwidth: {node_growth:.0f}x the nodes cost {growth:.2f}x the time "
            f"({'flat enough' if growth < node_growth / 2 else 'scaling with job size'})"
        )
    if len(depth) > 1:
        growth = depth[-1]["mean_s"] / max(depth[0]["mean_s"], 1e-9)
        rank_growth = depth[-1]["ranks_per_node"] / depth[0]["ranks_per_node"]
        print(
            f"depth: {rank_growth:.0f}x the ranks per node cost {growth:.2f}x the time"
        )
    print(
        f"\nFake driver and CRIU, {args.call_latency}s simulated per driver call. "
        f"Real device-memory copies dominate on hardware."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
