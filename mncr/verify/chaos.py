#!/usr/bin/env python3
"""Chaos matrix: inject one fault per run, assert the invariant for that phase.

There are exactly two invariants, and which one applies depends on which side of
the commit point the fault lands:

    before  the job survives - every rank alive, every rank unlocked, the epoch
            settled as ABORTED, and an AbortableError raised
    after   the failure is reported as terminal - the epoch settled as FAILED
            and a TerminalError raised naming the fallback

A run that "succeeds" when it should have failed is the worst outcome here, so
every case asserts that something was raised.
"""

import argparse
import json
import sys
import time

from mncr.errors import AbortableError, TerminalError
from mncr.proto import Phase

from .sim import SimCluster

BEFORE, AFTER = "before-commit", "after-commit"


def _fail_driver(action, node="node-1"):
    def inject(sim):
        agent = sim.agents[node]
        pid = agent.local_ranks(sim.job_id)[0]["host_pid"]
        agent.driver.fail_next(action, pid, f"injected {action} failure")
    return inject


def _fail_criu(op, node="node-0"):
    def inject(sim):
        sim.agents[node].criu.fail_next(op)
    return inject


def _kill_rank(index=0):
    def inject(sim):
        _rank, _node, proc = sim.procs[index]
        proc.kill()
        proc.wait(timeout=5)
        sim.cfg.quiesce_timeout = 3.0
    return inject


def _unreachable_agent(node="node-1"):
    def inject(sim):
        sim.coord.pool.agents[node] = "tcp:127.0.0.1:1"   # nothing listening
    return inject


CASES = [
    ("dirty-rank",        BEFORE, None,                        {"dirty_ranks": [0]}),
    ("lock-failure",      BEFORE, _fail_driver("lock"),        {}),
    ("rank-killed",       BEFORE, _kill_rank(),                {}),
    ("agent-unreachable", BEFORE, _unreachable_agent(),        {}),
    ("checkpoint-failure", AFTER, _fail_driver("checkpoint"),  {}),
    ("dump-failure",       AFTER, _fail_criu("dump"),          {}),
    ("resume-failure",     AFTER, _fail_driver("restore"),     {}),
    ("unlock-failure",     AFTER, _fail_driver("unlock"),      {}),
]


def run_case(name, side, inject, launch_kwargs, nodes=2, ranks_per_node=1):
    sim = SimCluster(nodes=nodes, ranks_per_node=ranks_per_node, step_seconds=0.002).start()
    outcome = {"case": name, "expected": side}
    try:
        sim.launch_ranks(**launch_kwargs)
        sim.wait_registered()
        if inject:
            inject(sim)

        raised = None
        try:
            sim.coord.checkpoint(sim.job_id, mode="continue")
        except AbortableError as exc:
            raised = ("abortable", str(exc))
        except TerminalError as exc:
            raised = ("terminal", str(exc))
        except Exception as exc:                       # noqa: BLE001
            raised = (type(exc).__name__, str(exc))

        epochs = sorted(
            sim.coord.store.list_epochs(sim.job_id), key=lambda e: e["created_at"]
        )
        phase = epochs[-1]["phase"] if epochs else None
        outcome["raised"] = raised[0] if raised else None
        outcome["phase"] = phase
        outcome["message"] = (raised[1][:160] if raised else "")

        if raised is None:
            outcome["ok"] = False
            outcome["why"] = "checkpoint reported success despite an injected fault"
        elif side == BEFORE:
            alive = sim.alive()
            expected_alive = nodes * ranks_per_node - (1 if name == "rank-killed" else 0)
            locked = [
                (node, record["rank"])
                for node, agent in sim.agents.items()
                for record in agent.local_ranks(sim.job_id)
                if agent.driver.state_of(record["host_pid"]) not in (None, "running")
            ]
            outcome["alive"] = alive
            outcome["still_locked"] = locked
            outcome["ok"] = (
                raised[0] == "abortable"
                and phase == Phase.ABORTED.value
                and alive == expected_alive
                and not locked
            )
            if not outcome["ok"]:
                outcome["why"] = (
                    f"expected an abort with the job intact; got raised={raised[0]} "
                    f"phase={phase} alive={alive}/{expected_alive} locked={locked}"
                )
        else:
            outcome["ok"] = raised[0] == "terminal" and phase == Phase.FAILED.value
            if not outcome["ok"]:
                outcome["why"] = (
                    f"expected a terminal failure; got raised={raised[0]} phase={phase}"
                )
        return outcome
    finally:
        sim.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="substring filter on case name")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    results = []
    started = time.time()
    for _ in range(args.repeat):
        for name, side, inject, launch in CASES:
            if args.only and args.only not in name:
                continue
            results.append(run_case(name, side, inject, launch))

    passed = sum(1 for r in results if r["ok"])
    report = {
        "cases": results,
        "passed": passed,
        "total": len(results),
        "seconds": round(time.time() - started, 2),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{'CASE':<20} {'SIDE':<14} {'RAISED':<10} {'PHASE':<10} RESULT")
        print("-" * 72)
        for r in results:
            mark = "pass" if r["ok"] else "FAIL"
            print(
                f"{r['case']:<20} {r['expected']:<14} {str(r['raised']):<10} "
                f"{str(r['phase']):<10} {mark}"
            )
            if not r["ok"]:
                print(f"    {r.get('why', '')}")
        print("-" * 72)
        print(f"{passed}/{len(results)} passed in {report['seconds']}s")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
