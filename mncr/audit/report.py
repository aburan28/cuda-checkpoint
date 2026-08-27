#!/usr/bin/env python3
"""Merge P0 evidence into a go/no-go report.

Inputs:
    --fleet   one or more JSON objects from fleet_audit.sh (file or glob)
    --calls   one or more JSONL files from cuda_audit.so
    --json    emit machine-readable output instead of text

The report answers one question per image and one per node: is there anything
here that the checkpoint path cannot handle, and if so what has to be torn down
or changed. Nothing in this file guesses - it only reports what the auditor and
the node inventory actually observed.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mncr.version import MIN_CRIU, MIN_DRIVER  # noqa: E402

# api -> (verdict, what to do about it)
REMEDIATION = {
    "cuMemCreate": (
        "blocker",
        "VMM allocation. Usually PyTorch expandable_segments or NCCL's default "
        "allocator. Free before the lock: destroy_process_group() then "
        "empty_cache(); disable expandable_segments if the driver rejects held "
        "segments.",
    ),
    "cuMemMap": (
        "blocker",
        "VMM mapping, same origin as cuMemCreate. Same remediation.",
    ),
    "cuMemExportToShareableHandle": (
        "blocker",
        "The documented hard limitation. If handle_type is FABRIC this is MNNVL "
        "and there is no workaround; if POSIX fd, destroying the owning "
        "communicator removes it.",
    ),
    "cuMemImportFromShareableHandle": (
        "blocker",
        "Receiving side of an unsupported export. Close before the lock.",
    ),
    "cuMemAllocManaged": (
        "blocker",
        "UVM is unsupported. Replace with explicit device allocations; add a CI "
        "gate so it cannot return.",
    ),
    "cuMulticastCreate": (
        "blocker",
        "NVLS multicast object. Full communicator teardown removes it; suspend "
        "alone does not (NCCL #2337).",
    ),
    "cuMulticastBindMem": ("blocker", "NVLS binding. Same as cuMulticastCreate."),
    "cuIpcGetMemHandle": (
        "conditional",
        "Legacy IPC. Supported from driver 610 when the processes were launched "
        "as one job; verify the job file is in place.",
    ),
    "cuIpcOpenMemHandle": ("conditional", "Same as cuIpcGetMemHandle."),
}


def _driver_major(version):
    try:
        return int(str(version).split(".")[0])
    except (ValueError, IndexError):
        return -1


def _criu_tuple(version):
    parts = []
    for chunk in str(version).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:2]) or (0, 0)


def check_node(node):
    """Findings for one node inventory object."""
    out = []
    host = node.get("host", "?")

    major = _driver_major(node.get("driver_version"))
    if major < 0:
        out.append(("blocker", "driver", "no NVIDIA driver detected"))
    elif major < MIN_DRIVER:
        out.append(
            (
                "blocker",
                "driver",
                f"driver {node['driver_version']} < {MIN_DRIVER}; job-file IPC "
                f"and the newest checkpoint fixes are unavailable",
            )
        )

    if node.get("mnnvl"):
        out.append(
            (
                "blocker",
                "mnnvl",
                "fabric/MNNVL state present. Fabric handles cannot be "
                "checkpointed and there is no workaround - exclude this node or "
                "disable MNNVL for checkpointable jobs",
            )
        )

    criu = node.get("criu_version", "none")
    if criu in ("none", "unknown"):
        out.append(("blocker", "criu", "criu not installed"))
    elif _criu_tuple(criu) < MIN_CRIU:
        out.append(("blocker", "criu", f"criu {criu} < {'.'.join(map(str, MIN_CRIU))}"))
    if not node.get("criu_cuda_plugin"):
        out.append(("blocker", "criu", "cuda_plugin.so not found in any criu libdir"))

    if node.get("cuda_checkpoint", "none") == "none":
        out.append(("blocker", "tooling", "cuda-checkpoint not on PATH"))

    if str(node.get("persistence_mode", "")).lower() not in ("enabled", "on"):
        out.append(
            (
                "warning",
                "persistence",
                "persistence mode is not enabled; GPU migration on restore "
                "requires it",
            )
        )

    gpu_mem = node.get("gpu_mem_total_mib", 0)
    headroom = node.get("ram_headroom_mib", 0)
    if gpu_mem and headroom < gpu_mem:
        out.append(
            (
                "warning",
                "sizing",
                f"host RAM headroom {headroom} MiB is below total device memory "
                f"{gpu_mem} MiB; a full-node checkpoint can exhaust host memory",
            )
        )

    if node.get("rdma_devices", 0) > 0:
        out.append(
            (
                "info",
                "rdma",
                f"{node['rdma_devices']} RDMA device(s); ranks must close verbs "
                f"fds before the lock",
            )
        )

    return host, out


def check_calls(paths):
    """Aggregate interposer output into per-api counts and verdicts."""
    counts, details = {}, {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "summary":
                    for api, n in (rec.get("counts") or {}).items():
                        counts[api] = counts.get(api, 0) + n
                elif rec.get("event") == "call":
                    api = rec.get("api")
                    if api == "cuGetProcAddress":
                        continue  # redirect bookkeeping, not an allocation
                    counts.setdefault(api, 0)
                    if rec.get("detail"):
                        details.setdefault(api, set()).add(rec["detail"])
    return counts, {k: sorted(v) for k, v in details.items()}


def build(fleet_paths, call_paths):
    nodes, node_findings = [], {}
    for path in fleet_paths:
        with open(path) as fh:
            text = fh.read().strip()
        for chunk in _split_json_objects(text):
            node = json.loads(chunk)
            nodes.append(node)
            host, findings = check_node(node)
            node_findings[host] = findings

    counts, details = check_calls(call_paths)
    api_findings = []
    for api, n in sorted(counts.items()):
        verdict, advice = REMEDIATION.get(api, ("info", "not classified"))
        api_findings.append(
            {
                "api": api,
                "calls": n,
                "verdict": verdict,
                "detail": details.get(api, []),
                "remediation": advice,
            }
        )

    blockers = sum(
        1 for f in node_findings.values() for sev, _, _ in f if sev == "blocker"
    ) + sum(1 for f in api_findings if f["verdict"] == "blocker")

    return {
        "nodes": nodes,
        "node_findings": {
            h: [{"severity": s, "area": a, "message": m} for s, a, m in f]
            for h, f in node_findings.items()
        },
        "api_findings": api_findings,
        "blockers": blockers,
        "verdict": "no-go" if blockers else "go",
    }


def _split_json_objects(text):
    """Accept a file holding one object, or several concatenated."""
    depth, start, out = 0, None, []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
                start = None
    return out


def render(report):
    lines = []
    add = lines.append
    add("=" * 72)
    add(f"P0 COMPATIBILITY REPORT   verdict: {report['verdict'].upper()}")
    add("=" * 72)

    add("")
    add(f"NODES ({len(report['nodes'])})")
    if not report["nodes"]:
        add("  (none supplied)")
    for node in report["nodes"]:
        host = node.get("host", "?")
        add(
            f"  {host:<28} driver={node.get('driver_version','?'):<12} "
            f"gpus={node.get('gpu_count',0):<3} criu={node.get('criu_version','?')}"
        )
        for f in report["node_findings"].get(host, []):
            add(f"      [{f['severity']:^8}] {f['area']}: {f['message']}")
        if not report["node_findings"].get(host):
            add("      [   ok   ] no findings")

    add("")
    add("OBSERVED ALLOCATIONS")
    if not report["api_findings"]:
        add("  (no interposer output supplied)")
    for f in report["api_findings"]:
        add(f"  [{f['verdict']:^11}] {f['api']}  x{f['calls']}")
        if f["detail"]:
            add(f"      seen: {', '.join(f['detail'][:6])}")
        add(f"      -> {f['remediation']}")

    add("")
    add(f"BLOCKERS: {report['blockers']}")
    if report["verdict"] == "go":
        add("Every observed resource is either supported or has a named teardown.")
    else:
        add("Resolve the blockers above before proceeding to P1.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fleet", action="append", default=[], help="fleet_audit.sh output")
    ap.add_argument("--calls", action="append", default=[], help="cuda_audit.so JSONL")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    fleet = [p for pat in args.fleet for p in sorted(glob.glob(pat)) or [pat]]
    calls = [p for pat in args.calls for p in sorted(glob.glob(pat)) or [pat]]
    fleet = [p for p in fleet if os.path.exists(p)]
    calls = [p for p in calls if os.path.exists(p)]

    report = build(fleet, calls)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if report["verdict"] == "no-go" else 0


if __name__ == "__main__":
    raise SystemExit(main())
