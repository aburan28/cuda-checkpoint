#!/usr/bin/env python3
"""Assert the interposer saw what the probe did.

Two properties, and the second is the one worth the whole test rig: the direct
calls are recorded, and so are the calls made through a pointer that
cuGetProcAddress returned. Interposing the symbol alone would satisfy the first
and silently fail the second, which is exactly the shape of an audit that comes
back clean on a workload full of unsupported allocations.
"""

import glob
import json
import sys

EXPECTED_DIRECT = {
    "cuMemCreate",
    "cuMemMap",
    "cuMemExportToShareableHandle",
    "cuMemAllocManaged",
    "cuMulticastCreate",
    "cuIpcGetMemHandle",
}


def main(prefix, expect_dlsym_hook):
    paths = sorted(glob.glob(f"{prefix}*"))
    if not paths:
        print(f"FAIL: the interposer wrote nothing to {prefix}*")
        print("      it was probably not loaded at all")
        return 1

    calls, summary, redirects, dlsym_hits = [], {}, set(), set()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["event"] == "summary":
                    for api, count in record["counts"].items():
                        summary[api] = summary.get(api, 0) + count
                elif record["event"] == "call":
                    if record["api"] == "cuGetProcAddress":
                        redirects.add(record["detail"])
                    elif record["api"] == "dlsym":
                        dlsym_hits.add(record["detail"])
                    else:
                        calls.append(record)

    seen = {c["api"] for c in calls}
    problems = []

    missing = EXPECTED_DIRECT - seen
    if missing:
        problems.append(f"never recorded: {', '.join(sorted(missing))}")

    if not redirects:
        problems.append(
            "cuGetProcAddress was never intercepted; a real workload resolving "
            "entry points that way would be invisible to the audit"
        )
    else:
        for wanted in ("cuMemCreate", "cuMemExportToShareableHandle"):
            if wanted not in redirects:
                problems.append(f"cuGetProcAddress({wanted}) was not redirected")

    # The probe calls cuMemCreate twice: once directly, once through the
    # pointer. Both must land, or the redirect returned the driver's function.
    # Resolution paths the probe exercises: direct, cuGetProcAddress, and -
    # where the platform supports interposing it - dlsym on a handle. The last
    # is glibc-only; macOS builds compile the hook out, so the expectation is
    # passed in rather than inferred. Inferring it would let a silently broken
    # hook pass by simply never firing.
    expected_creates = 3 if expect_dlsym_hook else 2
    creates = summary.get("cuMemCreate", 0)
    if creates < expected_creates:
        problems.append(
            f"cuMemCreate recorded {creates} time(s), expected {expected_creates} "
            f"- one of the resolution paths did not go through the wrapper"
        )
    if expect_dlsym_hook and "cuMemCreate" not in dlsym_hits:
        problems.append(
            "dlsym(cuMemCreate) was not intercepted; a workload that resolves "
            "driver entry points that way would be invisible to the audit"
        )

    fabric = [
        c
        for c in calls
        if c["api"] == "cuMemExportToShareableHandle" and "handle_type=8" in c["detail"]
    ]
    if not fabric:
        problems.append("the FABRIC handle type was not captured in the detail field")

    # Severities follow what was measured on hardware, not what the vendor
    # documentation says in the aggregate: holding VMM memory is fine, UVM is
    # not, and importing somebody else's handle is the one that cannot be
    # restored. See docs/findings-595-blackwell.md.
    severities = {c["api"]: c["severity"] for c in calls}
    expected = {
        "cuMemCreate": "conditional",
        "cuMemMap": "conditional",
        "cuMemExportToShareableHandle": "conditional",
        "cuMemAllocManaged": "blocker",
        "cuMulticastCreate": "blocker",
        "cuIpcGetMemHandle": "conditional",
    }
    for api, want in expected.items():
        if api in severities and severities[api] != want:
            problems.append(
                f"{api} classified {severities[api]!r}, expected {want!r}"
            )

    if problems:
        print("FAIL")
        for problem in problems:
            print(f"  {problem}")
        print(f"  observed: {json.dumps(summary, sort_keys=True)}")
        return 1

    print(
        f"interposer ok: {len(calls)} calls, "
        f"{len(redirects)} cuGetProcAddress redirects, "
        f"{len(dlsym_hits)} dlsym redirects, "
        f"{len(seen)} distinct APIs"
    )
    return 0


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mncr-audit-test"
    expect = len(sys.argv) > 2 and sys.argv[2] == "--expect-dlsym-hook"
    sys.exit(main(prefix, expect))
