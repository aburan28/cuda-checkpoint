"""Can this node actually do this?

Everything here uses the real backends. The simulator proves the protocol; this
proves the node. It is the check that turns "the code compiles" into "this
machine will not fail at the commit point", and it is meant to run as an init
container on the agent DaemonSet so a node that cannot participate never
advertises itself as one that can.

Findings use the same severity vocabulary as the P0 report, so a node inventory
and a preflight read the same way.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mncr import config, log  # noqa: E402
from mncr.errors import MncrError  # noqa: E402
from mncr.procscan import ProcScanner  # noqa: E402
from mncr.version import MIN_CRIU, MIN_DRIVER  # noqa: E402

from .jobfile import JobFiles  # noqa: E402
from .pids import PidResolver  # noqa: E402

_LOG = log.get("agent.preflight")

BLOCKER, WARNING, INFO = "blocker", "warning", "info"


class Check:
    def __init__(self, name):
        self.name = name
        self.findings = []

    def blocker(self, message, **detail):
        self.findings.append({"severity": BLOCKER, "check": self.name,
                              "message": message, **detail})

    def warning(self, message, **detail):
        self.findings.append({"severity": WARNING, "check": self.name,
                              "message": message, **detail})

    def info(self, message, **detail):
        self.findings.append({"severity": INFO, "check": self.name,
                              "message": message, **detail})


def _run(cmd, timeout=30):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]}: timed out"


def check_tooling(cfg):
    check = Check("tooling")
    binary = shutil.which(cfg.cuda_checkpoint)
    if not binary:
        check.blocker(f"{cfg.cuda_checkpoint} is not on PATH")
    else:
        code, out = _run([binary, "--help"])
        # The utility is a shim: it reports the driver's version, not its own,
        # and refuses with "Insufficient driver" when the driver is too old.
        if "Insufficient driver" in out:
            check.blocker("cuda-checkpoint reports the driver is too old", output=out.strip()[:200])
        elif code not in (0, 1):
            check.blocker(f"cuda-checkpoint --help exited {code}", output=out.strip()[:200])
        else:
            version = next(
                (line for line in out.splitlines() if line.startswith("Version")), ""
            )
            check.info(f"cuda-checkpoint present at {binary}", version=version.strip())
    return check


def check_driver(require_ipc=True):
    check = Check("driver")
    code, out = _run(["nvidia-smi", "--query-gpu=driver_version,name,persistence_mode",
                      "--format=csv,noheader"])
    if code != 0:
        check.blocker("nvidia-smi unavailable; no driver to checkpoint against")
        return check
    rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
    if not rows:
        check.blocker("nvidia-smi reported no GPUs")
        return check
    version = rows[0].split(",")[0].strip()
    try:
        major = int(version.split(".")[0])
    except ValueError:
        major = -1
    if major < MIN_DRIVER:
        # Measured on 595: single-process checkpoint, restore and --device-map
        # all work; what is missing is job-file IPC, which arrives at 610. So
        # this blocks only a fleet that needs multi-process IPC.
        message = (
            f"driver {version} is below {MIN_DRIVER}: single-process "
            f"checkpoint/restore and --device-map work, but job-file IPC does "
            f"not, and cuIpcGetMemHandle memory cannot be checkpointed"
        )
        if require_ipc:
            check.blocker(message, driver=version)
        else:
            check.warning(message, driver=version)
    else:
        check.info(f"driver {version}", gpus=len(rows))
    if not any("Enabled" in r for r in rows):
        check.warning("persistence mode is not enabled; GPU migration requires it")
    return check


def check_criu(cfg):
    check = Check("criu")
    binary = shutil.which(cfg.criu)
    if not binary:
        check.blocker(f"{cfg.criu} is not on PATH")
        return check
    code, out = _run([binary, "--version"])
    if code != 0:
        check.blocker(f"criu --version exited {code}", output=out.strip()[:200])
        return check
    raw = out.strip().splitlines()[0].split()[-1]
    parts = []
    for chunk in raw.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts[:2]) < MIN_CRIU:
        check.blocker(f"criu {raw} is below {'.'.join(map(str, MIN_CRIU))}")
    else:
        check.info(f"criu {raw}")

    plugin = os.path.join(cfg.criu_libdir, "cuda_plugin.so")
    if not os.path.exists(plugin):
        found = None
        for candidate in ("/usr/lib/criu", "/usr/local/lib/criu", "/usr/lib64/criu"):
            if os.path.exists(os.path.join(candidate, "cuda_plugin.so")):
                found = candidate
                break
        if found:
            check.warning(
                f"cuda_plugin.so is at {found}, not the configured {cfg.criu_libdir}",
                fix=f"set MNCR_CRIU_LIBDIR={found}",
            )
        else:
            check.blocker("cuda_plugin.so not found in any known criu libdir")
    else:
        check.info(f"cuda plugin at {plugin}")

    # criu check reports whether the kernel supports what it needs.
    code, out = _run([binary, "check", "--extra"], timeout=60)
    if code != 0:
        check.warning(
            "criu check --extra reported problems; dumps may fail",
            output=out.strip()[-300:],
        )
    return check


def check_privileges():
    check = Check("privileges")
    if os.geteuid() != 0:
        check.blocker("not running as root; CRIU and the driver calls both need it")
    scanner = ProcScanner("/proc")
    if not scanner.available():
        check.blocker("/proc is not readable; verification gates cannot run")
        return check

    # hostPID: without it the agent only sees its own namespace, and the pids it
    # would pass to the driver mean nothing.
    resolver = PidResolver("/proc")
    pids = resolver.all_pids()
    if len(pids) < 20:
        check.warning(
            f"only {len(pids)} pids visible; the agent is probably not in the "
            f"host pid namespace (hostPID: true)"
        )
    else:
        check.info(f"{len(pids)} pids visible")

    sample = next((p for p in pids if p > 1), None)
    if sample and not resolver._ns_pids(sample):
        check.warning("NSpid is absent from /proc/<pid>/status; pid translation "
                      "will fall back to guessing")
    return check


def check_paths(cfg):
    check = Check("paths")
    for label, path in (
        ("control", os.environ.get("MNCR_CONTROL_ROOT", "/run/mncr")),
        ("images", cfg.image_dir),
        ("cache", cfg.cache_dir),
        ("jobfiles", cfg.jobfile_dir),
    ):
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".mncr-preflight")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.unlink(probe)
            check.info(f"{label} writable", path=path)
        except OSError as exc:
            check.blocker(f"{label} directory not writable: {exc}", path=path)
    return check


def check_capacity(cfg):
    check = Check("capacity")
    code, out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"])
    device_mib = 0
    if code == 0:
        for row in out.strip().splitlines():
            digits = "".join(c for c in row if c.isdigit())
            device_mib += int(digits or 0)
    host_mib = 0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    host_mib = int(line.split()[1]) // 1024
                    break
    except FileNotFoundError:
        pass

    if device_mib and host_mib and host_mib < device_mib:
        check.blocker(
            f"host has {host_mib} MiB available but device memory totals "
            f"{device_mib} MiB; the driver copies device memory into host "
            f"allocations, so a full-node checkpoint would exhaust it",
            host_available_mib=host_mib,
            device_total_mib=device_mib,
        )
    elif device_mib:
        check.info(
            "host memory headroom is sufficient",
            host_available_mib=host_mib,
            device_total_mib=device_mib,
        )

    usage = shutil.disk_usage(cfg.image_dir) if os.path.isdir(cfg.image_dir) else None
    if usage and device_mib and (usage.free >> 20) < device_mib:
        check.warning(
            f"{cfg.image_dir} has {usage.free >> 20} MiB free; one full-node "
            f"image is about {device_mib} MiB",
        )
    return check


def check_jobfile(cfg):
    """Actually create a job file. The only way to know the driver will."""
    check = Check("jobfile")
    probe_dir = tempfile.mkdtemp(prefix="mncr-preflight-")
    jobfiles = JobFiles(probe_dir, cfg.cuda_checkpoint, fake=False)
    try:
        path = jobfiles.create("preflight")
        check.info("job file creation works", path=path)
    except MncrError as exc:
        check.blocker(f"job file creation failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        check.blocker(f"job file creation raised {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    return check


def check_target(cfg, pid):
    """Optional: confirm the driver recognises a specific running process."""
    check = Check("target")
    from .driver import CliBackend

    try:
        state = CliBackend(cfg.cuda_checkpoint).get_state(pid)
        check.info(f"pid {pid} is a CUDA process in state {state!r}")
    except MncrError as exc:
        check.blocker(f"cuda-checkpoint --get-state on pid {pid} failed: {exc}")
    return check


def run(cfg=None, target_pid=None, skip=(), require_ipc=True):
    cfg = cfg or config.load()
    checks = []
    if "tooling" not in skip:
        checks.append(check_tooling(cfg))
    if "driver" not in skip:
        checks.append(check_driver(require_ipc=require_ipc))
    if "criu" not in skip:
        checks.append(check_criu(cfg))
    if "privileges" not in skip:
        checks.append(check_privileges())
    if "paths" not in skip:
        checks.append(check_paths(cfg))
    if "capacity" not in skip:
        checks.append(check_capacity(cfg))
    if "jobfile" not in skip:
        checks.append(check_jobfile(cfg))
    if target_pid and "target" not in skip:
        checks.append(check_target(cfg, target_pid))

    findings = [f for check in checks for f in check.findings]
    blockers = [f for f in findings if f["severity"] == BLOCKER]
    return {
        "node": os.environ.get("NODE_NAME") or os.uname().nodename,
        "findings": findings,
        "blockers": len(blockers),
        "verdict": "no-go" if blockers else "go",
    }


def render(report):
    lines = [
        "=" * 72,
        f"PREFLIGHT {report['node']}   verdict: {report['verdict'].upper()}",
        "=" * 72,
    ]
    current = None
    for finding in report["findings"]:
        if finding["check"] != current:
            current = finding["check"]
            lines.append(f"\n{current.upper()}")
        detail = " ".join(
            f"{k}={v}"
            for k, v in finding.items()
            if k not in ("severity", "check", "message")
        )
        lines.append(f"  [{finding['severity']:^8}] {finding['message']}")
        if detail:
            lines.append(f"             {detail}")
    lines.append("")
    lines.append(
        f"BLOCKERS: {report['blockers']}"
        + ("" if report["blockers"] else "  - this node can participate")
    )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="validate a node against the real backends")
    ap.add_argument("--pid", type=int, default=None, help="also check a running CUDA process")
    ap.add_argument("--skip", default="", help="comma-separated checks to skip")
    ap.add_argument(
        "--no-ipc",
        action="store_true",
        help="the job does not use multi-process CUDA IPC, so a driver below "
             "610 is a warning rather than a blocker",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = run(
        target_pid=args.pid,
        skip=set(filter(None, args.skip.split(","))),
        require_ipc=not args.no_ipc,
    )
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if report["verdict"] == "no-go" else 0


if __name__ == "__main__":
    raise SystemExit(main())
