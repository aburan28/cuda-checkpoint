"""Inspect a process for resources the CUDA checkpoint path cannot handle.

What this can and cannot tell you, measured rather than assumed: an ordinary
CUDA process holds /dev/nvidia*, /dev/nvidiactl and /dev/nvidia-uvm the whole
time it is running, so none of those mean anything before a checkpoint. After
one, the driver has closed every last one, which makes their absence a real
precondition for the dump. Actual managed-memory *use* is not visible here at
all - that is what the P0 interposer is for.

Used from two directions and deliberately shared, so the rank's self-check and
the agent's external check can never disagree:

    torchckpt.assert_clean()   from inside the process, before it votes
    agent.verify               from outside, before CRIU is invoked

The two checks run at different points and enforce different bars, which is why
Severity distinguishes them:

    BEFORE_LOCK  must be gone before cuCheckpointProcessLock is called
    BEFORE_DUMP  must be gone before criu dump is invoked (the driver's
                 checkpoint releases these; if they are still here, the
                 checkpoint did not do what we think it did)
    INFO         worth recording, not disqualifying

`root` is injectable so this is testable on a host without /proc.
"""

import dataclasses
import enum
import os
import re


class Severity(enum.Enum):
    BEFORE_LOCK = "before_lock"
    BEFORE_DUMP = "before_dump"
    INFO = "info"


@dataclasses.dataclass(frozen=True)
class Finding:
    kind: str
    severity: Severity
    target: str
    why: str

    def describe(self):
        return f"{self.kind}({self.target}): {self.why}"

    def to_dict(self):
        return {
            "kind": self.kind,
            "severity": self.severity.value,
            "target": self.target,
            "why": self.why,
        }


# fd targets, matched against the resolved symlink
_FD_RULES = [
    (
        re.compile(r"^/dev/infiniband/(uverbs|rdma_cm|umad)"),
        "verbs_fd",
        Severity.BEFORE_LOCK,
        "RDMA resource; neither the driver nor CRIU can checkpoint queue pairs",
    ),
    (
        re.compile(r"^/dev/gdrdrv"),
        "gdrcopy_fd",
        Severity.BEFORE_LOCK,
        "GDRCopy pins device memory for the NIC; must be released first",
    ),
    (
        re.compile(r"^/dev/nvidia-uvm"),
        "uvm_fd",
        Severity.BEFORE_DUMP,
        # Measured on a real node: every CUDA process opens this, whether or not
        # it ever allocates managed memory - the runtime initialises UVM
        # support regardless. Treating it as a pre-lock blocker rejected a
        # process that checkpoints and restores perfectly. It is only
        # meaningful after the checkpoint, by which point the driver has closed
        # it along with every other GPU fd.
        "GPU fd still open after checkpoint",
    ),
    (
        re.compile(r"^/dev/nvidia(ctl|-modeset|\d+)"),
        "nvidia_fd",
        Severity.BEFORE_DUMP,
        "GPU still held; the driver checkpoint should have released it",
    ),
    (
        re.compile(r"^/dev/nvidia-caps"),
        "nvidia_caps_fd",
        Severity.BEFORE_DUMP,
        "GPU capability fd still open after checkpoint",
    ),
]

# mapping paths, matched against the pathname column of /proc/<pid>/maps
_MAP_RULES = [
    (
        re.compile(r"/dev/nvidia-uvm"),
        "uvm_mapping",
        Severity.BEFORE_DUMP,
        # Same measurement: present in an ordinary CUDA process that never
        # touches managed memory, so it says nothing before the lock.
        "UVM mapping still present after checkpoint",
    ),
    (
        re.compile(r"/dev/nvidia\d+"),
        "device_mapping",
        Severity.BEFORE_DUMP,
        "device memory still mapped after checkpoint",
    ),
]


class ProcScanner:
    def __init__(self, root="/proc"):
        self.root = root

    def available(self):
        return os.path.isdir(self.root)

    # ---------------------------------------------------------------- fds
    def _fd_targets(self, pid):
        fd_dir = os.path.join(self.root, str(pid), "fd")
        out = []
        try:
            names = os.listdir(fd_dir)
        except (FileNotFoundError, PermissionError):
            return out
        for name in names:
            path = os.path.join(fd_dir, name)
            try:
                out.append((name, os.readlink(path)))
            except OSError:
                continue
        return out

    # --------------------------------------------------------------- maps
    def _map_paths(self, pid):
        maps = os.path.join(self.root, str(pid), "maps")
        out = []
        try:
            with open(maps, "r") as fh:
                for line in fh:
                    parts = line.split(None, 5)
                    if len(parts) == 6:
                        out.append(parts[5].strip())
        except (FileNotFoundError, PermissionError):
            pass
        return out

    # -------------------------------------------------------------- public
    def scan(self, pid):
        """All findings for `pid`, unfiltered."""
        findings = []
        for fd, target in self._fd_targets(pid):
            for pattern, kind, sev, why in _FD_RULES:
                if pattern.match(target):
                    findings.append(Finding(kind, sev, f"fd {fd} -> {target}", why))
                    break
        seen_maps = set()
        for path in self._map_paths(pid):
            for pattern, kind, sev, why in _MAP_RULES:
                if pattern.search(path) and (kind, path) not in seen_maps:
                    seen_maps.add((kind, path))
                    findings.append(Finding(kind, sev, path, why))
                    break
        return findings

    def blocking(self, pid, gate):
        """Findings that block `gate`, which is a Severity.

        BEFORE_LOCK findings also block the dump, so the dump gate is the union.
        """
        found = self.scan(pid)
        if gate is Severity.BEFORE_LOCK:
            return [f for f in found if f.severity is Severity.BEFORE_LOCK]
        if gate is Severity.BEFORE_DUMP:
            return [
                f
                for f in found
                if f.severity in (Severity.BEFORE_LOCK, Severity.BEFORE_DUMP)
            ]
        return found


def summarize(findings):
    counts = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    return counts
