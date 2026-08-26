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
        # Measured: aws-ofi-nccl's libfabric opens this at plugin init - on a
        # node with no EFA, where the plugin then fails and NCCL falls back to
        # sockets - and never closes it. Nothing in the process can release
        # it after the fact; it has to be prevented at launch.
        "GDRCopy handle; CRIU cannot dump it and destroying the communicator "
        "does not close it. Launch with FI_HMEM_CUDA_USE_GDRCOPY=0 (keeps EFA) "
        "or NCCL_NET_PLUGIN=none",
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


_TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
    "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
    "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING",
}

# A TCP socket is a local address CRIU has to bind again on restore, and a
# bind that fails - the port taken by anything else on the target - fails the
# restore, past the commit point. Measured: NCCL's RAS subsystem leaves two
# listeners behind after every communicator is destroyed, one on
# 127.0.0.1:28028 and one on the node address, and a restore on the same node
# lost the epoch to exactly that. A clean rank holds none, so this is free.
_SOCKET_WHY = {
    "LISTEN": (
        "listening socket survives teardown; CRIU must rebind the port on "
        "restore. NCCL RAS is the usual owner: launch with NCCL_RAS_ENABLE=0"
    ),
    "*": (
        "TCP socket survives teardown; CRIU must rebind its local port on "
        "restore and its peer is at a different address after a migration"
    ),
}


def _decode_addr(hex_addr):
    host, _, port = hex_addr.partition(":")
    if len(host) == 8:
        return ".".join(str(int(host[i:i + 2], 16)) for i in (6, 4, 2, 0)) + f":{int(port, 16)}"
    return f"[{host}]:{int(port, 16)}"


class ProcScanner:
    def __init__(self, root="/proc"):
        self.root = root

    def available(self):
        return os.path.isdir(self.root)

    # ------------------------------------------------------------ sockets
    def _tcp_table(self, pid):
        """inode -> (local, remote, state), from the process's own net view."""
        table = {}
        for name in ("tcp", "tcp6"):
            path = os.path.join(self.root, str(pid), "net", name)
            try:
                with open(path) as fh:
                    next(fh, None)
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 10:
                            continue
                        table[parts[9]] = (
                            _decode_addr(parts[1]),
                            _decode_addr(parts[2]),
                            _TCP_STATES.get(parts[3], parts[3]),
                        )
            except (FileNotFoundError, PermissionError, OSError):
                continue
        return table

    def _socket_findings(self, pid, fd_targets):
        sockets = [(fd, t[8:-1]) for fd, t in fd_targets if t.startswith("socket:[")]
        if not sockets:
            return []
        table = self._tcp_table(pid)
        out = []
        for fd, inode in sockets:
            entry = table.get(inode)
            if entry is None:
                continue          # unix, udp, netlink: CRIU copes, or the peer is inside the tree
            local, remote, state = entry
            kind = "listening_socket" if state == "LISTEN" else "tcp_socket"
            why = _SOCKET_WHY["LISTEN" if state == "LISTEN" else "*"]
            out.append(
                Finding(kind, Severity.BEFORE_LOCK, f"fd {fd} {state} {local} -> {remote}", why)
            )
        return out

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
        fd_targets = self._fd_targets(pid)
        for fd, target in fd_targets:
            for pattern, kind, sev, why in _FD_RULES:
                if pattern.match(target):
                    findings.append(Finding(kind, sev, f"fd {fd} -> {target}", why))
                    break
        findings.extend(self._socket_findings(pid, fd_targets))
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
