"""Container pid to host pid translation.

The agent runs with hostPID and calls the driver with host pids; the job speaks
in its own namespace. /proc/<host_pid>/status carries an NSpid line listing the
pid in each nested namespace, innermost last, which is the only reliable mapping
that does not require entering the namespace.
"""

import os

from mncr import log

_LOG = log.get("agent.pids")


class PidResolver:
    def __init__(self, proc_root="/proc"):
        self.proc_root = proc_root

    def _ns_pids(self, host_pid):
        path = os.path.join(self.proc_root, str(host_pid), "status")
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("NSpid:"):
                        return [int(x) for x in line.split()[1:]]
        except (FileNotFoundError, PermissionError, ValueError):
            return []
        return []

    def all_pids(self):
        try:
            return [int(n) for n in os.listdir(self.proc_root) if n.isdigit()]
        except FileNotFoundError:
            return []

    def to_host_pid(self, container_pid, hint_cgroup=None):
        """Find the host pid whose innermost namespace pid is container_pid.

        hint_cgroup narrows the search when several containers share a pid
        number, which is common: every container's init is pid 1.
        """
        matches = []
        for host_pid in self.all_pids():
            ns = self._ns_pids(host_pid)
            if ns and ns[-1] == int(container_pid):
                if hint_cgroup and hint_cgroup not in self._cgroup(host_pid):
                    continue
                matches.append(host_pid)
        if not matches:
            return None
        if len(matches) > 1:
            _LOG.warn(
                "ambiguous pid translation",
                container_pid=container_pid,
                candidates=matches,
            )
        return matches[0]

    def _cgroup(self, host_pid):
        try:
            with open(os.path.join(self.proc_root, str(host_pid), "cgroup")) as fh:
                return fh.read()
        except (FileNotFoundError, PermissionError):
            return ""

    def resolve(self, reported):
        """The host pid behind a pid a rank reported about itself.

        A rank sends os.getpid(). In a pod that is the pid inside the pod's
        namespace, and the same number belongs to some other process on the
        host. Returns (host_pid, how), where `how` says which reading held:

            host        the number names a host-namespace process, so it is
                        already a host pid
            translated  exactly one process has that number as its innermost
                        pid, so it is that process's host pid

        Raises LookupError when neither reading is safe - no such process, or
        the same inner pid in more than one namespace. The remedy for the
        latter is the unix socket, where the kernel reports the peer's pid
        directly; see mncr.rpc.peer_pid.
        """
        reported = int(reported)
        ns = self._ns_pids(reported)
        if len(ns) == 1:
            return reported, "host"
        if not ns and os.path.isdir(os.path.join(self.proc_root, str(reported))):
            # A kernel without NSpid (pre-4.1): nothing to translate with.
            return reported, "unverified"
        matches = [
            pid for pid in self.all_pids()
            if (self._ns_pids(pid) or [None])[-1] == reported
        ]
        if len(matches) == 1:
            return matches[0], "translated"
        if not matches:
            raise LookupError(f"no process has pid {reported} in its innermost namespace")
        raise LookupError(
            f"pid {reported} is the innermost pid of {len(matches)} processes "
            f"({sorted(matches)[:6]}); have the rank register over the unix "
            f"socket so the agent can read the peer pid"
        )

    def pids_in_cgroup(self, needle):
        """Every host pid whose cgroup path contains `needle`.

        Used to find all the ranks of a pod when the job did not register.
        """
        return [p for p in self.all_pids() if needle in self._cgroup(p)]

    def alive(self, host_pid):
        return os.path.isdir(os.path.join(self.proc_root, str(host_pid)))
