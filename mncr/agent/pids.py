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

    def pids_in_cgroup(self, needle):
        """Every host pid whose cgroup path contains `needle`.

        Used to find all the ranks of a pod when the job did not register.
        """
        return [p for p in self.all_pids() if needle in self._cgroup(p)]

    def alive(self, host_pid):
        return os.path.isdir(os.path.join(self.proc_root, str(host_pid)))
