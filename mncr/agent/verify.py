"""External verification, run by the agent at both gates.

The rank checks itself before voting; this runs the same scan from outside at
two later moments. It exists because the rank's self-report is a claim about a
moment that has already passed - between the vote and the lock, a background
thread can open a verbs fd and nobody inside the process would notice.
"""

from mncr import log
from mncr.errors import NotCleanError
from mncr.procscan import ProcScanner, Severity

_LOG = log.get("agent.verify")


class Verifier:
    def __init__(self, proc_root="/proc", strict=True):
        self.scanner = ProcScanner(proc_root)
        self.strict = strict

    def _gate(self, host_pid, severity, gate_name):
        if not self.scanner.available():
            _LOG.warn("procfs unavailable; verification skipped", gate=gate_name)
            return []
        findings = self.scanner.blocking(host_pid, severity)
        if findings and self.strict:
            raise NotCleanError(findings)
        if findings:
            _LOG.warn(
                "gate findings ignored (strict=False)",
                gate=gate_name,
                pid=host_pid,
                findings=[f.describe() for f in findings],
            )
        return findings

    def before_lock(self, host_pid):
        """The rank claims it is clean. Confirm it before taking the lock."""
        return self._gate(host_pid, Severity.BEFORE_LOCK, "before_lock")

    def before_dump(self, host_pid):
        """After the driver checkpoint, the process must hold no GPU at all.

        A device fd still open here means the checkpoint did not do what we
        believe it did, and dumping anyway would produce an image that cannot be
        restored.
        """
        return self._gate(host_pid, Severity.BEFORE_DUMP, "before_dump")
