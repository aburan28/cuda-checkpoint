"""assert_clean() - the precondition that turns intent into a checked fact.

A rank votes CLEAN only if this passes. The agent re-runs the same scan from
outside before it calls the driver, so a rank that lies about itself, or that
acquires something after voting, is still caught.
"""

import os

from mncr import log
from mncr.errors import NotCleanError
from mncr.procscan import ProcScanner, Severity

_LOG = log.get("torchckpt.cleanroom")


def scan_self(root="/proc"):
    scanner = ProcScanner(root)
    if not scanner.available():
        _LOG.warn("procfs unavailable; self-check degraded", root=root)
        return []
    return scanner.blocking(os.getpid(), Severity.BEFORE_LOCK)


def assert_clean(root="/proc", strict=True):
    """Raise NotCleanError if this process still holds unsupported resources.

    strict=False downgrades to a warning, for bring-up on a node where the scan
    itself is not yet trustworthy. Production runs with strict=True; the whole
    point is that a dirty rank fails loudly at prepare time rather than at the
    checkpoint call, where failure is no longer recoverable in place.
    """
    findings = scan_self(root)
    if not findings:
        _LOG.debug("clean", pid=os.getpid())
        return []
    if strict:
        raise NotCleanError(findings)
    _LOG.warn(
        "not clean, continuing because strict=False",
        pid=os.getpid(),
        findings=[f.describe() for f in findings],
    )
    return findings


def findings_as_dicts(findings):
    return [f.to_dict() for f in findings]
