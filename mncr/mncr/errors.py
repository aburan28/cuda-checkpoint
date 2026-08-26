"""Error taxonomy.

The split that matters is Abortable vs Terminal. Everything up to and including
the driver lock can be undone by unlocking and resuming in place. Once
cuCheckpointProcessCheckpoint has been called the process has released its GPU
resources and the driver offers no way back, so a failure past that point ends
the epoch rather than retrying it.
"""


class MncrError(Exception):
    """Base for everything raised by this system."""


class AbortableError(MncrError):
    """Failed before the commit point. The epoch can be abandoned in place."""


class TerminalError(MncrError):
    """Failed after the commit point. The epoch is lost; ranks must be replaced."""


class UnreleasedAbortError(AbortableError):
    """Aborted before the commit point, but a node could not be reached to
    release its ranks. Nothing on a GPU was touched, so the epoch is not
    FAILED - yet ranks on that node are torn down and waiting for a token
    that will not come, and the job is not intact."""


class PreconditionError(AbortableError):
    """A rank or node did not meet a documented precondition."""


class NotCleanError(PreconditionError):
    """The process still holds resources the checkpoint path cannot handle."""

    def __init__(self, findings):
        self.findings = list(findings)
        detail = "; ".join(f.describe() for f in self.findings) or "unknown"
        super().__init__(f"process is not checkpoint-clean: {detail}")


class TimeoutError_(AbortableError):
    """A rank missed its deadline. Distinct from builtins.TimeoutError."""


class DriverError(MncrError):
    """cuda-checkpoint or the CUDA driver API returned failure."""


class CriuError(MncrError):
    """criu dump or restore returned failure."""


class PlacementError(AbortableError):
    """No target node satisfies the constraints recorded in the image manifest."""
