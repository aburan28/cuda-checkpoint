"""Wire protocol and the phase model.

The phase enum encodes the single most important property of the design: where
the commit point sits. Everything at or before LOCKED can be abandoned by
unlocking and resuming in place. From CHECKPOINTED onward the driver has already
released the process's GPU resources and provides no rollback, so failure is
terminal for the epoch.
"""

import enum
import json
import time
import uuid


class Phase(enum.Enum):
    RUNNING = "running"            # steady state, no epoch in flight
    PREPARING = "preparing"        # ranks driving to a safe point and tearing down
    PREPARED = "prepared"          # every rank has voted clean
    LOCKED = "locked"              # driver lock held on every pid
    CHECKPOINTED = "checkpointed"  # <-- commit point crossed
    DUMPED = "dumped"              # CRIU images written
    RESTORING = "restoring"        # images loaded, driver restore in flight
    RESUMED = "resumed"            # unlocked, communicators rebuilt
    ABORTED = "aborted"            # abandoned before the commit point
    FAILED = "failed"              # lost after the commit point

    @property
    def abortable_in_place(self):
        """True if reaching this phase can be undone without losing the ranks."""
        return self in _ABORTABLE

    @property
    def terminal(self):
        return self in (Phase.ABORTED, Phase.FAILED)


_ABORTABLE = frozenset(
    {Phase.RUNNING, Phase.PREPARING, Phase.PREPARED, Phase.LOCKED}
)

#: The phase whose entry crosses the commit point. Named once so the coordinator,
#: the agent and the tests cannot disagree about where the line is.
COMMIT_POINT = Phase.CHECKPOINTED

#: Legal forward transitions. Aborts and failures are handled separately because
#: they may be entered from many phases.
_FORWARD = {
    Phase.RUNNING: {Phase.PREPARING},
    Phase.PREPARING: {Phase.PREPARED},
    Phase.PREPARED: {Phase.LOCKED},
    Phase.LOCKED: {Phase.CHECKPOINTED},
    Phase.CHECKPOINTED: {Phase.DUMPED},
    Phase.DUMPED: {Phase.RESTORING, Phase.RUNNING},
    Phase.RESTORING: {Phase.RESUMED},
    Phase.RESUMED: {Phase.RUNNING},
    Phase.ABORTED: set(),
    Phase.FAILED: set(),
}


def can_transition(src, dst):
    if dst is Phase.ABORTED:
        return src.abortable_in_place
    if dst is Phase.FAILED:
        return not src.terminal
    return dst in _FORWARD[src]


def next_on_failure(src):
    """The phase an epoch lands in when something fails while in `src`."""
    return Phase.ABORTED if src.abortable_in_place else Phase.FAILED


class Vote(enum.Enum):
    CLEAN = "clean"        # rank reached a safe point and holds nothing unsupported
    DIRTY = "dirty"        # rank could not clean up; names what is left
    TIMEOUT = "timeout"    # rank never answered
    ERROR = "error"        # rank raised while quiescing


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Message(dict):
    """JSON-line request/response envelope.

    Requests carry {id, op, args}; responses carry {id, ok, result|error}.
    """

    @staticmethod
    def request(op, **args):
        return Message(id=new_id("req"), op=op, args=args, ts=time.time())

    @staticmethod
    def ok(req_id, **result):
        return Message(id=req_id, ok=True, result=result, ts=time.time())

    @staticmethod
    def err(req_id, error, kind="error"):
        return Message(id=req_id, ok=False, error=str(error), kind=kind, ts=time.time())

    def encode(self):
        return (json.dumps(self, default=str) + "\n").encode()

    @staticmethod
    def decode(line):
        return Message(json.loads(line))


class RankRef(dict):
    """Identity of one rank, as the coordinator and agent both see it."""

    @staticmethod
    def make(job_id, rank, node, pod=None, container=None, host_pid=None, gpu_uuids=None,
             ip=None):
        return RankRef(
            job_id=job_id,
            rank=int(rank),
            node=node,
            pod=pod,
            container=container,
            host_pid=host_pid,
            gpu_uuids=list(gpu_uuids or []),
            # Where the rank itself can be reached - its pod address when
            # ranks are not on the host network. The rendezvous for a rebuild
            # is built on rank 0's, since a peer has to connect to it.
            ip=ip,
        )

    @property
    def key(self):
        return f"{self['job_id']}/{self['rank']}"


class Epoch(dict):
    """One checkpoint attempt, as durable coordinator state."""

    @staticmethod
    def make(job_id, ranks, reason="manual", deadline_s=120.0):
        return Epoch(
            epoch_id=new_id("ep"),
            job_id=job_id,
            reason=reason,
            phase=Phase.RUNNING.value,
            created_at=time.time(),
            deadline_s=float(deadline_s),
            ranks=[dict(r) for r in ranks],
            votes={},
            image_id=None,
            error=None,
            history=[],
        )

    @property
    def phase(self):
        return Phase(self["phase"])

    def set_phase(self, phase, note=None):
        src = self.phase
        if not can_transition(src, phase):
            raise ValueError(f"illegal transition {src.value} -> {phase.value}")
        self["phase"] = phase.value
        self["history"].append(
            {"at": time.time(), "from": src.value, "to": phase.value, "note": note}
        )
        return self

    @property
    def past_commit_point(self):
        return not self.phase.abortable_in_place

    def rank_refs(self):
        return [RankRef(r) for r in self["ranks"]]

    def nodes(self):
        seen = []
        for r in self["ranks"]:
            if r["node"] not in seen:
                seen.append(r["node"])
        return seen

    def ranks_on(self, node):
        return [RankRef(r) for r in self["ranks"] if r["node"] == node]
