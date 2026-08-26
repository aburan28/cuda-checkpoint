"""Rank <-> agent control channel.

The rank holds no persistent connection and runs no listener. That is a design
constraint, not a simplification: any socket open across the CRIU dump is a
resource whose peer lives outside the dumped tree, and CRIU would need it
declared external. By polling a file and connecting out only to post a vote, a
quiesced rank holds nothing but its own memory.

Layout of the shared control directory, hostPath-mounted into both the agent
and every rank pod:

    <root>/jobs/<job>/request.json          agent -> all ranks
    <root>/jobs/<job>/ranks/<rank>.json     agent -> one rank (resume / abort)

Both files are written atomically via rename, so a rank never observes a partial
document.
"""

import json
import os
import tempfile
import time

from mncr import log, rpc

_LOG = log.get("torchckpt.channel")


def write_atomic(path, payload):
    """Write via a unique temp file in the same directory, then rename.

    The temp name must be unique per call, not per process: two threads writing
    the same path is a normal occurrence in the agent, and sharing a temp name
    means one rename wins and the other fails with ENOENT.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class ControlDir:
    def __init__(self, root, job_id):
        self.root = root
        self.job_id = job_id
        self._request_sig = None
        self._serviced = set()

    @property
    def job_dir(self):
        return os.path.join(self.root, "jobs", self.job_id)

    @property
    def request_path(self):
        return os.path.join(self.job_dir, "request.json")

    def rank_path(self, rank):
        return os.path.join(self.job_dir, "ranks", f"{rank}.json")

    # ------------------------------------------------------- rank side reads
    def poll_request(self):
        """Cheapest possible check, run once per training step.

        A stat on a tmpfs file is ~1us, which is noise next to a step. Only a
        changed file costs a read.

        The change signature includes the inode, not just mtime: writes go
        through mkstemp+rename so every write is a new inode, while mtime
        granularity is filesystem-dependent and can be as coarse as a second.
        Two epochs inside one mtime tick is exactly the case a retry produces.
        """
        try:
            st = os.stat(self.request_path)
        except FileNotFoundError:
            return None
        sig = (st.st_ino, st.st_mtime_ns, st.st_size)
        if sig == self._request_sig:
            return None
        self._request_sig = sig
        try:
            with open(self.request_path) as fh:
                request = json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
        epoch_id = request.get("epoch_id")
        if epoch_id in self._serviced:
            return None
        self._serviced.add(epoch_id)
        return request

    def mark_serviced(self, epoch_id):
        """Treat `epoch_id` as already seen, so a late file is not serviced twice."""
        self._serviced.add(epoch_id)

    def wait_for_request(self, timeout, poll_interval=0.02):
        """Block until a request appears, for a rank whose peers already have it."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = self.poll_request()
            if request is not None:
                return request
            time.sleep(poll_interval)
        return None

    def wait_for_token(self, rank, epoch_id, timeout, poll_interval=0.05):
        """Block until the agent leaves a token for this rank in this epoch.

        Sleeping in a poll loop is the one thing a process can do across a CRIU
        dump with no state to restore beyond its own stack.
        """
        path = self.rank_path(rank)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with open(path) as fh:
                    token = json.load(fh)
                if token.get("epoch_id") == epoch_id:
                    return token
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(poll_interval)
        return None

    # ------------------------------------------------------ agent side writes
    def put_request(self, payload):
        write_atomic(self.request_path, payload)

    def put_token(self, rank, payload):
        write_atomic(self.rank_path(rank), payload)

    def clear(self, epoch_id=None):
        """Remove the request, but only if it is still the one we handled.

        Without the epoch check, releasing epoch N can delete the request for
        epoch N+1 that a fast coordinator has already published.
        """
        try:
            if epoch_id is not None:
                with open(self.request_path) as fh:
                    if json.load(fh).get("epoch_id") != epoch_id:
                        return False
            os.unlink(self.request_path)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False


class AgentClient:
    """Short-lived outbound calls. Never held open across a checkpoint."""

    def __init__(self, addr, timeout=30.0):
        self.addr = addr
        self.timeout = timeout

    def post_vote(self, job_id, rank, epoch_id, vote, host_pid, findings=None, error=None):
        try:
            with rpc.Client(self.addr, timeout=self.timeout) as client:
                return client.call(
                    "rank_vote",
                    job_id=job_id,
                    rank=rank,
                    epoch_id=epoch_id,
                    vote=vote,
                    host_pid=host_pid,
                    findings=findings or [],
                    error=error,
                )
        except Exception as exc:
            _LOG.error("vote failed", rank=rank, epoch=epoch_id, error=str(exc))
            raise

    def register(self, job_id, rank, host_pid, world_size, gpu_uuids):
        with rpc.Client(self.addr, timeout=self.timeout) as client:
            return client.call(
                "rank_register",
                job_id=job_id,
                rank=rank,
                host_pid=host_pid,
                world_size=world_size,
                gpu_uuids=gpu_uuids,
            )
