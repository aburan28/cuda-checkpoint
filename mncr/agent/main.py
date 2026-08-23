"""The node agent.

Owns every privileged operation on this node and nothing else. It has no opinion
about global consistency - it reports what happened locally and does what the
coordinator tells it, in the order the driver requires.

Ops, in the order a checkpoint uses them:

    rank_register / rank_vote   inbound, from ranks on this node
    prepare                     publish the request, collect local votes
    lock                        verify, then lock each local rank in turn
    checkpoint                  checkpoint each local rank in turn  [COMMITTED]
    dump                        verify, then CRIU-dump each local rank
    restore                     CRIU restore, driver restore, unlock, tokens
    abort                       unlock whatever is locked, release the ranks
"""

import argparse
import os
import shutil
import threading
import time

from mncr import config, log, metrics, rpc
from mncr.errors import DriverError, PreconditionError
from mncr.proto import Vote
from torchckpt.channel import ControlDir

from imagestore.backends import LocalBackend, RemoteBackend, TieredBackend
from imagestore.manifest import Manifest
from imagestore.pipeline import Pipeline

from . import criu as criu_mod
from . import driver as driver_mod
from .jobfile import JobFiles
from .pids import PidResolver
from .verify import Verifier

_LOG = log.get("agent")


class RankRecord(dict):
    @staticmethod
    def make(job_id, rank, host_pid, world_size, gpu_uuids):
        return RankRecord(
            job_id=job_id,
            rank=int(rank),
            host_pid=int(host_pid),
            world_size=int(world_size),
            gpu_uuids=list(gpu_uuids or []),
            registered_at=time.time(),
        )


class Agent:
    def __init__(self, cfg=None, node_name=None, control_root=None):
        self.cfg = cfg or config.load()
        self.node = node_name or os.environ.get("NODE_NAME") or os.uname().nodename
        self.control_root = control_root or os.environ.get("MNCR_CONTROL_ROOT", "/run/mncr")
        self.driver = driver_mod.make(self.cfg)
        self.criu = criu_mod.make(self.cfg)
        self.jobfiles = JobFiles(self.cfg.jobfile_dir, self.cfg.cuda_checkpoint, self.cfg.fake)
        self.pids = PidResolver()
        self.verifier = Verifier(strict=self.cfg.strict_clean)
        self._pipeline = None

        self._lock = threading.Lock()
        self._ranks = {}      # (job_id, rank) -> RankRecord
        self._votes = {}      # (job_id, epoch_id) -> {rank: vote dict}
        self._vote_event = {}  # (job_id, epoch_id) -> threading.Event
        self._locked = {}     # (job_id, epoch_id) -> [host_pid]
        self._checkpointed = {}

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def _build_pipeline(self):
        """Local NVMe first, object store behind it when one is configured.

        Local-first is what ends the stop-the-world window at the dump rather
        than at the upload.
        """
        local = LocalBackend(os.path.join(self.cfg.cache_dir, "shards"))
        prefix = os.environ.get("MNCR_OBJECT_STORE")
        remote = (
            RemoteBackend(
                prefix,
                os.environ.get("MNCR_OBJECT_PUT"),
                os.environ.get("MNCR_OBJECT_GET"),
            )
            if prefix
            else None
        )
        return Pipeline(
            TieredBackend(local, remote),
            shard_bytes=self.cfg.shard_bytes,
            level=self.cfg.zstd_level,
            scratch=self.cfg.cache_dir,
        )

    def _manifest_path(self, image_root, epoch_id):
        return os.path.join(image_root, epoch_id, f"manifest-{self.node}.json")

    # ------------------------------------------------------------- inbound
    def _host_pid(self, reported, peer_pid):
        """The pid the driver, criu and /proc will be given for a rank.

        Never the number the rank sent, taken at face value: inside a pod
        that is the pod's pid, and on the host it is somebody else. The unix
        socket tells us the peer's real pid; over TCP the resolver has to
        work it out, and refuses when it cannot.
        """
        if peer_pid:
            return int(peer_pid), "peer"
        if self.cfg.fake:
            return int(reported), "fake"
        try:
            return self.pids.resolve(reported)
        except LookupError as exc:
            raise PreconditionError(f"cannot map rank pid {reported} to a host pid: {exc}")

    def rank_register(self, job_id, rank, host_pid, world_size, gpu_uuids=None,
                      peer_pid=None):
        host_pid, how = self._host_pid(host_pid, peer_pid)
        record = RankRecord.make(job_id, rank, host_pid, world_size, gpu_uuids)
        with self._lock:
            self._ranks[(job_id, int(rank))] = record
        # The fake backend has no way to discover CUDA processes, so a
        # registration is also how it learns a pid exists.
        if isinstance(self.driver, driver_mod.FakeBackend):
            self.driver.add_pid(host_pid)
        metrics.RANKS.set(len(self._ranks), node=self.node)
        _LOG.info("rank registered", job=job_id, rank=rank, pid=host_pid, pid_from=how,
                  node=self.node)
        return {"node": self.node, "registered": True, "host_pid": host_pid}

    def rank_vote(self, job_id, rank, epoch_id, vote, host_pid, findings=None, error=None,
                  peer_pid=None):
        host_pid, _how = self._host_pid(host_pid, peer_pid)
        key = (job_id, epoch_id)
        with self._lock:
            self._votes.setdefault(key, {})[int(rank)] = {
                "rank": int(rank),
                "vote": vote,
                "host_pid": int(host_pid),
                "findings": findings or [],
                "error": error,
                "at": time.time(),
            }
            # The pid a rank votes with is authoritative: after a restore it may
            # differ from what it registered with.
            record = self._ranks.get((job_id, int(rank)))
            if record:
                record["host_pid"] = int(host_pid)
            event = self._vote_event.get(key)
        if event:
            event.set()
        _LOG.info("vote", job=job_id, rank=rank, epoch=epoch_id, vote=vote)
        for finding in findings or []:
            metrics.GATE_FINDINGS.inc(
                gate="rank_self_check", kind=finding.get("kind", "unknown"),
                node=self.node,
            )
        return {"accepted": True}

    # ------------------------------------------------------------ local view
    def local_ranks(self, job_id, ranks=None):
        with self._lock:
            if ranks is None:
                return [
                    r for (j, _), r in sorted(self._ranks.items()) if j == job_id
                ]
            wanted = {int(x) for x in ranks}
            return [
                r
                for (j, rk), r in sorted(self._ranks.items())
                if j == job_id and rk in wanted
            ]

    def _registered_ranks(self, job_id, ranks):
        """local_ranks, refusing silently to narrow the set.

        The coordinator names the ranks it expects on this node. If one of
        them is not registered here, answering with the rest would let an
        epoch proceed - and commit - with a rank that never voted. That is the
        one thing the two-phase commit exists to prevent, so it is an error at
        the point where it is still free.
        """
        local = self.local_ranks(job_id, ranks)
        if ranks is not None:
            missing = sorted({int(x) for x in ranks} - {int(r["rank"]) for r in local})
            if missing:
                raise PreconditionError(
                    f"ranks {missing} of {job_id} are not registered on {self.node}"
                )
        return local

    def status(self, job_id=None):
        with self._lock:
            ranks = [dict(r) for (j, _), r in sorted(self._ranks.items())
                     if job_id is None or j == job_id]
        return {
            "node": self.node,
            "ranks": ranks,
            "driver": type(self.driver).__name__,
            "criu": type(self.criu).__name__,
            "fake": self.cfg.fake,
        }

    # -------------------------------------------------------------- pre-dump
    def pre_dump(self, job_id, epoch_id, image_root, ranks=None, external=None):
        """Copy pages while the ranks still run, to shrink the stop window.

        Runs before the lock, deliberately: once the driver has copied device
        memory into host allocations, a pre-dump would be racing the very pages
        that are about to change.
        """
        local = self.local_ranks(job_id, ranks)
        done = []
        for record in local:
            images_dir = os.path.join(
                image_root, epoch_id, f"rank-{record['rank']}", "pre"
            )
            self.criu.pre_dump(
                record["host_pid"], images_dir, external=external or ()
            )
            done.append({"rank": record["rank"], "path": images_dir})
        _LOG.info("pre-dumped", job=job_id, epoch=epoch_id, count=len(done))
        return {"node": self.node, "pre_dumps": done}

    # --------------------------------------------------------------- prepare
    def prepare(self, job_id, epoch_id, ranks=None, lookahead=1, wait_timeout=None,
                vote_timeout=None):
        """Publish the request and block until every local rank has voted."""
        local = self._registered_ranks(job_id, ranks)
        if not local:
            raise PreconditionError(f"no registered ranks for {job_id} on {self.node}")

        key = (job_id, epoch_id)
        with self._lock:
            self._votes.setdefault(key, {})
            self._vote_event[key] = threading.Event()

        control = ControlDir(self.control_root, job_id)
        control.put_request(
            {
                "epoch_id": epoch_id,
                "action": "checkpoint",
                "lookahead": int(lookahead),
                "wait_timeout": float(wait_timeout or self.cfg.checkpoint_timeout),
                "issued_at": time.time(),
            }
        )

        deadline = time.monotonic() + float(vote_timeout or self.cfg.quiesce_timeout)
        expected = {r["rank"] for r in local}
        while time.monotonic() < deadline:
            with self._lock:
                have = set(self._votes.get(key, {}))
                event = self._vote_event[key]
            if expected <= have:
                break
            event.clear()
            event.wait(timeout=min(1.0, max(0.01, deadline - time.monotonic())))

        with self._lock:
            votes = dict(self._votes.get(key, {}))
        missing = sorted(expected - set(votes))
        for rank in missing:
            votes[rank] = {"rank": rank, "vote": Vote.TIMEOUT.value, "host_pid": None}

        clean = all(v["vote"] == Vote.CLEAN.value for v in votes.values())
        _LOG.info(
            "prepare complete",
            job=job_id,
            epoch=epoch_id,
            node=self.node,
            clean=clean,
            missing=missing,
        )
        return {
            "node": self.node,
            "clean": clean,
            "votes": [votes[r] for r in sorted(votes)],
            "missing": missing,
        }

    # ------------------------------------------------------------------ lock
    def lock(self, job_id, epoch_id, ranks=None, timeout_ms=None):
        local = self._registered_ranks(job_id, ranks)
        timeout_ms = int(timeout_ms or self.cfg.lock_timeout_ms)
        locked = []
        try:
            for record in local:
                pid = record["host_pid"]
                self._gate(self.verifier.before_lock, pid, "before_lock")
                with metrics.DRIVER_SECONDS.time(action="lock", node=self.node):
                    self._driver_call(self.driver.lock, "lock", pid, timeout_ms)
                locked.append(pid)
        except Exception:
            # Still abortable: undo the partial lock before propagating.
            for pid in reversed(locked):
                try:
                    self.driver.unlock(pid)
                except DriverError as exc:
                    _LOG.error("unlock during rollback failed", pid=pid, error=str(exc))
            raise
        with self._lock:
            self._locked[(job_id, epoch_id)] = locked
        _LOG.info("locked", job=job_id, epoch=epoch_id, count=len(locked))
        return {"node": self.node, "locked": locked}

    # ------------------------------------------------------------ checkpoint
    def checkpoint(self, job_id, epoch_id, ranks=None):
        """Past this call there is no way back for the affected ranks."""
        local = self._registered_ranks(job_id, ranks)
        done = []
        for record in local:
            pid = record["host_pid"]
            with metrics.DRIVER_SECONDS.time(action="checkpoint", node=self.node):
                self._driver_call(self.driver.checkpoint, "checkpoint", pid)
            done.append(pid)
        with self._lock:
            self._checkpointed[(job_id, epoch_id)] = done
        _LOG.info("checkpointed", job=job_id, epoch=epoch_id, count=len(done))
        return {"node": self.node, "checkpointed": done}

    # ------------------------------------------------------------------ dump
    def dump(self, job_id, epoch_id, image_root, ranks=None, external=None,
             store=True):
        local = self._registered_ranks(job_id, ranks)
        images, entries = [], []
        for record in local:
            pid = record["host_pid"]
            self._gate(self.verifier.before_dump, pid, "before_dump")
            images_dir = os.path.join(image_root, epoch_id, f"rank-{record['rank']}")
            self.criu.dump(pid, images_dir, external=external or ())
            images.append({"rank": record["rank"], "path": images_dir, "pid": pid})
            entries.append(
                {
                    "rank": record["rank"],
                    "node": self.node,
                    "host_pid": pid,
                    "images_dir": images_dir,
                }
            )

        manifest_path = None
        if store and entries:
            store_started = time.time()
            # Shard, compress and checksum. A cross-node restore has no other
            # way to reach these images: they are on this node's disk.
            manifest = self.pipeline.store_epoch(
                epoch_id,
                job_id,
                epoch_id,
                entries,
                source_uuids={self.node: self._node_uuids(local)},
                world_size=len(entries),
            )
            manifest_path = manifest.save(self._manifest_path(image_root, epoch_id))
            metrics.IMAGE_SECONDS.observe(
                time.time() - store_started, node=self.node
            )
            for rank_image in manifest.ranks:
                metrics.IMAGE_BYTES.observe(rank_image.stored_bytes, node=self.node)

        _LOG.info(
            "dumped",
            job=job_id,
            epoch=epoch_id,
            count=len(images),
            stored=bool(manifest_path),
        )
        return {"node": self.node, "images": images, "manifest": manifest_path}

    def _node_uuids(self, ranks):
        seen = []
        for record in ranks:
            for uuid in record.get("gpu_uuids", []):
                if uuid not in seen:
                    seen.append(uuid)
        return seen

    # --------------------------------------------------------------- restore
    def restore(self, job_id, epoch_id, image_root, ranks=None, device_map=None,
                init_method=None, from_images=True, world_size=None):
        """Restore, unlock, and release the ranks - in checkpoint order.

        The vendor's r610 demo is explicit that processes must be restored and
        unlocked in the same order they were checkpointed, so local_ranks()
        returning a stable sort is a correctness property, not a tidiness one.
        """
        local = self._restore_targets(job_id, ranks, from_images)
        restored = []
        for record in local:
            if from_images:
                images_dir = self._materialize(
                    job_id, epoch_id, image_root, record["rank"]
                )
                # criu reports the pid it created. On a node that did not take
                # the checkpoint this is the only source for it.
                pid = self.criu.restore(images_dir)
                self._adopt(job_id, record, pid)
            else:
                pid = record["host_pid"]
            self.driver.restore(pid, device_map=device_map)
            self.driver.unlock(pid)
            restored.append(pid)

        self._release(
            job_id,
            epoch_id,
            local,
            restored=from_images,
            aborted=False,
            init_method=init_method,
            device_map=device_map,
            world_size=world_size,
        )
        _LOG.info("restored", job=job_id, epoch=epoch_id, count=len(restored))
        return {"node": self.node, "restored": restored}

    # ----------------------------------------------------------------- abort
    def abort(self, job_id, epoch_id, ranks=None, reason="aborted",
              init_method=None, world_size=None):
        """Undo an epoch that has not crossed the commit point."""
        with self._lock:
            locked = list(self._locked.pop((job_id, epoch_id), []))
            committed = (job_id, epoch_id) in self._checkpointed
        if committed:
            raise PreconditionError(
                f"epoch {epoch_id} is past the commit point on {self.node}; "
                f"it cannot be aborted in place"
            )
        for pid in reversed(locked):
            try:
                self.driver.unlock(pid)
            except DriverError as exc:
                _LOG.error("unlock during abort failed", pid=pid, error=str(exc))

        local = self.local_ranks(job_id, ranks)
        self._release(
            job_id,
            epoch_id,
            local,
            restored=False,
            aborted=True,
            init_method=init_method,
            device_map=None,
            world_size=world_size,
            reason=reason,
        )
        _LOG.info("aborted", job=job_id, epoch=epoch_id, unlocked=len(locked))
        return {"node": self.node, "unlocked": locked}

    def _materialize(self, job_id, epoch_id, image_root, rank):
        """Ensure this rank's image is on local disk, fetching it if not.

        On the node that took the checkpoint the directory is already there. On
        any other node it is not, and the shards have to come back out of the
        store - which is the case that makes migration work at all.
        """
        images_dir = os.path.join(image_root, epoch_id, f"rank-{rank}")
        if os.path.isdir(images_dir) and os.listdir(images_dir):
            return images_dir

        manifest = self._find_manifest(image_root, epoch_id, rank)
        if manifest is None:
            raise PreconditionError(
                f"no image for rank {rank} of {job_id} on {self.node}, and no "
                f"manifest under {image_root}/{epoch_id} lists that rank"
            )
        _LOG.info("fetching image", job=job_id, epoch=epoch_id, rank=rank, node=self.node)
        return self.pipeline.fetch_rank(manifest, rank, images_dir)

    def _find_manifest(self, image_root, epoch_id, rank):
        """The manifest holding `rank`.

        Each node writes its own manifest covering only its local ranks - no
        coordination needed at dump time. The cost is at read time: a node
        restoring a rank that migrated in has to look through all of them.
        """
        directory = os.path.join(image_root, epoch_id)
        try:
            names = sorted(os.listdir(directory))
        except FileNotFoundError:
            return None
        for name in names:
            if not (name.startswith("manifest-") and name.endswith(".json")):
                continue
            try:
                manifest = Manifest.load(os.path.join(directory, name))
            except (OSError, ValueError, KeyError):
                continue
            ranks = {
                (r["rank"] if isinstance(r, dict) else r.rank) for r in manifest.ranks
            }
            if rank in ranks:
                return manifest
        return None

    def _driver_call(self, fn, action, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            metrics.DRIVER_ERRORS.inc(action=action, node=self.node)
            raise

    def _gate(self, fn, pid, gate):
        try:
            return fn(pid)
        except Exception as exc:
            for finding in getattr(exc, "findings", []) or []:
                metrics.GATE_FINDINGS.inc(
                    gate=gate, kind=finding.kind, node=self.node
                )
            raise

    def _restore_targets(self, job_id, ranks, from_images):
        """Records to restore.

        A restore onto the original node uses the local registry. A restore onto
        a fresh node has no registry to consult, so the requested rank list is
        authoritative and the pids come from criu.
        """
        local = self.local_ranks(job_id, ranks)
        if not from_images:
            return local
        if not ranks:
            if local:
                return local
            raise PreconditionError(
                f"restore on {self.node} needs an explicit rank list: no ranks "
                f"of {job_id} are registered here"
            )
        # The requested list is authoritative. A rank already registered here
        # keeps its record; one arriving from elsewhere gets a fresh one and
        # its pid from criu. Returning only the registered ones would restore
        # part of what was asked and report success for all of it.
        known = {int(r["rank"]): r for r in local}
        return [
            known.get(rank) or RankRecord.make(job_id, rank, 0, len(ranks), [])
            for rank in sorted(int(r) for r in ranks)
        ]

    def _adopt(self, job_id, record, pid):
        """Take ownership of a rank restored onto this node."""
        record["host_pid"] = int(pid)
        with self._lock:
            self._ranks[(job_id, record["rank"])] = record
        if isinstance(self.driver, driver_mod.FakeBackend):
            self.driver.add_pid(pid, driver_mod.STATE_CHECKPOINTED)
        _LOG.info("rank adopted after restore", job=job_id,
                  rank=record["rank"], pid=pid, node=self.node)
        return record

    def _release(self, job_id, epoch_id, ranks, restored, aborted, init_method,
                 device_map, world_size, reason=None):
        """Write the token each rank is polling for.

        An aborted epoch still needs the rank to rebuild: the teardown already
        happened, so there is no such thing as resuming without re-initialising.
        """
        control = ControlDir(self.control_root, job_id)
        for record in ranks:
            control.put_token(
                record["rank"],
                {
                    "epoch_id": epoch_id,
                    "rank": record["rank"],
                    "world_size": int(world_size or record["world_size"]),
                    "restored": bool(restored),
                    "aborted": bool(aborted),
                    "init_method": init_method,
                    "device_map": device_map or {},
                    "reason": reason,
                    "at": time.time(),
                },
            )
        control.clear(epoch_id)

    def forget(self, job_id, ranks=None):
        """Relinquish ranks that now live somewhere else.

        Without this, a rank migrated to another node stays in the source
        agent's registry and the next epoch tries to lock a pid that is gone.
        The coordinator calls it as part of every restore that moves ranks.
        """
        with self._lock:
            if ranks is None:
                dropped = [k for k in self._ranks if k[0] == job_id]
            else:
                wanted = {int(r) for r in ranks}
                dropped = [
                    k for k in self._ranks if k[0] == job_id and k[1] in wanted
                ]
            for key in dropped:
                self._ranks.pop(key, None)
            metrics.RANKS.set(len(self._ranks), node=self.node)
        _LOG.info("ranks relinquished", job=job_id, node=self.node,
                  ranks=[k[1] for k in dropped])
        return {"node": self.node, "forgotten": [k[1] for k in dropped]}

    def reap(self, job_id=None):
        """Drop registrations whose process no longer exists.

        Belt and braces for the case where a node dies mid-migration and never
        gets a forget call.
        """
        with self._lock:
            dead = [
                key
                for key, record in self._ranks.items()
                if (job_id is None or key[0] == job_id)
                and not self.pids.alive(record["host_pid"])
            ]
            for key in dead:
                self._ranks.pop(key, None)
        if dead:
            _LOG.info("reaped dead ranks", node=self.node, count=len(dead))
        return {"node": self.node, "reaped": [k[1] for k in dead]}

    # ------------------------------------------------------------- retention
    def delete_images(self, job_id, epoch_id, image_root):
        """Remove this node's images for an epoch, shards included.

        Best effort by design: retention that stops at the first error leaves
        the disk full, which is the problem it exists to prevent.
        """
        directory = os.path.join(image_root, epoch_id)
        shards = {"removed": 0, "failed": []}
        manifest_path = self._manifest_path(image_root, epoch_id)
        if os.path.exists(manifest_path):
            try:
                shards = self.pipeline.delete_epoch(Manifest.load(manifest_path))
            except Exception as exc:  # noqa: BLE001
                _LOG.warn("shard deletion failed", epoch=epoch_id, error=str(exc))
        removed_dir = False
        try:
            shutil.rmtree(directory)
            removed_dir = True
        except FileNotFoundError:
            pass
        except OSError as exc:
            _LOG.warn("image directory not removed", path=directory, error=str(exc))
        _LOG.info(
            "images deleted",
            job=job_id,
            epoch=epoch_id,
            node=self.node,
            shards=shards["removed"],
            directory=removed_dir,
        )
        return {"node": self.node, "shards": shards["removed"], "directory": removed_dir}

    # ------------------------------------------------------------- job files
    def job_create(self, job_id):
        return {"path": self.jobfiles.create(job_id)}

    def job_env(self, job_id):
        return {"env": self.jobfiles.env_for(job_id)}

    def job_remove(self, job_id):
        return {"removed": self.jobfiles.remove(job_id)}


def build_server(agent, addr=None):
    server = rpc.Server(addr or agent.cfg.agent_addr, name="agent")
    for name in (
        "rank_register",
        "rank_vote",
        "pre_dump",
        "prepare",
        "lock",
        "checkpoint",
        "dump",
        "restore",
        "abort",
        "forget",
        "reap",
        "delete_images",
        "status",
        "job_create",
        "job_env",
        "job_remove",
    ):
        server.op(name)(getattr(agent, name))
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(description="mncr node agent")
    ap.add_argument("--addr", default=None, help="listen address")
    ap.add_argument("--rank-addr", default=None, help="address ranks connect to")
    ap.add_argument("--node", default=None)
    ap.add_argument("--control-root", default=None)
    ap.add_argument("--metrics-port", type=int, default=9180,
                    help="Prometheus endpoint; 0 disables it")
    args = ap.parse_args(argv)

    cfg = config.load()
    agent = Agent(cfg, node_name=args.node, control_root=args.control_root)
    servers = [build_server(agent, args.addr or cfg.agent_addr).start()]
    rank_addr = args.rank_addr or cfg.rank_addr
    if rank_addr and rank_addr != (args.addr or cfg.agent_addr):
        servers.append(build_server(agent, rank_addr).start())

    metrics_server = metrics.serve(args.metrics_port) if args.metrics_port else None
    _LOG.info(
        "agent ready", node=agent.node, fake=cfg.fake, metrics=args.metrics_port or None
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.stop()
        if metrics_server:
            metrics_server.shutdown()
            metrics_server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
