"""Two-phase commit across nodes.

The rule this file exists to enforce: collect every vote before issuing a single
checkpoint call. Before the commit point a failure costs one drained step; after
it, the affected ranks have released their GPU resources and the driver offers
no way back, so the epoch is lost and the job falls back to its last good image.

Everything else here is bookkeeping around that rule.
"""

import time

from mncr import log, metrics
from mncr.errors import AbortableError, TerminalError, UnreleasedAbortError
from mncr.proto import Epoch, Phase, Vote

from .rendezvous import Rendezvous

_LOG = log.get("coord.epoch")


class EpochRunner:
    def __init__(self, pool, store, cfg, node_info=None):
        self.pool = pool
        self.store = store
        self.cfg = cfg
        self.node_info = node_info if node_info is not None else {}
        # Reads node_info at call time, so agents registered later are seen.
        self.rendezvous = Rendezvous(self.node_info)
        # Recorded so an abort - which happens outside checkpoint()'s argument
        # list - releases ranks with the same collective backend they were
        # torn down from.
        self.default_backend = "nccl"

    # ------------------------------------------------------------- utilities
    def _save(self, epoch, phase=None, note=None, error=None):
        if phase is not None:
            previous = epoch.phase
            entered = epoch["history"][-1]["at"] if epoch["history"] else epoch["created_at"]
            epoch.set_phase(phase, note)
            metrics.PHASE_SECONDS.observe(
                max(0.0, time.time() - entered), phase=previous.value
            )
        if error is not None:
            epoch["error"] = str(error)
        self.store.put(epoch)
        return epoch

    def _per_node_ranks(self, epoch):
        return {
            node: {"ranks": [r["rank"] for r in epoch.ranks_on(node)]}
            for node in epoch.nodes()
        }

    def _abort(self, epoch, reason, init_method=None):
        """Undo an epoch that has not committed. Best effort by design."""
        _LOG.warn("aborting epoch", epoch=epoch["epoch_id"], reason=reason)
        results, errors = self.pool.fanout(
            epoch.nodes(),
            "abort",
            per_node_args=self._per_node_ranks(epoch),
            job_id=epoch["job_id"],
            epoch_id=epoch["epoch_id"],
            reason=reason,
            init_method=init_method,
            backend=self.default_backend,
            world_size=len(epoch["ranks"]),
        )
        # Whether every rank was released. The epoch is still aborted rather
        # than committed - no GPU state was touched - but ranks on a node the
        # abort could not reach have torn down and are waiting for a token
        # that will not come, and the job is not intact. The controller reads
        # this field; it must not report jobIntact=true on the phase alone.
        epoch["released"] = not errors
        if errors:
            _LOG.error("abort incomplete; ranks unreleased", epoch=epoch["epoch_id"],
                       errors=errors)
            reason = f"{reason}; abort incomplete on {sorted(errors)}, ranks there are unreleased"
        self._save(epoch, Phase.ABORTED, note=reason, error=reason)
        metrics.EPOCHS.inc(outcome="aborted", job=epoch["job_id"])
        metrics.EPOCH_SECONDS.observe(
            time.time() - epoch["created_at"], outcome="aborted"
        )
        if errors:
            # Raised here, ahead of the caller's own AbortableError, so the
            # distinction reaches whoever reports on the job.
            raise UnreleasedAbortError(
                f"epoch {epoch['epoch_id']} aborted ({reason}); the job is not intact"
            )
        return results, errors

    def _fail(self, epoch, reason):
        _LOG.error("epoch lost past commit point", epoch=epoch["epoch_id"], reason=reason)
        # Nobody releases the ranks after this, so nobody clears the request
        # file either. Do it here, best effort: a rank launched later on one
        # of these nodes must not find an epoch waiting for it.
        _results, errors = self.pool.fanout(
            epoch.nodes(), "clear_request",
            job_id=epoch["job_id"], epoch_id=epoch["epoch_id"], _timeout=30,
        )
        if errors:
            _LOG.warn("request files not cleared", epoch=epoch["epoch_id"], errors=errors)
        self._save(epoch, Phase.FAILED, note=reason, error=reason)
        metrics.EPOCHS.inc(outcome="failed", job=epoch["job_id"])
        metrics.EPOCH_SECONDS.observe(
            time.time() - epoch["created_at"], outcome="failed"
        )
        raise TerminalError(
            f"epoch {epoch['epoch_id']} failed after the commit point: {reason}. "
            f"Ranks cannot be resumed in place; restore the last good image."
        )

    # -------------------------------------------------------------- the flow
    def checkpoint(self, job_id, ranks, mode="continue", image_root=None,
                   init_method=None, reason="manual", pre_dump=False,
                   backend="nccl"):
        """Run one checkpoint epoch.

        mode="continue"  checkpoint and keep the job running (fault tolerance)
        mode="stop"      checkpoint and leave the ranks down (preemption)
        pre_dump=True    copy pages before the lock, shrinking the stop window
        """
        image_root = image_root or self.cfg.image_dir
        self.default_backend = backend
        epoch = Epoch.make(job_id, ranks, reason=reason)
        # A fresh address for the rebuild unless the caller chose one. The
        # same string goes to every rank, in the resume token and in the abort
        # token alike - both paths end in init_process_group.
        if init_method is None:
            init_method = self.rendezvous.new(ranks)
        epoch["init_method"] = init_method
        self.store.put(epoch)
        started = time.time()
        per_node = self._per_node_ranks(epoch)
        nodes = epoch.nodes()

        # --------------------------------------------------------- pre-dump
        if pre_dump:
            # Before the lock on purpose: this is the only window where the
            # process is still running and its pages are worth copying early.
            _results, errors = self.pool.fanout(
                nodes,
                "pre_dump",
                per_node_args=per_node,
                job_id=job_id,
                epoch_id=epoch["epoch_id"],
                image_root=image_root,
                _timeout=self.cfg.dump_timeout,
            )
            if errors:
                # A failed pre-dump costs time, not correctness. Carry on.
                _LOG.warn("pre-dump incomplete", epoch=epoch["epoch_id"], errors=errors)

        # ---------------------------------------------------------- prepare
        self._save(epoch, Phase.PREPARING)
        results, errors = self.pool.fanout(
            nodes,
            "prepare",
            per_node_args=per_node,
            job_id=job_id,
            epoch_id=epoch["epoch_id"],
            # The rank waits for its token through lock, checkpoint, dump and
            # restore; its budget has to outlast the dump, which is the slow
            # part, or a large image ends with every rank timing out just as
            # its token arrives.
            wait_timeout=self.cfg.checkpoint_timeout + self.cfg.dump_timeout,
            vote_timeout=self.cfg.quiesce_timeout,
            _timeout=self.cfg.quiesce_timeout + 60,
        )
        votes = {}
        for node, result in results.items():
            for vote in result.get("votes", []):
                votes[int(vote["rank"])] = vote
        epoch["votes"] = votes
        for vote in votes.values():
            metrics.VOTES.inc(verdict=vote["vote"], job=job_id)

        if errors:
            self._abort(epoch, f"prepare failed on {sorted(errors)}", init_method)
            raise AbortableError(f"prepare failed: {errors}")

        # The rule, checked rather than assumed: every rank in the epoch has a
        # vote. The agents refuse to narrow the rank set, so this should never
        # fire; it is here so that a bug on their side cannot commit an epoch
        # that is missing a rank.
        missing = sorted({int(r["rank"]) for r in epoch["ranks"]} - set(votes))
        if missing:
            self._abort(epoch, f"no vote from ranks {missing}", init_method)
            raise AbortableError(f"no vote from ranks {missing}")

        dirty = [v for v in votes.values() if v["vote"] != Vote.CLEAN.value]
        if dirty:
            detail = ", ".join(
                f"rank {v['rank']}: {v['vote']}"
                + (f" ({v.get('error')})" if v.get("error") else "")
                for v in dirty[:8]
            )
            self._abort(epoch, f"ranks not clean: {detail}", init_method)
            raise AbortableError(f"ranks not clean: {detail}")

        self._save(epoch, Phase.PREPARED, note=f"{len(votes)} clean votes")

        # ------------------------------------------------------------- lock
        results, errors = self.pool.fanout(
            nodes,
            "lock",
            per_node_args=per_node,
            job_id=job_id,
            epoch_id=epoch["epoch_id"],
            timeout_ms=self.cfg.lock_timeout_ms,
            _timeout=(self.cfg.lock_timeout_ms / 1000.0) + 120,
        )
        if errors:
            self._abort(epoch, f"lock failed on {sorted(errors)}", init_method)
            raise AbortableError(f"lock failed: {errors}")
        self._save(epoch, Phase.LOCKED)
        locked_at = time.time()

        # ------------------------------------------------ COMMIT POINT below
        # Recorded before the first checkpoint call goes out, not after the
        # last returns: a coordinator that dies during the fan-out must come
        # back believing the epoch committed, because some rank may have.
        # Classifying it as abortable would invite an unlock on a process
        # that has already released its GPU.
        self._save(epoch, Phase.CHECKPOINTED, note="checkpoint calls in flight")
        results, errors = self.pool.fanout(
            nodes,
            "checkpoint",
            per_node_args=per_node,
            job_id=job_id,
            epoch_id=epoch["epoch_id"],
            _timeout=self.cfg.checkpoint_timeout,
        )
        if errors:
            self._fail(epoch, f"checkpoint failed on {sorted(errors)}: {errors}")

        # ------------------------------------------------------------- dump
        results, errors = self.pool.fanout(
            nodes,
            "dump",
            per_node_args=per_node,
            job_id=job_id,
            epoch_id=epoch["epoch_id"],
            image_root=image_root,
            # Keeping the job running is the whole point of mode="continue",
            # and criu kills what it dumps unless told otherwise.
            leave_running=(mode == "continue"),
            _timeout=self.cfg.dump_timeout,
        )
        if errors:
            self._fail(epoch, f"dump failed on {sorted(errors)}: {errors}")

        images = []
        for node, result in results.items():
            for image in result.get("images", []):
                images.append({**image, "node": node})
        epoch["image_id"] = epoch["epoch_id"]
        epoch["images"] = images
        # Where each rank's image was written. `ranks` is rewritten by every
        # restore to say where the ranks are now; this is not.
        epoch["dumped_on"] = {str(i["rank"]): i["node"] for i in images}
        self._save(epoch, Phase.DUMPED, note=f"{len(images)} images")

        requirements = self._requirements_for(nodes)
        self.store.mark_last_good(job_id, epoch["epoch_id"], epoch["image_id"], requirements)

        # ----------------------------------------------------- put it back
        if mode == "continue":
            results, errors = self.pool.fanout(
                nodes,
                "restore",
                per_node_args=per_node,
                job_id=job_id,
                epoch_id=epoch["epoch_id"],
                image_root=image_root,
                from_images=False,   # the processes never died
                init_method=init_method,
                backend=backend,
                world_size=len(epoch["ranks"]),
                _timeout=self.cfg.checkpoint_timeout,
            )
            if errors:
                self._fail(epoch, f"resume failed on {sorted(errors)}: {errors}")
            self._save(epoch, Phase.RESTORING)
            self._save(epoch, Phase.RESUMED)
            self._save(epoch, Phase.RUNNING, note="checkpoint and continue")
        else:
            _LOG.info("epoch left stopped", epoch=epoch["epoch_id"], mode=mode)

        epoch["seconds"] = round(time.time() - started, 3)
        self.store.put(epoch)
        metrics.EPOCHS.inc(outcome="succeeded", job=job_id, mode=mode)
        metrics.EPOCH_SECONDS.observe(epoch["seconds"], outcome="succeeded")
        # The number an operator feels: how long the job was not running.
        metrics.STOPPED_SECONDS.observe(time.time() - locked_at, mode=mode)
        _LOG.info(
            "checkpoint complete",
            epoch=epoch["epoch_id"],
            job=job_id,
            mode=mode,
            seconds=epoch["seconds"],
            images=len(images),
        )
        return epoch

    # ----------------------------------------------------------- restore
    def restore(self, job_id, epoch_id, ranks, image_root=None, device_maps=None,
                init_method=None, backend="nccl"):
        """Bring a dumped job back, possibly onto different nodes.

        device_maps is {node: "old=new,..."} because each node's target UUIDs
        differ. A node restoring onto its original hardware still gets an
        explicit identity map rather than None - being explicit here is what
        makes a wrong map a placement error instead of a silent mis-restore.
        """
        image_root = image_root or self.cfg.image_dir
        stored = self.store.get(epoch_id)
        epoch = Epoch(stored or Epoch.make(job_id, ranks, reason="restore"))
        previous = {int(r["rank"]): r["node"] for r in epoch.get("ranks", [])}
        # A second restore of the same epoch must fetch from where the image
        # was dumped, not from where the first restore put the ranks.
        dumped_on = {
            int(k): v for k, v in (epoch.get("dumped_on") or {}).items()
        } or previous
        epoch["ranks"] = [dict(r) for r in ranks]
        epoch["phase"] = Phase.DUMPED.value
        # A fresh mark, so the RESTORING transition measures the restore and
        # not the time the image sat on disk.
        epoch["history"].append(
            {"at": time.time(), "from": Phase.DUMPED.value, "to": Phase.DUMPED.value,
             "note": "restore requested"}
        )
        # Rank 0 may have moved; the address is computed from where it is now.
        if init_method is None:
            init_method = self.rendezvous.new(ranks)
        epoch["init_method"] = init_method
        self._save(epoch, Phase.RESTORING)

        per_node = self._per_node_ranks(epoch)
        device_maps = device_maps or {}
        args = {
            node: {
                **per_node[node],
                "device_map": device_maps.get(node),
                # Where each rank's image was written, so a node that never
                # saw it can find the manifest in the store.
                "sources": {
                    int(r["rank"]): dumped_on.get(int(r["rank"]))
                    for r in epoch.ranks_on(node)
                    if dumped_on.get(int(r["rank"]))
                },
            }
            for node in epoch.nodes()
        }
        results, errors = self.pool.fanout(
            epoch.nodes(),
            "restore",
            per_node_args=args,
            job_id=job_id,
            epoch_id=epoch_id,
            image_root=image_root,
            from_images=True,
            init_method=init_method,
            backend=backend,
            world_size=len(epoch["ranks"]),
            _timeout=self.cfg.dump_timeout,
        )
        if errors:
            self._fail(epoch, f"restore failed on {sorted(errors)}: {errors}")

        self._relinquish(job_id, previous, ranks)
        self._save(epoch, Phase.RESUMED)
        self._save(epoch, Phase.RUNNING, note="restored from images")
        _LOG.info("restore complete", epoch=epoch_id, job=job_id, nodes=len(results))
        return epoch

    def _relinquish(self, job_id, previous, ranks):
        """Have the old nodes drop ranks that moved elsewhere."""
        now = {int(r["rank"]): r["node"] for r in ranks}
        moved = {}
        for rank, old_node in previous.items():
            new_node = now.get(rank)
            if new_node and new_node != old_node:
                moved.setdefault(old_node, []).append(rank)
        if not moved:
            return {}
        results, errors = self.pool.fanout(
            list(moved),
            "forget",
            per_node_args={node: {"ranks": ranks_} for node, ranks_ in moved.items()},
            job_id=job_id,
        )
        if errors:
            # A node that cannot let go is not fatal for this restore, but the
            # next epoch would try to lock a pid that has gone.
            _LOG.warn("relinquish incomplete", errors=errors, moved=moved)
        return results

    def _requirements_for(self, nodes):
        from .placement import ImageRequirements

        for node in nodes:
            info = self.node_info.get(node)
            if info:
                return ImageRequirements.from_node(info).to_dict()
        return {}
