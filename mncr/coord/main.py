"""Coordinator service and CLI.

Holds no GPU state and issues no privileged call. Its entire job is deciding
when it is safe to cross the commit point, and remembering what happened.
"""

import argparse
import json
import time

from mncr import config, log, metrics, rpc
from mncr.proto import RankRef

from .client import AgentPool
from .epoch import EpochRunner
from .placement import ImageRequirements, select
from .devicemap import build_pairs, to_cli
from .store import EpochStore

_LOG = log.get("coord")


class Coordinator:
    def __init__(self, cfg=None, agents=None, store=None, node_info=None):
        self.cfg = cfg or config.load()
        self.pool = AgentPool(agents or {})
        self.store = store or EpochStore()
        self.node_info = dict(node_info or {})
        self.runner = EpochRunner(self.pool, self.store, self.cfg, self.node_info)
        self._jobs = {}   # job_id -> [RankRef]

    # --------------------------------------------------------------- ops
    def register_agent(self, node, addr, info=None):
        """Add a node. Without `info`, the agent is asked to describe itself."""
        self.pool.add(node, addr)
        if info is None:
            try:
                info = self.pool.call(node, "describe", _timeout=60)
            except Exception as exc:  # noqa: BLE001
                _LOG.warn("agent did not describe itself", node=node, error=str(exc))
                info = None
        if info:
            self.node_info[node] = info
            self.runner.node_info[node] = info
        _LOG.info(
            "agent registered", node=node, addr=addr,
            ip=(info or {}).get("ip"), gpus=(info or {}).get("gpu_count"),
        )
        return {"registered": True, "nodes": self.pool.nodes(), "info": info or {}}

    def register_job(self, job_id, ranks):
        refs = [RankRef(r) for r in ranks]
        self._jobs[job_id] = refs
        _LOG.info("job registered", job=job_id, ranks=len(refs), nodes=len({r["node"] for r in refs}))
        return {"job_id": job_id, "ranks": len(refs)}

    def job_ranks(self, job_id):
        if job_id not in self._jobs:
            raise KeyError(f"unknown job {job_id}")
        return self._jobs[job_id]

    def checkpoint(self, job_id, mode="continue", image_root=None, init_method=None,
                   reason="manual", pre_dump=False, backend="nccl"):
        epoch = self.runner.checkpoint(
            job_id,
            self.job_ranks(job_id),
            mode=mode,
            image_root=image_root,
            init_method=init_method,
            reason=reason,
            pre_dump=pre_dump,
            backend=backend,
        )
        return {"epoch_id": epoch["epoch_id"], "phase": epoch["phase"],
                "images": len(epoch.get("images", [])), "seconds": epoch.get("seconds")}

    def restore(self, job_id, epoch_id, targets=None, image_root=None,
                init_method=None, backend="nccl"):
        """Restore onto `targets` ({node: [ranks]}), defaulting to where it ran."""
        source = self.store.get(epoch_id)
        if source is None:
            raise KeyError(f"unknown epoch {epoch_id}")

        if targets:
            ranks = self._replace_placement(source, targets)
        else:
            ranks = [RankRef(r) for r in source["ranks"]]
        device_maps = self._maps_for(source, ranks)

        epoch = self.runner.restore(
            job_id, epoch_id, ranks, image_root=image_root,
            device_maps=device_maps, init_method=init_method, backend=backend,
        )
        self._jobs[job_id] = ranks
        return {"epoch_id": epoch_id, "phase": epoch["phase"],
                "device_maps": device_maps}

    def _node_uuids(self, node):
        info = self.node_info.get(node) or {}
        raw = info.get("gpu_uuids") or ""
        return [u for u in str(raw).split("|") if u]

    def _replace_placement(self, source, targets):
        """Re-place ranks onto new nodes."""
        by_rank = {int(r["rank"]): r for r in source.get("ranks", [])}
        ranks = []
        for node, rank_ids in targets.items():
            for rank_id in rank_ids:
                original = by_rank[int(rank_id)]
                ranks.append(RankRef({**original, "node": node}))
        return ranks

    def _maps_for(self, source, ranks):
        """One device map per target node: the image's GPUs onto the node's.

        The image's GPUs are those of the node that dumped it, which
        `dumped_on` records and a later restore does not change. A node
        restoring an image it dumped itself gets an identity map; one
        restoring somebody else's gets the real thing - whether this is the
        first restore of the epoch or the third.
        """
        dumped = {int(k): v for k, v in (source.get("dumped_on") or {}).items()}
        fallback = {int(r["rank"]): r["node"] for r in source.get("ranks", [])}
        maps = {}
        for node in {r["node"] for r in ranks}:
            here = sorted(int(r["rank"]) for r in ranks if r["node"] == node)
            origins = sorted({dumped.get(rank) or fallback.get(rank) for rank in here} - {None})
            if len(origins) > 1:
                # The agent applies one map to every rank it restores; ranks
                # dumped on different nodes would each need their own.
                _LOG.warn(
                    "ranks placed on one node were dumped on several; using the "
                    "first one's map for all",
                    node=node, origins=origins,
                )
            src_uuids = self._node_uuids(origins[0]) if origins else []
            dst_uuids = self._node_uuids(node)
            if src_uuids and dst_uuids:
                maps[node] = to_cli(build_pairs(src_uuids, dst_uuids))
        return maps

    def plan_restore(self, epoch_id, node_count, allow_mnnvl=None):
        """Which nodes could host this image, and why the rest cannot."""
        last = self.store.get(epoch_id) or {}
        job_id = last.get("job_id")
        good = self.store.last_good(job_id) if job_id else None
        req = ImageRequirements(**(good or {}).get("requirements", {})) if good else None
        if req is None:
            raise KeyError(f"no recorded requirements for {epoch_id}")
        chosen, rejected = select(
            req,
            list(self.node_info.values()),
            node_count,
            allow_mnnvl=self.cfg.allow_mnnvl if allow_mnnvl is None else allow_mnnvl,
        )
        return {"nodes": [n.get("host") for n in chosen], "rejected": rejected}

    def gc(self, job_id, retain=3, image_root=None):
        """Reclaim images beyond the retention count.

        Two things are never deleted: the newest `retain` images, and whatever
        the job's last-good pointer names. Deleting the image a failed epoch
        would fall back to is the one mistake retention must not make.
        """
        retain = max(1, int(retain))
        keep_epochs = self.store.retained(job_id)[:retain]
        keep = {e["epoch_id"] for e in keep_epochs}
        good = self.store.last_good(job_id)
        if good and good.get("epoch_id"):
            keep.add(good["epoch_id"])

        candidates = [
            e for e in self.store.retained(job_id) if e["epoch_id"] not in keep
        ]
        deleted, incomplete = [], []
        for epoch in candidates:
            # The images live where they were dumped, which a later restore
            # onto other nodes does not change - `ranks` is the placement now,
            # `images` is where the bytes are.
            nodes = sorted(
                {i["node"] for i in epoch.get("images", []) if i.get("node")}
                or {r["node"] for r in epoch.get("ranks", [])}
            )
            _results, errors = self.pool.fanout(
                nodes,
                "delete_images",
                job_id=job_id,
                epoch_id=epoch["epoch_id"],
                image_root=image_root or self.cfg.image_dir,
            )
            if errors:
                # Not pruned: the bytes are still there, and an epoch marked
                # pruned is never looked at again. The next sweep retries.
                _LOG.warn(
                    "image deletion incomplete; will retry",
                    epoch=epoch["epoch_id"],
                    errors=errors,
                )
                incomplete.append(epoch["epoch_id"])
                continue
            self.store.mark_pruned(epoch["epoch_id"])
            deleted.append(epoch["epoch_id"])

        _LOG.info(
            "gc complete", job=job_id, retained=len(keep), deleted=len(deleted),
            incomplete=len(incomplete),
        )
        return {"kept": sorted(keep), "deleted": deleted, "incomplete": incomplete}

    def status(self, job_id=None):
        return {
            "nodes": self.pool.nodes(),
            "jobs": {j: len(r) for j, r in self._jobs.items()},
            "epochs": [
                {k: e.get(k) for k in ("epoch_id", "job_id", "phase", "seconds")}
                for e in self.store.list_epochs(job_id)[-20:]
            ],
        }

    def recover(self):
        """What a coordinator restart found. Reported, never auto-resumed."""
        recoverable, lost = self.store.in_flight()
        for epoch in lost:
            _LOG.error(
                "epoch was past the commit point at coordinator restart",
                epoch=epoch.get("epoch_id"),
                phase=epoch.get("phase"),
            )
        return {
            "abortable": [e["epoch_id"] for e in recoverable],
            "lost": [e["epoch_id"] for e in lost],
        }


def build_server(coord, addr=None):
    server = rpc.Server(addr or coord.cfg.coord_addr, name="coord")
    for name in (
        "register_agent",
        "register_job",
        "checkpoint",
        "restore",
        "plan_restore",
        "gc",
        "status",
        "recover",
    ):
        server.op(name)(getattr(coord, name))
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(description="mncr coordinator")
    ap.add_argument("--addr", default=None)
    ap.add_argument("--agents", default="", help="node=addr,node=addr")
    ap.add_argument("--epoch-dir", default=None)
    ap.add_argument("--metrics-port", type=int, default=9181)
    args = ap.parse_args(argv)

    cfg = config.load()
    agents = {}
    for pair in filter(None, args.agents.split(",")):
        node, _, addr = pair.partition("=")
        agents[node] = addr

    coord = Coordinator(
        cfg,
        agents=agents,
        store=EpochStore(args.epoch_dir) if args.epoch_dir else None,
    )
    server = build_server(coord, args.addr or cfg.coord_addr).start()
    metrics_server = metrics.serve(args.metrics_port) if args.metrics_port else None
    print(json.dumps(coord.recover()))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if metrics_server:
            metrics_server.shutdown()
            metrics_server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
