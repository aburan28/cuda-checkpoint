"""Reconciler for GpuCheckpoint, GpuRestore and CheckpointPolicy.

Deliberately thin. The controller decides *when*, the coordinator decides
*whether it is safe*, and the agent does the work. Keeping the commit-point
logic out of the controller means a controller crash can never leave a job in a
state only the controller understood.
"""

import argparse
import time

from mncr import config, log, metrics
from mncr.errors import AbortableError, TerminalError

from .client import K8sClient

_LOG = log.get("k8s.controller")

GROUP = "mncr.io"
VERSION = "v1alpha1"


class Controller:
    def __init__(self, coordinator, client=None, namespace=None, poll_interval=5.0):
        self.coord = coordinator
        self.client = client or K8sClient(namespace=namespace)
        self.namespace = namespace or self.client.namespace
        self.poll_interval = poll_interval
        self._policy_last = {}

    # ------------------------------------------------------------ discovery
    def discover_job(self, spec):
        """Map a job selector onto the ranks the agents already know about.

        Ranks register themselves with their node agent at startup, so the
        controller does not have to reconstruct the topology from pod specs -
        it only has to name the job.
        """
        job_id = spec["jobId"]
        return job_id

    # ---------------------------------------------------------- reconcilers
    def reconcile_checkpoint(self, obj):
        name = obj["metadata"]["name"]
        spec = obj.get("spec", {})
        status = obj.get("status", {})
        if status.get("phase") in ("Succeeded", "Failed"):
            return

        job_id = self.discover_job(spec)
        mode = spec.get("mode", "continue")
        self._set_status(
            "gpucheckpoints", name, {"phase": "Running", "startedAt": _now()}
        )
        try:
            result = self.coord.checkpoint(
                job_id,
                mode=mode,
                image_root=spec.get("imageRoot"),
                reason=spec.get("reason", f"GpuCheckpoint/{name}"),
            )
        except AbortableError as exc:
            # Nothing was lost. The job is still running.
            self._set_status(
                "gpucheckpoints",
                name,
                {
                    "phase": "Failed",
                    "reason": "Aborted",
                    "message": str(exc)[:900],
                    "jobIntact": True,
                    "finishedAt": _now(),
                },
            )
            _LOG.warn("checkpoint aborted", name=name, error=str(exc))
            return
        except TerminalError as exc:
            # The epoch crossed the commit point and did not complete.
            self._set_status(
                "gpucheckpoints",
                name,
                {
                    "phase": "Failed",
                    "reason": "LostPastCommitPoint",
                    "message": str(exc)[:900],
                    "jobIntact": False,
                    "finishedAt": _now(),
                },
            )
            _LOG.error("checkpoint lost the job", name=name, error=str(exc))
            return

        self._set_status(
            "gpucheckpoints",
            name,
            {
                "phase": "Succeeded",
                "epochId": result["epoch_id"],
                "imageId": result["epoch_id"],
                "images": result["images"],
                "seconds": result.get("seconds"),
                "jobIntact": mode == "continue",
                "finishedAt": _now(),
            },
        )
        if mode == "stop" and spec.get("deletePodsOnStop", True):
            self._delete_job_pods(spec)

    def reconcile_restore(self, obj):
        name = obj["metadata"]["name"]
        spec = obj.get("spec", {})
        if obj.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            return

        self._set_status("gpurestores", name, {"phase": "Running", "startedAt": _now()})
        try:
            targets = spec.get("targets")
            result = self.coord.restore(
                spec["jobId"],
                spec["epochId"],
                targets=targets,
                image_root=spec.get("imageRoot"),
            )
        except Exception as exc:
            self._set_status(
                "gpurestores",
                name,
                {"phase": "Failed", "message": str(exc)[:900], "finishedAt": _now()},
            )
            _LOG.error("restore failed", name=name, error=str(exc))
            return
        self._set_status(
            "gpurestores",
            name,
            {
                "phase": "Succeeded",
                "deviceMaps": result.get("device_maps", {}),
                "finishedAt": _now(),
            },
        )

    def reconcile_policy(self, obj):
        """Periodic checkpoints. Interval, not cron - a checkpoint that lands
        while the previous one is still uploading is worse than a late one."""
        name = obj["metadata"]["name"]
        spec = obj.get("spec", {})
        status = obj.get("status", {})

        if status.get("suspended"):
            return

        interval = float(spec.get("intervalSeconds", 3600))
        last = self._policy_last.get(name, 0)
        if time.time() - last < interval:
            return
        self._policy_last[name] = time.time()

        job_id = spec["jobId"]
        try:
            result = self.coord.checkpoint(
                job_id, mode="continue", reason=f"CheckpointPolicy/{name}"
            )
        except Exception as exc:
            failures = status.get("consecutiveFailures", 0) + 1
            limit = int(spec.get("suspendAfterFailures", 3))
            update = {
                "lastError": str(exc)[:500],
                "consecutiveFailures": failures,
                "lastAttemptAt": _now(),
            }
            metrics.POLICY_SUSPENDED.set(
                1 if failures >= limit else 0, policy=name, job=job_id
            )
            if failures >= limit:
                # Retrying a policy that has failed repeatedly turns one broken
                # job into a source of load on every node it touches. Stop, and
                # make the stop visible.
                update["suspended"] = True
                update["suspendedReason"] = (
                    f"{failures} consecutive failures reached "
                    f"suspendAfterFailures={limit}"
                )
                _LOG.error("policy suspended", name=name, failures=failures)
            else:
                _LOG.warn("policy checkpoint failed", name=name, failures=failures)
            self._set_status("checkpointpolicies", name, update)
            return

        update = {
            "lastEpochId": result["epoch_id"],
            "lastCheckpointAt": _now(),
            "consecutiveFailures": 0,
        }
        retain = int(spec.get("retain", 3))
        try:
            reclaimed = self.coord.gc(job_id, retain=retain)
            update["retainedImages"] = len(reclaimed["kept"])
            if reclaimed["deleted"]:
                _LOG.info(
                    "images reclaimed",
                    name=name,
                    deleted=len(reclaimed["deleted"]),
                    retained=len(reclaimed["kept"]),
                )
        except Exception as exc:  # noqa: BLE001
            # Retention failing must not mark a successful checkpoint failed.
            update["lastError"] = f"gc: {str(exc)[:300]}"
            _LOG.warn("gc failed", name=name, error=str(exc))
        self._set_status("checkpointpolicies", name, update)

    # ------------------------------------------------------------- helpers
    def _set_status(self, plural, name, status):
        try:
            self.client.patch_status(self.client.cr_path(plural, name), status)
        except Exception as exc:
            _LOG.warn("status update failed", plural=plural, name=name, error=str(exc))

    def _delete_job_pods(self, spec):
        selector = spec.get("podSelector")
        if not selector:
            return
        for pod in self.client.pods(selector=selector):
            name = pod["metadata"]["name"]
            try:
                self.client.delete_pod(name)
                _LOG.info("pod deleted after checkpoint-and-stop", pod=name)
            except Exception as exc:
                _LOG.error("pod delete failed", pod=name, error=str(exc))

    # ----------------------------------------------------------------- loop
    def poll_once(self):
        handled = 0
        for plural, handler in (
            ("gpucheckpoints", self.reconcile_checkpoint),
            ("gpurestores", self.reconcile_restore),
            ("checkpointpolicies", self.reconcile_policy),
        ):
            try:
                for obj in self.client.crs(plural, self.namespace, GROUP, VERSION):
                    handler(obj)
                    handled += 1
            except Exception as exc:
                _LOG.warn("list failed", plural=plural, error=str(exc))
        return handled

    def run(self):
        _LOG.info("controller running", namespace=self.namespace)
        while True:
            self.poll_once()
            time.sleep(self.poll_interval)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None):
    from coord.main import Coordinator

    ap = argparse.ArgumentParser(description="mncr kubernetes controller")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--agents", default="", help="node=addr,node=addr")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--metrics-port", type=int, default=9181)
    args = ap.parse_args(argv)

    agents = {}
    for pair in filter(None, args.agents.split(",")):
        node, _, addr = pair.partition("=")
        agents[node] = addr

    coord = Coordinator(config.load(), agents=agents)
    if args.metrics_port:
        metrics.serve(args.metrics_port)
    Controller(coord, namespace=args.namespace, poll_interval=args.interval).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
