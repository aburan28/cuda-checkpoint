"""Retention and policy suspension.

Both were declared in the CRD before they existed. These are the tests that
make the declaration true - and the first one guards the single mistake
retention must never make.
"""

import os
import unittest

from . import context  # noqa: F401
from k8s.controller import Controller
from mncr.proto import Phase
from verify.sim import SimCluster


class FakeK8s:
    """Enough of the API for the controller: list custom resources, patch status."""

    def __init__(self, namespace="test"):
        self.namespace = namespace
        self.objects = {}
        self.deleted_pods = []

    def add(self, plural, name, spec, status=None):
        self.objects.setdefault(plural, {})[name] = {
            "metadata": {"name": name},
            "spec": spec,
            "status": status or {},
        }
        return self

    def crs(self, plural, namespace=None, group=None, version=None):
        return list(self.objects.get(plural, {}).values())

    def cr_path(self, plural, name, namespace=None, group=None, version=None):
        return f"{plural}/{name}"

    def patch_status(self, path, status):
        plural, _, name = path.partition("/")
        self.objects[plural][name].setdefault("status", {}).update(status)
        return self.objects[plural][name]

    def pods(self, namespace=None, selector=None):
        return []

    def delete_pod(self, name, namespace=None, grace=0):
        self.deleted_pods.append(name)


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimCluster(nodes=1, ranks_per_node=1, step_seconds=0.002).start()
        self.sim.launch_ranks()
        self.sim.wait_registered()

    def tearDown(self):
        self.sim.stop()

    def _epochs(self, count):
        ids = []
        for _ in range(count):
            result = self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
            ids.append(result["epoch_id"])
            self.sim.wait_for_event("resumed", timeout=20)
        return ids

    def test_gc_keeps_the_newest_and_deletes_the_rest(self):
        ids = self._epochs(4)
        result = self.sim.coord.gc(self.sim.job_id, retain=2)
        # The two newest are kept; last_good is the newest, already in that set.
        self.assertEqual(len(result["deleted"]), 2)
        self.assertEqual(set(result["deleted"]), set(ids[:2]))
        for epoch_id in ids[2:]:
            self.assertIn(epoch_id, result["kept"])

    def test_gc_never_deletes_the_last_good_image(self):
        ids = self._epochs(3)
        # Pin last_good to the oldest epoch, which retain=1 would otherwise drop.
        self.sim.coord.store.mark_last_good(self.sim.job_id, ids[0], ids[0])
        result = self.sim.coord.gc(self.sim.job_id, retain=1)
        self.assertIn(ids[0], result["kept"])
        self.assertNotIn(ids[0], result["deleted"])

    def test_deleted_images_are_gone_from_disk(self):
        ids = self._epochs(2)
        oldest = ids[0]
        path = os.path.join(self.sim.image_root, oldest)
        self.assertTrue(os.path.isdir(path))
        self.sim.coord.gc(self.sim.job_id, retain=1)
        self.assertFalse(os.path.isdir(path), "gc did not reach the bytes")

    def test_pruned_epochs_are_not_reconsidered(self):
        self._epochs(3)
        first = self.sim.coord.gc(self.sim.job_id, retain=1)
        second = self.sim.coord.gc(self.sim.job_id, retain=1)
        self.assertTrue(first["deleted"])
        self.assertEqual(second["deleted"], [], "gc deleted the same images twice")

    def test_epoch_record_survives_its_image(self):
        ids = self._epochs(2)
        self.sim.coord.gc(self.sim.job_id, retain=1)
        record = self.sim.coord.store.get(ids[0])
        self.assertIsNotNone(record, "the audit record was deleted with the image")
        self.assertTrue(record["pruned"])
        self.assertEqual(record["phase"], Phase.RUNNING.value)


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimCluster(nodes=1, ranks_per_node=1, step_seconds=0.002).start()
        self.sim.launch_ranks()
        self.sim.wait_registered()
        self.k8s = FakeK8s()
        self.controller = Controller(self.sim.coord, client=self.k8s, namespace="test")

    def tearDown(self):
        self.sim.stop()

    def _policy(self, **spec):
        base = {"jobId": self.sim.job_id, "intervalSeconds": 0, "retain": 1,
                "suspendAfterFailures": 2}
        base.update(spec)
        self.k8s.add("checkpointpolicies", "p1", base)
        return self.k8s.objects["checkpointpolicies"]["p1"]

    def _tick(self):
        self.controller._policy_last.clear()   # interval is not what is under test
        self.controller.reconcile_policy(self.k8s.objects["checkpointpolicies"]["p1"])
        return self.k8s.objects["checkpointpolicies"]["p1"]["status"]

    def test_successful_policy_records_the_epoch_and_runs_gc(self):
        self._policy()
        status = self._tick()
        self.assertIn("lastEpochId", status)
        self.assertEqual(status["consecutiveFailures"], 0)
        self.assertEqual(status["retainedImages"], 1)

    def test_failures_accumulate_then_suspend(self):
        self._policy(suspendAfterFailures=2)
        agent = self.sim.agents["node-0"]

        for expected in (1, 2):
            pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
            agent.driver.fail_next("lock", pid)
            status = self._tick()
            self.assertEqual(status["consecutiveFailures"], expected)

        self.assertTrue(status["suspended"])
        self.assertIn("suspendAfterFailures=2", status["suspendedReason"])

    def test_suspended_policy_stops_attempting(self):
        self._policy(suspendAfterFailures=1)
        agent = self.sim.agents["node-0"]
        pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
        agent.driver.fail_next("lock", pid)
        self._tick()
        self.assertTrue(self.k8s.objects["checkpointpolicies"]["p1"]["status"]["suspended"])

        before = len(self.sim.coord.store.list_epochs(self.sim.job_id))
        self._tick()
        after = len(self.sim.coord.store.list_epochs(self.sim.job_id))
        self.assertEqual(before, after, "a suspended policy still ran a checkpoint")

    def test_a_success_clears_the_failure_count(self):
        self._policy(suspendAfterFailures=3)
        agent = self.sim.agents["node-0"]
        pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
        agent.driver.fail_next("lock", pid)
        self.assertEqual(self._tick()["consecutiveFailures"], 1)
        self.sim.wait_for_event("resumed", timeout=20)
        self.assertEqual(self._tick()["consecutiveFailures"], 0)


if __name__ == "__main__":
    unittest.main()
