"""Two-phase commit under failure.

These are the tests the whole design exists for. Each one injects a failure at a
different point and asserts the one property that matters there: before the
commit point the job survives; after it, the failure is reported as terminal
rather than papered over.
"""

import unittest

from . import context  # noqa: F401
from mncr.errors import AbortableError, PreconditionError, TerminalError
from mncr.proto import Phase
from verify.sim import SimCluster


class TwoPhaseCommitTest(unittest.TestCase):
    NODES = 2
    RANKS_PER_NODE = 1

    def setUp(self):
        self.sim = SimCluster(
            nodes=self.NODES, ranks_per_node=self.RANKS_PER_NODE, step_seconds=0.002
        ).start()

    def tearDown(self):
        self.sim.stop()

    def _launch(self, **kwargs):
        self.sim.launch_ranks(**kwargs)
        self.sim.wait_registered()

    def _epochs(self):
        return self.sim.coord.store.list_epochs(self.sim.job_id)

    def _last_epoch(self):
        return sorted(self._epochs(), key=lambda e: e["created_at"])[-1]

    # ----------------------------------------------------------- happy path
    def test_clean_checkpoint_commits_and_resumes(self):
        self._launch()
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertEqual(result["phase"], Phase.RUNNING.value)
        self.assertTrue(self.sim.wait_for_event("resumed", timeout=20))
        self.assertEqual(self.sim.alive(), self.NODES * self.RANKS_PER_NODE)

    def test_driver_calls_are_ordered_lock_all_then_checkpoint_all(self):
        self._launch()
        self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        for node, calls in self.sim.driver_calls().items():
            actions = [action for action, _pid in calls]
            self.assertLess(
                max(i for i, a in enumerate(actions) if a == "lock"),
                min(i for i, a in enumerate(actions) if a == "checkpoint"),
                f"{node}: a checkpoint was issued before every lock completed",
            )

    def test_last_good_image_recorded_on_success(self):
        self._launch()
        result = self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        good = self.sim.coord.store.last_good(self.sim.job_id)
        self.assertEqual(good["epoch_id"], result["epoch_id"])

    # -------------------------------------------------- abort before commit
    def test_dirty_rank_aborts_before_the_commit_point(self):
        self._launch(dirty_ranks=[0])
        with self.assertRaises(AbortableError) as ctx:
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertIn("not clean", str(ctx.exception))

        epoch = self._last_epoch()
        self.assertEqual(epoch["phase"], Phase.ABORTED.value)
        # The job is intact: no rank was ever locked.
        for calls in self.sim.driver_calls().values():
            self.assertNotIn("checkpoint", [a for a, _ in calls])
        self.assertEqual(self.sim.alive(), self.NODES * self.RANKS_PER_NODE)

    def test_lock_failure_unlocks_everything_and_aborts(self):
        self._launch()
        victim = self.sim.agents["node-1"]
        pid = victim.local_ranks(self.sim.job_id)[0]["host_pid"]
        victim.driver.fail_next("lock", pid, "lock timed out draining work")

        with self.assertRaises(AbortableError):
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")

        self.assertEqual(self._last_epoch()["phase"], Phase.ABORTED.value)
        # Every rank that did get locked was unlocked again.
        for node, agent in self.sim.agents.items():
            for record in agent.local_ranks(self.sim.job_id):
                self.assertEqual(
                    agent.driver.state_of(record["host_pid"]),
                    "running",
                    f"{node} rank {record['rank']} left locked after abort",
                )

    def test_aborted_ranks_still_rebuild(self):
        """An abort still tore the communicator down, so the rank must rebuild."""
        self._launch()
        agent = self.sim.agents["node-1"]
        pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
        agent.driver.fail_next("lock", pid)
        with self.assertRaises(AbortableError):
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertTrue(self.sim.wait_for_event("resumed", timeout=20))
        for progress in self.sim.all_progress():
            resumed = [e for e in progress["events"] if e["event"] == "resumed"]
            self.assertTrue(resumed and resumed[-1]["aborted"])

    def test_missing_vote_aborts(self):
        self._launch()
        # A rank that dies before voting must not be waited on forever.
        rank_id, _node, proc = self.sim.procs[0]
        proc.kill()
        proc.wait(timeout=5)
        self.sim.cfg.quiesce_timeout = 3.0
        with self.assertRaises(AbortableError):
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertEqual(self._last_epoch()["phase"], Phase.ABORTED.value)

    # --------------------------------------------------- failure past commit
    def test_checkpoint_failure_is_terminal(self):
        self._launch()
        agent = self.sim.agents["node-1"]
        pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
        agent.driver.fail_next("checkpoint", pid, "device memory copy failed")

        with self.assertRaises(TerminalError) as ctx:
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertIn("after the commit point", str(ctx.exception))
        self.assertIn("last good image", str(ctx.exception))
        self.assertEqual(self._last_epoch()["phase"], Phase.FAILED.value)

    def test_dump_failure_is_terminal(self):
        self._launch()
        self.sim.agents["node-0"].criu.fail_next("dump")
        with self.assertRaises(TerminalError):
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")
        self.assertEqual(self._last_epoch()["phase"], Phase.FAILED.value)

    def test_agent_refuses_to_abort_past_the_commit_point(self):
        self._launch()
        agent = self.sim.agents["node-0"]
        ranks = [r["rank"] for r in agent.local_ranks(self.sim.job_id)]
        agent.prepare(self.sim.job_id, "ep-manual", ranks=ranks, vote_timeout=10)
        agent.lock(self.sim.job_id, "ep-manual", ranks=ranks)
        agent.checkpoint(self.sim.job_id, "ep-manual", ranks=ranks)
        with self.assertRaises(PreconditionError) as ctx:
            agent.abort(self.sim.job_id, "ep-manual", ranks=ranks)
        self.assertIn("past the commit point", str(ctx.exception))

    # --------------------------------------------------------- store recovery
    def test_store_separates_recoverable_from_lost_epochs(self):
        self._launch()
        agent = self.sim.agents["node-1"]
        pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
        agent.driver.fail_next("checkpoint", pid)
        with self.assertRaises(TerminalError):
            self.sim.coord.checkpoint(self.sim.job_id, mode="continue")

        recoverable, lost = self.sim.coord.store.in_flight()
        self.assertEqual(recoverable, [])
        self.assertEqual(lost, [])   # FAILED is a settled state, not in flight

        report = self.sim.coord.recover()
        self.assertEqual(report["abortable"], [])
        self.assertEqual(report["lost"], [])


if __name__ == "__main__":
    unittest.main()
