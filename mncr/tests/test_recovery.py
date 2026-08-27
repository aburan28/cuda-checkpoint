"""Coordinator restart, and a rank left waiting.

Both are cases where a component dies with an epoch in flight. What matters is
that the survivor classifies the epoch correctly - and, past the commit point,
that it refuses to pretend the job can be resumed.
"""

import os
import tempfile
import unittest

from . import context  # noqa: F401
from coord.main import Coordinator
from coord.store import EpochStore
from mncr import config
from mncr.errors import TimeoutError_
from mncr.proto import Epoch, Phase, RankRef
from torchckpt.channel import ControlDir


def epoch_in(phase, job="job", store=None):
    ranks = [RankRef.make(job, 0, "node-a", host_pid=10)]
    epoch = Epoch.make(job, ranks)
    for step in (Phase.PREPARING, Phase.PREPARED, Phase.LOCKED,
                 Phase.CHECKPOINTED, Phase.DUMPED):
        epoch.set_phase(step)
        if step is phase:
            break
    if store:
        store.put(epoch)
    return epoch


class CoordinatorRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mncr-recover-")
        self.store = EpochStore(os.path.join(self.tmp, "epochs"))

    def _coord(self):
        return Coordinator(config.Config(), agents={}, store=self.store)

    def test_an_epoch_stuck_before_the_commit_point_is_abortable(self):
        epoch = epoch_in(Phase.LOCKED, store=self.store)
        report = self._coord().recover()
        self.assertEqual(report["abortable"], [epoch["epoch_id"]])
        self.assertEqual(report["lost"], [])

    def test_an_epoch_stuck_past_the_commit_point_is_lost(self):
        epoch = epoch_in(Phase.CHECKPOINTED, store=self.store)
        report = self._coord().recover()
        self.assertEqual(report["lost"], [epoch["epoch_id"]])
        self.assertEqual(report["abortable"], [])

    def test_a_dumped_epoch_is_also_lost_not_abortable(self):
        """The image exists, but the ranks that made it are already gone."""
        epoch = epoch_in(Phase.DUMPED, store=self.store)
        report = self._coord().recover()
        self.assertEqual(report["lost"], [epoch["epoch_id"]])

    def test_settled_epochs_are_not_reported(self):
        finished = epoch_in(Phase.DUMPED, store=self.store)
        finished.set_phase(Phase.RESTORING).set_phase(Phase.RESUMED)
        finished.set_phase(Phase.RUNNING)
        self.store.put(finished)
        report = self._coord().recover()
        self.assertEqual(report, {"abortable": [], "lost": []})

    def test_recovery_never_resumes_anything_by_itself(self):
        epoch = epoch_in(Phase.LOCKED, store=self.store)
        self._coord().recover()
        # Still exactly where it was: recovery reports, an operator decides.
        self.assertEqual(self.store.get(epoch["epoch_id"])["phase"], Phase.LOCKED.value)

    def test_the_store_survives_a_new_coordinator(self):
        epoch = epoch_in(Phase.PREPARED, store=self.store)
        reopened = EpochStore(os.path.join(self.tmp, "epochs"))
        self.assertEqual(reopened.get(epoch["epoch_id"])["phase"], Phase.PREPARED.value)


class RankWaitTest(unittest.TestCase):
    """A rank that is released, and one that never is."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mncr-wait-")
        self.control = ControlDir(self.tmp, "job")

    def test_a_token_for_another_epoch_is_ignored(self):
        self.control.put_token(0, {"epoch_id": "other", "restored": False})
        self.assertIsNone(self.control.wait_for_token(0, "mine", timeout=0.3))

    def test_the_right_token_is_returned(self):
        self.control.put_token(0, {"epoch_id": "mine", "restored": True})
        token = self.control.wait_for_token(0, "mine", timeout=2)
        self.assertTrue(token["restored"])

    def test_waiting_gives_up_rather_than_hanging(self):
        started = __import__("time").monotonic()
        self.assertIsNone(self.control.wait_for_token(0, "never", timeout=0.4))
        self.assertLess(__import__("time").monotonic() - started, 5)

    def test_the_rank_library_raises_when_it_is_never_released(self):
        """A rank whose coordinator vanished must fail loudly.

        Silently resuming would put it back to work while its peers are still
        quiesced - which is worse than dying.
        """
        import torchckpt
        from torchckpt.api import _Runtime

        runtime = _Runtime()
        runtime.init(
            job_id="job",
            rank=0,
            world_size=1,
            agent_addr="tcp:127.0.0.1:1",
            control_root=self.tmp,
            register=False,
            proc_root=os.path.join(self.tmp, "no-proc"),
        )
        runtime.pending = {
            "epoch_id": "orphan",
            "target_step": 0,
            "request": {"wait_timeout": 0.3},
        }
        with self.assertRaises(TimeoutError_) as ctx:
            runtime._service()
        self.assertIn("no resume token", str(ctx.exception))
        self.assertEqual(runtime.status.state, torchckpt.RankState.FAILED)


if __name__ == "__main__":
    unittest.main()
