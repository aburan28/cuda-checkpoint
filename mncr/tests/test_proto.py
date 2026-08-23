"""The phase model, and the commit point it exists to encode."""

import unittest

from . import context  # noqa: F401
from mncr.proto import (
    COMMIT_POINT,
    Epoch,
    Phase,
    RankRef,
    can_transition,
    next_on_failure,
)


class TestPhases(unittest.TestCase):
    def test_commit_point_is_checkpointed(self):
        self.assertIs(COMMIT_POINT, Phase.CHECKPOINTED)

    def test_abortable_set_stops_at_lock(self):
        for phase in (Phase.RUNNING, Phase.PREPARING, Phase.PREPARED, Phase.LOCKED):
            self.assertTrue(phase.abortable_in_place, phase)
        for phase in (Phase.CHECKPOINTED, Phase.DUMPED, Phase.RESTORING, Phase.RESUMED):
            self.assertFalse(phase.abortable_in_place, phase)

    def test_failure_before_commit_aborts_after_commit_fails(self):
        self.assertIs(next_on_failure(Phase.LOCKED), Phase.ABORTED)
        self.assertIs(next_on_failure(Phase.CHECKPOINTED), Phase.FAILED)
        self.assertIs(next_on_failure(Phase.DUMPED), Phase.FAILED)

    def test_cannot_abort_after_commit(self):
        self.assertTrue(can_transition(Phase.LOCKED, Phase.ABORTED))
        self.assertFalse(can_transition(Phase.CHECKPOINTED, Phase.ABORTED))

    def test_forward_transitions_are_constrained(self):
        self.assertTrue(can_transition(Phase.PREPARED, Phase.LOCKED))
        self.assertFalse(can_transition(Phase.PREPARED, Phase.CHECKPOINTED))
        self.assertFalse(can_transition(Phase.RUNNING, Phase.DUMPED))


class TestEpoch(unittest.TestCase):
    def _epoch(self):
        ranks = [
            RankRef.make("job", 0, "node-a", host_pid=10),
            RankRef.make("job", 1, "node-b", host_pid=11),
        ]
        return Epoch.make("job", ranks)

    def test_illegal_transition_raises(self):
        epoch = self._epoch()
        with self.assertRaises(ValueError):
            epoch.set_phase(Phase.DUMPED)

    def test_history_records_each_move(self):
        epoch = self._epoch()
        epoch.set_phase(Phase.PREPARING).set_phase(Phase.PREPARED)
        self.assertEqual(len(epoch["history"]), 2)
        self.assertEqual(epoch["history"][-1]["to"], "prepared")

    def test_past_commit_point_flag(self):
        epoch = self._epoch()
        for phase in (Phase.PREPARING, Phase.PREPARED, Phase.LOCKED):
            epoch.set_phase(phase)
            self.assertFalse(epoch.past_commit_point)
        epoch.set_phase(Phase.CHECKPOINTED)
        self.assertTrue(epoch.past_commit_point)

    def test_node_grouping(self):
        epoch = self._epoch()
        self.assertEqual(epoch.nodes(), ["node-a", "node-b"])
        self.assertEqual([r["rank"] for r in epoch.ranks_on("node-b")], [1])


if __name__ == "__main__":
    unittest.main()
