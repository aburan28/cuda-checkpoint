"""What a rank does with a request file, and when it must not."""

import json
import os
import shutil
import tempfile
import time
import unittest

from . import context  # noqa: F401
from torchckpt.api import _Runtime
from torchckpt.channel import ControlDir


class FreshnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rt = _Runtime()
        self.rt.init(job_id="j", rank=0, world_size=1, agent_addr="tcp:127.0.0.1:1",
                     control_root=self.tmp, register=False, auto_teardown=False,
                     proc_root=os.path.join(self.tmp, "no-proc"))
        self.control = ControlDir(self.tmp, "j")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _request(self, epoch_id, issued_at):
        self.control.put_request({"epoch_id": epoch_id, "action": "checkpoint",
                                  "lookahead": 1, "issued_at": issued_at})
        return self.rt.control.poll_request()

    def test_a_request_issued_before_the_rank_started_is_ignored(self):
        stale = self._request("ep-000000000001", self.rt.started_at - 10)
        self.assertIsNone(self.rt._fresh(stale))
        fresh = self._request("ep-000000000002", time.time() + 1)
        self.assertIsNotNone(self.rt._fresh(fresh))

    def test_a_restore_token_moves_the_line(self):
        """A restored process keeps its original started_at. The request that
        led to its dump is older than the token that brought it back, and
        must not be serviced again."""
        dump_request_at = self.rt.started_at + 5
        token_at = dump_request_at + 30
        self.rt.agent.register = lambda *a, **k: None
        self.rt._resume({"epoch_id": "ep-00000000000a", "rank": 0, "world_size": 1,
                         "restored": True, "aborted": False, "at": token_at}, "ep-00000000000a")
        self.assertGreaterEqual(self.rt.started_at, token_at)
        leftover = self._request("ep-00000000000a", dump_request_at)
        self.assertIsNone(self.rt._fresh(leftover))
        later = self._request("ep-00000000000b", token_at + 1)
        self.assertIsNotNone(self.rt._fresh(later))


if __name__ == "__main__":
    unittest.main()
