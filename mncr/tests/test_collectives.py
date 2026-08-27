"""The consistent cut, against a real process group.

Every other test falls through the no-process-group path, where agreement is
trivially the local answer. These use gloo on CPU so the coupling is real: a
rank that stops issuing collectives blocks its peers at the next one, which is
exactly the dynamic the design has to survive.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from . import context  # noqa: F401
from verify.sim import SimCluster

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGREE_PROBE = os.path.join(ROOT, "verify", "fake", "agree_probe.py")


def torch_available():
    try:
        import torch  # noqa: F401
        import torch.distributed as dist

        return dist.is_available() and dist.is_gloo_available()
    except Exception:
        return False


def free_address():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"tcp://127.0.0.1:{port}"


@unittest.skipUnless(torch_available(), "needs torch with the gloo backend")
class StepAgreementTest(unittest.TestCase):
    def _run(self, local_steps, lookahead=1, timeout=120):
        world = len(local_steps)
        init_method = free_address()
        tmp = tempfile.mkdtemp(prefix="mncr-agree-")
        procs = []
        for rank, local_step in enumerate(local_steps):
            out = os.path.join(tmp, f"rank-{rank}.json")
            procs.append(
                (
                    out,
                    subprocess.Popen(
                        [
                            sys.executable, AGREE_PROBE,
                            "--rank", str(rank),
                            "--world-size", str(world),
                            "--init-method", init_method,
                            "--local-step", str(local_step),
                            "--lookahead", str(lookahead),
                            "--out", out,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    ),
                )
            )
        results = []
        for out, proc in procs:
            _stdout, stderr = proc.communicate(timeout=timeout)
            self.assertEqual(
                proc.returncode, 0, f"probe failed: {stderr.decode()[-500:]}"
            )
            with open(out) as fh:
                results.append(json.load(fh))
        return sorted(results, key=lambda r: r["rank"])

    def test_ranks_at_different_steps_agree_on_one(self):
        results = self._run([100, 101, 100, 99])
        targets = {r["target"] for r in results}
        self.assertEqual(
            len(targets), 1, f"ranks disagreed on where to stop: {targets}"
        )
        self.assertEqual(targets.pop(), 102, "agreement did not take the maximum")

    def test_every_rank_saw_a_live_process_group(self):
        results = self._run([5, 5])
        for result in results:
            self.assertTrue(
                result["distributed_ready"],
                "the probe fell through the no-process-group path, so this "
                "test proved nothing",
            )

    def test_lookahead_is_honoured(self):
        results = self._run([10, 12], lookahead=4)
        self.assertEqual({r["target"] for r in results}, {16})

    def test_default_teardown_destroys_the_group(self):
        results = self._run([1, 1])
        for result in results:
            self.assertTrue(result["teardown"].get("process_group_destroyed"))


@unittest.skipUnless(torch_available(), "needs torch with the gloo backend")
class CollectiveLifecycleTest(unittest.TestCase):
    RANKS = 3

    def setUp(self):
        self.sim = SimCluster(
            nodes=1, ranks_per_node=self.RANKS, step_seconds=0.01
        ).start()
        self.sim.launch_ranks(gloo=True)
        self.sim.wait_registered(timeout=120)
        time.sleep(0.5)   # let the ranks get into the loop

    def tearDown(self):
        self.sim.stop()

    def _quiesce_steps(self):
        steps = []
        for rank in range(self.RANKS):
            events = [
                e for e in self.sim.progress(rank)["events"] if e["event"] == "quiesced"
            ]
            steps.append(events[-1]["step"] if events else None)
        return steps

    def _resumed(self):
        out = []
        for rank in range(self.RANKS):
            events = [
                e for e in self.sim.progress(rank)["events"] if e["event"] == "resumed"
            ]
            out.append(events[-1] if events else None)
        return out

    def test_all_ranks_stop_at_the_same_step(self):
        self.sim.coord.checkpoint(
            self.sim.job_id, mode="continue",
            init_method=self.sim.new_rendezvous(), backend="gloo",
        )
        self.assertTrue(self.sim.wait_for_event("resumed", timeout=120))
        steps = self._quiesce_steps()
        self.assertEqual(
            len(set(steps)), 1,
            f"ranks quiesced at different steps {steps}; the restored job would "
            f"have ranks on different iterations",
        )

    def test_the_group_works_again_after_resume(self):
        self.sim.coord.checkpoint(
            self.sim.job_id, mode="continue",
            init_method=self.sim.new_rendezvous(), backend="gloo",
        )
        self.assertTrue(self.sim.wait_for_event("resumed", timeout=120))
        for event in self._resumed():
            self.assertIsNotNone(event)
            self.assertTrue(
                event["rejoined"],
                "the rebuilt process group did not produce a correct collective",
            )

    def test_an_abort_also_leaves_a_working_group(self):
        """The teardown already happened, so an abort must rebuild too."""
        agent = self.sim.agents["node-0"]
        pid = agent.local_ranks(self.sim.job_id)[0]["host_pid"]
        agent.driver.fail_next("lock", pid)
        from mncr.errors import AbortableError

        with self.assertRaises(AbortableError):
            self.sim.coord.checkpoint(
                self.sim.job_id, mode="continue",
                init_method=self.sim.new_rendezvous(), backend="gloo",
            )
        self.assertTrue(self.sim.wait_for_event("resumed", timeout=120))
        for event in self._resumed():
            self.assertTrue(event["aborted"])
            self.assertTrue(
                event["rejoined"], "an aborted epoch left the group broken"
            )


if __name__ == "__main__":
    unittest.main()
